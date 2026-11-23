"""Run the shared SLiM engine with explicit phases and independent seed streams."""
from __future__ import annotations
import json, math, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from .runtime import PACKAGE, run_command, dump_json, write_csv

MODEL=PACKAGE/'models/quantitative_resistance.slim'

def eidos(value) -> str:
    if isinstance(value,(Path,str)):
        s=str(value)
        if any(c in s for c in ["'",'\\','\n','\r']): raise ValueError('Unsupported character in Eidos string')
        return "'"+s+"'"
    if isinstance(value,(bool,np.bool_)): return 'T' if value else 'F'
    if isinstance(value,(int,np.integer)): return str(int(value))
    value=float(value)
    if not math.isfinite(value): raise ValueError('Nonfinite simulation parameter')
    return repr(value)

def invoke(cfg: dict, phase: str, defs: dict, seed: int, log: Path) -> None:
    executable=shutil.which(str(cfg['slim']))
    if not executable: raise FileNotFoundError(f'SLiM executable not found: {cfg["slim"]}. Load SLiM 5.1 or set slim in config/local.yaml.')
    cmd=[executable,'-s',str(int(seed))]
    for k,v in {'phase':phase,**defs}.items(): cmd.extend(['-d',f'{k}={eidos(v)}'])
    cmd.append(str(MODEL)); run_command(cmd,log)

def base_defs(cfg: dict, row: dict, idcol: str='patient_id') -> dict:
    p=cfg['primary']
    return {'patient_id':int(row[idcol]),'base_mut_rate':float(row['base_mut_rate']),
        'dose_drug_1_mono':float(row.get('dose_drug_1',row.get('dose_drug_1_mono'))),
        'dose_drug_2_mono':float(row.get('dose_drug_2',row.get('dose_drug_2_mono',0))),
        'patient_turnover':float(row.get('turnover_rate',row.get('patient_turnover'))),
        **{k:float(row[k]) for k in ['draw_m1','draw_m2','draw_m3','cross_resistance']},
        **{k:p[k] for k in ['Q','expansion_time','prefix_length','treatment_time','relapse_cap_factor','m1_meanlog','m1_sdlog','m2_meanlog','m2_sdlog']}}

def expansion(cfg: dict, row: dict, directory: Path, idcol: str='patient_id') -> None:
    directory.mkdir(parents=True,exist_ok=True)
    snapshot=(directory/'pretreatment.bin').resolve(); meta=(directory/'expansion.csv').resolve()
    original=int(row['expansion_seed']); resolved=row.get('resolved_expansion_seed')
    seeds=[int(resolved)] if pd.notna(resolved) else [original+i*cfg['primary']['expansion_seed_step'] for i in range(cfg['primary']['expansion_attempts'])]
    for attempt,seed in enumerate(seeds):
        snapshot.unlink(missing_ok=True); meta.unlink(missing_ok=True)
        invoke(cfg,'expand',{**base_defs(cfg,row,idcol),'expansion_seed':seed,'snapshot_file':snapshot,'meta_file':meta},seed,directory/f'expand_{attempt}.log')
        if not meta.is_file(): raise RuntimeError(f'Expansion produced no metadata: {directory}')
        rec=pd.read_csv(meta).iloc[0].to_dict()
        if rec['status']=='viable':
            if not snapshot.is_file(): raise RuntimeError('Viable expansion has no checkpoint')
            rec.update(original_expansion_seed=original,resolved_expansion_seed=seed,recovery_attempt=attempt)
            dump_json(directory/'expansion.json',rec); return
        if rec['status']!='pre_treatment_extinction': raise RuntimeError(f'Unexpected expansion status: {rec}')
    raise RuntimeError(f'Expansion failed after {len(seeds)} biological retries for {idcol}={row[idcol]}; parameters retained.')

def prefix(cfg: dict, row: dict, directory: Path, idcol: str='patient_id') -> None:
    snap=(directory/'pretreatment.bin').resolve(); out=(directory/'prefix.csv').resolve()
    if not snap.is_file(): raise FileNotFoundError(snap)
    invoke(cfg,'prefix',{**base_defs(cfg,row,idcol),'snapshot_file':snap,'prefix_file':out,'d1_seed':int(row['d1_seed'])},int(row['d1_seed']),directory/'prefix.log')
    x=pd.read_csv(out)
    if not len(x) or not {'N','t_treatment'}<=set(x): raise RuntimeError(f'Invalid prefix {out}')

def outcome(cfg: dict, row: dict, directory: Path, lag: int, out: Path) -> None:
    records=[]; work=out.parent/(out.stem+'_replicates');work.mkdir(parents=True,exist_ok=True)
    for rep in range(1,int(cfg['primary']['n_replicates'])+1):
        seed=int(cfg['primary']['treatment_seed_base'])+int(row['patient_id'])*100000+int(lag)*100+rep
        f=(work/f'{rep:03d}.csv').resolve()
        invoke(cfg,'outcome',{**base_defs(cfg,row),'snapshot_file':(directory/'pretreatment.bin').resolve(),
            'outcome_file':f,'d1_seed':int(row['d1_seed']),'second_strike_lag':int(lag),
            'replicate_id':rep,'treatment_seed':seed},int(row['d1_seed']),work/f'{rep:03d}.log')
        x=pd.read_csv(f)
        if len(x)!=1: raise RuntimeError(f'Expected one outcome row: {f}')
        r=x.iloc[0]
        if (int(r.patient_id),int(r.lag),int(r.replicate))!=(int(row['patient_id']),int(lag),rep): raise RuntimeError('Outcome key mismatch')
        records.append(x)
    write_csv(pd.concat(records,ignore_index=True),out)

def reference_batch(cfg: dict, manifest: Path, batch: int, out: Path) -> None:
    df=pd.read_csv(manifest); size=int(cfg['reference']['batch_size']); rows=[];metadata=[]
    for _,r in df.iloc[batch*size:(batch+1)*size].iterrows():
        row=r.to_dict(); rid=int(row['ref_id']); work=out.parent/'raw'/f'ref_{rid:05d}'
        expansion(cfg,row,work,'ref_id');prefix(cfg,row,work,'ref_id')
        z=pd.read_csv(work/'prefix.csv'); z['ref_id']=rid; rows.append(z)
        metadata.append({'ref_id':rid,**json.loads((work/'expansion.json').read_text())})
        # References never branch to Drug 2. Their temporary checkpoint is disposable.
        (work/'pretreatment.bin').unlink()
    if not rows: raise ValueError('Empty reference batch')
    write_csv(pd.concat(rows,ignore_index=True),out)
    write_csv(pd.DataFrame(metadata),out.with_name(out.stem+'_metadata.csv'))

def clinical_batch(cfg: dict, manifest: Path, batch: int, directory: Path, preset: dict) -> None:
    m=pd.read_csv(manifest);size=int(preset['batch_size']); records=[]
    mode=preset['phase']
    for _,r in m.iloc[batch*size:(batch+1)*size].iterrows():
        row=r.to_dict();pid=int(row['patient_id']);pdir=(directory/'raw'/f'patient_{pid:05d}').resolve();pdir.mkdir(parents=True,exist_ok=True)
        success=False
        for retry in range(int(preset['max_retries'])+1):
            seed=int(row['patient_seed_base'])+retry
            (pdir/'trajectory.csv').unlink(missing_ok=True)
            (pdir/'meta.csv').unlink(missing_ok=True)
            defs=base_defs(cfg,row)
            defs.update({k:row[k] for k in ['base_patient_id','target_decay_rate','m1_meanlog','m1_sdlog','m1_effect_mean_target']})
            defs.update(phase=mode,scenario_name=str(row['scenario']),prefix_length=int(preset['prefix_length']),patient_relapse_factor=float(preset['relapse_factor']),patient_seed=seed,output_file=pdir/'trajectory.csv',meta_file=pdir/'meta.csv')
            defs.pop('phase')
            if mode=='clinical_single':
                defs.update({k:row[k] for k in ['source_k5_patient_id','single_age','n_founders','standing_target']})
                for k in ['source_k5_patient_id','single_age','n_founders','base_patient_id']:defs[k]=int(defs[k])
            else:
                for j in range(1,6): defs[f'age{j}']=int(row[f'age{j}']);defs[f'standing{j}']=float(row[f'standing{j}'])
                defs['base_patient_id']=int(defs['base_patient_id'])
            invoke(cfg,mode,defs,seed,pdir/f'retry_{retry}.log')
            meta=pd.read_csv(pdir/'meta.csv').iloc[-1]
            if meta['status']=='viable':
                x=pd.read_csv(pdir/'trajectory.csv')
                if str(x.iloc[-1]['status']) not in {'extinct','relapse','horizon'}:raise RuntimeError('Unfinished clinical trajectory')
                success=True;break
            if not str(meta['status']).startswith('pretreatment'): raise RuntimeError('Unexpected clinical initialization status')
        result={'patient_id':pid,'status':'finished' if success else 'failed_after_retries','seed_used':seed,'retries':retry}
        dump_json(pdir/'done.json',result); records.append(result)
    write_csv(pd.DataFrame(records),directory/'batches'/f'{batch:04d}.csv')
