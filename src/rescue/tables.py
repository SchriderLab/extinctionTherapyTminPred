"""Strict outcome assembly and reference features from treatment trajectories."""
from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from .runtime import write_csv, dump_json

def trajectory(df: pd.DataFrame) -> pd.DataFrame:
    tcol=next((k for k in ['t_treatment','t','treatment_t'] if k in df),None)
    if tcol is None: raise ValueError('Explicit treatment time is required; absolute cycle is not silently reinterpreted.')
    x=df.rename(columns={tcol:'t'}).copy()
    for c in ['t','N']:x[c]=pd.to_numeric(x[c],errors='raise')
    if not np.isfinite(x[['t','N']]).all().all() or (x.N<0).any():raise ValueError('Invalid trajectory values')
    if x.t.duplicated().any():raise ValueError('Duplicate treatment times')
    return x.sort_values('t').reset_index(drop=True)

def reference_features(long: pd.DataFrame, scans: list[int]) -> tuple[pd.DataFrame,pd.DataFrame]:
    records=[]
    for rid,g in long.groupby('ref_id',sort=True):
        x=trajectory(g);j=int(x.N.idxmin());tm=float(x.loc[j,'t']);last=float(x.t.max())
        rec={'ref_id':int(rid),'n_points':len(x),'max_t':last,'t_min_true':tm,'N_min':float(x.loc[j,'N']),
             'tmin_boundary':int(np.isclose(tm,last)),'usable':0,'problem':''}
        if not set(scans)<=set(x.t):rec['problem']='missing_early_observation'
        elif tm<=max(scans):rec['problem']='nadir_not_after_early_window'
        else:
            y=x.set_index('t').loc[scans,'N'].to_numpy(float)
            for k,(n,z) in enumerate(zip(y,np.log10(y+1))):rec[f'N{k}']=n;rec[f'logN{k}']=z
            for k,z in enumerate(np.diff(np.log10(y+1))):rec[f'dlog{k}{k+1}']=z
            rec['usable']=1
        payload='\n'.join(f'{t:.12g},{n:.12g}' for t,n in x[['t','N']].itertuples(index=False,name=None))
        rec['trajectory_hash']=hashlib.sha256(payload.encode()).hexdigest()
        records.append(rec)
    allref=pd.DataFrame(records);clean=allref[(allref.usable==1)&(allref.tmin_boundary==0)].copy()
    return allref,clean

def assemble_reference(cfg: dict, files: list[Path], out: Path) -> None:
    long=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
    if long.ref_id.nunique()!=int(cfg['reference']['n']):raise ValueError('Reference count does not match attempted design')
    allref,clean=reference_features(long,cfg['abc']['scan_times'])
    if len(clean)<max(2,int(cfg['abc']['folds'])):raise ValueError('Too few usable future/interior reference nadirs')
    write_csv(allref,out/'features_all.csv');write_csv(clean,out/'features.csv');write_csv(long,out/'trajectories.csv')
    dump_json(out/'summary.json',{'attempted':len(allref),'early_usable':int(allref.usable.sum()),'boundary_removed':int(((allref.usable==1)&(allref.tmin_boundary==1)).sum()),'clean':len(clean)})

def assemble_patients(cfg: dict, prefixes: list[Path], outcomes: list[Path], out: Path) -> None:
    frames=[]
    for f in prefixes:
        x=trajectory(pd.read_csv(f)); frames.append(x)
    long=pd.concat(frames,ignore_index=True)
    raw=pd.concat([pd.read_csv(f) for f in outcomes],ignore_index=True)
    expected=int(cfg['primary']['n_patients'])*len(cfg['primary']['lags'])*int(cfg['primary']['n_replicates'])
    if len(raw)!=expected or raw[['patient_id','lag','replicate']].duplicated().any():raise ValueError('Missing/duplicate outcome keys')
    allowed={'extinct','pre_switch_extinct','relapse','pre_switch_relapse','unresolved'}
    if not set(raw.status)<=allowed:raise ValueError('Unknown outcome status')
    raw['extinct'] = raw.status.isin(['extinct','pre_switch_extinct']).astype(int)
    e=raw.groupby(['patient_id','lag'],as_index=False).agg(n_runs=('extinct','size'),n_extinct=('extinct','sum'))
    if not e.n_runs.eq(int(cfg['primary']['n_replicates'])).all():raise ValueError('Unequal replicate counts')
    e['extinction_probability']=e.n_extinct/e.n_runs
    chars=[]
    for pid,g in long.groupby('patient_id',sort=True):
        g=g.sort_values('t');r=g.loc[g.N.idxmin()]; tm=float(r.t); profile=e[e.patient_id==pid].set_index('lag').extinction_probability
        lag=int(np.rint(tm))+1
        if lag not in profile.index:raise ValueError(f'Patient {pid} nadir lag {lag} has no counterfactual outcome; expand the configured lag grid.')
        best=profile[profile==profile.max()].index
        char={'patient_id':int(pid),'tmin':tm,'lag_at_tmin':lag,'N_start':float(g.iloc[0].N),'N_min':float(r.N),
              'N_end':float(g.iloc[-1].N),'oracle_p_ext':float(profile.max()),'p_ext_at_tmin':float(profile.loc[lag]),
              'oracle_lag_earliest':int(min(best)),'oracle_lag_latest':int(max(best)),
              'mean_R2_eff_at_tmin':float(r.get('mean_R2_eff',np.nan))}
        chars.append(char)
    write_csv(long,out/'trajectories.csv');write_csv(raw,out/'outcomes.csv');write_csv(e,out/'extinction.csv');write_csv(pd.DataFrame(chars),out/'patients.csv')
