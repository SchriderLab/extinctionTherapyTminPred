"""Parameter designs, independent seed streams, and frozen-manifest import."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .runtime import write_csv, dump_json, execution_record, digest

ALIASES = {'dose_drug_1_mono':'dose_drug_1', 'dose_drug_2_mono':'dose_drug_2',
           'patient_turnover':'turnover_rate', 'expansion_seed_base':'expansion_seed'}

def normalized_manifest(df: pd.DataFrame) -> pd.DataFrame:
    x=df.copy()
    for old,new in ALIASES.items():
        if old in x and new not in x: x=x.rename(columns={old:new})
    return x

def lhs(n: int, dimensions: int, seed: int) -> np.ndarray:
    rng=np.random.default_rng(seed)
    return np.column_stack([(rng.permutation(n)+rng.random(n))/n for _ in range(dimensions)])

def sampled_design(p: dict, n: int, seed: int, expansion_base: int, d1_base: int, idcol: str) -> pd.DataFrame:
    u=lhs(n,5,seed)
    def scale(j,key): lo,hi=p[key]; return lo+u[:,j]*(hi-lo)
    d1=scale(1,'dose1'); m2=scale(3,'p_m2'); ids=np.arange(n,dtype=int)
    df=pd.DataFrame({idcol:ids,'base_mut_rate':scale(0,'mutation_rate'),
        'dose_drug_1':d1,'dose_drug_2':p['dose2_min']+u[:,4]*(d1-p['dose2_gap']-p['dose2_min']),
        'turnover_rate':scale(2,'turnover'),'draw_m1':1-m2-p['p_m3'],
        'draw_m2':m2,'draw_m3':p['p_m3'],'cross_resistance':p['cross_resistance'],
        'expansion_time':p['expansion_time'],'expansion_seed':expansion_base+ids,'d1_seed':d1_base+ids})
    return df

def validate(df: pd.DataFrame, idcol: str) -> None:
    required={idcol,'base_mut_rate','dose_drug_1','dose_drug_2','turnover_rate','draw_m1','draw_m2','draw_m3','cross_resistance','expansion_seed','d1_seed'}
    if required-set(df): raise ValueError(f'Manifest lacks {sorted(required-set(df))}')
    if df[idcol].duplicated().any(): raise ValueError('Duplicate IDs')
    for c in required:
        if not np.isfinite(pd.to_numeric(df[c],errors='coerce')).all(): raise ValueError(f'Invalid {c}')
    weights=df[['draw_m1','draw_m2','draw_m3']].to_numpy(float)
    if (weights<0).any() or not np.allclose(weights.sum(1),1): raise ValueError('Mutation weights must sum to one')
    for c in ['dose_drug_1','dose_drug_2','turnover_rate','cross_resistance']:
        if not df[c].between(0,1).all(): raise ValueError(f'{c} outside [0,1]')
    if not (df.dose_drug_2<df.dose_drug_1).all(): raise ValueError('Primary D2 must be weaker than D1')
    if not (df.base_mut_rate>=0).all(): raise ValueError('Negative mutation rate')
    for c in [idcol,'expansion_seed','d1_seed']:
        if not np.equal(df[c],np.rint(df[c])).all(): raise ValueError(f'{c} must be integral')

def manifests(cfg: dict, out: Path) -> None:
    p=cfg['primary']; r=cfg['reference']
    patients=sampled_design(p,int(p['n_patients']),int(p['lhs_seed']),int(p['expansion_seed_base']),int(p['d1_seed_base']),'patient_id')
    # The original patient generator writes 12 significant digits before SLiM reads the table.
    write_csv(patients,out/'patients.csv',float_format='%.12g')
    refs=None; provenance=execution_record(cfg)
    if r['design']=='frozen_manifest':
        f=Path(r['manifest'])
        if not f.is_file(): raise FileNotFoundError(f'{f}: import the original reference manifest with scripts/configure_inputs.py; use config/fresh.yaml for a new LHS realization.')
        refs=normalized_manifest(pd.read_csv(f))
        if len(refs)!=int(r['n']): raise ValueError(f'Expected {r["n"]} reference rows, got {len(refs)}')
        provenance['reference_manifest_sha256']=digest(f)
        if r.get('resolved_seeds') and Path(r['resolved_seeds']).is_file():
            s=pd.read_csv(r['resolved_seeds'])
            if {'ref_id','resolved_expansion_seed'}<=set(s):
                refs=refs.merge(s[['ref_id','resolved_expansion_seed']],on='ref_id',how='left',validate='one_to_one')
                provenance['reference_seed_table_sha256']=digest(Path(r['resolved_seeds']))
    elif r['design']=='lhs':
        refs=sampled_design(p,int(r['n']),int(r['lhs_seed']),int(r['expansion_seed_base']),int(r['d1_seed_base']),'ref_id')
        provenance['reference_design']='Fresh LHS; historical accepted counts and manuscript metrics are not enforced.'
    else: raise ValueError('reference.design must be frozen_manifest or lhs')
    validate(patients,'patient_id'); validate(refs,'ref_id')
    if set(patients.d1_seed)&set(refs.d1_seed): raise ValueError('Patient/reference D1 seeds overlap')
    if set(patients.expansion_seed)&set(refs.expansion_seed): raise ValueError('Patient/reference expansion seeds overlap')
    write_csv(refs,out/'references.csv')
    dump_json(out/'design.json',provenance)
