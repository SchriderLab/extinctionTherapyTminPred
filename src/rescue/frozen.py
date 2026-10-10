"""Read explicitly selected simulation tables from the original publication layout."""
from pathlib import Path
import shutil
import pandas as pd
from .tables import trajectory
from .runtime import write_csv,dump_json,digest

def import_tables(cfg: dict,root: Path) -> None:
    c=Path(cfg['frozen']['cohort']);r=Path(cfg['frozen']['reference'])
    names={'extinction.csv':'extinction_by_lag_binary300.csv','patients.csv':'patient_oracle_characterization.csv','outcomes.csv':'all_outcomes_resolved_to300.csv'}
    sources={}
    for dest,source in names.items():
        f=c/source
        if not f.is_file():raise FileNotFoundError(f)
        x=pd.read_csv(f)
        if 'patient_id' not in x:raise ValueError(f'Not a simulation table: {f}')
        write_csv(x,root/'cohort'/dest);sources[str(f)]=digest(f)
    p=pd.read_csv(root/'cohort/patients.csv')
    if len(p)!=cfg['primary']['n_patients']:raise ValueError('Frozen patient count differs from configuration')
    curves=[]
    for pid in sorted(p.patient_id.astype(int)):
        f=c/'patients'/f'patient_{pid:03d}'/'drug1_prefix.csv'
        if not f.is_file():raise FileNotFoundError(f)
        x=trajectory(pd.read_csv(f));x['patient_id']=pid;curves.append(x);sources[str(f)]=digest(f)
    write_csv(pd.concat(curves,ignore_index=True),root/'cohort/trajectories.csv')
    f=r/'abc_reference_features_usable.csv'
    x=pd.read_csv(f);clean=x[pd.to_numeric(x.get('tmin_boundary',0),errors='raise')!=1].copy() if 'tmin_boundary' in x else x.copy()
    if 'usable' in clean:clean=clean[clean.usable==1]
    write_csv(clean,root/'reference/features.csv');write_csv(x,root/'reference/features_all.csv')
    dump_json(root/'reference/summary.json',{'supplied_features':len(x),'clean':len(clean),'source':str(f),'sha256':digest(f)})
    dump_json(root/'cohort/frozen_provenance.json',sources)
