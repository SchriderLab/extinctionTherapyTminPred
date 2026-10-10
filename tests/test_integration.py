"""Small deterministic tables for software integration; not biological simulations."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rescue.runtime import PACKAGE,load_config
from rescue.tables import reference_features
from rescue.abc import tune,patient_prediction
from rescue.evaluate import evaluate,noise
from rescue.report import report
from rescue.plotting import render,PRIMARY_FIGURES

def make_tables(root,cfg):
    (root/'cohort').mkdir();(root/'reference').mkdir()
    records=[];chars=[];out=[]
    for pid in range(3):
        t=np.arange(20);n=10000*np.exp(-(.65+.04*pid)*t)+(1+pid*.1)*np.exp(.50*t)
        tm=int(t[np.argmin(n)]);chars.append({'patient_id':pid,'tmin':tm,'lag_at_tmin':tm+1,'oracle_p_ext':1.,'p_ext_at_tmin':1.,'oracle_lag_earliest':tm+1})
        records.extend({'patient_id':pid,'t':int(ti),'N':float(ni)} for ti,ni in zip(t,n))
        for lag in range(1,21):
            e=float(np.round(np.exp(-((lag-(tm+1))**2)/12),1));out.append({'patient_id':pid,'lag':lag,'n_runs':10,'n_extinct':int(e*10),'extinction_probability':e})
    pd.DataFrame(records).to_csv(root/'cohort/trajectories.csv',index=False);pd.DataFrame(chars).to_csv(root/'cohort/patients.csv',index=False);pd.DataFrame(out).to_csv(root/'cohort/extinction.csv',index=False)
    rr=[]
    for rid in range(50):
        t=np.arange(20);n=(8000+rid*150)*np.exp(-(.45+rid*.008)*t)+(1+rid*.015)*np.exp(.5*t)
        rr.extend({'ref_id':rid,'t':int(ti),'N':float(ni)} for ti,ni in zip(t,n))
    allref,clean=reference_features(pd.DataFrame(rr),[0,1,2,3]);clean.to_csv(root/'reference/features.csv',index=False)

def test_saved_table_pipeline_through_all_simulation_figures(tmp_path):
    cfg=load_config(PACKAGE/'config/smoke.yaml');cfg['run_dir']=str(tmp_path);cfg['policies']['bootstrap']=5
    make_tables(tmp_path,cfg)
    tune(cfg,tmp_path/'reference/features.csv',tmp_path/'abc')
    patient_prediction(cfg,tmp_path/'reference/features.csv',tmp_path/'cohort/trajectories.csv',tmp_path/'abc/selected.json',tmp_path/'abc/predictions.csv')
    evaluate(cfg,tmp_path);noise(cfg,tmp_path);report(cfg,tmp_path);render(cfg,tmp_path)
    assert len(list((tmp_path/'figures').glob('*.pdf')))==len(PRIMARY_FIGURES)
    s=pd.read_csv(tmp_path/'noise/summary.csv');assert len(s)==10
    assert s.mean_extinction.between(0,1).all()
    assert (tmp_path/'report/RESULTS.md').is_file()
