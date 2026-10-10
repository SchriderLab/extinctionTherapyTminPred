#!/usr/bin/env python3
"""Small command-line dispatch for Snakemake rules; no work is done at import."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rescue.runtime import load_config,dump_json,execution_record,config_digest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--threads',type=int,default=1)
    ap.add_argument('action',choices=['preflight','manifests','expand','prefix','outcome','reference_batch','reference_table','cohort_table','frozen','tune','predict','evaluate','noise','plot','report','clinical_stage','clinical','clinical_design','clinical_batch','clinical_figures'])
    ap.add_argument('arguments',nargs='*');args=ap.parse_args();cfg=load_config(args.config);root=Path(cfg['run_dir']).resolve();a=args.arguments
    if args.action=='preflight':
        import os,platform
        host=platform.node().lower()
        if 'longleaf-login' in host and not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run computation through hpc/submit.sh or an allocated compute node.')
        p=cfg['primary']
        if cfg['abc']['scan_times']!=[0,1,2,3]:raise ValueError('This manuscript predictor uses exactly cycles 0–3.')
        if p['lags']!=list(range(1,len(p['lags'])+1)):raise ValueError('Use a contiguous lag grid starting at 1.')
        if not 0<=p['turnover'][0]<=p['turnover'][1]<.5:raise ValueError('Expansion requires turnover below 0.5.')
        if p['Q']<=0 or p['n_patients']<2 or p['n_replicates']<1:raise ValueError('Invalid simulation configuration')
        if cfg['clinical']['enabled']:
            from rescue.clinical import settings,staged_inputs
            cc=settings(cfg);missing=[k for k in staged_inputs(cfg) if not cc['inputs'].get(k) or not Path(cc['inputs'][k]).exists()]
            if missing:raise FileNotFoundError('Clinical inputs missing: '+', '.join(missing))
        if cfg['mode']=='regenerate' and cfg['reference']['design']=='frozen_manifest' and not Path(cfg['reference']['manifest']).is_file():raise FileNotFoundError('Run configure_inputs.py to select the original reference manifest, or choose config/fresh.yaml.')
        if cfg['mode']=='frozen':
            for k in ['cohort','reference']:
                if not Path(cfg['frozen'][k]).is_dir():raise FileNotFoundError(cfg['frozen'][k])
        identity=dict(cfg)
        if cfg['clinical']['enabled']:
            from rescue.clinical import settings
            identity['clinical_parameters']=settings(cfg)
        f=root/'run.json'
        if f.exists() and json.loads(f.read_text())['config_sha256']!=config_digest(identity):raise RuntimeError('Configuration changed inside an existing run directory. Choose a new run_dir; old results remain intact.')
        dump_json(f,execution_record(identity));return
    if args.action=='manifests':
        from rescue.design import manifests
        manifests(cfg,root/'design');return
    if args.action in ['expand','prefix','outcome']:
        import pandas as pd
        from rescue import simulate
        pid=int(a[0]);m=pd.read_csv(root/'design/patients.csv');row=m[m.patient_id==pid].iloc[0].to_dict();d=root/'simulation/patients'/f'{pid:03d}'
        if args.action=='outcome':simulate.outcome(cfg,row,d,int(a[1]),root/'simulation/outcomes'/f'{pid:03d}_{int(a[1]):02d}.csv')
        else:getattr(simulate, 'expansion' if args.action == 'expand' else 'prefix')(cfg,row,d)
        return
    if args.action=='reference_batch':
        from rescue.simulate import reference_batch
        reference_batch(cfg,root/'design/references.csv',int(a[0]),root/'simulation/references'/f'{int(a[0]):04d}.csv');return
    if args.action=='reference_table':
        from rescue.tables import assemble_reference
        import math
        fs=[root/'simulation/references'/f'{b:04d}.csv' for b in range(math.ceil(cfg['reference']['n']/cfg['reference']['batch_size']))]
        assemble_reference(cfg,fs,root/'reference');return
    if args.action=='cohort_table':
        from rescue.tables import assemble_patients
        n=cfg['primary']['n_patients'];ps=[root/'simulation/patients'/f'{p:03d}'/'prefix.csv' for p in range(n)];os=[root/'simulation/outcomes'/f'{p:03d}_{l:02d}.csv' for p in range(n) for l in cfg['primary']['lags']]
        assemble_patients(cfg,ps,os,root/'cohort');return
    if args.action=='frozen':
        from rescue.frozen import import_tables
        import_tables(cfg,root);return
    if args.action=='tune':
        from rescue.abc import tune
        tune(cfg,root/'reference/features.csv',root/'abc');return
    if args.action=='predict':
        from rescue.abc import patient_prediction
        patient_prediction(cfg,root/'reference/features.csv',root/'cohort/trajectories.csv',root/'abc/selected.json',root/'abc/predictions.csv');return
    if args.action in ['evaluate','noise']:
        from rescue import evaluate
        getattr(evaluate,args.action)(cfg,root);return
    if args.action=='plot':
        from rescue.plotting import render
        render(cfg,root);return
    if args.action=='report':
        from rescue.report import report
        report(cfg,root);return
    if args.action=='clinical_stage':
        from rescue.clinical import stage
        if not cfg['clinical']['enabled']:raise RuntimeError('Enable clinical in config/paper.yaml or your configuration.')
        stage(cfg);return
    if args.action=='clinical':
        from rescue.clinical import run_task
        run_task(cfg,a[0],args.threads);return
    if args.action in ['clinical_design','clinical_batch']:
        from rescue.clinical import settings,C,K,CLOUD
        from rescue.clinical_design import generate
        from rescue.simulate import clinical_batch
        cc=settings(cfg);key=a[0];p=cc[key];ws=Path(cfg['clinical']['workspace']).resolve();d=ws/(C if key=='calibration' else K);manifest=d/('patient_manifest.csv' if key=='calibration' else 'reference_manifest.csv')
        if args.action=='clinical_design':generate(ws/CLOUD,p,manifest)
        else:clinical_batch(cfg,manifest,int(a[1]),d,p)
        return
    if args.action=='clinical_figures':
        from rescue.figure_runner import clinical_figures
        clinical_figures(cfg);return
if __name__=='__main__':main()
