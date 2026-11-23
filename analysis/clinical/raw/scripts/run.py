#!/usr/bin/env python3
from common import *
import argparse,subprocess,shutil,traceback,zipfile,datetime,importlib.metadata

def preflight(c,out):
    from xgboost import XGBRegressor
    # Fail before simulations if the installed XGBoost cannot run absolute-error loss.
    XGBRegressor(objective='reg:absoluteerror',n_estimators=2,n_jobs=1).fit(np.arange(12)[:,None],np.arange(12)%4)
    slim=shutil.which(c['slim'])
    if not slim:raise FileNotFoundError('SLiM executable unavailable; activate the existing extinction-paper environment or set slim in config.json.')
    p=subprocess.run([slim,'-v'],capture_output=True,text=True,check=True)
    versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','xgboost','matplotlib','joblib','PyYAML']}
    dump(out/'ENVIRONMENT.json',dict(python=sys.version,platform=platform.platform(),slim=p.stdout+p.stderr,packages=versions))

def collect(c,out):
    dest=out/'VENICE_TREB_RESULTS.zip'
    folders=['cohort','training','evaluation','predictions','results','figures','steps']
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in folders:
            for p in sorted((out/folder).rglob('*')):
                if p.is_file():z.write(p,p.relative_to(out))
        for name in ['CONFIG_FROZEN.json','ENVIRONMENT.json','SOURCE_HASHES.json','PREPARED_INPUTS_LOCK.json']:
            if (out/name).exists():z.write(out/name,name)
        for name in ['reference_manifest.csv','reference_manifest.design.json','DESIGN.json','PRESET.json',
                     'full_fit_parameters.csv','attrition.csv','REFERENCE_META.json','REFERENCE_LOCK.json']:
            if (out/'simulation'/name).exists():z.write(out/'simulation'/name,'simulation/'+name)
    print('Results package:',dest,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--stage',required=True,choices=['prepare','prepare_venice','prepare_mainsail','simulate','assemble','fixed','empirical','clinical_checks','reference_size','cross_trial','sequential','report','analysis','collect'])
    ap.add_argument('--batch',type=int,default=0);args=ap.parse_args();c,out=config(args.config)
    if args.stage!='collect':require_compute()
    frozen=out/'CONFIG_FROZEN.json'
    if frozen.exists() and json.loads(frozen.read_text())!=plain(c):raise RuntimeError('Configuration changed. Use a NEW output directory; do not reuse this split/run.')
    if not frozen.exists():dump(frozen,c)
    source={str(p.relative_to(PACKAGE)):digest(p) for folder in ['scripts','vendor'] for p in (PACKAGE/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.slim','.yaml']}
    src=out/'SOURCE_HASHES.json'
    if src.exists() and json.loads(src.read_text())!=source:raise RuntimeError('Source changed for an existing run. Use a new output directory.')
    if not src.exists():dump(src,source)
    prepared=out/'PREPARED_INPUTS_LOCK.json'
    if args.stage not in ['prepare','prepare_venice','prepare_mainsail','collect']:
        if not prepared.exists():raise RuntimeError('Preparation lock missing; finish both VENICE and MAINSAIL preparation first.')
        for relative,sha in json.loads(prepared.read_text()).items():
            if not (out/relative).exists() or digest(out/relative)!=sha:
                raise RuntimeError('Prepared input changed: '+relative)
    import prepare,simulation,predict,empirical,sequential,report
    jobs=max(1,int(os.environ.get('SLURM_CPUS_PER_TASK','1')))
    funcs={'prepare_venice':lambda:prepare.venice(c,out,jobs),'prepare_mainsail':lambda:prepare.mainsail(c,out),
        'simulate':lambda:simulation.batch(c,out,args.batch),'assemble':lambda:simulation.assemble(c,out,jobs),
        'fixed':lambda:predict.fixed(c,out),'empirical':lambda:empirical.run(c,out),
        'clinical_checks':lambda:predict.clinical_checks(c,out),'reference_size':lambda:predict.reference_size(c,out),
        'cross_trial':lambda:predict.cross_trial(c,out),
        'sequential':lambda:sequential.run(c,out),'report':lambda:report.run(c,out),'collect':lambda:collect(c,out)}
    stages=['prepare_venice','prepare_mainsail'] if args.stage=='prepare' else (
        ['assemble','fixed','empirical','clinical_checks','reference_size','sequential','cross_trial','report','collect'] if args.stage=='analysis' else [args.stage])
    if args.stage=='prepare':
        preflight(c,out)
        from recover import recover_venice
        recover_venice(c,out)
    for stage in stages:
        key=stage+(f'_{args.batch:04d}' if stage=='simulate' else '')
        record=dict(stage=stage,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        try:
            if stage=='report':dump(out/'predictions/ALL_PREDICTIONS_LOCK.json',
                {p.name:digest(p) for p in sorted((out/'predictions').glob('*.csv'))})
            funcs[stage]()
            if stage=='prepare_mainsail':
                files=[p for directory in ['cohort','training','queries','evaluation'] for p in (out/directory).glob('*.csv')]
                frozen_inputs={str(p.relative_to(out)):digest(p) for p in sorted(files)}
                if prepared.exists() and json.loads(prepared.read_text())!=frozen_inputs:
                    raise RuntimeError('Prepared input lock would change. Use a new run directory.')
                if not prepared.exists():dump(prepared,frozen_inputs)
            record['status']='SUCCESS'
        except Exception as exc:
            record.update(status='FAILED',error=str(exc));dump(out/'steps'/f'{key}.json',record);raise
        record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();dump(out/'steps'/f'{key}.json',record)
    if args.stage=='analysis':collect(c,out) # include the completion records themselves

if __name__=='__main__':main()
