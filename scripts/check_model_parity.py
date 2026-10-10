#!/usr/bin/env python3
"""Compare shared-engine outputs to supplied original SLiM models on a compute node.

Original models are read from the private source archive into a temporary test
folder; they are not additional maintained/public simulation implementations.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,tempfile,zipfile
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rescue.runtime import load_config,run_command,dump_json
from rescue.simulate import base_defs,eidos,invoke

NAMES={'expand':'hybrid_regen80_expand.slim','prefix':'hybrid_regen80_prefix.slim','outcome':'hybrid_regen80_outcome_extend300.slim',
       'clinical_single':'venice_singlepop_m1fitness_v1.slim','clinical_compartments':'venice_multicolony_m1fitness_v1.slim'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-archive',type=Path,required=True);ap.add_argument('--slim',required=True)
    ap.add_argument('--out',type=Path,default=Path('results/runs/model_parity'));ap.add_argument('--full-size',action='store_true')
    args=ap.parse_args();out=args.out.resolve()
    if out.exists():raise FileExistsError('Use a fresh --out directory for the parity check')
    out.mkdir(parents=True);cfg=load_config(ROOT/'config/primary.yaml');cfg['slim']=args.slim;cfg['primary']['expansion_time']=20 if args.full_size else 10
    seedbase=680000001;records=[]
    with tempfile.TemporaryDirectory(prefix='original_models_',dir=out) as temp:
        temp=Path(temp);old={}
        with zipfile.ZipFile(args.source_archive) as z:
            for phase,name in NAMES.items():
                content=z.read('sources/cancer_sims/sims/'+name).decode();sha=hashlib.sha256(content.encode()).hexdigest()
                if not args.full_size and phase in ['expand','prefix','outcome']:
                    expected='defineConstant("expansion_time", 20);'
                    if content.count(expected)!=1:raise RuntimeError('Unexpected original expansion constant')
                    content=content.replace(expected,'defineConstant("expansion_time", 10);')
                f=temp/name;f.write_text(content);old[phase]=f;records.append({'phase':phase,'source_sha256':sha})
        def run(which,phase,defs,seed,log):
            if which=='new':invoke(cfg,phase,defs,seed,log)
            else:
                # Remove configurable constants that the original declares unconditionally.
                omit={'Q','expansion_time','treatment_time','relapse_cap_factor','m2_meanlog','m2_sdlog'}
                if phase in ['expand','prefix','outcome']:omit|={'prefix_length','m1_meanlog','m1_sdlog'}
                cmd=[args.slim,'-s',str(seed)]
                for k,v in defs.items():
                    if k not in omit:cmd+=['-d',f'{k}={eidos(v)}']
                cmd.append(str(old[phase]));run_command(cmd,log)
        def equal(a,b,label):
            x=pd.read_csv(a);y=pd.read_csv(b);pd.testing.assert_frame_equal(x,y,check_dtype=False,rtol=0,atol=0)
            records.append({'comparison':label,'rows':len(x),'result':'exact_csv_match'})
        row={'patient_id':0,'base_mut_rate':5e-8 if args.full_size else 5e-7,'dose_drug_1':.7,'dose_drug_2':.6,'turnover_rate':.2,
             'draw_m1':.45,'draw_m2':.45,'draw_m3':.1,'cross_resistance':.3,'d1_seed':780000001,'expansion_seed':seedbase}
        d=base_defs(cfg,row)
        for attempt in range(20):
            seed=seedbase+10000*attempt
            for which in ['old','new']:
                path=out/which;path.mkdir(exist_ok=True)
                defs={**d,'snapshot_file':path/'snapshot.bin','meta_file':path/'meta.csv','expansion_seed':seed}
                run(which,'expand',defs,seed,path/'expand.log')
            equal(out/'old/meta.csv',out/'new/meta.csv','expansion_metadata')
            if pd.read_csv(out/'old/meta.csv').iloc[0]['status']=='viable':break
        else:raise RuntimeError('No viable parity-test expansion')
        for which in ['old','new']:
            path=out/which;defs={**d,'snapshot_file':path/'snapshot.bin','prefix_file':path/'prefix.csv','d1_seed':row['d1_seed']}
            run(which,'prefix',defs,row['d1_seed'],path/'prefix.log')
        equal(out/'old/prefix.csv',out/'new/prefix.csv','complete_Drug1_history')
        for lag in [1,6,12]:
            for rep in [1,2]:
                for which in ['old','new']:
                    path=out/which;defs={**d,'snapshot_file':path/'snapshot.bin','outcome_file':path/f'outcome_{lag}_{rep}.csv',
                      'd1_seed':row['d1_seed'],'second_strike_lag':lag,'replicate_id':rep,'treatment_seed':1000000000+lag*100+rep}
                    run(which,'outcome',defs,row['d1_seed'],path/f'outcome_{lag}_{rep}.log')
                equal(out/'old'/f'outcome_{lag}_{rep}.csv',out/'new'/f'outcome_{lag}_{rep}.csv',f'outcome_lag{lag}_rep{rep}')
        for phase in ['clinical_single','clinical_compartments']:
            for attempt in range(20):
                seed=4200000100+attempt
                for which in ['old','new']:
                    path=out/which;defs={**d,'base_patient_id':0,'scenario_name':'parity','target_decay_rate':.4,
                     'm1_meanlog':-.8,'m1_sdlog':.3,'m1_effect_mean_target':.47,'patient_seed':seed,
                     'prefix_length':30,'patient_relapse_factor':4.,'output_file':path/f'{phase}.csv','meta_file':path/f'{phase}_meta.csv'}
                    if phase=='clinical_single':defs.update(source_k5_patient_id=-1,single_age=12,n_founders=5,standing_target=.05)
                    else:
                        for j in range(1,6):defs[f'age{j}']=10+j%3;defs[f'standing{j}']=.05
                    run(which,phase,defs,seed,path/f'{phase}.log')
                equal(out/'old'/f'{phase}_meta.csv',out/'new'/f'{phase}_meta.csv',phase+'_initialization')
                if pd.read_csv(out/'old'/f'{phase}_meta.csv').iloc[0]['status']=='viable':
                    equal(out/'old'/f'{phase}.csv',out/'new'/f'{phase}.csv',phase+'_history');break
            else:raise RuntimeError(f'No viable {phase} parity case')
    dump_json(out/'parity_report.json',{'status':'PASS','full_size':args.full_size,'comparisons':records})
    print('PASS:',out/'parity_report.json')
if __name__=='__main__':main()
