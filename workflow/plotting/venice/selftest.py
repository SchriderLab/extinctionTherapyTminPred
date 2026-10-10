#!/usr/bin/env python3
"""Synthetic software fixtures ONLY, not study results. Run on a compute node."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd
from plot_core import Blocked,col,boolean,unique

def build(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(__file__).with_name('inputs.json').read_text());cfg.update(cohort_expected_counts=None,fixed_scan_expected_n=40,dpi=100,test_fixture=True)
    rng=np.random.default_rng(1701);ids=[f'FIXTURE-{i:03d}' for i in range(60)]
    def save(key,d):
        path=folder/(key+'.csv');d.to_csv(path,index=False);cfg['sources'][key]['path']=str(path);cfg['sources'][key].pop('directories',None)
    save('adsl',pd.DataFrame({'USUBJID':ids,'RANDOM':'Y','ARMCD':'PLACEBO','TMPSA':100.}));labs=[];targets=[]
    for i,id in enumerate(ids):
        event=5+i%8;targets.append(event)
        for t in range(1,25):labs.append(dict(USUBJID=id,LBTESTCD='PSA',LBSTRESN=100.*10**(-.18*min(t,event)+.12*max(t-event,0)),ANLVISN=t,EPOCH='Treatment Phase',LBDY2=t*21,LBDY=t*21))
    save('adlb',pd.DataFrame(labs));targets=np.array(targets,float)
    save('direct_abc',pd.DataFrame({'USUBJID':ids,'supported':True,'within_horizon_observed':True,'observed_target_cycle':targets,'pred_abc_idw':targets+rng.normal(0,2,len(ids)),'pred_reference_median':7.}))
    save('early_clinical',pd.DataFrame({'USUBJID':ids,'observed':targets,'ridge':targets+rng.normal(0,2,len(ids)),'median_baseline':7.}))
    save('gp',pd.DataFrame({'USUBJID':ids,'observed':targets,'simulation_plus_gp':targets+rng.normal(0,2,len(ids)),'simulation_abc':targets+rng.normal(0,2,len(ids))}))
    save('exp_cloud',pd.DataFrame({'id':ids,'q_hat':rng.uniform(.001,.05,60),'decline_rate_hat':rng.uniform(.1,.7,60),'growth_rate_hat':rng.uniform(.1,.3,60),'rmse_log10':rng.uniform(.01,.2,60),'tmin_fit_hat':targets,'fit_tmin_in_observed_window':np.arange(60)%5!=0,'growth_near_lower_bound':np.arange(60)%5==0,'q_near_lower_bound':False}))
    save('exp_agreement',pd.DataFrame({'id':ids,'valid_nonboundary':True,'observed_nadir_cycle':targets,'exp_tmin':targets+rng.normal(0,.8,60)}))
    save('sim_fits',pd.DataFrame({'patient_id':range(100),'scenario':'broad_mix','fit_success':True,'mean_standing_realized':rng.uniform(.001,.04,100),'m1_effect_mean_target':rng.uniform(.1,.8,100),'patient_turnover':rng.uniform(.1,.3,100),'d_hat':rng.uniform(.05,1,100),'q_hat':rng.uniform(.001,.04,100),'g_hat':rng.uniform(.04,1,100)}))
    save('matches',pd.DataFrame({'clinical_id':ids,'clinical_d':rng.uniform(.05,1,60),'clinical_q':rng.uniform(.001,.04,60),'clinical_g':rng.uniform(.04,1,60),'support_class':['well_supported','edge_of_support','outside_support']*20}))
    scans=[];binary=[];eb=[]
    for k in range(2,7):
        for i,id in enumerate(ids[:40]):
            t=7+i%8;p=t+rng.normal(0,2);scans.append({'id':id,'scan_count':k,'global_fit_tmin':t,'pred_tmin_weighted':p})
            pr=float(np.clip(.15+.07*(t-7)+rng.normal(0,.1),.01,.99));binary.append({'id':id,'scan_count':k,'true_tmin':t,'true_late':int(t>8),'prob_late':pr,'probability_threshold':.5,'pred_late':int(pr>=.5)})
            eb.append({'id':id,'scan_count':k,'exp_tmin':t,'observed_nadir_cycle':t+.5,'pred_tmin_median':p,'posterior_q025':p-4,'posterior_q975':p+4})
    sd=pd.DataFrame(scans);save('scan_predictions',sd)
    save('scan_summary',pd.DataFrame([{'scan_count':k,'MAE':np.mean(abs(g.pred_tmin_weighted-g.global_fit_tmin)),'baseline_MAE':2.8} for k,g in sd.groupby('scan_count')]))
    save('scan_reference_cv',pd.DataFrame({'scan_count':range(2,7),'subset':'well_supported_future','reference_cv_MAE':np.linspace(3,1.8,5)}))
    save('binary',pd.DataFrame(binary));save('eb',pd.DataFrame(eb))
    save('qg_summary',pd.DataFrame([{'scan_count':k,'mode':m,'n':40,'MAE_exp_tmin':3-k*.15+(m=='qg_shuffled')*.2,'Spearman_exp_tmin':.1+k*.04,'AUROC_tmin_gt8':.6+k*.02} for k in range(2,7) for m in ['joint','qg_shuffled']]))
    save('covariates',pd.DataFrame({'feature_set':['relative_psa','baseline_only','relative_psa_plus_full_baseline'],'AUROC':[.73,.56,.65],'Brier':[.21,.26,.23],'balanced_accuracy_05':[.71,.55,.62]}))
    save('learning',pd.DataFrame({'mean_train_n_per_fold':[26,42,57,73,89,105],'train_fraction':[.25,.4,.55,.7,.85,1.],'repeats':[50]*5+[1],'AUROC_mean':[.68,.7,.72,.73,.74,.75],'AUROC_q025':[.6,.62,.64,.65,.66,.75],'AUROC_q975':[.76,.77,.78,.79,.8,.75]}))
    ps=[];raw=[]
    for i,id in enumerate(ids):
        event=12+i%5
        for t in range(event+5):raw.append({'id':id,'cycle':t,'calendar_day':21*t,'psa':100*10**(-.16*min(event,t)+.08*max(t-event,0))})
        for t in range(4,event+2):
            p=float(np.clip(.15+.022*(t-4)+rng.normal(0,.08),.01,.99));base=float(np.clip(.25+.004*t+rng.normal(0,.02),.01,.99))
            row=dict(id=id,risk_index=t,event_index=event,risk_cycle=t,risk_calendar_day=21*t,outer_fold=i%5+1,p_physics_reinforced=p,p_time_only=base,scans_before_event=event-t,state_velocity_mean=-.2+.01*t)
            ps.append({**row,'y':int(abs(t-event)<=1)})
    save('pm1',pd.DataFrame(ps));save('audit_psa',pd.DataFrame(raw).rename(columns={'id':'USUBJID'}))
    for key in ['vm_reg','mv_reg']:save(key,pd.DataFrame({'patient_id':ids,'target_t':targets,'prediction':targets+rng.normal(0,2,60),'source_median':7.}))
    for key in ['main_bin','vm_bin','mv_bin']:save(key,pd.DataFrame({'patient_id':ids,'target_t':targets,'late':(targets>=7).astype(int),'p_late':np.clip(.1+.07*(targets-4)+rng.normal(0,.09,60),.01,.99)}))
    save('rejection',pd.DataFrame({'scan_count':range(2,7),'n':40,'AUROC':np.linspace(.6,.77,5),'balanced_accuracy':.65,'early_recall':.6,'late_recall':.7,'weighted_ABC_AUROC':np.linspace(.61,.78,5),'weighted_ABC_balanced_accuracy':.67,'weighted_ABC_early_recall':.59,'weighted_ABC_late_recall':.75}))
    save('xgb',pd.DataFrame({'scan_count':range(2,7),'MAE':np.linspace(4,5,5),'reference_cv_MAE':np.linspace(4,2,5),'ABC_MAE':np.linspace(4,3.5,5),'Spearman_rho':.2,'ABC_Spearman_rho':np.linspace(.1,.4,5)}))
    save('downsampling',pd.DataFrame({'reference_size':[500,1000,2000,4000,6000,7856],'MAE_mean':[3.3,3.5,3.7,3.6,3.7,3.6],'MAE_q025':[3.1,3.2,3.4,3.5,3.5,3.6],'MAE_q975':[3.5,3.8,3.9,3.8,3.9,3.6],'Spearman_mean':np.linspace(.25,.45,6)}))
    config=folder/'fixture_inputs.json';config.write_text(json.dumps(cfg,indent=2));return config

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--out',required=True,type=Path);a.add_argument('--experiments',default='all');args=a.parse_args()
    for fn in [lambda: col(pd.DataFrame({'d':[1.],'d_hat':[2.]}),['d','d_hat'],'conflict'),lambda: boolean(pd.Series(['perhaps'])),lambda: unique(pd.DataFrame({'id':['x','x']}),['id'],'duplicate')]:
        try:fn()
        except Blocked:pass
        else:raise AssertionError('A negative source-validation test did not fail.')
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'SYNTHETIC_FIXTURES_ONLY.txt').write_text('SOFTWARE TEST OUTPUT ONLY. NOT SCIENTIFIC RESULTS. DO NOT USE IN THE MANUSCRIPT.\n')
    config=build(args.out/'fixtures')
    result=subprocess.run([sys.executable,str(Path(__file__).with_name('make_figures.py')),'--config',str(config),'--out',str(args.out/'plots'),'--experiments',args.experiments,'--dpi','100'])
    raise SystemExit(result.returncode)
if __name__=='__main__':main()
