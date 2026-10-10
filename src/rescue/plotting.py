"""Manuscript simulation figures from workflow tables; no hidden analysis reruns."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PRIMARY_FIGURES=['Fig2_patient_timing_landscapes','Fig4_ABC_nadir_prediction','Fig5_timing_regret',
 'Fig6_prediction_vs_reactive_monitoring','Fig_new_persistent_vs_NED_reactive',
 'Fig_new_observation_noise_biomarkers','FigS1_ABC_reference_support','Fig3_illustrative_patient_windows']

def render(cfg: dict,root: Path) -> None:
    out=root/'figures';out.mkdir(parents=True,exist_ok=True)
    t=pd.read_csv(root/'cohort/trajectories.csv');p=pd.read_csv(root/'cohort/patients.csv')
    e=pd.read_csv(root/'cohort/extinction.csv');a=pd.read_csv(root/'abc/predictions.csv');a=a[a.method=='logN_plus_slope']
    d=pd.read_csv(root/'evaluation/policy_by_patient.csv');d=d[d.method=='logN_plus_slope']
    reactive=pd.read_csv(root/'evaluation/reactive_by_patient.csv');summary=json.loads((root/'evaluation/summary.json').read_text())
    ns=pd.read_csv(root/'noise/summary.csv');nd=pd.read_csv(root/'noise/cohort_draws.csv');ref=pd.read_csv(root/'reference/features.csv')
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,
                        'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    def panels(n=3,rows=1):
        f,axs=plt.subplots(rows,n,figsize=(3.45*n,3.05*rows+.25),layout='constrained',squeeze=False)
        for j,ax in enumerate(axs.flat):ax.text(-.12,1.08,chr(65+j),transform=ax.transAxes,weight='bold',fontsize=12)
        return f,list(axs.flat)
    def save(f,name):
        for ext in ['pdf','png','svg']:f.savefig(out/f'{name}.{ext}',dpi=300,bbox_inches='tight')
        plt.close(f)
    def ci(vals,seed):
        rng=np.random.default_rng(seed);b=rng.choice(np.asarray(vals),size=(cfg['policies']['bootstrap'],len(vals)),replace=True).mean(1)
        return np.quantile(b,[.025,.975])
    f,ax=panels()
    for pid,g in t.groupby('patient_id'):
        ax[0].plot(g.t,np.log10(g.N+1),alpha=.25,lw=.6)
    ax[0].set(xlabel='Drug-1 cycle',ylabel=r'$\log_{10}(N+1)$',title='Heterogeneous Drug-1 responses')
    rel=e.merge(p[['patient_id','lag_at_tmin']],on='patient_id');rel['relative_t']=rel.lag-rel.lag_at_tmin
    z=rel.groupby('relative_t').extinction_probability.mean();ax[1].plot(z.index,z.values,'o-');ax[1].axvline(0,ls='--',lw=.8)
    ax[1].set(xlabel='Switch time relative to nadir',ylabel='Mean extinction probability',title='A nadir-associated switching region',ylim=(0,1))
    mat=e.pivot(index='patient_id',columns='lag',values='extinction_probability').sort_index()
    im=ax[2].imshow(mat,aspect='auto',origin='lower',vmin=0,vmax=1,extent=[mat.columns.min()-.5,mat.columns.max()+.5,-.5,len(mat)-.5])
    f.colorbar(im,ax=ax[2],label='Extinction probability',shrink=.85)
    ax[2].set(xlabel='Second-strike lag',ylabel='Patient',title='Patient-specific treatment outcomes');save(f,PRIMARY_FIGURES[0])
    f,ax=panels(2)
    lo=min(a.true_tmin.min(),a.pred_tmin_weighted.min())-.5;hi=max(a.true_tmin.max(),a.pred_tmin_weighted.max())+.5
    ax[0].scatter(a.true_tmin,a.pred_tmin_weighted,s=20,alpha=.7);ax[0].plot([lo,hi],[lo,hi],'--',lw=.8)
    ax[0].set(xlabel='True nadir cycle',ylabel='Predicted nadir cycle',title='Four early observations',xlim=(lo,hi),ylim=(lo,hi))
    ax[0].text(.04,.96,f'MAE = {summary["MAE_weighted"]:.3f} cycles',transform=ax[0].transAxes,va='top')
    err=a.pred_tmin_weighted-a.true_tmin;ax[1].hist(err,bins=12);ax[1].set(xlabel='Predicted minus true nadir (cycles)',ylabel='Patients',title='Timing error');save(f,PRIMARY_FIGURES[1])
    f,ax=panels(2)
    cols=['abc_extinction','tmin_extinction','oracle_extinction'];vals=d[cols].mean().to_numpy();cis=np.array([ci(d[c],100+j) for j,c in enumerate(cols)])
    ax[0].bar(range(3),vals,yerr=np.maximum(0,np.array([vals-cis[:,0],cis[:,1]-vals])),capsize=3)
    ax[0].set(xticks=range(3),xticklabels=['ABC','True nadir','Empirical best lag'],ylabel='Mean extinction probability',ylim=(0,1),title='Timing-policy outcomes')
    ax[1].hist(d.total_oracle_regret,bins=np.linspace(0,1,11));ax[1].set(xlabel='Empirical best-lag score minus ABC',ylabel='Patients',title='Patient-level regret');save(f,PRIMARY_FIGURES[2])
    f,ax=panels(3)
    for j,(column,label) in enumerate([('extinction','Mean extinction probability'),('delay','Switch time minus nadir (cycles)')]):
        rr=[]
        for k,g in reactive.groupby('n_scans'):rr.append((k,g[column].mean(),*ci(g[column],200+int(k))))
        rr=np.asarray(rr);ax[j].plot(rr[:,0],rr[:,1],'o-');ax[j].fill_between(rr[:,0],rr[:,2],rr[:,3],alpha=.15)
        ax[j].set(xlabel='Reactive observations',ylabel=label,title=['Reactive scan density','Detection delay'][j])
    ax[0].axhline(summary['abc_extinction'],ls='--',label='Four-scan ABC');ax[0].legend(frameon=False)
    dense=reactive[reactive.n_scans==reactive.n_scans.max()].merge(d[['patient_id','abc_extinction']],on='patient_id')
    ax[2].scatter(dense.extinction,dense.abc_extinction,alpha=.7);ax[2].plot([0,1],[0,1],'--',lw=.8)
    ax[2].set(xlabel='Dense reactive extinction probability',ylabel='Four-scan ABC extinction probability',title='Paired patient outcomes');save(f,PRIMARY_FIGURES[3])
    f,ax=panels(2,2)
    pid=39 if 39 in set(p.patient_id) else int(p.iloc[0].patient_id);g=t[t.patient_id==pid].sort_values('t');scans=np.asarray(cfg['policies']['threshold_scans'])
    ys=np.interp(scans,g.t,g.N);nadir_observed=ys.min()
    ax[0].plot(g.t,np.log10(g.N+1),lw=1.5);ax[0].scatter(scans,np.log10(ys+1),zorder=3)
    for multiplier in [cfg['policies']['persistent_multiplier'],cfg['policies']['ned_multiplier']]:
        ax[0].axhline(np.log10(multiplier*nadir_observed+1),ls='--',label=f'{multiplier:g} × sampled minimum')
    ax[0].set(xlim=(0,max(scans)),xlabel='Drug-1 cycle',ylabel=r'$\log_{10}(N+1)$',title=f'Reactive surveillance: patient {pid}');ax[0].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.2),ncol=1)
    z=ns[(ns.sigma==0)&(ns.n_biomarkers==1)].set_index('policy')
    values=[z.loc['Reactive_1.25x','mean_extinction'],z.loc['Reactive_NED_10x','mean_extinction'],summary['abc_extinction'],summary['true_nadir_extinction'],summary['empirical_best_lag']]
    ax[1].bar(range(5),values);ax[1].set(xticks=range(5),xticklabels=['1.25×','10×','ABC','True\nnadir','Best\nlag'],ylim=(0,1),ylabel='Within-window policy score',title='Noiseless comparison')
    for j,v in enumerate(values):ax[1].text(j,v+.015,f'{v:.3f}',ha='center',fontsize=8)
    for policy,label in [('Reactive_1.25x','1.25×; final-scan fallback'),('Reactive_NED_10x','10×; no forced switch')]:
        z=ns[ns.policy==policy];ax[2].plot(z.sigma,z.trigger_fraction,'o-',label=label);ax[3].plot(z.sigma,z.mean_switch_delay,'o-',label=label)
    ax[2].set(xlabel='Observation-noise SD',ylabel='Fraction triggering',title='Threshold crossings',ylim=(0,1));ax[2].legend(frameon=False)
    ax[3].set(xlabel='Observation-noise SD',ylabel='Mean switch delay (cycles)',title='Delay among scored switches')
    f.supxlabel('10×: untriggered cases score zero; its delay includes triggered cases only.',fontsize=8);save(f,PRIMARY_FIGURES[4])
    f,ax=panels(3)
    pid=int(p.iloc[0].patient_id);g=t[t.patient_id==pid];times=np.asarray(cfg['noise']['observation_times']);truth=np.log10(np.interp(times,g.t,g.N)+1)
    signals=truth[None,:]+np.random.default_rng(cfg['noise']['seed']+1).normal(0,.1,(3,len(times)))
    ax[0].plot(times,truth,label='Latent burden',lw=1.8)
    for j,sig in enumerate(signals):ax[0].scatter(times,sig,s=10,label=f'Biomarker {j+1}',alpha=.65)
    ax[0].set(xlabel='Drug-1 cycle',ylabel=r'$\log_{10}(N+1)$',title='Independent noisy observations');ax[0].legend(frameon=False)
    for m in cfg['noise']['biomarkers']:
        z=ns[(ns.policy=='ABC')&(ns.n_biomarkers==m)]
        for j,col in [(1,'tmin_MAE'),(2,'mean_extinction')]:
            ax[j].plot(z.sigma,z[col],'o-',label=f'{m} biomarker'+('s' if m>1 else ''))
            q=nd[(nd.policy=='ABC')&(nd.n_biomarkers==m)].groupby('sigma')[col].quantile([.025,.975]).unstack()
            ax[j].fill_between(q.index,q[.025],q[.975],alpha=.12)
    r=ns[ns.policy=='Reactive_1.25x'];ax[2].plot(r.sigma,r.mean_extinction,'--',label='1.25× reactive')
    ax[1].set(xlabel='Noise SD on log burden',ylabel='Nadir MAE (cycles)',title='Timing accuracy');ax[1].legend(frameon=False)
    ax[2].set(xlabel='Noise SD on log burden',ylabel='Mean extinction probability',title='Treatment outcome');ax[2].legend(frameon=False)
    save(f,PRIMARY_FIGURES[5])
    f,ax=panels(2);ax[0].hist(ref.t_min_true,bins=np.arange(3.5,cfg['primary']['prefix_length']+.5,1));ax[0].set(xlabel='Reference nadir cycle',ylabel='Trajectories',title='Independent reference support')
    ax[1].scatter(ref.logN0,ref.logN3,s=2,alpha=.12,label='References');ax[1].scatter(a.logN0,a.logN3,s=12,label='Patients');ax[1].set(xlabel='log burden at cycle 0',ylabel='log burden at cycle 3',title='Early-observation coverage');ax[1].legend(frameon=False);save(f,PRIMARY_FIGURES[6])
    f,ax=panels(2,2)
    candidates=p.sort_values('oracle_p_ext',ascending=False).head(2).patient_id.tolist()
    for j,pid in enumerate(candidates):
        g=t[t.patient_id==pid];v=e[e.patient_id==pid];pp=p[p.patient_id==pid].iloc[0]
        ax[j].plot(g.t,np.log10(g.N+1));ax[j].axvline(pp.tmin,ls='--');ax[j].set(xlabel='Drug-1 cycle',ylabel='log burden',title=f'Patient {pid}')
        ax[j+2].plot(v.lag,v.extinction_probability,'o-');ax[j+2].axvline(pp.lag_at_tmin,ls='--');ax[j+2].set(xlabel='Second-strike lag',ylabel='Extinction probability',ylim=(0,1))
    save(f,PRIMARY_FIGURES[7])
