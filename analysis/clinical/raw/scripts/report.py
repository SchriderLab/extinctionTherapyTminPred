from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def binary(y,p,thr=.5,cutoff=8):
    y=np.asarray(y);p=np.asarray(p,float);ok=np.isfinite(p)&np.isfinite(y);y=(y[ok]>cutoff).astype(int);p=p[ok]
    if not len(p):return {'n_binary':0}
    if np.any(p < -1e-8) or np.any(p > 1+1e-8):raise ValueError('Invalid probability outside floating-point tolerance')
    p=np.clip(p,0,1)
    z=dict(n_binary=len(p),n_early=int(sum(y==0)),n_late=int(sum(y==1)),Brier=brier_score_loss(y,p))
    if len(np.unique(y))==2:
        z.update(AUROC=roc_auc_score(y,p),balanced_accuracy=balanced_accuracy_score(y,p>=thr),
            early_recall=np.mean(p[y==0]<thr),late_recall=np.mean(p[y==1]>=thr))
    return z

def read_targets(out):
    v=pd.read_csv(out/'evaluation/heldout_reference_PRIVATE.csv',dtype={'id':str});v['dataset']='venice'
    m=pd.read_csv(out/'evaluation/mainsail_reference_PRIVATE.csv',dtype={'id':str});m['dataset']='mainsail'
    return pd.concat([v,m],ignore_index=True)

def run(c,out):
    pdir=out/'predictions';lock=json.loads((pdir/'FIXED_LOCK.json').read_text())
    if lock['prediction_sha256']!=digest(pdir/'fixed_PRIVATE.csv'):raise RuntimeError('Predictions changed after lock')
    target=read_targets(out);files=[pdir/n for n in ['fixed_PRIVATE.csv','empirical_PRIVATE.csv','covariates_PRIVATE.csv','cross_trial_PRIVATE.csv']]
    files+=sorted(pdir.glob('early_fit_*_PRIVATE.csv'))
    pred=pd.concat([pd.read_csv(p,dtype={'id':str}) for p in files],ignore_index=True)
    if pred.duplicated(['dataset','id','scan_count','model']).any():raise RuntimeError('Duplicate prediction keys')
    pred=pred.merge(target[['dataset','id','Treb']],on=['dataset','id'],how='left',validate='many_to_one')
    if pred.Treb.isna().any():raise RuntimeError('Prediction lacks endpoint')
    fixed=pred[(pred.model=='Constant_TRAIN')&(pred.scan_count==6)&(pred.Treb>5)][['dataset','id']]
    common=set(zip(fixed.dataset,fixed.id));rows=[]
    for (dataset,n,model),z in pred.groupby(['dataset','scan_count','model']):
        groups={'all_assessable':z,'future_Treb':z[z.Treb>n-1],
            'fixed_future6':z[[tuple(x) in common for x in z[['dataset','id']].to_numpy()]]}
        for stratum,g in groups.items():
            row=dict(dataset=dataset,scan_count=n,model=model,stratum=stratum,n_target=len(g),n_missing_forecast=int(g.pred.isna().sum()))
            row.update(metrics(g.Treb,g.pred));row.update(binary(g.Treb,g.prob_late8,float(g.threshold.iloc[0]) if len(g) else .5))
            if {'lo','hi'}<=set(g):
                valid=g[['lo','hi']].notna().all(axis=1)
                row.update(interval_n=int(valid.sum()),interval_coverage=np.mean((g.loc[valid,'Treb']>=g.loc[valid,'lo'])&(g.loc[valid,'Treb']<=g.loc[valid,'hi'])) if valid.any() else np.nan,
                    median_interval_width=np.median(g.loc[valid,'hi']-g.loc[valid,'lo']) if valid.any() else np.nan)
            rows.append(row)
    scores=pd.DataFrame(rows);csv(out/'results/METRICS.csv',scores)
    csv(out/'evaluation/scored_predictions_PRIVATE.csv',pred)
    agreement(c,out);paired(c,out,pred);learning(c,out,target);seq_report(c,out,target)
    figures(c,out,scores)
    size=pd.read_csv(pdir/'reference_size_PRIVATE.csv',dtype={'id':str}).merge(target[['dataset','id','Treb']],on=['dataset','id'],validate='many_to_one')
    srows=[]
    for (dataset,nn,rep),g in size.groupby(['dataset','reference_n','repeat']):
        g=g[g.Treb>5]
        srows.append(dict(dataset=dataset,reference_n=nn,repeat=rep,**metrics(g.Treb,g.pred),**binary(g.Treb,g.prob_late8)))
    csv(out/'results/REFERENCE_SIZE.csv',pd.DataFrame(srows))
    dump(out/'results/COMPLETION.json',dict(status='SUCCESS',target='Treb',prediction_lock=lock,
        interpretation='Internal held-out reanalysis of a retrospective fit-eligible VENICE cohort; MAINSAIL cross-trial reanalysis. No independent untouched validation claim.',
        primary_exact='Future_Treb at each budget; fixed_future6 for comparing budgets on identical patients. All-assessable results include patients whose event is already observable.',
        primary_binary='Treb>8 in BOTH cohorts; report future-Treb and all-assessable strata, with counts.',
        uncertainty='Patient bootstrap conditional on frozen fitted models; does not include model-selection or simulation-generation uncertainty.'))
    selected=scores[(scores.dataset=='venice')&(scores.scan_count.isin([4,6]))&(scores.stratum=='future_Treb')&
        scores.model.isin(['Constant_TRAIN','ABC_noisy_joint','ABC_GP','XGBoost_absolute','Clinical_PSA','EB_Treb_joint'])]
    text='Completed internal held-out VENICE reanalysis; this is not untouched independent validation.\nClinical target: first local low after cycle 3 confirmed by ONE strict rise (Treb).\nTfit labels occur only in explicitly named proxy/fit-agreement outputs.\n\n'
    text+=selected[['scan_count','model','n_target','n','MAE','Spearman_rho','AUROC']].to_string(index=False)+'\n\n'
    text+='All-parent and future-target subsets are in METRICS.csv. Fixed-future6 gives an identical patient set across measurement budgets.\nRead Tfit_vs_Treb_agreement.csv for the retrospective fit comparison, and PAIRED_MAE_REDUCTION.csv for paired uncertainty.\nThe source cohort, full-fit diagnostics, model settings, prediction hashes and attrition are included.\n'
    (out/'results/READ_ME_RESULTS.txt').write_text(text)

def agreement(c,out):
    parent=pd.read_csv(out/'cohort/PARENT_231_PRIVATE.csv');rows=[]
    for name,g in [('TRAIN',parent[parent.split=='TRAIN']),('TEST',parent[parent.split=='TEST']),('ALL_descriptive',parent)]:
        for subset,z in [('all_parent',g),('fitted_nadir_in_window',g[bools(g.fit_tmin_in_observed_window)])]:
            rows.append(dict(split=name,subset=subset,negative_Tfit=int(sum(z.tmin_fit_hat<0)),
                **metrics(z.Treb,z.tmin_fit_hat)))
    csv(out/'results/Tfit_vs_Treb_agreement.csv',pd.DataFrame(rows))
    # Ratios describe fitted shape; neither a U shape nor fitted q is proof of evolutionary rescue.
    dump(out/'results/FIT_SHAPE_COUNTS.json',dict(parent_n=len(parent),positive_in_window_n=int(bools(parent.fit_tmin_in_observed_window).sum()),
        negative_n=int(sum(parent.tmin_fit_hat<0)),beyond_followup_n=int(sum(parent.tmin_fit_hat>parent.last_cycle)),
        rmse_log10_quantiles=parent.rmse_log10.quantile([0,.25,.5,.75,1]).to_dict()))

def paired(c,out,pred):
    rows=[];arows=[];rng=np.random.default_rng(c['seed']+40)
    pairs=[('ABC_noisy_joint','Constant_TRAIN'),('ABC_GP','ABC_noisy_joint'),('XGBoost_absolute','ABC_noisy_joint'),
        ('XGBoost_absolute','XGBoost_squared'),('EB_Treb_joint','ABC_noisy_joint'),('ABC_noisy_joint','ABC_clean_joint'),
        ('EB_Treb_joint','EB_Treb_qg_shuffled')]
    for dataset in ['venice','mainsail']:
        for n in [4,6]:
            z=pred[(pred.dataset==dataset)&(pred.scan_count==n)&(pred.Treb>n-1)]
            for a,b in pairs:
                x=z[z.model==a][['id','Treb','pred']].merge(z[z.model==b][['id','pred']],on='id',suffixes=('_a','_b')).dropna()
                if not len(x):continue
                gain=abs(x.pred_b-x.Treb)-abs(x.pred_a-x.Treb)
                samples=np.mean(gain.to_numpy()[rng.integers(len(x),size=(c['bootstrap_n'],len(x)))],axis=1)
                rows.append(dict(dataset=dataset,scan_count=n,model=a,comparator=b,n=len(x),
                    MAE_reduction=np.mean(gain),lo=np.quantile(samples,.025),hi=np.quantile(samples,.975)))
                zz=z[z.model==a][['id','Treb','prob_late8']].merge(z[z.model==b][['id','prob_late8']],on='id',suffixes=('_a','_b')).dropna()
                if len(zz) and zz.Treb.gt(8).nunique()==2:
                    vals=[];yy=zz.Treb.to_numpy()>8;pa=zz.prob_late8_a.to_numpy();pb=zz.prob_late8_b.to_numpy()
                    for _ in range(c['bootstrap_n']):
                        ix=rng.integers(len(zz),size=len(zz))
                        if len(np.unique(yy[ix]))==2:vals.append(roc_auc_score(yy[ix],pa[ix])-roc_auc_score(yy[ix],pb[ix]))
                    arows.append(dict(dataset=dataset,scan_count=n,model=a,comparator=b,n=len(zz),
                        AUROC_gain=roc_auc_score(yy,pa)-roc_auc_score(yy,pb),lo=np.quantile(vals,.025),hi=np.quantile(vals,.975)))
    csv(out/'results/PAIRED_MAE_REDUCTION.csv',pd.DataFrame(rows))
    csv(out/'results/PAIRED_AUROC_GAIN.csv',pd.DataFrame(arows))

def learning(c,out,target):
    x=pd.read_csv(out/'predictions/learning_PRIVATE.csv',dtype={'id':str}).merge(target[['dataset','id','Treb']],on=['dataset','id'],validate='many_to_one')
    rows=[]
    for (frac,rep,n),g in x.groupby(['fraction','repeat','n_training']):
        for name,z in [('all_assessable',g),('future_Treb',g[g.Treb>5])]:
            rows.append(dict(fraction=frac,repeat=rep,n_training=n,stratum=name,**binary(z.Treb,z.prob_late8)))
    csv(out/'results/LEARNING_CURVE.csv',pd.DataFrame(rows))

def seq_report(c,out,target):
    x=pd.read_csv(out/'predictions/sequential_PRIVATE.csv',dtype={'id':str}).merge(target[['dataset','id','Treb','event_index']],on=['dataset','id'],validate='many_to_one')
    x['window']=(abs(x.risk_index-x.event_index)<=1).astype(int)
    x['before_confirmation']=x.risk_index<=x.event_index
    csv(out/'evaluation/scored_sequential_PRIVATE.csv',x)
    rows=[];cis=[];rng=np.random.default_rng(c['seed']+41)
    for dataset,z in x.groupby('dataset'):
        for scope,g in [('confirmation_inclusive',z),('before_confirmation',z[z.before_confirmation])]:
            for name,col in [('state','p_window_state'),('time','p_window_time')]:
                met=binary(g.window.to_numpy(),g[col],float(g['threshold_'+name].iloc[0]),cutoff=.5)
                rows.append(dict(dataset=dataset,scope=scope,model=name,patients=g.id.nunique(),**met))
            by={pid:v for pid,v in g.groupby('id')};ids=list(by)
            for i in range(c['bootstrap_n']):
                sample=pd.concat([by[pid] for pid in rng.choice(ids,len(ids),replace=True)],ignore_index=True)
                if sample.window.nunique()<2:continue
                cis.append(dict(dataset=dataset,scope=scope,rep=i,delta_AUROC=roc_auc_score(sample.window,sample.p_window_state)-roc_auc_score(sample.window,sample.p_window_time)))
        future=z[z.Treb>z.risk_cycle]
        rows.append(dict(dataset=dataset,scope='future_Treb_exact',model='sequential_EB',patients=future.id.nunique(),
            **metrics(future.Treb,future.pred),patient_mean_MAE=future.assign(ae=abs(future.pred-future.Treb)).groupby('id').ae.mean().mean()))
    csv(out/'results/SEQUENTIAL_METRICS.csv',pd.DataFrame(rows))
    b=pd.DataFrame(cis)
    if len(b):csv(out/'results/SEQUENTIAL_PAIRED_AUC.csv',b.groupby(['dataset','scope']).delta_AUROC.agg(
        mean='mean',lo=lambda v:v.quantile(.025),hi=lambda v:v.quantile(.975)).reset_index())

def save(fig,out,name):
    p=out/'figures';p.mkdir(exist_ok=True)
    fig.savefig(p/(name+'.pdf'),bbox_inches='tight');fig.savefig(p/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)

def figures(c,out,scores):
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10,'pdf.fonttype':42})
    names=['Constant_TRAIN','ABC_clean_d_only','ABC_clean_joint','ABC_noisy_joint','ABC_GP','XGBoost_absolute','Clinical_PSA','EB_Treb_joint']
    for dataset in ['venice','mainsail']:
        fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for model in names:
            g=scores[(scores.dataset==dataset)&(scores.model==model)&(scores.stratum=='fixed_future6')].sort_values('scan_count')
            if len(g):
                ax[0].plot(g.scan_count,g.MAE,'o-',label=model)
                if 'AUROC' in g:ax[1].plot(g.scan_count,g.AUROC,'o-',label=model)
        ax[0].set(xlabel='Measurements (baseline included)',ylabel='MAE versus observed Treb (cycles)')
        ax[1].set(xlabel='Measurements (baseline included)',ylabel='AUROC: Treb > 8');ax[1].axhline(.5,c='gray',ls=':')
        ax[0].legend(fontsize=6.5);fig.suptitle(dataset.upper()+': fixed subset with Treb after cycle 5')
        save(fig,out,'Forecasting_'+dataset)
    parent=pd.read_csv(out/'cohort/PARENT_231_PRIVATE.csv')
    fig,axs=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for ax,name in zip(axs,['TRAIN','TEST']):
        g=parent[(parent.split==name)&bools(parent.fit_tmin_in_observed_window)]
        ax.scatter(g.Treb,g.tmin_fit_hat,s=16,alpha=.7);lim=max(g.Treb.max(),g.tmin_fit_hat.max())+1
        ax.plot([0,lim],[0,lim],c='gray',ls=':');ax.set(xlabel='Observed Treb (cycles)',ylabel='Full-fit Tfit (cycles)',title=f'{name}: in-window fits, n={len(g)}')
    fig.suptitle('Retrospective agreement; all-parent failures reported separately')
    save(fig,out,'Tfit_vs_Treb')
    train=parent[parent.split=='TRAIN'].sort_values('rmse_log10');long=pd.read_csv(out/'training/long_PRIVATE.csv')
    fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for i,(ax,q) in enumerate(zip(axs.flat,[.1,.4,.7,.9])):
        r=train.iloc[int(q*(len(train)-1))];g=long[long.id==r.id].sort_values('cycle');t=np.linspace(0,g.cycle.max(),400)
        ax.plot(g.cycle,g.psa/g.psa.iloc[0],'o',ms=3,label='Observed PSA')
        ax.plot(t,np.exp(log_curve(t,r.q_hat,r.decline_rate_hat,r.growth_rate_hat)),label='Full fit')
        ax.axvline(r.Treb,c='#2574a5',ls='--',label='Observed Treb');ax.set_yscale('log')
        if 0<=r.tmin_fit_hat<=g.cycle.max():ax.axvline(r.tmin_fit_hat,c='#a56818',ls=':',label='Fitted Tfit')
        ax.set(title=f'TRAIN fit-error percentile {int(q*100)}',xlabel='Cycle',ylabel='PSA / baseline');ax.legend(fontsize=7)
    save(fig,out,'Biexponential_examples_TRAIN')
    info=pd.read_csv(out/'training/parameter_information_summary.csv')
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    for p,g in info.groupby('parameter'):ax.plot(g.scan_count,g.median_width_ratio,'o-',label=p)
    ax.axhline(1,c='gray',ls=':');ax.set(xlabel='Measurements',ylabel='Posterior / prior 95% width',title='TRAIN-only leave-one-patient-out diagnostic');ax.legend()
    save(fig,out,'Early_parameter_information')
