from common import *
from predict import query_sets

def prior(fits,n,rng,shuffle=False):
    z=fits[['q_hat','decline_rate_hat','growth_rate_hat']].to_numpy(float)
    if shuffle:
        nb=min(5,max(2,len(z)//20));bins=pd.qcut(pd.Series(z[:,1]).rank(method='first'),nb,labels=False)
        for b in np.unique(bins):
            ix=np.flatnonzero(np.asarray(bins)==b);z[ix,2]=rng.permutation(z[ix,2])
    idx=rng.integers(len(z),size=n);draw=z[idx]
    q=expit(logit(np.clip(draw[:,0],1e-8,1-1e-8))+rng.normal(0,.08,n))
    d=np.exp(np.log(draw[:,1])+rng.normal(0,.08,n));g=np.exp(np.log(draw[:,2])+rng.normal(0,.08,n))
    tf=(np.log1p(-q)+np.log(d)-np.log(q)-np.log(g))/(d+g)
    return q,d,g,tf

def posterior(pr,t,y,sigma):
    q,d,g,tf=pr
    mu=log_curve(np.asarray(t)[None,:],q[:,None],d[:,None],g[:,None])
    ll=student_t.logpdf((np.asarray(y)[None,:]-mu)/sigma,df=4).sum(axis=1)
    w=np.exp(ll-logsumexp(ll));return w

def predictive(pr,w,t,y,noise,horizon):
    """Condition recorded past exactly; simulate scheduled future observations.
    Treb is re-extracted from each full observed history + simulated future.
    Unknown future visit schedule is nominal one-per-cycle, not the TEST schedule.
    """
    q,d,g,tf=pr;t=np.asarray(t,int);y=np.asarray(y,float)
    future=np.arange(int(t[-1])+1,horizon+1);times=np.r_[t,future]
    if len(times)<3:return dict(pred=np.nan,prob_late8=np.nan,undefined_mass=1.)
    full=log_curve(times[None,:],q[:,None],d[:,None],g[:,None])+noise[:,times]
    full[:,:len(t)]=y
    event=batch_endpoint(full,times);valid=np.isfinite(event)
    mass=float(w[~valid].sum())
    if not valid.any():return dict(pred=np.nan,prob_late8=np.nan,undefined_mass=mass)
    ww=w[valid]/w[valid].sum();e=event[valid]
    return dict(pred=wquant(e,ww),prob_late8=float(ww[e>8].sum()),lo=wquant(e,ww,.025),hi=wquant(e,ww,.975),
        undefined_mass=mass,ESS=float(1/np.sum(w*w)),latent_Tfit=wquant(tf,w))

def run(c,out):
    fits=pd.read_csv(out/'training/fits_PRIVATE.csv',dtype={'id':str});long=pd.read_csv(out/'training/long_PRIVATE.csv',dtype={'id':str})
    blocks=residual_blocks(fits,long);res=np.concatenate(blocks)
    sigma=float(np.clip(1.4826*np.median(abs(res-np.median(res))),.1,1.))
    rows=[];h=c['simulation_horizon']
    for mode,shuffle in [('joint',False),('qg_shuffled',True)]:
        rng=np.random.default_rng(c['seed']+20);pr=prior(fits,c['empirical_draws'],rng,shuffle)
        nd=noise_draw(blocks,len(pr[0]),h+1,np.random.default_rng(c['seed']+21))
        for n in c['scan_counts']:
            for dataset,(ids,Y) in query_sets(out,n-1).items():
                for pid,y in zip(ids,Y):
                    t=np.arange(n);w=posterior(pr,t[1:],y[1:],sigma)
                    ans=predictive(pr,w,t,y,nd,h)
                    rows.append(dict(dataset=dataset,id=pid,scan_count=n,model='EB_Treb_'+mode,threshold=.5,**ans))
                    rows.append(dict(dataset=dataset,id=pid,scan_count=n,model='EB_Tfit_proxy_'+mode,
                        threshold=.5,pred=wquant(pr[3],w),prob_late8=np.sum(w[pr[3]>8]),lo=wquant(pr[3],w,.025),hi=wquant(pr[3],w,.975)))
    csv(out/'predictions/empirical_PRIVATE.csv',pd.DataFrame(rows))
    dump(out/'training/EMPIRICAL_METHOD.json',dict(noise_scale=sigma,draws=c['empirical_draws'],
        prior='Joint bootstrap of TRAIN (logit(q), log(d), log(g)), independent SD=.08 jitter. No fitted-nadir sign filter removes patients from the prior.',
        likelihood='Student-t(df=4), log-relative PSA; TRAIN residual MAD clipped to [.1,1].',
        target='Posterior-predictive Treb uses fixed observed past and simulated future. Latent Tfit is a separately labelled proxy, scored against the same clinical Treb.',
        ablation='Shuffle g within up to five d-rank bins before joint bootstrap, as in the old q-g ablation; preserves d-g dependence only approximately.',
        interval='Conditional predictive 95% interval among draws with evaluable rebound by cycle80; undefined mass explicitly saved.'))
    info(c,out,fits,long,blocks,sigma)
    direct_prefix(c,out)

def direct_prefix(c,out):
    """Unregularized early-fit Tfit proxy; original kernel requires >=5 points."""
    for n in [5,6]:
        for dataset,(ids,Y) in query_sets(out,n-1).items():
            items=[(pid,np.arange(n,dtype=float),np.exp(y)) for pid,y in zip(ids,Y)]
            f=parallel_fits(items,int(os.environ.get('SLURM_CPUS_PER_TASK','2')))
            f['dataset']=dataset;f['scan_count']=n;f['model']='Early_fit_Tfit_proxy'
            f['pred']=f.get('tmin_fit_hat',np.nan);f['prob_late8']=np.nan;f['threshold']=.5
            csv(out/f'predictions/early_fit_{dataset}_{n}_PRIVATE.csv',f)

def info(c,out,fits,long,blocks,sigma):
    """TRAIN-only leave-one-patient-out parameter information diagnostic."""
    rows=[]
    for i,r in enumerate(fits.itertuples()):
        others=fits[fits.id!=r.id];rng=np.random.default_rng(c['seed']+100+i)
        pr=prior(others,min(c['empirical_draws'],4000),rng)
        for n in [4,6]:
            Y=matrix(long,[r.id],n-1)
            if not np.isfinite(Y).all():continue
            # LOO noise estimate also excludes this patient's residual sequence.
            rr=np.concatenate(residual_blocks(others,long));ss=float(np.clip(1.4826*np.median(abs(rr-np.median(rr))),.1,1.))
            w=posterior(pr,np.arange(1,n),Y[0,1:],ss);uniform=np.ones(len(w))/len(w)
            for name,values,truth in [('logit_q',logit(pr[0]),logit(r.q_hat)),('log_d',np.log(pr[1]),np.log(r.decline_rate_hat)),('log_g',np.log(pr[2]),np.log(r.growth_rate_hat))]:
                plo,phi=wquant(values,w,.025),wquant(values,w,.975)
                width=wquant(values,uniform,.975)-wquant(values,uniform,.025)
                rows.append(dict(id=r.id,scan_count=n,parameter=name,prior_width=width,posterior_width=phi-plo,
                    width_ratio=(phi-plo)/width if width>0 else np.nan,full_fit_reference=truth,
                    posterior_median=wquant(values,w),reference_within_interval=plo<=truth<=phi))
    tab=pd.DataFrame(rows);csv(out/'training/parameter_information_PRIVATE.csv',tab)
    csv(out/'training/parameter_information_summary.csv',tab.groupby(['scan_count','parameter']).agg(
        patients=('id','nunique'),median_width_ratio=('width_ratio','median'),
        full_fit_reference_coverage=('reference_within_interval','mean')).reset_index())
    dump(out/'training/IDENTIFIABILITY_INTERPRETATION.json',dict(
        initial_net_slope='q*g - (1-q)*d',
        scope='Posterior concentration under an empirical prior, not proof of structural identifiability or biological ground truth. Full-trajectory fitted parameters are themselves estimates.',
        comparison='Each patient is excluded from its own diagnostic parameter prior and noise estimate; TEST outcomes are unused.'))
