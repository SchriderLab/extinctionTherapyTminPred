from common import *
from empirical import prior,posterior,predictive
from sklearn.linear_model import HuberRegressor,LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

QFUNC=load_functions('run_venice_sequential_nadir_risk.py',['weighted_mad','robust_quadratic_features'],{'HuberRegressor':HuberRegressor})
quadratic=QFUNC['robust_quadratic_features']
TUNE=load_functions('run_venice_bayesian_state_nadir.py',['tune_hyperparams'])['tune_hyperparams']
PHYS=['near_zero_velocity','toward_zero_1','smooth_toward_zero_2']

def histories(long):
    """Rows and histories use past-only entry, stopping, and calendar-time checks."""
    rows=[];hist={};issues=[]
    for pid,g in long.groupby('id',sort=True):
        g=g.sort_values('cycle').reset_index(drop=True)
        for j in range(3,len(g)):
            z=g.iloc[:j+1]
            if z.cycle.iloc[-1]<3:continue
            days=z.day.to_numpy(float)
            if not np.isfinite(days).all() or np.any(np.diff(days)<=0):
                issues.append(dict(id=pid,first_unusable_cycle=g.cycle.iloc[j],reason='missing/nonincreasing actual days'));break
            t=(days-days[0])/21.;psa=z.psa.to_numpy(float);cyc=z.cycle.to_numpy(int)
            feat=quadratic(t,psa)
            if feat is None:continue
            observed,confirmation,idx=endpoint(cyc,psa)
            row=dict(id=pid,risk_index=j,risk_cycle=cyc[-1],risk_calendar_day=days[-1],
                elapsed_cycles=(days[-1]-days[3])/21.,past_event_confirmed=np.isfinite(observed),**feat)
            rows.append(row);hist[(pid,j)]=(t,np.log(psa/psa[0]),cyc)
            if np.isfinite(observed):break
    return pd.DataFrame(rows),hist,pd.DataFrame(issues)

def state_features(rows,hist,obs,jerk):
    x=rows.copy();vv=[]
    for r in rows.itertuples():
        t,y,_=hist[(r.id,r.risk_index)];states,_=robust_filter(t,y,obs,jerk)
        v=states[-1][0][1]
        dv1=(v-states[-2][0][1])/(t[-1]-t[-2])
        dv2=(states[-2][0][1]-states[-3][0][1])/(t[-2]-t[-3])
        vv.append([-abs(v),max(dv1,0.),max(min(dv1,dv2),0.)])
    x[PHYS]=np.array(vv);return x

def select_threshold(y,p):
    grid=np.unique(np.r_[.01,np.quantile(p,np.linspace(0,1,101)),.99])
    return float(grid[np.argmax([balanced_accuracy_score(y,p>=v) for v in grid])])

def run(c,out):
    fits=pd.read_csv(out/'training/fits_PRIVATE.csv',dtype={'id':str});long=pd.read_csv(out/'training/long_PRIVATE.csv',dtype={'id':str})
    tr,trainhist,issues=histories(long)
    params,_=TUNE(fits.id.to_list(),{},tr);obs=params['obs_sigma'];jerk=params['jerk_sigma']
    tr=state_features(tr,trainhist,obs,jerk).merge(fits[['id','event_index']],on='id',validate='many_to_one')
    tr['window']=(abs(tr.risk_index-tr.event_index)<=1).astype(int)
    phys=MonotoneLogistic(l2=1.).fit(tr[PHYS].to_numpy(),tr.window.to_numpy())
    time=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=1,max_iter=5000))]).fit(tr[['elapsed_cycles']],tr.window)
    pthr=select_threshold(tr.window,phys.predict_proba(tr[PHYS].to_numpy())[:,1])
    tthr=select_threshold(tr.window,time.predict_proba(tr[['elapsed_cycles']])[:,1])
    blocks=residual_blocks(fits,long);res=np.concatenate(blocks);sigma=float(np.clip(1.4826*np.median(abs(res-np.median(res))),.1,1.))
    rng=np.random.default_rng(c['seed']+20);pr=prior(fits,c['empirical_draws'],rng)
    noise=noise_draw(blocks,len(pr[0]),c['simulation_horizon']+1,np.random.default_rng(c['seed']+21))
    allrows=[];allissues=[issues.assign(dataset='TRAIN')]
    for name in ['venice','mainsail']:
        qlong=pd.read_csv(out/f'queries/{name}_long_PRIVATE.csv',dtype={'id':str})
        qr,qhist,qi=histories(qlong);allissues.append(qi.assign(dataset=name))
        qr=state_features(qr,qhist,obs,jerk);qr['dataset']=name
        qr['p_window_state']=phys.predict_proba(qr[PHYS].to_numpy())[:,1]
        qr['p_window_time']=time.predict_proba(qr[['elapsed_cycles']])[:,1]
        qr['threshold_state']=pthr;qr['threshold_time']=tthr
        estimates=[]
        for r in qr.itertuples():
            _,y,cycles=qhist[(r.id,r.risk_index)]
            w=posterior(pr,cycles[1:],y[1:],sigma)
            estimates.append(predictive(pr,w,cycles,y,noise,c['simulation_horizon']))
        qr=pd.concat([qr.reset_index(drop=True),pd.DataFrame(estimates)],axis=1)
        allrows.append(qr)
    csv(out/'predictions/sequential_PRIVATE.csv',pd.concat(allrows,ignore_index=True))
    csv(out/'cohort/sequential_calendar_issues_PRIVATE.csv',pd.concat(allissues,ignore_index=True))
    dump(out/'training/SEQUENTIAL_METHOD.json',dict(obs_sigma=obs,jerk_sigma=jerk,
        coefficient=phys.coef_,features=PHYS,threshold_state=pthr,threshold_time=tthr,
        original_methods_retained='Robust Student-t Kalman update, white-jerk state [level,velocity,acceleration], empirical TRAIN noise tuning, nonnegative-coefficient logistic window classifier, time-only comparator.',
        changes='Fixed 231 parent cohort/split; start after baseline plus first 3 measurements; one-rise Treb; no four-decline entry gate; stop after first confirming rise. Actual-day problems stop only affected future updates.',
        window='Within one available measurement of the universal Treb; confirmation-inclusive and preconfirmation results reported separately.',
        exact_timing='Sequential empirical posterior-predictive Treb with nominal future visit schedule; actual past observed cycles retained.',
        baseline_scope='All parent patients have an observed event by design, so this is conditional monitoring, not evaluation among patients who never rebound.'))
