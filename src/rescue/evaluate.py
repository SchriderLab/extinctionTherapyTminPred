"""Treatment-policy evaluation using existing patient-specific counterfactual tables."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .abc import metrics, LEVELS, scales
from .runtime import write_csv,dump_json

def reactive(obs_log: np.ndarray, scans: list[int], multiplier: float, fallback: bool) -> tuple[np.ndarray,np.ndarray]:
    n=np.maximum(10**obs_log[...,scans]-1,0)
    shape=n.shape[:-1];t=np.full(shape,np.nan);hit=np.zeros(shape,bool);minimum=n[...,0].copy()
    for j in range(1,len(scans)):
        minimum=np.minimum(minimum,n[...,j]);new=(~hit)&(n[...,j]>=multiplier*minimum)
        t[new]=scans[j];hit[new]=True
    if fallback:t[~hit]=scans[-1]
    return t,hit

def lookup(surface: np.ndarray, times: np.ndarray) -> np.ndarray:
    times=np.asarray(times,float)
    if times.shape[-1]!=surface.shape[0]:raise ValueError('Patient axis mismatch')
    out=np.full(times.shape,np.nan);valid=np.isfinite(times)
    lag=np.clip(np.where(valid,times,0).astype(int)+1,1,surface.shape[1])
    allvals=surface[np.broadcast_to(np.arange(surface.shape[0]),times.shape),lag-1]
    out[valid]=allvals[valid]
    return out

def surface_and_trajectories(extinction: Path, trajectories: Path, patients: Path, times: list[int]):
    e=pd.read_csv(extinction);p=pd.read_csv(patients).sort_values('patient_id');t=pd.read_csv(trajectories)
    ids=p.patient_id.to_numpy(int);mat=e.pivot(index='patient_id',columns='lag',values='extinction_probability').reindex(ids)
    if list(mat.columns)!=list(range(1,len(mat.columns)+1)) or mat.isna().any().any():raise ValueError('Incomplete contiguous extinction landscape')
    curves=[]
    for pid in ids:
        x=t[t.patient_id==pid].sort_values('t')
        # The saved noise scripts use interpolation, including endpoint holding.
        curves.append(np.interp(times,x.t,x.N))
    return ids,mat.to_numpy(float),p.tmin.to_numpy(float),np.log10(np.asarray(curves)+1)

def evaluate(cfg: dict, root: Path) -> None:
    a=pd.read_csv(root/'abc/predictions.csv');e=pd.read_csv(root/'cohort/extinction.csv');p=pd.read_csv(root/'cohort/patients.csv')
    look=e.set_index(['patient_id','lag']).extinction_probability;best=e.groupby('patient_id').extinction_probability.max()
    records=[]
    for r in a.itertuples():
        abc=float(look.loc[r.patient_id,r.pred_switch_lag_round]);true=float(look.loc[r.patient_id,r.true_switch_lag_at_tmin]);oracle=float(best.loc[r.patient_id])
        records.append({'patient_id':r.patient_id,'method':r.method,'k':r.k,'slope_weight':r.slope_weight,
            'true_tmin':r.true_tmin,'pred_tmin':r.pred_tmin_weighted,'abs_tmin_error':r.abs_error_weighted,
            'true_tmin_lag':r.true_switch_lag_at_tmin,'abc_lag':r.pred_switch_lag_round,'oracle_extinction':oracle,
            'tmin_extinction':true,'abc_extinction':abc,'total_oracle_regret':oracle-abc,
            'oracle_minus_true_tmin':oracle-true,'true_tmin_minus_abc':true-abc})
    d=pd.DataFrame(records);write_csv(d,root/'evaluation/policy_by_patient.csv')
    ref=pd.read_csv(root/'reference/features.csv');end=int(np.ceil(ref.t_min_true.quantile(cfg['policies']['monitoring_quantile'])))
    if end>=len(cfg['primary']['lags']):raise ValueError('Monitoring endpoint exceeds available lag grid')
    ids,mat,tm,logs=surface_and_trajectories(root/'cohort/extinction.csv',root/'cohort/trajectories.csv',root/'cohort/patients.csv',list(range(end+1)))
    n=10**logs-1;rr=[]
    for count in range(cfg['policies']['min_scan_count'],end+2):
        schedule=np.rint(np.linspace(0,end,count)).astype(int)
        if len(np.unique(schedule))!=count:raise ValueError('Duplicate scan schedule')
        for i,pid in enumerate(ids):
            increases=np.flatnonzero(np.diff(n[i,schedule])>0)
            st=int(schedule[increases[0]+1]) if len(increases) else end
            rr.append({'patient_id':int(pid),'n_scans':count,'schedule':','.join(map(str,schedule)),
                'switch_t':st,'triggered':bool(len(increases)),'delay':float(st-tm[i]),'extinction':float(mat[i,st])})
    rd=pd.DataFrame(rr);write_csv(rd,root/'evaluation/reactive_by_patient.csv')
    summary=rd.groupby('n_scans',as_index=False).agg(mean_extinction=('extinction','mean'),mean_delay=('delay','mean'))
    write_csv(summary,root/'evaluation/reactive_summary.csv')
    ap=a[a.method=='logN_plus_slope'];dp=d[d.method=='logN_plus_slope'];z=metrics(ap.true_tmin,ap.pred_tmin_weighted)
    z.update(abc_extinction=float(dp.abc_extinction.mean()),true_nadir_extinction=float(dp.tmin_extinction.mean()),
             empirical_best_lag=float(dp.oracle_extinction.mean()),dense_reactive=float(summary.iloc[-1].mean_extinction),monitoring_end=end)
    dump_json(root/'evaluation/summary.json',z)

def noise_predict(ref, y, query, k, weight, batch, eps):
    """Matrix-distance form used by the original noise driver, including its clipping."""
    sd=scales(ref);rl=ref/sd[0];rs=np.diff(ref,axis=1)/sd[1];rl2=np.sum(rl*rl,1);rs2=np.sum(rs*rs,1)
    out=np.empty(len(query));k=min(k,len(y))
    for start in range(0,len(query),batch):
        q=query[start:start+batch];ql=q/sd[0];qs=np.diff(q,axis=1)/sd[1]
        dl=np.sqrt(np.maximum(np.sum(ql*ql,1)[:,None]+rl2[None,:]-2*ql@rl.T,0)/4)
        ds=np.sqrt(np.maximum(np.sum(qs*qs,1)[:,None]+rs2[None,:]-2*qs@rs.T,0)/3)
        dist=dl+weight*ds;idx=np.argpartition(dist,k-1,axis=1)[:,:k];d=np.take_along_axis(dist,idx,1);w=1/(d+eps)
        out[start:start+batch]=np.sum(w*y[idx],1)/w.sum(1)
    return out

def noise(cfg: dict, root: Path) -> None:
    c=cfg['noise'];pol=cfg['policies'];times=c['observation_times']
    if times!=list(range(len(times))):raise ValueError('Noise observation times must be consecutive from zero')
    ids,mat,tm,truth=surface_and_trajectories(root/'cohort/extinction.csv',root/'cohort/trajectories.csv',root/'cohort/patients.csv',times)
    ref=pd.read_csv(root/'reference/features.csv');x=ref[LEVELS].to_numpy(float);y=ref.t_min_true.to_numpy(float)
    rng=np.random.default_rng(c['seed']);rows=[];means=[];summ=[];dcount=int(c['draws']);n=len(ids)
    def add(sigma,policy,m,sw,hit,score,pred=None):
        delay=sw-tm[None,:];mae=np.abs(pred-tm[None,:]) if pred is not None else np.full(sw.shape,np.nan)
        ms=float(np.nanmean(delay)) if np.isfinite(delay).any() else None
        summ.append({'sigma':sigma,'policy':policy,'n_biomarkers':m,'trigger_fraction':float(hit.mean()),
            'mean_switch_delay':ms,'mean_extinction':float(score.mean()),
            'mean_extinction_triggered':float(score[hit].mean()) if hit.any() else np.nan,
            'tmin_MAE':float(mae.mean()) if pred is not None else np.nan})
        for d in range(dcount):
            means.append({'sigma':sigma,'policy':policy,'n_biomarkers':m,'draw':d,'mean_extinction':float(score[d].mean()),
                          'tmin_MAE':float(mae[d].mean()) if pred is not None else np.nan})
        rows.append(pd.DataFrame({'sigma':sigma,'policy':policy,'n_biomarkers':m,'draw':np.repeat(np.arange(dcount),n),
            'patient_id':np.tile(ids,dcount),'switch_t':sw.ravel(),'triggered':hit.ravel(),'extinction':score.ravel(),
            'tmin_error':mae.ravel()}))
    for sigma in c['sigma']:
        print(f'Noise sigma={sigma}',flush=True)
        draws=truth[None,:,None,:]+rng.normal(0,sigma,size=(dcount,n,max(c['biomarkers']),len(times)))
        for policy,mult,fallback in [('Reactive_1.25x',pol['persistent_multiplier'],True),('Reactive_NED_10x',pol['ned_multiplier'],False)]:
            sw,hit=reactive(draws[:,:,0,:],pol['threshold_scans'],mult,fallback)
            score=np.nan_to_num(lookup(mat,sw),nan=0.);add(sigma,policy,1,sw,hit,score)
        for m in c['biomarkers']:
            q=draws[:,:,:m,:].mean(2)[:,:,:4].reshape(dcount*n,4)
            pred=noise_predict(x,y,q,int(c['neighbors']),float(c['slope_weight']),int(c['query_batch_size']),cfg['abc']['epsilon']).reshape(dcount,n)
            sw=np.clip(np.rint(pred),0,mat.shape[1]-1);score=lookup(mat,sw)
            add(sigma,'ABC',m,sw,np.ones(sw.shape,bool),score,pred)
    write_csv(pd.DataFrame(summ),root/'noise/summary.csv');write_csv(pd.DataFrame(means),root/'noise/cohort_draws.csv')
    write_csv(pd.concat(rows,ignore_index=True),root/'noise/patient_draws.csv')
