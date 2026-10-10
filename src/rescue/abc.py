"""Reference-only ABC tuning and prediction; no treatment outcomes are read here."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .runtime import dump_json, write_csv

LEVELS=['logN0','logN1','logN2','logN3']

def scales(x: np.ndarray) -> tuple[float,float]:
    a=float(np.std(x.ravel())); b=float(np.std(np.diff(x,axis=1).ravel()))
    return (a if np.isfinite(a) and a>0 else 1.,b if np.isfinite(b) and b>0 else 1.)

def distances(ref: np.ndarray, query: np.ndarray, sd: tuple[float,float]) -> tuple[np.ndarray,np.ndarray]:
    dl=np.sqrt(np.mean(((query[:,None,:]-ref[None,:,:])/sd[0])**2,axis=2))
    ds=np.sqrt(np.mean(((np.diff(query,axis=1)[:,None,:]-np.diff(ref,axis=1)[None,:,:])/sd[1])**2,axis=2))
    return dl,ds

def estimate(d: np.ndarray, y: np.ndarray, k: int, eps: float) -> np.ndarray:
    k=min(int(k),len(y))
    if k<1:raise ValueError('ABC requires a positive neighbor count')
    idx=np.argpartition(d,k-1,axis=1)[:,:k]
    vals=np.take_along_axis(d,idx,axis=1)
    order=np.argsort(vals,axis=1)
    idx=np.take_along_axis(idx,order,axis=1); vals=np.take_along_axis(vals,order,axis=1)
    w=1/(vals+eps)
    return (w*y[idx]).sum(1)/w.sum(1)

def predict(ref: np.ndarray,y: np.ndarray,query: np.ndarray,k: int,weight: float,eps: float=1e-8,batch: int=48) -> np.ndarray:
    result=np.empty(len(query));sd=scales(ref)
    for start in range(0,len(query),batch):
        dl,ds=distances(ref,query[start:start+batch],sd)
        result[start:start+batch]=estimate(dl+weight*ds,y,k,eps)
    return result

def metrics(y,p) -> dict:
    e=np.asarray(p)-np.asarray(y);ae=np.abs(e)
    return {'n':len(e),'MAE_weighted':float(ae.mean()),'RMSE_weighted':float(np.sqrt(np.mean(e*e))),
            'bias_weighted':float(e.mean()),'within_1':float((ae<=1).mean()),'within_2':float((ae<=2).mean())}

def tune(cfg: dict, reference: Path, out: Path) -> None:
    a=cfg['abc'];ref=pd.read_csv(reference);x=ref[LEVELS].to_numpy(float);y=ref.t_min_true.to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite reference feature')
    folds=np.empty(len(ref),dtype=int);perm=np.random.default_rng(a['seed']).permutation(len(ref));folds[perm]=np.arange(len(ref))%a['folds']
    candidates=[('baseline_logN',0.,int(k)) for k in a['neighbors']]+[('logN_plus_slope',float(w),int(k)) for w in a['slope_weights'] for k in a['neighbors']]
    preds={c:np.full(len(y),np.nan) for c in candidates}
    for fold in range(a['folds']):
        tr=folds!=fold;va=np.flatnonzero(~tr);sd=scales(x[tr])
        print(f'Reference CV fold {fold+1}/{a["folds"]}',flush=True)
        for start in range(0,len(va),a['batch_size']):
            ids=va[start:start+a['batch_size']];dl,ds=distances(x[tr],x[ids],sd)
            for weight in [0.]+list(a['slope_weights']):
                dist=dl+weight*ds;maxk=min(max(a['neighbors']),int(tr.sum()))
                idx=np.argpartition(dist,maxk-1,axis=1)[:,:maxk]
                vals=np.take_along_axis(dist,idx,axis=1);order=np.argsort(vals,axis=1)
                idx=np.take_along_axis(idx,order,axis=1);vals=np.take_along_axis(vals,order,axis=1)
                method='baseline_logN' if weight==0 else 'logN_plus_slope'
                for k in a['neighbors']:
                    use=min(k,maxk);w=1/(vals[:,:use]+a['epsilon'])
                    preds[(method,float(weight),int(k))][ids]=(w*y[tr][idx[:,:use]]).sum(1)/w.sum(1)
    rows=[];best={}
    for c,p in preds.items():
        if not np.isfinite(p).all():raise RuntimeError('Missing held-out reference predictions')
        rows.append({'method':c[0],'slope_weight':c[1],'k':c[2],**metrics(y,p)})
    summary=pd.DataFrame(rows).sort_values(['MAE_weighted','RMSE_weighted','k'],kind='stable')
    for method in ['baseline_logN','logN_plus_slope']:
        r=summary[summary.method==method].iloc[0];best[method]={'k':int(r.k),'slope_weight':float(r.slope_weight)}
        c=(method,float(r.slope_weight),int(r.k))
        write_csv(pd.DataFrame({'ref_id':ref.ref_id,'fold':folds,'true_tmin':y,'pred_tmin':preds[c]}),out/f'{method}_reference_oof.csv')
    write_csv(summary,out/'cv_summary.csv');dump_json(out/'selected.json',best)

def patient_prediction(cfg: dict, reference: Path, trajectory_file: Path, selected: Path, out: Path) -> None:
    import json
    a=cfg['abc'];refs=pd.read_csv(reference);trajs=pd.read_csv(trajectory_file)
    ids=[];x=[];ys=[]
    for pid,g in trajs.groupby('patient_id',sort=True):
        g=g.sort_values('t'); grid=g.set_index('t')
        if not set(a['scan_times'])<=set(grid.index):raise ValueError(f'Missing patient observation: {pid}')
        ids.append(int(pid));x.append(np.log10(grid.loc[a['scan_times'],'N'].to_numpy(float)+1));ys.append(float(g.loc[g.N.idxmin(),'t']))
    x=np.asarray(x);y=np.asarray(ys);best=json.loads(selected.read_text());frames=[]
    for method,b in best.items():
        pred=predict(refs[LEVELS].to_numpy(float),refs.t_min_true.to_numpy(float),x,b['k'],b['slope_weight'],a['epsilon'],a['batch_size'])
        round_t=np.rint(pred).astype(int);lags=np.asarray(cfg['primary']['lags'],int)
        # Mapping remains round(t)+1; a non-contiguous lag grid is rejected, not nearest-imputed.
        pl=np.clip(round_t+1,min(lags),max(lags));tl=np.rint(y).astype(int)+1
        if not set(pl)<=set(lags) or not set(tl)<=set(lags):raise ValueError('A selected/nadir lag is absent')
        z=pd.DataFrame({'patient_id':ids,'method':method,'k':b['k'],'slope_weight':b['slope_weight'],
            'true_tmin':y,'true_switch_lag_at_tmin':tl,'pred_tmin_weighted':pred,'pred_tmin_round':round_t,
            'pred_switch_lag_round':pl,'error_weighted':pred-y,'abs_error_weighted':np.abs(pred-y)})
        for j in range(4):z[f'logN{j}']=x[:,j]
        frames.append(z)
    write_csv(pd.concat(frames,ignore_index=True),out)
