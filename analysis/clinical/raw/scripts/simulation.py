from common import *
from scipy.spatial import cKDTree

def batch(c,out,index):
    from rescue import simulate
    import yaml
    simulate.MODEL=PACKAGE/'vendor/models/quantitative_resistance.slim'
    preset=json.loads((out/'simulation/PRESET.json').read_text())
    manifest=out/'simulation/reference_manifest.csv'
    m=pd.read_csv(manifest);n=int(np.ceil(len(m)/preset['batch_size']))
    if not 0<=index<n:raise ValueError('Batch outside manifest')
    marker=out/'simulation/batches'/f'{index:04d}.csv'
    if marker.exists():
        x=pd.read_csv(marker)
        expected=m.iloc[index*preset['batch_size']:(index+1)*preset['batch_size']].patient_id
        assert_ids(x.patient_id,expected,'Batch manifest mismatch')
        if all((out/'simulation/raw'/f'patient_{int(pid):05d}'/'done.json').exists() for pid in expected):
            print('Existing completed batch',index);return
    cfg=yaml.safe_load((PACKAGE/'vendor/primary.yaml').read_text());cfg['slim']=c['slim']
    simulate.clinical_batch(cfg,manifest,index,out/'simulation',preset)

def assemble(c,out,jobs):
    m=pd.read_csv(out/'simulation/reference_manifest.csv');nb=int(np.ceil(len(m)/c['simulation_batch_size']))
    missing=[i for i in range(nb) if not (out/'simulation/batches'/f'{i:04d}.csv').exists()]
    if missing:raise RuntimeError(f'Missing simulation batches: {missing}')
    records=[];items=[];paths={};Y={}
    for pid in m.patient_id:
        pdir=out/'simulation/raw'/f'patient_{int(pid):05d}';done=json.loads((pdir/'done.json').read_text())
        if done['status']!='finished':records.append(dict(ref_id=pid,status=done['status']));continue
        path=pdir/'trajectory.csv';x=pd.read_csv(path).sort_values('t_treatment')
        if not str(x.status.iloc[-1]) in ['extinct','relapse','horizon']:raise RuntimeError('Incomplete simulation')
        t=x.t_treatment.to_numpy(float);y=x.N.to_numpy(float);ok=(y>0)&np.isfinite(y)&np.isfinite(t)
        t=t[ok];y=y[ok]
        if len(t)<5 or t[0]!=0:records.append(dict(ref_id=pid,status='insufficient_positive_observations'));continue
        items.append((str(pid),t,y));paths[str(pid)]=path
        arr=np.full(c['simulation_horizon']+1,np.nan);arr[t.astype(int)]=np.log(y/y[0]);Y[str(pid)]=arr
    fits=parallel_fits(items,jobs)
    csv(out/'simulation/full_fit_parameters.csv',fits)
    fits=fits[bools(fits.fit_success)].copy().sort_values('id',key=lambda s:s.astype(int))
    if len(fits)<100:raise RuntimeError('Fewer than 100 successful simulation fits; inspect simulation attrition')
    train=pd.read_csv(out/'training/fits_PRIVATE.csv');long=pd.read_csv(out/'training/long_PRIVATE.csv')
    C=np.c_[train.decline_rate_hat,np.log10(train.q_hat),train.growth_rate_hat]
    S=np.c_[fits.decline_rate_hat,np.log10(fits.q_hat),fits.growth_rate_hat]
    center=np.median(C,axis=0);scale=np.maximum(np.quantile(C,.75,axis=0)-np.quantile(C,.25,axis=0),1e-6)
    ds,ix=cKDTree((S-center)/scale).query((C-center)/scale,k=min(25,len(S)))
    weights=np.zeros(len(S))
    for di,ii in zip(ds,ix):
        wi=1/np.maximum(di,1e-6);wi/=wi.sum();np.add.at(weights,ii,wi/len(C))
    weights=.9*weights+.1/len(S);weights/=weights.sum()
    clean=np.array([Y[str(pid)] for pid in fits.id]);blocks=residual_blocks(train,long)
    noisy=clean+noise_draw(blocks,len(clean),clean.shape[1],np.random.default_rng(c['seed']+2))
    np.savez_compressed(out/'simulation/REFERENCE.npz',ref_id=fits.id.astype(str).to_numpy(dtype=str),
        clean=clean,noisy=noisy,Treb_clean=batch_endpoint(clean),Treb_noisy=batch_endpoint(noisy),
        q=fits.q_hat.to_numpy(),d=fits.decline_rate_hat.to_numpy(),g=fits.growth_rate_hat.to_numpy(),weights=weights)
    csv(out/'simulation/attrition.csv',pd.DataFrame(records,columns=['ref_id','status']))
    dump(out/'simulation/REFERENCE_META.json',dict(attempted=len(m),fit_success=len(fits),
        clean_targets=int(np.isfinite(batch_endpoint(clean)).sum()),noisy_targets=int(np.isfinite(batch_endpoint(noisy)).sum()),
        weighted_effective_n=1/np.sum(weights**2),nearest_training_distance_quantiles=np.quantile(ds[:,0],[0,.5,.9,1]),
        guidance='90% equal-patient allocation to 25 nearest simulated full fits, inverse distance; 10% uniform mixture. TRAIN IQR scaling only.',
        observation_sensitivity='Add length-two moving blocks of centered TRAIN full-fit log residuals. Includes model discrepancy, not a validated assay-error model.',
        target='One-rise local-low Treb extracted independently from each clean/noisy recorded simulation; never analytic Tfit labels.',
        sha256={str(p.relative_to(out)):digest(p) for p in [out/'training/fits_PRIVATE.csv',out/'training/long_PRIVATE.csv',out/'simulation/reference_manifest.csv']}))
    # Freeze predictions/design choices before any held-out outcome scoring.
    dump(out/'simulation/REFERENCE_LOCK.json',{'sha256':digest(out/'simulation/REFERENCE.npz')})
