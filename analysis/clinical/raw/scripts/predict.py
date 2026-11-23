from common import *
from sklearn.model_selection import KFold,StratifiedKFold,GridSearchCV,StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,Matern,WhiteKernel
from scipy.spatial.distance import cdist
import joblib

def abc_space(Y,slope):return np.c_[Y[:,1:],slope*np.diff(Y,axis=1)]

def abc_predict(R,y,w,Q,k,slope,rejection=False,cutoff=8):
    # Feature scales derived exclusively from the eligible simulation bank.
    sr=np.maximum(np.std(R[:,1:],axis=0,ddof=1),1e-8)
    sd=np.maximum(np.std(np.diff(R,axis=1),axis=0,ddof=1),1e-8)
    def feat(x):return np.c_[x[:,1:]/sr,slope*np.diff(x,axis=1)/sd]
    dist=cdist(feat(Q),feat(R));k=min(int(k),len(R))
    ix=np.argsort(dist,axis=1,kind='stable')[:,:k];d=np.take_along_axis(dist,ix,axis=1)
    ww=np.ones_like(d) if rejection else w[ix]/(d+1e-8)
    ww/=ww.sum(axis=1,keepdims=True)
    yy=y[ix];pred=np.sum(yy*ww,axis=1);prob=np.sum((yy>cutoff)*ww,axis=1)
    med=np.array([wquant(yi,wi) for yi,wi in zip(yy,ww)])
    return pred,med,prob

def threshold(y,p):
    labels=(np.asarray(y)>8).astype(int)
    if len(np.unique(labels))<2:return .5
    grid=np.unique(np.r_[.01,np.quantile(p,np.linspace(0,1,101)),.99])
    scores=[balanced_accuracy_score(labels,np.asarray(p)>=v) for v in grid]
    return float(grid[np.argmax(scores)])

def tune_abc(R,y,w,seed):
    splits=list(KFold(5,shuffle=True,random_state=seed).split(R));rows=[];best={}
    ks=[k for k in [10,25,50,100] if k<len(R)*.8]
    for slope in [0.,.5,1.,2.]:
        cache={(r,k):(np.full(len(R),np.nan),np.full(len(R),np.nan)) for r in [False,True] for k in ks}
        for tr,va in splits:
            sr=np.maximum(np.std(R[tr,1:],axis=0,ddof=1),1e-8)
            sd=np.maximum(np.std(np.diff(R[tr],axis=1),axis=0,ddof=1),1e-8)
            F=lambda z:np.c_[z[:,1:]/sr,slope*np.diff(z,axis=1)/sd]
            dist=cdist(F(R[va]),F(R[tr]));ix=np.argsort(dist,axis=1,kind='stable')[:,:max(ks)]
            dd=np.take_along_axis(dist,ix,axis=1);yy=y[tr][ix];wp=w[tr][ix]
            for rejection in [False,True]:
                for k in ks:
                    ww=np.ones_like(dd[:,:k]) if rejection else wp[:,:k]/(dd[:,:k]+1e-8)
                    ww/=ww.sum(axis=1,keepdims=True)
                    cache[rejection,k][0][va]=np.sum(ww*yy[:,:k],axis=1)
                    cache[rejection,k][1][va]=np.sum(ww*(yy[:,:k]>8),axis=1)
        for (rejection,k),(pred,prob) in cache.items():
            rows.append(dict(rejection=rejection,k=k,slope=slope,MAE=np.mean(abs(pred-y)),
                AUROC=roc_auc_score(y>8,prob) if len(np.unique(y>8))>1 else np.nan,threshold=threshold(y,prob)))
    # Rejection ABC has its own reference-selected settings and threshold.
    for rejection in [False,True]:
        records=[r for r in rows if r['rejection']==rejection]
        # Regression settings: MAE. Binary settings: AUROC. No clinical TEST tuning.
        reg=min(records,key=lambda z:(z['MAE'],z['k'],z['slope']))
        binary=max(records,key=lambda z:(z['AUROC'] if np.isfinite(z['AUROC']) else -np.inf,-z['k']))
        best['rejection' if rejection else 'weighted']=dict(regression=reg,binary=binary)
    return best,pd.DataFrame(rows)

def clinical_fit(X,y,classification=False,seed=1):
    if classification:
        yy=(np.asarray(y)>8).astype(int);minor=int(np.bincount(yy,minlength=2).min())
        if minor<2:return None
        model=LogisticRegression(max_iter=3000,solver='liblinear',class_weight='balanced')
        grid={'model__C':np.logspace(-3,3,13)};cv=StratifiedKFold(min(4,minor),shuffle=True,random_state=seed);score='roc_auc'
    else:
        yy=np.asarray(y);model=Ridge();grid={'model__alpha':np.logspace(-3,3,13)}
        cv=KFold(min(5,len(y)),shuffle=True,random_state=seed);score='neg_mean_absolute_error'
    pipe=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler()),('model',model)])
    gs=GridSearchCV(pipe,grid,cv=cv,scoring=score,n_jobs=1).fit(X,yy)
    return gs

def query_sets(out,last):
    ans={}
    for name in ['venice','mainsail']:
        p=out/f'queries/{name}_long_PRIVATE.csv'
        if not p.exists():raise FileNotFoundError(p)
        long=pd.read_csv(p,dtype={'id':str});ids=sorted(long.id.unique());Y=matrix(long,ids,last)
        ok=np.isfinite(Y).all(axis=1);ans[name]=(np.array(ids)[ok],Y[ok])
    return ans

def fixed(c,out):
    from xgboost import XGBRegressor
    bank=np.load(out/'simulation/REFERENCE.npz',allow_pickle=False)
    lock=json.loads((out/'simulation/REFERENCE_LOCK.json').read_text())
    if lock['sha256']!=digest(out/'simulation/REFERENCE.npz'):raise RuntimeError('Reference bank changed after locking')
    train=pd.read_csv(out/'training/fits_PRIVATE.csv',dtype={'id':str});long=pd.read_csv(out/'training/long_PRIVATE.csv',dtype={'id':str})
    predictions=[];modelmeta=[]
    def add(dataset,ids,n,model,p,prob=None,thr=.5,extra=None):
        pp=pd.DataFrame(dict(dataset=dataset,id=ids,scan_count=n,model=model,pred=np.asarray(p),
            prob_late8=np.nan if prob is None else prob,threshold=thr))
        if extra:
            for k,v in extra.items():pp[k]=v
        predictions.append(pp)
    for n in c['scan_counts']:
        Q=query_sets(out,n-1);Ytr=matrix(long,train.id,n-1);complete=np.isfinite(Ytr).all(axis=1)
        tr=train[complete].copy();Ytr=Ytr[complete];yt=tr.Treb.to_numpy();Xtr=features(Ytr)
        primary_anchor={}
        for variant,field,joint in [('clean_d_only','clean',False),('clean_joint','clean',True),('noisy_joint','noisy',True)]:
            R=bank[field][:,:n];yr=bank['Treb_'+field];ok=np.isfinite(R).all(axis=1)&np.isfinite(yr)
            R=R[ok];yr=yr[ok];w=bank['weights'][ok] if joint else np.ones(sum(ok));w=w/w.sum()
            if len(R)<100:raise RuntimeError(f'{variant}: fewer than 100 eligible references')
            best,cv=tune_abc(R,yr,w,c['seed']+n);csv(out/f'training/abc_cv_{variant}_{n}.csv',cv)
            modelmeta.append(dict(scan_count=n,variant=variant,reference_n=len(R),settings=best))
            settings=best['weighted']['regression'];bs=best['weighted']['binary']
            rr=best['rejection']['regression'];rs=best['rejection']['binary']
            if variant=='noisy_joint':
                primary_anchor['TRAIN']=abc_predict(R,yr,w,Ytr,settings['k'],settings['slope'])[0]
            for name,(ids,Yq) in Q.items():
                p,med,_=abc_predict(R,yr,w,Yq,settings['k'],settings['slope'])
                bp=abc_predict(R,yr,w,Yq,bs['k'],bs['slope'])[2]
                add(name,ids,n,'ABC_'+variant,p,bp,bs['threshold'])
                add(name,ids,n,'ABC_median_'+variant,med,bp,bs['threshold'])
                add(name,ids,n,'Constant_reference_'+variant,np.repeat(wquant(yr,w),len(ids)),np.repeat(np.sum(w[yr>8]),len(ids)))
                if variant=='noisy_joint':
                    primary_anchor[name]=p
                    rp=abc_predict(R,yr,w,Yq,rr['k'],rr['slope'],True)[0]
                    rb=abc_predict(R,yr,w,Yq,rs['k'],rs['slope'],True)[2]
                    add(name,ids,n,'Rejection_ABC',rp,rb,rs['threshold'])
            if variant=='noisy_joint':
                for objective,tag in [('reg:absoluteerror','absolute'),('reg:squarederror','squared')]:
                    mdl=XGBRegressor(objective=objective,n_estimators=500,max_depth=4,learning_rate=.03,
                        min_child_weight=5,subsample=.85,colsample_bytree=.9,reg_lambda=1.,reg_alpha=0.,
                        tree_method='hist',random_state=20260919,n_jobs=1,eval_metric='mae')
                    mdl.fit(features(R,True),yr,sample_weight=w*len(w))
                    for name,(ids,Yq) in Q.items():add(name,ids,n,'XGBoost_'+tag,mdl.predict(features(Yq,True)))
                    mp=out/f'models/xgb_{tag}_{n}.json';mp.parent.mkdir(parents=True,exist_ok=True);mdl.save_model(mp)
        # Same discrepancy-GP family as the original paper, now trained only on TRAIN.
        scaler=StandardScaler().fit(Xtr)
        gp=GaussianProcessRegressor(kernel=ConstantKernel(1.,(1e-3,1e3))*Matern(length_scale=1.,nu=1.5)+
            WhiteKernel(noise_level=1.,noise_level_bounds=(1e-5,1e3)),normalize_y=True,
            n_restarts_optimizer=2,random_state=c['seed'])
        gp.fit(scaler.transform(Xtr),yt-primary_anchor['TRAIN'])
        ridge=clinical_fit(Xtr,yt,seed=c['seed']);clf=clinical_fit(Xtr,yt,True,c['seed'])
        for name,(ids,Yq) in Q.items():
            add(name,ids,n,'ABC_GP',primary_anchor[name]+gp.predict(scaler.transform(features(Yq))))
            p=ridge.predict(features(Yq));pr=None if clf is None else clf.predict_proba(features(Yq))[:,1]
            add(name,ids,n,'Clinical_PSA',p,pr)
            add(name,ids,n,'Constant_TRAIN',np.repeat(np.median(yt),len(ids)),np.repeat(np.mean(yt>8),len(ids)))
        joblib.dump(dict(gp=gp,scaler=scaler,ridge=ridge,classifier=clf),out/f'models/clinical_{n}.joblib')
        print('Fixed-scan models complete:',n,flush=True)
    csv(out/'predictions/fixed_PRIVATE.csv',pd.concat(predictions,ignore_index=True))
    dump(out/'training/MODEL_SETTINGS.json',modelmeta)
    dump(out/'predictions/FIXED_LOCK.json',dict(prediction_sha256=digest(out/'predictions/fixed_PRIVATE.csv'),
        training_sha256=digest(out/'training/fits_PRIVATE.csv'),reference_sha256=lock['sha256'],
        target='Treb everywhere. XGBoost trained on simulated observed Treb, not continuous fitted Tfit.',
        primary_method='ABC_noisy_joint; clean alternatives and median are prespecified sensitivities, not test-selected winners.'))

def clinical_checks(c,out):
    train=pd.read_csv(out/'training/fits_PRIVATE.csv',dtype={'id':str});long=pd.read_csv(out/'training/long_PRIVATE.csv',dtype={'id':str})
    cov=pd.read_csv(out/'training/baseline_PRIVATE.csv',dtype={'id':str}).set_index('id')
    qcov=pd.read_csv(out/'queries/venice_baseline_PRIVATE.csv',dtype={'id':str}).set_index('id')
    Y=matrix(long,train.id,5);ok=np.isfinite(Y).all(axis=1);tr=train[ok];Y=Y[ok]
    ids,Q=query_sets(out,5)['venice'];X=features(Y);Z=features(Q);y=tr.Treb.to_numpy()
    cols=cov.columns[cov.reindex(tr.id).notna().sum()>=10].tolist()
    B=cov.reindex(tr.id)[cols].to_numpy();Bq=qcov.reindex(ids)[cols].to_numpy();pred=[]
    for name,xx,zz in [('PSA',X,Z),('baseline',B,Bq),('PSA_baseline',np.c_[X,B],np.c_[Z,Bq])]:
        model=clinical_fit(xx,y,True,c['seed'])
        if model is None:raise RuntimeError('Insufficient TRAIN classes for covariate comparison')
        pred.append(pd.DataFrame(dict(dataset='venice',id=ids,scan_count=6,model='Covariates_'+name,pred=np.nan,
            prob_late8=model.predict_proba(zz)[:,1],threshold=.5)))
    csv(out/'predictions/covariates_PRIVATE.csv',pd.concat(pred))
    curve=[]
    for fraction in [.25,.40,.55,.70,.85,1.]:
        for rep in range(1 if fraction==1 else c['learning_repeats']):
            ix=np.arange(len(y)) if fraction==1 else next(StratifiedShuffleSplit(1,train_size=fraction,
                random_state=c['seed']+rep+int(fraction*1000)).split(X,y>8))[0]
            model=clinical_fit(X[ix],y[ix],True,c['seed']+rep)
            if model is None:continue
            curve.append(pd.DataFrame(dict(dataset='venice',id=ids,scan_count=6,model='Clinical_PSA_learning',
                fraction=fraction,repeat=rep,n_training=len(ix),pred=np.nan,prob_late8=model.predict_proba(Z)[:,1],threshold=.5)))
    csv(out/'predictions/learning_PRIVATE.csv',pd.concat(curve))
    dump(out/'training/CLINICAL_CHECKS.json',dict(baseline_columns_used=cols,training_only_column_selection_min_nonmissing=10,
        repeats=c['learning_repeats'],note='PSA/baseline comparison and training-size experiments repeat Section 3.8 with Treb>8 and the fixed patient split.'))

def cross_trial(c,out):
    """MAINSAIL -> held-out VENICE and internal MAINSAIL, separate from primary transfer."""
    m=pd.read_csv(out/'evaluation/mainsail_reference_PRIVATE.csv',dtype={'id':str})
    ml=pd.read_csv(out/'queries/mainsail_long_PRIVATE.csv',dtype={'id':str});pred=[]
    for n in [4,6]:
        Y=matrix(ml,m.id,n-1);ok=np.isfinite(Y).all(axis=1);tab=m[ok];X=features(Y[ok]);y=tab.Treb.to_numpy()
        ids,Q=query_sets(out,n-1)['venice'];Z=features(Q)
        r=clinical_fit(X,y,seed=c['seed']);cl=clinical_fit(X,y,True,c['seed'])
        for name,p,pr in [('MAINSAIL_to_VENICE',r.predict(Z),cl.predict_proba(Z)[:,1] if cl else np.repeat(np.mean(y>8),len(Z))),
            ('MAINSAIL_constant',np.repeat(np.median(y),len(Z)),np.repeat(np.mean(y>8),len(Z)))]:
            pred.append(pd.DataFrame(dict(dataset='venice',id=ids,scan_count=n,model=name,pred=p,prob_late8=pr,threshold=.5)))
        for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=c['seed']).split(X,y>8)):
            r=clinical_fit(X[tr],y[tr],seed=c['seed']+fold);cl=clinical_fit(X[tr],y[tr],True,c['seed']+fold)
            pred.append(pd.DataFrame(dict(dataset='mainsail',id=tab.id.to_numpy()[va],scan_count=n,
                model='MAINSAIL_internal_CV',pred=r.predict(X[va]),prob_late8=cl.predict_proba(X[va])[:,1],threshold=.5)))
    csv(out/'predictions/cross_trial_PRIVATE.csv',pd.concat(pred))
    dump(out/'predictions/CROSS_TRIAL_DESIGN.json',dict(cutoff=8,target='Treb>8 in BOTH trials',
        note='Primary VENICE->MAINSAIL forecasts are already in fixed_PRIVATE.csv, from VENICE TRAIN only. Reverse/internal models do not modify those forecasts.'))

def reference_size(c,out):
    bank=np.load(out/'simulation/REFERENCE.npz',allow_pickle=False);n=6
    R=bank['noisy'][:,:n];yr=bank['Treb_noisy'];valid=np.isfinite(R).all(axis=1)&np.isfinite(yr)
    R=R[valid];yr=yr[valid];w=bank['weights'][valid];w/=w.sum()
    settings=[v for v in json.loads((out/'training/MODEL_SETTINGS.json').read_text()) if v['variant']=='noisy_joint' and v['scan_count']==6][0]['settings']['weighted']
    reg=settings['regression'];bi=settings['binary'];rng=np.random.default_rng(c['seed']+72)
    sizes=sorted(set([s for s in [250,500,1000] if s<len(R)]+[len(R)]));rows=[]
    for size in sizes:
        for rep in range(1 if size==len(R) else 20):
            ix=np.arange(len(R)) if size==len(R) else rng.choice(len(R),size,replace=False)
            if len(ix)<max(reg['k'],bi['k']):continue
            for dataset,(ids,Q) in query_sets(out,5).items():
                p,_,_=abc_predict(R[ix],yr[ix],w[ix],Q,reg['k'],reg['slope'])
                pr=abc_predict(R[ix],yr[ix],w[ix],Q,bi['k'],bi['slope'])[2]
                rows.append(pd.DataFrame(dict(dataset=dataset,id=ids,scan_count=6,reference_n=size,repeat=rep,
                    pred=p,prob_late8=pr,threshold=bi['threshold'])))
    csv(out/'predictions/reference_size_PRIVATE.csv',pd.concat(rows,ignore_index=True))
