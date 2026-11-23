"""Shared endpoint, immutable I/O, and original fit/state kernels."""
from pathlib import Path
import ast, hashlib, json, os, sys, platform
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from scipy.special import expit, logit, logsumexp
from scipy.stats import spearmanr, t as student_t
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, brier_score_loss

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE/'vendor'))

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def plain(x):
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [plain(v) for v in x]
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.bool_,)):return bool(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,Path):return str(x)
    return x

def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(plain(x),indent=2,allow_nan=False)+'\n');temp.replace(p)

def csv(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    x.to_csv(p,index=False)

def bools(s):return s.astype(str).str.lower().isin(['true','1','yes'])

def config(path):
    c=json.loads(Path(path).read_text());out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
    for key,rel in [('adsl','data/venice_csv/adsl.csv'),('adlb','data/venice_csv/adlb.csv'),('mainsail_lab','data/mainsail_pc002/lab.sas7bdat')]:
        c[key]=c.get(key) or str(Path(c['historical_root'])/rel)
    return c,out

def require_compute():
    import socket
    if 'login' in socket.gethostname().lower() and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit this calculation with sbatch/srun; do not run it on a login node.')

def load_functions(filename,names,extra=None):
    """Load only named pure definitions, never a legacy script's analysis body."""
    p=PACKAGE/'vendor/clinical'/filename;tree=ast.parse(p.read_text())
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in selected}!=set(names):raise RuntimeError('Vendor kernel contract changed')
    ns=dict(np=np,pd=pd,least_squares=least_squares,minimize=minimize,expit=expit,
            student_t=student_t,math=__import__('math'),EPS=1e-9,NU=4.0,
            BOUNDS_LO=np.array([1e-8,1e-5,1e-5]),BOUNDS_HI=np.array([.95,3.,3.]))
    if extra:ns.update(extra)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(p),'exec'),ns)
    return ns

FIT=load_functions('build_venice_exponential_cloud.py',
    ['model','analytic_nadir','nadir_depth','residuals_log10','fit_curve'])
fit_curve=FIT['fit_curve'];analytic_nadir=FIT['analytic_nadir']
STATE=load_functions('run_venice_bayesian_state_nadir.py',['transition','process_cov','robust_filter','sequence_score'])
robust_filter=STATE['robust_filter']
MonotoneLogistic=load_functions('run_venice_physics_reinforced_nadir.py',['MonotoneLogistic'])['MonotoneLogistic']

def log_curve(t,q,d,g):
    return np.logaddexp(np.log1p(-q)-d*t,np.log(q)+g*t)

def endpoint(t,y):
    """Earliest observed local low after cycle 3, confirmed by ONE strict rise.
    Adjacent AVAILABLE measurements; ties allowed on the declining side.
    Returns (low_cycle, confirmation_cycle, index), or (nan,nan,-1).
    """
    t=np.asarray(t,float);y=np.asarray(y,float)
    if len(t)!=len(y) or np.any(np.diff(t)<=0):raise ValueError('Unordered/duplicate endpoint times')
    good=np.isfinite(t)&np.isfinite(y);t=t[good];y=y[good]
    for i in range(1,len(y)-1):
        if t[i]>3 and y[i]<=y[i-1] and y[i+1]>y[i]:return float(t[i]),float(t[i+1]),i
    return np.nan,np.nan,-1

def batch_endpoint(Y,t=None):
    Y=np.asarray(Y,float)
    t=np.arange(Y.shape[1],dtype=float) if t is None else np.asarray(t,float)
    ok=(Y[:,1:-1]<=Y[:,:-2])&(Y[:,2:]>Y[:,1:-1])
    ok[:,t[1:-1]<=3]=False
    any_ok=ok.any(axis=1);ans=t[np.argmax(ok,axis=1)+1].astype(float)
    ans[~any_ok]=np.nan
    return ans

def fit_one(item):
    pid,t,y=item
    fit=fit_curve(t,y/y[0])
    if fit is None:return {'id':pid,'fit_success':False}
    q,d,g=fit['params'];tf=fit['t_hat'];inside=bool(np.isfinite(tf) and 0<=tf<=max(t))
    return dict(id=pid,fit_success=True,q_hat=q,decline_rate_hat=d,growth_rate_hat=g,
        tmin_fit_hat=tf,rmse_log10=fit['rmse_log10'],sigma_log10=fit['sigma_log10'],
        max_observed_cycle=max(t),n_points=len(t),fit_tmin_in_observed_window=inside,
        growth_near_lower_bound=g<=1.01e-5,q_near_lower_bound=q<=1.01e-8,
        retained=not ((not inside) and (g<=1.01e-5 or q<=1.01e-8)))

def parallel_fits(items,njobs):
    from joblib import Parallel,delayed
    return pd.DataFrame(Parallel(n_jobs=njobs)(delayed(fit_one)(x) for x in items))

def matrix(long,ids,last):
    w=long.pivot(index='id',columns='cycle',values='psa').reindex(index=list(ids),columns=range(last+1))
    return np.log(w.to_numpy(float)/w[0].to_numpy(float)[:,None])

def features(Y,curvature=False):
    Y=np.asarray(Y,float);z=[Y[:,1:],np.diff(Y,axis=1)]
    if curvature and Y.shape[1]>2:z.append(np.diff(Y,n=2,axis=1))
    return np.column_stack(z)

def wquant(y,w,p=.5):
    good=np.isfinite(y)&np.isfinite(w)&(w>=0);y=np.asarray(y)[good];w=np.asarray(w)[good]
    if not len(y) or w.sum()<=0:return np.nan
    order=np.argsort(y);return float(np.interp(p,np.cumsum(w[order])/w.sum(),y[order]))

def metrics(y,p,prob=None,cutoff=8,threshold=.5):
    y=np.asarray(y,float);p=np.asarray(p,float);ok=np.isfinite(y)&np.isfinite(p);y=y[ok];p=p[ok]
    if not len(y):return {'n':0}
    err=p-y;out=dict(n=len(y),MAE=np.mean(abs(err)),RMSE=np.sqrt(np.mean(err**2)),bias=np.mean(err),
        Spearman_rho=spearmanr(y,p).statistic if np.std(y)>0 and np.std(p)>0 else np.nan,
        within1=np.mean(abs(err)<=1),within2=np.mean(abs(err)<=2))
    if prob is not None:
        z=np.asarray(prob,float)[ok];valid=np.isfinite(z);truth=(y[valid]>cutoff).astype(int);z=np.clip(z[valid],0,1)
        out.update(n_binary=len(z),n_late=int(truth.sum()))
        if len(z):out['Brier']=brier_score_loss(truth,z)
        if len(np.unique(truth))==2:
            out.update(AUROC=roc_auc_score(truth,z),balanced_accuracy=balanced_accuracy_score(truth,z>=threshold),
                early_recall=np.mean(z[truth==0]<threshold),late_recall=np.mean(z[truth==1]>=threshold))
    return plain(out)

def assert_ids(x,y,message):
    if set(map(str,x))!=set(map(str,y)):raise RuntimeError(message)

def residual_blocks(fits,long):
    blocks=[]
    for r in fits.itertuples():
        z=long[long.id==r.id].sort_values('cycle')
        if len(z)<2:continue
        residual=np.log(z.psa.to_numpy()/z.psa.iloc[0])-log_curve(z.cycle.to_numpy(),r.q_hat,r.decline_rate_hat,r.growth_rate_hat)
        residual=residual[1:];residual=residual-np.median(residual)
        if len(residual):blocks.append(residual)
    if not blocks:raise RuntimeError('No TRAIN residual sequences')
    return blocks

def noise_draw(blocks,n,length,rng):
    """Prespecified moving blocks of two TRAIN residuals; not assay-error estimates."""
    out=np.zeros((n,length))
    for i in range(n):
        seq=blocks[rng.integers(len(blocks))];dest=[]
        while len(dest)<length-1:
            j=rng.integers(len(seq));dest.extend([seq[j],seq[(j+1)%len(seq)]])
        out[i,1:]=dest[:length-1]
    return out
