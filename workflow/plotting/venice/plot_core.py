"""Plotting and source-validation primitives. No fitting or simulated results."""
from __future__ import annotations
import hashlib, json, math, re, textwrap, tempfile, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

def json_ready(x):
    if isinstance(x,dict):return {str(k):json_ready(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [json_ready(v) for v in x]
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,(np.bool_,)):return bool(x)
    if isinstance(x,Path):return str(x)
    return x

class Blocked(RuntimeError):
    """A source/definition must be resolved before this experiment can be plotted."""


def boolean(s):
    if pd.api.types.is_bool_dtype(s): return s.fillna(False)
    v=s.astype('string').str.strip().str.lower()
    bad=v.notna() & ~v.isin(['true','false','1','0','1.0','0.0','yes','no','y','n',''])
    if bad.any(): raise Blocked(f'Unrecognized boolean values: {v[bad].unique().tolist()}')
    return v.isin(['true','1','1.0','yes','y']).fillna(False)


def col(df, names, role, optional=False):
    """Exact, case-insensitive aliases only. Prefer an explicit override in inputs.json."""
    lookup={str(k).casefold():k for k in df.columns}
    present=list(dict.fromkeys(lookup[x.casefold()] for x in names if x.casefold() in lookup))
    if not present:
        if optional: return None
        raise Blocked(f'{role}: expected one of {names}; found columns {list(df.columns)}')
    # Aliases may coexist only when they represent the same values.
    first=present[0]
    for other in present[1:]:
        a,b=df[first],df[other]
        an,bn=pd.to_numeric(a,errors='coerce'),pd.to_numeric(b,errors='coerce')
        numeric_equal=(an.notna().equals(bn.notna()) and np.allclose(an.dropna(),bn.dropna())
                       and an.notna().any())
        if not a.fillna('').astype(str).equals(b.fillna('').astype(str)) and not numeric_equal:
            raise Blocked(f'{role}: conflicting columns {present}. Set an exact column override; not choosing silently.')
    return first


def numeric(df, column):
    x=pd.to_numeric(df[column], errors='coerce')
    if x.isna().any() or not np.isfinite(x).all():
        raise Blocked(f'Column {column} contains missing/nonfinite/non-numeric values in the selected analysis rows.')
    return x.astype(float)


def unique(df, keys, what):
    if df.duplicated(keys).any():
        raise Blocked(f'{what}: duplicate {keys}; do not pool CV repeats or different experiments silently.')


def rhoval(y,p):
    if len(np.unique(y))<2 or len(np.unique(p))<2:return float('nan')
    return float(spearmanr(y,p).statistic)


def regression_metrics(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float); e=p-y
    return dict(n=len(y),MAE=float(np.mean(abs(e))),median_AE=float(np.median(abs(e))),
                RMSE=float(np.sqrt(np.mean(e**2))),bias=float(np.mean(e)),rho=rhoval(y,p),
                within_1=float(np.mean(abs(e)<=1)),within_2=float(np.mean(abs(e)<=2)))


def cluster_interval(df,value,cluster='id',B=1000,seed=713):
    """Percentile CI of a row-weighted mean, resampling independent patients."""
    z=df.groupby(cluster,sort=True)[value].agg(['sum','count'])
    if len(z)<3:return np.nan,np.nan
    rng=np.random.default_rng(seed); out=[]
    for _ in range(B):
        j=rng.integers(0,len(z),len(z)); out.append(z['sum'].to_numpy()[j].sum()/z['count'].to_numpy()[j].sum())
    return tuple(np.quantile(out,[.025,.975]))


def group_curve(df,x,y):
    rows=[]
    for k,g in df.groupby(x,sort=True):
        lo,hi=cluster_interval(g,y,B=500)
        rows.append({x:k,'mean':g[y].mean(),'lo':lo,'hi':hi,'n_patients':g.id.nunique()})
    return pd.DataFrame(rows)


class Store:
    def __init__(self,root,repo,config,out):
        self.root,self.repo,self.cfg,self.out=Path(root),Path(repo),config,Path(out)
        self.used={}; self.headers={}; self.cache={}
    def path(self,key):
        spec=self.cfg['sources'][key]
        paths=[spec['path']] if spec.get('path') else spec.get('paths',[])
        found=[]
        for p in paths:
            p=Path(p.replace('{root}',str(self.root)).replace('{repo}',str(self.repo)))
            if p.is_file() and p.resolve() not in found:found.append(p.resolve())
        if spec.get('path') and not found:
            raise Blocked(f'{key}: explicit source path does not exist: {paths[0]}; not substituting another run.')
        # A bounded directory lookup is allowed only for this registered experiment.
        if not found and spec.get('directories'):
            for folder in spec['directories']:
                folder=Path(folder.replace('{root}',str(self.root)).replace('{repo}',str(self.repo)))
                if not folder.is_dir():continue
                files=sorted(folder.glob(spec.get('glob','*.csv')))
                if len(files)>100:raise Blocked(f'{key}: >100 candidates in {folder}; set an exact path.')
                for p in files:
                    try:h=pd.read_csv(p,nrows=0).columns.tolist()
                    except Exception:continue
                    self.headers[str(p)]=h
                    requirements=spec.get('required_header',[])
                    if requirements and all(any(a in h for a in group) for group in requirements):
                        if p.resolve() not in found:found.append(p.resolve())
        if not found: raise Blocked(f'{key}: source missing. Expected: {paths or spec.get("directories",[])}')
        if len(found)>1:
            hashes={hashlib.sha256(p.read_bytes()).hexdigest() for p in found}
            if len(hashes)>1:raise Blocked(f'{key}: conflicting source copies: {[str(p) for p in found]}; set sources.{key}.path.')
        p=found[0]
        self.used[key]={'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        return p
    def read(self,key,usecols=None):
        tag=(key,tuple(usecols or []))
        if tag in self.cache:return self.cache[tag].copy()
        p=self.path(key)
        if p.suffix.lower()!='.csv':raise Blocked(f'{key}: expected CSV, got {p}')
        d=pd.read_csv(p,low_memory=False,usecols=usecols)
        self.headers[str(p)]=d.columns.tolist(); self.cache[tag]=d
        return d.copy()
    def column(self,key,df,role,aliases,optional=False):
        override=self.cfg['sources'][key].get('columns',{}).get(role)
        return col(df,[override] if override else aliases,f'{key}.{role}',optional)
    def export(self,stem,df):
        d=self.out/'source_data';d.mkdir(parents=True,exist_ok=True)
        df.to_csv(d/(stem+'.csv'),index=False)


# Paper-width panels with dedicated title and legend strips; no legends on data.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,
 'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':.8,'lines.linewidth':1.5,'lines.markersize':4,'pdf.fonttype':42,
 'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})

class Page:
    def __init__(self,n=4,columns=2,width=8.4):
        self.n,self.columns=n,columns;self.rows=math.ceil(n/columns)
        self.width=width;self.rowheight=3.98; self.height=self.rows*self.rowheight+.35
        self.fig=plt.figure(figsize=(width,self.height),dpi=150)
        self.axes=[];self.titles=[];self.info=[];self.legends=[];self.own=[]
        self.labels=[]
        cw=width/columns
        for k in range(n):
            r,c=divmod(k,columns); left=c*cw+.64;right=(c+1)*cw-.22
            top=self.height-.78-r*self.rowheight;bottom=top-2.42
            ax=self.fig.add_axes([left/width,bottom/self.height,(right-left)/width,2.42/self.height])
            self.axes.append(ax)
            t=self.fig.text((left+.16)/width,(top+.47)/self.height,'',ha='left',va='center',fontsize=9.3)
            letter=self.fig.text((left-.04)/width,(top+.47)/self.height,chr(65+k),weight='bold',ha='right',va='center',fontsize=11)
            subtitle=self.fig.text(left/width,(top+.085)/self.height,'',ha='left',va='bottom',fontsize=7.4)
            self.titles.append(t);self.info.append(subtitle);self.labels.append(letter)
            self.own.append((left,right,bottom,top))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5));ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        self.caption='';self.stats={}
    def __getitem__(self,k):return self.axes[k]
    def title(self,k,text,subtitle=''):
        # Deliberate short panel titles. Render-measured wrapping as a second guard.
        self.titles[k].set_text(text);self.info[k].set_text(subtitle)
    def reserve_ylabels(self,k,inches=.65):
        ax=self.axes[k];box=ax.get_position();dx=inches/self.width
        ax.set_position([box.x0+dx,box.y0,box.width-dx,box.height])
    def key(self,k,**kw):
        ax=self.axes[k];handles,labels=ax.get_legend_handles_labels()
        if not handles:return
        d=dict(zip(labels,handles));labels=list(d);handles=list(d.values())
        left,right,bottom,_=self.own[k]
        labels=['\n'.join(textwrap.wrap(s,29)) for s in labels]
        leg=self.fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(left/self.width,(bottom-.51)/self.height),
                            bbox_transform=self.fig.transFigure,frameon=False,fontsize=7.2,
                            ncol=1 if len(labels)<=3 else 2,handlelength=2.2,borderaxespad=0,**kw)
        self.legends.append((k,leg))
    def finish(self,out,stem,caption,stats=None,dpi=600):
        self.fig.canvas.draw();ren=self.fig.canvas.get_renderer()
        for k,t in enumerate(self.titles):
            left,right,_,_=self.own[k];maxw=(right-left-.17)*self.fig.dpi
            original=t.get_text()
            for width in [45,36,31,27,23]:
                if t.get_window_extent(ren).width<=maxw:break
                t.set_text('\n'.join(textwrap.wrap(original,width=width)))
                t.set_fontsize(9.0);self.fig.canvas.draw()
            if t.get_window_extent(ren).width>maxw:raise Blocked(f'Layout: title too wide in {stem} panel {k+1}')
        # Fit metadata into its own strip; no arbitrary clipping of long subtitles.
        for k,t in enumerate(self.info):
            left,right,_,_=self.own[k];maxw=(right-left)*self.fig.dpi
            original=t.get_text()
            for width in [65,58,51,45]:
                if t.get_window_extent(ren).width<=maxw:break
                t.set_text('\n'.join(textwrap.wrap(original,width=width)))
                self.fig.canvas.draw()
            if t.get_window_extent(ren).width>maxw:raise Blocked(f'Layout: metadata too wide in {stem} panel {k+1}')
        # Check text geometry at final positions, including ticks, headers and legends.
        self.fig.canvas.draw();ren=self.fig.canvas.get_renderer();groups=[]
        for k,ax in enumerate(self.axes):
            box=ax.get_tightbbox(ren);groups.append((f'axes{k}',k,box))
            for label,t in [('title',self.titles[k]),('panel',self.labels[k]),('subtitle',self.info[k])]:
                if t.get_text():groups.append((f'{label}{k}',k,t.get_window_extent(ren)))
        for k,leg in self.legends:groups.append((f'legend{k}',k,leg.get_window_extent(ren)))
        collisions=[]
        bounds=self.fig.bbox
        for name,k,b in groups:
            if b.x0 < -1 or b.y0 < -1 or b.x1 > bounds.x1+1 or b.y1 > bounds.y1+1:
                collisions.append(f'{name} outside page')
        for i,(an,ai,a) in enumerate(groups):
            for bn,bi,b in groups[i+1:]:
                if an.startswith('axes') and bn.startswith('axes') and ai==bi:continue
                if min(a.x1,b.x1)-max(a.x0,b.x0)>2 and min(a.y1,b.y1)-max(a.y0,b.y0)>2:
                    collisions.append(f'{an} overlaps {bn}')
        if collisions:
            debug=Path(out)/'_layout_debug';debug.mkdir(parents=True,exist_ok=True)
            self.fig.savefig(debug/(stem+'.png'),dpi=150)
            plt.close(self.fig)
            raise Blocked(f'Layout check failed for {stem}: {collisions}. Debug image saved; no final figure emitted.')
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        # Stage all formats before publishing any final artifact for this figure.
        with tempfile.TemporaryDirectory(prefix='.render_',dir=out) as tmp:
            tmp=Path(tmp)
            for ext in ['pdf','svg','png']:
                self.fig.savefig(tmp/f'{stem}.{ext}',dpi=dpi if ext=='png' else 300)
            (tmp/(stem+'.caption.md')).write_text(caption+'\n',encoding='utf-8')
            (tmp/(stem+'.metrics.json')).write_text(json.dumps(json_ready(stats or {}),indent=2,allow_nan=False))
            for file in tmp.iterdir():os.replace(file,out/file.name)
        plt.close(self.fig)
        return stem


def scatter_prediction(ax,y,p):
    m=regression_metrics(y,p);ax.scatter(y,p,s=17,alpha=.65,edgecolors='none')
    lo=min(np.min(y),np.min(p));hi=max(np.max(y),np.max(p));pad=max(.5,(hi-lo)*.06)
    ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],ls='--',alpha=.55)
    ax.set_xlim(lo-pad,hi+pad);ax.set_ylim(lo-pad,hi+pad)
    ax.set_xlabel('Observed target time (cycles)');ax.set_ylabel('Predicted target time (cycles)')
    return m


def ecdf(ax,values,label):
    x=np.sort(np.asarray(values,float));ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',label=label)


def paired_regression_page(d,pred='prediction',base='baseline',model='Model',baseline='Population baseline'):
    unique(d,['id'],'Regression comparison');y=d.target.to_numpy();p=d[pred].to_numpy();b=d[base].to_numpy()
    page=Page(4);m=scatter_prediction(page[0],y,p)
    page.title(0,'Individualized nadir prediction',f'n = {len(d)}; MAE = {m["MAE"]:.2f}; Spearman rho = {m["rho"]:.2f}')
    ax=page[1];delta=np.abs(b-y)-np.abs(p-y);ax.hist(delta,bins='auto');ax.axvline(0,ls='--',alpha=.5)
    ax.set_xlabel('Absolute-error reduction (cycles)');ax.set_ylabel('Patients')
    z=pd.DataFrame({'id':d.id,'delta':delta});lo,hi=cluster_interval(z,'delta')
    page.title(1,'Benefit over the matched baseline',f'Mean reduction {delta.mean():.2f}; patient bootstrap 95% CI [{lo:.2f}, {hi:.2f}]')
    ax=page[2];ecdf(ax,np.abs(p-y),model);ecdf(ax,np.abs(b-y),baseline)
    ax.set_xlabel('Absolute prediction error (cycles)');ax.set_ylabel('Fraction of patients');ax.set_ylim(0,1.03)
    page.title(2,'How often is the prediction close?');page.key(2)
    ax=page[3];ax.scatter(y,p-y,s=17,alpha=.6,edgecolors='none');ax.axhline(0,ls='--',alpha=.5)
    ax.set_xlabel('Observed target time (cycles)');ax.set_ylabel('Prediction minus target (cycles)')
    page.title(3,'Timing bias across the target range',f'Positive residual = prediction too late; mean bias = {m["bias"]:.2f}')
    stats={model:m,baseline:regression_metrics(y,b),'paired_mean_error_reduction':delta.mean(),'paired_CI95':[lo,hi]}
    return page,stats


def verify_prob(y,p):
    if not set(np.unique(y)).issubset({0,1}):raise Blocked('Nonbinary event label.')
    if not np.isfinite(p).all() or np.min(p)<0 or np.max(p)>1:raise Blocked('Invalid probability values; not clipping.')
    if len(np.unique(y))!=2:raise Blocked('Only one outcome class is present.')


def roc(ax,y,p,label):
    verify_prob(y,p);a=roc_auc_score(y,p);x,z,_=roc_curve(y,p);ax.plot(x,z,label=f'{label}: {a:.3f}')
    ax.set_xlabel('False-positive rate');ax.set_ylabel('True-positive rate');ax.set_xlim(0,1);ax.set_ylim(0,1)
    return a


def calibration(ax,y,p,label):
    verify_prob(y,p);d=pd.DataFrame({'y':y,'p':p});d['bin']=pd.qcut(d.p,8,duplicates='drop')
    g=d.groupby('bin',observed=True).agg(predicted=('p','mean'),observed=('y','mean'),n=('y','size'))
    ax.plot(g.predicted,g.observed,'o-',label=label)
    ax.set_xlabel('Mean predicted probability');ax.set_ylabel('Observed event frequency');ax.set_xlim(0,1);ax.set_ylim(0,1)
    return g.reset_index(drop=True)
