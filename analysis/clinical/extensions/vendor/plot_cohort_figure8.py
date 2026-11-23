#!/usr/bin/env python3
"""Redraw historical VENICE cohort Figure 8 with the requested panel titles.

Uses the original historical cohort reconstruction. No forecasting model is run.
Raw patient files are read locally and are not copied into the figure package.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def reconstruct(adsl,adlb):
    a=pd.read_csv(adsl,dtype={'USUBJID':str})
    if a.USUBJID.duplicated().any():raise ValueError('Duplicate ADSL subject IDs')
    selected=a.RANDOM.astype(str).str.upper().eq('Y') & a.ARMCD.astype(str).str.upper().eq('PLACEBO')
    r=a[selected].copy();r['id']=r.USUBJID.astype(str)
    baseline=pd.DataFrame({'id':r.id,'cycle':0.,'psa':pd.to_numeric(r.TMPSA,errors='coerce'),'day':0.})
    baseline=baseline[baseline.psa.gt(0)&np.isfinite(baseline.psa)]
    columns=['USUBJID','LBTESTCD','LBSTRESN','ANLVISN','EPOCH','LBDY2','LBDY']
    l=pd.read_csv(adlb,usecols=columns,dtype={'USUBJID':str},low_memory=False)
    l=l[l.USUBJID.isin(r.id)&l.LBTESTCD.astype(str).str.upper().eq('PSA')&l.EPOCH.astype(str).str.lower().eq('treatment phase')].copy()
    l['id']=l.USUBJID;l['cycle']=pd.to_numeric(l.ANLVISN,errors='coerce');l['psa']=pd.to_numeric(l.LBSTRESN,errors='coerce')
    l['day']=pd.to_numeric(l.LBDY2,errors='coerce').fillna(pd.to_numeric(l.LBDY,errors='coerce'))
    l=l[l.cycle.ge(1)&l.psa.gt(0)&np.isfinite(l.psa)].copy()
    l['delta']=(l.day-21*l.cycle).abs().fillna(np.inf)
    l=l.sort_values(['id','cycle','delta','day'],kind='stable').drop_duplicates(['id','cycle'])
    t=pd.concat([baseline,l[['id','cycle','psa','day']]],ignore_index=True).sort_values(['id','cycle'])
    n=t[t.cycle.isin([0,1,2,3])].groupby('id').cycle.nunique()
    early=set(n[n.eq(4)].index);post=set(l[l.cycle.gt(3)].id)&early
    rows=[]
    for pid in r.id:
        g=t[t.id.eq(pid)].sort_values('cycle');target=np.nan;inside=False
        if pid in early:
            tv=g.cycle.to_numpy();v=g.psa.to_numpy();j=np.argmin(v)
            inside=bool(tv[j]>3 and j<len(g)-1)
            if inside:
                for k in range(1,len(g)-1):
                    if tv[k]>3 and v[k]<=v[k-1] and v[k+1]>v[k]:target=tv[k];break
        if pid not in early:reason='Incomplete early PSA'
        elif pid not in post:reason='No post-cycle-3 PSA'
        elif not inside:reason='No future interior global minimum'
        elif not np.isfinite(target):reason='First rebound target unavailable'
        else:reason='Eligible first-rebound target'
        rows.append(dict(id=pid,locked=inside,target_t=target,group=reason,n_psa=len(g)))
    m=pd.DataFrame(rows)
    counts=[len(a),len(r),len(early),len(post),int(m.locked.sum()),int(m.target_t.notna().sum())]
    if counts!=[936,612,506,482,268,263]:
        raise RuntimeError(f'Historical cohort changed: {counts}; do not alter the filters to force counts.')
    return m,t,counts


def plot(adsl,adlb,out,split=None):
    m,t,counts=reconstruct(adsl,adlb)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
        'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
        'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42})
    fig,a=plt.subplots(2,2,figsize=(11.6,8.0),layout='constrained')
    for i,ax in enumerate(a.flat):
        ax.text(-.12,1.07,chr(65+i),transform=ax.transAxes,weight='bold',fontsize=12)
    ax=a[0,0];ys=np.arange(len(counts))[::-1]
    ax.plot(np.zeros(len(counts)),ys,'o-',color='#3269A8',ms=5)
    names=['Available subject records','Randomized comparator arm','Positive PSA at cycles 0, 1, 2, 3',
           'At least one later treatment PSA','Global minimum after cycle 3, with later PSA','Evaluable first local rebound']
    for i,(n,y,label) in enumerate(zip(counts,ys,names)):
        ax.text(.08,y+.12,f'{n:,}  {label}',fontsize=9,va='center')
        if i:ax.text(.08,y-.24,f'{counts[i-1]-n:,} excluded',fontsize=8,color='#69727A')
    ax.set(xlim=(-.08,1.9),ylim=(-.5,5.7));ax.axis('off')
    ax.set_title('Selection criteria for ABC prediction test',loc='left')
    order=['Eligible first-rebound target','First rebound target unavailable','No future interior global minimum','No post-cycle-3 PSA','Incomplete early PSA']
    if split is not None:
        membership=pd.read_csv(split,dtype={'id':str})
        test_ids=set(membership.loc[membership.split.eq('TEST'),'id'])
        train_ids=set(membership.loc[membership.split.eq('TRAIN'),'id'])
        if len(test_ids)!=47 or len(train_ids)!=184 or test_ids & train_ids:
            raise ValueError('Expected disjoint TRAIN184/TEST47 in frozen split')
        m.loc[m.target_t.notna(),'group']='Other eligible (32)'
        m.loc[m.id.isin(train_ids),'group']='Later TRAIN (184)'
        m.loc[m.id.isin(test_ids),'group']='Later TEST (47)'
        m.loc[m.target_t.isna(),'group']='Other comparator (349)'
        order=['Later TEST (47)','Later TRAIN (184)','Other eligible (32)','Other comparator (349)']
    z=m.copy();z['order']=z.group.map(dict(zip(order,range(5))))
    z=z.sort_values(['order','n_psa','id'],ascending=[True,False,True])
    visits=np.arange(int(t.cycle.max())+1);matrix=np.full((len(z),len(visits)),np.nan)
    index={pid:i for i,pid in enumerate(z.id)}
    for pid,g in t.groupby('id'):
        v=g.cycle.to_numpy();v=v[np.isclose(v,np.round(v))].astype(int)
        matrix[index[pid],v]=1
    ax=a[0,1]
    ax.imshow(np.ma.masked_invalid(matrix),origin='upper',aspect='auto',interpolation='nearest',
              extent=[-.5,visits[-1]+.5,len(z)-.5,-.5],cmap='Blues',vmin=0,vmax=1)
    end=0
    for group in order:
        end+=int(z.group.eq(group).sum())
        if 0<end<len(z):ax.axhline(end-.5,color='#586773',ls='--',lw=.6)
    ax.axvline(3.5,color='#797F85',ls=':',lw=1)
    ax.set(xlabel='Scheduled treatment cycle',ylabel='Comparator patients, grouped by eligibility',
           title='Duration of PSA follow-up per patient')
    if split is not None:
        sizes=[int(z.group.eq(group).sum()) for group in order]
        centers=np.cumsum(sizes)-np.array(sizes)/2-.5
        ax.set_yticks(centers,order,fontsize=8)
        ax.set_ylabel('Clinical analysis groups')
    ax.text(.5,-.18,'612 patients; white = no retained PSA',transform=ax.transAxes,ha='center',fontsize=8)
    ax=a[1,0]
    for label,ids,color in [('Eligible',set(m[m.target_t.notna()].id),'#3269A8'),
                           ('Other comparator',set(m[m.target_t.isna()].id),'#C56528')]:
        g=t[t.id.isin(ids)].groupby('cycle').id.nunique().reindex(visits,fill_value=0)
        ax.plot(visits,g/len(ids),color=color,label=f'{label} (n={len(ids)})',lw=1.8)
    ax.set(xlabel='Scheduled treatment cycle',ylabel='Fraction with a retained PSA value',ylim=(0,1.04),
           title='Summary of PSA observation windows');ax.legend()
    ax=a[1,1];v=m.target_t.dropna();edges=np.arange(v.min()-.5,v.max()+1.5)
    ax.hist(v,bins=edges,color='#3269A8',edgecolor='white')
    ax.axvline(3,color='#6A747C',ls='--',label='Last early input (cycle 3)')
    ax.set(xlabel=r'First local rebound nadir $T_{\mathrm{reb}}$ (cycle)',ylabel='Eligible patients',
           title=r'Distribution of $T_{\mathrm{reb}}$ in selected patient records')
    ax.text(.98,.94,f'n = {len(v)}\nMedian = {v.median():g}\nIQR = {v.quantile(.25):g}-{v.quantile(.75):g}',
            ha='right',va='top',transform=ax.transAxes,fontsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True));ax.legend(loc='upper center')
    fig.savefig(out/'E01_cohort_eligibility.pdf',bbox_inches='tight')
    fig.savefig(out/'E01_cohort_eligibility.png',bbox_inches='tight',dpi=220);plt.close(fig)
    (out/'FIGURE8_VALIDATION.json').write_text(json.dumps({'counts':counts,'endpoint':'First post-cycle-3 local low with previous>=low and next>low, after historical global-minimum gate',
        'input_sha256':{str(adsl):sha256(adsl),str(adlb):sha256(adlb)},'n_abc_after_separate_support_horizon_filters':90},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adsl',type=Path,required=True);p.add_argument('--adlb',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--split',type=Path,required=True)
    args=p.parse_args()
    if 'login' in socket.gethostname().lower() and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit the plotting job with sbatch or srun on a compute node.')
    plot(args.adsl,args.adlb,args.out,args.split)
