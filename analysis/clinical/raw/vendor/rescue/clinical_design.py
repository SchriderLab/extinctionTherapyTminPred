"""The two clinical simulation presets retained in the manuscript.

The calibration generator consumes the historical core-regime random draws before
constructing broad_mix. Only broad_mix is simulated; preserving that RNG order
preserves the reported broad_mix parameter design.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from .runtime import write_csv,dump_json

def as_bool(s): return s.astype(str).str.lower().isin(['true','1','t','yes'])

def clean_d(cloud,p):
    d=pd.to_numeric(cloud.decline_rate_hat,errors='coerce');keep=np.isfinite(d)
    needed={'fit_tmin_in_observed_window','growth_near_lower_bound','q_near_lower_bound'}
    if needed<=set(cloud):keep &= ~((~as_bool(cloud.fit_tmin_in_observed_window))&(as_bool(cloud.growth_near_lower_bound)|as_bool(cloud.q_near_lower_bound)))
    x=d[keep].to_numpy(float);lo,hi=np.quantile(x,p['d_quantiles']);x=x[(x>=lo)&(x<=hi)]
    if len(x)<20:raise ValueError('Too few clean empirical decline rates')
    return x

def stream(n,rng):return (rng.permutation(n)+rng.random(n))/n

def effective_age(ages,turnover,founders):
    r=2*(1-float(turnover))
    if r<=1:return int(round(float(np.mean(ages))))
    return max(1,int(round(math.log(max(float(np.sum(r**ages))/founders,1e-12))/math.log(r))))

def generate(cloud: Path,preset: dict,out: Path) -> None:
    p=preset;n=int(p['n']);rng=np.random.default_rng(p['seed']);d=clean_d(pd.read_csv(cloud),p)
    multi=p['phase']=='clinical_compartments'
    if multi:
        # Historical RNG stream and broad_mix IDs are preserved; no core-regime simulations are run.
        rng.choice(d,size=n,replace=True);target=rng.choice(d,size=n,replace=True)
        for _ in range(5):stream(n,rng)
        rng.lognormal(np.log(.033),.28,n);rng.normal(.47,.045,n);rng.normal(.25,.045,n)
        ut=stream(n,rng);um=stream(n,rng);stream(n,rng);ur=stream(n,rng);uv=stream(n,rng)
        ids=np.arange(n)+n
    else:
        target=rng.choice(d,size=n,replace=True)
        ut=stream(n,rng);um=stream(n,rng);ur=stream(n,rng);uv=stream(n,rng);ids=np.arange(n)
    def scale(u,key):lo,hi=p[key];return lo+u*(hi-lo)
    turnover=scale(ut,'turnover');lo,hi=p['mutation_rate'];mu=10**(np.log10(lo)+um*(np.log10(hi)-np.log10(lo)))
    s=p['standing'];standing=np.clip(rng.lognormal(np.log(s['median']),s['sdlog'],n),*s['clip']);standing[rng.random(n)<s['zero_probability']]=0
    effect=scale(ur,'m1_mean_effect');sd=scale(uv,'m1_sdlog');ages=rng.integers(p['age_range'][0],p['age_range'][1]+1,size=(n,5))
    dose=1-np.exp(-target)/(2*(1-turnover))
    if not ((dose>0)&(dose<1)).all():raise ValueError('Clinical d/turnover combination yields invalid dose')
    rows=[]
    for j,pid in enumerate(ids):
        r={'patient_id':int(pid),'base_patient_id':j,'scenario':'broad_mix' if multi else 'broad_mix_k1_abc_ref',
           'standing_frac':float(standing[j]),'m1_effect_mean_target':float(effect[j]),'m1_meanlog':math.log(float(effect[j]))-.5*float(sd[j])**2,
           'm1_sdlog':float(sd[j]),'target_decay_rate':float(target[j]),'base_mut_rate':float(mu[j]),
           'dose_drug_1_mono':float(dose[j]),'patient_turnover':float(turnover[j]),'draw_m1':1.,'draw_m2':0.,'draw_m3':0.,'cross_resistance':0.,
           'patient_seed_base':int(p['seed_base'])+int(pid)*20}
        if multi:
            r['K']=5
            for k in range(1,6):r[f'age{k}']=int(ages[j,k-1]);r[f'standing{k}']=float(standing[j])
        else:
            r.update(source_k5_patient_id=-1,standing_target=float(standing[j]),n_founders=int(p['n_founders']),single_age=effective_age(ages[j],turnover[j],p['n_founders']))
            for k in range(1,6):r[f'pseudo_age{k}']=int(ages[j,k-1])
        rows.append(r)
    write_csv(pd.DataFrame(rows),out);dump_json(out.with_suffix('.design.json'),p)
