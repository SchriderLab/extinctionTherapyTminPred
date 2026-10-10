"""Experiment-specific figures. Each output has one target definition and input set."""
from __future__ import annotations
import json,re
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from plot_core import (Page,Blocked,boolean,col,numeric,unique,regression_metrics,rhoval,
                        paired_regression_page,ecdf,roc,calibration,cluster_interval,group_curve)
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

REGISTRY={}

def experiment(code,title):
    def wrap(fn): REGISTRY[code]=(title,fn);return fn
    return wrap

def finish(s,page,stem,caption,stats=None):
    if s.cfg.get('test_fixture'):
        page.fig.text(.5,.008,'SOFTWARE TEST FIXTURE - NOT STUDY RESULTS',ha='center',va='bottom',fontsize=8,weight='bold')
    return page.finish(s.out,stem,caption,stats,dpi=s.cfg.get('dpi',600))

def pid(s,key,d):
    c=s.column(key,d,'id',['id','USUBJID','patient_id','clinical_id']);v=d[c]
    if v.isna().any():raise Blocked(f'{key}: missing IDs')
    return v.astype(str).str.strip()

def get(s,key,d,role,aliases,optional=False):
    c=s.column(key,d,role,aliases,optional);return None if c is None else numeric(d,c)

def as_standard(s,key,d,fields):
    z=pd.DataFrame({'id':pid(s,key,d)})
    for role,aliases in fields.items():z[role]=get(s,key,d,role,aliases)
    return z


def cohort(s):
    if hasattr(s,'cohort_cache'):return tuple(x.copy() for x in s.cohort_cache)
    a=s.read('adsl');required=['USUBJID','RANDOM','ARMCD','TMPSA']
    if not set(required)<=set(a):raise Blocked('Cohort extraction requires ADSL columns '+str(required))
    unique(a,['USUBJID'],'ADSL subject file')
    rand=a.RANDOM.astype(str).str.upper().eq('Y')&a.ARMCD.astype(str).str.upper().eq('PLACEBO')
    r=a[rand].copy();r['id']=r.USUBJID.astype(str);base=pd.to_numeric(r.TMPSA,errors='coerce')
    b=pd.DataFrame({'id':r.id,'cycle':0.,'psa':base,'day':0.});b=b[b.psa.gt(0)&np.isfinite(b.psa)].copy()
    use=['USUBJID','LBTESTCD','LBSTRESN','ANLVISN','EPOCH','LBDY2','LBDY']
    l=s.read('adlb',usecols=use)
    l=l[l.USUBJID.astype(str).isin(r.id)&l.LBTESTCD.astype(str).str.upper().eq('PSA')&l.EPOCH.astype(str).str.lower().eq('treatment phase')].copy()
    l['id']=l.USUBJID.astype(str);l['cycle']=pd.to_numeric(l.ANLVISN,errors='coerce');l['psa']=pd.to_numeric(l.LBSTRESN,errors='coerce')
    l['day']=pd.to_numeric(l.LBDY2,errors='coerce').fillna(pd.to_numeric(l.LBDY,errors='coerce'))
    l=l[l.cycle.ge(1)&l.psa.gt(0)&np.isfinite(l.psa)].copy()
    l['delta']=(l.day-21*l.cycle).abs().fillna(np.inf)
    l=l.sort_values(['id','cycle','delta','day'],kind='stable').drop_duplicates(['id','cycle'])
    t=pd.concat([b,l[['id','cycle','psa','day']]],ignore_index=True).sort_values(['id','cycle'])
    n=t[t.cycle.isin([0,1,2,3])].groupby('id').cycle.nunique();early=set(n[n.eq(4)].index)
    post=set(l[l.cycle.gt(3)].id)&early
    rows=[]
    for id in r.id:
        g=t[t.id.eq(id)].sort_values('cycle');target=np.nan;global_t=np.nan;inside=False
        if id in early:
            tv=g.cycle.to_numpy();v=g.psa.to_numpy();j=np.argmin(v);global_t=tv[j];inside=bool(global_t>3 and j<len(g)-1)
            if inside:
                for k in range(1,len(g)-1):
                    if tv[k]>3 and v[k]<=v[k-1] and v[k+1]>v[k]:target=tv[k];break
        if id not in early:reason='Incomplete early PSA'
        elif id not in post:reason='No post-cycle-3 PSA'
        elif not inside:reason='No future interior global minimum'
        elif not np.isfinite(target):reason='First rebound target unavailable'
        else:reason='Eligible first-rebound target'
        rows.append(dict(id=id,early_complete=id in early,has_post3=id in post,locked=inside,
                         global_nadir_cycle=global_t,target_t=target,group=reason,n_psa=len(g)))
    m=pd.DataFrame(rows)
    counts=pd.DataFrame({'stage':['Subject records in supplied file','Randomized comparator arm','Complete baseline + cycles 1-3','Any PSA after cycle 3','Future interior global minimum','First rebound target available'],
        'n':[len(a),len(r),len(early),len(post),int(m.locked.sum()),int(m.target_t.notna().sum())]})
    expected=s.cfg.get('cohort_expected_counts')
    if expected is not None and counts.n.tolist()!=expected:
        s.export('E01_failed_cohort_counts',counts)
        raise Blocked(f'Frozen cohort audit changed: {counts.n.tolist()}, expected {expected}. Do not adjust filters to force counts.')
    s.cohort_cache=(m,t,counts)
    return m.copy(),t.copy(),counts.copy()


@experiment('E01','Clinical source population and forecasting eligibility')
def e01(s):
    m,t,c=cohort(s);p=Page(4);ax=p[0]
    # A simple count-flow, with explicit exclusions, not an implied treatment/no-treatment split.
    ys=np.arange(len(c))[::-1];ax.plot(np.zeros(len(c)),ys,'o-',markersize=5)
    ax.set_xlim(-.16,1.75);ax.set_ylim(-.6,len(c)-.4);ax.axis('off')
    names=['Available subject records','Randomized PLACEBO/comparator','Positive PSA at cycles 0, 1, 2, 3','At least one later treatment PSA','Global nadir > 3, with a later PSA','First post-3 rebound target']
    for i,(n,y,label) in enumerate(zip(c.n,ys,names)):
        ax.text(.08,y+.16,f'{n:,}  {label}',fontsize=8,va='center')
        if i>0:ax.text(.08,y-.27,f'{int(c.n.iloc[i-1]-n):,} excluded at this step',fontsize=7,alpha=.75)
    p.title(0,'Who entered the original ABC test?')
    ax=p[1];order=['Eligible first-rebound target','First rebound target unavailable','No future interior global minimum','No post-cycle-3 PSA','Incomplete early PSA']
    z=m.copy();z['order']=z.group.map(dict(zip(order,range(len(order)))));z=z.sort_values(['order','n_psa','id'],ascending=[True,False,True])
    visits=np.arange(int(t.cycle.max())+1);matrix=np.full((len(z),len(visits)),np.nan);idx={id:i for i,id in enumerate(z.id)}
    for id,g in t.groupby('id'):
        if id in idx:
            v=g.cycle.to_numpy();v=v[np.isclose(v,np.round(v))].astype(int);matrix[idx[id],v]=1
    ax.imshow(np.ma.masked_invalid(matrix),origin='upper',aspect='auto',interpolation='nearest',extent=[-.5,visits[-1]+.5,len(z)-.5,-.5])
    end=0
    for group in order:
        end+=int(z.group.eq(group).sum())
        if 0<end<len(z):ax.axhline(end-.5,ls='--',lw=.6,alpha=.8)
    ax.axvline(3.5,ls=':',lw=1);ax.set_xlabel('Scheduled treatment cycle');ax.set_ylabel('Randomized patients, grouped by eligibility')
    p.title(1,'PSA availability across follow-up',f'n = {len(z)}; eligible first; white = no retained PSA')
    ax=p[2]
    for label,ids in [('Eligible',set(m[m.target_t.notna()].id)),('Other randomized',set(m[m.target_t.isna()].id))]:
        if not ids:continue
        gg=t[t.id.isin(ids)].groupby('cycle').id.nunique().reindex(visits,fill_value=0)
        ax.plot(visits,gg/len(ids),label=f'{label} (n={len(ids)})')
    ax.set_xlabel('Scheduled treatment cycle');ax.set_ylabel('Fraction with a retained PSA value');ax.set_ylim(0,1.04)
    p.title(2,'Observation opportunity by eligibility');p.key(2)
    ax=p[3];v=m.target_t.dropna();edges=np.arange(v.min()-.5,v.max()+1.5,1);ax.hist(v,bins=edges)
    ax.axvline(3,ls='--',label='End of four-point input');ax.set_xlabel('First rebound-associated nadir (cycle)');ax.set_ylabel('Eligible patients')
    p.title(3,'What future event was forecast?',f'n = {len(v)}; median = {v.median():g}; IQR = {v.quantile(.25):g}-{v.quantile(.75):g}');p.key(3)
    s.export('E01_selection_by_patient',m);s.export('E01_counts',c);s.export('E01_retained_psa',t)
    s.export('E01_availability_order',z[['id','group','n_psa']])
    caption=('Original VENICE evaluation cohort, reconstructed from the supplied subject and laboratory files. PLACEBO is the randomized comparator assignment in this extract, not a claim of no background treatment. '
      'Baseline is ADSL TMPSA at cycle 0; treatment-phase PSA duplicates are resolved by proximity to 21 x cycle days. The first occurrence of the global minimum must be after cycle 3 and have a subsequent measurement. '
      'The endpoint is then the first post-cycle-3 local low followed by one higher measurement. This is retrospective outcome-evaluability selection, NOT a prospective clinical eligibility rule or a population prevalence estimate. '
      'The treated/safety count is not used as an additional filter. Panels B-C retain all randomized comparator patients; panel D uses evaluable targets. Selection strata in B are listed in source_data/E01_availability_order.csv.')
    return [finish(s,p,'E01_cohort_eligibility',caption,{'counts':c.to_dict('records')})]


@experiment('E02','Original direct ABC forecasting, frozen support and horizon')
def e02(s):
    key='direct_abc';d=s.read(key);sp=s.column(key,d,'supported',['supported']);hp=s.column(key,d,'horizon',['within_horizon_observed'])
    all_n=len(d);n_support=int(boolean(d[sp]).sum());d=d[boolean(d[sp])&boolean(d[hp])].copy()
    z=as_standard(s,key,d,{'target':['observed_target_cycle'],'prediction':['pred_abc_idw']})
    bc=s.column(key,d,'baseline',['pred_reference_median','pred_ref_median','pred_naive_median','pred_baseline','reference_median_target'],True)
    if bc is not None:z['baseline']=numeric(d,bc)
    else:
        text=s.path('direct_summary').read_text();hits=re.findall(r'(?m)^\s*(?:reference_median_target|reference_target_median)\s*=\s*([\d.eE+-]+)\s*$',text)
        if not hits or len(set(float(x) for x in hits))!=1:raise Blocked('No unique frozen reference-median target in original direct-ABC source. Do not substitute the clinical median.')
        z['baseline']=float(hits[0])
    p,stats=paired_regression_page(z,model='Direct ABC',baseline='Reference-median predictor')
    p[0].set_xlabel('Observed first-rebound nadir (visit)');p[0].set_ylabel('ABC-predicted nadir (visit)')
    p[1].set_xlabel('Absolute-error reduction (visits)');p[2].set_xlabel('Absolute prediction error (visits)')
    p[3].set_xlabel('Observed first-rebound nadir (visit)');p[3].set_ylabel('Prediction minus target (visits)')
    stats.update(all_source_rows=all_n,supported=n_support,evaluable=len(z))
    s.export('E02_exact_prediction_pairs',z)
    return [finish(s,p,'E02_direct_ABC_exact_timing',
      'Original direct SLiM/ABC transfer experiment. Four inputs: baseline and cycles 1-3. Target: first post-cycle-3 local low followed by a higher measurement. '
      'The plotted patients satisfy the SAVED prospective-support and observed-horizon flags; they are not the later q,d,g-supported or clinical-KNN cohort. '
      'Reference-median predictions are read from the saved original output, not estimated from the evaluation outcomes. Panel B: positive = smaller absolute error under ABC. '
      'Its 95% interval resamples patients and describes uncertainty for fixed saved predictions; it does not refit or retune ABC. Parameter expansion and the forecasting horizon were informed by VENICE, so this is model development/adequacy, not untouched external validation.',stats)]


@experiment('E03a','Early within-VENICE clinical forecasting')
def e03a(s):
    key='early_clinical';d=s.read(key)
    z=as_standard(s,key,d,{'target':['observed','target_t','observed_target_cycle','y_true'],
                          'prediction':['pred_ridge','ridge_prediction','ridge','predicted'],
                          'baseline':['median_baseline','pred_median','median_prediction','baseline','baseline_pred']})
    p,stats=paired_regression_page(z,model='Clinical Ridge',baseline='Training-fold median')
    s.export('E03a_clinical_prediction_pairs',z)
    return [finish(s,p,'E03a_clinical_only_exact_forecast','Early within-VENICE Ridge experiment on the original first-rebound target. Only saved held-out predictions are plotted. '+
       'This is not the later 211-patient KNN experiment. Per-patient repeated-CV means are used only when provided as such by the source; duplicate patient rows block the plot. '+
       'Error-reduction confidence intervals resample patients with predictions fixed.',stats)]


@experiment('E03b','GP correction of the original simulation predictor')
def e03b(s):
    key='gp';d=s.read(key);z=as_standard(s,key,d,{'target':['observed'],'prediction':['simulation_plus_gp'],'baseline':['simulation_abc']})
    p,stats=paired_regression_page(z,model='ABC + GP correction',baseline='Uncorrected ABC')
    p.title(1,'Does clinical correction improve ABC?',p.info[1].get_text());s.export('E03b_gp_prediction_pairs',z)
    return [finish(s,p,'E03b_ABC_GP_correction','GP discrepancy correction of the original supported/horizon-restricted ABC predictions. Source: saved repeated out-of-fold predictions averaged per patient. '
       'Positive absolute-error reduction favors GP correction. These results are not combined with the differently selected clinical Ridge cohort.',stats)]


def exp_cloud(s):
    d=s.read('exp_cloud');d['id']=pid(s,'exp_cloud',d)
    for role,aliases in {'q':['q_hat'],'d':['decline_rate_hat'],'g':['growth_rate_hat'],'rmse':['rmse_log10'],'fit_tmin':['tmin_fit_hat']}.items():
        c=s.column('exp_cloud',d,role,aliases);d[role]=pd.to_numeric(d[c],errors='coerce')
    unique(d,['id'],'Clinical exponential fits');return d


@experiment('E04','Biexponential curve description and retrospective nadir agreement')
def e04(s):
    cloud=exp_cloud(s);m,t,_=cohort(s);base=t[t.cycle.eq(0)].set_index('id').psa
    valid=cloud[cloud.id.isin(base.index)&np.isfinite(cloud.rmse)&cloud.q.gt(0)&cloud.q.lt(1)&cloud.d.gt(0)&cloud.g.gt(0)].copy()
    if len(valid)<4:raise Blocked('Too few fitted patients with the original PSA baseline for fit examples.')
    chosen=[]
    for q in [.1,.4,.7,.9]:
        vv=valid[~valid.id.isin(chosen)].assign(distance=lambda x:abs(x.rmse-valid.rmse.quantile(q))).sort_values(['distance','id']);chosen.append(vv.id.iloc[0])
    p=Page(4);audit=[]
    for j,id in enumerate(chosen):
        r=valid[valid.id.eq(id)].iloc[0];g=t[t.id.eq(id)].sort_values('cycle');ax=p[j];xx=g.cycle.to_numpy();yy=np.log10(g.psa.to_numpy()/base[id]);grid=np.linspace(xx.min(),xx.max(),500)
        # Stable analytic rendering; no refit and no parameter clipping.
        logfit=np.logaddexp(np.log1p(-r.q)-r.d*grid,np.log(r.q)+r.g*grid)/np.log(10)
        ax.plot(xx,yy,'o',label='Observed PSA');ax.plot(grid,logfit,label='Saved biexponential fit')
        ax.set_xlabel('Treatment cycle');ax.set_ylabel('log10(PSA / baseline PSA)')
        p.title(j,f'Fit-error quantile {int([.1,.4,.7,.9][j]*100)}%',f'Patient {id[-11:]}; log10 RMSE = {r.rmse:.3f}');p.key(j)
        audit.append({'id':id,'chosen_rmse_quantile':[.1,.4,.7,.9][j],'rmse':r.rmse,'q':r.q,'d':r.d,'g':r.g})
    s.export('E04_example_selection',pd.DataFrame(audit))
    out=[finish(s,p,'E04a_biexponential_fit_examples','Retrospective biexponential descriptions of real PSA. Examples are chosen nearest the 10th, 40th, 70th and 90th percentiles of saved log10 fit RMSE, not by forecasting success. '
       'Observed PSA is normalized by the original ADSL baseline at treatment cycle 0. Curves use the saved q,d,g without refitting. Full-trajectory information is allowed here. '
       'This is a model-description diagnostic, NOT a prospective prediction experiment.',{'example_ids':chosen})]
    d=s.read('exp_agreement');flag=s.column('exp_agreement',d,'valid',['valid_nonboundary']);d=d[boolean(d[flag])].copy()
    z=as_standard(s,'exp_agreement',d,{'target':['observed_nadir_cycle'],'prediction':['exp_tmin']});unique(z,['id'],'Retrospective nadir comparison')
    p=Page(4);stats=regression_metrics(z.target,z.prediction)
    ax=p[0];ax.scatter(z.target,z.prediction,s=17,alpha=.6,edgecolors='none');lims=[min(z.target.min(),z.prediction.min())-.5,max(z.target.max(),z.prediction.max())+.5];ax.plot(lims,lims,ls='--',alpha=.5);ax.set_xlim(lims);ax.set_ylim(lims)
    ax.set_xlabel('Observed global PSA nadir (cycle)');ax.set_ylabel('Retrospectively fitted nadir (cycle)')
    p.title(0,'Does the fitted nadir match the data?',f'n = {len(z)}; MAE = {stats["MAE"]:.2f}; rho = {stats["rho"]:.3f}')
    ax=p[1];ecdf(ax,abs(z.prediction-z.target),'Absolute nadir difference');ax.set_xlabel('Absolute discrepancy (cycles)');ax.set_ylabel('Fraction of nonboundary patients')
    p.title(1,'Distribution of nadir agreement',f'Within 1 cycle: {stats["within_1"]:.1%}; within 2: {stats["within_2"]:.1%}')
    ax=p[2];good=cloud.rmse[np.isfinite(cloud.rmse)];ax.hist(good,bins='auto');ax.set_xlabel('Saved log10 fit RMSE');ax.set_ylabel('Patients')
    p.title(2,'Curve-fit quality across the fitted cohort',f'All finite saved fits, n = {len(good)}; not the nadir-agreement subset')
    keys=['fit_tmin_in_observed_window','growth_near_lower_bound','q_near_lower_bound']
    if not set(keys)<=set(cloud):raise Blocked('Missing saved boundary flags needed for fit-identifiability diagnostics.')
    within=boolean(cloud[keys[0]]);boundary=boolean(cloud[keys[1]])|boolean(cloud[keys[2]])
    counts=pd.Series({'Inside observed window':int(within.sum()),'Outside + lower-bound fit':int((~within&boundary).sum()),'Outside, other fit':int((~within&~boundary).sum())})
    p.reserve_ylabels(3);ax=p[3];ax.barh(np.arange(3),counts.values);ax.set_yticks(np.arange(3));ax.set_yticklabels(['Inside window','Outside + boundary','Outside, other']);ax.set_xlabel('Patients');ax.invert_yaxis();p.title(3,'Turning-point identifiability flags')
    s.export('E04_nadir_agreement',z);s.export('E04_boundary_counts',counts.rename_axis('category').reset_index(name='n'))
    out.append(finish(s,p,'E04b_biexponential_nadir_diagnostics','A-B: saved valid_nonboundary cohort and its observed GLOBAL PSA minimum; no further outlier removal, winsorization or target substitution. '
       'C-D: all saved clinical exponential fits, a different denominator. The flagged pathological rule is outside-window fitted nadir AND growth or q at its lower bound. '
       'All panels are retrospective. Do not compare their error directly with an early-forecasting model on a different cohort.',stats))
    return out


@experiment('E05','Calibration relationships and matched clinical support')
def e05(s):
    d=s.read('sim_fits');scenario=s.cfg.get('calibration_scenario','broad_mix')
    if 'scenario' not in d:raise Blocked('Simulation fits lack scenario: cannot identify the calibrated library.')
    d=d[d.scenario.eq(scenario)].copy()
    if 'fit_success' in d:d=d[boolean(d.fit_success)]
    for c in ['d_hat','q_hat','g_hat','mean_standing_realized','m1_effect_mean_target','patient_turnover']:
        if c not in d:raise Blocked(f'Simulation calibration input missing {c}')
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d[np.isfinite(d[['d_hat','q_hat','g_hat','mean_standing_realized','m1_effect_mean_target','patient_turnover']]).all(axis=1)&d.q_hat.gt(0)&d.d_hat.gt(0)&d.g_hat.gt(0)]
    if len(d)<10:raise Blocked('Not enough successful calibrated fits.')
    p=Page(4)
    for j,(x,y,xlab,ylab,title) in enumerate([
       ('mean_standing_realized','q_hat','Realized standing resistant fraction','Fitted rebound weight q','Standing resistance and rebound weight'),
       ('m1_effect_mean_target','g_hat','Target mean resistance effect','Fitted growth rate g','Resistance effect and rebound growth'),
       ('patient_turnover','d_hat','Turnover','Fitted decline rate d','Turnover and fitted decline'),
       ('d_hat','g_hat','Fitted decline rate d','Fitted growth rate g','Resulting decline-rebound geometry')]):
        ax=p[j];ax.scatter(d[x],d[y],s=11,alpha=.35,edgecolors='none');ax.set_xlabel(xlab);ax.set_ylabel(ylab)
        if y=='q_hat':ax.set_yscale('log')
        p.title(j,title,f'n = {len(d)}; Spearman rho = {rhoval(d[x],d[y]):.2f}')
    outputs=[finish(s,p,'E05a_simulator_parameter_relationships','Input-to-fitted-parameter relationships within the saved '+scenario+' simulation scenario. '
       'These are multivariable calibration associations, NOT controlled one-parameter causal sweeps. Other settings vary. Only fit_success rows with finite positive fitted q,d,g are displayed. '+
       'This figure should not be described as proving a unique mapping from each biological parameter to d,q,g.',{'scenario':scenario,'n':len(d)})]
    m=s.read('matches');req=['clinical_d','clinical_q','clinical_g','support_class']
    if not set(req)<=set(m):raise Blocked('Matched support table missing '+str(req))
    for c in req[:3]:m[c]=numeric(m,c)
    if not m.clinical_q.gt(0).all():raise Blocked('Nonpositive clinical q in matched cohort.')
    p=Page(4)
    for j,(cx,sx,clab,cy,sy,ylab,title) in enumerate([
      ('clinical_d','d_hat','Decline rate d','clinical_g','g_hat','Growth rate g','Decline and growth distributions'),
      ('clinical_q','q_hat','log10(rebound weight q)','clinical_g','g_hat','Growth rate g','Rebound weight and growth'),
      ('clinical_d','d_hat','Decline rate d','clinical_q','q_hat','log10(rebound weight q)','Decline and rebound weight')]):
        xx=np.log10(d[sx]) if sx=='q_hat' else d[sx];yy=np.log10(d[sy]) if sy=='q_hat' else d[sy]
        cxv=np.log10(m[cx]) if cx=='clinical_q' else m[cx];cyv=np.log10(m[cy]) if cy=='clinical_q' else m[cy]
        ax=p[j];ax.scatter(xx,yy,s=10,alpha=.2,edgecolors='none',label='Calibrated simulations');ax.scatter(cxv,cyv,s=15,alpha=.65,edgecolors='none',label='Matched clinical cohort')
        ax.set_xlabel(clab);ax.set_ylabel(ylab);p.title(j,title);p.key(j)
    order=['well_supported','edge_of_support','outside_support'];unknown=set(m.support_class)-set(order)
    if unknown:raise Blocked('Unexpected support categories '+str(unknown))
    counts=m.support_class.value_counts().reindex(order,fill_value=0);p.reserve_ylabels(3);ax=p[3];ax.barh(np.arange(3),counts.values);ax.set_yticks(np.arange(3));ax.set_yticklabels(['Well supported','Edge','Outside']);ax.invert_yaxis();ax.set_xlabel('Patients')
    p.title(3,'Support in this specific calibrated cloud',f'n = {len(m)}; well + edge = {(counts.iloc[:2].sum()/len(m)):.1%}')
    s.export('E05_matched_clinical_parameters',m);s.export('E05_simulation_cloud',d)
    outputs.append(finish(s,p,'E05b_clinical_simulation_support','The clinical points come directly from the SAME matched table as the support counts. The simulation points are successful '+scenario+' fits from the associated heterogeneous library. '+
       'This is full-trajectory, retrospective q,d,g support. It is NOT early-input support and is not the later K=1 ABC reference cohort. The configured calibration description is: '+s.cfg.get('calibration_description','not specified')+'.',{'support_counts':counts.to_dict(),'simulation_n':len(d)}))
    return outputs


def scan_predictions(s):
    key='scan_predictions';d=s.read(key)
    subset=s.cfg['sources'][key].get('filters',{})
    for k,v in subset.items():
        if k not in d:raise Blocked(f'{key}: missing filter column {k}')
        d=d[d[k].eq(v)].copy()
    z=as_standard(s,key,d,{'scan':['scan_count','n_scans'],'target':['global_fit_tmin','true_tmin','target_tmin'],'prediction':['pred_tmin_weighted','pred_tmin']})
    # Membership is taken from a saved fixed-cohort ID set, not reselected by outcome.
    if s.cfg.get('fixed_scan_ids_from') == 'binary':
        b=s.read('binary');bi=pid(s,'binary',b);bs=get(s,'binary',b,'scan',['scan_count']);bt=get(s,'binary',b,'target',['true_tmin'])
        frozen=pd.DataFrame({'id':bi,'scan':bs,'target':bt})
        if frozen.groupby('scan').id.apply(set).map(lambda x:x!=set(frozen.id)).any():
            raise Blocked('The saved binary-cohort IDs are not fixed across scan counts.')
        truth=frozen.groupby('id').target.first();z=z[z.id.isin(truth.index)].copy()
        if not np.allclose(z.target, z.id.map(truth)):
            raise Blocked('Scan-study fitted targets differ from the shared saved fixed-cohort targets.')
    unique(z,['id','scan'],'Fixed-cohort scan predictions')
    sets=[set(g.id) for _,g in z.groupby('scan')]
    if not sets or any(g!=sets[0] for g in sets):raise Blocked('Scan budgets have different patient IDs; bind the frozen fixed-common cohort, not changing eligible subsets.')
    n_expected=s.cfg.get('fixed_scan_expected_n',133)
    if n_expected and len(sets[0])!=n_expected:raise Blocked(f'Expected fixed scan cohort n={n_expected}; found {len(sets[0])}.')
    if z.groupby('id').target.nunique().max()!=1:raise Blocked('Target changes across scan budgets.')
    if not (z.target>z.scan.max()-1).all():raise Blocked('Fixed scan cohort contains fitted nadirs at/before the last input cycle.')
    return z


@experiment('E06','Calibrated K=1 exact ABC versus early observation count')
def e06(s):
    z=scan_predictions(s);summary=s.read('scan_summary')
    sc=s.column('scan_summary',summary,'scan',['scan_count']);mc=s.column('scan_summary',summary,'mae',['MAE'])
    if summary[sc].duplicated().any():raise Blocked('Fixed-common summary has multiple rows per scan count.')
    rows=[]
    for k,g in z.groupby('scan'):
        met=regression_metrics(g.target,g.prediction);met['scan']=k;rows.append(met)
        r=summary[summary[sc].eq(k)]
        if len(r)!=1 or not np.isclose(float(r[mc].iloc[0]),met['MAE'],atol=.005):raise Blocked('Per-patient scan predictions disagree with frozen fixed-common summary.')
    a=pd.DataFrame(rows);p=Page(4);ax=p[0];ax.plot(a.scan,a.MAE,'o-',label='Clinical ABC MAE')
    bc=s.column('scan_summary',summary,'baseline_mae',['reference_median6_MAE','naive_MAE','baseline_MAE','reference_median_MAE'],True)
    if bc:ax.plot(summary[sc],summary[bc],'s--',label='Saved constant-predictor MAE')
    cv=s.read('scan_reference_cv')
    if not {'scan_count','subset','reference_cv_MAE'}<=set(cv):raise Blocked('Missing registered reference-CV summary fields.')
    cv=cv[cv['subset'].eq('well_supported_future')].sort_values('scan_count')
    unique(cv,['scan_count'],'Reference-CV comparison')
    if set(cv.scan_count)!=set(a.scan):raise Blocked('Reference CV and clinical study have different scan budgets.')
    ax.plot(cv.scan_count,cv.reference_cv_MAE,'^--',label='Within-reference CV MAE')
    ax.set_xlabel('Number of early PSA measurements');ax.set_ylabel('Mean absolute error (cycles)');p.title(0,'Does more early PSA improve exact timing?',f'Identical {z.id.nunique()} patients at every budget');p.key(0)
    ax=p[1];ax.plot(a.scan,a.rho,'o-');ax.axhline(0,ls='--',alpha=.5);ax.set_xlabel('Number of early PSA measurements');ax.set_ylabel('Spearman correlation');p.title(1,'Patient-specific rank information')
    last=z[z.scan.eq(z.scan.max())];ax=p[2];met=regression_metrics(last.target,last.prediction)
    from plot_core import scatter_prediction
    scatter_prediction(ax,last.target,last.prediction);ax.set_xlabel('Full-trajectory fitted nadir (cycle)');p.title(2,'Final-budget exact predictions',f'{int(z.scan.max())} measurements; MAE = {met["MAE"]:.2f}; rho = {met["rho"]:.2f}')
    first=4 if 4 in set(z.scan) else int(z.scan.min());wide=z.pivot(index='id',columns='scan',values='prediction');truth=z.groupby('id').target.first();delta=abs(wide[first]-truth)-abs(wide[z.scan.max()]-truth)
    ax=p[3];ax.hist(delta,bins='auto');ax.axvline(0,ls='--',alpha=.5);ax.set_xlabel('Absolute-error reduction (cycles)');ax.set_ylabel('Patients');p.title(3,f'Paired change: {first} to {int(z.scan.max())} inputs','Positive = improvement with the larger input budget')
    s.export('E06_fixed_patient_predictions',z);s.export('E06_recomputed_metrics',a)
    return [finish(s,p,'E06_calibrated_ABC_scan_budget','K=1 calibrated ABC experiment predicting the full-trajectory FITTED biexponential nadir. The saved fixed-common cohort, not the changing eligible cohort, is used at every budget. '+
       'The final fitted nadir is later than the entire early input window for all displayed patients. Panel D is paired by patient ID. If the saved summary does not contain the frozen baseline MAE, panel A shows ABC only; no outcome-derived baseline is fabricated. '+
       'Retrospective clinical support/calibration means this is exploratory transportability analysis.',{'per_scan':a.to_dict('records')})]


@experiment('E07','Binary early/late ABC on the fixed 133-patient cohort')
def e07(s):
    key='binary';d=s.read(key);z=as_standard(s,key,d,{'scan':['scan_count'],'target':['true_tmin'],'y':['true_late'],'p':['prob_late'],'threshold':['probability_threshold']})
    unique(z,['id','scan'],'Binary ABC');sets=[set(g.id) for _,g in z.groupby('scan')]
    if any(x!=sets[0] for x in sets):raise Blocked('Binary cohorts differ across scan budgets.')
    if not np.array_equal(z.y.to_numpy(),(z.target>8).astype(float).to_numpy()):raise Blocked('Binary label is not fitted t_min > 8.')
    if not (z.target>z.scan.max()-1).all():raise Blocked('Binary fitted target is not prospective at the final scan budget.')
    rows=[]
    for k,g in z.groupby('scan'):
        y=g.y.to_numpy(int);pp=g.p.to_numpy();pred=pp>=g.threshold.to_numpy()
        from plot_core import verify_prob
        verify_prob(y,pp);rows.append({'scan':k,'n':len(g),'AUROC':roc_auc_score(y,pp),'early_recall':np.mean(~pred[y==0]),'late_recall':np.mean(pred[y==1])})
    a=pd.DataFrame(rows);p=Page(4);ax=p[0];ax.plot(a.scan,a.AUROC,'o-');ax.axhline(.5,ls='--',alpha=.5);ax.set_ylim(.45,1);ax.set_xlabel('Early PSA measurements');ax.set_ylabel('AUROC');p.title(0,'Coarse timing information by input count',f'Fixed n = {z.id.nunique()}; early: 5 < fitted nadir <= 8')
    ax=p[1];ax.plot(a.scan,a.early_recall,'o-',label='Early recall');ax.plot(a.scan,a.late_recall,'s-',label='Late recall');ax.set_ylim(0,1);ax.set_xlabel('Early PSA measurements');ax.set_ylabel('Recall');p.title(1,'Recall at simulation-selected thresholds');p.key(1)
    g=z[z.scan.eq(z.scan.max())];roc(p[2],g.y,g.p,'Weighted ABC');p[2].plot([0,1],[0,1],ls='--',alpha=.5);p.title(2,'Six-measurement discrimination');p.key(2)
    cal=calibration(p[3],g.y,g.p,'Weighted ABC');p[3].plot([0,1],[0,1],ls='--',alpha=.5);p.title(3,'Calibration of late-nadir probabilities')
    s.export('E07_binary_rows',z);s.export('E07_metrics_by_scan',a);s.export('E07_calibration',cal)
    return [finish(s,p,'E07_binary_ABC_early_late','Weighted binary ABC, not a direct clinical logistic classifier. Saved inverse-distance probabilities and simulation-selected probability thresholds are used without retuning on VENICE. '+
       'Positive label: fitted global nadir > 8; early class: fitted nadir > 5 and <= 8. The same patient IDs occur at all scan counts. The cutoff/support selections were exploratory. '+
       'Calibration bins are descriptive, with their counts exported; they are not an additional trained calibration model.',{'per_scan':a.to_dict('records')})]


@experiment('E08','Empirical-Bayes posterior forecast from early PSA')
def e08(s):
    key='eb';d=s.read(key);z=as_standard(s,key,d,{'scan':['scan_count'],'target':['exp_tmin'],'observed':['observed_nadir_cycle'],
      'prediction':['pred_tmin_median'],'lower':['posterior_q025'],'upper':['posterior_q975']})
    unique(z,['id','scan'],'Empirical Bayes');sets=[set(g.id) for _,g in z.groupby('scan')]
    if any(v!=sets[0] for v in sets):raise Blocked('Empirical-Bayes cohorts change across scan counts.')
    if (z.lower>z.upper).any():raise Blocked('Reversed posterior interval bounds.')
    rows=[]
    for k,g in z.groupby('scan'):
        for target in ['target','observed']:
            met=regression_metrics(g[target],g.prediction);met.update(scan=k,target=target,median_width=float((g.upper-g.lower).median()),coverage=float(((g[target]>=g.lower)&(g[target]<=g.upper)).mean()));rows.append(met)
    a=pd.DataFrame(rows);p=Page(4)
    for j,(metric,ylabel,title) in enumerate([('MAE','Mean absolute error (cycles)','Point prediction under the two targets'),('rho','Spearman correlation','Patient-specific information'),('coverage','Observed coverage of 95% interval','Interval coverage by target')]):
        ax=p[[0,1,3][j]]
        for target,label in [('target','Fitted biexponential nadir'),('observed','Observed global PSA nadir')]:
            g=a[a.target.eq(target)];ax.plot(g.scan,g[metric],'o-',label=label)
        ax.set_xlabel('Number of early PSA measurements');ax.set_ylabel(ylabel);p.title([0,1,3][j],title);p.key([0,1,3][j])
        if metric=='coverage':ax.axhline(.95,ls='--',alpha=.4);ax.set_ylim(0,1.02)
    ax=p[2];g=a[a.target.eq('target')];ax.plot(g.scan,g.median_width,'o-');ax.set_xlabel('Number of early PSA measurements');ax.set_ylabel('Median posterior 95% width (cycles)');p.title(2,'Does the posterior concentrate?')
    s.export('E08_posterior_predictions',z);s.export('E08_posterior_metrics',a)
    return [finish(s,p,'E08_empirical_Bayes_forecasting','Cross-validated empirical-prior biexponential forecasting, using the saved posterior MEDIAN as point estimate. Priors were learned from other patients, and early PSA updated the held-out patient posterior. '+
       'The fitted exponential nadir and observed global PSA nadir are different targets and are kept separate throughout. A one-measurement point, when saved, is the baseline-only/prior condition. '+
       'Coverage is evaluated separately for each target; posterior uncertainty about the latent nadir is not the same as an observation-level prediction interval for noisy PSA.',{'per_scan':a.to_dict('records')})]


@experiment('E08b','Joint versus shuffled q-g prior')
def e08b(s):
    d=s.read('qg_summary');req=['scan_count','mode','n','MAE_exp_tmin','Spearman_exp_tmin','AUROC_tmin_gt8']
    if not set(req)<=set(d):raise Blocked('qg summary schema needs '+str(req))
    if set(d['mode'])!={'joint','qg_shuffled'}:raise Blocked('Unexpected prior modes; do not mix shrinkage and shuffled-prior experiments.')
    if d.groupby('scan_count').n.nunique().max()!=1:raise Blocked('Joint/shuffled summaries have unequal denominators.')
    p=Page(4)
    for j,(metric,ylab,title) in enumerate([('MAE_exp_tmin','MAE (cycles)','Effect on exact fitted-nadir timing'),('Spearman_exp_tmin','Spearman correlation','Effect on patient ranking'),('AUROC_tmin_gt8','AUROC','Effect on coarse timing')]):
        for mode,label in [('joint','Joint empirical prior'),('qg_shuffled','q-g shuffled prior')]:
            g=d[d['mode'].eq(mode)].sort_values('scan_count');p[j].plot(g.scan_count,g[metric],'o-',label=label)
        p[j].set_xlabel('Early PSA measurements');p[j].set_ylabel(ylab);p.title(j,title);p.key(j)
    a=d.pivot(index='scan_count',columns='mode',values='MAE_exp_tmin');delta=a.qg_shuffled-a.joint
    p[3].plot(delta.index,delta,'o-');p[3].axhline(0,ls='--',alpha=.5);p[3].set_xlabel('Early PSA measurements');p[3].set_ylabel('Shuffled minus joint MAE (cycles)');p.title(3,'Benefit of retaining joint structure','Positive = lower error under joint prior')
    s.export('E08b_joint_prior_summary',d)
    return [finish(s,p,'E08b_joint_qg_prior_ablation','Saved q-g dependence ablation on the same reported fixed cohort and early likelihood. This plots aggregate results only; no patient-level confidence interval is invented from summaries. '+
      'Fitted q-g correlation can include a parameter tradeoff of the curve fit; these results do not by themselves establish a biological q-g relationship.',{'source_rows':len(d)})]


@experiment('E09a','Clinical covariate addition to six-scan logistic classification')
def e09a(s):
    key='covariates';d=s.read(key)
    f=s.column(key,d,'feature_set',['feature_set','model']);auc=s.column(key,d,'auc',['AUROC']);brier=s.column(key,d,'brier',['Brier']);bal=s.column(key,d,'balanced_accuracy',['balanced_accuracy_05','balanced_accuracy','balacc'])
    labels=['relative_psa','baseline_only','relative_psa_plus_full_baseline'];names=['PSA trajectory','Baseline only','PSA + baseline']
    if d[f].duplicated().any():raise Blocked('Covariate summary contains duplicate feature sets; specify exact experiment.')
    d=d.set_index(f).reindex(labels)
    if d[[auc,brier,bal]].isna().any().any():raise Blocked('Missing one of the three exact clinical feature-set rows.')
    p=Page(4)
    for j,(metric,ylab,title) in enumerate([(auc,'AUROC','Discrimination'),(bal,'Balanced accuracy','Classification at the saved threshold'),(brier,'Brier score (lower is better)','Probability error')]):
        p.reserve_ylabels(j);p[j].plot(d[metric],range(3),'o',markersize=6);p[j].set_yticks(range(3));p[j].set_yticklabels(names);p[j].invert_yaxis();p[j].set_xlabel(ylab);p.title(j,title)
    delta=d[auc].iloc[1:].to_numpy()-d[auc].iloc[0];p.reserve_ylabels(3);p[3].barh(range(2),delta);p[3].axvline(0,ls='--',alpha=.5);p[3].set_yticks(range(2));p[3].set_yticklabels(names[1:]);p[3].invert_yaxis();p[3].set_xlabel('AUROC minus PSA-only AUROC');p.title(3,'Incremental value relative to PSA')
    s.export('E09a_covariate_summary',d.reset_index())
    return [finish(s,p,'E09a_clinical_covariate_test','Direct clinical six-scan classifier feature ablation. These are not weighted binary ABC results and must not be presented as the same 133-patient analysis. '+
      'The baseline-only panel and combined panel use the exact saved feature-set labels. Summary-only plots do not include invented uncertainty. Lower performance after adding features applies to these data and training procedure, not all possible biomarkers.',{'rows':d.reset_index().to_dict('records')})]


@experiment('E09b','Training-size sensitivity of the direct clinical classifier')
def e09b(s):
    d=s.read('learning');req=['mean_train_n_per_fold','AUROC_mean','AUROC_q025','AUROC_q975','train_fraction','repeats']
    if not set(req)<=set(d):raise Blocked('Learning-curve summary missing '+str(req))
    d=d.sort_values('mean_train_n_per_fold');p=Page(2);ax=p[0]
    ax.plot(d.mean_train_n_per_fold,d.AUROC_mean,'o-');ax.fill_between(d.mean_train_n_per_fold,d.AUROC_q025,d.AUROC_q975,alpha=.14)
    ax.set_xlabel('Mean training patients per fold');ax.set_ylabel('Held-out AUROC');p.title(0,'Effect of clinical training-set size','Shading: 2.5th-97.5th percentiles across subsampling runs')
    ax=p[1];ax.plot(d.train_fraction,d.repeats,'o-');ax.set_xlabel('Fraction of the outer training set');ax.set_ylabel('Number of subsampling runs');p.title(1,'Replication behind each curve point','A single full-size run does not estimate sampling uncertainty')
    s.export('E09b_learning_curve',d)
    return [finish(s,p,'E09b_clinical_training_size','Training-size experiment for the direct clinical classifier only. Shaded intervals are variation across the saved repeated training subsets, not a patient-bootstrap confidence interval or evidence about unlimited future sample sizes. '+
      'The number of runs at every fraction is displayed explicitly. No extrapolation beyond the tested range is shown.',{'n_points':len(d)})]


def sequential(s,key,pm1=False):
    d=s.read(key);fields={'risk_index':['risk_index'],'event_index':['event_index'],
       'p':['p_physics_reinforced'],'base':['p_time_only'],'fold':['outer_fold'],'risk_day':['risk_calendar_day']}
    z=pd.DataFrame({'id':pid(s,key,d)})
    for role,aliases in fields.items():
        c=s.column(key,d,role,aliases);z[role]=pd.to_numeric(d[c],errors='coerce')
    yc=s.column(key,d,'y',['y','y_pm1']);z['y']=pd.to_numeric(d[yc],errors='coerce')
    if not pm1:z['horizon']=get(s,key,d,'horizon',['horizon_scans'])
    else:z['horizon']=0
    z=z[z.y.notna()].copy()
    for c in ['risk_index','p','base','fold','risk_day','horizon','y']:
        if not np.isfinite(z[c]).all():raise Blocked(f'{key}: missing {c} in resolved prediction rows.')
    unique(z,['id','risk_index','horizon'],'Sequential predictions')
    if z.groupby('id').fold.nunique().max()!=1:raise Blocked('A patient occurs in more than one outer fold.')
    if z.groupby('id').event_index.nunique().max()>1:raise Blocked('Inconsistent confirmed local nadir index within patient.')
    z['relative_scan']=z.risk_index-z.event_index
    flag=z.event_index.notna()
    expected=(z.relative_scan.abs()<=1) if pm1 else ((z.event_index-z.risk_index).ge(1)&(z.event_index-z.risk_index).le(z.horizon))
    if not np.array_equal(z.loc[flag,'y'],expected[flag].astype(float)):raise Blocked('Saved event labels disagree with the registered local-event/scan-index definition.')
    if not pm1 and (z.loc[flag,'risk_index']>=z.loc[flag,'event_index']).any():raise Blocked('Strict future warning includes at/post-event predictions.')
    for _,g in z.groupby('horizon'):
        from plot_core import verify_prob
        verify_prob(g.y,g.p);verify_prob(g.y,g.base)
    return z


def auc_bootstrap(z,B=600):
    ids=z.id.unique();positions={id:np.flatnonzero(z.id.to_numpy()==id) for id in ids};rng=np.random.default_rng(391)
    y=z.y.to_numpy();p=z.p.to_numpy();b=z.base.to_numpy();vals=[]
    for _ in range(B):
        ii=np.concatenate([positions[id] for id in rng.choice(ids,len(ids),replace=True)])
        if len(np.unique(y[ii]))==2:vals.append(roc_auc_score(y[ii],p[ii])-roc_auc_score(y[ii],b[ii]))
    return list(np.quantile(vals,[.025,.975]))




@experiment('E11','Immediate nadir-region classification and held-out examples')
def e11(s):
    z=sequential(s,'pm1',True);p=Page(4);a=roc(p[0],z.y,z.p,'Latent-state');b=roc(p[0],z.y,z.base,'Time only');p[0].plot([0,1],[0,1],ls='--',alpha=.5)
    ci=auc_bootstrap(z);p.title(0,'Within one scan of the confirmed nadir',f'{z.id.nunique()} patients; {len(z)} decisions; delta AUC = {a-b:.3f}');p.key(0)
    cal=[]
    for c,label in [('p','Latent-state'),('base','Time only')]:
        q=calibration(p[1],z.y,z[c],label);q['model']=label;cal.append(q)
    p[1].plot([0,1],[0,1],ls='--',alpha=.5);p.title(1,'Probability calibration');p.key(1)
    # Event-aligned descriptive display is restricted to patients with an event.
    aligned=z[z.event_index.notna()&z.relative_scan.between(-8,1)].copy()
    ax=p[2]
    for c,label in [('p','Latent-state'),('base','Time only')]:
        q=group_curve(aligned,'relative_scan',c);line=ax.plot(q.relative_scan,q['mean'],'o-',label=label)[0];ax.fill_between(q.relative_scan,q.lo,q.hi,alpha=.12,color=line.get_color())
        s.export('E11_event_alignment_'+c,q)
    ax.axvspan(-1,1,alpha=.07);ax.axvline(0,ls='--',alpha=.5);ax.set_xlabel('Current scan minus confirmed nadir scan');ax.set_ylabel('Mean out-of-fold probability');p.title(2,'Risk near the event, aligned retrospectively','Event patients only; intervals resample patients');p.key(2)
    ax=p[3];fd=[]
    for fold,g in z.groupby('fold'):
        if g.y.nunique()<2:continue
        fd.append({'fold':int(fold),'latent':roc_auc_score(g.y,g.p),'time':roc_auc_score(g.y,g.base),'n':len(g)})
    fd=pd.DataFrame(fd)
    ax.plot(fd.fold,fd.latent,'o-',label='Latent-state');ax.plot(fd.fold,fd.time,'s-',label='Time only');ax.axhline(.5,ls='--',alpha=.4);ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.set_xlabel('Outer patient CV fold');ax.set_ylabel('AUROC');p.title(3,'Consistency across held-out patient folds');p.key(3)
    s.export('E11_pm1_predictions',z);s.export('E11_calibration',pd.concat(cal));s.export('E11_fold_metrics',fd)
    out=[finish(s,p,'E11a_local_nadir_window','Exploratory +/-1-SCAN region classification: one scan before, at, or one scan after the first local nadir confirmed by two subsequent higher PSA values. '
      'This includes contemporaneous and post-nadir detection and is NOT equivalent to predicting a still-future event. No claim of an optimal clinical switching time is made. '
      'The point estimate is row-weighted held-out AUROC, with patients as bootstrap clusters. Event-aligned curves use future event times solely for retrospective display; event definitions must not enter prediction features. '+
      'Window width was explored, so internal CV uncertainty does not cover window-selection optimism.',{'n_patients':z.id.nunique(),'n_rows':len(z),'AUROC':a,'time_AUROC':b,'delta_CI95':ci})]
    # True sequential-example mapping. Never use the older first-rebound CSV.
    raw=s.read('audit_psa');ic=s.column('audit_psa',raw,'id',['id']);tc=s.column('audit_psa',raw,'calendar_day',['calendar_day']);cc=s.column('audit_psa',raw,'cycle',['cycle']);pc=s.column('audit_psa',raw,'psa',['psa','PSA'])
    raw=raw.rename(columns={ic:'id',tc:'calendar_day',cc:'cycle',pc:'psa'});raw.id=raw.id.astype(str)
    for c in ['calendar_day','cycle','psa']:raw[c]=pd.to_numeric(raw[c],errors='coerce')
    ids=aligned.groupby('id').agg(n=('risk_index','size'),event=('event_index','first'));ids=ids[ids.n.ge(5)].sort_values(['event'])
    examples=[]
    for frac in [.3,.7]:
        candidates=ids.assign(dist=abs(ids.event-ids.event.quantile(frac))).sort_values(['dist','event'])
        for id in candidates.index:
            if id not in examples and id in set(raw.id):examples.append(id);break
    if len(examples)<2:raise Blocked('Fewer than two confirmed-event patients with >=5 OOF risk updates and raw audit PSA.')
    p=Page(4);selections=[]
    for j,id in enumerate(examples):
        g=raw[raw.id.eq(id)].sort_values(['calendar_day','cycle'],kind='stable').reset_index(drop=True);g['scan_index']=np.arange(len(g));zz=z[z.id.eq(id)].sort_values('risk_index')
        if g[['calendar_day','cycle','psa']].isna().any().any() or not g.psa.gt(0).all():raise Blocked('Raw sequential example contains missing/nonpositive PSA.')
        ri=zz.risk_index.to_numpy(int);ei=int(zz.event_index.dropna().iloc[0])
        if ri.max()>=len(g) or ei+2>=len(g):raise Blocked('Sequential event/risk indices cannot be mapped to the raw audit records.')
        if not np.allclose(g.loc[ri,'calendar_day'],zz.risk_day):raise Blocked('Raw record ordering disagrees with saved risk_calendar_day. Do not mark the event using another time origin.')
        if not (g.psa.iloc[ei+1:ei+3]>g.psa.iloc[ei]).all():raise Blocked('Example event lacks two higher subsequent PSA observations in the audit trajectory.')
        for k in [j*2,j*2+1]:
            p[k].axvspan(ei-1,ei+1,alpha=.08);p[k].axvline(ei,ls='--',alpha=.5)
        ax=p[j*2];ax.plot(g.scan_index,np.log10(g.psa),'o-',label='Observed PSA');ax.scatter([ei],[np.log10(g.psa.iloc[ei])],marker='*',s=75)
        ax.set_xlabel('PSA measurement index');ax.set_ylabel('log10(PSA)');p.title(j*2,f'Held-out patient {id[-11:]}','Star: saved confirmed local nadir; shade: +/-1 scan')
        ax=p[j*2+1];ax.plot(zz.risk_index,zz.p,'o-',label='Latent-state');ax.plot(zz.risk_index,zz.base,'s-',label='Time only');ax.set_xlabel('PSA measurement index');ax.set_ylabel('Out-of-fold window probability');ax.set_ylim(0,1)
        p.title(j*2+1,'Probabilities available at each risk visit');p.key(j*2+1)
        selections.append({'id':id,'event_index':ei,'fold':int(zz.fold.iloc[0]),'n_updates':len(zz)})
    s.export('E11_example_selection',pd.DataFrame(selections))
    out.append(finish(s,p,'E11b_heldout_window_examples','Held-out patients selected near the 30th and 70th percentiles of event index among patients with at least five risk updates, without selecting on model success. '+
      'The raw PSA order is checked against saved risk_calendar_day, and the plotted star comes from the SAME event_index as the window-model output. '+
      'Future observations are shown only to reveal the retrospectively confirmed event, never as predictor input. The horizontal scale is PSA measurement index, not a fixed-duration treatment cycle.',{'examples':selections}))
    return out


@experiment('E03c','Cross-trial exact-timing transfer with source-trial baselines')
def e03c(s):
    p=Page(4);stats={}
    for j,(key,title) in enumerate([('vm_reg','VENICE to MAINSAIL'),('mv_reg','MAINSAIL to VENICE')]):
        d=s.read(key);z=as_standard(s,key,d,{'target':['target_t'],'prediction':['prediction'],'baseline':['source_median']});unique(z,['id'],'Cross-trial regression')
        from plot_core import scatter_prediction
        met=scatter_prediction(p[j*2],z.target,z.prediction);p.title(j*2,title,f'n = {len(z)}; MAE = {met["MAE"]:.2f}; rho = {met["rho"]:.2f}')
        ax=p[j*2+1];ecdf(ax,abs(z.prediction-z.target),'Transferred Ridge');ecdf(ax,abs(z.baseline-z.target),'Source-trial median')
        ax.set_xlabel('Absolute prediction error (cycles)');ax.set_ylabel('Fraction of held-out patients');ax.set_ylim(0,1.02);p.title(j*2+1,'Does transfer improve on source timing?');p.key(j*2+1)
        stats[title]={'model':met,'source_baseline':regression_metrics(z.target,z.baseline)};s.export('E03c_'+key,z)
    return [finish(s,p,'E03c_exact_cross_trial_transfer','Earlier first-rebound-nadir forecasting experiment, not the later global-nadir or sequential target. The source-trial median is read from each saved cross-trial prediction CSV, not taken from a within-MAINSAIL CV summary. '
      'Models are trained in one trial and evaluated in the other; no numerical fallback from the chat is used. IDs and paired baseline predictions are preserved.',stats)]


@experiment('E03d','Earlier cutoff-7 classification and cross-trial transfer')
def e03d(s):
    p=Page(4);stats={};results=[]
    for j,(key,title) in enumerate([('main_bin','MAINSAIL held-out prediction'),('vm_bin','VENICE to MAINSAIL'),('mv_bin','MAINSAIL to VENICE')]):
        d=s.read(key);z=as_standard(s,key,d,{'target':['target_t'],'y':['late'],'p':['p_late']});unique(z,['id'],'Earlier cutoff-7 classifier')
        if not np.array_equal(z.y.to_numpy(),(z.target>=7).astype(float).to_numpy()):raise Blocked('Earlier cross-trial binary label is not first-rebound target >= 7.')
        a=roc(p[j],z.y,z.p,'Clinical classifier');p[j].plot([0,1],[0,1],ls='--',alpha=.5);p.title(j,title,f'n = {len(z)}; late prevalence = {z.y.mean():.1%}');p.key(j)
        stats[title]={'n':len(z),'AUROC':a,'prevalence':z.y.mean()};q=calibration(p[3],z.y,z.p,title);q['analysis']=title;results.append(q);s.export('E03d_'+key,z)
    p[3].plot([0,1],[0,1],ls='--',alpha=.5);p.title(3,'Calibration across validation settings');p.key(3)
    s.export('E03d_calibration',pd.concat(results,ignore_index=True))
    return [finish(s,p,'E03d_binary_cross_trial_transfer','Earlier clinical cutoff-7 task: first-rebound nadir < 7 versus >= 7. MAINSAIL internal predictions are saved repeated-CV patient means; the two transfer curves use models trained on the other trial. '
      'This is NOT the later cutoff-8 weighted ABC or +/-1-scan state-classification task. Calibration curves are descriptive quantile bins with bin sizes exported.',stats)]


@experiment('E07b','Weighted versus rejection ABC at the same fitted-nadir cutoff')
def e07b(s):
    d=s.read('rejection');req=['scan_count','n','AUROC','balanced_accuracy','early_recall','late_recall','weighted_ABC_AUROC','weighted_ABC_balanced_accuracy','weighted_ABC_early_recall','weighted_ABC_late_recall']
    if not set(req)<=set(d):raise Blocked('Rejection summary lacks matched weighted-ABC comparison columns: '+str(sorted(set(req)-set(d))))
    unique(d,['scan_count'],'Rejection summary');d=d.sort_values('scan_count');p=Page(4)
    for j,(c,ylabel,title) in enumerate([('AUROC','AUROC','Overall discrimination'),('balanced_accuracy','Balanced accuracy','Balanced classification performance'),('early_recall','Early-nadir recall','Recall of early nadirs'),('late_recall','Late-nadir recall','Recall of late nadirs')]):
        ax=p[j];ax.plot(d.scan_count,d[c],'s-',label='Rejection ABC');ax.plot(d.scan_count,d['weighted_ABC_'+c],'o-',label='Weighted ABC');ax.set_xlabel('Early PSA measurements');ax.set_ylabel(ylabel);ax.set_ylim(0,1)
        p.title(j,title);p.key(j)
    s.export('E07b_abc_weighting_summary',d)
    return [finish(s,p,'E07b_ABC_weighting_control','Two saved ABC weighting schemes on the same fixed cutoff-8 clinical evaluation. Decision thresholds came from the respective reference-validation procedures. '
      'Changes in early/late recall are shown separately from aggregate discrimination; no current clinical threshold re-optimization is performed.',{'source_rows':len(d)})]


@experiment('E06b','XGBoost versus calibrated ABC on the same clinical scan budgets')
def e06b(s):
    d=s.read('xgb');req=['scan_count','MAE','reference_cv_MAE','ABC_MAE','Spearman_rho','ABC_Spearman_rho']
    if not set(req)<=set(d):raise Blocked('XGBoost summary missing same-cohort comparison columns '+str(sorted(set(req)-set(d))))
    unique(d,['scan_count'],'XGBoost comparison');d=d.sort_values('scan_count');p=Page(4)
    for j,(cols,labels,ylabel,title) in enumerate([
       (['MAE','ABC_MAE'],['XGBoost','ABC'],'Clinical MAE (cycles)','Patient-level transfer error'),
       (['Spearman_rho','ABC_Spearman_rho'],['XGBoost','ABC'],'Clinical Spearman correlation','Patient-specific rank information'),
       (['reference_cv_MAE','MAE'],['Reference CV','VENICE'],'XGBoost MAE (cycles)','Within-simulation versus clinical error')]):
        for c,label in zip(cols,labels):p[j].plot(d.scan_count,d[c],'o-',label=label)
        p[j].set_xlabel('Early PSA measurements');p[j].set_ylabel(ylabel);p.title(j,title);p.key(j)
    p[3].plot(d.scan_count,d.MAE-d.ABC_MAE,'o-');p[3].axhline(0,ls='--',alpha=.5);p[3].set_xlabel('Early PSA measurements');p[3].set_ylabel('XGBoost minus ABC MAE (cycles)');p.title(3,'Does a different predictor improve transfer?','Positive = larger error under XGBoost')
    s.export('E06b_xgb_comparison',d)
    return [finish(s,p,'E06b_predictor_transfer_control','Saved XGBoost and ABC evaluations from the same fixed clinical scan-budget analysis. Reference cross-validation uses simulations and does not have the clinical target distribution. '
      'The reference-to-clinical error gap is shown descriptively; it alone does not prove overfitting rather than distributional/model mismatch.',{'source_rows':len(d)})]


@experiment('E06c','Simulation-reference-library size sensitivity')
def e06c(s):
    d=s.read('downsampling');req=['reference_size','MAE_mean','MAE_q025','MAE_q975','Spearman_mean']
    if not set(req)<=set(d):raise Blocked('Reference downsampling summary missing '+str(sorted(set(req)-set(d))))
    unique(d,['reference_size'],'Reference-size experiment');d=d.sort_values('reference_size');p=Page(2)
    ax=p[0];ax.plot(d.reference_size,d.MAE_mean,'o-');ax.fill_between(d.reference_size,d.MAE_q025,d.MAE_q975,alpha=.13);ax.set_xlabel('Number of reference simulations');ax.set_ylabel('Clinical MAE (cycles)');p.title(0,'Does a larger library improve exact timing?','Shading: central 95% of saved subsampling results')
    p[1].plot(d.reference_size,d.Spearman_mean,'o-');p[1].set_xlabel('Number of reference simulations');p[1].set_ylabel('Mean clinical Spearman correlation');p.title(1,'Effect on patient-specific timing ranks')
    s.export('E06c_reference_size_summary',d)
    return [finish(s,p,'E06c_reference_library_size','Reference-library subsampling at the original fixed neighborhood settings. Intervals summarize repeated reference subsets, not a patient-bootstrap confidence interval. '
      'Keeping k fixed changes the fraction of the library used as library size changes. This does not prove that an arbitrarily large or retuned library could not improve prediction.',{'source_rows':len(d)})]
