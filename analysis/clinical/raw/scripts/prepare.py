from common import *

def venice(c,out,jobs):
    a=pd.read_csv(c['adsl'],low_memory=False);l=pd.read_csv(c['adlb'],low_memory=False)
    for x in [a,l]:x['USUBJID']=x.USUBJID.astype(str).str.strip()
    a=a[a.RANDOM.astype(str).str.upper().eq('Y')&a.ARMCD.astype(str).str.upper().eq('PLACEBO')].copy()
    baseline=pd.to_numeric(a.set_index('USUBJID').TMPSA,errors='coerce')
    baseline=baseline[baseline>0]
    l=l[l.LBTESTCD.astype(str).str.upper().str.contains('PSA',na=False)&
        l.EPOCH.astype(str).str.strip().str.casefold().eq('treatment phase')].copy()
    l['cycle']=pd.to_numeric(l.ANLVISN,errors='coerce');l['psa']=pd.to_numeric(l.LBSTRESN,errors='coerce')
    l['day']=pd.to_numeric(l.LBDY2,errors='coerce').fillna(pd.to_numeric(l.LBDY,errors='coerce'))
    l=l[l.cycle.notna()&(l.psa>0)&l.USUBJID.isin(baseline.index)].copy();l['cycle']=l.cycle.astype(int)
    l=l[l.cycle>=1];l['day_error']=(l.day-21*l.cycle).abs().fillna(np.inf)
    l=l.sort_values(['USUBJID','cycle','day_error','day']).drop_duplicates(['USUBJID','cycle'])
    l=l.rename(columns={'USUBJID':'id'})[['id','cycle','psa','day']]
    base=pd.DataFrame({'id':baseline.index,'cycle':0,'psa':baseline.values,'day':0.})
    long=pd.concat([base,l]).sort_values(['id','cycle']).reset_index(drop=True)
    rows=[];items=[];early=0;post=0;locked=0
    for pid,z in long.groupby('id',sort=True):
        t=z.cycle.to_numpy(float);y=z.psa.to_numpy(float)
        if not set([0,1,2,3])<=set(t):continue
        early+=1
        if np.any(t>3):post+=1
        j=int(np.argmin(y))
        if t[j]<=3 or not np.any(t>t[j]):continue
        locked+=1;treb,confirm,idx=endpoint(t,y)
        if not np.isfinite(treb):continue
        rows.append(dict(id=pid,Treb=treb,confirm_cycle=confirm,event_index=idx,
            global_cycle=t[j],last_cycle=max(t)))
        items.append((pid,t,y))
    target=pd.DataFrame(rows);counts=dict(comparator=len(a),early=early,post3=post,locked=locked,local_target=len(target))
    if counts!=dict(comparator=612,early=506,post3=482,locked=268,local_target=263):
        dump(out/'COHORT_REPRODUCTION_FAILED.json',counts);raise RuntimeError('Historical cohort counts differ')
    audit=Path(c['audit_case'])/'audit'/c['audit_pass']/'cohorts'
    flags=pd.read_csv(audit/'manuscript_comparator_patient_flags_PRIVATE.csv')
    assert_ids(target.id,flags.loc[bools(flags.historical_selected),'id'],'Historical 263 membership differs')
    chk=target.merge(flags[['id','local_low_cycle']],on='id',validate='one_to_one')
    if not np.array_equal(chk.Treb,chk.local_low_cycle):raise RuntimeError('Historical targets differ')
    input_hash={str(Path(c[k])):digest(c[k]) for k in ['adsl','adlb']}
    fitpath=out/'cohort/fit263_PRIVATE.csv';cachemeta=out/'cohort/FIT_INPUTS.json'
    if fitpath.exists():
        if not cachemeta.exists() or json.loads(cachemeta.read_text())!=input_hash:raise RuntimeError('Fit cache input mismatch')
        fits=pd.read_csv(fitpath)
    else:
        print('Refitting all 263 with original 120-start point-fit algorithm; bootstrap omitted.',flush=True)
        fits=parallel_fits(items,jobs);csv(fitpath,fits);dump(cachemeta,input_hash)
    if not bools(fits.fit_success).all():raise RuntimeError('A historical fit failed')
    ids=fits.loc[bools(fits.retained),'id'];expected=pd.read_csv(audit/'rerun_calibration231_membership_PRIVATE.csv')
    expids=expected.loc[bools(expected.audit_nonpathological),'id']
    csv(out/'cohort/membership_comparison_PRIVATE.csv',pd.DataFrame({'id':sorted(set(fits.id)|set(expected.id))}).assign(
        reproduced=lambda x:x.id.isin(ids),historical=lambda x:x.id.isin(expids)))
    assert_ids(ids,expids,'Re-fitted 231 membership differs; inspect comparison. No split or models created.')
    if len(ids)!=231:raise RuntimeError('Expected 231')
    parent=target.merge(fits[fits.id.isin(ids)],on='id',validate='one_to_one').sort_values('id').reset_index(drop=True)
    perm=np.random.default_rng(c['seed']).permutation(parent.id.to_numpy());test=set(perm[:c['test_n']])
    parent['split']=np.where(parent.id.isin(test),'TEST','TRAIN')
    split=parent[['id','split']];splitpath=out/'cohort/SPLIT_PRIVATE.csv'
    if splitpath.exists():
        old=pd.read_csv(splitpath).sort_values('id').reset_index(drop=True)
        if not split.equals(old):raise RuntimeError('Refusing to change frozen patient split')
    csv(splitpath,split);csv(out/'cohort/PARENT_231_PRIVATE.csv',parent)
    training=parent[parent.split=='TRAIN'].copy();testref=parent[parent.split=='TEST'].copy()
    csv(out/'training/fits_PRIVATE.csv',training);csv(out/'evaluation/heldout_reference_PRIVATE.csv',testref)
    csv(out/'training/long_PRIVATE.csv',long[long.id.isin(training.id)])
    csv(out/'queries/venice_long_PRIVATE.csv',long[long.id.isin(testref.id)])
    # Query data may contain a whole recorded history; every feature routine truncates at its forecast time.
    covariates(c,out,training.id,testref.id)
    retained_counts={}
    for n in c['scan_counts']:
        for label,tab in [('TRAIN',training),('TEST',testref)]:
            Y=matrix(long,tab.id,n-1);complete=np.isfinite(Y).all(axis=1)
            retained_counts[f'{label}_{n}_scans']=dict(complete=int(complete.sum()),future_Treb=int((complete&(tab.Treb.to_numpy()>n-1)).sum()))
    stats=dict(**counts,fit_retained=231,train_n=len(training),test_n=len(testref),seed=c['seed'],
        negative_fitted_nadir_n=int((parent.tmin_fit_hat<0).sum()),
        retained_with_fitted_nadir_in_window=int(bools(parent.fit_tmin_in_observed_window).sum()),
        availability=retained_counts,input_sha256=input_hash,
        interpretation='Frozen retrospective fit-eligible cohort; TEST is an internal reanalysis, not historically untouched validation.',
        endpoint='First observed local low after cycle 3: low<=previous and next>low. One confirming rise; adjacent available visits.')
    dump(out/'cohort/COHORT_REPRODUCTION.json',stats)
    dump(out/'training/AGREEMENT_Tfit_vs_Treb.json',metrics(training.Treb,training.tmin_fit_hat))
    print(json.dumps(plain(stats),indent=2),flush=True)
    make_manifest(c,out,training)

def covariates(c,out,train_ids,test_ids):
    a=pd.read_csv(c['adsl'],low_memory=False);a['id']=a.USUBJID.astype(str)
    b=pd.DataFrame({'id':a.id,'log_abs_psa0':np.log(pd.to_numeric(a.TMPSA,errors='coerce').clip(lower=1e-12)),
        'ECOG':pd.to_numeric(a.ECOGIVRN,errors='coerce'),
        'BMI':pd.to_numeric(a.WEIGHTBL,errors='coerce')/(pd.to_numeric(a.HEIGHTBL,errors='coerce')/100)**2})
    l=pd.read_csv(c['adlb'],low_memory=False);l['id']=l.USUBJID.astype(str)
    names=['TESTOSTERONE','HEMOGLOBIN','ALKALINE PHOSPHATASE','CREATININE','PHOSPHORUS','CALCIUM','ALBUMIN']
    l=l[l.LBBLFL.astype(str).str.upper().eq('Y')&l.LBTEST.astype(str).str.upper().isin(names)].copy()
    l['name']=l.LBTEST.astype(str).str.upper();l['value']=pd.to_numeric(l.LBSTRESN,errors='coerce')
    w=l.pivot_table(index='id',columns='name',values='value',aggfunc='median').reindex(columns=names)
    w['TESTOSTERONE']=np.log(w.TESTOSTERONE.clip(lower=1e-6));w['ALKALINE PHOSPHATASE']=np.log1p(w['ALKALINE PHOSPHATASE'].clip(lower=0))
    b=b.merge(w,on='id',how='left').drop_duplicates('id')
    csv(out/'training/baseline_PRIVATE.csv',b[b.id.isin(train_ids)])
    csv(out/'queries/venice_baseline_PRIVATE.csv',b[b.id.isin(test_ids)])
    dump(out/'training/BASELINE_FEATURE_POLICY.json',dict(features=[x for x in b if x!='id'],
        laboratory_rule='LBBLFL=Y; median of baseline flagged rows; original Section 3.8 feature definitions.',
        missing_fraction_training=b[b.id.isin(train_ids)].drop(columns='id').isna().mean().to_dict()))

def make_manifest(c,out,training):
    from rescue.clinical_design import generate
    preset=dict(phase='clinical_single',n=c['simulation_n'],seed=c['seed']+1,seed_base=6200000000,
        n_founders=5,batch_size=c['simulation_batch_size'],max_retries=c['simulation_max_retries'],
        prefix_length=c['simulation_horizon'],relapse_factor=4.,d_quantiles=[.05,.95],turnover=[.05,.40],
        mutation_rate=[1e-9,1e-7],m1_mean_effect=[.20,.90],m1_sdlog=[.15,.80],age_range=[15,23],
        standing=dict(median=.035,sdlog=1.20,clip=[.0001,.35],zero_probability=.10))
    generate(out/'training/fits_PRIVATE.csv',preset,out/'simulation/reference_manifest.csv')
    dump(out/'simulation/PRESET.json',preset)
    dump(out/'simulation/DESIGN.json',dict(source_commit=c['clinical_source_commit'],
        engine_sha256=digest(PACKAGE/'vendor/models/quantitative_resistance.slim'),
        training_ids_sha256=hashlib.sha256('\n'.join(sorted(training.id)).encode()).hexdigest(),
        training_fits_sha256=digest(out/'training/fits_PRIVATE.csv'),
        design='Original single-population model and broad biological priors; TRAIN fitted d sets dose. Joint TRAIN (d,log10q,g) shapes subsequently weight simulated fits. No q-to-resistant-fraction inversion.',
        cell_cycle_to_clinical_cycle='Original Section 3.6 convention: one treatment step per nominal clinical cycle; unchanged and unvalidated biologically.'))

def mainsail_ids(values):
    """Keep RPT identifiers textual, including leading zeros.

    Strip surrounding padding and the historical integer-float suffix only;
    never convert identifiers to integers or equate '001' with '1'.
    """
    raw=pd.Series(values,copy=True)
    if raw.isna().any():raise RuntimeError('Missing MAINSAIL patient identifier')
    raw=raw.map(lambda v:v.decode('latin1') if isinstance(v,bytes) else str(v))
    result=raw.str.strip().str.replace(r'\.0$','',regex=True)
    if result.eq('').any():raise RuntimeError('Empty MAINSAIL patient identifier')
    mapping=pd.DataFrame({'raw':raw,'id':result}).drop_duplicates()
    if mapping.groupby('id').raw.nunique().gt(1).any():
        raise RuntimeError('MAINSAIL ID normalization would merge distinct source identifiers')
    return result

def compare_mainsail(targets,historical,out):
    """Verify identities AND targets; save diagnostics before any mismatch error."""
    historical=Path(historical)
    # Crucial: default CSV inference can erase leading zeros that SAS preserves.
    h=pd.read_csv(historical,dtype={'RPT':'string'},keep_default_na=False)
    h['target_t']=pd.to_numeric(h.target_t,errors='coerce')
    h=h[h.target_t.notna()].copy();h['id']=mainsail_ids(h.RPT)
    if targets.id.duplicated().any() or h.id.duplicated().any():
        raise RuntimeError('Duplicate MAINSAIL target patient IDs; comparison is not one-to-one')
    comparison=targets[['id','Treb']].merge(h[['id','target_t']],on='id',how='outer',
        validate='one_to_one',indicator=True).sort_values('id')
    comparison['target_match']=comparison.Treb.eq(comparison.target_t)
    csv(out/'cohort/MAINSAIL_ID_COMPARISON_PRIVATE.csv',comparison)
    # Reproduce the OLD reader solely to diagnose the reported failure.
    legacy=pd.read_csv(historical,usecols=['RPT','target_t'])
    legacy=legacy[pd.to_numeric(legacy.target_t,errors='coerce').notna()].copy()
    oldids=legacy.RPT.astype(str).str.replace(r'\.0$','',regex=True)
    same_ids=bool(comparison['_merge'].eq('both').all())
    same_targets=same_ids and bool(comparison.target_match.all())
    old_match=set(targets.id)==set(oldids)
    info=dict(rebuilt_n=len(targets),historical_n=len(h),
        matched_ids=int(comparison['_merge'].eq('both').sum()),
        only_rebuilt=int(comparison['_merge'].eq('left_only').sum()),
        only_historical=int(comparison['_merge'].eq('right_only').sum()),
        mismatched_targets=int((comparison['_merge'].eq('both')&~comparison.target_match).sum()),
        old_csv_inference_dtype=str(legacy.RPT.dtype),old_reader_exact_id_match=old_match,
        rebuilt_leading_zero_ids=int(targets.id.str.match(r'^0\d').sum()),
        historical_leading_zero_ids=int(h.id.str.match(r'^0\d').sum()),
        exact_membership_match=same_ids,exact_targets_match=same_targets,
        historical_file=str(historical),historical_sha256=digest(historical),
        diagnosis=('Old CSV type inference/ID formatting caused the mismatch; text-preserving read matches exactly.'
            if same_targets and not old_match else
            'Exact IDs and targets matched.' if same_targets else
            'Mismatch remains after preserving IDs. Inspect MAINSAIL_ID_COMPARISON_PRIVATE.csv; analysis stopped.'))
    dump(out/'cohort/MAINSAIL_ID_CHECK.json',info)
    if not same_ids:
        raise RuntimeError('MAINSAIL membership differs after text-preserving ID read; see cohort/MAINSAIL_ID_CHECK.json and MAINSAIL_ID_COMPARISON_PRIVATE.csv')
    if not same_targets:
        raise RuntimeError('MAINSAIL targets differ from historical audit; see cohort/MAINSAIL_ID_COMPARISON_PRIVATE.csv')
    print('MAINSAIL:',json.dumps(info),flush=True)
    return info

def mainsail(c,out):
    p=Path(c['mainsail_lab'])
    if not p.exists():raise FileNotFoundError('MAINSAIL input missing: '+str(p))
    x=pd.read_sas(p,format='sas7bdat',encoding='latin1');x=x[x.TST_NAME.astype(str).str.upper().eq('PSA')].copy()
    x['psa']=pd.to_numeric(x.US_RES.astype(str).str.strip().str.removeprefix('<').str.strip(),errors='coerce')
    x['cycle']=pd.to_numeric(x.VISNAME.astype(str).str.strip().str.extract(r'(?i)^Cycle\s+(\d+)\s+Day\s+1$')[0],errors='coerce')
    x=x[(x.psa>0)&x.cycle.notna()].copy();x['cycle']=x.cycle.astype(int)
    b=x[x.cycle==1].sort_values(['RPT','LABDAY']).drop_duplicates('RPT')[['RPT','LABDAY']].rename(columns={'LABDAY':'day0'})
    x=x.merge(b,on='RPT',how='left');x['error']=(x.LABDAY-x.day0-21*(x.cycle-1)).abs()
    x=x.sort_values(['RPT','cycle','error','LABDAY']).drop_duplicates(['RPT','cycle'])
    x['id']=mainsail_ids(x.RPT);x['cycle']=x.cycle-1;x['day']=x.LABDAY-x.day0
    long=x[['id','cycle','psa','day']].copy();rows=[]
    for pid,z in long.groupby('id'):
        z=z.sort_values('cycle');t=z.cycle.to_numpy(float);y=z.psa.to_numpy(float)
        if not set([0,1,2,3])<=set(t) or np.sum(t>3)<2:continue
        target,confirm,j=endpoint(t,y)
        if np.isfinite(target):rows.append(dict(id=pid,Treb=target,confirm_cycle=confirm,event_index=j,last_cycle=max(t)))
    targets=pd.DataFrame(rows,columns=['id','Treb','confirm_cycle','event_index','last_cycle'])
    csv(out/'cohort/MAINSAIL_REBUILT_PRIVATE.csv',targets)
    if len(targets)!=181:raise RuntimeError(f'MAINSAIL historical target cohort changed: {len(targets)} rather than 181')
    historical=Path(c['audit_case'])/'workspace/data/source_data/mainsail/mainsail_first_rebound_nadirs.csv'
    check='historical target file unavailable; count and unchanged extraction rules checked'
    if historical.exists():
        compare_mainsail(targets,historical,out)
        check='exact historical patient IDs and targets matched'
    csv(out/'queries/mainsail_long_PRIVATE.csv',long[long.id.isin(targets.id)])
    csv(out/'evaluation/mainsail_reference_PRIVATE.csv',targets)
    dump(out/'cohort/MAINSAIL.json',dict(n=181,input_sha256=digest(p),historical_comparison=check,
        baseline='Cycle 1 Day 1 mapped to time 0; subsequent cycles minus one.',
        target='Same one-rise post-cycle-3 local low as VENICE; no global-minimum cohort gate added.',
        cross_trial_rule='Frozen VENICE training models; no MAINSAIL calibration. Reverse-direction models are separately labelled experiments.'))
