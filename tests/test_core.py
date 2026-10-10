from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from rescue.runtime import load_config,PACKAGE
from rescue.design import sampled_design,validate,lhs
from rescue.abc import distances,scales,predict,estimate,tune
from rescue.evaluate import reactive,lookup,noise_predict
from rescue.tables import trajectory,reference_features
from rescue.clinical import TASKS

CFG=load_config(PACKAGE/'config/primary.yaml')

def test_patient_lhs_reproduces_source_first_patient():
    p=CFG['primary'];x=sampled_design(p,80,20260908,680000001,780000001,'patient_id')
    np.testing.assert_allclose(x.iloc[0][['base_mut_rate','dose_drug_1','dose_drug_2','turnover_rate']].to_numpy(float),[4.951929817e-8,.780768064268,.517160476699,.178776566222],rtol=1e-10)
    validate(x,'patient_id')

@pytest.mark.parametrize('n',[3,10,80,100])
def test_stratification(n):
    x=lhs(n,5,31)
    for j in range(5):assert sorted(np.floor(x[:,j]*n).astype(int))==list(range(n))

@pytest.mark.parametrize('n',[3,80])
def test_design_weights_and_drug_order(n):
    p=CFG['primary'];x=sampled_design(p,n,100,680000001,780000001,'patient_id');validate(x,'patient_id')
    assert (x.dose_drug_2<x.dose_drug_1).all()
    np.testing.assert_allclose(x[['draw_m1','draw_m2','draw_m3']].sum(axis=1),1)

@pytest.mark.parametrize('bad',['draw_m1','dose_drug_1','patient_id','expansion_seed'])
def test_invalid_manifest_rejected(bad):
    p=CFG['primary'];x=sampled_design(p,3,31,100,200,'patient_id');x.loc[0,bad]=np.nan
    with pytest.raises(ValueError):validate(x,'patient_id')

def test_distance_is_rms_levels_plus_rms_slopes():
    r=np.array([[5,4,3,2],[5,4.5,4,3.5]],float);q=np.array([[5,4.2,3.4,2.6]]);sd=scales(r)
    a,b=distances(r,q,sd)
    np.testing.assert_allclose(a[0],np.sqrt(np.mean(((q-r)/sd[0])**2,axis=1)))
    np.testing.assert_allclose(b[0],np.sqrt(np.mean(((np.diff(q)-np.diff(r))/sd[1])**2,axis=1)))

def test_prediction_matches_inverse_distance_definition():
    r=np.array([[5,4,3,2],[5,4.4,3.8,3.2],[5,4.8,4.6,4.4]],float);y=np.array([5,8,12]);q=np.array([[5,4.1,3.2,2.3]])
    dl,ds=distances(r,q,scales(r));d=dl[0]+2*ds[0];idx=np.argsort(d)[:2];w=1/(d[idx]+1e-8)
    np.testing.assert_allclose(predict(r,y,q,2,2),(w*y[idx]).sum()/w.sum())

@pytest.mark.parametrize('batch',[1,3,96])
def test_noise_matrix_distance_agrees_with_primary_distance(batch):
    rng=np.random.default_rng(59);r=rng.normal(3,.7,(50,4));y=rng.uniform(4,15,50);q=rng.normal(3,.7,(7,4))
    np.testing.assert_allclose(noise_predict(r,y,q,5,2,batch,1e-8),predict(r,y,q,5,2),rtol=1e-11,atol=1e-11)

def test_reactive_threshold_and_fallback():
    # Last axis has four actual sampled cycles; use 0,1,2,3 for this unit test.
    obs=np.log10(np.array([[[100,40,10,13],[100,40,10,11]]])+1)
    t,hit=reactive(obs,[0,1,2,3],1.25,True)
    np.testing.assert_equal(t,[[3,3]]);np.testing.assert_equal(hit,[[True,False]])
    t,hit=reactive(obs,[0,1,2,3],10,False);assert np.isnan(t).all();assert not hit.any()

def test_lag_offset_and_no_switch():
    surface=np.array([[.1,.2,.3],[.8,.7,.6]])
    np.testing.assert_allclose(lookup(surface,np.array([[0.,2.],[1.,np.nan]])),[[.1,.6],[.2,np.nan]],equal_nan=True)

def test_reference_filter_uses_future_interior_nadir_not_full_length():
    rows=[]
    for rid,vals in enumerate([[100,50,20,10,5,7],[100,50,20,10,5,3],[100,50,20,10,0]]):
        rows += [dict(ref_id=rid,t_treatment=t,N=n) for t,n in enumerate(vals)]
    allref,clean=reference_features(pd.DataFrame(rows),[0,1,2,3])
    assert allref.usable.sum()==3
    assert list(clean.ref_id)==[0]
    assert clean.iloc[0].t_min_true==4

def test_absolute_cycle_not_implicitly_treatment_time():
    with pytest.raises(ValueError):trajectory(pd.DataFrame({'cycle':[100,101],'N':[10,5]}))

def test_duplicate_times_rejected():
    with pytest.raises(ValueError):trajectory(pd.DataFrame({'t':[0,0],'N':[10,5]}))

def test_cohort_and_reference_seeds_disjoint():
    p=CFG['primary'];r=CFG['reference'];patients=sampled_design(p,80,p['lhs_seed'],p['expansion_seed_base'],p['d1_seed_base'],'patient_id')
    refs=sampled_design(p,100,r['lhs_seed'],r['expansion_seed_base'],r['d1_seed_base'],'ref_id')
    assert not (set(patients.d1_seed)&set(refs.d1_seed))

def test_clinical_graph_has_no_cycles_or_e10_target():
    done=set()
    def visit(k,stack):
        assert k not in stack
        for d in TASKS[k]['deps']:visit(d,stack|{k})
        done.add(k)
    for k in TASKS:visit(k,set())
    assert done==set(TASKS)
    assert all('E10' not in t['script'] for t in TASKS.values())

def test_all_clinical_outputs_have_single_owner():
    o=[p for t in TASKS.values() for p in t['out']];assert len(o)==len(set(o))
    for t in TASKS.values():assert (PACKAGE/'workflow/clinical'/t['script']).is_file()

def test_only_one_slim_implementation():
    files=list((PACKAGE/'models').glob('*.slim'));assert len(files)==1
    s=files[0].read_text();assert s.count('initializeMutationType("m1"')==1
    assert 'base_mut_rate * Q' in s
    assert 'm1.mutationStackPolicy = "s"' in s
    assert 'm2.mutationStackPolicy = "s"' in s
    assert 'R1 > 1.0' in s and 'R2 > 1.0' in s
    assert 'setSeed(' in s

def test_tiny_reference_cv(tmp_path):
    cfg=load_config(PACKAGE/'config/smoke.yaml');r=np.random.default_rng(3)
    df=pd.DataFrame(r.normal(4,.3,(30,4)),columns=['logN0','logN1','logN2','logN3']);df['ref_id']=range(30);df['t_min_true']=r.integers(4,15,30)
    f=tmp_path/'ref.csv';df.to_csv(f,index=False);tune(cfg,f,tmp_path/'abc')
    b=json.loads((tmp_path/'abc/selected.json').read_text());assert set(b)=={'baseline_logN','logN_plus_slope'}
    o=pd.read_csv(tmp_path/'abc/logN_plus_slope_reference_oof.csv');assert len(o)==30 and o.pred_tmin.notna().all()


def test_expansion_cli_dispatch(tmp_path, monkeypatch):
    """Public 'expand' action calls the expansion function without launching SLiM."""
    import importlib.util
    import sys
    import yaml
    from rescue import simulate
    from rescue.runtime import PACKAGE
    root = tmp_path / 'run'
    (root / 'design').mkdir(parents=True)
    pd.DataFrame([{'patient_id': 0}]).to_csv(root / 'design/patients.csv', index=False)
    config = tmp_path / 'dispatch.yaml'
    config.write_text(yaml.safe_dump({'run_dir': str(root)}))
    spec = importlib.util.spec_from_file_location('rescue_test_driver', PACKAGE/'scripts/run_task.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(simulate, 'expansion', lambda cfg, row, path: calls.append((row['patient_id'], path)))
    monkeypatch.setattr(sys, 'argv', ['run_task.py', '--config', str(config), 'expand', '000'])
    module.main()
    assert calls == [(0, root/'simulation/patients/000')]
