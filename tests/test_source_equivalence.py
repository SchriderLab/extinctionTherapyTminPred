"""Optional direct checks against the supplied source archive (no SLiM execution)."""
import ast,os,sys,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from rescue.runtime import PACKAGE,load_config
from rescue.abc import distances,scales
from rescue.clinical_design import generate

@pytest.fixture
def original_archive():
    path=os.environ.get('RESCUE_SOURCE_ARCHIVE')
    if not path:pytest.skip('Set RESCUE_SOURCE_ARCHIVE for original-source comparisons')
    with zipfile.ZipFile(path) as z:yield z

def test_distance_function_matches_uploaded_source(original_archive):
    source=original_archive.read('sources/cancer_sims/scripts/tune_and_run_hybrid_abc_80.py').decode();tree=ast.parse(source)
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='distance_matrices')
    ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'original','exec'),ns)
    rng=np.random.default_rng(42);r=rng.normal(size=(10,4));q=rng.normal(size=(6,4));sd=scales(r)
    old=ns['distance_matrices'](r,np.diff(r,axis=1),q,np.diff(q,axis=1),*sd)
    new=distances(r,q,sd)
    for a,b in zip(old,new):np.testing.assert_array_equal(a,b)

@pytest.mark.parametrize('key,name,flag,narg',[
 ('calibration','generate_venice_heterogeneous_cloud_manifest.py','--cloud','--n-per-regime'),
 ('forecasting','generate_venice_k1_abc_reference_manifest.py','--clinical','--n')])
def test_clinical_parameter_preset_matches_uploaded_generator(original_archive,tmp_path,monkeypatch,key,name,flag,narg):
    cloud=pd.DataFrame({'decline_rate_hat':np.linspace(.1,.9,100),'fit_tmin_in_observed_window':True,'growth_near_lower_bound':False,'q_near_lower_bound':False})
    cp=tmp_path/'cloud.csv';cloud.to_csv(cp,index=False);preset=load_config(PACKAGE/'config/clinical.yaml')[key];preset['n']=31
    generated=tmp_path/'new.csv';generate(cp,preset,generated)
    source=original_archive.read('sources/cancer_sims/scripts/'+name).decode();ns={'__name__':'original_generator','__file__':str(tmp_path/name)};exec(compile(source,name,'exec'),ns)
    monkeypatch.setattr(sys,'argv',[name,flag,str(cp),'--outdir',str(tmp_path/'old'),narg,'31','--seed',str(preset['seed'])]);ns['main']()
    old=pd.read_csv(tmp_path/'old'/('patient_manifest.csv' if key=='calibration' else 'reference_manifest.csv'))
    if key=='calibration':old=old[old.scenario=='broad_mix'].reset_index(drop=True)
    new=pd.read_csv(generated)[old.columns]
    pd.testing.assert_frame_equal(old,new,check_dtype=False,rtol=1e-14,atol=1e-14)
