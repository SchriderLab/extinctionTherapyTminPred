#!/usr/bin/env python3
"""Bind existing paper inputs without modifying/copying the research checkout."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rescue.runtime import load_config

def first(paths):return next((str(p.resolve()) for p in paths if p.exists()),None)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--working-repo',type=Path,required=True)
    ap.add_argument('--slim',default='/nas/longleaf/rhel9/apps/slim/5.1/bin/slim');ap.add_argument('--force',action='store_true')
    a=ap.parse_args();s=a.source.resolve();r=a.working_repo.resolve()
    if s==ROOT:raise ValueError('Clinical source directory must not be the package/repository itself')
    dest=ROOT/'config/local.yaml';cdest=ROOT/'config/clinical.local.yaml'
    if (dest.exists() or cdest.exists()) and not a.force:raise FileExistsError('Local configuration exists. Review it or use --force; research files are never changed.')
    ref=first([r/'data/reference',s/'abc_ref_hybrid_regen80_v1_10k']);cohort=first([r/'data/cohort',s/'fixed80_hybrid_heterogeneous_regenerated_v1'])
    manifest=first([r/'data/reference/manifests/manifest_abc_ref_hybrid_regen80_v1_10k.csv',s/'abc_ref_hybrid_regen80_v1_10k/manifest_abc_ref_hybrid_regen80_v1_10k.csv'])
    seeds=first([r/'data/reference/abc_reference_features_all.csv',s/'abc_ref_hybrid_regen80_v1_10k/abc_reference_features_all.csv'])
    paths={
      'adsl':'data/venice_csv/adsl.csv','adlb':'data/venice_csv/adlb.csv','mainsail_lab':'data/mainsail_pc002/lab.sas7bdat',
      'direct_library':'abc_ref_clinical_expanded_v2_screen5k/reference_trajectories.csv',
      'qg_folds':'venice_k1_abc_reference_10k/venice_real_discrete_nadir_cv_v1/binary_tmin8_real_cv_predictions.csv',
      'calibration_manifest':'venice_heterogeneous_cloud_v1/patient_manifest.csv','calibration_raw':'venice_heterogeneous_cloud_v1/raw',
      'forecasting_manifest':'venice_k1_abc_reference_10k/reference_manifest.csv','forecasting_raw':'venice_k1_abc_reference_10k/raw'}
    inputs={k:first([s/v,r/v]) for k,v in paths.items()}
    cc={'extends':['clinical.yaml'],'inputs':inputs};cdest.write_text(yaml.safe_dump(cc,sort_keys=False))
    local={'extends':['primary.yaml'],'slim':a.slim,'reference':{'manifest':manifest or 'MISSING_REFERENCE_MANIFEST','resolved_seeds':seeds},
           'frozen':{'cohort':cohort or 'MISSING_COHORT','reference':ref or 'MISSING_REFERENCE'},
           'clinical':{'config':'config/clinical.local.yaml'}}
    dest.write_text(yaml.safe_dump(local,sort_keys=False))
    # Convenience overlays retain the user's explicit source bindings.
    (ROOT/'config/local_frozen.yaml').write_text(yaml.safe_dump({'extends':['local.yaml'],'mode':'frozen','run_dir':'results/runs/from_frozen'},sort_keys=False))
    (ROOT/'config/local_smoke.yaml').write_text(yaml.safe_dump({'extends':['smoke.yaml'],'slim':a.slim},sort_keys=False))
    missing=[k for k,v in inputs.items() if v is None]
    print('WROTE',dest);print('WROTE',cdest)
    print('Original primary reference manifest:',manifest or 'MISSING')
    print('Missing clinical inputs:',', '.join(missing) or 'none')
    print('No research files, Git files, or patient tables were copied or changed.')
if __name__=='__main__':main()
