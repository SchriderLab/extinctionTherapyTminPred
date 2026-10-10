"""Manuscript clinical task registry and isolated execution contracts."""
from __future__ import annotations
import os,sys,json
from pathlib import Path
from .runtime import PACKAGE,load_config,run_command,dump_json,digest

S='data/source_data/'
K='venice_k1_abc_reference_10k/'
C='venice_heterogeneous_cloud_v1/'
CLOUD=S+'exponential_cloud/venice_exponential_cloud_per_patient.csv'
QUERY=K+'venice_real_4scan/venice_real_4scan_abc_predictions.csv'
FEATURES=K+'abc_reference_features_usable.csv'
CACHE=K+'venice_scan_budget_global_tmin_v1/reference_scan_features_cycles0_5.csv'
# Input/output contracts refer to an ignored work directory, never the original research directory.
TASKS={
 'cohort':dict(script='rebuild_frozen_venice_targets.py',deps=[],out=[S+'venice_locked_268_ids.csv',S+'venice_first_rebound_nadirs.csv',S+'venice_psa_scheduled_long.csv']),
 'direct':dict(script='run_venice_abc_horizon.py',deps=['cohort'],args=['--reference','abc_ref_clinical_expanded_v2_screen5k/reference_trajectories.csv','--clinical-dir','data/venice_csv'],out=[S+'venice_abc_horizon/venice_abc_horizon_predictions.csv',S+'venice_abc_horizon/venice_abc_horizon_summary.txt']),
 'ridge':dict(script='run_venice_within_cohort_cv.py',deps=['cohort'],out=[S+'venice_within_cohort_cv/venice_repeated_cv_predictions_by_patient.csv',S+'venice_within_cohort_cv/venice_within_cohort_cv_summary.csv']),
 'gp':dict(script='run_venice_simulation_gp.py',deps=['direct'],out=[S+'venice_simulation_gp/simulation_gp_cv_predictions_by_patient.csv',S+'venice_simulation_gp/simulation_gp_summary.csv']),
 'mainsail_cohort':dict(script='build_mainsail_forecasting_cohort.py',deps=[],out=[S+'mainsail/mainsail_early_features.csv',S+'mainsail/mainsail_first_rebound_nadirs.csv',S+'mainsail/mainsail_forecasting_ids.csv']),
 'cross_trial':dict(script='run_mainsail_4step_validation.py',deps=['cohort','mainsail_cohort'],out=[S+'mainsail_validation/'+x for x in ['step1_mainsail_regression_all_predictions.csv','step2_mainsail_binary_cutoff7_predictions.csv','step3_regression_VENICE_to_MAINSAIL.csv','step3_regression_MAINSAIL_to_VENICE.csv','step3_binary_VENICE_to_MAINSAIL.csv','step3_binary_MAINSAIL_to_VENICE.csv','mainsail_manuscript_validation_summary.json']]),
 'fit_cloud':dict(script='build_venice_exponential_cloud.py',deps=['cohort'],out=[CLOUD]),
 'calibration_fits':dict(script='analyze_venice_heterogeneous_cloud.py',deps=['fit_cloud'],args=['--library',C,'--cloud',CLOUD,'--outdir',C+'analysis'],out=[C+'analysis/patient_fits.csv',C+'analysis/parameter_to_exponential_correlations.csv']),
 'matching':dict(script='match_venice_to_simulations.py',deps=['calibration_fits'],args=['--clinical',CLOUD,'--sim-fits',C+'analysis/patient_fits.csv','--outdir',C+'patient_matching','--scenario','broad_mix'],out=[C+'patient_matching/venice_patient_mechanistic_matches.csv']),
 'reference_features':dict(script='build_venice_k1_abc_reference_features.py',deps=[],args=['--library',K],out=[FEATURES]),
 'query':dict(script='run_venice_k1_abc_real_4scan.py',deps=['matching','reference_features','cohort'],out=[QUERY]),
 'fit_agreement':dict(script='evaluate_exponential_nadir_vs_observed_psa.py',deps=['query'],out=[K+'exponential_nadir_vs_observed_psa/exponential_vs_observed_psa_patient_level.csv',K+'exponential_nadir_vs_observed_psa/exponential_vs_observed_psa_summary.csv']),
 'scan_budget':dict(script='run_venice_k1_abc_scan_budget.py',deps=['query'],out=[CACHE,K+'venice_scan_budget_global_tmin_v1/scan_budget_clinical_predictions.csv',K+'venice_scan_budget_global_tmin_v1/scan_budget_fixed_common_summary.csv',K+'venice_scan_budget_global_tmin_v1/scan_budget_clinical_summary.csv']),
 'xgboost':dict(script='run_venice_xgboost_tmin.py',deps=['scan_budget'],out=[K+'venice_xgboost_tmin_v1/xgb_venice_summary.csv']),
 'reference_size':dict(script='run_venice_abc_reference_downsampling.py',deps=['scan_budget'],out=[K+'venice_abc_reference_downsampling_v1/downsampling_summary.csv']),
 'binary':dict(script='run_venice_k1_binary_tmin8.py',deps=['scan_budget'],out=[K+'venice_binary_tmin8_v1/binary_tmin8_venice_predictions.csv',K+'venice_binary_tmin8_v1/binary_tmin8_venice_summary.csv']),
 'rejection':dict(script='run_venice_k1_rejection_abc_tmin8.py',deps=['binary'],out=[K+'venice_rejection_abc_tmin8_v1/rejection_venice_summary.csv']),
 'empirical_bayes':dict(script='run_venice_hierarchical_biexponential_empirical.py',deps=['fit_agreement'],out=[K+'hierarchical_biexponential_empirical_v1/hierarchical_biexponential_predictions.csv',K+'hierarchical_biexponential_empirical_v1/hierarchical_biexponential_summary.csv']),
 'qg_ablation':dict(script='run_venice_qg_joint_prior_ablation.py',deps=['fit_cloud'],out=[K+'qg_joint_prior_ablation_v1/qg_joint_vs_shuffled_summary.csv']),
 'covariates_training':dict(script='run_venice_binary_final_checks.py',deps=['empirical_bayes'],out=[K+'venice_binary_final_checks_v1/binary_feature_ablation_summary.csv',K+'venice_binary_final_checks_v1/binary_learning_curve_summary.csv']),
 'decline_audit':dict(script='reproduce_venice_decline_audit.py',deps=[],args=['--data-dir','data/venice_csv','--outdir','venice_decline_filter_audit','--declines','4'],out=['venice_decline_filter_audit/all_scheduled_psa.csv','venice_decline_filter_audit/selected_early_features.csv']),
 'window':dict(script='run_venice_nadir_window_pm1.py',deps=['decline_audit'],out=[S+'venice_nadir_window_pm1_v1/pm1_window_oof_predictions.csv',S+'venice_nadir_window_pm1_v1/pm1_window_summary.csv',S+'venice_nadir_window_pm1_v1/pm1_window_bootstrap_summary.csv'])
}

INPUT_DESTINATIONS={
 'adsl':'data/venice_csv/adsl.csv', 'adlb':'data/venice_csv/adlb.csv','mainsail_lab':'data/mainsail_pc002/lab.sas7bdat',
 'direct_library':'abc_ref_clinical_expanded_v2_screen5k/reference_trajectories.csv',
 'qg_folds':'data/clinical/qg_folds.csv',
 'calibration_manifest':C+'patient_manifest.csv','calibration_raw':C+'raw',
 'forecasting_manifest':K+'reference_manifest.csv','forecasting_raw':K+'raw'
}

def settings(cfg):return load_config(cfg['clinical']['config'])

def staged_inputs(cfg: dict) -> list[str]:
    cc=settings(cfg);keys=list(INPUT_DESTINATIONS)
    if cc['library_mode']=='generate':keys=[x for x in keys if not x.startswith(('calibration_','forecasting_'))]
    return keys

def stage(cfg: dict) -> None:
    c=settings(cfg);ws=Path(cfg['clinical']['workspace']).resolve();ws.mkdir(parents=True,exist_ok=True)
    missing=[k for k in staged_inputs(cfg) if not c['inputs'].get(k) or not Path(c['inputs'][k]).exists()]
    if missing:raise FileNotFoundError('Missing clinical inputs: '+', '.join(missing)+'. Run scripts/configure_inputs.py or edit config/clinical.local.yaml.')
    records=[]
    for k in staged_inputs(cfg):
        src=Path(c['inputs'][k]).resolve();target=ws/INPUT_DESTINATIONS[k]
        # Directory links are read-only library/raw inputs; all generated analyses have separate output paths.
        target.parent.mkdir(parents=True,exist_ok=True)
        if k in {'calibration_manifest','qg_folds'}:
            import pandas as pd
            from .runtime import write_csv
            if target.is_symlink(): raise RuntimeError(f'Refusing to overwrite linked input {target}')
            table=pd.read_csv(src)
            if k=='calibration_manifest': table=table[table['scenario']=='broad_mix'].copy()
            else: table=table[['id','fold','exp_tmin','scan_count']].copy()
            write_csv(table,target)
        elif target.is_symlink():
            if target.resolve()!=src:raise RuntimeError(f'Existing link differs: {target}')
        elif target.exists():raise FileExistsError(f'Refusing to replace {target}')
        else:target.symlink_to(src,target_is_directory=src.is_dir())
        records.append({'key':k,'source':str(src),'destination':str(target),'sha256':digest(src) if src.is_file() else None})
    dump_json(ws/'.steps/staged.json',records)

def run_task(cfg: dict,name: str,threads: int=1) -> None:
    if name not in TASKS:raise ValueError(f'Unknown manuscript task: {name}')
    task=TASKS[name];cc=settings(cfg);ws=Path(cfg['clinical']['workspace']).resolve()
    script=PACKAGE/'workflow/clinical'/task['script'];args=task.get('args',[])+cc.get('arguments',{}).get(name,[])
    env=os.environ.copy();env.update(RESCUE_WORKSPACE=str(ws),RESCUE_THREADS=str(threads),MPLBACKEND='Agg',
        PYTHONPATH=os.pathsep.join([str(PACKAGE/'src'),str(PACKAGE/'workflow/clinical'),env.get('PYTHONPATH','')]))
    # Bound BLAS threads separately from estimator/joblib workers to avoid oversubscription.
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='1'
    run_command([sys.executable,str(script),*args],ws/'logs'/f'{name}.log',cwd=ws,env=env)
    missing=[str(ws/p) for p in task['out'] if not (ws/p).is_file() or not (ws/p).stat().st_size]
    if missing:raise RuntimeError(f'{name} exited without declared outputs: {missing}')
    dump_json(ws/'.steps'/f'{name}.json',{'task':name,'script_sha256':digest(script),'outputs':{p:digest(ws/p) for p in task['out']}})
