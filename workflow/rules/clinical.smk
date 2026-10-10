# Generated from the explicit clinical.TASKS registry; no historical launchers.
rule clinical_stage:
    input: R+'/run.json', RUN['clinical']['config'], KERNELS, CODE,
           [str(Path(v).resolve()) for k,v in CC.get('inputs',{}).items() if v and (CC['library_mode']=='saved' or not k.startswith(('calibration_','forecasting_')))]
    output: W+'/.steps/staged.json'
    params: digest=CHASH
    resources: mem_mb=4000, runtime=30
    shell: 'python {DRIVER:q} --config {CFG:q} clinical_stage'

if CC['library_mode']=='generate':
    rule calibration_design:
        input: W+'/'+CLOUD, W+'/.steps/staged.json',CODE
        output: W+'/'+C+'patient_manifest.csv'
        resources: mem_mb=2000,runtime=15
        shell: 'python {DRIVER:q} --config {CFG:q} clinical_design calibration'

    rule forecasting_design:
        input: W+'/'+CLOUD, W+'/.steps/staged.json',CODE
        output: W+'/'+K+'reference_manifest.csv'
        resources: mem_mb=2000,runtime=15
        shell: 'python {DRIVER:q} --config {CFG:q} clinical_design forecasting'

    rule calibration_batch:
        input: W+'/'+C+'patient_manifest.csv',MODEL,CODE
        output: W+'/'+C+'batches/{batch}.csv'
        wildcard_constraints: batch=r'\d{4}'
        resources: mem_mb=8000,runtime=480
        shell: 'python {DRIVER:q} --config {CFG:q} clinical_batch calibration {wildcards.batch}'

    rule forecasting_batch:
        input: W+'/'+K+'reference_manifest.csv',MODEL,CODE
        output: W+'/'+K+'batches/{batch}.csv'
        wildcard_constraints: batch=r'\d{4}'
        resources: mem_mb=8000,runtime=480
        shell: 'python {DRIVER:q} --config {CFG:q} clinical_batch forecasting {wildcards.batch}'

rule clinical_cohort:
    input:
        staged=W+'/.steps/staged.json',
        deps=[W+'/'+p for p in []],
        code=KERNELS+CODE
    output:
        data=[W+'/'+p for p in ['data/source_data/venice_locked_268_ids.csv', 'data/source_data/venice_first_rebound_nadirs.csv', 'data/source_data/venice_psa_scheduled_long.csv']],
        record=W+'/.steps/cohort.json'
    threads: 1
    resources: mem_mb=16000, runtime=720
    shell: 'python {DRIVER:q} --config {CFG:q} --threads {threads} clinical cohort'

rule clinical_direct:
    input:
        staged=W+'/.steps/staged.json',
        deps=[W+'/'+p for p in ['data/source_data/venice_locked_268_ids.csv', 'data/source_data/venice_first_rebound_nadirs.csv', 'data/source_data/venice_psa_scheduled_long.csv']],
        code=KERNELS+CODE
    output:
        data=[W+'/'+p for p in ['data/source_data/venice_abc_horizon/venice_abc_horizon_predictions.csv', 'data/source_data/venice_abc_horizon/venice_abc_horizon_summary.txt']],
        record=W+'/.steps/direct.json'
    threads: 1
    resources: mem_mb=16000, runtime=720
    shell: 'python {DRIVER:q} --config {CFG:q} --threads {threads} clinical direct'

rule clinical_ridge:
    input:
        staged=W+'/.steps/staged.json',
        deps=[W+'/'+p for p in ['data/source_data/venice_locked_268_ids.csv', 'data/source_data/venice_first_rebound_nadirs.csv', 'data/source_data/venice_psa_scheduled_long.csv']],
        code=KERNELS+CODE
    output:
        data=[W+'/'+p for p in ['data/source_data/venice_within_cohort_cv/venice_repeated_cv_predictions_by_patient.csv', 'data/source_data/venice_within_cohort_cv/venice_within_cohort_cv_summary.csv']],
        record=W+'/.steps/ridge.json'
    threads: 4
    resources: mem_mb=16000, runtime=720
    shell: 'python {DRIVER:q} --config {CFG:q} --threads {threads} clinical ridge'

rule clinical_gp:
    input:
        staged=W+'/.steps/staged.json',
        deps=[W+'/'+p for p in ['data/source_data/venice_abc_horizon/venice_abc_horizon_predictions.csv', 'data/source_data/venice_abc_horizon/venice_abc_horizon_summary.txt']],
        code=KERNELS+CODE
    output:
        data=[W+'/'+p for p in ['data/source_data/venice_simulation_gp/simulation_gp_cv_predictions_by_patient.csv', 'data/source_data/venice_simulation_gp/simulation_gp_summary.csv']],
        record=W+'/.steps/gp.json'
    threads: 4
    resources: mem_mb=16000, runtime=720
    shell: 'python {DRIVER:q} --config {CFG:q} --threads {threads} clinical gp'

rule clinical_mainsail_cohort:
    input:
        staged=W+'/.steps/staged.json',
        deps=[W+'/'+p for p in []],
        code=KERNELS+CODE
    output:
        data=[W+'/'+p for p in ['data/source_data/mainsail/mainsail_early_features.csv', 'data/source_data/mainsail/mainsail_first_rebound_nadirs.csv', 'data/source_data/mainsail/mainsail_forecasting_ids.csv']],
        record=W+'/.steps/mainsail_cohort.json'
    threads: 1
    resources: mem_mb=16000, runtime=720
    shell: 'python {DRIVER:q} --config {CFG:q} --threads {threads} clinical mainsail_cohort'

rule clinical_cross_trial:
    input:
        staged=W+'/.steps/staged.json',
        deps=[W+'/'+p for p in ['data/source_data/venice_locked_268_ids.csv', 'data/source_data/venice_first_rebound_nadirs.csv', 'data/source_data/venice_psa_scheduled_long.csv', 'data/source_data/mainsail/mainsail_early_features.csv', 'data/source_data/mainsail/mainsail_first_rebound_nadirs.csv', 'data/source_data/mainsail/mainsail_forecasting_ids.csv']],
        code=KERNELS+CODE
    output:
        data=[W+'/'+p for p in ['data/source_data/mainsail_validation/step1_mainsail_regression_all_predictions.csv', 'data/source_data/mainsail_validation/step2_mainsail_binary_cutoff7_predictions.csv', 'data/source_data/mainsail_validation/step3_regression_VENICE_to_MAINSAIL.csv', 'data/source_data/mainsail_validation/step3_regression_MAINSAIL_to_VENICE.csv', 'data/source_data/mainsail_validation/step3_binary_VENICE_to_MAINSAIL.csv', 'data/source_data/mainsail_validation/step3_binary_MAINSAIL_to_VENICE.csv', 'data/source_data/mainsail_validation/mainsail_manuscript_validation_summary.json']],
        record=W+'/.steps/cross_trial.json'
    threads: 4
    resources: mem_mb=16000, runtime=720
    shell: 'python {DRIVER:q} --config {CFG:q} --threads {threads} clinical cross_trial'

rule clinical_fit_cloud:
    input:
        staged=W+'/.steps/staged.json',
        deps=[W+'/'+p for p in ['data/source_data/venice_locked_268_ids.csv', 'data/source_data/venice_first_rebound_nadirs.csv', 'data/source_data/venice_psa_scheduled_long.csv']],
        code=KERNELS+CODE
    output:
        data=[W+'/'+p for p in ['data/source_data/exponential_cloud/venice_exponential_cloud_per_patient.csv']],
        record=W+'/.steps/fit_cloud.json'
    threads: 1
    resources: mem_mb=16000, runtime=720
    shell: 'python {DRIVER:q} --config {CFG:q} --threads {threads} clinical fit_cloud'

rule clinical_calibration_fits:
    input:
        staged=W+'/.steps/staged.json',
        deps=[W+'/'+p for p in ['data/source_data/exponential_cloud/venice_exponential_cloud_per_patient.csv']],
        code=KERNELS+CODE,
        batches=(expand(W+'/venice_heterogeneous_cloud_v1/batches/{batch}.csv',batch=[f'{i:04d}' for i in range(math.ceil(CC['calibration']['n']/CC['calibration']['batch_size']))]) if CC['library_mode']=='generate' else [])
    output:
        data=[W+'/'+p for p in ['venice_heterogeneous_cloud_v1/analysis/patient_fits.csv', 'venice_heterogeneous_cloud_v1/analysis/parameter_to_exponential_correlations.csv']],
        record=W+'/.steps/calibration_fits.json'
    threads: 1
    resources: mem_mb=16000, runtime=720
    shell: 'python {DRIVER:q} --config {CFG:q} --threads {threads} clinical calibration_fits'

rule clinical_matching:
    input:
        staged=W+'/.steps/staged.json',
        deps=[W+'/'+p for p in ['venice_heterogeneous_cloud_v1/analysis/patient_fits.csv', 'venice_heterogeneous_cloud_v1/analysis/parameter_to_exponential_correlations.csv']],
        code=KERNELS+CODE
    output:
        data=[W+'/'+p for p in ['venice_heterogeneous_cloud_v1/patient_matching/venice_patient_mechanistic_matches.csv']],
        record=W+'/.steps/matching.json'
    threads: 1
    resources: mem_mb=16000, runtime=720
    shell: 'python {DRIVER:q} --config {CFG:q} --threads {threads} clinical matching'

rule clinical_reference_features:
    input:
        staged=W+'/.steps/staged.json',
        deps=[W+'/'+p for p in []],
        code=KERNELS+CODE,
        batches=(expand(W+'/venice_k1_abc_reference_10k/batches/{batch}.csv',batch=[f'{i:04d}' for i in range(math.ceil(CC['forecasting']['n']/CC['forecasting']['batch_size']))]) if CC['library_mode']=='generate' else [])
    output:
        data=[W+'/'+p for p in ['venice_k1_abc_reference_10k/abc_reference_features_usable.csv']],
        record=W+'/.steps/reference_features.json'
    threads: 1
    resources: mem_mb=16000, runtime=720
    shell: 'python {DRIVER:q} --config {CFG:q} --threads {threads} clinical reference_features'

rule clinical_query:
    input:
        staged=W+'/.steps/staged.json',
        deps=[W+'/'+p for p in ['venice_heterogeneous_cloud_v1/patient_matching/venice_patient_mechanistic_matches.csv', 'venice_k1_abc_reference_10k/abc_reference_features_usable.csv', 'data/source_data/venice_locked_268_ids.csv', 'data/source_data/venice_first_rebound_nadirs.csv', 'data/source_data/venice_psa_scheduled_long.csv']],
        code=KERNELS+CODE
    output:
        data=[W+'/'+p for p in ['venice_k1_abc_reference_10k/venice_real_4scan/venice_real_4scan_abc_predictions.csv']],
        record=W+'/.steps/query.json'
    threads: 1
    resources: mem_mb=16000, runtime=720
    shell: 'python {DRIVER:q} --config {CFG:q} --threads {threads} clinical query'

rule clinical_fit_agreement:
    input:
        staged=W+'/.steps/staged.json',
        deps=[W+'/'+p for p in ['venice_k1_abc_reference_10k/venice_real_4scan/venice_real_4scan_abc_predictions.csv']],
        code=KERNELS+CODE
    output:
        data=[W+'/'+p for p in ['venice_k1_abc_reference_10k/exponential_nadir_vs_observed_psa/exponential_vs_observed_psa_patient_level.csv', 'venice_k1_abc_reference_10k/exponential_nadir_vs_observed_psa/exponential_vs_observed_psa_summary.csv']],
        record=W+'/.steps/fit_agreement.json'
    threads: 1
    resources: mem_mb=16000, runtime=720
    shell: 'python {DRIVER:q} --config {CFG:q} --threads {threads} clinical fit_agreement'

rule clinical_scan_budget:
    input:
        staged=W+'/.steps/staged.json',
        deps=[W+'/'+p for p in ['venice_k1_abc_reference_10k/venice_real_4scan/venice_real_4scan_abc_predictions.csv']],
        code=KERNELS+CODE
    output:
        data=[W+'/'+p for p in ['venice_k1_abc_reference_10k/venice_scan_budget_global_tmin_v1/reference_scan_features_cycles0_5.csv', 'venice_k1_abc_reference_10k/venice_scan_budget_global_tmin_v1/scan_budget_clinical_predictions.csv', 'venice_k1_abc_reference_10k/venice_scan_budget_global_tmin_v1/scan_budget_fixed_common_summary.csv', 'venice_k1_abc_reference_10k/venice_scan_budget_global_tmin_v1/scan_budget_clinical_summary.csv']],
        record=W+'/.steps/scan_budget.json'
    threads: 1
    resources: mem_mb=16000, runtime=720
    shell: 'python {DRIVER:q} --config {CFG:q} --threads {threads} clinical scan_budget'

rule clinical_xgboost:
    input:
        staged=W+'/.steps/staged.json',
        deps=[W+'/'+p for p in ['venice_k1_abc_reference_10k/venice_scan_budget_global_tmin_v1/reference_scan_features_cycles0_5.csv', 'venice_k1_abc_reference_10k/venice_scan_budget_global_tmin_v1/scan_budget_clinical_predictions.csv', 'venice_k1_abc_reference_10k/venice_scan_budget_global_tmin_v1/scan_budget_fixed_common_summary.csv', 'venice_k1_abc_reference_10k/venice_scan_budget_global_tmin_v1/scan_budget_clinical_summary.csv']],
        code=KERNELS+CODE
    output:
        data=[W+'/'+p for p in ['venice_k1_abc_reference_10k/venice_xgboost_tmin_v1/xgb_venice_summary.csv']],
        record=W+'/.steps/xgboost.json'
    threads: 1
    resources: mem_mb=16000, runtime=720
    shell: 'python {DRIVER:q} --config {CFG:q} --threads {threads} clinical xgboost'

rule clinical_reference_size:
    input:
        staged=W+'/.steps/staged.json',
        deps=[W+'/'+p for p in ['venice_k1_abc_reference_10k/venice_scan_budget_global_tmin_v1/reference_scan_features_cycles0_5.csv', 'venice_k1_abc_reference_10k/venice_scan_budget_global_tmin_v1/scan_budget_clinical_predictions.csv', 'venice_k1_abc_reference_10k/venice_scan_budget_global_tmin_v1/scan_budget_fixed_common_summary.csv', 'venice_k1_abc_reference_10k/venice_scan_budget_global_tmin_v1/scan_budget_clinical_summary.csv']],
        code=KERNELS+CODE
    output:
        data=[W+'/'+p for p in ['venice_k1_abc_reference_10k/venice_abc_reference_downsampling_v1/downsampling_summary.csv']],
        record=W+'/.steps/reference_size.json'
    threads: 1
    resources: mem_mb=16000, runtime=720
    shell: 'python {DRIVER:q} --config {CFG:q} --threads {threads} clinical reference_size'

rule clinical_binary:
    input:
        staged=W+'/.steps/staged.json',
        deps=[W+'/'+p for p in ['venice_k1_abc_reference_10k/venice_scan_budget_global_tmin_v1/reference_scan_features_cycles0_5.csv', 'venice_k1_abc_reference_10k/venice_scan_budget_global_tmin_v1/scan_budget_clinical_predictions.csv', 'venice_k1_abc_reference_10k/venice_scan_budget_global_tmin_v1/scan_budget_fixed_common_summary.csv', 'venice_k1_abc_reference_10k/venice_scan_budget_global_tmin_v1/scan_budget_clinical_summary.csv']],
        code=KERNELS+CODE
    output:
        data=[W+'/'+p for p in ['venice_k1_abc_reference_10k/venice_binary_tmin8_v1/binary_tmin8_venice_predictions.csv', 'venice_k1_abc_reference_10k/venice_binary_tmin8_v1/binary_tmin8_venice_summary.csv']],
        record=W+'/.steps/binary.json'
    threads: 1
    resources: mem_mb=16000, runtime=720
    shell: 'python {DRIVER:q} --config {CFG:q} --threads {threads} clinical binary'

rule clinical_rejection:
    input:
        staged=W+'/.steps/staged.json',
        deps=[W+'/'+p for p in ['venice_k1_abc_reference_10k/venice_binary_tmin8_v1/binary_tmin8_venice_predictions.csv', 'venice_k1_abc_reference_10k/venice_binary_tmin8_v1/binary_tmin8_venice_summary.csv']],
        code=KERNELS+CODE
    output:
        data=[W+'/'+p for p in ['venice_k1_abc_reference_10k/venice_rejection_abc_tmin8_v1/rejection_venice_summary.csv']],
        record=W+'/.steps/rejection.json'
    threads: 1
    resources: mem_mb=16000, runtime=720
    shell: 'python {DRIVER:q} --config {CFG:q} --threads {threads} clinical rejection'

rule clinical_empirical_bayes:
    input:
        staged=W+'/.steps/staged.json',
        deps=[W+'/'+p for p in ['venice_k1_abc_reference_10k/exponential_nadir_vs_observed_psa/exponential_vs_observed_psa_patient_level.csv', 'venice_k1_abc_reference_10k/exponential_nadir_vs_observed_psa/exponential_vs_observed_psa_summary.csv']],
        code=KERNELS+CODE
    output:
        data=[W+'/'+p for p in ['venice_k1_abc_reference_10k/hierarchical_biexponential_empirical_v1/hierarchical_biexponential_predictions.csv', 'venice_k1_abc_reference_10k/hierarchical_biexponential_empirical_v1/hierarchical_biexponential_summary.csv']],
        record=W+'/.steps/empirical_bayes.json'
    threads: 1
    resources: mem_mb=16000, runtime=720
    shell: 'python {DRIVER:q} --config {CFG:q} --threads {threads} clinical empirical_bayes'

rule clinical_qg_ablation:
    input:
        staged=W+'/.steps/staged.json',
        deps=[W+'/'+p for p in ['data/source_data/exponential_cloud/venice_exponential_cloud_per_patient.csv']],
        code=KERNELS+CODE
    output:
        data=[W+'/'+p for p in ['venice_k1_abc_reference_10k/qg_joint_prior_ablation_v1/qg_joint_vs_shuffled_summary.csv']],
        record=W+'/.steps/qg_ablation.json'
    threads: 1
    resources: mem_mb=16000, runtime=720
    shell: 'python {DRIVER:q} --config {CFG:q} --threads {threads} clinical qg_ablation'

rule clinical_covariates_training:
    input:
        staged=W+'/.steps/staged.json',
        deps=[W+'/'+p for p in ['venice_k1_abc_reference_10k/hierarchical_biexponential_empirical_v1/hierarchical_biexponential_predictions.csv', 'venice_k1_abc_reference_10k/hierarchical_biexponential_empirical_v1/hierarchical_biexponential_summary.csv']],
        code=KERNELS+CODE
    output:
        data=[W+'/'+p for p in ['venice_k1_abc_reference_10k/venice_binary_final_checks_v1/binary_feature_ablation_summary.csv', 'venice_k1_abc_reference_10k/venice_binary_final_checks_v1/binary_learning_curve_summary.csv']],
        record=W+'/.steps/covariates_training.json'
    threads: 4
    resources: mem_mb=16000, runtime=720
    shell: 'python {DRIVER:q} --config {CFG:q} --threads {threads} clinical covariates_training'

rule clinical_decline_audit:
    input:
        staged=W+'/.steps/staged.json',
        deps=[W+'/'+p for p in []],
        code=KERNELS+CODE
    output:
        data=[W+'/'+p for p in ['venice_decline_filter_audit/all_scheduled_psa.csv', 'venice_decline_filter_audit/selected_early_features.csv']],
        record=W+'/.steps/decline_audit.json'
    threads: 1
    resources: mem_mb=16000, runtime=720
    shell: 'python {DRIVER:q} --config {CFG:q} --threads {threads} clinical decline_audit'

rule clinical_window:
    input:
        staged=W+'/.steps/staged.json',
        deps=[W+'/'+p for p in ['venice_decline_filter_audit/all_scheduled_psa.csv', 'venice_decline_filter_audit/selected_early_features.csv']],
        code=KERNELS+CODE
    output:
        data=[W+'/'+p for p in ['data/source_data/venice_nadir_window_pm1_v1/pm1_window_oof_predictions.csv', 'data/source_data/venice_nadir_window_pm1_v1/pm1_window_summary.csv', 'data/source_data/venice_nadir_window_pm1_v1/pm1_window_bootstrap_summary.csv']],
        record=W+'/.steps/window.json'
    threads: 4
    resources: mem_mb=16000, runtime=720
    shell: 'python {DRIVER:q} --config {CFG:q} --threads {threads} clinical window'

rule clinical_figures:
    input: [W+'/'+p for name in TASKS for p in TASKS[name]['out']],CFIGCODE,CODE
    output: CFIGS,R+'/report/clinical_figures.json'
    resources: mem_mb=10000,runtime=120
    shell: 'python {DRIVER:q} --config {CFG:q} clinical_figures'
