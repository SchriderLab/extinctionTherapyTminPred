"""Manuscript-only workflow. Run the controller on a compute node on Longleaf."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(workflow.basedir) / 'src'))
from rescue.runtime import load_config, config_digest
from rescue.clinical import TASKS, settings, C, K, CLOUD
from rescue.plotting import PRIMARY_FIGURES
from rescue.figure_runner import CLINICAL_FIGURES

CFG = str(Path(config.get('config_path', 'config/primary.yaml')).resolve())
RUN = load_config(CFG)
R = str(Path(RUN['run_dir']).resolve())
W = str(Path(RUN['clinical']['workspace']).resolve())
CC = settings(RUN)
CODE = sorted(str(p) for p in (Path(workflow.basedir)/'src/rescue').glob('*.py'))
KERNELS = sorted(str(p) for p in (Path(workflow.basedir)/'workflow/clinical').glob('*.py'))
MODEL = str(Path(workflow.basedir)/'models/quantitative_resistance.slim')
DRIVER = str(Path(workflow.basedir)/'scripts/run_task.py')
CFIGCODE = sorted(str(p) for p in (Path(workflow.basedir)/'workflow/plotting/venice').glob('*') if p.is_file())
SIMFIGS = [R+'/figures/'+x+'.pdf' for x in PRIMARY_FIGURES]
CFIGS = [R+'/figures/'+x+'.pdf' for x in CLINICAL_FIGURES]
PIDS = [f'{i:03d}' for i in range(RUN['primary']['n_patients'])]
LAGS = [f'{i:02d}' for i in RUN['primary']['lags']]
BATCHES = [f'{i:04d}' for i in range(math.ceil(RUN['reference']['n']/RUN['reference']['batch_size']))]
HASH = config_digest(RUN)
CHASH = config_digest(CC)

rule all:
    input: R+'/report/RESULTS.md', SIMFIGS

rule simulation:
    input: R+'/report/RESULTS.md', SIMFIGS

rule clinical:
    input: R+'/report/clinical_figures.json', CFIGS

rule paper:
    input: R+'/report/RESULTS.md', SIMFIGS, R+'/report/clinical_figures.json', CFIGS, R+'/figures/Fig1_study_design_workflow_updated.pdf'

if RUN['mode'] == 'frozen':
    rule reproduce_frozen:
        input: R+'/report/RESULTS.md', SIMFIGS
else:
    rule reproduce_frozen:
        resources: mem_mb=1000, runtime=5
        shell: "echo 'Use config/local_frozen.yaml for reproduce_frozen.' >&2; exit 2"

rule preflight:
    input: CFG, CODE, DRIVER, MODEL
    output: R+'/run.json'
    params: digest=HASH, clinical_digest=CHASH
    resources: mem_mb=2000, runtime=10
    shell: 'python {DRIVER:q} --config {CFG:q} preflight'

rule static_workflow_figure:
    input: str(Path(workflow.basedir)/'figures/Fig1_study_design_workflow_updated.pdf')
    output: R+'/figures/Fig1_study_design_workflow_updated.pdf'
    resources: mem_mb=1000, runtime=5
    shell: 'mkdir -p "$(dirname {output:q})"; cp {input:q} {output:q}'

if RUN['mode'] == 'regenerate':
    rule parameter_design:
        input:
            preflight=R+'/run.json',
            manifest=([RUN['reference']['manifest']] if RUN['reference']['design']=='frozen_manifest' else [])
        output: patients=R+'/design/patients.csv', references=R+'/design/references.csv', record=R+'/design/design.json'
        resources: mem_mb=2000, runtime=10
        shell: 'python {DRIVER:q} --config {CFG:q} manifests'

    rule patient_expansion:
        input: manifest=R+'/design/patients.csv', model=MODEL, code=CODE
        output: snapshot=R+'/simulation/patients/{pid}/pretreatment.bin', meta=R+'/simulation/patients/{pid}/expansion.json'
        wildcard_constraints: pid=r'\d{3}'
        resources: mem_mb=6000, runtime=120
        shell: 'python {DRIVER:q} --config {CFG:q} expand {wildcards.pid}'

    rule patient_prefix:
        input: snapshot=R+'/simulation/patients/{pid}/pretreatment.bin', manifest=R+'/design/patients.csv', model=MODEL, code=CODE
        output: R+'/simulation/patients/{pid}/prefix.csv'
        wildcard_constraints: pid=r'\d{3}'
        resources: mem_mb=6000, runtime=120
        shell: 'python {DRIVER:q} --config {CFG:q} prefix {wildcards.pid}'

    rule patient_outcomes:
        input: snapshot=R+'/simulation/patients/{pid}/pretreatment.bin', manifest=R+'/design/patients.csv', model=MODEL, code=CODE
        output: R+'/simulation/outcomes/{pid}_{lag}.csv'
        wildcard_constraints: pid=r'\d{3}', lag=r'\d{2}'
        resources: mem_mb=6000, runtime=240
        shell: 'python {DRIVER:q} --config {CFG:q} outcome {wildcards.pid} {wildcards.lag}'

    rule reference_batch:
        input: manifest=R+'/design/references.csv', model=MODEL, code=CODE
        output: trajectories=R+'/simulation/references/{batch}.csv', metadata=R+'/simulation/references/{batch}_metadata.csv'
        wildcard_constraints: batch=r'\d{4}'
        resources: mem_mb=6000, runtime=720
        shell: 'python {DRIVER:q} --config {CFG:q} reference_batch {wildcards.batch}'

    rule reference_table:
        input: expand(R+'/simulation/references/{batch}.csv',batch=BATCHES), CODE
        output: R+'/reference/features.csv',R+'/reference/features_all.csv',R+'/reference/trajectories.csv',R+'/reference/summary.json'
        resources: mem_mb=8000, runtime=60
        shell: 'python {DRIVER:q} --config {CFG:q} reference_table'

    rule cohort_table:
        input: expand(R+'/simulation/patients/{pid}/prefix.csv',pid=PIDS), expand(R+'/simulation/outcomes/{pid}_{lag}.csv',pid=PIDS,lag=LAGS), CODE
        output: R+'/cohort/trajectories.csv',R+'/cohort/outcomes.csv',R+'/cohort/extinction.csv',R+'/cohort/patients.csv'
        resources: mem_mb=4000, runtime=60
        shell: 'python {DRIVER:q} --config {CFG:q} cohort_table'
else:
    rule frozen_tables:
        input: R+'/run.json', CODE,
               RUN['frozen']['cohort']+'/extinction_by_lag_binary300.csv',
               RUN['frozen']['cohort']+'/patient_oracle_characterization.csv',
               RUN['frozen']['cohort']+'/all_outcomes_resolved_to300.csv',
               RUN['frozen']['reference']+'/abc_reference_features_usable.csv'
        output: R+'/cohort/trajectories.csv',R+'/cohort/outcomes.csv',R+'/cohort/extinction.csv',R+'/cohort/patients.csv',
                R+'/reference/features.csv',R+'/reference/features_all.csv',R+'/reference/summary.json',R+'/cohort/frozen_provenance.json'
        resources: mem_mb=4000, runtime=30
        shell: 'python {DRIVER:q} --config {CFG:q} frozen'

rule reference_cv:
    input: reference=R+'/reference/features.csv',code=CODE,preflight=R+'/run.json'
    output: R+'/abc/selected.json',R+'/abc/cv_summary.csv',R+'/abc/baseline_logN_reference_oof.csv',R+'/abc/logN_plus_slope_reference_oof.csv'
    threads: 1
    resources: mem_mb=8000, runtime=480
    shell: 'python {DRIVER:q} --config {CFG:q} tune'

rule predict_patients:
    input: R+'/reference/features.csv',R+'/cohort/trajectories.csv',R+'/abc/selected.json',CODE
    output: R+'/abc/predictions.csv'
    resources: mem_mb=6000, runtime=120
    shell: 'python {DRIVER:q} --config {CFG:q} predict'

rule evaluate_policies:
    input: R+'/abc/predictions.csv',R+'/cohort/extinction.csv',R+'/cohort/patients.csv',R+'/cohort/trajectories.csv',R+'/reference/features.csv',CODE
    output: R+'/evaluation/policy_by_patient.csv',R+'/evaluation/reactive_by_patient.csv',R+'/evaluation/reactive_summary.csv',R+'/evaluation/summary.json'
    resources: mem_mb=3000, runtime=30
    shell: 'python {DRIVER:q} --config {CFG:q} evaluate'

rule observation_noise:
    input: R+'/cohort/extinction.csv',R+'/cohort/patients.csv',R+'/cohort/trajectories.csv',R+'/reference/features.csv',R+'/run.json',CODE
    output: R+'/noise/summary.csv',R+'/noise/cohort_draws.csv',R+'/noise/patient_draws.csv'
    threads: 4
    resources: mem_mb=10000, runtime=480
    shell: 'OPENBLAS_NUM_THREADS={threads} OMP_NUM_THREADS={threads} python {DRIVER:q} --config {CFG:q} noise'

rule simulation_figures:
    input: R+'/evaluation/summary.json',R+'/evaluation/policy_by_patient.csv',R+'/evaluation/reactive_by_patient.csv',
           R+'/noise/summary.csv',R+'/noise/cohort_draws.csv',R+'/cohort/extinction.csv',R+'/cohort/patients.csv',R+'/cohort/trajectories.csv',R+'/abc/predictions.csv',R+'/reference/features.csv',CODE
    output: SIMFIGS
    resources: mem_mb=6000, runtime=90
    shell: 'python {DRIVER:q} --config {CFG:q} plot'

rule numerical_report:
    input: R+'/evaluation/summary.json',R+'/noise/summary.csv',CODE
    output: R+'/report/RESULTS.md',R+'/report/manuscript_comparison.csv',R+'/report/run.json'
    resources: mem_mb=2000, runtime=10
    shell: 'python {DRIVER:q} --config {CFG:q} report'

include: 'workflow/rules/clinical.smk'
