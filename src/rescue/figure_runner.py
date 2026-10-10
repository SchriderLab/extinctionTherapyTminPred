"""Run the retained clinical figure generator and verify the complete figure set."""
from pathlib import Path
import tempfile,os,sys,shutil,json
import pandas as pd
from .runtime import PACKAGE,run_command,dump_json,digest

CLINICAL_FIGURES=[
'E01_cohort_eligibility','E02_direct_ABC_exact_timing','E03a_clinical_only_exact_forecast','E03b_ABC_GP_correction',
'E03c_exact_cross_trial_transfer','E03d_binary_cross_trial_transfer','E04a_biexponential_fit_examples',
'E04b_biexponential_nadir_diagnostics','E05a_simulator_parameter_relationships','E05b_clinical_simulation_support',
'E06_calibrated_ABC_scan_budget','E06b_predictor_transfer_control','E06c_reference_library_size',
'E07_binary_ABC_early_late','E07b_ABC_weighting_control','E08_empirical_Bayes_forecasting',
'E08b_joint_qg_prior_ablation','E09a_clinical_covariate_test','E09b_clinical_training_size',
'E11a_local_nadir_window','E11b_heldout_window_examples']

def clinical_figures(cfg):
    ws=Path(cfg['clinical']['workspace']).resolve();root=Path(cfg['run_dir']).resolve();parent=ws/'figure_builds';parent.mkdir(parents=True,exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix='run_',dir=parent));env=os.environ.copy();env['MPLBACKEND']='Agg'
    script=PACKAGE/'workflow/plotting/venice/make_figures.py'
    run_command([sys.executable,str(script),'--root',str(ws),'--repo-root',str(ws),'--out',str(out),'--experiments','manuscript'],ws/'logs/clinical_figures.log',cwd=ws,env=env)
    r=pd.read_csv(out/'run_report.csv')
    if not r.status.eq('SUCCESS').all():raise RuntimeError(f'Clinical figures incomplete: {out}')
    target=root/'figures';target.mkdir(parents=True,exist_ok=True);record={}
    for stem in CLINICAL_FIGURES:
        if not (out/f'{stem}.pdf').is_file():raise FileNotFoundError(out/f'{stem}.pdf')
        for ext in ['pdf','svg','png']:
            f=out/f'{stem}.{ext}'
            if f.is_file():shutil.copy2(f,target/f.name)
        record[stem]=digest(target/f'{stem}.pdf')
    # Patient-level figure source tables remain in the private workspace.
    dump_json(root/'report/clinical_figures.json',record)
