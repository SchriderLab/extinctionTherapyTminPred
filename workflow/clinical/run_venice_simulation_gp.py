from rescue.runtime import workspace_path, allocated_threads
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
CLINICAL_DIR = workspace_path('data')
LOCKED_IDS = Path('data/source_data/venice_locked_268_ids.csv')
ABC_PRED = Path('data/source_data/venice_abc_horizon/venice_abc_horizon_predictions.csv')
OUTDIR = Path('data/source_data/venice_simulation_gp')
N_SPLITS = 5
N_REPEATS = 20
SEED = 20260915
HORIZON = 12

def find_csv(root, basename):
    hits = list(root.rglob(basename))
    if len(hits) != 1:
        raise RuntimeError(f'Expected one {basename}; found {len(hits)}')
    return hits[0]
locked = pd.read_csv(LOCKED_IDS)
locked['USUBJID'] = locked['USUBJID'].astype(str)
locked_set = set(locked['USUBJID'])
adsl = pd.read_csv(find_csv(CLINICAL_DIR, 'adsl.csv'), low_memory=False)
adlb = pd.read_csv(find_csv(CLINICAL_DIR, 'adlb.csv'), low_memory=False)
adsl['USUBJID'] = adsl['USUBJID'].astype(str)
adlb['USUBJID'] = adlb['USUBJID'].astype(str)
base = adsl.loc[adsl['USUBJID'].isin(locked_set), ['USUBJID', 'TMPSA']].copy()
base['TMPSA'] = pd.to_numeric(base['TMPSA'], errors='coerce')
base = base[base['TMPSA'].notna() & np.isfinite(base['TMPSA']) & (base['TMPSA'] > 0)].drop_duplicates('USUBJID')
lab = adlb[adlb['USUBJID'].isin(locked_set)].copy()
lab = lab[lab['LBTESTCD'].astype(str).str.upper().eq('PSA')].copy()
if 'EPOCH' in lab.columns:
    treat = lab['EPOCH'].astype(str).str.upper().str.contains('TREAT', na=False)
    if treat.any():
        lab = lab[treat].copy()
lab['cycle'] = pd.to_numeric(lab['ANLVISN'], errors='coerce')
lab['PSA'] = pd.to_numeric(lab['LBSTRESN'], errors='coerce')
if 'LBDY2' in lab.columns:
    d2 = pd.to_numeric(lab['LBDY2'], errors='coerce')
else:
    d2 = pd.Series(np.nan, index=lab.index)
if 'LBDY' in lab.columns:
    d1 = pd.to_numeric(lab['LBDY'], errors='coerce')
else:
    d1 = pd.Series(np.nan, index=lab.index)
lab['day'] = d2.where(d2.notna(), d1)
lab = lab[lab['cycle'].isin([1, 2, 3]) & lab['PSA'].notna() & np.isfinite(lab['PSA']) & (lab['PSA'] > 0)].copy()
lab['cycle'] = lab['cycle'].astype(int)
lab['nominal_day'] = 21 * lab['cycle']
lab['day_missing'] = lab['day'].isna()
lab['day_distance'] = np.abs(lab['day'] - lab['nominal_day'])
lab = lab.sort_values(['USUBJID', 'cycle', 'day_missing', 'day_distance'], kind='mergesort').drop_duplicates(['USUBJID', 'cycle'], keep='first')
wide = lab.pivot(index='USUBJID', columns='cycle', values='PSA')
wide.columns = [f'psa{int(x)}' for x in wide.columns]
wide = wide.reset_index()
feat = base.merge(wide, on='USUBJID', how='inner')
feat = feat.dropna(subset=['TMPSA', 'psa1', 'psa2', 'psa3']).copy()
for j in [1, 2, 3]:
    feat[f'z{j}'] = np.log10(feat[f'psa{j}'] / feat['TMPSA'])
slopes = []
for r in feat.itertuples(index=False):
    yy = np.array([0.0, r.z1, r.z2, r.z3])
    slopes.append(np.polyfit(np.arange(4), yy, 1)[0])
feat['slope'] = slopes
abc = pd.read_csv(ABC_PRED)
abc['USUBJID'] = abc['USUBJID'].astype(str)
df = feat.merge(abc[['USUBJID', 'supported', 'observed_target_cycle', 'within_horizon_observed', 'pred_abc_idw']], on='USUBJID', how='inner')
if df['supported'].dtype != bool:
    df['supported'] = df['supported'].astype(str).str.lower().eq('true')
if df['within_horizon_observed'].dtype != bool:
    df['within_horizon_observed'] = df['within_horizon_observed'].astype(str).str.lower().eq('true')
df = df[df['supported'] & df['within_horizon_observed']].copy()
df = df.sort_values('USUBJID').reset_index(drop=True)
FEATURES = ['z1', 'z2', 'z3', 'slope']
X = df[FEATURES].to_numpy(float)
y = pd.to_numeric(df['observed_target_cycle'], errors='coerce').to_numpy(float)
sim_pred = pd.to_numeric(df['pred_abc_idw'], errors='coerce').to_numpy(float)
print('Hybrid-analysis patients:', len(df))
cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
records = []
for split_id, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
    Xtrain = X[train_idx]
    Xtest = X[test_idx]
    ytrain = y[train_idx]
    sim_train = sim_pred[train_idx]
    sim_test = sim_pred[test_idx]
    residual_train = ytrain - sim_train
    scaler = StandardScaler()
    Xtrain_s = scaler.fit_transform(Xtrain)
    Xtest_s = scaler.transform(Xtest)
    kernel = ConstantKernel(1.0, (0.01, 100.0)) * Matern(length_scale=np.ones(X.shape[1]), length_scale_bounds=(0.01, 100.0), nu=1.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(0.001, 100.0))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=SEED + split_id)
    gp.fit(Xtrain_s, residual_train)
    correction, correction_sd = gp.predict(Xtest_s, return_std=True)
    hybrid = sim_test + correction
    hybrid = np.clip(hybrid, 4, HORIZON)
    for j, gi in enumerate(test_idx):
        records.append({'split_id': split_id, 'USUBJID': df.iloc[gi]['USUBJID'], 'observed': y[gi], 'simulation_abc': sim_test[j], 'gp_correction': correction[j], 'gp_correction_sd': correction_sd[j], 'simulation_plus_gp': hybrid[j]})
pred_long = pd.DataFrame(records)
OUTDIR.mkdir(parents=True, exist_ok=True)
pred_long.to_csv(OUTDIR / 'simulation_gp_cv_predictions_long.csv', index=False)
pred = pred_long.groupby(['USUBJID', 'observed'], as_index=False)[['simulation_abc', 'gp_correction', 'gp_correction_sd', 'simulation_plus_gp']].mean()
pred.to_csv(OUTDIR / 'simulation_gp_cv_predictions_by_patient.csv', index=False)
yy = pred['observed'].to_numpy(float)
abc_p = pred['simulation_abc'].to_numpy(float)
hyb_p = pred['simulation_plus_gp'].to_numpy(float)

def metrics(name, pp):
    err = pp - yy
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    rho = spearmanr(pp, yy)
    return {'model': name, 'n': len(yy), 'MAE': mae, 'RMSE': rmse, 'bias': np.mean(err), 'within_1_visit': np.mean(np.abs(err) <= 1), 'within_2_visits': np.mean(np.abs(err) <= 2), 'Spearman_rho': rho.statistic, 'Spearman_p': rho.pvalue}
summary = pd.DataFrame([metrics('Simulation ABC', abc_p), metrics('Simulation + GP', hyb_p)])
abc_abs = np.abs(abc_p - yy)
hyb_abs = np.abs(hyb_p - yy)
delta = abc_abs - hyb_abs
nz = delta[np.abs(delta) > 1e-12]
if len(nz):
    test = wilcoxon(nz, alternative='greater')
    paired_p = test.pvalue
else:
    paired_p = np.nan
print()
print('=' * 72)
print('SIMULATION + GP DISCREPANCY TEST')
print('=' * 72)
print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print()
print('MAE improvement (ABC - hybrid):', f'{abc_abs.mean() - hyb_abs.mean():.4f}')
print('Wilcoxon H1: Simulation+GP absolute error < ABC absolute error:', f'p={paired_p:.6g}')
print()
print('Mean absolute GP correction:', f"{np.mean(np.abs(pred['gp_correction'])):.3f}")
print('Mean GP correction SD:', f"{np.mean(pred['gp_correction_sd']):.3f}")
summary['paired_vs_ABC_p'] = [np.nan, paired_p]
summary.to_csv(OUTDIR / 'simulation_gp_summary.csv', index=False)
print()
print('WROTE')
print(OUTDIR / 'simulation_gp_summary.csv')
print(OUTDIR / 'simulation_gp_cv_predictions_by_patient.csv')
