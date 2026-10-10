from rescue.runtime import workspace_path, allocated_threads
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, wilcoxon
from sklearn.model_selection import RepeatedKFold, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib.pyplot as plt
CLINICAL_DIR = workspace_path('data')
LOCKED_IDS = Path('data/source_data/venice_locked_268_ids.csv')
TARGETS = Path('data/source_data/venice_first_rebound_nadirs.csv')
OUTDIR = Path('data/source_data/venice_within_cohort_cv')
FIGDIR = Path('figures/venice_within_cohort_cv')
N_SPLITS = 5
N_REPEATS = 20
SEED = 20260915
HORIZON = 12

def find_csv(root, basename):
    hits = list(root.rglob(basename))
    if len(hits) != 1:
        raise RuntimeError(f'Expected exactly one {basename}; found {len(hits)}')
    return hits[0]

def as_bool(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().map({'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False}).fillna(False)

def bootstrap_ci(y, pred, metric, nboot=3000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        pp = pred[idx]
        if metric == 'mae':
            v = np.mean(np.abs(pp - yy))
        elif metric == 'rmse':
            v = np.sqrt(np.mean((pp - yy) ** 2))
        elif metric == 'spearman':
            if len(np.unique(yy)) < 2 or len(np.unique(pp)) < 2:
                continue
            v = spearmanr(pp, yy).statistic
            if not np.isfinite(v):
                continue
        else:
            raise ValueError(metric)
        vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return tuple(np.quantile(vals, [0.025, 0.975]))

def summarize(name, y, pred, baseline_pred):
    err = pred - y
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    bias = np.mean(err)
    within1 = np.mean(np.abs(err) <= 1)
    within2 = np.mean(np.abs(err) <= 2)
    if len(np.unique(y)) > 1 and len(np.unique(pred)) > 1:
        sp = spearmanr(pred, y)
        pe = pearsonr(pred, y)
        rho = sp.statistic
        rho_p = sp.pvalue
        r = pe.statistic
        r_p = pe.pvalue
    else:
        rho = rho_p = r = r_p = np.nan
    baseline_mae = np.mean(np.abs(baseline_pred - y))
    improvement = baseline_mae - mae
    mae_ci = bootstrap_ci(y, pred, 'mae')
    rho_ci = bootstrap_ci(y, pred, 'spearman')
    if name == 'Median baseline':
        wilcox_p = np.nan
    else:
        delta = np.abs(baseline_pred - y) - np.abs(pred - y)
        delta = delta[np.abs(delta) > 1e-12]
        if len(delta):
            try:
                wilcox_p = wilcoxon(delta, alternative='greater').pvalue
            except Exception:
                wilcox_p = np.nan
        else:
            wilcox_p = np.nan
    return {'model': name, 'n': len(y), 'MAE': mae, 'MAE_CI_low': mae_ci[0], 'MAE_CI_high': mae_ci[1], 'RMSE': rmse, 'bias': bias, 'within_1_visit': within1, 'within_2_visits': within2, 'Spearman_rho': rho, 'Spearman_p': rho_p, 'Spearman_CI_low': rho_ci[0], 'Spearman_CI_high': rho_ci[1], 'Pearson_r': r, 'Pearson_p': r_p, 'baseline_MAE': baseline_mae, 'MAE_improvement_vs_baseline': improvement, 'Wilcoxon_H1_model_better_p': wilcox_p}
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)
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
wide.columns = [f'psa{int(c)}' for c in wide.columns]
wide = wide.reset_index()
df = base.merge(wide, on='USUBJID', how='inner')
df = df.dropna(subset=['TMPSA', 'psa1', 'psa2', 'psa3']).copy()
for j in [1, 2, 3]:
    df[f'z{j}'] = np.log10(df[f'psa{j}'] / df['TMPSA'])
slopes = []
for r in df.itertuples(index=False):
    y = np.array([0.0, r.z1, r.z2, r.z3])
    slope = np.polyfit(np.arange(4), y, 1)[0]
    slopes.append(slope)
df['slope'] = slopes
targets = pd.read_csv(TARGETS)
targets['USUBJID'] = targets['USUBJID'].astype(str)
targets['target_available'] = as_bool(targets['target_available'])
targets['target_cycle'] = pd.to_numeric(targets['target_cycle'], errors='coerce')
df = df.merge(targets[['USUBJID', 'target_available', 'target_cycle']], on='USUBJID', how='inner')
df = df[df['target_available'] & df['target_cycle'].notna() & (df['target_cycle'] > 3)].copy()
df = df.sort_values('USUBJID').reset_index(drop=True)
FEATURES = ['z1', 'z2', 'z3', 'slope']
X = df[FEATURES].to_numpy(float)
y = df['target_cycle'].to_numpy(float)
print('Patients with complete early features + target:', len(df))
print('Target median:', np.median(y))
print('Target range:', np.min(y), 'to', np.max(y))
outer = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
records = []
for split_id, (train_idx, test_idx) in enumerate(outer.split(X), start=1):
    Xtrain = X[train_idx]
    ytrain = y[train_idx]
    Xtest = X[test_idx]
    baseline_value = np.median(ytrain)
    baseline_pred = np.full(len(test_idx), baseline_value)
    inner = KFold(n_splits=5, shuffle=True, random_state=SEED + split_id)
    ridge = Pipeline([('scale', StandardScaler()), ('model', Ridge())])
    ridge_grid = {'model__alpha': np.logspace(-3, 3, 13)}
    ridge_search = GridSearchCV(ridge, ridge_grid, scoring='neg_mean_absolute_error', cv=inner, n_jobs=allocated_threads())
    ridge_search.fit(Xtrain, ytrain)
    ridge_pred = ridge_search.predict(Xtest)
    for local_i, global_i in enumerate(test_idx):
        records.append({'split_id': split_id, 'USUBJID': df.iloc[global_i]['USUBJID'], 'observed': y[global_i], 'median_baseline': baseline_pred[local_i], 'ridge': ridge_pred[local_i]})
pred_long = pd.DataFrame(records)
pred_long.to_csv(OUTDIR / 'venice_repeated_cv_predictions_long.csv', index=False)
pred = pred_long.groupby(['USUBJID', 'observed'], as_index=False)[['median_baseline', 'ridge']].mean()
pred.to_csv(OUTDIR / 'venice_repeated_cv_predictions_by_patient.csv', index=False)

def evaluate_subset(subset, label):
    yy = subset['observed'].to_numpy(float)
    baseline = subset['median_baseline'].to_numpy(float)
    rows = []
    rows.append(summarize('Median baseline', yy, baseline, baseline))
    rows.append(summarize('Ridge', yy, subset['ridge'].to_numpy(float), baseline))
    out = pd.DataFrame(rows)
    out.insert(0, 'analysis', label)
    return out
all_metrics = evaluate_subset(pred, 'all_targets')
horizon_pred = pred[pred['observed'] <= HORIZON].copy()
horizon_metrics = evaluate_subset(horizon_pred, f'targets_visit_4_to_{HORIZON}')
metrics = pd.concat([all_metrics, horizon_metrics], ignore_index=True)
metrics.to_csv(OUTDIR / 'venice_within_cohort_cv_summary.csv', index=False)
for label, tab in metrics.groupby('analysis', sort=False):
    print()
    print('=' * 76)
    print(label)
    print('=' * 76)
    show = tab[['model', 'n', 'MAE', 'RMSE', 'MAE_improvement_vs_baseline', 'Spearman_rho', 'Spearman_p', 'within_1_visit', 'within_2_visits', 'Wilcoxon_H1_model_better_p']].copy()
    print(show.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
