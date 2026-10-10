from rescue.runtime import workspace_path, allocated_threads
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from scipy.stats import t as student_t
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path('scripts').resolve()))
from run_venice_k1_abc_scan_budget import normalize_id, numeric, load_clinical_long, build_clinical_features
SEED = 20260921
N_FOLDS = 5
N_DRAWS = 100000
DF_NOISE = 4.0
JITTER_SCALE = 0.08
MAX_LAST_CYCLE = 5
SCAN_COUNTS = [1, 2, 3, 4, 5, 6]
ROOT = Path('.')
ORACLE_FILE = ROOT / 'venice_k1_abc_reference_10k' / 'exponential_nadir_vs_observed_psa' / 'exponential_vs_observed_psa_patient_level.csv'
CLINICAL_LONG = ROOT / 'data' / 'source_data' / 'venice_psa_scheduled_long.csv'
ADSL = ROOT / 'data' / 'venice_csv' / 'adsl.csv'
OUTDIR = ROOT / 'venice_k1_abc_reference_10k' / 'hierarchical_biexponential_empirical_v1'
OUTDIR.mkdir(parents=True, exist_ok=True)

def model_y(q, d, g, t):
    """
    log-relative biexponential:
      y(t) = log[(1-q)e^(-dt) + q e^(gt)]
    """
    q = np.asarray(q, dtype=float)
    d = np.asarray(d, dtype=float)
    g = np.asarray(g, dtype=float)
    return np.logaddexp(np.log1p(-q) - d * t, np.log(q) + g * t)

def analytic_tmin(q, d, g):
    q = np.asarray(q, dtype=float)
    d = np.asarray(d, dtype=float)
    g = np.asarray(g, dtype=float)
    numerator = np.log1p(-q) + np.log(d) - np.log(q) - np.log(g)
    return numerator / (d + g)

def transform_params(df):
    q = np.clip(df['clinical_q'].to_numpy(float), 1e-10, 1.0 - 1e-10)
    d = np.clip(df['clinical_d'].to_numpy(float), 1e-08, None)
    g = np.clip(df['clinical_g'].to_numpy(float), 1e-08, None)
    z = np.column_stack([np.log(q / (1.0 - q)), np.log(d), np.log(g)])
    return z

def empirical_prior_coordinates(train):
    q = np.clip(train['clinical_q'].to_numpy(float), 1e-10, 1.0 - 1e-10)
    d = np.clip(train['clinical_d'].to_numpy(float), 1e-08, None)
    g = np.clip(train['clinical_g'].to_numpy(float), 1e-08, None)
    tmin = analytic_tmin(q, d, g)
    z = np.column_stack([tmin, np.log(d), np.log(g)])
    valid = np.all(np.isfinite(z), axis=1)
    return z[valid]

def robust_scale(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        scale = np.std(x, ddof=1)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return float(scale)

def estimate_noise_scale(train):
    residuals = []
    for _, r in train.iterrows():
        q = float(r['clinical_q'])
        d = float(r['clinical_d'])
        g = float(r['clinical_g'])
        for cycle in range(1, MAX_LAST_CYCLE + 1):
            col = f'y{cycle}'
            if col not in r.index:
                continue
            y = r[col]
            if not np.isfinite(y):
                continue
            pred = model_y(q, d, g, cycle)
            residuals.append(float(y - pred))
    residuals = np.asarray(residuals, dtype=float)
    if len(residuals) == 0:
        return 0.2
    center = np.median(residuals)
    mad = np.median(np.abs(residuals - center))
    sigma = 1.4826 * mad
    sigma = float(np.clip(sigma, 0.1, 1.0))
    return sigma

def draw_empirical_population_prior(train, rng):
    z = empirical_prior_coordinates(train)
    if len(z) == 0:
        raise RuntimeError('No valid training q,d,g triples.')
    anchor_idx = rng.integers(0, len(z), size=N_DRAWS)
    draws = z[anchor_idx].copy()
    scales = np.array([robust_scale(z[:, 0]), robust_scale(z[:, 1]), robust_scale(z[:, 2])], dtype=float)
    draws += JITTER_SCALE * rng.normal(size=draws.shape) * scales
    lower = np.min(z, axis=0)
    upper = np.max(z, axis=0)
    draws[:, 0] = np.clip(draws[:, 0], max(0.0, lower[0]), upper[0])
    draws[:, 1] = np.clip(draws[:, 1], lower[1], upper[1])
    draws[:, 2] = np.clip(draws[:, 2], lower[2], upper[2])
    tmin = draws[:, 0]
    d = np.exp(draws[:, 1])
    g = np.exp(draws[:, 2])
    exponent = np.clip((d + g) * tmin, -700, 700)
    q = d / (d + g * np.exp(exponent))
    valid = np.isfinite(q) & np.isfinite(d) & np.isfinite(g) & np.isfinite(tmin) & (q > 1e-10) & (q < 1.0) & (d > 0) & (g > 0) & (tmin >= 0)
    return {'q': q[valid], 'd': d[valid], 'g': g[valid], 'tmin': tmin[valid]}

def posterior_weights(draws, row, scan_count, sigma):
    n = len(draws['tmin'])
    if scan_count == 1:
        return np.full(n, 1.0 / n, dtype=float)
    loglik = np.zeros(n, dtype=float)
    last_cycle = scan_count - 1
    for cycle in range(1, last_cycle + 1):
        y_obs = float(row[f'y{cycle}'])
        mu = model_y(draws['q'], draws['d'], draws['g'], cycle)
        resid = (y_obs - mu) / sigma
        loglik += student_t.logpdf(resid, df=DF_NOISE) - np.log(sigma)
    loglik -= logsumexp(loglik)
    return np.exp(loglik)

def weighted_quantile_sorted(sorted_values, sorted_weights, probs):
    cumulative = np.cumsum(sorted_weights)
    cumulative /= cumulative[-1]
    return np.interp(probs, cumulative, sorted_values)

def posterior_summary(draws, weights, tmin_order, last_cycle):
    ess = 1.0 / np.sum(weights ** 2)
    w_sorted = weights[tmin_order]
    t_sorted = draws['tmin'][tmin_order]
    q025, q25, median, q75, q975 = weighted_quantile_sorted(t_sorted, w_sorted, [0.025, 0.25, 0.5, 0.75, 0.975])
    return {'pred_tmin_mean': float(np.sum(weights * draws['tmin'])), 'pred_tmin_median': float(median), 'posterior_q025': float(q025), 'posterior_q25': float(q25), 'posterior_q75': float(q75), 'posterior_q975': float(q975), 'posterior_ci95_width': float(q975 - q025), 'prob_tmin_future': float(np.sum(weights[draws['tmin'] > last_cycle])), 'prob_tmin_gt8': float(np.sum(weights[draws['tmin'] > 8.0])), 'posterior_ESS': float(ess), 'q_post_mean': float(np.sum(weights * draws['q'])), 'd_post_mean': float(np.sum(weights * draws['d'])), 'g_post_mean': float(np.sum(weights * draws['g']))}

def evaluate_group(x, target_col, label):
    x = x[np.isfinite(x[target_col]) & np.isfinite(x['pred_tmin_median'])].copy()
    if len(x) == 0:
        return {'target': target_col, 'cohort': label, 'n': 0}
    true = x[target_col].to_numpy(float)
    pred = x['pred_tmin_median'].to_numpy(float)
    err = pred - true
    ae = np.abs(err)
    rho = spearmanr(true, pred).statistic
    out = {'target': target_col, 'cohort': label, 'n': len(x), 'bias': float(np.mean(err)), 'MAE': float(np.mean(ae)), 'median_AE': float(np.median(ae)), 'RMSE': float(np.sqrt(np.mean(err ** 2))), 'within_1': float(np.mean(ae <= 1.0)), 'within_2': float(np.mean(ae <= 2.0)), 'Spearman_rho': float(rho), 'median_CI95_width': float(np.median(x['posterior_ci95_width'])), 'median_ESS': float(np.median(x['posterior_ESS']))}
    cls = (true > 8.0).astype(int)
    if len(np.unique(cls)) == 2:
        out['AUROC_tmin8'] = float(roc_auc_score(cls, x['prob_tmin_gt8']))
    else:
        out['AUROC_tmin8'] = np.nan
    return out
oracle = pd.read_csv(ORACLE_FILE)
oracle['id'] = oracle['id'].map(normalize_id)
for c in ['clinical_q', 'clinical_d', 'clinical_g', 'exp_tmin', 'observed_nadir_cycle']:
    oracle[c] = numeric(oracle[c])
if oracle['valid_nonboundary'].dtype != bool:
    oracle['valid_nonboundary'] = oracle['valid_nonboundary'].astype(str).str.lower().isin(['true', '1', 'yes'])
oracle = oracle[oracle['valid_nonboundary'] & np.isfinite(oracle['clinical_q']) & np.isfinite(oracle['clinical_d']) & np.isfinite(oracle['clinical_g']) & (oracle['clinical_q'] > 0) & (oracle['clinical_q'] < 1) & (oracle['clinical_d'] > 0) & (oracle['clinical_g'] > 0)].copy()
oracle = oracle.drop_duplicates('id').reset_index(drop=True)
print('Nonboundary rescue-like patients:', len(oracle))
clinical_long = load_clinical_long(CLINICAL_LONG, ADSL)
features = build_clinical_features(clinical_long, MAX_LAST_CYCLE)
features['id'] = features['id'].map(normalize_id)
data = oracle.merge(features, on='id', how='left')
Y_COLS = [f'y{i}' for i in range(1, MAX_LAST_CYCLE + 1)]
print('Patients after feature merge:', len(data))
print('Complete through cycle 5:', int(data[Y_COLS].notna().all(axis=1).sum()))
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
data['fold'] = -1
for fold, (train_idx, valid_idx) in enumerate(kf.split(data)):
    data.loc[valid_idx, 'fold'] = fold
fixed = data[(data['support_class_full_trajectory'] == 'well_supported') & (data['exp_tmin'] > MAX_LAST_CYCLE)].dropna(subset=Y_COLS).copy()
fixed_ids = set(fixed['id'])
print('\nFixed well-supported cohort:', len(fixed))
if len(fixed) != 133:
    print('WARNING: previous ABC cohort had n=133.')
strict_observed_future_ids = set(fixed.loc[fixed['observed_nadir_cycle'] > MAX_LAST_CYCLE, 'id'])
print('Of these, observed PSA nadir > cycle 5:', len(strict_observed_future_ids))
rng_master = np.random.default_rng(SEED)
prediction_rows = []
prior_rows = []
for fold in range(N_FOLDS):
    print('\n' + '=' * 72)
    print(f'FOLD {fold + 1}/{N_FOLDS}')
    train = data[data['fold'] != fold].copy()
    valid = data[data['fold'] == fold].copy()
    valid = valid[valid['id'].isin(fixed_ids)].copy()
    sigma = estimate_noise_scale(train)
    fold_rng = np.random.default_rng(rng_master.integers(0, 2 ** 32 - 1))
    draws = draw_empirical_population_prior(train, fold_rng)
    tmin_order = np.argsort(draws['tmin'])
    print('training patients:', len(train))
    print('held-out fixed patients:', len(valid))
    print('prior draws retained:', len(draws['tmin']))
    print('Student-t noise scale:', f'{sigma:.3f}')
    print('empirical prior tmin 2.5/50/97.5%:', np.quantile(draws['tmin'], [0.025, 0.5, 0.975]))
    prior_rows.append({'fold': fold, 'n_train': len(train), 'n_valid_fixed': len(valid), 'noise_sigma': sigma, 'prior_draws': len(draws['tmin']), 'jitter_scale': JITTER_SCALE, 'prior_tmin_q025': np.quantile(draws['tmin'], 0.025), 'prior_tmin_median': np.median(draws['tmin']), 'prior_tmin_q975': np.quantile(draws['tmin'], 0.975)})
    for _, row in valid.iterrows():
        for scan_count in SCAN_COUNTS:
            last_cycle = scan_count - 1
            weights = posterior_weights(draws, row, scan_count, sigma)
            post = posterior_summary(draws, weights, tmin_order, last_cycle)
            prediction_rows.append({'id': row['id'], 'fold': fold, 'scan_count': scan_count, 'last_cycle': last_cycle, 'observed_nadir_cycle': row['observed_nadir_cycle'], 'exp_tmin': row['exp_tmin'], 'clinical_q': row['clinical_q'], 'clinical_d': row['clinical_d'], 'clinical_g': row['clinical_g'], 'support_class_full_trajectory': row['support_class_full_trajectory'], 'observed_future_after_cycle5': row['observed_nadir_cycle'] > MAX_LAST_CYCLE, **post})
pred = pd.DataFrame(prediction_rows)
priors = pd.DataFrame(prior_rows)
summary_rows = []
for scan_count in SCAN_COUNTS:
    x = pred[pred['scan_count'] == scan_count].copy()
    for target in ['exp_tmin', 'observed_nadir_cycle']:
        result = evaluate_group(x, target, 'fixed_133')
        result['scan_count'] = scan_count
        summary_rows.append(result)
        strict = x[x['id'].isin(strict_observed_future_ids)]
        result = evaluate_group(strict, target, 'strict_observed_future')
        result['scan_count'] = scan_count
        summary_rows.append(result)
summary = pd.DataFrame(summary_rows)
pred.to_csv(OUTDIR / 'hierarchical_biexponential_predictions.csv', index=False)
summary.to_csv(OUTDIR / 'hierarchical_biexponential_summary.csv', index=False)
priors.to_csv(OUTDIR / 'hierarchical_biexponential_fold_priors.csv', index=False)
print('\n' + '=' * 120)
print('CROSS-VALIDATED EMPIRICAL-PRIOR HIERARCHICAL BIEXPONENTIAL RESULTS')
print('Each held-out patient contributes ONLY early PSA to their posterior.')
print("Population priors are learned from OTHER patients' full-trajectory q,d,g fits.")
print('\nPRIMARY TARGET: ACTUAL OBSERVED PSA NADIR')
primary = summary[(summary['target'] == 'observed_nadir_cycle') & (summary['cohort'] == 'fixed_133')][['scan_count', 'n', 'bias', 'MAE', 'median_AE', 'within_1', 'within_2', 'Spearman_rho', 'AUROC_tmin8', 'median_CI95_width', 'median_ESS']]
print(primary.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print('\nSECONDARY TARGET: FULL-TRAJECTORY EXPONENTIAL NADIR')
secondary = summary[(summary['target'] == 'exp_tmin') & (summary['cohort'] == 'fixed_133')][['scan_count', 'n', 'bias', 'MAE', 'median_AE', 'within_1', 'within_2', 'Spearman_rho', 'AUROC_tmin8', 'median_CI95_width', 'median_ESS']]
print(secondary.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print('\nSTRICT SUBSET: ACTUAL OBSERVED NADIR IS AFTER CYCLE 5')
strict = summary[(summary['target'] == 'observed_nadir_cycle') & (summary['cohort'] == 'strict_observed_future')][['scan_count', 'n', 'MAE', 'median_AE', 'within_1', 'within_2', 'Spearman_rho', 'AUROC_tmin8']]
print(strict.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print('\nInterpretation of scan_count=1:')
print('baseline only -> prior-only prediction; no patient-specific trajectory information.')
print('\nWrote:')
print(OUTDIR / 'hierarchical_biexponential_predictions.csv')
print(OUTDIR / 'hierarchical_biexponential_summary.csv')
print(OUTDIR / 'hierarchical_biexponential_fold_priors.csv')
