from rescue.runtime import workspace_path, allocated_threads
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
ROOT = workspace_path('')
DATA = ROOT / 'data'
SRC = DATA / 'source_data'
OUT = SRC / 'exponential_cloud'
OUT.mkdir(parents=True, exist_ok=True)
TARGET_FILE = SRC / 'venice_first_rebound_nadirs.csv'
LONG_FILE = SRC / 'venice_psa_scheduled_long.csv'
ADSL_FILE = DATA / 'venice_csv/adsl.csv'
N_BOOT = 100
SEED = 20260916
QLO = 0.05
QHI = 0.95
rng = np.random.default_rng(SEED)

def model(t, q, d, g):
    t = np.asarray(t, dtype=float)
    return (1.0 - q) * np.exp(-d * t) + q * np.exp(g * t)

def analytic_nadir(q, d, g):
    ratio = d * (1.0 - q) / (g * q)
    if not np.isfinite(ratio) or ratio <= 0:
        return np.nan
    return np.log(ratio) / (d + g)

def nadir_depth(q, d, g):
    tmin = analytic_nadir(q, d, g)
    if not np.isfinite(tmin):
        return np.nan
    val = model(np.asarray([tmin]), q, d, g)[0]
    if not np.isfinite(val) or val <= 0:
        return np.nan
    return float(val)

def residuals_log10(params, t, log_y):
    q, d, g = params
    pred = model(t, q, d, g)
    if np.any(~np.isfinite(pred)) or np.any(pred <= 0):
        return np.full(len(t), 1000000.0)
    return np.log10(pred) - log_y
BOUNDS_LO = np.array([1e-08, 1e-05, 1e-05])
BOUNDS_HI = np.array([0.95, 3.0, 3.0])

def fit_curve(t, rel_y, warm_start=None, full_multistart=True):
    t = np.asarray(t, dtype=float)
    rel_y = np.asarray(rel_y, dtype=float)
    good = np.isfinite(t) & np.isfinite(rel_y) & (rel_y > 0)
    t = t[good]
    rel_y = rel_y[good]
    if len(t) < 5:
        return None
    log_y = np.log10(rel_y)
    starts = []
    if warm_start is not None:
        warm_start = np.asarray(warm_start, dtype=float)
        if np.all(np.isfinite(warm_start)):
            warm_start = np.clip(warm_start, BOUNDS_LO * 1.001, BOUNDS_HI * 0.999)
            starts.append(warm_start)
            for scale in [0.8, 1.2]:
                starts.append(np.clip(warm_start * scale, BOUNDS_LO * 1.001, BOUNDS_HI * 0.999))
    if full_multistart:
        for q0 in [1e-05, 0.0001, 0.001, 0.01, 0.05, 0.2]:
            for d0 in [0.05, 0.2, 0.5, 1.0]:
                for g0 in [0.02, 0.1, 0.3, 0.7, 1.5]:
                    starts.append([q0, d0, g0])
    if len(starts) == 0:
        return None
    best = None
    for p0 in starts:
        try:
            fit = least_squares(residuals_log10, x0=p0, bounds=(BOUNDS_LO, BOUNDS_HI), args=(t, log_y), max_nfev=5000)
        except Exception:
            continue
        resid = residuals_log10(fit.x, t, log_y)
        ss = float(np.sum(resid ** 2))
        if best is None or ss < best[0]:
            best = (ss, fit)
    if best is None:
        return None
    _, fit = best
    q, d, g = fit.x
    pred = model(t, q, d, g)
    resid = np.log10(pred) - log_y
    t_hat = analytic_nadir(q, d, g)
    depth = nadir_depth(q, d, g)
    return {'q': float(q), 'decline_rate': float(d), 'growth_rate': float(g), 't_hat': float(t_hat), 'nadir_fraction': float(depth), 'rmse_log10': float(np.sqrt(np.mean(resid ** 2))), 'sigma_log10': float(np.std(resid, ddof=1) if len(resid) > 1 else 0.0), 'params': np.array([q, d, g]), 'residuals': resid}

def normalize_id(x):
    return pd.Series(x).astype(str).str.strip().str.replace('\\.0$', '', regex=True)

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(f'Could not find any of: {candidates}')

def quantile_or_nan(x, q):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.quantile(x, q))
target = pd.read_csv(TARGET_FILE)
target_col = pick_col(target, ['target_t', 'observed_tmin'])
target[target_col] = pd.to_numeric(target[target_col], errors='coerce')
target = target[target[target_col].notna()].copy()
target_id_col = pick_col(target, ['USUBJID', 'id'])
target['id'] = normalize_id(target[target_id_col])
long = pd.read_csv(LONG_FILE, low_memory=False)
id_col = pick_col(long, ['USUBJID', 'id'])
cycle_col = pick_col(long, ['cycle', 'ANLVISN', 'target_t'])
psa_col = pick_col(long, ['value', 'psa', 'PSA', 'LBSTRESN'])
long['id'] = normalize_id(long[id_col])
long['cycle'] = pd.to_numeric(long[cycle_col], errors='coerce')
long['psa'] = pd.to_numeric(long[psa_col], errors='coerce')
long = long[long['cycle'].notna() & long['psa'].notna() & (long['cycle'] >= 0) & (long['psa'] > 0)].copy()
adsl = pd.read_csv(ADSL_FILE, low_memory=False)
if 'RANDOM' in adsl.columns and 'ARMCD' in adsl.columns:
    adsl = adsl[adsl['RANDOM'].astype(str).str.upper().eq('Y') & adsl['ARMCD'].astype(str).str.upper().eq('PLACEBO')].copy()
adsl['id'] = normalize_id(adsl['USUBJID'])
baseline_col = pick_col(adsl, ['TMPSA', 'baseline_psa'])
adsl['baseline_psa'] = pd.to_numeric(adsl[baseline_col], errors='coerce')
baseline_map = adsl.dropna(subset=['baseline_psa']).set_index('id')['baseline_psa'].to_dict()
rows = []
for i, row in target.iterrows():
    pid = row['id']
    observed_tmin = float(row[target_col])
    gdf = long[long['id'].eq(pid)][['cycle', 'psa']].copy()
    has_zero = bool(np.any(np.isclose(gdf['cycle'], 0.0)))
    if not has_zero and pid in baseline_map and (baseline_map[pid] > 0):
        gdf = pd.concat([pd.DataFrame({'cycle': [0.0], 'psa': [baseline_map[pid]]}), gdf], ignore_index=True)
    gdf = gdf.groupby('cycle', as_index=False)['psa'].median().sort_values('cycle').reset_index(drop=True)
    if len(gdf) < 5:
        continue
    zero_idx = gdf['cycle'].sub(0.0).abs().idxmin()
    y0 = float(gdf.loc[zero_idx, 'psa'])
    if not np.isfinite(y0) or y0 <= 0:
        continue
    t = gdf['cycle'].to_numpy(dtype=float)
    rel = gdf['psa'].to_numpy(dtype=float) / y0
    fit = fit_curve(t, rel)
    if fit is None:
        continue
    q_hat = fit['q']
    d_hat = fit['decline_rate']
    g_hat = fit['growth_rate']
    t_hat = fit['t_hat']
    depth_hat = fit['nadir_fraction']
    pred = model(t, q_hat, d_hat, g_hat)
    pred_log = np.log10(pred)
    sigma = fit['sigma_log10']
    boot_q = []
    boot_d = []
    boot_g = []
    boot_t = []
    boot_depth = []
    for b in range(N_BOOT):
        noise = rng.normal(0.0, sigma, size=len(t))
        pseudo_log = pred_log + noise
        pseudo_rel = 10.0 ** pseudo_log
        bf = fit_curve(t, pseudo_rel, warm_start=fit['params'], full_multistart=False)
        if bf is None:
            continue
        if not np.all(np.isfinite([bf['q'], bf['decline_rate'], bf['growth_rate'], bf['t_hat'], bf['nadir_fraction']])):
            continue
        boot_q.append(bf['q'])
        boot_d.append(bf['decline_rate'])
        boot_g.append(bf['growth_rate'])
        boot_t.append(bf['t_hat'])
        boot_depth.append(bf['nadir_fraction'])
    n_success = len(boot_q)
    tmax = float(np.max(t))
    rows.append({'id': pid, 'n_points': len(t), 'max_observed_cycle': tmax, 'observed_tmin': observed_tmin, 'rmse_log10': fit['rmse_log10'], 'sigma_log10': sigma, 'bootstrap_success_n': n_success, 'bootstrap_success_fraction': n_success / N_BOOT, 'q_hat': q_hat, 'q_p05': quantile_or_nan(boot_q, QLO), 'q_p95': quantile_or_nan(boot_q, QHI), 'decline_rate_hat': d_hat, 'decline_rate_p05': quantile_or_nan(boot_d, QLO), 'decline_rate_p95': quantile_or_nan(boot_d, QHI), 'growth_rate_hat': g_hat, 'growth_rate_p05': quantile_or_nan(boot_g, QLO), 'growth_rate_p95': quantile_or_nan(boot_g, QHI), 'tmin_fit_hat': t_hat, 'tmin_fit_p05': quantile_or_nan(boot_t, QLO), 'tmin_fit_p95': quantile_or_nan(boot_t, QHI), 'nadir_fraction_hat': depth_hat, 'nadir_fraction_p05': quantile_or_nan(boot_depth, QLO), 'nadir_fraction_p95': quantile_or_nan(boot_depth, QHI), 'fit_tmin_in_observed_window': bool(np.isfinite(t_hat) and 0 <= t_hat <= tmax), 'growth_near_lower_bound': bool(g_hat <= 1.01e-05), 'q_near_lower_bound': bool(q_hat <= 1.01e-08)})
cloud = pd.DataFrame(rows)
cloud = cloud.sort_values('id').reset_index(drop=True)
patient_out = OUT / 'venice_exponential_cloud_per_patient.csv'
cloud.to_csv(patient_out, index=False)
global_rows = []
for parameter in ['q_hat', 'decline_rate_hat', 'growth_rate_hat', 'tmin_fit_hat', 'nadir_fraction_hat', 'sigma_log10']:
    x = pd.to_numeric(cloud[parameter], errors='coerce')
    x = x[np.isfinite(x)]
    global_rows.append({'parameter': parameter, 'min': x.min(), 'p05': x.quantile(0.05), 'p25': x.quantile(0.25), 'median': x.quantile(0.5), 'p75': x.quantile(0.75), 'p95': x.quantile(0.95), 'max': x.max()})
global_cloud = pd.DataFrame(global_rows)
global_out = OUT / 'venice_exponential_cloud_global_ranges.csv'
global_cloud.to_csv(global_out, index=False)
pd.set_option('display.max_columns', None)
print('=' * 80)
print('VENICE EXPONENTIAL CLOUD')
print('=' * 80)
print(f'Patients: {len(cloud)}')
print('\nPer-patient output:')
print(patient_out)
print('\nOverall parameter ranges:')
print(global_cloud.to_string(index=False))
print('\nBootstrap success:')
print(cloud['bootstrap_success_fraction'].describe())
print('\nFits with tmin inside observed time window:', int(cloud['fit_tmin_in_observed_window'].sum()), '/', len(cloud))
print('\nWrote:')
print(patient_out)
print(global_out)
