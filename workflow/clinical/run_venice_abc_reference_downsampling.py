from rescue.runtime import workspace_path, allocated_threads
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path('scripts').resolve()))
from run_venice_k1_abc_scan_budget import normalize_id, numeric, analytic_tmin, load_clinical_long, build_clinical_features, feature_arrays, scales_from_reference, batch_distances
SEED = 20260919
N_REPS = 20
LAST_CYCLE = 5
SCAN_COUNT = 6
K = 25
SLOPE_WEIGHT = 2.0
EPS = 1e-08
REQUESTED_SIZES = [500, 1000, 2000, 4000, 6000, 8000]
REF_DIR = Path('venice_k1_abc_reference_10k')
REF_META = REF_DIR / 'abc_reference_features_usable.csv'
SCAN_DIR = REF_DIR / 'venice_scan_budget_global_tmin_v1'
REF_SCAN = SCAN_DIR / 'reference_scan_features_cycles0_5.csv'
PRIOR_PRED = REF_DIR / 'venice_real_4scan' / 'venice_real_4scan_abc_predictions.csv'
CLINICAL_LONG = Path('data/source_data/venice_psa_scheduled_long.csv')
ADSL = Path('data/venice_csv/adsl.csv')
OUTDIR = REF_DIR / 'venice_abc_reference_downsampling_v1'
OUTDIR.mkdir(parents=True, exist_ok=True)

def evaluate(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - y
    ae = np.abs(err)
    rho = spearmanr(y, pred).statistic
    return {'bias': float(np.mean(err)), 'MAE': float(np.mean(ae)), 'median_AE': float(np.median(ae)), 'RMSE': float(np.sqrt(np.mean(err ** 2))), 'within_1': float(np.mean(ae <= 1.0)), 'within_2': float(np.mean(ae <= 2.0)), 'Spearman_rho': float(rho)}
ref_meta = pd.read_csv(REF_META)
ref_meta['ref_id'] = numeric(ref_meta['ref_id'])
ref_meta['t_min_true'] = numeric(ref_meta['t_min_true'])
if 'tmin_boundary' in ref_meta.columns:
    ref_meta = ref_meta[numeric(ref_meta['tmin_boundary']) != 1].copy()
ref_meta = ref_meta[ref_meta['ref_id'].notna() & ref_meta['t_min_true'].notna()].copy()
ref_meta['ref_id'] = ref_meta['ref_id'].astype(int)
ref_scan = pd.read_csv(REF_SCAN)
ref_scan['ref_id'] = numeric(ref_scan['ref_id'])
ref_scan = ref_scan[ref_scan['ref_id'].notna()].copy()
ref_scan['ref_id'] = ref_scan['ref_id'].astype(int)
ref = ref_scan.merge(ref_meta[['ref_id', 't_min_true']], on='ref_id', how='inner', suffixes=('', '_meta'))
if 't_min_true_meta' in ref.columns:
    ref['t_min_true'] = ref['t_min_true_meta']
    ref = ref.drop(columns=['t_min_true_meta'])
ref = ref[np.isfinite(ref['t_min_true'])].reset_index(drop=True)
print('Clean reference trajectories:', len(ref))
meta = pd.read_csv(PRIOR_PRED)
meta['id'] = meta['id'].map(normalize_id)
meta = meta[['id', 'support_class_full_trajectory', 'clinical_d', 'clinical_q', 'clinical_g']].drop_duplicates('id').copy()
for c in ['clinical_d', 'clinical_q', 'clinical_g']:
    meta[c] = numeric(meta[c])
meta['global_fit_tmin'] = analytic_tmin(meta['clinical_q'].to_numpy(float), meta['clinical_d'].to_numpy(float), meta['clinical_g'].to_numpy(float))
clinical_long = load_clinical_long(CLINICAL_LONG, ADSL)
clinical_features = build_clinical_features(clinical_long, LAST_CYCLE)
clinical = meta.merge(clinical_features, on='id', how='left')
needed = []
for i in range(1, LAST_CYCLE + 1):
    needed += [f'y{i}', f'dy{i - 1}{i}']
clinical = clinical[(clinical['support_class_full_trajectory'] == 'well_supported') & np.isfinite(clinical['global_fit_tmin']) & (clinical['global_fit_tmin'] > LAST_CYCLE)].dropna(subset=needed).reset_index(drop=True)
print('Fixed VENICE cohort:', len(clinical))
if len(clinical) != 133:
    print('WARNING: expected 133 patients')
ref_level, ref_slope, _, _ = feature_arrays(ref, LAST_CYCLE)
clinical_level, clinical_slope, _, _ = feature_arrays(clinical, LAST_CYCLE)
ref_y = ref['t_min_true'].to_numpy(float)
clinical_y = clinical['global_fit_tmin'].to_numpy(float)
sd_level, sd_slope = scales_from_reference(ref_level, ref_slope)
n_ref = len(ref)
sizes = [x for x in REQUESTED_SIZES if x < n_ref]
sizes.append(n_ref)
print('Reference sizes:', sizes)
print(f'Frozen ABC settings: k={K}, slope_weight={SLOPE_WEIGHT}')

def predict_from_subset(subset_idx):
    rlev = ref_level[subset_idx]
    rslo = ref_slope[subset_idx]
    ry = ref_y[subset_idx]
    level_d, slope_d = batch_distances(clinical_level, clinical_slope, rlev, rslo, sd_level, sd_slope)
    distance = level_d + np.float32(SLOPE_WEIGHT) * slope_d
    kk = min(K, len(subset_idx))
    idx = np.argpartition(distance, kth=kk - 1, axis=1)[:, :kk]
    rows = np.arange(len(clinical))[:, None]
    d = distance[rows, idx]
    y = ry[idx]
    weights = 1.0 / (d + EPS)
    pred = np.sum(weights * y, axis=1) / np.sum(weights, axis=1)
    return pred
results = []
for rep in range(N_REPS):
    rng = np.random.default_rng(SEED + rep)
    permutation = rng.permutation(n_ref)
    for size in sizes:
        if size == n_ref and rep > 0:
            continue
        if size == n_ref:
            subset_idx = np.arange(n_ref)
        else:
            subset_idx = permutation[:size]
        pred = predict_from_subset(subset_idx)
        m = evaluate(clinical_y, pred)
        results.append({'reference_size': size, 'fraction_full': size / n_ref, 'replicate': rep, 'n_patients': len(clinical), 'k': K, 'slope_weight': SLOPE_WEIGHT, **m})
        print(f"rep={rep:02d} Nref={size:5d} MAE={m['MAE']:.3f} medianAE={m['median_AE']:.3f} rho={m['Spearman_rho']:.3f}")
raw = pd.DataFrame(results)
summary_rows = []
for size, g in raw.groupby('reference_size', sort=True):
    summary_rows.append({'reference_size': int(size), 'fraction_full': float(g['fraction_full'].iloc[0]), 'replicates': len(g), 'MAE_mean': g['MAE'].mean(), 'MAE_sd': g['MAE'].std(ddof=1), 'MAE_q025': g['MAE'].quantile(0.025), 'MAE_q975': g['MAE'].quantile(0.975), 'median_AE_mean': g['median_AE'].mean(), 'Spearman_mean': g['Spearman_rho'].mean(), 'within_1_mean': g['within_1'].mean(), 'within_2_mean': g['within_2'].mean()})
summary = pd.DataFrame(summary_rows)
raw.to_csv(OUTDIR / 'downsampling_replicates.csv', index=False)
summary.to_csv(OUTDIR / 'downsampling_summary.csv', index=False)
print('\n' + '=' * 100)
print('REFERENCE-SIZE DOWNSAMPLING SUMMARY')
print(summary.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
full = raw[raw['reference_size'] == n_ref].iloc[0]
print('\nFull-reference sanity check:')
print(f"Nref={n_ref}, MAE={full['MAE']:.3f}, median AE={full['median_AE']:.3f}, rho={full['Spearman_rho']:.3f}")
print('Expected approximately: MAE 3.63, median AE 2.26, rho 0.43.')
print('\nWrote:')
print(OUTDIR / 'downsampling_replicates.csv')
print(OUTDIR / 'downsampling_summary.csv')
