from rescue.runtime import workspace_path, allocated_threads
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
sys.path.insert(0, str(Path('scripts').resolve()))
from run_venice_k1_abc_scan_budget import normalize_id, numeric, load_clinical_long
ROOT = Path('.')
FIT_FILE = ROOT / 'venice_k1_abc_reference_10k' / 'venice_real_4scan' / 'venice_real_4scan_abc_predictions.csv'
CLINICAL_LONG = ROOT / 'data' / 'source_data' / 'venice_psa_scheduled_long.csv'
ADSL = ROOT / 'data' / 'venice_csv' / 'adsl.csv'
OUTDIR = ROOT / 'venice_k1_abc_reference_10k' / 'exponential_nadir_vs_observed_psa'
OUTDIR.mkdir(parents=True, exist_ok=True)

def exponential_tmin(q, d, g):
    q = np.asarray(q, dtype=float)
    d = np.asarray(d, dtype=float)
    g = np.asarray(g, dtype=float)
    out = np.full(np.broadcast(q, d, g).shape, np.nan, dtype=float)
    valid = np.isfinite(q) & np.isfinite(d) & np.isfinite(g) & (q > 0.0) & (q < 1.0) & (d > 0.0) & (g > 0.0)
    ratio = np.full_like(out, np.nan, dtype=float)
    ratio[valid] = (1.0 - q[valid]) * d[valid] / (q[valid] * g[valid])
    valid = valid & np.isfinite(ratio) & (ratio > 0.0)
    out[valid] = np.log(ratio[valid]) / (d[valid] + g[valid])
    return out

def pick_column(df, candidates):
    lookup = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise RuntimeError(f'Could not find any of {candidates}. Available columns: {list(df.columns)}')

def summarize(df, label):
    x = df[np.isfinite(df['exp_tmin']) & np.isfinite(df['observed_nadir_cycle'])].copy()
    if len(x) == 0:
        return {'cohort': label, 'n': 0}
    err = x['exp_tmin'].to_numpy(float) - x['observed_nadir_cycle'].to_numpy(float)
    ae = np.abs(err)
    rho = spearmanr(x['observed_nadir_cycle'], x['exp_tmin']).statistic
    r = pearsonr(x['observed_nadir_cycle'], x['exp_tmin']).statistic
    rounded = np.rint(x['exp_tmin'].to_numpy(float))
    rounded_ae = np.abs(rounded - x['observed_nadir_cycle'].to_numpy(float))
    return {'cohort': label, 'n': len(x), 'bias': float(np.mean(err)), 'MAE': float(np.mean(ae)), 'median_AE': float(np.median(ae)), 'RMSE': float(np.sqrt(np.mean(err ** 2))), 'within_0p5': float(np.mean(ae <= 0.5)), 'within_1': float(np.mean(ae <= 1.0)), 'within_2': float(np.mean(ae <= 2.0)), 'Spearman_rho': float(rho), 'Pearson_r': float(r), 'rounded_exact': float(np.mean(rounded_ae == 0)), 'rounded_within_1': float(np.mean(rounded_ae <= 1))}
fit = pd.read_csv(FIT_FILE)
fit['id'] = fit['id'].map(normalize_id)
needed_fit = ['id', 'clinical_q', 'clinical_d', 'clinical_g']
if 'support_class_full_trajectory' in fit.columns:
    needed_fit.append('support_class_full_trajectory')
fit = fit[needed_fit].drop_duplicates('id').copy()
for c in ['clinical_q', 'clinical_d', 'clinical_g']:
    fit[c] = numeric(fit[c])
fit['exp_tmin'] = exponential_tmin(fit['clinical_q'].to_numpy(float), fit['clinical_d'].to_numpy(float), fit['clinical_g'].to_numpy(float))
print('Patients with q,d,g:', len(fit))
print('Finite exponential tmin:', int(np.isfinite(fit['exp_tmin']).sum()))
long = load_clinical_long(CLINICAL_LONG, ADSL)
print('\nClinical long columns:', list(long.columns))
id_col = pick_column(long, ['id', 'patient_id', 'usubjid'])
cycle_col = pick_column(long, ['cycle', 'cycle_num', 'cycle_number', 'visit_cycle', 'time'])
psa_col = pick_column(long, ['psa', 'psa_value', 'psa_raw', 'value', 'aval'])
psa = long[[id_col, cycle_col, psa_col]].copy()
psa.columns = ['id', 'cycle', 'psa']
psa['id'] = psa['id'].map(normalize_id)
psa['cycle'] = numeric(psa['cycle'])
psa['psa'] = numeric(psa['psa'])
psa = psa[psa['id'].notna() & np.isfinite(psa['cycle']) & np.isfinite(psa['psa']) & (psa['psa'] > 0)].copy()
psa = psa.groupby(['id', 'cycle'], as_index=False)['psa'].median()
rows = []
for pid, gdf in psa.groupby('id', sort=False):
    gdf = gdf.sort_values('cycle').reset_index(drop=True)
    if len(gdf) < 2:
        continue
    cycles = gdf['cycle'].to_numpy(float)
    values = gdf['psa'].to_numpy(float)
    min_psa = float(np.min(values))
    minimum_mask = np.isclose(values, min_psa, rtol=1e-10, atol=max(1e-12, abs(min_psa) * 1e-12))
    minimum_cycles = cycles[minimum_mask]
    observed_nadir_cycle = float(np.min(minimum_cycles))
    last_cycle = float(np.max(cycles))
    observed_nadir_at_boundary = bool(np.any(np.isclose(minimum_cycles, last_cycle)))
    rows.append({'id': pid, 'n_psa_observations': len(gdf), 'first_cycle': float(np.min(cycles)), 'last_cycle': last_cycle, 'observed_nadir_cycle': observed_nadir_cycle, 'observed_nadir_psa': min_psa, 'observed_nadir_at_boundary': observed_nadir_at_boundary})
observed = pd.DataFrame(rows)
print('\nPatients with observed PSA trajectories:', len(observed))
print('Observed nadir at final measurement:', int(observed['observed_nadir_at_boundary'].sum()))
merged = fit.merge(observed, on='id', how='inner')
merged['error'] = merged['exp_tmin'] - merged['observed_nadir_cycle']
merged['abs_error'] = np.abs(merged['error'])
merged['valid_nonboundary'] = np.isfinite(merged['exp_tmin']) & np.isfinite(merged['observed_nadir_cycle']) & (merged['exp_tmin'] >= 0) & ~merged['observed_nadir_at_boundary']
valid = merged[merged['valid_nonboundary']].copy()
summary_rows = []
summary_rows.append(summarize(valid, 'all_nonboundary'))
if 'support_class_full_trajectory' in valid.columns:
    well = valid[valid['support_class_full_trajectory'] == 'well_supported']
    summary_rows.append(summarize(well, 'well_supported'))
    well_edge = valid[valid['support_class_full_trajectory'].isin(['well_supported', 'edge_supported', 'edge_of_support', 'edge'])]
    summary_rows.append(summarize(well_edge, 'well_plus_edge'))
summary = pd.DataFrame(summary_rows)
merged.to_csv(OUTDIR / 'exponential_vs_observed_psa_patient_level.csv', index=False)
summary.to_csv(OUTDIR / 'exponential_vs_observed_psa_summary.csv', index=False)
print('\n' + '=' * 110)
print('EXPONENTIAL NADIR vs ACTUAL OBSERVED GLOBAL PSA NADIR')
print('Boundary minima at the final observed PSA measurement are excluded.')
print(summary.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print('\nClosest 10:')
print(valid[['id', 'observed_nadir_cycle', 'exp_tmin', 'abs_error']].sort_values('abs_error').head(10).to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print('\nWorst 10:')
print(valid[['id', 'observed_nadir_cycle', 'exp_tmin', 'abs_error']].sort_values('abs_error', ascending=False).head(10).to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print('\nWrote:')
print(OUTDIR / 'exponential_vs_observed_psa_patient_level.csv')
print(OUTDIR / 'exponential_vs_observed_psa_summary.csv')
