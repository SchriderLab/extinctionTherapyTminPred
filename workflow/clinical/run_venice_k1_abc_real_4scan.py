from __future__ import annotations
from rescue.runtime import workspace_path, allocated_threads
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
try:
    from scipy.stats import spearmanr, pearsonr
except Exception:
    spearmanr = None
    pearsonr = None
EPS = 1e-08
K = 25
SLOPE_WEIGHT = 2.0
LEVEL_COLS = ['y1', 'y2', 'y3']
SLOPE_COLS = ['dy01', 'dy12', 'dy23']
PARAM_COLS = ['target_decay_rate', 'base_mut_rate', 'dose_drug_1_mono', 'patient_turnover', 'standing_target', 'standing_realized', 'm1_effect_mean_target', 'm1_meanlog', 'm1_sdlog', 'single_age', 'n_founders']

def normalize_id(s):
    return pd.Series(s).astype(str).str.strip().str.replace('\\.0$', '', regex=True)

def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise RuntimeError('Could not identify required column among ' + ', '.join(candidates) + '\nAvailable columns:\n' + str(df.columns.tolist()))
    return None

def as_bool(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({'true', '1', 't', 'yes', 'y'})

def top_indices(distance, k):
    k = min(int(k), len(distance))
    idx = np.argpartition(distance, k - 1)[:k]
    return idx[np.argsort(distance[idx])]

def prediction_stats(y, d):
    y = np.asarray(y, float)
    d = np.asarray(d, float)
    w = 1.0 / (d + EPS)
    w /= w.sum()
    return {'pred_tmin_weighted': float(np.sum(w * y)), 'pred_tmin_mean': float(np.mean(y)), 'pred_tmin_median': float(np.median(y)), 'posterior_q025': float(np.quantile(y, 0.025)), 'posterior_q25': float(np.quantile(y, 0.25)), 'posterior_q75': float(np.quantile(y, 0.75)), 'posterior_q975': float(np.quantile(y, 0.975)), 'best_distance': float(np.min(d)), 'mean_distance_k': float(np.mean(d)), 'weights': w}

def summarize(df, label, reference_median):
    if len(df) == 0:
        return {'subset': label, 'n': 0}
    true = df['target_t'].to_numpy(float)
    pred = df['pred_tmin_weighted'].to_numpy(float)
    err = pred - true
    ae = np.abs(err)
    lo = df['posterior_q025'].to_numpy(float)
    hi = df['posterior_q975'].to_numpy(float)
    baseline_ae = np.abs(reference_median - true)
    row = {'subset': label, 'n': int(len(df)), 'ABC_bias': float(np.mean(err)), 'ABC_MAE': float(np.mean(ae)), 'ABC_median_AE': float(np.median(ae)), 'ABC_RMSE': float(np.sqrt(np.mean(err ** 2))), 'ABC_within_0p5': float(np.mean(ae <= 0.5)), 'ABC_within_1': float(np.mean(ae <= 1.0)), 'ABC_within_2': float(np.mean(ae <= 2.0)), 'ABC_coverage_95': float(np.mean((true >= lo) & (true <= hi))), 'ABC_median_ci95_width': float(np.median(hi - lo)), 'reference_median_prediction': float(reference_median), 'reference_median_MAE': float(np.mean(baseline_ae)), 'ABC_minus_reference_median_MAE': float(np.mean(ae) - np.mean(baseline_ae)), 'median_best_distance': float(df['best_distance'].median())}
    if spearmanr is not None and len(df) >= 3:
        sp = spearmanr(true, pred, nan_policy='omit')
        row['Spearman_rho'] = float(sp.statistic)
        row['Spearman_p'] = float(sp.pvalue)
    if pearsonr is not None and len(df) >= 3:
        pr = pearsonr(true, pred)
        row['Pearson_r'] = float(pr.statistic)
        row['Pearson_p'] = float(pr.pvalue)
    return row

def build_early_features(long_path: Path):
    long = pd.read_csv(long_path, low_memory=False)
    id_col = pick_col(long, ['USUBJID', 'id', 'patient_id', 'subject_id'])
    cycle_col = pick_col(long, ['cycle', 'ANLVISN', 'visit_cycle', 'target_t'])
    psa_col = pick_col(long, ['value', 'psa', 'PSA', 'LBSTRESN'])
    long = long.copy()
    long['id'] = normalize_id(long[id_col])
    long['cycle'] = pd.to_numeric(long[cycle_col], errors='coerce')
    long['psa'] = pd.to_numeric(long[psa_col], errors='coerce')
    long = long[long['cycle'].notna() & long['psa'].notna() & (long['psa'] > 0)].copy()
    adsl_path = long_path.parents[1] / 'venice_csv' / 'adsl.csv'
    adsl = pd.read_csv(adsl_path, low_memory=False)
    aid = pick_col(adsl, ['USUBJID', 'id', 'patient_id', 'subject_id'])
    if 'RANDOM' in adsl.columns:
        adsl = adsl[adsl['RANDOM'].astype(str).str.upper().eq('Y')].copy()
    if 'ARMCD' in adsl.columns:
        adsl = adsl[adsl['ARMCD'].astype(str).str.upper().eq('PLACEBO')].copy()
    base = pd.DataFrame({'id': normalize_id(adsl[aid]), 'cycle': 0.0, 'psa': pd.to_numeric(adsl['TMPSA'], errors='coerce')})
    base = base[base['psa'].notna() & (base['psa'] > 0)].drop_duplicates('id')
    long = pd.concat([base, long[~np.isclose(long['cycle'], 0.0, atol=1e-09)]], ignore_index=True)
    long = long.sort_values(['id', 'cycle']).drop_duplicates(['id', 'cycle'], keep='first')
    rows = []
    for pid, g in long.groupby('id', sort=False):
        m = {}
        for cyc in [0, 1, 2, 3]:
            q = g[np.isclose(g['cycle'], cyc, atol=1e-09)]
            if len(q) != 1:
                m = None
                break
            m[cyc] = float(q.iloc[0]['psa'])
        if m is None or m[0] <= 0:
            continue
        vals = np.array([m[0], m[1], m[2], m[3]], float)
        y = np.log((vals + 1e-12) / (vals[0] + 1e-12))
        rows.append({'id': pid, 'PSA0': vals[0], 'PSA1': vals[1], 'PSA2': vals[2], 'PSA3': vals[3], 'y0': float(y[0]), 'y1': float(y[1]), 'y2': float(y[2]), 'y3': float(y[3]), 'dy01': float(y[1] - y[0]), 'dy12': float(y[2] - y[1]), 'dy23': float(y[3] - y[2])})
    return pd.DataFrame(rows)

def get_cloud_ids(cloud_path: Path, matches_path: Path):
    cloud = pd.read_csv(cloud_path)
    matches = pd.read_csv(matches_path)
    if 'clinical_source_row' not in matches.columns:
        raise RuntimeError('Mechanistic-matches file must contain clinical_source_row.')
    if 'support_class' not in matches.columns:
        raise RuntimeError('Mechanistic-matches file must contain support_class.')
    id_col = pick_col(cloud, ['USUBJID', 'id', 'patient_id', 'subject_id', 'subject', 'patient'])
    cloud = cloud.copy()
    cloud['__id'] = normalize_id(cloud[id_col])
    m = matches.copy()
    m['clinical_source_row'] = pd.to_numeric(m['clinical_source_row'], errors='coerce')
    m = m[m['clinical_source_row'].notna()].copy()
    m['clinical_source_row'] = m['clinical_source_row'].astype(int)
    valid = m[(m['clinical_source_row'] >= 0) & (m['clinical_source_row'] < len(cloud))].copy()
    valid['id'] = [cloud.iloc[int(i)]['__id'] for i in valid['clinical_source_row']]
    keep_cols = ['id', 'clinical_match_id', 'clinical_source_row', 'support_class', 'nearest_distance', 'clinical_d', 'clinical_q', 'clinical_g']
    keep_cols = [c for c in keep_cols if c in valid.columns]
    return valid[keep_cols].drop_duplicates('id')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference-dir', type=Path, default=Path('venice_k1_abc_reference_10k'))
    ap.add_argument('--clinical-long', type=Path, default=Path('data/source_data/venice_psa_scheduled_long.csv'))
    ap.add_argument('--targets', type=Path, default=Path('data/source_data/venice_first_rebound_nadirs.csv'))
    ap.add_argument('--clinical-cloud', type=Path, default=Path('data/source_data/exponential_cloud/venice_exponential_cloud_per_patient.csv'))
    ap.add_argument('--mechanistic-matches', type=Path, default=Path('venice_heterogeneous_cloud_v1/patient_matching/venice_patient_mechanistic_matches.csv'))
    ap.add_argument('--outdir', type=Path, default=Path('venice_k1_abc_reference_10k/venice_real_4scan'))
    ap.add_argument('--k', type=int, default=K)
    ap.add_argument('--slope-weight', type=float, default=SLOPE_WEIGHT)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    ref_path = args.reference_dir / 'abc_reference_features_usable.csv'
    ref = pd.read_csv(ref_path)
    if 'tmin_boundary' in ref.columns:
        ref = ref[pd.to_numeric(ref['tmin_boundary'], errors='coerce') != 1].copy()
    ref = ref.dropna(subset=LEVEL_COLS + SLOPE_COLS + ['t_min_true']).reset_index(drop=True)
    if len(ref) < args.k:
        raise RuntimeError(f'Only {len(ref)} clean references; cannot use k={args.k}.')
    ref_level = ref[LEVEL_COLS].to_numpy(float)
    ref_slope = ref[SLOPE_COLS].to_numpy(float)
    ref_tmin = ref['t_min_true'].to_numpy(float)
    sd_level = float(np.std(ref_level.ravel()))
    sd_slope = float(np.std(ref_slope.ravel()))
    if not np.isfinite(sd_level) or sd_level <= 0:
        sd_level = 1.0
    if not np.isfinite(sd_slope) or sd_slope <= 0:
        sd_slope = 1.0
    reference_median = float(np.median(ref_tmin))
    print('Clean K=1 ABC references:', len(ref))
    print('Frozen k:', args.k)
    print('Frozen slope weight:', args.slope_weight)
    print('Reference tmin median:', reference_median)
    print('Reference tmin q05/median/q95:', np.quantile(ref_tmin, [0.05, 0.5, 0.95]))
    early = build_early_features(args.clinical_long)
    print('\nVENICE patients with complete positive cycles 0-3:', len(early))
    support = get_cloud_ids(args.clinical_cloud, args.mechanistic_matches)
    print('Clinical exponential-cloud patients mapped to IDs:', len(support))
    print(support['support_class'].value_counts(dropna=False).to_string())
    data = support.merge(early, on='id', how='inner')
    print('\nCloud patients with usable early PSA:', len(data))
    print('Original well-supported with usable early PSA:', int((data['support_class'] == 'well_supported').sum()))
    target = pd.read_csv(args.targets)
    tid = pick_col(target, ['USUBJID', 'id', 'patient_id', 'subject_id'])
    if 'target_t' not in target.columns:
        raise RuntimeError(f'Expected target file to contain target_t. Columns are: {target.columns.tolist()}')
    target = target.copy()
    target['id'] = normalize_id(target[tid])
    target['target_t'] = pd.to_numeric(target['target_t'], errors='coerce')
    target = target[target['target_t'].notna()][['id', 'target_t']].drop_duplicates('id')
    data = data.merge(target, on='id', how='left')
    query_level = data[LEVEL_COLS].to_numpy(float)
    query_slope = data[SLOPE_COLS].to_numpy(float)
    pred_rows = []
    neighbor_rows = []
    for i in range(len(data)):
        ql = query_level[i]
        qs = query_slope[i]
        level_dist = np.sqrt(np.mean(((ref_level - ql[None, :]) / sd_level) ** 2, axis=1))
        slope_dist = np.sqrt(np.mean(((ref_slope - qs[None, :]) / sd_slope) ** 2, axis=1))
        d = level_dist + args.slope_weight * slope_dist
        use = top_indices(d, args.k)
        stats = prediction_stats(ref_tmin[use], d[use])
        src = data.iloc[i]
        row = {'id': src['id'], 'support_class_full_trajectory': src['support_class'], 'clinical_match_id': src.get('clinical_match_id', np.nan), 'clinical_source_row': src.get('clinical_source_row', np.nan), 'clinical_d': src.get('clinical_d', np.nan), 'clinical_q': src.get('clinical_q', np.nan), 'clinical_g': src.get('clinical_g', np.nan), 'target_t': src.get('target_t', np.nan), 'PSA0': src['PSA0'], 'PSA1': src['PSA1'], 'PSA2': src['PSA2'], 'PSA3': src['PSA3'], 'y1': src['y1'], 'y2': src['y2'], 'y3': src['y3'], 'dy01': src['dy01'], 'dy12': src['dy12'], 'dy23': src['dy23'], 'k': int(args.k), 'slope_weight': float(args.slope_weight), 'pred_tmin_weighted': stats['pred_tmin_weighted'], 'pred_tmin_mean': stats['pred_tmin_mean'], 'pred_tmin_median': stats['pred_tmin_median'], 'pred_tmin_round': int(round(stats['pred_tmin_weighted'])), 'pred_tmin_ceil': int(math.ceil(stats['pred_tmin_weighted'])), 'posterior_q025': stats['posterior_q025'], 'posterior_q25': stats['posterior_q25'], 'posterior_q75': stats['posterior_q75'], 'posterior_q975': stats['posterior_q975'], 'posterior_ci95_width': stats['posterior_q975'] - stats['posterior_q025'], 'best_distance': stats['best_distance'], 'mean_distance_k': stats['mean_distance_k']}
        if np.isfinite(row['target_t']):
            row['error'] = row['pred_tmin_weighted'] - row['target_t']
            row['abs_error'] = abs(row['error'])
        else:
            row['error'] = np.nan
            row['abs_error'] = np.nan
        w = stats['weights']
        for c in PARAM_COLS:
            if c in ref.columns:
                vals = pd.to_numeric(ref.iloc[use][c], errors='coerce').to_numpy(float)
                ok = np.isfinite(vals)
                if ok.any():
                    ww = w[ok]
                    ww = ww / ww.sum()
                    row[f'{c}_posterior_weighted_mean'] = float(np.sum(ww * vals[ok]))
                    row[f'{c}_neighbor_q25'] = float(np.quantile(vals[ok], 0.25))
                    row[f'{c}_neighbor_q75'] = float(np.quantile(vals[ok], 0.75))
        pred_rows.append(row)
        for rank, (idx_ref, weight) in enumerate(zip(use, w), start=1):
            rr = ref.iloc[int(idx_ref)]
            nr = {'id': src['id'], 'support_class_full_trajectory': src['support_class'], 'rank': rank, 'ref_id': int(rr['ref_id']), 'distance': float(d[int(idx_ref)]), 'weight': float(weight), 'ref_tmin': float(rr['t_min_true'])}
            for c in PARAM_COLS:
                if c in rr.index:
                    nr[f'ref_{c}'] = rr[c]
            neighbor_rows.append(nr)
    pred = pd.DataFrame(pred_rows)
    neigh = pd.DataFrame(neighbor_rows)
    pred_path = args.outdir / 'venice_real_4scan_abc_predictions.csv'
    neigh_path = args.outdir / 'venice_real_4scan_abc_neighbors.csv'
    pred.to_csv(pred_path, index=False)
    neigh.to_csv(neigh_path, index=False)
    eval_df = pred[pred['target_t'].notna()].copy()
    summary_rows = []
    summary_rows.append(summarize(eval_df, 'all_231_cloud_patients_with_target', reference_median))
    summary_rows.append(summarize(eval_df[eval_df['support_class_full_trajectory'] == 'well_supported'], 'original_164_well_supported_with_target', reference_median))
    summary_rows.append(summarize(eval_df[eval_df['support_class_full_trajectory'].isin(['well_supported', 'edge_of_support'])], 'well_plus_edge_with_target', reference_median))
    summary = pd.DataFrame(summary_rows)
    summary_path = args.outdir / 'venice_real_4scan_abc_summary.csv'
    summary.to_csv(summary_path, index=False)
    support_rows = []
    for sc, g in eval_df.groupby('support_class_full_trajectory', dropna=False):
        support_rows.append(summarize(g, f'support_class={sc}', reference_median))
    support_summary = pd.DataFrame(support_rows)
    support_summary_path = args.outdir / 'venice_real_4scan_abc_by_support_class.csv'
    support_summary.to_csv(support_summary_path, index=False)
    print('\n=== VENICE REAL-PATIENT ABC SUMMARY ===')
    print(summary.to_string(index=False))
    print('\n=== BY FULL-TRAJECTORY SUPPORT CLASS ===')
    print(support_summary.to_string(index=False))
    print('\nImportant interpretation:')
    print('target_t is the observed first rebound-associated PSA nadir, not a directly observed biological tumor-burden t_min.')
    print('The original 164-patient subset was selected using full-trajectory d/q/g support, so its performance is exploratory rather than a strictly prospective subset-selection analysis.')
    print('\nWROTE:')
    print(pred_path)
    print(neigh_path)
    print(summary_path)
    print(support_summary_path)
if __name__ == '__main__':
    main()
