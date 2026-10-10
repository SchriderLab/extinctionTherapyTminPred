"""
VENICE K=1 ABC scan-budget experiment.

Primary question
----------------
How does prediction of the full-trajectory biexponential global t_min change
as we increase the number of early PSA scans?

Default budgets:
    2 scans = baseline + cycle 1
    3 scans = baseline + cycles 1-2
    4 scans = baseline + cycles 1-3
    5 scans = baseline + cycles 1-4
    6 scans = baseline + cycles 1-5

The default stops at 6 total scans because the existing VENICE diagnostics
show a much sharper loss of subjects by cycle 6. You can extend to 7 total
scans with --max-scans 7.

Important design choices
------------------------
1. The target is the analytic minimum of the FULL-TRAJECTORY biexponential fit:
       t_min = log(((1-q)d)/(qg)) / (d+g)
2. A scan-budget result is "prospective" only when fitted global t_min occurs
   after the last scan used. These future-only results are the primary output.
3. ABC hyperparameters are re-selected separately for every scan budget using
   REFERENCE-ONLY cross-validation. VENICE never tunes k or slope weight.
4. We report both:
   - per-budget eligible cohort: maximum usable data for that budget
   - fixed common cohort: patients eligible at the largest budget, so changes
     across budgets are not caused by changing patient composition.
5. The script validates reconstructed reference trajectories against the
   existing frozen 4-scan reference features before doing any ABC analysis.
"""
from __future__ import annotations
from rescue.runtime import workspace_path, allocated_threads
import argparse
import math
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from scipy.stats import spearmanr, pearsonr
except Exception:
    spearmanr = None
    pearsonr = None
from sklearn.model_selection import KFold
EPS = 1e-08
K_GRID = [10, 25, 50, 100, 200, 400]
SLOPE_GRID = [0.0, 0.25, 0.5, 1.0, 2.0]
CV_FOLDS = 5
CV_SEED = 20260910
CHUNK = 128
DEFAULT_REF_DIR = Path('venice_k1_abc_reference_10k')
DEFAULT_REF_FEATURES = DEFAULT_REF_DIR / 'abc_reference_features_usable.csv'
DEFAULT_CLINICAL_LONG = Path('data/source_data/venice_psa_scheduled_long.csv')
DEFAULT_ADSL = Path('data/venice_csv/adsl.csv')
DEFAULT_PRIOR_PRED = DEFAULT_REF_DIR / 'venice_real_4scan' / 'venice_real_4scan_abc_predictions.csv'
DEFAULT_OUTDIR = DEFAULT_REF_DIR / 'venice_scan_budget_global_tmin_v1'

def normalize_id(x):
    s = str(x).strip()
    return re.sub('\\.0$', '', s)

def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise RuntimeError('Could not identify required column among ' + ', '.join(candidates) + '\nAvailable columns:\n' + str(df.columns.tolist()))
    return None

def numeric(s):
    return pd.to_numeric(s, errors='coerce')

def weighted_prediction(y, d):
    y = np.asarray(y, float)
    d = np.asarray(d, float)
    w = 1.0 / (d + EPS)
    w /= w.sum()
    return float(np.sum(w * y))

def weighted_quantiles(values, weights, probs=(0.025, 0.25, 0.5, 0.75, 0.975)):
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    weights = weights / weights.sum()
    cdf = np.cumsum(weights)
    out = []
    for p in probs:
        out.append(float(values[np.searchsorted(cdf, p, side='left')]))
    return out

def analytic_tmin(q, d, g):
    q = np.asarray(q, float)
    d = np.asarray(d, float)
    g = np.asarray(g, float)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return np.log((1.0 - q) * d / (q * g)) / (d + g)

def metrics(true, pred):
    true = np.asarray(true, float)
    pred = np.asarray(pred, float)
    ok = np.isfinite(true) & np.isfinite(pred)
    true = true[ok]
    pred = pred[ok]
    if len(true) == 0:
        return {'n': 0, 'bias': np.nan, 'MAE': np.nan, 'median_AE': np.nan, 'RMSE': np.nan, 'within_0p5': np.nan, 'within_1': np.nan, 'within_2': np.nan, 'Spearman_rho': np.nan, 'Spearman_p': np.nan, 'Pearson_r': np.nan, 'Pearson_p': np.nan}
    err = pred - true
    ae = np.abs(err)
    row = {'n': int(len(true)), 'bias': float(np.mean(err)), 'MAE': float(np.mean(ae)), 'median_AE': float(np.median(ae)), 'RMSE': float(np.sqrt(np.mean(err ** 2))), 'within_0p5': float(np.mean(ae <= 0.5)), 'within_1': float(np.mean(ae <= 1.0)), 'within_2': float(np.mean(ae <= 2.0)), 'Spearman_rho': np.nan, 'Spearman_p': np.nan, 'Pearson_r': np.nan, 'Pearson_p': np.nan}
    if len(true) >= 3 and spearmanr is not None:
        x = spearmanr(true, pred)
        row['Spearman_rho'] = float(x.statistic)
        row['Spearman_p'] = float(x.pvalue)
    if len(true) >= 3 and pearsonr is not None:
        x = pearsonr(true, pred)
        row['Pearson_r'] = float(x.statistic)
        row['Pearson_p'] = float(x.pvalue)
    return row

def load_clinical_long(long_path: Path, adsl_path: Path):
    long = pd.read_csv(long_path, low_memory=False)
    adsl = pd.read_csv(adsl_path, low_memory=False)
    id_col = pick_col(long, ['USUBJID', 'id', 'patient_id', 'subject_id'])
    cycle_col = pick_col(long, ['cycle', 'ANLVISN', 'visit_cycle', 'target_t'])
    psa_col = pick_col(long, ['value', 'psa', 'PSA', 'LBSTRESN'])
    long = long.copy()
    long['id'] = long[id_col].map(normalize_id)
    long['cycle'] = numeric(long[cycle_col])
    long['psa'] = numeric(long[psa_col])
    long = long[long['cycle'].notna() & long['psa'].notna() & (long['psa'] > 0)].copy()
    aid = pick_col(adsl, ['USUBJID', 'id', 'patient_id', 'subject_id'])
    if 'RANDOM' in adsl.columns:
        adsl = adsl[adsl['RANDOM'].astype(str).str.upper().eq('Y')].copy()
    if 'ARMCD' in adsl.columns:
        adsl = adsl[adsl['ARMCD'].astype(str).str.upper().eq('PLACEBO')].copy()
    if 'TMPSA' not in adsl.columns:
        raise RuntimeError('ADSL does not contain TMPSA baseline PSA.')
    base = pd.DataFrame({'id': adsl[aid].map(normalize_id), 'cycle': 0.0, 'psa': numeric(adsl['TMPSA'])})
    base = base[base['psa'].notna() & (base['psa'] > 0)].drop_duplicates('id')
    long_nonzero = long[~np.isclose(long['cycle'], 0.0, atol=1e-09)][['id', 'cycle', 'psa']].copy()
    out = pd.concat([base[['id', 'cycle', 'psa']], long_nonzero], ignore_index=True)
    out = out.sort_values(['id', 'cycle']).drop_duplicates(['id', 'cycle'], keep='first').reset_index(drop=True)
    return out

def build_clinical_features(clin_long, max_last_cycle):
    rows = []
    for pid, g in clin_long.groupby('id', sort=False):
        vals = {}
        complete = True
        for cyc in range(max_last_cycle + 1):
            q = g[np.isclose(g['cycle'], cyc, atol=1e-09)]
            if len(q) != 1:
                complete = False
                break
            vals[cyc] = float(q.iloc[0]['psa'])
        if 0 not in vals:
            continue
        max_have = -1
        vals = {}
        for cyc in range(max_last_cycle + 1):
            q = g[np.isclose(g['cycle'], cyc, atol=1e-09)]
            if len(q) != 1:
                break
            vals[cyc] = float(q.iloc[0]['psa'])
            max_have = cyc
        if max_have < 1:
            continue
        arr = np.array([vals[i] for i in range(max_have + 1)], float)
        y = np.log((arr + 1e-12) / (arr[0] + 1e-12))
        row = {'id': pid, 'max_consecutive_cycle': int(max_have), 'PSA0': float(arr[0])}
        for j in range(1, max_have + 1):
            row[f'y{j}'] = float(y[j])
            row[f'dy{j - 1}{j}'] = float(y[j] - y[j - 1])
        rows.append(row)
    return pd.DataFrame(rows)

def ref_id_from_filename(path: Path):
    nums = re.findall('\\d+', path.stem)
    if not nums:
        return None
    return int(nums[-1])

def detect_trajectory_columns(cols):
    cols = list(cols)
    id_col = None
    for c in ['ref_id', 'reference_id', 'sim_id', 'simulation_id', 'patient_id', 'id']:
        if c in cols:
            id_col = c
            break
    cycle_col = None
    for c in ['drug1_cycle', 'drug_1_cycle', 'treatment_cycle', 'cycle', 'time', 't']:
        if c in cols:
            cycle_col = c
            break
    pop_col = None
    for c in ['N', 'n', 'population', 'population_size', 'pop_size', 'total_N', 'N_total']:
        if c in cols:
            pop_col = c
            break
    phase_col = None
    for c in ['phase', 'treatment_phase', 'drug_phase']:
        if c in cols:
            phase_col = c
            break
    return (id_col, cycle_col, pop_col, phase_col)

def normalize_reference_cycles(df, cycle_col, phase_col):
    x = df.copy()
    if phase_col is not None:
        p = x[phase_col].astype(str).str.lower()
        mask = p.str.contains('drug1', regex=False) | p.str.contains('drug_1', regex=False) | p.str.contains('mono', regex=False) | p.str.contains('treat', regex=False)
        if mask.any():
            x = x[mask].copy()
    x['__cycle'] = numeric(x[cycle_col])
    x = x[x['__cycle'].notna()].copy()
    if len(x) == 0:
        return x
    vals = np.sort(x['__cycle'].dropna().unique())
    if len(vals) and vals[0] > 20:
        x['__cycle'] = x['__cycle'] - vals[0]
    return x

def load_reference_from_feature_columns(ref_features, max_last_cycle):
    """
    If the frozen feature file already contains enough y-columns, use them.
    """
    needed = [f'y{i}' for i in range(1, max_last_cycle + 1)]
    if all((c in ref_features.columns for c in needed)):
        out = ref_features[['ref_id', 't_min_true'] + needed].copy()
        for i in range(1, max_last_cycle + 1):
            out[f'dy{i - 1}{i}'] = out[f'y{i}'] - (0.0 if i == 1 else out[f'y{i - 1}'])
        return (out, 'frozen feature CSV')
    ncols = [f'N{i}' for i in range(max_last_cycle + 1)]
    if all((c in ref_features.columns for c in ncols)):
        out = ref_features[['ref_id', 't_min_true'] + ncols].copy()
        n0 = numeric(out['N0'])
        for i in range(1, max_last_cycle + 1):
            ni = numeric(out[f'N{i}'])
            out[f'y{i}'] = np.log((ni + 1e-12) / (n0 + 1e-12))
            out[f'dy{i - 1}{i}'] = out[f'y{i}'] - (0.0 if i == 1 else out[f'y{i - 1}'])
        keep = ['ref_id', 't_min_true'] + [f'y{i}' for i in range(1, max_last_cycle + 1)] + [f'dy{i - 1}{i}' for i in range(1, max_last_cycle + 1)]
        return (out[keep], 'N columns in frozen feature CSV')
    return (None, None)

def discover_reference_trajectory_csvs(ref_dir: Path):
    files = []
    for p in ref_dir.rglob('*.csv'):
        name = p.name.lower()
        bad = ['manifest', 'feature', 'summary', 'validation', 'prediction', 'neighbor', 'match', 'cv_', 'scan_budget']
        if any((x in name for x in bad)):
            continue
        files.append(p)
    return files

def reconstruct_reference_features(ref_dir, ref_features, max_last_cycle, cache_path):
    if cache_path.exists():
        print(f'Using cached reference scan features: {cache_path}')
        cached = pd.read_csv(cache_path)
        needed = [f'y{i}' for i in range(1, max_last_cycle + 1)]
        if all((c in cached.columns for c in needed)):
            return cached
    direct, source = load_reference_from_feature_columns(ref_features, max_last_cycle)
    if direct is not None:
        print(f'Reference scan features recovered from {source}.')
        direct.to_csv(cache_path, index=False)
        return direct
    print('Frozen feature CSV does not contain scans beyond cycle 3.')
    print('Searching raw K=1 reference trajectory CSVs...')
    candidates = discover_reference_trajectory_csvs(ref_dir)
    if not candidates:
        raise RuntimeError(f'No candidate raw reference CSVs found under {ref_dir}.')
    long_parts = []
    n_examined = 0
    n_used = 0
    for path in candidates:
        n_examined += 1
        try:
            head = pd.read_csv(path, nrows=10)
        except Exception:
            continue
        id_col, cycle_col, pop_col, phase_col = detect_trajectory_columns(head.columns)
        if cycle_col is None or pop_col is None:
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        df = normalize_reference_cycles(df, cycle_col, phase_col)
        if len(df) == 0:
            continue
        df['__N'] = numeric(df[pop_col])
        df = df[df['__N'].notna() & (df['__N'] > 0) & df['__cycle'].notna()].copy()
        if len(df) == 0:
            continue
        if id_col is not None:
            df['__ref_id'] = numeric(df[id_col])
        else:
            rid = ref_id_from_filename(path)
            if rid is None:
                continue
            df['__ref_id'] = rid
        df = df[df['__ref_id'].notna()].copy()
        df = df[(df['__cycle'] >= 0) & (df['__cycle'] <= max_last_cycle)][['__ref_id', '__cycle', '__N']].copy()
        if len(df) == 0:
            continue
        long_parts.append(df)
        n_used += 1
    if not long_parts:
        raise RuntimeError('Found CSV files, but none contained recognizable reference trajectory columns. Run:\nfind venice_k1_abc_reference_10k -maxdepth 3 -type f | head -100\nand send me the output so I can patch the loader.')
    print(f'Reference CSV discovery: examined={n_examined}, trajectory_files_used={n_used}')
    ref_long = pd.concat(long_parts, ignore_index=True)
    ref_long['__ref_id'] = ref_long['__ref_id'].astype(int)
    ref_long = ref_long.sort_values(['__ref_id', '__cycle']).drop_duplicates(['__ref_id', '__cycle'], keep='first')
    wide = ref_long.pivot(index='__ref_id', columns='__cycle', values='__N')
    wide.columns = [int(round(float(c))) for c in wide.columns]
    need_cycles = list(range(max_last_cycle + 1))
    missing_cols = [c for c in need_cycles if c not in wide.columns]
    if missing_cols:
        raise RuntimeError(f'Raw reference logs were found but these treatment cycles are missing: {missing_cols}. Detected cycle columns: {sorted(wide.columns.tolist())[:30]}')
    wide = wide[need_cycles].dropna().copy()
    wide.index.name = 'ref_id'
    wide = wide.reset_index()
    n0 = wide[0].to_numpy(float)
    out = wide[['ref_id']].copy()
    for i in range(1, max_last_cycle + 1):
        out[f'y{i}'] = np.log((wide[i].to_numpy(float) + 1e-12) / (n0 + 1e-12))
        out[f'dy{i - 1}{i}'] = out[f'y{i}'] - (0.0 if i == 1 else out[f'y{i - 1}'])
    out = out.merge(ref_features[['ref_id', 't_min_true']], on='ref_id', how='inner')
    overlap_cols = [c for c in ['y1', 'y2', 'y3'] if c in ref_features.columns]
    if overlap_cols:
        check = out[['ref_id'] + overlap_cols].merge(ref_features[['ref_id'] + overlap_cols], on='ref_id', suffixes=('_reconstructed', '_frozen'))
        diffs = []
        for c in overlap_cols:
            diffs.extend(np.abs(check[f'{c}_reconstructed'].to_numpy(float) - check[f'{c}_frozen'].to_numpy(float)).tolist())
        med = float(np.nanmedian(diffs))
        q95 = float(np.nanquantile(diffs, 0.95))
        print(f'Reference reconstruction check against frozen 4-scan features: median |delta|={med:.6g}, q95={q95:.6g}')
        if not np.isfinite(med) or med > 0.02 or q95 > 0.1:
            raise RuntimeError('Reconstructed reference cycles do not match the frozen 4-scan features closely enough. Stop here rather than run a misaligned scan-budget experiment.')
    out.to_csv(cache_path, index=False)
    print(f'Wrote reference scan-feature cache: {cache_path}')
    return out

def feature_arrays(df, last_cycle):
    level_cols = [f'y{i}' for i in range(1, last_cycle + 1)]
    slope_cols = [f'dy{i - 1}{i}' for i in range(1, last_cycle + 1)]
    xlev = df[level_cols].to_numpy(np.float32)
    xslo = df[slope_cols].to_numpy(np.float32)
    return (xlev, xslo, level_cols, slope_cols)

def scales_from_reference(xlev, xslo):
    sd_level = float(np.std(xlev.ravel()))
    sd_slope = float(np.std(xslo.ravel()))
    if not np.isfinite(sd_level) or sd_level <= 0:
        sd_level = 1.0
    if not np.isfinite(sd_slope) or sd_slope <= 0:
        sd_slope = 1.0
    return (sd_level, sd_slope)

def batch_distances(qlev, qslo, rlev, rslo, sd_level, sd_slope):
    level = np.sqrt(np.mean(((rlev[None, :, :] - qlev[:, None, :]) / sd_level) ** 2, axis=2)).astype(np.float32)
    slope = np.sqrt(np.mean(((rslo[None, :, :] - qslo[:, None, :]) / sd_slope) ** 2, axis=2)).astype(np.float32)
    return (level, slope)

def cv_tune_budget(ref_df, last_cycle, cv_max_queries=None):
    xlev, xslo, _, _ = feature_arrays(ref_df, last_cycle)
    y = ref_df['t_min_true'].to_numpy(float)
    sd_level, sd_slope = scales_from_reference(xlev, xslo)
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    fold_ids = np.full(len(ref_df), -1, int)
    splits = list(kf.split(np.arange(len(ref_df))))
    for f, (_, va) in enumerate(splits):
        fold_ids[va] = f
    rng = np.random.default_rng(CV_SEED + last_cycle)
    if cv_max_queries is None or cv_max_queries <= 0 or cv_max_queries >= len(ref_df):
        selected_queries = set(range(len(ref_df)))
    else:
        chosen = rng.choice(np.arange(len(ref_df)), size=int(cv_max_queries), replace=False)
        selected_queries = set((int(x) for x in chosen))
    collectors = {(sw, k): {'true': [], 'pred': []} for sw in SLOPE_GRID for k in K_GRID}
    for fold, (train_idx, valid_idx_all) in enumerate(splits):
        valid_idx = np.array([i for i in valid_idx_all if int(i) in selected_queries], dtype=int)
        if len(valid_idx) == 0:
            continue
        rlev = xlev[train_idx]
        rslo = xslo[train_idx]
        ry = y[train_idx]
        for start in range(0, len(valid_idx), CHUNK):
            qidx = valid_idx[start:start + CHUNK]
            qlev = xlev[qidx]
            qslo = xslo[qidx]
            lev_d, slo_d = batch_distances(qlev, qslo, rlev, rslo, sd_level, sd_slope)
            for sw in SLOPE_GRID:
                d = lev_d + np.float32(sw) * slo_d
                max_k = min(max(K_GRID), d.shape[1])
                part = np.argpartition(d, kth=max_k - 1, axis=1)[:, :max_k]
                row = np.arange(len(qidx))[:, None]
                part_d = d[row, part]
                order = np.argsort(part_d, axis=1)
                neigh = part[row, order]
                neigh_d = part_d[row, order]
                for k in K_GRID:
                    kk = min(k, max_k)
                    ni = neigh[:, :kk]
                    nd = neigh_d[:, :kk]
                    vals = ry[ni]
                    w = 1.0 / (nd.astype(float) + EPS)
                    w /= w.sum(axis=1, keepdims=True)
                    p = np.sum(w * vals, axis=1)
                    collectors[sw, k]['true'].extend(y[qidx].tolist())
                    collectors[sw, k]['pred'].extend(p.tolist())
    rows = []
    for (sw, k), z in collectors.items():
        m = metrics(z['true'], z['pred'])
        rows.append({'last_cycle': int(last_cycle), 'scan_count': int(last_cycle + 1), 'slope_weight': float(sw), 'k': int(k), 'sd_level': sd_level, 'sd_slope': sd_slope, **m})
    cv = pd.DataFrame(rows)
    cv = cv.sort_values(['MAE', 'median_AE', 'RMSE', 'k', 'slope_weight'], ascending=[True, True, True, True, True]).reset_index(drop=True)
    return (cv, sd_level, sd_slope)

def abc_predict_queries(query_df, ref_df, last_cycle, k, slope_weight, sd_level, sd_slope):
    qlev, qslo, _, _ = feature_arrays(query_df, last_cycle)
    rlev, rslo, _, _ = feature_arrays(ref_df, last_cycle)
    ry = ref_df['t_min_true'].to_numpy(float)
    ref_ids = ref_df['ref_id'].to_numpy(int)
    rows = []
    for start in range(0, len(query_df), CHUNK):
        qpart = query_df.iloc[start:start + CHUNK].copy()
        alev = qlev[start:start + CHUNK]
        aslo = qslo[start:start + CHUNK]
        lev_d, slo_d = batch_distances(alev, aslo, rlev, rslo, sd_level, sd_slope)
        d = lev_d + np.float32(slope_weight) * slo_d
        kk = min(int(k), len(ref_df))
        part = np.argpartition(d, kth=kk - 1, axis=1)[:, :kk]
        rr = np.arange(len(qpart))[:, None]
        part_d = d[rr, part]
        order = np.argsort(part_d, axis=1)
        neigh = part[rr, order]
        neigh_d = part_d[rr, order]
        for j, (_, src) in enumerate(qpart.iterrows()):
            ni = neigh[j]
            nd = neigh_d[j].astype(float)
            vals = ry[ni]
            w = 1.0 / (nd + EPS)
            w /= w.sum()
            pred = float(np.sum(w * vals))
            q025, q25, q50, q75, q975 = weighted_quantiles(vals, w)
            rows.append({'id': src['id'], 'scan_count': int(last_cycle + 1), 'last_cycle': int(last_cycle), 'k': int(k), 'slope_weight': float(slope_weight), 'pred_tmin': pred, 'posterior_q025': q025, 'posterior_q25': q25, 'posterior_q50': q50, 'posterior_q75': q75, 'posterior_q975': q975, 'posterior_ci95_width': q975 - q025, 'best_distance': float(nd[0]), 'mean_distance_k': float(np.mean(nd)), 'neighbor_tmin_sd': float(np.std(vals)), 'neighbor_tmin_iqr': float(np.quantile(vals, 0.75) - np.quantile(vals, 0.25)), 'neighbor_min_tmin': float(np.min(vals)), 'neighbor_max_tmin': float(np.max(vals)), 'nearest_ref_id': int(ref_ids[ni[0]])})
    return pd.DataFrame(rows)

def summarize_prediction_set(df, label, ref_median):
    if len(df) == 0:
        return {'subset': label, 'n': 0}
    m = metrics(df['global_fit_tmin'], df['pred_tmin'])
    base = np.abs(df['global_fit_tmin'].to_numpy(float) - ref_median)
    return {'subset': label, **m, 'reference_median_prediction': float(ref_median), 'reference_median_MAE': float(np.mean(base)), 'ABC_minus_reference_median_MAE': float(m['MAE']) - float(np.mean(base)), 'median_best_distance': float(df['best_distance'].median()), 'median_neighbor_tmin_sd': float(df['neighbor_tmin_sd'].median()), 'median_ci95_width': float(df['posterior_ci95_width'].median())}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference-dir', type=Path, default=DEFAULT_REF_DIR)
    ap.add_argument('--reference-features', type=Path, default=DEFAULT_REF_FEATURES)
    ap.add_argument('--clinical-long', type=Path, default=DEFAULT_CLINICAL_LONG)
    ap.add_argument('--adsl', type=Path, default=DEFAULT_ADSL)
    ap.add_argument('--prior-predictions', type=Path, default=DEFAULT_PRIOR_PRED, help='Existing 4-scan prediction CSV used only to recover the frozen 231-patient support labels and fitted d/q/g.')
    ap.add_argument('--outdir', type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument('--min-scans', type=int, default=2)
    ap.add_argument('--max-scans', type=int, default=6, help='Default 6 = baseline through cycle 5. Use 7 to include cycle 6 sensitivity analysis.')
    ap.add_argument('--cv-max-queries', type=int, default=0, help='0 = full reference-only 5-fold CV. For a quick smoke test use e.g. 2000.')
    args = ap.parse_args()
    if args.min_scans < 2:
        raise ValueError('--min-scans must be >=2')
    if args.max_scans < args.min_scans:
        raise ValueError('--max-scans must be >= --min-scans')
    args.outdir.mkdir(parents=True, exist_ok=True)
    max_last_cycle = args.max_scans - 1
    ref0 = pd.read_csv(args.reference_features)
    if 'ref_id' not in ref0.columns or 't_min_true' not in ref0.columns:
        raise RuntimeError('Frozen reference feature CSV must contain ref_id and t_min_true.')
    ref0['ref_id'] = numeric(ref0['ref_id'])
    ref0['t_min_true'] = numeric(ref0['t_min_true'])
    if 'tmin_boundary' in ref0.columns:
        ref0 = ref0[numeric(ref0['tmin_boundary']) != 1].copy()
    ref0 = ref0[ref0['ref_id'].notna() & ref0['t_min_true'].notna()].copy()
    ref0['ref_id'] = ref0['ref_id'].astype(int)
    print('Frozen clean reference rows:', len(ref0))
    cache = args.outdir / f'reference_scan_features_cycles0_{max_last_cycle}.csv'
    refscan = reconstruct_reference_features(args.reference_dir, ref0, max_last_cycle, cache)
    refscan = refscan.merge(ref0[['ref_id']], on='ref_id', how='inner')
    refscan['t_min_true'] = numeric(refscan['t_min_true'])
    print(f'Reference rows with complete scans through cycle {max_last_cycle}: {len(refscan)}')
    meta = pd.read_csv(args.prior_predictions)
    meta['id'] = meta['id'].map(normalize_id)
    needed_meta = ['id', 'support_class_full_trajectory', 'clinical_d', 'clinical_q', 'clinical_g']
    for c in needed_meta:
        if c not in meta.columns:
            raise RuntimeError(f'Missing {c} from {args.prior_predictions}')
    meta = meta[needed_meta].drop_duplicates('id').copy()
    meta['clinical_d'] = numeric(meta['clinical_d'])
    meta['clinical_q'] = numeric(meta['clinical_q'])
    meta['clinical_g'] = numeric(meta['clinical_g'])
    meta['global_fit_tmin'] = analytic_tmin(meta['clinical_q'].to_numpy(float), meta['clinical_d'].to_numpy(float), meta['clinical_g'].to_numpy(float))
    meta['global_tmin_valid'] = np.isfinite(meta['global_fit_tmin']) & (meta['global_fit_tmin'] >= 0)
    print('\nFrozen support cohort:', len(meta))
    print(meta['support_class_full_trajectory'].value_counts(dropna=False).to_string())
    print('Valid finite nonnegative global fit tmin:', int(meta['global_tmin_valid'].sum()))
    invalid_global = meta[~meta['global_tmin_valid']].copy()
    invalid_path = args.outdir / 'invalid_global_tmin_patients.csv'
    invalid_global.to_csv(invalid_path, index=False)
    print('Invalid global tmin by support class:\n' + invalid_global['support_class_full_trajectory'].value_counts(dropna=False).to_string())
    clin_long = load_clinical_long(args.clinical_long, args.adsl)
    clin_feat = build_clinical_features(clin_long, max_last_cycle)
    clinical = meta.merge(clin_feat, on='id', how='left')
    cv_all = []
    pred_all = []
    retention_rows = []
    summary_rows = []
    for scan_count in range(args.min_scans, args.max_scans + 1):
        last_cycle = scan_count - 1
        level_cols = [f'y{i}' for i in range(1, last_cycle + 1)]
        slope_cols = [f'dy{i - 1}{i}' for i in range(1, last_cycle + 1)]
        ref_budget = refscan.dropna(subset=level_cols + slope_cols + ['t_min_true']).copy()
        clin_budget = clinical.dropna(subset=level_cols + slope_cols).copy()
        eligible = clin_budget[clin_budget['global_tmin_valid'] & (clin_budget['global_fit_tmin'] > last_cycle)].copy()
        print('\n' + '=' * 72)
        print(f'{scan_count} scans: baseline through cycle {last_cycle}')
        print('Reference rows:', len(ref_budget))
        print('Clinical with complete scans:', len(clin_budget))
        print('Clinical future-global-tmin eligible:', len(eligible))
        retention_rows.append({'scan_count': scan_count, 'last_cycle': last_cycle, 'clinical_231_total': len(meta), 'valid_global_tmin_total': int(meta['global_tmin_valid'].sum()), 'complete_scans_all_support': len(clin_budget), 'future_global_tmin_all_support': len(eligible), 'future_global_tmin_well_supported': int((eligible['support_class_full_trajectory'] == 'well_supported').sum()), 'future_global_tmin_well_plus_edge': int(eligible['support_class_full_trajectory'].isin(['well_supported', 'edge_of_support']).sum())})
        cv, sd_level, sd_slope = cv_tune_budget(ref_budget, last_cycle, cv_max_queries=None if args.cv_max_queries <= 0 else args.cv_max_queries)
        best = cv.iloc[0]
        print('Best reference-only CV:', f"k={int(best['k'])}, slope_weight={best['slope_weight']}, MAE={best['MAE']:.3f}, median_AE={best['median_AE']:.3f}")
        cv_all.append(cv)
        if len(eligible) == 0:
            continue
        pred = abc_predict_queries(eligible, ref_budget, last_cycle, int(best['k']), float(best['slope_weight']), sd_level, sd_slope)
        keep_meta = ['id', 'support_class_full_trajectory', 'global_fit_tmin', 'clinical_d', 'clinical_q', 'clinical_g']
        pred = pred.merge(eligible[keep_meta], on='id', how='left')
        pred['error'] = pred['pred_tmin'] - pred['global_fit_tmin']
        pred['abs_error'] = pred['error'].abs()
        pred['future_target'] = True
        pred['cv_MAE_selected'] = float(best['MAE'])
        pred['cv_median_AE_selected'] = float(best['median_AE'])
        pred_all.append(pred)
        ref_median = float(np.median(ref_budget['t_min_true']))
        groups = {'all_support_future': pred, 'well_supported_future': pred[pred['support_class_full_trajectory'] == 'well_supported'], 'well_plus_edge_future': pred[pred['support_class_full_trajectory'].isin(['well_supported', 'edge_of_support'])]}
        for label, g in groups.items():
            z = summarize_prediction_set(g, label, ref_median)
            z.update({'scan_count': scan_count, 'last_cycle': last_cycle, 'selected_k': int(best['k']), 'selected_slope_weight': float(best['slope_weight']), 'reference_cv_MAE': float(best['MAE'])})
            summary_rows.append(z)
    cv_out = pd.concat(cv_all, ignore_index=True)
    pred_out = pd.concat(pred_all, ignore_index=True) if pred_all else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    retention = pd.DataFrame(retention_rows)
    if len(pred_out):
        max_budget = args.max_scans
        fixed_ids = set(pred_out[(pred_out['scan_count'] == max_budget) & (pred_out['support_class_full_trajectory'] == 'well_supported')]['id'])
        print(f'\nFixed common well-supported cohort eligible through {max_budget} scans: n={len(fixed_ids)}')
        fixed_rows = []
        for scan_count in range(args.min_scans, args.max_scans + 1):
            g = pred_out[(pred_out['scan_count'] == scan_count) & pred_out['id'].isin(fixed_ids)].copy()
            if len(g) == 0:
                continue
            ref_budget = refscan.copy()
            ref_median = float(np.median(ref_budget['t_min_true']))
            z = summarize_prediction_set(g, 'fixed_common_well_supported', ref_median)
            z.update({'scan_count': scan_count, 'last_cycle': scan_count - 1})
            fixed_rows.append(z)
        fixed_summary = pd.DataFrame(fixed_rows)
    else:
        fixed_summary = pd.DataFrame()
    paths = {'retention': args.outdir / 'scan_budget_cohort_retention.csv', 'cv': args.outdir / 'scan_budget_reference_cv.csv', 'predictions': args.outdir / 'scan_budget_clinical_predictions.csv', 'summary': args.outdir / 'scan_budget_clinical_summary.csv', 'fixed': args.outdir / 'scan_budget_fixed_common_summary.csv'}
    retention.to_csv(paths['retention'], index=False)
    cv_out.to_csv(paths['cv'], index=False)
    pred_out.to_csv(paths['predictions'], index=False)
    summary.to_csv(paths['summary'], index=False)
    fixed_summary.to_csv(paths['fixed'], index=False)
    print('\n' + '=' * 72)
    print('COHORT RETENTION')
    print(retention.to_string(index=False))
    print('\nPRIMARY PER-BUDGET RESULTS')
    if len(summary):
        show = summary[summary['subset'] == 'well_supported_future'].copy()
        cols = ['scan_count', 'last_cycle', 'n', 'selected_k', 'selected_slope_weight', 'MAE', 'median_AE', 'RMSE', 'within_1', 'within_2', 'Spearman_rho', 'reference_median_MAE', 'ABC_minus_reference_median_MAE', 'median_neighbor_tmin_sd']
        print(show[cols].to_string(index=False))
    print('\nFIXED COMMON WELL-SUPPORTED COHORT')
    if len(fixed_summary):
        cols = ['scan_count', 'n', 'MAE', 'median_AE', 'RMSE', 'within_1', 'within_2', 'Spearman_rho', 'reference_median_MAE', 'ABC_minus_reference_median_MAE', 'median_neighbor_tmin_sd']
        print(fixed_summary[cols].to_string(index=False))
    print('\nWROTE')
    for k, p in paths.items():
        print(f'{k}: {p}')
    print('invalid global tmin:', invalid_path)
if __name__ == '__main__':
    main()
