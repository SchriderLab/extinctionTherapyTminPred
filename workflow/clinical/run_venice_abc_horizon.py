"""
run_venice_abc_horizon.py

Run the VENICE ABC/KNN forecasting analysis using the frozen,
horizon-restricted clinical-transfer setup:

  - clinical early window: baseline + visits 1,2,3
  - summary vector: [z1, z2, z3, slope_weight * slope]
  - z_t = log10(PSA_t / PSA_0)
  - simulation-to-clinical time scale: default 2.25
  - clinically relevant first-rebound horizon: default visit 12
  - reference eligibility: first simulated rebound-associated nadir
    must occur at visits 4..horizon
  - ABC neighborhood: K=25
  - support: clinical Kth-neighbor distance <= 95th percentile of
    reference leave-one-out Kth-neighbor distance
  - ABC point prediction: inverse-distance weighted mean of the K
    eligible reference target visits

This is a clinical model-adequacy / transfer diagnostic, not an untouched
external validation, because the parameter expansion and the global horizon
were informed by VENICE.

Outputs:
  venice_abc_horizon_predictions.csv
  venice_abc_horizon_summary.txt
  Fig_venice_abc_observed_vs_predicted.pdf/png
  Fig_venice_abc_error_distribution.pdf/png
"""
from __future__ import annotations
from rescue.runtime import workspace_path, allocated_threads
import argparse
from pathlib import Path
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr, pearsonr, wilcoxon

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--clinical-dir', type=Path, default=workspace_path('data'))
    p.add_argument('--locked-ids', type=Path, default=Path('data/source_data/venice_locked_268_ids.csv'))
    p.add_argument('--clinical-targets', type=Path, default=Path('data/source_data/venice_first_rebound_nadirs.csv'))
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--outdir', type=Path, default=Path('data/source_data/venice_abc_horizon'))
    p.add_argument('--figure-dir', type=Path, default=Path('figures/venice_abc_horizon'))
    p.add_argument('--scale', type=float, default=2.25)
    p.add_argument('--forecast-horizon', type=int, default=12)
    p.add_argument('--k', type=int, default=25)
    p.add_argument('--support-quantile', type=float, default=0.95)
    p.add_argument('--slope-weight', type=float, default=2.0)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=20260915)
    return p.parse_args()

def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    vals = s.astype(str).str.strip().str.lower()
    return vals.map({'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False}).fillna(False)

def find_csv(root: Path, basename: str) -> Path:
    hits = list(root.rglob(basename))
    if len(hits) != 1:
        raise RuntimeError(f'Expected exactly one {basename} under {root}; found {len(hits)}: {hits}')
    return hits[0]

def load_clinical_features(clinical_dir, locked_ids_path, slope_weight):
    ids = pd.read_csv(locked_ids_path)
    if 'USUBJID' not in ids.columns:
        raise RuntimeError(f'{locked_ids_path} lacks USUBJID')
    ids['USUBJID'] = ids['USUBJID'].astype(str)
    locked = set(ids['USUBJID'])
    adsl = pd.read_csv(find_csv(clinical_dir, 'adsl.csv'), low_memory=False)
    adlb = pd.read_csv(find_csv(clinical_dir, 'adlb.csv'), low_memory=False)
    adsl['USUBJID'] = adsl['USUBJID'].astype(str)
    adlb['USUBJID'] = adlb['USUBJID'].astype(str)
    base = adsl.loc[adsl['USUBJID'].isin(locked), ['USUBJID', 'TMPSA']].copy()
    base['TMPSA'] = pd.to_numeric(base['TMPSA'], errors='coerce')
    base = base[base['TMPSA'] > 0].dropna().drop_duplicates('USUBJID')
    x = adlb[adlb['USUBJID'].isin(locked)].copy()
    if 'LBTESTCD' in x.columns:
        x = x[x['LBTESTCD'].astype(str).str.upper().eq('PSA')].copy()
    x['cycle'] = pd.to_numeric(x['ANLVISN'], errors='coerce')
    x['PSA'] = pd.to_numeric(x['LBSTRESN'], errors='coerce')
    if 'LBDY2' in x.columns:
        d2 = pd.to_numeric(x['LBDY2'], errors='coerce')
    else:
        d2 = pd.Series(np.nan, index=x.index)
    if 'LBDY' in x.columns:
        d1 = pd.to_numeric(x['LBDY'], errors='coerce')
    else:
        d1 = pd.Series(np.nan, index=x.index)
    x['day'] = d2.where(d2.notna(), d1)
    x = x[x['cycle'].isin([1, 2, 3]) & x['PSA'].notna() & np.isfinite(x['PSA']) & (x['PSA'] > 0)].copy()
    x['cycle'] = x['cycle'].astype(int)
    x['nominal_day'] = 21.0 * x['cycle']
    x['day_missing'] = x['day'].isna()
    x['day_distance'] = np.abs(x['day'] - x['nominal_day'])
    x = x.sort_values(['USUBJID', 'cycle', 'day_missing', 'day_distance'], kind='mergesort').drop_duplicates(['USUBJID', 'cycle'], keep='first')
    wide = x.pivot(index='USUBJID', columns='cycle', values='PSA')
    wide.columns = [f'psa{int(c)}' for c in wide.columns]
    wide = wide.reset_index()
    c = base.merge(wide, on='USUBJID', how='inner')
    need = ['TMPSA', 'psa1', 'psa2', 'psa3']
    c = c.dropna(subset=need).copy()
    if len(c) != len(locked):
        missing = sorted(locked - set(c['USUBJID']))
        raise RuntimeError(f'Expected {len(locked)} locked patients with complete early data; recovered {len(c)}. First missing: {missing[:10]}')
    for j in [1, 2, 3]:
        c[f'z{j}'] = np.log10(c[f'psa{j}'] / c['TMPSA'])
    vx = np.arange(4, dtype=float)
    slopes = []
    for r in c.itertuples(index=False):
        y = np.array([0.0, r.z1, r.z2, r.z3], dtype=float)
        slopes.append(np.polyfit(vx, y, 1)[0])
    c['slope'] = slopes
    c['slope_weighted'] = slope_weight * c['slope']
    c = c.sort_values('USUBJID').reset_index(drop=True)
    X = c[['z1', 'z2', 'z3', 'slope_weighted']].to_numpy(float)
    if not np.isfinite(X).all():
        raise RuntimeError('Non-finite clinical feature values')
    return (c, X)

def load_reference_arrays(reference_path):
    r = pd.read_csv(reference_path, low_memory=False)
    required = {'ref_id', 't', 'N'}
    miss = required - set(r.columns)
    if miss:
        raise RuntimeError(f'Reference file missing columns: {sorted(miss)}')
    r = r[['ref_id', 't', 'N']].copy()
    r['ref_id'] = r['ref_id'].astype(str)
    r['t'] = pd.to_numeric(r['t'], errors='coerce')
    r['N'] = pd.to_numeric(r['N'], errors='coerce')
    r = r.dropna(subset=['t', 'N'])
    r = r[np.isfinite(r['N']) & (r['N'] >= 0)]
    r = r.groupby(['ref_id', 't'], as_index=False)['N'].median().sort_values(['ref_id', 't'])
    out = {}
    for rid, g in r.groupby('ref_id', sort=False):
        t = g['t'].to_numpy(float)
        n = g['N'].to_numpy(float)
        o = np.argsort(t)
        out[str(rid)] = (t[o], n[o])
    return out

def interp_logN(t, n, q):
    t = np.asarray(t, float)
    n = np.asarray(n, float)
    q = np.asarray(q, float)
    pos = n > 0
    if pos.sum() < 2:
        return np.full(len(q), np.nan)
    tp = t[pos]
    lp = np.log10(n[pos])
    y = np.interp(q, tp, lp)
    y[(q < tp.min()) | (q > tp.max())] = np.nan
    return y

def reference_table(refs, scale, horizon, slope_weight):
    rows = []
    for rid, (t, n) in refs.items():
        early_q = np.array([0, scale, 2 * scale, 3 * scale], float)
        ly = interp_logN(t, n, early_q)
        if not np.isfinite(ly).all():
            continue
        z = ly - ly[0]
        slope = np.polyfit(np.arange(4, dtype=float), z, 1)[0]
        target = np.nan
        max_visit = int(np.floor(np.max(t) / scale + 1e-12))
        max_j = min(horizon, max_visit - 1)
        if max_j >= 4:
            visits = np.arange(max_visit + 1, dtype=float)
            lyall = interp_logN(t, n, visits * scale)
            for j in range(4, max_j + 1):
                if not np.all(np.isfinite(lyall[j - 1:j + 2])):
                    continue
                if lyall[j] <= lyall[j - 1] and lyall[j + 1] > lyall[j]:
                    target = float(j)
                    break
        rows.append({'ref_id': rid, 'z1': z[1], 'z2': z[2], 'z3': z[3], 'slope': slope, 'slope_weighted': slope_weight * slope, 'target_visit': target, 'eligible': np.isfinite(target)})
    return pd.DataFrame(rows)

def feature_matrix(df):
    return df[['z1', 'z2', 'z3', 'slope_weighted']].to_numpy(float)

def loo_support_threshold(X, k, q):
    if len(X) <= k:
        raise RuntimeError(f'Need > K reference rows; got n={len(X)}, K={k}')
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X)
    d, _ = nn.kneighbors(X)
    return float(np.quantile(d[:, k], q))

def idw_mean(distances, values):
    d = np.asarray(distances, float)
    y = np.asarray(values, float)
    zero = d <= 1e-12
    if zero.any():
        return float(np.mean(y[zero]))
    w = 1.0 / d
    w = w / w.sum()
    return float(np.sum(w * y))

def weighted_quantile(values, weights, q):
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float(np.quantile(v, q))
    cw = cw / cw[-1]
    return float(np.interp(q, cw, v))

def abc_neighbor_summary(distances, targets):
    d = np.asarray(distances, float)
    y = np.asarray(targets, float)
    zero = d <= 1e-12
    if zero.any():
        w = zero.astype(float)
    else:
        w = 1.0 / d
    w = w / w.sum()
    return {'pred_abc_idw': float(np.sum(w * y)), 'neighbor_mean': float(np.mean(y)), 'neighbor_median': float(np.median(y)), 'neighbor_q10': weighted_quantile(y, w, 0.1), 'neighbor_q90': weighted_quantile(y, w, 0.9), 'neighbor_target_min': float(np.min(y)), 'neighbor_target_max': float(np.max(y))}

def bootstrap_metrics(y, pred, nboot, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    out = {'mae': [], 'rmse': [], 'bias': [], 'spearman': []}
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        pp = pred[idx]
        e = pp - yy
        out['mae'].append(float(np.mean(np.abs(e))))
        out['rmse'].append(float(np.sqrt(np.mean(e * e))))
        out['bias'].append(float(np.mean(e)))
        if np.unique(yy).size > 1 and np.unique(pp).size > 1:
            rho = spearmanr(pp, yy).statistic
            if np.isfinite(rho):
                out['spearman'].append(float(rho))

    def ci(a):
        a = np.asarray(a, float)
        if len(a) == 0:
            return (np.nan, np.nan)
        return tuple(np.quantile(a, [0.025, 0.975]))
    return {k: ci(v) for k, v in out.items()}

def fmt_ci(x, ci):
    return f'{x:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]'

def main():
    args = parse_args()
    if args.forecast_horizon <= 3:
        raise RuntimeError('--forecast-horizon must be > 3')
    if args.k < 1:
        raise RuntimeError('--k must be >=1')
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    clin, Xclin = load_clinical_features(args.clinical_dir, args.locked_ids, args.slope_weight)
    ct = pd.read_csv(args.clinical_targets, low_memory=False)
    ct['USUBJID'] = ct['USUBJID'].astype(str)
    ct['target_available'] = as_bool(ct['target_available'])
    ct['target_cycle'] = pd.to_numeric(ct['target_cycle'], errors='coerce')
    clin = clin.merge(ct[['USUBJID', 'target_available', 'target_cycle']], on='USUBJID', how='left')
    refs = load_reference_arrays(args.reference)
    ref_all = reference_table(refs, args.scale, args.forecast_horizon, args.slope_weight)
    ref = ref_all[ref_all['eligible']].copy().reset_index(drop=True)
    if len(ref) <= args.k:
        raise RuntimeError(f'Only {len(ref)} horizon-eligible references; cannot run K={args.k}')
    Xref = feature_matrix(ref)
    threshold = loo_support_threshold(Xref, args.k, args.support_quantile)
    nn = NearestNeighbors(n_neighbors=args.k)
    nn.fit(Xref)
    d, idx = nn.kneighbors(Xclin)
    kth = d[:, -1]
    supported = kth <= threshold
    rows = []
    for i in range(len(clin)):
        neigh = ref.iloc[idx[i]]
        targets = neigh['target_visit'].to_numpy(float)
        summ = abc_neighbor_summary(d[i], targets)
        row = {'USUBJID': clin.iloc[i]['USUBJID'], 'supported': bool(supported[i]), 'support_distance': float(kth[i]), 'support_threshold': float(threshold), 'support_ratio': float(kth[i] / threshold), 'observed_target_available': bool(clin.iloc[i]['target_available']) if pd.notna(clin.iloc[i]['target_available']) else False, 'observed_target_cycle': clin.iloc[i]['target_cycle'], **summ}
        rows.append(row)
    pred = pd.DataFrame(rows)
    pred['within_horizon_observed'] = pred['observed_target_available'] & pred['observed_target_cycle'].notna() & (pred['observed_target_cycle'] >= 4) & (pred['observed_target_cycle'] <= args.forecast_horizon)
    ev = pred[pred['supported'] & pred['within_horizon_observed']].copy()
    pred_path = args.outdir / 'venice_abc_horizon_predictions.csv'
    pred.to_csv(pred_path, index=False)
    lines = []
    lines.append('=== VENICE ABC: HORIZON-RESTRICTED CLINICAL TRANSFER ===')
    lines.append(f'scale={args.scale:g}')
    lines.append(f'forecast_horizon_visit={args.forecast_horizon}')
    lines.append(f'K={args.k}')
    lines.append(f'slope_weight={args.slope_weight:g}')
    lines.append(f'support_quantile={args.support_quantile:g}')
    lines.append('')
    lines.append('REFERENCE LIBRARY')
    lines.append(f'reference_early_valid={len(ref_all)}')
    lines.append(f'reference_horizon_eligible={len(ref)}')
    lines.append(f'fraction_horizon_eligible={len(ref) / len(ref_all):.8f}' if len(ref_all) else 'fraction_horizon_eligible=nan')
    lines.append(f"reference_target_median={ref['target_visit'].median():.3f}")
    lines.append(f"reference_target_q1={ref['target_visit'].quantile(0.25):.3f}")
    lines.append(f"reference_target_q3={ref['target_visit'].quantile(0.75):.3f}")
    lines.append('')
    lines.append('SUPPORT')
    lines.append(f'support_threshold={threshold:.8f}')
    lines.append(f"clinical_supported={int(pred['supported'].sum())}/{len(pred)} ({pred['supported'].mean():.2%})")
    lines.append(f"median_support_ratio={pred['support_ratio'].median():.3f}")
    lines.append(f'supported_with_observed_target_inside_horizon={len(ev)}')
    if len(ev):
        y = ev['observed_target_cycle'].to_numpy(float)
        p = ev['pred_abc_idw'].to_numpy(float)
        err = p - y
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))
        mederr = float(np.median(err))
        within1 = float((np.abs(err) <= 1).mean())
        within2 = float((np.abs(err) <= 2).mean())
        if np.unique(p).size > 1 and np.unique(y).size > 1:
            sp = spearmanr(p, y)
            pe = pearsonr(p, y)
            rho, rho_p = (float(sp.statistic), float(sp.pvalue))
            r, r_p = (float(pe.statistic), float(pe.pvalue))
        else:
            rho = rho_p = r = r_p = np.nan
        coverage = float(((y >= ev['neighbor_q10'].to_numpy(float)) & (y <= ev['neighbor_q90'].to_numpy(float))).mean())
        ref_median = float(ref['target_visit'].median())
        baseline_pred = np.full_like(y, ref_median)
        baseline_abs = np.abs(baseline_pred - y)
        baseline_mae = float(np.mean(baseline_abs))
        baseline_rmse = float(np.sqrt(np.mean((baseline_pred - y) ** 2)))
        abc_abs = np.abs(err)
        delta_mae = baseline_mae - mae
        nz = baseline_abs - abc_abs
        nz = nz[np.abs(nz) > 1e-12]
        if len(nz):
            try:
                wt = wilcoxon(nz, alternative='greater', zero_method='wilcox')
                baseline_w_p = float(wt.pvalue)
            except Exception:
                baseline_w_p = np.nan
        else:
            baseline_w_p = np.nan
        cis = bootstrap_metrics(y, p, args.bootstrap, args.seed)
        ev['error_pred_minus_observed'] = err
        ev['abs_error'] = np.abs(err)
        lines.append('')
        lines.append('PRIMARY ABC PERFORMANCE')
        lines.append(f'n_evaluable={len(ev)}')
        lines.append(f"MAE={fmt_ci(mae, cis['mae'])}")
        lines.append(f"RMSE={fmt_ci(rmse, cis['rmse'])}")
        lines.append(f"bias_pred_minus_observed={fmt_ci(bias, cis['bias'])}")
        lines.append(f'median_error_pred_minus_observed={mederr:.3f}')
        lines.append(f'within_1_visit={within1:.2%}')
        lines.append(f'within_2_visits={within2:.2%}')
        lines.append(f"Spearman_pred_vs_observed={rho:.3f} (p={rho_p:.6g}; bootstrap95%={cis['spearman'][0]:.3f},{cis['spearman'][1]:.3f})")
        lines.append(f'Pearson_pred_vs_observed={r:.3f} (p={r_p:.6g})')
        lines.append(f'weighted_neighbor_10_90_coverage={coverage:.2%}')
        lines.append('')
        lines.append('REFERENCE-MEDIAN BASELINE')
        lines.append(f'reference_median_target={ref_median:.3f}')
        lines.append(f'baseline_MAE={baseline_mae:.3f}')
        lines.append(f'baseline_RMSE={baseline_rmse:.3f}')
        lines.append(f'MAE_improvement_baseline_minus_ABC={delta_mae:.3f}')
        lines.append(f'Wilcoxon H1: baseline absolute error > ABC absolute error: p={baseline_w_p:.6g}')
        fig, ax = plt.subplots(figsize=(6.2, 5.8))
        ax.scatter(y, p, alpha=0.75)
        lo = min(np.min(y), np.min(p)) - 0.5
        hi = max(np.max(y), np.max(p)) + 0.5
        ax.plot([lo, hi], [lo, hi], linestyle='--')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel('Observed first rebound-associated PSA nadir visit')
        ax.set_ylabel('ABC-predicted nadir visit')
        ax.set_title(f'VENICE ABC prediction: supported patients within visit {args.forecast_horizon}')
        ax.text(0.03, 0.97, f'n={len(ev)}\nMAE={mae:.2f}\nSpearman ρ={rho:.2f}', transform=ax.transAxes, va='top')
        fig.tight_layout()
        fig.savefig(args.figure_dir / 'Fig_venice_abc_observed_vs_predicted.pdf')
        fig.savefig(args.figure_dir / 'Fig_venice_abc_observed_vs_predicted.png', dpi=250)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        ax.hist(err, bins=np.arange(math.floor(err.min()) - 0.5, math.ceil(err.max()) + 1.5, 1.0))
        ax.axvline(0, linestyle='--')
        ax.set_xlabel('Prediction error (predicted − observed visit)')
        ax.set_ylabel('Patients')
        ax.set_title('VENICE ABC prediction errors')
        fig.tight_layout()
        fig.savefig(args.figure_dir / 'Fig_venice_abc_error_distribution.pdf')
        fig.savefig(args.figure_dir / 'Fig_venice_abc_error_distribution.png', dpi=250)
        plt.close(fig)
    lines.append('')
    lines.append('INTERPRETATION LIMITS')
    lines.append('Primary performance is reported only for patients inside the simulation support and whose observed first rebound-associated nadir occurs within the frozen forecasting horizon.')
    lines.append('The parameter expansion and forecasting horizon were informed by VENICE; therefore this analysis is a clinically calibrated transfer/model-adequacy test, not untouched external validation.')
    lines.append('Patients outside support are not counted as successful predictions.')
    summary_path = args.outdir / 'venice_abc_horizon_summary.txt'
    summary_path.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print()
    print('WROTE', pred_path)
    print('WROTE', summary_path)
    if len(ev):
        print('WROTE', args.figure_dir / 'Fig_venice_abc_observed_vs_predicted.pdf')
        print('WROTE', args.figure_dir / 'Fig_venice_abc_error_distribution.pdf')
if __name__ == '__main__':
    main()
