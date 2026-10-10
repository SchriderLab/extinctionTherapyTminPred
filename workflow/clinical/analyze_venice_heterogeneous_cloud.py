from __future__ import annotations
from rescue.runtime import workspace_path, allocated_threads
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance, spearmanr

def as_bool(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin({'true', '1', 't', 'yes'})

def clinical_clean(cloud):

    def pick(names):
        for n in names:
            if n in cloud.columns:
                return n
    dc = pick(['decline_rate_hat', 'd_hat', 'd'])
    qc = pick(['q_hat', 'growth_weight_hat', 'resistant_weight_hat', 'q'])
    gc = pick(['growth_rate_hat', 'g_hat', 'g'])
    if not (dc and qc and gc):
        raise ValueError('Could not identify clinical d/q/g columns')
    out = pd.DataFrame({'d': pd.to_numeric(cloud[dc], errors='coerce'), 'q': pd.to_numeric(cloud[qc], errors='coerce'), 'g': pd.to_numeric(cloud[gc], errors='coerce')})
    keep = np.isfinite(out).all(axis=1) & (out.q > 0)
    if {'fit_tmin_in_observed_window', 'growth_near_lower_bound', 'q_near_lower_bound'}.issubset(cloud.columns):
        pathological = ~as_bool(cloud['fit_tmin_in_observed_window']) & (as_bool(cloud['growth_near_lower_bound']) | as_bool(cloud['q_near_lower_bound']))
        keep &= ~pathological
    return out.loc[keep].reset_index(drop=True)

def pred_log(theta, t):
    d, lq, g = theta
    q = 1 / (1 + np.exp(-lq))
    return np.logaddexp(np.log1p(-q) - d * t, np.log(q) + g * t)

def fit_one(t, y):
    ok = np.isfinite(t) & np.isfinite(y)
    t = t[ok]
    y = y[ok]
    if len(t) < 6 or len(np.unique(t)) < 6:
        return {'fit_success': False}
    q0 = 0.006
    x0 = np.array([0.38, np.log(q0 / (1 - q0)), 0.305])
    lo = np.array([1e-05, np.log(1e-08 / (1 - 1e-08)), 1e-05])
    hi = np.array([3.0, np.log(0.95 / 0.05), 3.0])
    try:
        z = least_squares(lambda th: pred_log(th, t) - y, x0=x0, bounds=(lo, hi), max_nfev=3000)
        d, lq, g = z.x
        q = 1 / (1 + np.exp(-lq))
        return {'fit_success': bool(z.success), 'd_hat': float(d), 'q_hat': float(q), 'g_hat': float(g), 'fit_rmse_log': float(np.sqrt(np.mean((pred_log(z.x, t) - y) ** 2)))}
    except Exception:
        return {'fit_success': False}

def summarize(fits):
    rows = []
    for sc, x in fits.groupby('scenario', sort=False):
        z = x[x.fit_success]
        r = {'scenario': sc, 'n': len(x), 'n_fit_success': len(z), 'fit_success_rate': len(z) / len(x), 'median_final_t': float(z.final_t.median()) if len(z) else np.nan, 'median_fit_rmse_log': float(z.fit_rmse_log.median()) if len(z) else np.nan}
        for p in ['d_hat', 'q_hat', 'g_hat']:
            for name, qv in [('q05', 0.05), ('median', 0.5), ('q95', 0.95)]:
                r[f'{p}_{name}'] = float(z[p].quantile(qv)) if len(z) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)

def cloud_metrics(fits, clin):
    eps = 1e-12
    C = np.column_stack([clin.d.to_numpy(float), np.log10(np.maximum(clin.q.to_numpy(float), eps)), clin.g.to_numpy(float)])
    med = np.median(C, axis=0)
    scale = np.maximum(np.quantile(C, 0.75, axis=0) - np.quantile(C, 0.25, axis=0), 1e-06)
    Csc = (C - med) / scale
    treeC = cKDTree(Csc)
    rows = []
    for sc, x in fits[fits.fit_success].groupby('scenario', sort=False):
        S = np.column_stack([x.d_hat.to_numpy(float), np.log10(np.maximum(x.q_hat.to_numpy(float), eps)), x.g_hat.to_numpy(float)])
        Ssc = (S - med) / scale
        treeS = cKDTree(Ssc)
        sim_to_clin = treeC.query(Ssc, k=1)[0]
        clin_to_sim = treeS.query(Csc, k=1)[0]
        wd = wasserstein_distance(C[:, 0], S[:, 0]) / scale[0]
        wq = wasserstein_distance(C[:, 1], S[:, 1]) / scale[1]
        wg = wasserstein_distance(C[:, 2], S[:, 2]) / scale[2]
        rows.append({'scenario': sc, 'n': len(x), 'sim_to_clin_median_nn': float(np.median(sim_to_clin)), 'sim_to_clin_q90_nn': float(np.quantile(sim_to_clin, 0.9)), 'clin_to_sim_median_nn': float(np.median(clin_to_sim)), 'clin_to_sim_q90_nn': float(np.quantile(clin_to_sim, 0.9)), 'wasserstein_d_scaled': float(wd), 'wasserstein_log10q_scaled': float(wq), 'wasserstein_g_scaled': float(wg), 'mean_scaled_wasserstein': float(np.mean([wd, wq, wg]))})
    return pd.DataFrame(rows)

def correlations(fits):
    z = fits[fits.fit_success].copy()
    rows = []
    predictors = ['standing_frac', 'm1_effect_mean_target', 'm1_sdlog', 'patient_turnover', 'base_mut_rate', 'mean_age']
    outcomes = ['d_hat', 'q_hat', 'g_hat']
    for sc, x in z.groupby('scenario', sort=False):
        for a in predictors:
            for b in outcomes:
                aa = pd.to_numeric(x[a], errors='coerce')
                bb = pd.to_numeric(x[b], errors='coerce')
                ok = np.isfinite(aa) & np.isfinite(bb)
                if ok.sum() < 10:
                    continue
                rho, p = spearmanr(aa[ok], bb[ok])
                rows.append({'scenario': sc, 'predictor': a, 'outcome': b, 'n': int(ok.sum()), 'spearman_rho': float(rho), 'spearman_p': float(p)})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--library', type=Path, required=True)
    ap.add_argument('--cloud', type=Path, required=True)
    ap.add_argument('--outdir', type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.library / 'patient_manifest.csv')
    rec = []
    for j, row in manifest.iterrows():
        pid = int(row.patient_id)
        f = args.library / 'raw' / f'patient_{pid:05d}' / 'trajectory.csv'
        r = {'patient_id': pid, 'base_patient_id': int(row.base_patient_id), 'scenario': str(row.scenario), 'standing_frac': float(row.standing_frac), 'm1_effect_mean_target': float(row.m1_effect_mean_target), 'm1_sdlog': float(row.m1_sdlog), 'patient_turnover': float(row.patient_turnover), 'base_mut_rate': float(row.base_mut_rate), 'mean_age': float(np.mean([row[f'age{i}'] for i in range(1, 6)])), 'fit_success': False}
        if f.exists():
            try:
                x = pd.read_csv(f)
                if len(x) >= 6 and float(x.iloc[0].N0_total) > 0:
                    t = pd.to_numeric(x.t_treatment, errors='coerce').to_numpy(float)
                    N = pd.to_numeric(x.N, errors='coerce').to_numpy(float)
                    y = np.log(np.maximum(N, 1e-12) / float(x.iloc[0].N0_total))
                    r.update({'n_points': len(x), 'final_t': float(t[-1]), 'final_status': str(x.iloc[-1].status), 'N0_total': float(x.iloc[0].N0_total)})
                    rr = [float(x.iloc[0][f'standing{i}_realized']) for i in range(1, 6)]
                    r['mean_standing_realized'] = float(np.mean(rr))
                    r.update(fit_one(t, y))
            except Exception as e:
                r['parse_error'] = repr(e)
        rec.append(r)
        if (j + 1) % 200 == 0:
            print(f'processed {j + 1}/{len(manifest)}', flush=True)
    fits = pd.DataFrame(rec)
    fits.to_csv(args.outdir / 'patient_fits.csv', index=False)
    summ = summarize(fits)
    summ.to_csv(args.outdir / 'fit_summary.csv', index=False)
    clin = clinical_clean(pd.read_csv(args.cloud))
    cs = pd.DataFrame({'parameter': ['d', 'q', 'g'], 'q05': [clin[c].quantile(0.05) for c in ['d', 'q', 'g']], 'median': [clin[c].median() for c in ['d', 'q', 'g']], 'q95': [clin[c].quantile(0.95) for c in ['d', 'q', 'g']], 'n': [len(clin)] * 3})
    cs.to_csv(args.outdir / 'clinical_summary.csv', index=False)
    cm = cloud_metrics(fits, clin)
    cm.to_csv(args.outdir / 'cloud_distance_metrics.csv', index=False)
    cr = correlations(fits)
    cr.to_csv(args.outdir / 'parameter_to_exponential_correlations.csv', index=False)
    prs = []
    for sc, x in manifest.groupby('scenario', sort=False):
        for p in ['standing_frac', 'm1_effect_mean_target', 'm1_sdlog', 'patient_turnover', 'base_mut_rate']:
            a = pd.to_numeric(x[p], errors='coerce')
            prs.append({'scenario': sc, 'parameter': p, 'q05': a.quantile(0.05), 'median': a.median(), 'q95': a.quantile(0.95)})
    pd.DataFrame(prs).to_csv(args.outdir / 'mechanistic_parameter_summary.csv', index=False)
    print('\n=== CLINICAL ===')
    print(cs.to_string(index=False))
    print('\n=== FIT SUMMARY ===')
    print(summ.to_string(index=False))
    print('\n=== CLOUD DISTANCE / COVERAGE ===')
    print(cm.sort_values('mean_scaled_wasserstein').to_string(index=False))
    print('\n=== STRONGEST PARAMETER RELATIONSHIPS ===')
    tmp = cr.assign(absrho=cr.spearman_rho.abs()).sort_values(['scenario', 'absrho'], ascending=[True, False])
    print(tmp.groupby('scenario').head(8).drop(columns='absrho').to_string(index=False))
if __name__ == '__main__':
    main()
