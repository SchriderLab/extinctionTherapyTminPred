from rescue.runtime import workspace_path, allocated_threads
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit, gammaln, logsumexp
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
ROOT = Path('.')
CLOUD = ROOT / 'data/source_data/exponential_cloud/venice_exponential_cloud_per_patient.csv'
PRED = ROOT / 'data/clinical/qg_folds.csv'
ADSL = ROOT / 'data/venice_csv/adsl.csv'
ADLB = ROOT / 'data/venice_csv/adlb.csv'
OUT = ROOT / 'venice_k1_abc_reference_10k/qg_joint_prior_ablation_v1'
OUT.mkdir(parents=True, exist_ok=True)
N_DRAWS = 50000
JITTER = 0.08
DF_T = 4.0
SEED = 73191
SCAN_COUNTS = [2, 3, 4, 5, 6]

def as_bool(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin(['true', '1', 't', 'yes', 'y'])

def t_logpdf(resid, scale, df=DF_T):
    scale = max(float(scale), 0.0001)
    z = resid / scale
    return gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(df * np.pi) - np.log(scale) - (df + 1) / 2 * np.log1p(z * z / df)

def weighted_quantile(x, w, q=0.5):
    order = np.argsort(x)
    xs = x[order]
    ws = w[order]
    c = np.cumsum(ws)
    if c[-1] <= 0 or not np.isfinite(c[-1]):
        return np.nan
    c /= c[-1]
    return float(np.interp(q, c, xs))

def make_base_prior(train, mode, n_draws, rng):
    tr = train[['q_hat', 'decline_rate_hat', 'growth_rate_hat']].copy()
    tr.columns = ['q', 'd', 'g']
    tr = tr.replace([np.inf, -np.inf], np.nan).dropna()
    tr = tr[(tr.q > 0) & (tr.q < 1) & (tr.d > 0) & (tr.g > 0)].reset_index(drop=True)
    if mode == 'qg_shuffled':
        nb = min(5, max(2, len(tr) // 20))
        bins = pd.qcut(tr['d'].rank(method='first'), q=nb, labels=False, duplicates='drop')
        gnew = tr['g'].to_numpy().copy()
        for b in np.unique(bins):
            ii = np.where(np.asarray(bins) == b)[0]
            gnew[ii] = rng.permutation(gnew[ii])
        tr['g'] = gnew
    idx = rng.integers(0, len(tr), n_draws)
    q = tr.loc[idx, 'q'].to_numpy(float)
    d = tr.loc[idx, 'd'].to_numpy(float)
    g = tr.loc[idx, 'g'].to_numpy(float)
    lq = logit(np.clip(q, 1e-08, 1 - 1e-08)) + rng.normal(0, JITTER, n_draws)
    ld = np.log(d) + rng.normal(0, JITTER, n_draws)
    lg = np.log(g) + rng.normal(0, JITTER, n_draws)
    q = expit(lq)
    d = np.exp(ld)
    g = np.exp(lg)
    tmin = np.log((1 - q) * d / (q * g)) / (d + g)
    ok = np.isfinite(tmin) & (tmin > 0) & (tmin < 50)
    return (q[ok], d[ok], g[ok], tmin[ok])

def posterior_one(y, times, prior, scale):
    q, d, g, tmin = prior
    ll = np.zeros(len(q), dtype=float)
    for yy, tt in zip(y, times):
        a = np.log1p(-q) - d * tt
        b = np.log(q) + g * tt
        pred = np.logaddexp(a, b)
        ll += t_logpdf(yy - pred, scale)
    lw = ll - logsumexp(ll)
    w = np.exp(lw)
    ess = 1.0 / np.sum(w * w)
    return (weighted_quantile(tmin, w, 0.5), float(np.sum(w * (tmin > 8.0))), float(ess))
cloud = pd.read_csv(CLOUD, low_memory=False)
pathological = ~as_bool(cloud['fit_tmin_in_observed_window']) & (as_bool(cloud['growth_near_lower_bound']) | as_bool(cloud['q_near_lower_bound']))
cloud = cloud.loc[~pathological].copy()
cloud['id'] = cloud['id'].astype(str)
cloud = cloud[(cloud.q_hat > 0) & (cloud.q_hat < 1) & (cloud.decline_rate_hat > 0) & (cloud.growth_rate_hat > 0)].copy()
print('Non-pathological prior pool n =', len(cloud))
print('q-g Spearman in prior pool =', round(spearmanr(cloud.q_hat, cloud.growth_rate_hat).statistic, 3))
p = pd.read_csv(PRED, low_memory=False)
base = p[p['scan_count'].eq(6)][['id', 'fold', 'exp_tmin']].drop_duplicates('id').copy()
base['id'] = base['id'].astype(str)
base['late8'] = (base['exp_tmin'] > 8).astype(int)
print('Fixed cohort n =', len(base), 'class counts =', base.late8.value_counts().sort_index().to_dict())
adsl = pd.read_csv(ADSL, low_memory=False)
adsl['id'] = adsl['USUBJID'].astype(str)
adsl['TMPSA'] = pd.to_numeric(adsl['TMPSA'], errors='coerce')
adsl = adsl[['id', 'TMPSA']].drop_duplicates('id')
lb = pd.read_csv(ADLB, low_memory=False)
lb['id'] = lb['USUBJID'].astype(str)
is_psa = lb['LBTESTCD'].astype(str).str.upper().eq('PSA') | lb['LBTEST'].astype(str).str.upper().eq('PSA')
lb = lb[is_psa].copy()
lb['cycle'] = pd.to_numeric(lb['ANLVISN'], errors='coerce')
lb['psa'] = pd.to_numeric(lb['LBSTRESN'], errors='coerce')
lb = lb[lb['cycle'].between(1, 5) & lb['psa'].gt(0)]
wide = lb.pivot_table(index='id', columns='cycle', values='psa', aggfunc='median')
wide = wide.rename(columns={float(c): f'psa{c}' for c in range(1, 6)}).reset_index()
d = base.merge(adsl, on='id', how='left').merge(wide, on='id', how='left')
for c in range(1, 6):
    d[f'y{c}'] = np.log(pd.to_numeric(d[f'psa{c}'], errors='coerce') / d['TMPSA'])
needed = ['TMPSA'] + [f'y{c}' for c in range(1, 6)]
if d[needed].isna().any(axis=None):
    bad = d.loc[d[needed].isna().any(axis=1), 'id'].tolist()
    raise RuntimeError(f'Missing PSA values for fixed cohort: {bad[:10]}')
rows = []
fold_cache = {}
for fold in sorted(d['fold'].dropna().unique()):
    test_ids = set(d.loc[d['fold'].eq(fold), 'id'])
    train = cloud.loc[~cloud['id'].isin(test_ids)].copy()
    scale = float(np.nanmedian(pd.to_numeric(train['sigma_log10'], errors='coerce')) * np.log(10.0))
    if not np.isfinite(scale) or scale <= 0:
        scale = 0.15
    print(f'fold {fold}: prior n={len(train)}, ln-scale={scale:.4f}')
    for mode_i, mode in enumerate(['joint', 'qg_shuffled']):
        rng = np.random.default_rng(SEED + int(fold) * 100 + mode_i)
        prior = make_base_prior(train, mode, N_DRAWS, rng)
        print(f'  {mode}: usable draws={len(prior[0])}, prior tmin median={np.median(prior[3]):.3f}')
        fold_cache[fold, mode] = (prior, scale)
for scan in SCAN_COUNTS:
    print('\nscan', scan)
    for mode in ['joint', 'qg_shuffled']:
        pred_t = np.full(len(d), np.nan)
        prob8 = np.full(len(d), np.nan)
        ess = np.full(len(d), np.nan)
        for i, r in d.iterrows():
            prior, scale = fold_cache[r['fold'], mode]
            times = np.arange(scan, dtype=float)
            y = np.array([0.0] + [r[f'y{c}'] for c in range(1, scan)], dtype=float)
            pred_t[i], prob8[i], ess[i] = posterior_one(y, times, prior, scale)
        ytrue = d['exp_tmin'].to_numpy(float)
        ybin = d['late8'].to_numpy(int)
        mae = float(np.mean(np.abs(pred_t - ytrue)))
        rho = float(spearmanr(pred_t, ytrue).statistic)
        auc = float(roc_auc_score(ybin, prob8))
        bal = float(balanced_accuracy_score(ybin, (prob8 >= 0.5).astype(int)))
        print(f'  {mode:12s} MAE={mae:.3f} rho={rho:.3f} AUROC8={auc:.3f} balacc={bal:.3f} medianESS={np.median(ess):.0f}')
        rows.append({'scan_count': scan, 'mode': mode, 'n': len(d), 'MAE_exp_tmin': mae, 'Spearman_exp_tmin': rho, 'AUROC_tmin_gt8': auc, 'balanced_accuracy_05': bal, 'median_ESS': float(np.median(ess))})
summary = pd.DataFrame(rows)
summary.to_csv(OUT / 'qg_joint_vs_shuffled_summary.csv', index=False)
wideout = summary.pivot(index='scan_count', columns='mode', values=['MAE_exp_tmin', 'Spearman_exp_tmin', 'AUROC_tmin_gt8', 'balanced_accuracy_05'])
print('\nSUMMARY')
print(summary.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print('\nDELTA joint - shuffled (positive favors joint for rho/AUC/balacc; negative favors joint for MAE)')
for scan in SCAN_COUNTS:
    a = summary[(summary.scan_count == scan) & (summary['mode'] == 'joint')].iloc[0]
    b = summary[(summary.scan_count == scan) & (summary['mode'] == 'qg_shuffled')].iloc[0]
    print(f'scan {scan}: dMAE={a.MAE_exp_tmin - b.MAE_exp_tmin:+.3f}, dRho={a.Spearman_exp_tmin - b.Spearman_exp_tmin:+.3f}, dAUC={a.AUROC_tmin_gt8 - b.AUROC_tmin_gt8:+.3f}, dBalAcc={a.balanced_accuracy_05 - b.balanced_accuracy_05:+.3f}')
print('\nWrote', OUT / 'qg_joint_vs_shuffled_summary.csv')
