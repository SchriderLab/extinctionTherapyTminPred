from rescue.runtime import workspace_path, allocated_threads
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors
ROOT = workspace_path('')
MAIN = ROOT / 'data/source_data/mainsail'
M_FEATURES = MAIN / 'mainsail_early_features.csv'
M_TARGETS = MAIN / 'mainsail_first_rebound_nadirs.csv'
M_FORECAST_IDS = MAIN / 'mainsail_forecasting_ids.csv'
VENICE_ROOT = ROOT / 'data/venice_csv'
V_TARGETS = ROOT / 'data/source_data/venice_first_rebound_nadirs.csv'
V_LOCKED = ROOT / 'data/source_data/venice_locked_268_ids.csv'
REFDIR = ROOT / 'abc_ref_clinical_expanded_v2_screen5k'
REF_TRAJ = REFDIR / 'reference_trajectories.csv'
OUT = ROOT / 'data/source_data/mainsail_validation'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260915
FEATURES = ['z1', 'z2', 'z3', 'slope']
BINARY_CUTOFF = 7
ABC_SCALE = 2.25
ABC_H = 12
ABC_K = 25
ABC_SLOPE_WEIGHT = 2.0
ABC_SUPPORT_THRESHOLD = 0.09268436
RIDGE_ALPHAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

def norm_id(s):
    return s.astype(str).str.strip().str.replace('\\.0$', '', regex=True)

def bool_series(s):
    if pd.api.types.is_bool_dtype(s):
        return s
    return s.astype(str).str.strip().str.lower().isin(['true', '1', 'yes', 'y'])

def regression_metrics(y, pred):
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    rho, rho_p = spearmanr(y, pred)
    r, r_p = pearsonr(y, pred)
    return {'n': len(y), 'MAE': float(mean_absolute_error(y, pred)), 'RMSE': float(np.sqrt(mean_squared_error(y, pred))), 'bias_pred_minus_obs': float(np.mean(pred - y)), 'median_abs_error': float(np.median(np.abs(pred - y))), 'within_1': float(np.mean(np.abs(pred - y) <= 1)), 'within_2': float(np.mean(np.abs(pred - y) <= 2)), 'Spearman_rho': float(rho), 'Spearman_p': float(rho_p), 'Pearson_r': float(r), 'Pearson_p': float(r_p)}

def bootstrap_auc(y, p, B=3000, seed=SEED):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(roc_auc_score(yy, p[idx]))
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))

def classification_metrics(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    lo, hi = bootstrap_auc(y, p)
    return {'n': len(y), 'n_early': int((y == 0).sum()), 'n_late': int((y == 1).sum()), 'late_prevalence': float(y.mean()), 'AUROC': float(roc_auc_score(y, p)), 'AUROC_CI_low': lo, 'AUROC_CI_high': hi, 'average_precision': float(average_precision_score(y, p)), 'balanced_accuracy': float(balanced_accuracy_score(y, p >= 0.5))}

def target_col(df):
    preferred = ['target_t', 'target_visit', 'target_cycle', 'first_rebound_visit', 'first_rebound_cycle', 'nadir_visit', 'nadir_cycle', 'tmin', 't_min']
    lower = {c.lower(): c for c in df.columns}
    for c in preferred:
        if c.lower() in lower:
            return lower[c.lower()]
    candidates = [c for c in df.columns if ('target' in c.lower() or 'nadir' in c.lower() or 'tmin' in c.lower()) and pd.api.types.is_numeric_dtype(df[c])]
    if not candidates:
        raise RuntimeError(f'Could not identify target column.\nColumns: {df.columns.tolist()}')
    return candidates[0]

def id_col(df):
    for c in ['USUBJID', 'RPT', 'patient_id', 'id']:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in ['usubjid', 'rpt', 'patient_id']:
        if c in lower:
            return lower[c]
    raise RuntimeError(f'Could not identify patient ID column: {df.columns.tolist()}')

def load_mainsail():
    feat = pd.read_csv(M_FEATURES)
    tar = pd.read_csv(M_TARGETS)
    fid = id_col(feat)
    tid = id_col(tar)
    feat['patient_id'] = norm_id(feat[fid])
    tar['patient_id'] = norm_id(tar[tid])
    if 'target_available' in tar.columns:
        tar = tar[bool_series(tar['target_available'])].copy()
    tc = target_col(tar)
    tar['target_t'] = pd.to_numeric(tar[tc], errors='coerce')
    cols = ['patient_id'] + FEATURES
    out = feat[cols].merge(tar[['patient_id', 'target_t']], on='patient_id', how='inner')
    for c in FEATURES + ['target_t']:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out = out.dropna(subset=FEATURES + ['target_t'])
    print('\nMAINSAIL target-positive cohort')
    print(out['target_t'].describe())
    return out

def find_csv_with_columns(root, required):
    for p in root.rglob('*.csv'):
        try:
            h = pd.read_csv(p, nrows=3)
        except Exception:
            continue
        if set(required).issubset(h.columns):
            return p
    raise FileNotFoundError(f'Could not find CSV under {root} with columns {required}')

def load_venice_features():
    adsl_path = find_csv_with_columns(VENICE_ROOT, ['USUBJID', 'TMPSA'])
    adlb_path = find_csv_with_columns(VENICE_ROOT, ['USUBJID', 'LBSTRESN', 'ANLVISN'])
    print('\nVENICE ADSL:', adsl_path)
    print('VENICE ADLB:', adlb_path)
    adsl = pd.read_csv(adsl_path)
    adlb = pd.read_csv(adlb_path)
    adsl['patient_id'] = norm_id(adsl['USUBJID'])
    adlb['patient_id'] = norm_id(adlb['USUBJID'])
    adsl['psa0'] = pd.to_numeric(adsl['TMPSA'], errors='coerce')
    base = adsl.loc[adsl['psa0'].notna() & (adsl['psa0'] > 0), ['patient_id', 'psa0']].drop_duplicates('patient_id')
    if 'LBTESTCD' in adlb.columns:
        mask = adlb['LBTESTCD'].astype(str).str.upper().str.contains('PSA', na=False)
        if mask.any():
            adlb = adlb[mask].copy()
    adlb['cycle'] = pd.to_numeric(adlb['ANLVISN'], errors='coerce')
    adlb['psa'] = pd.to_numeric(adlb['LBSTRESN'], errors='coerce')
    if 'LBDY2' in adlb.columns:
        d2 = pd.to_numeric(adlb['LBDY2'], errors='coerce')
    else:
        d2 = pd.Series(np.nan, index=adlb.index)
    if 'LBDY' in adlb.columns:
        d1 = pd.to_numeric(adlb['LBDY'], errors='coerce')
    else:
        d1 = pd.Series(np.nan, index=adlb.index)
    adlb['day'] = d2.where(d2.notna(), d1)
    early = adlb[adlb['cycle'].isin([1, 2, 3]) & adlb['psa'].notna() & (adlb['psa'] > 0)].copy()
    early['nominal_day'] = 21.0 * early['cycle']
    early['day_error'] = (early['day'] - early['nominal_day']).abs()
    early = early.sort_values(['patient_id', 'cycle', 'day_error']).drop_duplicates(['patient_id', 'cycle'], keep='first')
    wide = early.pivot(index='patient_id', columns='cycle', values='psa').rename(columns={1: 'psa1', 2: 'psa2', 3: 'psa3'}).reset_index()
    x = base.merge(wide, on='patient_id', how='inner')
    x = x.dropna(subset=['psa0', 'psa1', 'psa2', 'psa3'])
    for t in [1, 2, 3]:
        x[f'z{t}'] = np.log10(x[f'psa{t}'] / x['psa0'])
    tt = np.array([0.0, 1.0, 2.0, 3.0])

    def calc_slope(r):
        yy = np.array([0.0, r['z1'], r['z2'], r['z3']])
        return np.polyfit(tt, yy, 1)[0]
    x['slope'] = x.apply(calc_slope, axis=1)
    tar = pd.read_csv(V_TARGETS)
    tid = id_col(tar)
    tar['patient_id'] = norm_id(tar[tid])
    if 'target_available' in tar.columns:
        tar = tar[bool_series(tar['target_available'])].copy()
    tc = target_col(tar)
    tar['target_t'] = pd.to_numeric(tar[tc], errors='coerce')
    y = x[['patient_id'] + FEATURES].merge(tar[['patient_id', 'target_t']], on='patient_id', how='inner')
    y = y.dropna(subset=FEATURES + ['target_t'])
    print('\nVENICE target-positive cohort')
    print(y['target_t'].describe())
    if len(y) != 263:
        warnings.warn(f'Expected 263 VENICE targets, got {len(y)}. Check target/feature reconstruction before interpreting cross-trial analyses.')
    return (x[['patient_id'] + FEATURES], y)

def repeated_ridge_cv(df, label):
    X = df[FEATURES].to_numpy(float)
    y = df['target_t'].to_numpy(float)
    rkf = RepeatedKFold(n_splits=5, n_repeats=20, random_state=SEED)
    pred_sum = np.zeros(len(df))
    base_sum = np.zeros(len(df))
    count = np.zeros(len(df))
    for train, test in rkf.split(X):
        model = GridSearchCV(Pipeline([('scale', StandardScaler()), ('ridge', Ridge())]), {'ridge__alpha': RIDGE_ALPHAS}, cv=5, scoring='neg_mean_absolute_error', n_jobs=allocated_threads())
        model.fit(X[train], y[train])
        pred = model.predict(X[test])
        baseline = np.repeat(np.median(y[train]), len(test))
        pred_sum[test] += pred
        base_sum[test] += baseline
        count[test] += 1
    pred = pred_sum / count
    base = base_sum / count
    m_model = regression_metrics(y, pred)
    m_base = regression_metrics(y, base)
    summary = {'analysis': label, 'model': m_model, 'training_fold_median_baseline': m_base, 'baseline_minus_ridge_MAE': m_base['MAE'] - m_model['MAE']}
    out = df[['patient_id', 'target_t']].copy()
    out['ridge_prediction'] = pred
    out['fold_median_prediction'] = base
    return (summary, out)

def repeated_binary_cv(df, cutoff=BINARY_CUTOFF):
    X = df[FEATURES].to_numpy(float)
    y = (df['target_t'].to_numpy(float) >= cutoff).astype(int)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
    pred_sum = np.zeros(len(df))
    count = np.zeros(len(df))
    for train, test in cv.split(X, y):
        model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=5000))])
        model.fit(X[train], y[train])
        pred_sum[test] += model.predict_proba(X[test])[:, 1]
        count[test] += 1
    p = pred_sum / count
    metrics = classification_metrics(y, p)
    out = df[['patient_id', 'target_t']].copy()
    out['late'] = y
    out['p_late'] = p
    return (metrics, out)

def cross_trial_regression(train_df, test_df, train_name, test_name):
    Xtr = train_df[FEATURES].to_numpy(float)
    ytr = train_df['target_t'].to_numpy(float)
    Xte = test_df[FEATURES].to_numpy(float)
    yte = test_df['target_t'].to_numpy(float)
    model = GridSearchCV(Pipeline([('scale', StandardScaler()), ('ridge', Ridge())]), {'ridge__alpha': RIDGE_ALPHAS}, cv=5, scoring='neg_mean_absolute_error', n_jobs=allocated_threads())
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    baseline = np.repeat(np.median(ytr), len(yte))
    result = {'task': 'regression', 'train': train_name, 'test': test_name, 'selected_alpha': float(model.best_params_['ridge__alpha']), 'model': regression_metrics(yte, pred), 'source_median_baseline': regression_metrics(yte, baseline)}
    result['baseline_minus_model_MAE'] = result['source_median_baseline']['MAE'] - result['model']['MAE']
    out = test_df[['patient_id', 'target_t']].copy()
    out['prediction'] = pred
    out['source_median'] = baseline
    return (result, out)

def cross_trial_binary(train_df, test_df, train_name, test_name, cutoff=BINARY_CUTOFF):
    Xtr = train_df[FEATURES].to_numpy(float)
    ytr = (train_df['target_t'].to_numpy(float) >= cutoff).astype(int)
    Xte = test_df[FEATURES].to_numpy(float)
    yte = (test_df['target_t'].to_numpy(float) >= cutoff).astype(int)
    model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=5000))])
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    result = {'task': f'binary_cutoff_{cutoff}', 'train': train_name, 'test': test_name, **classification_metrics(yte, p)}
    out = test_df[['patient_id', 'target_t']].copy()
    out['late'] = yte
    out['p_late'] = p
    return (result, out)

def detect_ref_columns(df):
    lower = {c.lower(): c for c in df.columns}

    def choose(options):
        for x in options:
            if x.lower() in lower:
                return lower[x.lower()]
        return None
    rid = choose(['ref_id', 'reference_id', 'sim_id', 'simulation_id', 'replicate', 'id'])
    tc = choose(['t', 'time', 'cycle', 'drug1_cycle'])
    nc = choose(['N', 'population_size', 'pop_size', 'population', 'n_cells'])
    if rid is None or tc is None or nc is None:
        raise RuntimeError(f'Could not detect reference columns.\nColumns: {df.columns.tolist()}')
    return (rid, tc, nc)

def build_frozen_reference():
    if not REF_TRAJ.exists():
        raise FileNotFoundError(f'Missing reference trajectory file:\n{REF_TRAJ}')
    print('\nLoading frozen reference:')
    print(REF_TRAJ)
    df = pd.read_csv(REF_TRAJ)
    rid, tc, nc = detect_ref_columns(df)
    df[tc] = pd.to_numeric(df[tc], errors='coerce')
    df[nc] = pd.to_numeric(df[nc], errors='coerce')
    df = df.dropna(subset=[rid, tc, nc])
    rows = []
    query_early = np.arange(4, dtype=float) * ABC_SCALE
    query_visits = np.arange(0, ABC_H + 2, dtype=int)
    query_times = query_visits.astype(float) * ABC_SCALE
    for ref_id, g in df.groupby(rid):
        g = g.sort_values(tc).drop_duplicates(tc, keep='last')
        t = g[tc].to_numpy(float)
        n = g[nc].to_numpy(float)
        good = np.isfinite(t) & np.isfinite(n) & (n > 0)
        t = t[good]
        n = n[good]
        if len(t) < 2:
            continue
        order = np.argsort(t)
        t = t[order]
        n = n[order]
        logn = np.log10(n)
        if query_early[0] < t.min() or query_early[-1] > t.max():
            continue
        early_log = np.interp(query_early, t, logn)
        z = early_log - early_log[0]
        slope = np.polyfit(np.array([0.0, 1.0, 2.0, 3.0]), z, 1)[0]
        vals = np.full(len(query_times), np.nan)
        inside = (query_times >= t.min()) & (query_times <= t.max())
        vals[inside] = np.interp(query_times[inside], t, logn)
        target = np.nan
        for j in range(4, ABC_H + 1):
            if not np.all(np.isfinite([vals[j - 1], vals[j], vals[j + 1]])):
                continue
            if vals[j] <= vals[j - 1] and vals[j + 1] > vals[j]:
                target = float(j)
                break
        rows.append({'ref_id': ref_id, 'z1': z[1], 'z2': z[2], 'z3': z[3], 'slope': slope, 'target_t': target})
    ref = pd.DataFrame(rows)
    early_valid = len(ref)
    eligible = ref[ref['target_t'].notna() & (ref['target_t'] <= ABC_H)].copy()
    print('\nReference reconstruction')
    print('early-valid:', early_valid)
    print('H<=12 target-eligible:', len(eligible))
    print('target median:', eligible['target_t'].median())
    expected_early = 4040
    expected_eligible = 2744
    if early_valid != expected_early or len(eligible) != expected_eligible:
        raise RuntimeError(f'\nREFERENCE RECONSTRUCTION DOES NOT MATCH THE FROZEN VENICE PIPELINE.\nObserved early={early_valid}, eligible={len(eligible)}\nExpected early={expected_early}, eligible={expected_eligible}\n\nDo NOT reinterpret MAINSAIL ABC from a different reference construction. We should instead copy the exact feature-building function from your existing run_venice_abc_horizon.py.')
    return eligible

def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v = np.asarray(values)[order]
    w = np.asarray(weights)[order]
    c = np.cumsum(w)
    c /= c[-1]
    return float(np.interp(q, c, v))

def run_frozen_abc(query_df, reference):
    R = reference[['z1', 'z2', 'z3', 'slope']].to_numpy(float)
    R[:, 3] *= ABC_SLOPE_WEIGHT
    yref = reference['target_t'].to_numpy(float)
    nn = NearestNeighbors(n_neighbors=ABC_K, metric='euclidean')
    nn.fit(R)
    Q = query_df[['z1', 'z2', 'z3', 'slope']].to_numpy(float)
    Q[:, 3] *= ABC_SLOPE_WEIGHT
    dist, ind = nn.kneighbors(Q)
    rows = []
    for i in range(len(query_df)):
        d = dist[i]
        yi = yref[ind[i]]
        if np.any(d < 1e-12):
            exact = d < 1e-12
            pred = float(np.mean(yi[exact]))
            q10 = float(np.min(yi[exact]))
            q90 = float(np.max(yi[exact]))
        else:
            w = 1.0 / (d + 1e-12)
            w /= w.sum()
            pred = float(np.sum(w * yi))
            q10 = weighted_quantile(yi, w, 0.1)
            q90 = weighted_quantile(yi, w, 0.9)
        kth = float(d[-1])
        rows.append({'patient_id': query_df.iloc[i]['patient_id'], 'abc_prediction': pred, 'neighbor_q10': q10, 'neighbor_q90': q90, 'k25_distance': kth, 'supported': kth <= ABC_SUPPORT_THRESHOLD, 'distance_over_threshold': kth / ABC_SUPPORT_THRESHOLD})
    return pd.DataFrame(rows)

def venice_locked_features(v_all_features):
    locked = pd.read_csv(V_LOCKED)
    lid = id_col(locked)
    locked['patient_id'] = norm_id(locked[lid])
    return locked[['patient_id']].merge(v_all_features, on='patient_id', how='inner')

def mainsail_forecasting_features():
    f = pd.read_csv(M_FEATURES)
    ids = pd.read_csv(M_FORECAST_IDS)
    fid = id_col(f)
    iid = id_col(ids)
    f['patient_id'] = norm_id(f[fid])
    ids['patient_id'] = norm_id(ids[iid])
    return ids[['patient_id']].merge(f[['patient_id'] + FEATURES], on='patient_id', how='inner')

def main():
    mainsail = load_mainsail()
    v_features_all, venice = load_venice_features()
    master_summary = {}
    print('\n' + '=' * 80)
    print('STEP 1: MAINSAIL INTERNAL REGRESSION')
    print('=' * 80)
    s1_all, p1_all = repeated_ridge_cv(mainsail, 'MAINSAIL_all_targets')
    m_H = mainsail[mainsail['target_t'] <= ABC_H].copy()
    s1_H, p1_H = repeated_ridge_cv(m_H, 'MAINSAIL_target_4_to_12')
    print(json.dumps(s1_all, indent=2))
    print(json.dumps(s1_H, indent=2))
    p1_all.to_csv(OUT / 'step1_mainsail_regression_all_predictions.csv', index=False)
    p1_H.to_csv(OUT / 'step1_mainsail_regression_H12_predictions.csv', index=False)
    master_summary['step1'] = {'all_targets': s1_all, 'H12': s1_H}
    print('\n' + '=' * 80)
    print('STEP 2: MAINSAIL BINARY CUTOFF 7')
    print('=' * 80)
    s2, p2 = repeated_binary_cv(mainsail, BINARY_CUTOFF)
    print(json.dumps(s2, indent=2))
    p2.to_csv(OUT / 'step2_mainsail_binary_cutoff7_predictions.csv', index=False)
    master_summary['step2'] = s2
    print('\n' + '=' * 80)
    print('STEP 3: CROSS-TRIAL TRANSFER')
    print('=' * 80)
    transfer = []
    r_vm, p_vm = cross_trial_regression(venice, mainsail, 'VENICE', 'MAINSAIL')
    r_mv, p_mv = cross_trial_regression(mainsail, venice, 'MAINSAIL', 'VENICE')
    transfer.extend([r_vm, r_mv])
    p_vm.to_csv(OUT / 'step3_regression_VENICE_to_MAINSAIL.csv', index=False)
    p_mv.to_csv(OUT / 'step3_regression_MAINSAIL_to_VENICE.csv', index=False)
    b_vm, bp_vm = cross_trial_binary(venice, mainsail, 'VENICE', 'MAINSAIL')
    b_mv, bp_mv = cross_trial_binary(mainsail, venice, 'MAINSAIL', 'VENICE')
    transfer.extend([b_vm, b_mv])
    bp_vm.to_csv(OUT / 'step3_binary_VENICE_to_MAINSAIL.csv', index=False)
    bp_mv.to_csv(OUT / 'step3_binary_MAINSAIL_to_VENICE.csv', index=False)
    print(json.dumps(transfer, indent=2))
    master_summary['step3'] = transfer
    with open(OUT / 'mainsail_manuscript_validation_summary.json', 'w') as f:
        json.dump(master_summary, f, indent=2)
if __name__ == '__main__':
    main()
