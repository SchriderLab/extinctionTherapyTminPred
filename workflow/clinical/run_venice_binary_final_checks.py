from rescue.runtime import workspace_path, allocated_threads
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, brier_score_loss
ROOT = Path('.')
PRED = ROOT / 'venice_k1_abc_reference_10k' / 'hierarchical_biexponential_empirical_v1' / 'hierarchical_biexponential_predictions.csv'
ADSL = ROOT / 'data' / 'venice_csv' / 'adsl.csv'
ADLB = ROOT / 'data' / 'venice_csv' / 'adlb.csv'
OUT = ROOT / 'venice_k1_abc_reference_10k' / 'venice_binary_final_checks_v1'
OUT.mkdir(parents=True, exist_ok=True)
C_GRID = np.logspace(-3, 3, 13)
BOOT = 5000
LEARNING_REPS = 50
FRACS = [0.25, 0.4, 0.55, 0.7, 0.85, 1.0]

def make_model(seed, inner_splits=4):
    return GridSearchCV(Pipeline([('imp', SimpleImputer(strategy='median')), ('scale', StandardScaler()), ('lr', LogisticRegression(class_weight='balanced', max_iter=5000, solver='liblinear'))]), {'lr__C': C_GRID}, cv=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed), scoring='roc_auc', n_jobs=allocated_threads(), refit=True)

def metrics(y, prob):
    y = np.asarray(y, int)
    prob = np.asarray(prob, float)
    pred = (prob >= 0.5).astype(int)
    return {'n': len(y), 'n_early': int((y == 0).sum()), 'n_late': int((y == 1).sum()), 'AUROC': float(roc_auc_score(y, prob)), 'balanced_accuracy_05': float(balanced_accuracy_score(y, pred)), 'macro_F1_05': float(f1_score(y, pred, average='macro')), 'Brier': float(brier_score_loss(y, prob))}

def paired_bootstrap(y, p0, p1, metric='auc', nboot=5000, seed=31):
    y = np.asarray(y, int)
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        if metric == 'auc':
            v = roc_auc_score(yy, p1[idx]) - roc_auc_score(yy, p0[idx])
        else:
            v = balanced_accuracy_score(yy, (p1[idx] >= 0.5).astype(int)) - balanced_accuracy_score(yy, (p0[idx] >= 0.5).astype(int))
        vals.append(v)
    vals = np.asarray(vals)
    return (float(vals.mean()), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))
p = pd.read_csv(PRED)
base = p[p['scan_count'].eq(6)].drop_duplicates('id').copy()
base['id'] = base['id'].astype(str)
base = base[['id', 'fold', 'exp_tmin']]
base['late8'] = (base['exp_tmin'] > 8.0).astype(int)
a = pd.read_csv(ADSL, low_memory=False).copy()
a['id'] = a['USUBJID'].astype(str)
for c in ['TMPSA', 'ECOGIVRN', 'HEIGHTBL', 'WEIGHTBL']:
    a[c] = pd.to_numeric(a[c], errors='coerce')
a['log_abs_psa0'] = np.log(np.clip(a['TMPSA'], 1e-12, None))
a['BMI'] = a['WEIGHTBL'] / (a['HEIGHTBL'] / 100.0) ** 2
a = a[['id', 'TMPSA', 'log_abs_psa0', 'ECOGIVRN', 'BMI']].drop_duplicates('id')
l = pd.read_csv(ADLB, low_memory=False).copy()
l['id'] = l['USUBJID'].astype(str)
l['LBSTRESN'] = pd.to_numeric(l['LBSTRESN'], errors='coerce')
LABS = ['TESTOSTERONE', 'HEMOGLOBIN', 'ALKALINE PHOSPHATASE', 'CREATININE', 'PHOSPHORUS', 'CALCIUM', 'ALBUMIN']
bl = l[l['LBBLFL'].astype(str).str.upper().eq('Y') & l['LBTEST'].astype(str).str.upper().isin(LABS)].copy()
bl['LBTEST'] = bl['LBTEST'].astype(str).str.upper()
wlab = bl.pivot_table(index='id', columns='LBTEST', values='LBSTRESN', aggfunc='median').reset_index()
wlab.columns.name = None
wlab['log_TESTOSTERONE'] = np.log(np.clip(pd.to_numeric(wlab['TESTOSTERONE'], errors='coerce'), 1e-06, None))
wlab['log_ALP'] = np.log1p(pd.to_numeric(wlab['ALKALINE PHOSPHATASE'], errors='coerce'))
is_psa = l['LBTESTCD'].astype(str).str.upper().eq('PSA') | l['LBTES1CD'].astype(str).str.upper().eq('PSA') | l['LBTEST'].astype(str).str.upper().eq('PSA')
lp = l[is_psa].copy()
lp['cycle'] = pd.to_numeric(lp['ANLVISN'], errors='coerce')
lp['psa'] = pd.to_numeric(lp['LBSTRESN'], errors='coerce')
lp = lp[lp['cycle'].between(1, 5) & lp['psa'].gt(0)]
wp = lp.pivot_table(index='id', columns='cycle', values='psa', aggfunc='median')
wp = wp.rename(columns={float(c): f'psa{c}' for c in range(1, 6)}).reset_index()
d = base.merge(a, on='id', how='left').merge(wlab, on='id', how='left').merge(wp, on='id', how='left')
d['y0'] = 0.0
for c in range(1, 6):
    d[f'y{c}'] = np.log(pd.to_numeric(d[f'psa{c}'], errors='coerce') / d['TMPSA'])
    d[f'slope{c}'] = d[f'y{c}'] - d[f'y{c - 1}']
PSA_FEATURES = [f'y{c}' for c in range(1, 6)] + [f'slope{c}' for c in range(1, 6)]
BASELINE_FEATURES = ['log_abs_psa0', 'ECOGIVRN', 'BMI', 'log_TESTOSTERONE', 'HEMOGLOBIN', 'log_ALP', 'CREATININE', 'PHOSPHORUS', 'CALCIUM', 'ALBUMIN']
FEATURE_SETS = {'relative_psa': PSA_FEATURES, 'baseline_only': BASELINE_FEATURES, 'relative_psa_plus_full_baseline': PSA_FEATURES + BASELINE_FEATURES}
print('Fixed cohort n =', len(d))
print('Class counts =', d['late8'].value_counts().sort_index().to_dict())
print('\n' + '=' * 110)
print('EXPERIMENT 1: 6-SCAN BINARY FEATURE ABLATION')
pred_rows = []
summary_rows = []
all_probs = {}
for model_name, features in FEATURE_SETS.items():
    oof = np.full(len(d), np.nan)
    chosen_c = []
    for fold in sorted(d['fold'].dropna().unique()):
        tr = d['fold'].ne(fold)
        te = d['fold'].eq(fold)
        ytr = d.loc[tr, 'late8'].to_numpy(int)
        inner = int(min(4, pd.Series(ytr).value_counts().min()))
        gs = make_model(4100 + int(fold), inner)
        gs.fit(d.loc[tr, features], ytr)
        oof[te.to_numpy()] = gs.predict_proba(d.loc[te, features])[:, 1]
        chosen_c.append(gs.best_params_['lr__C'])
    m = metrics(d['late8'], oof)
    all_probs[model_name] = oof.copy()
    summary_rows.append({'model': model_name, **m, 'median_C': float(np.median(chosen_c))})
    print(f"{model_name:34s} AUROC={m['AUROC']:.3f} balacc={m['balanced_accuracy_05']:.3f} macroF1={m['macro_F1_05']:.3f} Brier={m['Brier']:.3f} medianC={np.median(chosen_c):.4g}")
    for i in range(len(d)):
        pred_rows.append({'id': d.iloc[i]['id'], 'fold': d.iloc[i]['fold'], 'exp_tmin': d.iloc[i]['exp_tmin'], 'late8': d.iloc[i]['late8'], 'model': model_name, 'prob_late8': oof[i]})
y = d['late8'].to_numpy(int)
for comp in ['baseline_only', 'relative_psa_plus_full_baseline']:
    da, lo, hi = paired_bootstrap(y, all_probs['relative_psa'], all_probs[comp], 'auc', BOOT, 31)
    db, lob, hib = paired_bootstrap(y, all_probs['relative_psa'], all_probs[comp], 'balacc', BOOT, 32)
    print(f'{comp} minus relative_psa: delta AUROC={da:.3f} [{lo:.3f},{hi:.3f}], delta balacc={db:.3f} [{lob:.3f},{hib:.3f}]')
pd.DataFrame(pred_rows).to_csv(OUT / 'binary_feature_ablation_predictions.csv', index=False)
pd.DataFrame(summary_rows).to_csv(OUT / 'binary_feature_ablation_summary.csv', index=False)
print('\n' + '=' * 110)
print('EXPERIMENT 2: 6-SCAN REAL-VENICE TRAINING-SIZE CURVE')
raw_curve = []
for frac in FRACS:
    reps = 1 if frac == 1.0 else LEARNING_REPS
    print(f'training fraction={frac:.2f}, repeats={reps}')
    for rep in range(reps):
        oof = np.full(len(d), np.nan)
        train_ns = []
        for fold in sorted(d['fold'].dropna().unique()):
            train_mask = d['fold'].ne(fold)
            test_mask = d['fold'].eq(fold)
            train_idx = np.where(train_mask.to_numpy())[0]
            y_train_full = d.iloc[train_idx]['late8'].to_numpy(int)
            if frac < 1.0:
                splitter = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=900000 + int(frac * 1000) * 1000 + rep * 10 + int(fold))
                sub_rel, _ = next(splitter.split(np.zeros(len(train_idx)), y_train_full))
                sub_idx = train_idx[sub_rel]
            else:
                sub_idx = train_idx
            train_ns.append(len(sub_idx))
            ysub = d.iloc[sub_idx]['late8'].to_numpy(int)
            inner = int(min(4, pd.Series(ysub).value_counts().min()))
            gs = make_model(8000 + rep * 10 + int(fold), inner)
            gs.fit(d.iloc[sub_idx][PSA_FEATURES], ysub)
            test_idx = np.where(test_mask.to_numpy())[0]
            oof[test_idx] = gs.predict_proba(d.iloc[test_idx][PSA_FEATURES])[:, 1]
        ok = np.isfinite(oof)
        if ok.sum() != len(d):
            continue
        m = metrics(d.loc[ok, 'late8'], oof[ok])
        raw_curve.append({'train_fraction': frac, 'repeat': rep, 'mean_train_n_per_fold': float(np.mean(train_ns)), **m})
curve = pd.DataFrame(raw_curve)
curve.to_csv(OUT / 'binary_learning_curve_raw.csv', index=False)
rows = []
for frac, z in curve.groupby('train_fraction'):
    row = {'train_fraction': frac, 'repeats': len(z), 'mean_train_n_per_fold': z['mean_train_n_per_fold'].mean()}
    for metric in ['AUROC', 'balanced_accuracy_05', 'macro_F1_05', 'Brier']:
        row[f'{metric}_mean'] = z[metric].mean()
        row[f'{metric}_q025'] = z[metric].quantile(0.025)
        row[f'{metric}_q975'] = z[metric].quantile(0.975)
    rows.append(row)
curve_summary = pd.DataFrame(rows).sort_values('train_fraction')
curve_summary.to_csv(OUT / 'binary_learning_curve_summary.csv', index=False)
print('\nLEARNING CURVE SUMMARY')
print(curve_summary.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print('\nWrote:')
for f in sorted(OUT.glob('*.csv')):
    print(f)
