from rescue.runtime import workspace_path, allocated_threads
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
try:
    from xgboost import XGBRegressor
except ImportError:
    raise SystemExit('xgboost is not installed. Run: pip install xgboost')
sys.path.insert(0, str(Path('scripts').resolve()))
from run_venice_k1_abc_scan_budget import normalize_id, numeric, analytic_tmin, load_clinical_long, build_clinical_features
SEED = 20260919
N_FOLDS = 5
REF_DIR = Path('venice_k1_abc_reference_10k')
REF_META = REF_DIR / 'abc_reference_features_usable.csv'
SCAN_DIR = REF_DIR / 'venice_scan_budget_global_tmin_v1'
REF_SCAN = SCAN_DIR / 'reference_scan_features_cycles0_5.csv'
PRIOR_PRED = REF_DIR / 'venice_real_4scan' / 'venice_real_4scan_abc_predictions.csv'
CLINICAL_LONG = Path('data/source_data/venice_psa_scheduled_long.csv')
ADSL = Path('data/venice_csv/adsl.csv')
OUTDIR = REF_DIR / 'venice_xgboost_tmin_v1'
OUTDIR.mkdir(parents=True, exist_ok=True)
MAX_SCANS = 6
MAX_LAST_CYCLE = 5
FEATURE_SETS = ['levels', 'levels_slopes', 'levels_slopes_curvature']
PARAM_GRID = [{'max_depth': 2, 'learning_rate': 0.03, 'n_estimators': 300, 'min_child_weight': 1}, {'max_depth': 2, 'learning_rate': 0.05, 'n_estimators': 500, 'min_child_weight': 5}, {'max_depth': 3, 'learning_rate': 0.03, 'n_estimators': 500, 'min_child_weight': 1}, {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300, 'min_child_weight': 5}, {'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500, 'min_child_weight': 5}]

def make_model(params):
    return XGBRegressor(objective='reg:squarederror', eval_metric='mae', tree_method='hist', subsample=0.85, colsample_bytree=0.9, reg_lambda=1.0, reg_alpha=0.0, random_state=SEED, n_jobs=1, verbosity=0, **params)

def feature_matrix(df, last_cycle, feature_set):
    parts = []
    names = []
    for i in range(1, last_cycle + 1):
        col = f'y{i}'
        parts.append(df[col].to_numpy(float))
        names.append(col)
    if feature_set in ('levels_slopes', 'levels_slopes_curvature'):
        for i in range(1, last_cycle + 1):
            col = f'dy{i - 1}{i}'
            parts.append(df[col].to_numpy(float))
            names.append(col)
    if feature_set == 'levels_slopes_curvature':
        for i in range(2, last_cycle + 1):
            prev_slope = df[f'dy{i - 2}{i - 1}'].to_numpy(float)
            curr_slope = df[f'dy{i - 1}{i}'].to_numpy(float)
            parts.append(curr_slope - prev_slope)
            names.append(f'curvature_{i - 2}_{i - 1}_{i}')
    X = np.column_stack(parts)
    return (X, names)

def metrics(y, pred):
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    ae = np.abs(pred - y)
    rho = spearmanr(y, pred).statistic
    return {'bias': float(np.mean(pred - y)), 'MAE': float(np.mean(ae)), 'median_AE': float(np.median(ae)), 'RMSE': float(np.sqrt(mean_squared_error(y, pred))), 'within_1': float(np.mean(ae <= 1.0)), 'within_2': float(np.mean(ae <= 2.0)), 'Spearman_rho': float(rho)}

def cv_one(X, y, params):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan, dtype=float)
    for train_idx, valid_idx in kf.split(X):
        model = make_model(params)
        model.fit(X[train_idx], y[train_idx])
        oof[valid_idx] = model.predict(X[valid_idx])
    return (metrics(y, oof), oof)
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
ref = ref[np.isfinite(ref['t_min_true'])].copy()
print('Clean reference trajectories:', len(ref))
meta = pd.read_csv(PRIOR_PRED)
meta['id'] = meta['id'].map(normalize_id)
meta = meta[['id', 'support_class_full_trajectory', 'clinical_d', 'clinical_q', 'clinical_g']].drop_duplicates('id').copy()
meta['clinical_d'] = numeric(meta['clinical_d'])
meta['clinical_q'] = numeric(meta['clinical_q'])
meta['clinical_g'] = numeric(meta['clinical_g'])
meta['global_fit_tmin'] = analytic_tmin(meta['clinical_q'].to_numpy(float), meta['clinical_d'].to_numpy(float), meta['clinical_g'].to_numpy(float))
clinical_long = load_clinical_long(CLINICAL_LONG, ADSL)
clinical_features = build_clinical_features(clinical_long, MAX_LAST_CYCLE)
clinical = meta.merge(clinical_features, on='id', how='left')
needed_max = []
for i in range(1, MAX_LAST_CYCLE + 1):
    needed_max.append(f'y{i}')
    needed_max.append(f'dy{i - 1}{i}')
clinical = clinical[(clinical['support_class_full_trajectory'] == 'well_supported') & np.isfinite(clinical['global_fit_tmin']) & (clinical['global_fit_tmin'] > MAX_LAST_CYCLE)].dropna(subset=needed_max).copy()
print('Fixed VENICE cohort:', len(clinical))
if len(clinical) != 133:
    print('WARNING: expected 133 fixed patients, found', len(clinical))
cv_rows = []
summary_rows = []
prediction_rows = []
importance_rows = []
for scan_count in range(2, MAX_SCANS + 1):
    last_cycle = scan_count - 1
    needed = []
    for i in range(1, last_cycle + 1):
        needed.append(f'y{i}')
        needed.append(f'dy{i - 1}{i}')
    ref_budget = ref.dropna(subset=needed + ['t_min_true']).copy()
    y_ref = ref_budget['t_min_true'].to_numpy(float)
    best = None
    print('\n' + '=' * 72)
    print(f'{scan_count} scans: cycles 0-{last_cycle}')
    for feature_set in FEATURE_SETS:
        X_ref, feature_names = feature_matrix(ref_budget, last_cycle, feature_set)
        for param_id, params in enumerate(PARAM_GRID):
            m, _ = cv_one(X_ref, y_ref, params)
            row = {'scan_count': scan_count, 'last_cycle': last_cycle, 'feature_set': feature_set, 'param_id': param_id, **params, **{f'cv_{k}': v for k, v in m.items()}}
            cv_rows.append(row)
            candidate = {'feature_set': feature_set, 'feature_names': feature_names, 'param_id': param_id, 'params': params, 'metrics': m}
            if best is None:
                best = candidate
            elif m['MAE'] < best['metrics']['MAE']:
                best = candidate
            elif np.isclose(m['MAE'], best['metrics']['MAE']) and m['median_AE'] < best['metrics']['median_AE']:
                best = candidate
    print('Selected:', best['feature_set'], 'param', best['param_id'])
    print('Reference CV MAE:', f"{best['metrics']['MAE']:.3f}")
    print('Reference CV median AE:', f"{best['metrics']['median_AE']:.3f}")
    X_ref, feature_names = feature_matrix(ref_budget, last_cycle, best['feature_set'])
    model = make_model(best['params'])
    model.fit(X_ref, y_ref)
    X_clin, _ = feature_matrix(clinical, last_cycle, best['feature_set'])
    pred = model.predict(X_clin)
    true = clinical['global_fit_tmin'].to_numpy(float)
    cm = metrics(true, pred)
    print('VENICE MAE:', f"{cm['MAE']:.3f}")
    print('VENICE median AE:', f"{cm['median_AE']:.3f}")
    print('VENICE Spearman rho:', f"{cm['Spearman_rho']:.3f}")
    print('Within +/-1:', f"{cm['within_1']:.3f}")
    print('Within +/-2:', f"{cm['within_2']:.3f}")
    summary_rows.append({'scan_count': scan_count, 'last_cycle': last_cycle, 'n': len(clinical), 'feature_set': best['feature_set'], 'param_id': best['param_id'], 'reference_cv_MAE': best['metrics']['MAE'], 'reference_cv_median_AE': best['metrics']['median_AE'], **cm})
    for pid, target, p in zip(clinical['id'], true, pred):
        prediction_rows.append({'id': pid, 'scan_count': scan_count, 'true_tmin': target, 'xgb_pred_tmin': float(p), 'absolute_error': float(abs(p - target))})
    for fname, importance in zip(feature_names, model.feature_importances_):
        importance_rows.append({'scan_count': scan_count, 'feature_set': best['feature_set'], 'feature': fname, 'importance': float(importance)})
cv_df = pd.DataFrame(cv_rows)
summary = pd.DataFrame(summary_rows)
predictions = pd.DataFrame(prediction_rows)
importance = pd.DataFrame(importance_rows)
abc_file = SCAN_DIR / 'scan_budget_fixed_common_summary.csv'
if abc_file.exists():
    abc = pd.read_csv(abc_file)
    abc = abc[['scan_count', 'MAE', 'median_AE', 'Spearman_rho']].copy()
    abc = abc.rename(columns={'MAE': 'ABC_MAE', 'median_AE': 'ABC_median_AE', 'Spearman_rho': 'ABC_Spearman_rho'})
    summary = summary.merge(abc, on='scan_count', how='left')
    summary['XGB_minus_ABC_MAE'] = summary['MAE'] - summary['ABC_MAE']
cv_df.to_csv(OUTDIR / 'xgb_reference_cv.csv', index=False)
summary.to_csv(OUTDIR / 'xgb_venice_summary.csv', index=False)
predictions.to_csv(OUTDIR / 'xgb_venice_predictions.csv', index=False)
importance.to_csv(OUTDIR / 'xgb_feature_importance.csv', index=False)
print('\n' + '=' * 90)
print('FINAL FIXED-COHORT VENICE RESULTS')
cols = ['scan_count', 'n', 'feature_set', 'reference_cv_MAE', 'MAE', 'median_AE', 'within_1', 'within_2', 'Spearman_rho']
if 'ABC_MAE' in summary.columns:
    cols += ['ABC_MAE', 'XGB_minus_ABC_MAE']
print(summary[cols].to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print('\nNegative XGB_minus_ABC_MAE means XGBoost beat ABC.')
print('\nWrote:')
print(OUTDIR / 'xgb_reference_cv.csv')
print(OUTDIR / 'xgb_venice_summary.csv')
print(OUTDIR / 'xgb_venice_predictions.csv')
print(OUTDIR / 'xgb_feature_importance.csv')
