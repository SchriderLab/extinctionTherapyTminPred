from rescue.runtime import workspace_path, allocated_threads
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve, confusion_matrix, brier_score_loss
sys.path.insert(0, str(Path('scripts').resolve()))
from run_venice_k1_abc_scan_budget import normalize_id, numeric, analytic_tmin, load_clinical_long, build_clinical_features, feature_arrays, scales_from_reference, batch_distances
TMIN_CUTOFF = 8.0
MIN_SCANS = 2
MAX_SCANS = 6
MAX_LAST_CYCLE = 5
SLOPE_GRID = [0.0, 0.25, 0.5, 1.0, 2.0]
ACCEPT_FRAC_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
CV_FOLDS = 5
SEED = 20260919
CHUNK = 128
REF_DIR = Path('venice_k1_abc_reference_10k')
REF_META = REF_DIR / 'abc_reference_features_usable.csv'
SCAN_DIR = REF_DIR / 'venice_scan_budget_global_tmin_v1'
REF_SCAN = SCAN_DIR / 'reference_scan_features_cycles0_5.csv'
PRIOR_PRED = REF_DIR / 'venice_real_4scan' / 'venice_real_4scan_abc_predictions.csv'
CLINICAL_LONG = Path('data/source_data/venice_psa_scheduled_long.csv')
ADSL = Path('data/venice_csv/adsl.csv')
OUTDIR = REF_DIR / 'venice_rejection_abc_tmin8_v1'
OUTDIR.mkdir(parents=True, exist_ok=True)

def choose_probability_threshold(y_true, prob):
    y_true = np.asarray(y_true, dtype=int)
    prob = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    fpr, tpr, thresholds = roc_curve(y_true, prob)
    finite = np.isfinite(thresholds)
    fpr = fpr[finite]
    tpr = tpr[finite]
    thresholds = thresholds[finite]
    youden = tpr - fpr
    best_value = np.max(youden)
    candidates = np.flatnonzero(np.isclose(youden, best_value))
    best = candidates[np.argmin(np.abs(thresholds[candidates] - 0.5))]
    return float(thresholds[best])

def classification_metrics(y_true, prob, threshold):
    y_true = np.asarray(y_true, dtype=int)
    prob = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    early_recall = tn / (tn + fp) if tn + fp else np.nan
    late_recall = tp / (tp + fn) if tp + fn else np.nan
    return {'n': len(y_true), 'n_early': int(np.sum(y_true == 0)), 'n_late': int(np.sum(y_true == 1)), 'probability_threshold': threshold, 'accuracy': accuracy_score(y_true, pred), 'balanced_accuracy': balanced_accuracy_score(y_true, pred), 'AUROC': roc_auc_score(y_true, prob), 'early_recall': early_recall, 'late_recall': late_recall, 'brier': brier_score_loss(y_true, prob), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

def rejection_cv(ref, last_cycle):
    ref_level, ref_slope, _, _ = feature_arrays(ref, last_cycle)
    y = ref['late_class'].to_numpy(int)
    sd_level, sd_slope = scales_from_reference(ref_level, ref_slope)
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    collectors = {(sw, frac): {'true': [], 'prob': []} for sw in SLOPE_GRID for frac in ACCEPT_FRAC_GRID}
    for train_idx, valid_idx in splitter.split(ref_level, y):
        train_level = ref_level[train_idx]
        train_slope = ref_slope[train_idx]
        train_y = y[train_idx]
        n_train = len(train_idx)
        accept_counts = {frac: max(5, int(round(frac * n_train))) for frac in ACCEPT_FRAC_GRID}
        for start in range(0, len(valid_idx), CHUNK):
            query_idx = valid_idx[start:start + CHUNK]
            level_d, slope_d = batch_distances(ref_level[query_idx], ref_slope[query_idx], train_level, train_slope, sd_level, sd_slope)
            for sw in SLOPE_GRID:
                distance = level_d + np.float32(sw) * slope_d
                order = np.argsort(distance, axis=1)
                sorted_labels = train_y[order]
                cumulative_late = np.cumsum(sorted_labels, axis=1)
                for frac in ACCEPT_FRAC_GRID:
                    n_accept = min(accept_counts[frac], n_train)
                    prob = cumulative_late[:, n_accept - 1] / n_accept
                    collectors[sw, frac]['true'].extend(y[query_idx].tolist())
                    collectors[sw, frac]['prob'].extend(prob.tolist())
    rows = []
    for (sw, frac), values in collectors.items():
        threshold = choose_probability_threshold(values['true'], values['prob'])
        m = classification_metrics(values['true'], values['prob'], threshold)
        rows.append({'scan_count': last_cycle + 1, 'last_cycle': last_cycle, 'slope_weight': sw, 'accept_fraction': frac, 'approx_accept_n_full': int(round(frac * len(ref))), 'sd_level': sd_level, 'sd_slope': sd_slope, **m})
    out = pd.DataFrame(rows)
    out = out.sort_values(['balanced_accuracy', 'AUROC', 'brier'], ascending=[False, False, True]).reset_index(drop=True)
    return (out, sd_level, sd_slope)

def predict_rejection(clinical, ref, last_cycle, slope_weight, accept_fraction, sd_level, sd_slope):
    query_level, query_slope, _, _ = feature_arrays(clinical, last_cycle)
    ref_level, ref_slope, _, _ = feature_arrays(ref, last_cycle)
    ref_y = ref['late_class'].to_numpy(int)
    level_d, slope_d = batch_distances(query_level, query_slope, ref_level, ref_slope, sd_level, sd_slope)
    distance = level_d + np.float32(slope_weight) * slope_d
    n_accept = max(5, int(round(accept_fraction * len(ref))))
    n_accept = min(n_accept, len(ref))
    idx = np.argpartition(distance, kth=n_accept - 1, axis=1)[:, :n_accept]
    accepted_labels = ref_y[idx]
    prob_late = np.mean(accepted_labels, axis=1)
    return (prob_late, n_accept)
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
ref = ref[np.isfinite(ref['t_min_true']) & (ref['t_min_true'] > MAX_LAST_CYCLE)].reset_index(drop=True)
ref['late_class'] = (ref['t_min_true'] > TMIN_CUTOFF).astype(int)
meta = pd.read_csv(PRIOR_PRED)
meta['id'] = meta['id'].map(normalize_id)
meta = meta[['id', 'support_class_full_trajectory', 'clinical_d', 'clinical_q', 'clinical_g']].drop_duplicates('id').copy()
for col in ['clinical_d', 'clinical_q', 'clinical_g']:
    meta[col] = numeric(meta[col])
meta['global_fit_tmin'] = analytic_tmin(meta['clinical_q'].to_numpy(float), meta['clinical_d'].to_numpy(float), meta['clinical_g'].to_numpy(float))
clinical_long = load_clinical_long(CLINICAL_LONG, ADSL)
clinical_features = build_clinical_features(clinical_long, MAX_LAST_CYCLE)
clinical = meta.merge(clinical_features, on='id', how='left')
needed = []
for i in range(1, MAX_LAST_CYCLE + 1):
    needed += [f'y{i}', f'dy{i - 1}{i}']
clinical = clinical[(clinical['support_class_full_trajectory'] == 'well_supported') & np.isfinite(clinical['global_fit_tmin']) & (clinical['global_fit_tmin'] > MAX_LAST_CYCLE)].dropna(subset=needed).reset_index(drop=True)
clinical['late_class'] = (clinical['global_fit_tmin'] > TMIN_CUTOFF).astype(int)
print('=' * 72)
print('REJECTION ABC: t_min cutoff = 8')
print('early = 5 < t_min <= 8')
print('late  = t_min > 8')
print('\nREFERENCE')
print('n =', len(ref))
print(ref['late_class'].value_counts().rename(index={0: 'early', 1: 'late'}))
print('\nVENICE FIXED COHORT')
print('n =', len(clinical))
print(clinical['late_class'].value_counts().rename(index={0: 'early', 1: 'late'}))
all_cv = []
summary_rows = []
prediction_rows = []
for scan_count in range(MIN_SCANS, MAX_SCANS + 1):
    last_cycle = scan_count - 1
    cv, sd_level, sd_slope = rejection_cv(ref, last_cycle)
    best = cv.iloc[0]
    all_cv.append(cv)
    prob, n_accept = predict_rejection(clinical, ref, last_cycle, float(best['slope_weight']), float(best['accept_fraction']), sd_level, sd_slope)
    threshold = float(best['probability_threshold'])
    m = classification_metrics(clinical['late_class'], prob, threshold)
    summary_rows.append({'scan_count': scan_count, 'last_cycle': last_cycle, 'tmin_cutoff': TMIN_CUTOFF, 'selected_slope_weight': float(best['slope_weight']), 'selected_accept_fraction': float(best['accept_fraction']), 'selected_accept_n': int(n_accept), 'selected_probability_threshold': threshold, 'reference_cv_balanced_accuracy': float(best['balanced_accuracy']), 'reference_cv_AUROC': float(best['AUROC']), **m})
    pred_class = (prob >= threshold).astype(int)
    for pid, true_tmin, true_class, p, pc in zip(clinical['id'], clinical['global_fit_tmin'], clinical['late_class'], prob, pred_class):
        prediction_rows.append({'id': pid, 'scan_count': scan_count, 'true_tmin': true_tmin, 'true_late': true_class, 'prob_late': p, 'probability_threshold': threshold, 'pred_late': pc})
    print('\n' + '-' * 72)
    print(f'{scan_count} scans')
    print(f"slope={best['slope_weight']}, accept={100 * best['accept_fraction']:.1f}% (n={n_accept}), prob cutoff={threshold:.3f}")
    print(f"Reference CV: balanced={best['balanced_accuracy']:.3f}, AUROC={best['AUROC']:.3f}")
    print(f"VENICE: balanced={m['balanced_accuracy']:.3f}, AUROC={m['AUROC']:.3f}, early recall={m['early_recall']:.3f}, late recall={m['late_recall']:.3f}")
cv_out = pd.concat(all_cv, ignore_index=True)
summary = pd.DataFrame(summary_rows)
predictions = pd.DataFrame(prediction_rows)
weighted_file = REF_DIR / 'venice_binary_tmin8_v1' / 'binary_tmin8_venice_summary.csv'
if weighted_file.exists():
    weighted = pd.read_csv(weighted_file)
    weighted = weighted[['scan_count', 'balanced_accuracy', 'AUROC', 'early_recall', 'late_recall']].rename(columns={'balanced_accuracy': 'weighted_ABC_balanced_accuracy', 'AUROC': 'weighted_ABC_AUROC', 'early_recall': 'weighted_ABC_early_recall', 'late_recall': 'weighted_ABC_late_recall'})
    summary = summary.merge(weighted, on='scan_count', how='left')
    summary['rejection_minus_weighted_AUROC'] = summary['AUROC'] - summary['weighted_ABC_AUROC']
    summary['rejection_minus_weighted_balanced'] = summary['balanced_accuracy'] - summary['weighted_ABC_balanced_accuracy']
cv_out.to_csv(OUTDIR / 'rejection_reference_cv.csv', index=False)
summary.to_csv(OUTDIR / 'rejection_venice_summary.csv', index=False)
predictions.to_csv(OUTDIR / 'rejection_venice_predictions.csv', index=False)
print('\n' + '=' * 120)
print('FINAL REJECTION ABC RESULTS')
columns = ['scan_count', 'n', 'n_early', 'n_late', 'selected_accept_fraction', 'selected_accept_n', 'selected_probability_threshold', 'early_recall', 'late_recall', 'balanced_accuracy', 'AUROC', 'reference_cv_balanced_accuracy', 'reference_cv_AUROC']
if 'weighted_ABC_AUROC' in summary.columns:
    columns += ['weighted_ABC_balanced_accuracy', 'weighted_ABC_AUROC', 'rejection_minus_weighted_balanced', 'rejection_minus_weighted_AUROC']
print(summary[columns].to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print('\nPositive rejection-minus-weighted values mean rejection ABC performed better.')
print('\nWrote:')
print(OUTDIR / 'rejection_reference_cv.csv')
print(OUTDIR / 'rejection_venice_summary.csv')
print(OUTDIR / 'rejection_venice_predictions.csv')
