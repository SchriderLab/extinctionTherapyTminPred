"""
VENICE sequential nadir-window classifier: +/- 1 observed scan around the
first confirmed local PSA nadir.

Primary label at a current scan t:
    y_pm1(t) = 1  if event_index - t is in {-1, 0, +1}
             = 0  if t is earlier than one scan before the event and the label
                  is ascertainable
             = NA if follow-up is insufficient to rule out an event in-window.

Interpretation:
  +1 = one observed scan BEFORE the nadir
   0 = at the nadir
  -1 = one observed scan AFTER the nadir

The nadir event is inherited from the sequential analysis:
  first local PSA minimum after the selected declining phase that is followed
  by two higher PSA observations.

Predictors are causal and use PSA history only through the current scan.
The physics-informed model uses a robust Bayesian state-space filter and a
monotone hazard mapping with three pre-specified features:
  - near_zero_velocity      = -|latent log-PSA velocity|
  - toward_zero_1           = positive recent movement of velocity toward 0
  - smooth_toward_zero_2    = persistent movement toward 0 across 2 intervals

All filter hyperparameters and classifiers are estimated using OUTER TRAINING
patients only. The outer held-out patient never contributes to its own model.
A time-only model is fit on the same folds as the baseline.

Outputs:
  pm1_window_summary.csv
  pm1_window_outer_fold_metrics.csv
  pm1_window_oof_predictions.csv
  pm1_window_filter_hyperparameters.csv
  pm1_window_hazard_coefficients.csv
  pm1_window_patient_cluster_bootstrap.csv
  pm1_window_bootstrap_summary.csv
"""
from __future__ import annotations
from rescue.runtime import workspace_path, allocated_threads
import argparse
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from run_venice_sequential_nadir_risk import load_inputs, build_person_period
from run_venice_bayesian_state_nadir import tune_hyperparams
from run_venice_physics_reinforced_nadir import MonotoneLogistic, build_patient_sequences_for_tuning, reinforced_state_features
CV_SEED = 20260924
N_SPLITS = 5
EPS = 1e-09
PHYS_FEATURES = ['near_zero_velocity', 'toward_zero_1', 'smooth_toward_zero_2']

def safe_auc(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan

def safe_ap(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    return float(average_precision_score(y, p)) if np.sum(y == 1) > 0 else np.nan

def choose_balanced_accuracy_threshold(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    grid = np.unique(np.r_[0.001, np.quantile(p, np.linspace(0, 1, 201)), 0.999])
    vals = np.asarray([balanced_accuracy_score(y, p >= q) for q in grid], float)
    return float(grid[int(np.nanargmax(vals))])

def classification_metrics(y, p, threshold):
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-08, 1 - 1e-08)
    yh = (p >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return {'n_rows': int(len(y)), 'positives': int(y.sum()), 'prevalence': float(y.mean()), 'AUROC': safe_auc(y, p), 'average_precision': safe_ap(y, p), 'Brier': float(brier_score_loss(y, p)), 'log_loss': float(log_loss(y, p, labels=[0, 1])), 'balanced_accuracy': float(balanced_accuracy_score(y, yh)), 'accuracy': float(accuracy_score(y, yh)), 'sensitivity': float(tp / (tp + fn)) if tp + fn else np.nan, 'specificity': float(tn / (tn + fp)) if tn + fp else np.nan, 'PPV': float(tp / (tp + fp)) if tp + fp else np.nan, 'NPV': float(tn / (tn + fn)) if tn + fn else np.nan}

def make_folds(manifest):
    """Same patient-level fold construction used in the exploratory window test.

    Stratify on broad event category only, not event timing.
    """
    m = manifest.drop_duplicates('id').copy()
    cls = np.where(m['event_at_entry'].astype(bool), 'entry_event', np.where(m['future_event_observed'].astype(bool), 'future_event', 'no_event'))
    ids = m['id'].astype(str).to_numpy()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    fold_map = {}
    for fold, (_, te) in enumerate(skf.split(ids, cls), start=1):
        for ix in te:
            fold_map[str(ids[ix])] = int(fold)
    return fold_map

def build_pm1_rows(psa, manifest):
    """Build causal current-scan rows and +/-1-window labels.

    For no-event patients, a negative at current scan t is only declared if an
    event at t+1 could have been confirmed; because event confirmation needs two
    higher later observations, this requires records through t+3.
    """
    mm = manifest.drop_duplicates('id').set_index('id')
    out = []
    for pid, mr in mm.iterrows():
        pid = str(pid)
        g = psa[psa['id'].astype(str).eq(pid)].sort_values(['calendar_day', 'cycle']).reset_index(drop=True)
        if len(g) == 0:
            continue
        ai = int(mr['anchor_index'])
        ei = int(mr['entry_index'])
        event_idx = None if pd.isna(mr.get('event_index', np.nan)) else int(mr['event_index'])
        end = len(g) - 1 if event_idx is None else min(len(g) - 1, event_idx + 1)
        for t in range(ei, end + 1):
            if t - ai + 1 < 5:
                continue
            y = np.nan
            rel_to_event = np.nan
            if event_idx is not None:
                d = event_idx - t
                rel_to_event = float(d)
                if -1 <= d <= 1:
                    y = 1.0
                elif d > 1:
                    y = 0.0
            elif t + 3 < len(g):
                y = 0.0
            out.append({'id': pid, 'risk_index': int(t), 'risk_calendar_day': float(g.loc[t, 'calendar_day']), 'risk_cycle': float(g.loc[t, 'cycle']), 'elapsed_cycles': float((g.loc[t, 'calendar_day'] - g.loc[ei, 'calendar_day']) / 21.0), 'event_index': np.nan if event_idx is None else int(event_idx), 'scans_before_event': rel_to_event, 'y_pm1': y})
    return pd.DataFrame(out)

def fit_time_model(X, y):
    mdl = Pipeline([('scale', StandardScaler()), ('lr', LogisticRegression(C=1.0, solver='lbfgs', max_iter=5000))])
    mdl.fit(X, y)
    return mdl

def patient_cluster_bootstrap(pred, n_boot, seed):
    """Resample patients with replacement; keep all their scan rows together."""
    rng = np.random.default_rng(seed)
    ids = pred['id'].astype(str).unique()
    by_id = {pid: pred[pred['id'].astype(str).eq(pid)] for pid in ids}
    recs = []
    for b in range(int(n_boot)):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        parts = []
        for j, pid in enumerate(sampled):
            z = by_id[str(pid)].copy()
            z['boot_id'] = f'{pid}__{j}'
            parts.append(z)
        d = pd.concat(parts, ignore_index=True)
        y = d['y'].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            continue
        pt = d['p_time_only'].to_numpy(float)
        pp = d['p_physics_reinforced'].to_numpy(float)
        recs.append({'bootstrap': int(b), 'AUROC_time': safe_auc(y, pt), 'AUROC_physics': safe_auc(y, pp), 'delta_AUROC_physics_minus_time': safe_auc(y, pp) - safe_auc(y, pt), 'AP_time': safe_ap(y, pt), 'AP_physics': safe_ap(y, pp), 'delta_AP_physics_minus_time': safe_ap(y, pp) - safe_ap(y, pt), 'Brier_time': float(brier_score_loss(y, pt)), 'Brier_physics': float(brier_score_loss(y, pp)), 'delta_Brier_physics_minus_time': float(brier_score_loss(y, pp) - brier_score_loss(y, pt))})
    return pd.DataFrame(recs)

def quantile_ci(x):
    x = pd.to_numeric(pd.Series(x), errors='coerce').dropna().to_numpy(float)
    if len(x) == 0:
        return (np.nan, np.nan)
    return (float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audit-dir', type=Path, default=Path('venice_decline_filter_audit'))
    ap.add_argument('--sequential-dir', type=Path, default=Path('data/source_data/venice_sequential_nadir_risk_v1'))
    ap.add_argument('--outdir', type=Path, default=Path('data/source_data/venice_nadir_window_pm1_v1'))
    ap.add_argument('--l2', type=float, default=1.0)
    ap.add_argument('--cluster-bootstrap', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=20260924)
    ap.add_argument('--rise-frac', type=float, default=0.0)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    psa, gate = load_inputs(args.audit_dir)
    for d in (psa, gate):
        d['id'] = d['id'].astype(str)
    manifest_path = args.sequential_dir / 'patient_event_manifest.csv'
    sequential_rows_path = args.sequential_dir / 'sequential_risk_rows.csv'
    if manifest_path.exists() and sequential_rows_path.exists():
        manifest = pd.read_csv(manifest_path, low_memory=False)
        sequential_rows = pd.read_csv(sequential_rows_path, low_memory=False)
    else:
        manifest, sequential_rows = build_person_period(psa, gate, rise_frac=args.rise_frac)
    manifest['id'] = manifest['id'].astype(str)
    sequential_rows['id'] = sequential_rows['id'].astype(str)
    gate_ids = set(gate.loc[gate['new_record_low_at_forecast'].astype(bool), 'id'].astype(str))
    manifest = manifest[manifest['id'].isin(gate_ids)].drop_duplicates('id').copy()
    pm1 = build_pm1_rows(psa, manifest)
    pm1['id'] = pm1['id'].astype(str)
    fold_map = make_folds(manifest)
    pm1['outer_fold'] = pm1['id'].map(fold_map)
    sequences = build_patient_sequences_for_tuning(psa, manifest)
    pred_parts = []
    fold_rows = []
    hyper_rows = []
    coef_rows = []
    for fold in range(1, N_SPLITS + 1):
        test_ids = {pid for pid, f in fold_map.items() if f == fold}
        train_ids = [pid for pid in manifest['id'].astype(str) if pid not in test_ids]
        best, _ = tune_hyperparams(train_ids, sequences, sequential_rows)
        hyper_rows.append({'outer_fold': fold, 'obs_sigma': best['obs_sigma'], 'jerk_sigma': best['jerk_sigma'], 'train_patients': len(train_ids), 'test_patients': len(test_ids)})
        sf = reinforced_state_features(psa, manifest, pm1, obs_sigma=best['obs_sigma'], jerk_sigma=best['jerk_sigma'])
        dat = pm1.merge(sf, on=['id', 'risk_index'], how='inner')
        dat = dat[dat['y_pm1'].notna()].copy()
        dat['y'] = dat['y_pm1'].astype(int)
        dat['outer_fold'] = dat['id'].map(fold_map).astype(int)
        tr = dat[dat['outer_fold'].ne(fold)].copy()
        te = dat[dat['outer_fold'].eq(fold)].copy()
        if len(tr) == 0 or len(te) == 0 or tr['y'].nunique() < 2:
            continue
        time_model = fit_time_model(tr[['elapsed_cycles']], tr['y'])
        p_time_tr = time_model.predict_proba(tr[['elapsed_cycles']])[:, 1]
        p_time_te = time_model.predict_proba(te[['elapsed_cycles']])[:, 1]
        thr_time = choose_balanced_accuracy_threshold(tr['y'], p_time_tr)
        phys = MonotoneLogistic(l2=args.l2).fit(tr[PHYS_FEATURES].to_numpy(float), tr['y'].to_numpy(int))
        p_phys_tr = phys.predict_proba(tr[PHYS_FEATURES].to_numpy(float))[:, 1]
        p_phys_te = phys.predict_proba(te[PHYS_FEATURES].to_numpy(float))[:, 1]
        thr_phys = choose_balanced_accuracy_threshold(tr['y'], p_phys_tr)
        for feat, beta in zip(PHYS_FEATURES, phys.coef_):
            coef_rows.append({'outer_fold': fold, 'feature': feat, 'standardized_monotone_logit_coef': float(beta)})
        out = te[['id', 'risk_index', 'risk_cycle', 'risk_calendar_day', 'elapsed_cycles', 'event_index', 'scans_before_event', 'y', 'state_velocity_mean', 'state_acceleration_mean', 'state_velocity_sd', 'state_acceleration_sd', 'near_zero_velocity', 'velocity_change_rate_1', 'velocity_change_rate_2', 'toward_zero_1', 'smooth_toward_zero_2', 'toward_zero_consistent']].copy()
        out['outer_fold'] = fold
        out['p_time_only'] = p_time_te
        out['p_physics_reinforced'] = p_phys_te
        out['threshold_time_only'] = thr_time
        out['threshold_physics_reinforced'] = thr_phys
        pred_parts.append(out)
        mt = classification_metrics(te['y'], p_time_te, thr_time)
        mp = classification_metrics(te['y'], p_phys_te, thr_phys)
        fold_rows.append({'outer_fold': fold, 'model': 'time_only', 'n_patients': te['id'].nunique(), **mt})
        fold_rows.append({'outer_fold': fold, 'model': 'physics_reinforced_pm1', 'n_patients': te['id'].nunique(), **mp})
        print(f"fold {fold}: train={tr['id'].nunique()} test={te['id'].nunique()} obs_sigma={best['obs_sigma']:.4f} jerk_sigma={best['jerk_sigma']:.4f} AUC time={mt['AUROC']:.3f} phys={mp['AUROC']:.3f}")
    pred = pd.concat(pred_parts, ignore_index=True)
    pred.to_csv(args.outdir / 'pm1_window_oof_predictions.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(args.outdir / 'pm1_window_outer_fold_metrics.csv', index=False)
    pd.DataFrame(hyper_rows).to_csv(args.outdir / 'pm1_window_filter_hyperparameters.csv', index=False)
    pd.DataFrame(coef_rows).to_csv(args.outdir / 'pm1_window_hazard_coefficients.csv', index=False)
    summary_rows = []
    for model, pcol, tcol in [('time_only', 'p_time_only', 'threshold_time_only'), ('physics_reinforced_pm1', 'p_physics_reinforced', 'threshold_physics_reinforced')]:
        y = pred['y'].astype(int).to_numpy()
        p = pred[pcol].to_numpy(float)
        yh = (p >= pred[tcol].to_numpy(float)).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
        summary_rows.append({'model': model, 'window': '[-1,+1] scans around confirmed local nadir', 'n_rows': len(y), 'n_patients': pred['id'].nunique(), 'positives': int(y.sum()), 'prevalence': float(y.mean()), 'AUROC': safe_auc(y, p), 'average_precision': safe_ap(y, p), 'Brier': float(brier_score_loss(y, p)), 'log_loss': float(log_loss(y, np.clip(p, 1e-08, 1 - 1e-08), labels=[0, 1])), 'balanced_accuracy': float(balanced_accuracy_score(y, yh)), 'accuracy': float(accuracy_score(y, yh)), 'sensitivity': float(tp / (tp + fn)) if tp + fn else np.nan, 'specificity': float(tn / (tn + fp)) if tn + fp else np.nan, 'PPV': float(tp / (tp + fp)) if tp + fp else np.nan, 'NPV': float(tn / (tn + fn)) if tn + fn else np.nan})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.outdir / 'pm1_window_summary.csv', index=False)
    boot = patient_cluster_bootstrap(pred, args.cluster_bootstrap, args.seed)
    boot.to_csv(args.outdir / 'pm1_window_patient_cluster_bootstrap.csv', index=False)
    bs = []
    for col in ['AUROC_time', 'AUROC_physics', 'delta_AUROC_physics_minus_time', 'AP_time', 'AP_physics', 'delta_AP_physics_minus_time', 'Brier_time', 'Brier_physics', 'delta_Brier_physics_minus_time']:
        lo, hi = quantile_ci(boot[col])
        bs.append({'metric': col, 'bootstrap_mean': float(pd.to_numeric(boot[col], errors='coerce').mean()), 'CI95_low': lo, 'CI95_high': hi})
    pd.DataFrame(bs).to_csv(args.outdir / 'pm1_window_bootstrap_summary.csv', index=False)
    print('\n=== POOLED OUT-OF-FOLD +/-1 WINDOW RESULTS ===')
    print(summary.to_string(index=False))
    print('\n=== PATIENT-CLUSTER BOOTSTRAP ===')
    print(pd.DataFrame(bs).to_string(index=False))
    print(f'\nWrote outputs to: {args.outdir}')
if __name__ == '__main__':
    main()
