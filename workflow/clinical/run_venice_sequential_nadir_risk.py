"""
Sequential VENICE nadir-risk test (simple version).

Scientific question
-------------------
After a patient has entered a clearly declining PSA phase, can the PSA history
available *up to the current scan* predict whether the first confirmed local
nadir will occur at the next scan or within the next two scans?

Primary endpoint
----------------
The event is the FIRST local PSA minimum at/after decline entry that:
  1) is lower than the immediately preceding PSA, and
  2) is followed by two subsequent PSA values that are both strictly above it.
The event time is the nadir scan itself, not the later confirmation scan.

Decline entry
-------------
To avoid changing the cohort and outcome definition simultaneously, this first
implementation uses the audited Gate-A cohort:
  - existing 5 measurements / 4 consecutive declines,
  - new_record_low_at_forecast == True.
The fifth measurement is the entry/forecast scan. Patients whose confirmed
local nadir is already that entry scan are counted, but they do not contribute
future-risk rows because the event has already occurred.

Features at each risk scan
--------------------------
Only information available through that scan is used. Starting at the audited
anchor scan, fit a robust quadratic to log(PSA) versus actual elapsed time,
centered at the current scan:
    z(x) = a + b*x + c*x^2
using Huber regression.
Features:
  current_slope      = b
  slope_change       = change in derivative from previous scan to current scan
  trend_uncertainty  = robust residual MAD around the fitted trend
  elapsed_cycles     = actual elapsed 21-day cycle-equivalents since entry

The quadratic smoother plus Huber loss is deliberately insensitive to a single
bumpy PSA point while allowing a persistent flattening trend to accumulate.

Outcomes / censoring
--------------------
For prediction at scan t:
  next1 = event occurs at t+1
  next2 = event occurs at t+1 or t+2
A negative next1 label is used only if event status at t+1 can be ascertained,
which requires two confirmation scans after t+1 (through t+3).
A negative next2 label requires enough follow-up to ascertain an event at t+2
(through t+4). Otherwise the row is censored for that horizon.

Validation
----------
Five-fold PATIENT-LEVEL cross-validation. All rows from one patient remain in
one fold. Two fixed logistic models are compared:
  - time-only baseline: elapsed_cycles
  - trend model: current_slope + slope_change + trend_uncertainty + elapsed_cycles
No held-out future observations enter feature construction or model fitting.

Outputs
-------
  patient_event_manifest.csv
  sequential_risk_rows.csv
  oof_predictions_next1.csv
  oof_predictions_next2.csv
  model_summary.csv
  calibration_next1.csv
  calibration_next2.csv
  feature_coefficients_by_fold.csv
  run_summary.txt
"""
from __future__ import annotations
from rescue.runtime import workspace_path, allocated_threads
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None
EPS = 1e-12
CV_SEED = 20260924
N_SPLITS = 5

def weighted_mad(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))

def robust_quadratic_features(times_cycle: np.ndarray, psa: np.ndarray):
    """Fit robust quadratic log-PSA trend centered at current time."""
    t = np.asarray(times_cycle, float)
    y = np.log(np.asarray(psa, float))
    if len(t) < 3 or np.any(~np.isfinite(y)):
        return None
    x = t - t[-1]
    X = np.column_stack([x, x * x])
    try:
        model = HuberRegressor(epsilon=1.35, alpha=0.05, fit_intercept=True, max_iter=1000, tol=1e-08)
        model.fit(X, y)
        pred = model.predict(X)
        b = float(model.coef_[0])
        c = float(model.coef_[1])
    except Exception:
        Xd = np.column_stack([np.ones(len(x)), x, x * x])
        ridge = np.diag([0.0, 0.05, 0.05])
        beta = np.linalg.solve(Xd.T @ Xd + ridge, Xd.T @ y)
        pred = Xd @ beta
        b = float(beta[1])
        c = float(beta[2])
    resid = y - pred
    uncertainty = weighted_mad(resid)
    if not np.isfinite(uncertainty):
        uncertainty = float(np.std(resid)) if len(resid) > 1 else 0.0
    if len(x) >= 2:
        x_prev = float(x[-2])
        previous_slope_same_smooth = b + 2.0 * c * x_prev
        slope_change = b - previous_slope_same_smooth
    else:
        slope_change = 0.0
    return {'current_slope': b, 'slope_change': float(slope_change), 'trend_uncertainty': float(max(uncertainty, 0.0)), 'curvature': float(2.0 * c)}

def first_confirmed_local_nadir(g: pd.DataFrame, entry_idx: int, rise_frac: float=0.0):
    """
    First j >= entry_idx with decline into j and two later PSAs both above j.
    rise_frac=0 requires strictly higher; e.g. 0.05 requires > 1.05*nadir.
    """
    p = g['psa'].to_numpy(float)
    threshold_mult = 1.0 + float(rise_frac)
    for j in range(max(1, entry_idx), len(g) - 2):
        if p[j] < p[j - 1] and p[j + 1] > threshold_mult * p[j] and (p[j + 2] > threshold_mult * p[j]):
            return j
    return None

def load_inputs(audit_dir: Path):
    psa_path = audit_dir / 'all_scheduled_psa.csv'
    early_path = audit_dir / 'selected_early_features.csv'
    if not psa_path.exists() or not early_path.exists():
        raise FileNotFoundError(f'Expected {psa_path} and {early_path}. Extract the audit bundle first.')
    psa = pd.read_csv(psa_path, low_memory=False)
    early = pd.read_csv(early_path, low_memory=False)
    psa = psa.rename(columns={'USUBJID': 'id'}).copy()
    for c in ['cycle', 'psa', 'calendar_day']:
        psa[c] = pd.to_numeric(psa[c], errors='coerce')
    psa = psa[psa['id'].notna() & psa['psa'].notna() & (psa['psa'] > 0) & psa['calendar_day'].notna()].copy()
    psa['id'] = psa['id'].astype(str)
    psa = psa.sort_values(['id', 'calendar_day', 'cycle']).reset_index(drop=True)
    gate = early[early['new_record_low_at_forecast'].astype(bool)].copy()
    gate['id'] = gate['id'].astype(str)
    return (psa, gate)

def locate_index(g: pd.DataFrame, calendar_day: float, cycle: float):
    a = np.where(np.isclose(g['calendar_day'].to_numpy(float), float(calendar_day), atol=1e-08))[0]
    if len(a):
        return int(a[0])
    a = np.where(np.isclose(g['cycle'].to_numpy(float), float(cycle), atol=1e-08))[0]
    if len(a):
        return int(a[0])
    return None

def build_person_period(psa: pd.DataFrame, gate: pd.DataFrame, rise_frac: float):
    manifests = []
    rows = []
    for _, er in gate.iterrows():
        pid = str(er['id'])
        g = psa[psa['id'].eq(pid)].sort_values(['calendar_day', 'cycle']).reset_index(drop=True)
        if len(g) == 0:
            continue
        anchor_idx = locate_index(g, er['anchor_calendar_day'], er['anchor_cycle'])
        entry_idx = locate_index(g, er['forecast_calendar_day'], er['forecast_cycle'])
        if anchor_idx is None or entry_idx is None or anchor_idx > entry_idx:
            continue
        event_idx = first_confirmed_local_nadir(g, entry_idx, rise_frac=rise_frac)
        event_at_entry = event_idx is not None and event_idx == entry_idx
        future_event = event_idx is not None and event_idx > entry_idx
        manifests.append({'id': pid, 'n_total_records': int(len(g)), 'anchor_index': int(anchor_idx), 'entry_index': int(entry_idx), 'anchor_cycle': float(g.loc[anchor_idx, 'cycle']), 'entry_cycle': float(g.loc[entry_idx, 'cycle']), 'anchor_calendar_day': float(g.loc[anchor_idx, 'calendar_day']), 'entry_calendar_day': float(g.loc[entry_idx, 'calendar_day']), 'event_observed': bool(event_idx is not None), 'event_at_entry': bool(event_at_entry), 'future_event_observed': bool(future_event), 'event_index': np.nan if event_idx is None else int(event_idx), 'event_cycle': np.nan if event_idx is None else float(g.loc[event_idx, 'cycle']), 'event_calendar_day': np.nan if event_idx is None else float(g.loc[event_idx, 'calendar_day']), 'event_remaining_scans_from_entry': np.nan if event_idx is None else int(event_idx - entry_idx), 'event_remaining_cycle_equiv_from_entry': np.nan if event_idx is None else float((g.loc[event_idx, 'calendar_day'] - g.loc[entry_idx, 'calendar_day']) / 21.0), 'last_calendar_day': float(g['calendar_day'].iloc[-1]), 'followup_cycle_equiv_from_entry': float((g['calendar_day'].iloc[-1] - g.loc[entry_idx, 'calendar_day']) / 21.0), 'n_after_entry': int(len(g) - entry_idx - 1)})
        if event_at_entry:
            continue
        last_risk_idx = event_idx - 1 if event_idx is not None else len(g) - 1
        for t_idx in range(entry_idx, last_risk_idx + 1):
            hist = g.iloc[anchor_idx:t_idx + 1]
            if len(hist) < 5:
                continue
            time_cycle = (hist['calendar_day'].to_numpy(float) - float(hist['calendar_day'].iloc[0])) / 21.0
            feat = robust_quadratic_features(time_cycle, hist['psa'].to_numpy(float))
            if feat is None:
                continue
            y1 = np.nan
            if event_idx is not None and event_idx == t_idx + 1:
                y1 = 1.0
            elif t_idx + 3 < len(g):
                y1 = 0.0
            y2 = np.nan
            if event_idx is not None and event_idx in (t_idx + 1, t_idx + 2):
                y2 = 1.0
            elif t_idx + 4 < len(g):
                y2 = 0.0
            rows.append({'id': pid, 'risk_index': int(t_idx), 'risk_cycle': float(g.loc[t_idx, 'cycle']), 'risk_calendar_day': float(g.loc[t_idx, 'calendar_day']), 'n_history_from_anchor': int(len(hist)), 'elapsed_cycles': float((g.loc[t_idx, 'calendar_day'] - g.loc[entry_idx, 'calendar_day']) / 21.0), 'current_psa': float(g.loc[t_idx, 'psa']), 'log_current_psa': float(np.log(g.loc[t_idx, 'psa'])), **feat, 'y_next1': y1, 'y_next2': y2, 'event_index': np.nan if event_idx is None else int(event_idx), 'event_cycle': np.nan if event_idx is None else float(g.loc[event_idx, 'cycle']), 'event_at_entry': False})
    return (pd.DataFrame(manifests), pd.DataFrame(rows))

def make_patient_folds(manifest: pd.DataFrame, rows: pd.DataFrame):
    contributing = sorted(set(rows.loc[rows[['y_next1', 'y_next2']].notna().any(axis=1), 'id'].astype(str)))
    m = manifest[manifest['id'].isin(contributing)].drop_duplicates('id').copy()
    y = m['future_event_observed'].astype(int).to_numpy()
    ids = m['id'].astype(str).to_numpy()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    out = {}
    for fold, (_, te) in enumerate(skf.split(ids, y), start=1):
        for idx in te:
            out[str(ids[idx])] = fold
    return out

def build_models():
    base = Pipeline([('scale', StandardScaler()), ('lr', LogisticRegression(C=1.0, solver='lbfgs', max_iter=5000))])
    trend = clone(base)
    return (base, trend)

def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan

def safe_ap(y, p):
    return float(average_precision_score(y, p)) if np.sum(y == 1) > 0 else np.nan

def evaluate(y, p):
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-08, 1 - 1e-08)
    return {'n_rows': int(len(y)), 'positives': int(y.sum()), 'prevalence': float(y.mean()) if len(y) else np.nan, 'Brier': float(brier_score_loss(y, p)), 'log_loss': float(log_loss(y, p, labels=[0, 1])), 'AUROC': safe_auc(y, p), 'average_precision': safe_ap(y, p)}

def calibration_table(df: pd.DataFrame, prob_col: str, n_bins: int=10):
    x = df[['y', prob_col]].dropna().copy()
    if len(x) == 0:
        return pd.DataFrame()
    try:
        x['bin'] = pd.qcut(x[prob_col], q=min(n_bins, len(x)), duplicates='drop')
    except Exception:
        x['bin'] = 0
    out = x.groupby('bin', observed=True).agg(n=('y', 'size'), mean_pred=(prob_col, 'mean'), observed_rate=('y', 'mean')).reset_index(drop=True)
    return out

def run_cv(rows: pd.DataFrame, manifest: pd.DataFrame, horizon: int, fold_map: dict):
    label = 'y_next1' if horizon == 1 else 'y_next2'
    d = rows[rows[label].notna()].copy()
    d['y'] = d[label].astype(int)
    d['outer_fold'] = d['id'].map(fold_map)
    d = d[d['outer_fold'].notna()].copy()
    d['outer_fold'] = d['outer_fold'].astype(int)
    base_features = ['elapsed_cycles']
    trend_features = ['current_slope', 'slope_change', 'trend_uncertainty', 'elapsed_cycles']
    base_model, trend_model = build_models()
    pred_parts = []
    coef_rows = []
    for fold in sorted(d['outer_fold'].unique()):
        tr = d[d['outer_fold'] != fold].copy()
        te = d[d['outer_fold'] == fold].copy()
        if len(tr) == 0 or len(te) == 0 or tr['y'].nunique() < 2:
            continue
        bm = clone(base_model)
        tm = clone(trend_model)
        bm.fit(tr[base_features], tr['y'])
        tm.fit(tr[trend_features], tr['y'])
        te['p_time_only'] = bm.predict_proba(te[base_features])[:, 1]
        te['p_trend'] = tm.predict_proba(te[trend_features])[:, 1]
        te['horizon_scans'] = horizon
        pred_parts.append(te)
        lr = tm.named_steps['lr']
        for feat, coef in zip(trend_features, lr.coef_[0]):
            coef_rows.append({'horizon_scans': horizon, 'outer_fold': int(fold), 'feature': feat, 'standardized_logit_coef': float(coef)})
    pred = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    coefs = pd.DataFrame(coef_rows)
    summaries = []
    if len(pred):
        for model_name, col in [('time_only', 'p_time_only'), ('robust_trend', 'p_trend')]:
            met = evaluate(pred['y'].to_numpy(), pred[col].to_numpy())
            met.update({'horizon_scans': horizon, 'model': model_name, 'n_patients': int(pred['id'].nunique())})
            summaries.append(met)
    return (pred, pd.DataFrame(summaries), coefs)
