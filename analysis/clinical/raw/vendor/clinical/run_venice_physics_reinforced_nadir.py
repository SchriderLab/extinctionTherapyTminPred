"""
Physics-informed reinforcement test for sequential VENICE nadir prediction.

This script does NOT force the latent PSA filter itself to bend toward a nadir.
Instead it leaves the robust Bayesian state-space filter unchanged, then
reinforces the *hazard mapping* with a pre-specified physical rule:

  1) imminent nadir risk should rise as latent log-PSA velocity approaches 0;
  2) risk should rise when velocity has recently moved toward 0;
  3) risk should rise further when that movement toward 0 is sustained across
     the last two scan intervals.

The reinforced model uses monotone-constrained logistic regression so these
three physical features cannot receive coefficients with the opposite sign.
All filtering, scaling, model fitting, and tuning remain outer-training-only.
Held-out outcomes are never used to construct the state or select coefficients.

Comparators:
  - time_only
  - robust_quadratic_best (current_slope + slope_change)
  - state_va (latent velocity + acceleration)
  - reinforced_unconstrained
  - reinforced_monotone

Endpoint/censoring/folds are inherited from the previous sequential analysis.
"""
from __future__ import annotations
from rescue.runtime import workspace_path, allocated_threads
import argparse
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from run_venice_sequential_nadir_risk import load_inputs, build_person_period, make_patient_folds
from run_venice_bayesian_state_nadir import robust_filter, tune_hyperparams, make_lr, safe_metrics, cluster_bootstrap
EPS = 1e-09

def reinforced_state_features(psa, manifest, rows, obs_sigma, jerk_sigma):
    """Causal latent state + pre-specified 'smooth approach to zero' features.

    Uses latent velocity estimates at current, previous, and previous-previous
    observed scans. Previous states can be pre-entry anchor-window states; no
    future observations are used.
    """
    manifest_idx = manifest.drop_duplicates('id').set_index('id')
    parts = []
    for pid, rpid in rows.groupby(rows['id'].astype(str), sort=False):
        if pid not in manifest_idx.index:
            continue
        mr = manifest_idx.loc[pid]
        g = psa[psa['id'].astype(str).eq(pid)].sort_values(['calendar_day', 'cycle']).reset_index(drop=True)
        if len(g) == 0:
            continue
        ai = int(mr['anchor_index'])
        max_idx = min(int(rpid['risk_index'].max()), len(g) - 1)
        gg = g.iloc[ai:max_idx + 1].copy()
        t = (gg['calendar_day'].to_numpy(float) - float(gg['calendar_day'].iloc[0])) / 21.0
        ly = np.log(gg['psa'].to_numpy(float))
        y = ly - ly[0]
        states, _ = robust_filter(t, y, obs_sigma, jerk_sigma, return_all=True)
        for _, rr in rpid.iterrows():
            gi = int(rr['risk_index'])
            li = gi - ai
            if li < 0 or li >= len(states):
                continue
            m, P, w = states[li]
            v = float(m[1])
            a = float(m[2])
            sdv = math.sqrt(max(float(P[1, 1]), EPS))
            sda = math.sqrt(max(float(P[2, 2]), EPS))
            p_declining = float(norm.cdf((0.0 - v) / max(sdv, 1e-08)))
            dv1 = np.nan
            dv2 = np.nan
            if li >= 1:
                mp1 = states[li - 1][0]
                dt1 = float(t[li] - t[li - 1])
                if np.isfinite(dt1) and dt1 > 0:
                    dv1 = float((v - float(mp1[1])) / dt1)
            if li >= 2:
                mp1 = states[li - 1][0]
                mp2 = states[li - 2][0]
                dt2 = float(t[li - 1] - t[li - 2])
                if np.isfinite(dt2) and dt2 > 0:
                    dv2 = float((float(mp1[1]) - float(mp2[1])) / dt2)
            near_zero = float(-abs(v))
            toward_zero_1 = float(max(dv1, 0.0)) if np.isfinite(dv1) else 0.0
            smooth_toward_zero_2 = float(max(min(dv1, dv2), 0.0)) if np.isfinite(dv1) and np.isfinite(dv2) else 0.0
            toward_zero_consistent = float(np.isfinite(dv1) and np.isfinite(dv2) and (dv1 > 0) and (dv2 > 0))
            parts.append({'id': pid, 'risk_index': gi, 'state_velocity_mean': v, 'state_acceleration_mean': a, 'state_velocity_sd': sdv, 'state_acceleration_sd': sda, 'last_obs_robust_weight': float(w), 'p_declining': p_declining, 'near_zero_velocity': near_zero, 'velocity_change_rate_1': dv1, 'velocity_change_rate_2': dv2, 'toward_zero_1': toward_zero_1, 'smooth_toward_zero_2': smooth_toward_zero_2, 'toward_zero_consistent': toward_zero_consistent})
    return pd.DataFrame(parts)

class MonotoneLogistic:
    """Standardized logistic regression with beta_j >= 0 for all features."""

    def __init__(self, l2=1.0):
        self.l2 = float(l2)
        self.mean_ = None
        self.scale_ = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)
        self.scale_[~np.isfinite(self.scale_) | (self.scale_ < 1e-08)] = 1.0
        Z = (X - self.mean_) / self.scale_
        Z = np.where(np.isfinite(Z), Z, 0.0)
        p0 = np.clip(float(np.mean(y)), 0.0001, 1 - 0.0001)
        theta0 = np.r_[math.log(p0 / (1 - p0)), np.zeros(Z.shape[1])]

        def obj(theta):
            eta = theta[0] + Z @ theta[1:]
            nll = np.sum(np.logaddexp(0.0, eta) - y * eta)
            pen = 0.5 * self.l2 * np.sum(theta[1:] ** 2)
            return float(nll + pen)

        def grad(theta):
            eta = theta[0] + Z @ theta[1:]
            p = expit(eta)
            r = p - y
            g0 = np.sum(r)
            gb = Z.T @ r + self.l2 * theta[1:]
            return np.r_[g0, gb]
        bounds = [(None, None)] + [(0.0, None)] * Z.shape[1]
        res = minimize(obj, theta0, jac=grad, method='L-BFGS-B', bounds=bounds, options={'maxiter': 5000})
        if not res.success:
            raise RuntimeError(f'Monotone logistic failed: {res.message}')
        self.intercept_ = float(res.x[0])
        self.coef_ = np.asarray(res.x[1:], float)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        Z = (X - self.mean_) / self.scale_
        Z = np.where(np.isfinite(Z), Z, 0.0)
        p = expit(self.intercept_ + Z @ self.coef_)
        return np.c_[1.0 - p, p]

def build_patient_sequences_for_tuning(psa, manifest):
    out = {}
    mm = manifest.drop_duplicates('id').set_index('id')
    for pid, mr in mm.iterrows():
        g = psa[psa['id'].astype(str).eq(str(pid))].sort_values(['calendar_day', 'cycle']).reset_index(drop=True)
        if len(g) == 0:
            continue
        ai = int(mr['anchor_index'])
        if ai < 0 or ai >= len(g):
            continue
        if pd.notna(mr.get('event_index', np.nan)):
            end = min(len(g) - 1, int(mr['event_index']) + 2)
        else:
            end = len(g) - 1
        gg = g.iloc[ai:end + 1]
        tt = (gg['calendar_day'].to_numpy(float) - float(gg['calendar_day'].iloc[0])) / 21.0
        ly = np.log(gg['psa'].to_numpy(float))
        out[str(pid)] = (tt, ly - ly[0])
    return out

def summarize_velocity_alignment(pred, rows):
    h1 = pred[pred['horizon_scans'].eq(1)].copy()
    ev = rows[['id', 'risk_index', 'event_index']].drop_duplicates()
    d = h1.merge(ev, on=['id', 'risk_index'], how='left')
    d = d[d['event_index'].notna()].copy()
    d['scans_before_event'] = d['event_index'] - d['risk_index']
    ds = d[d['scans_before_event'].between(1, 6)].copy()
    s = ds.groupby('scans_before_event').agg(n_patients=('id', 'nunique'), mean_velocity=('state_velocity_mean', 'mean'), median_velocity=('state_velocity_mean', 'median'), q25_velocity=('state_velocity_mean', lambda x: np.quantile(x, 0.25)), q75_velocity=('state_velocity_mean', lambda x: np.quantile(x, 0.75)), mean_acceleration=('state_acceleration_mean', 'mean'), median_acceleration=('state_acceleration_mean', 'median')).reset_index().sort_values('scans_before_event', ascending=False)
    return (s, d)
