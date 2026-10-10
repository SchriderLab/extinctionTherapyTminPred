"""
Robust Bayesian state-space test for sequential VENICE nadir prediction.

This is a fast approximate-Bayesian proof of concept, not full MCMC.

Latent dynamics (log PSA relative to the audited anchor):
    state_t = [level, velocity, acceleration]

    level(t+dt) ~= level + velocity*dt + 0.5*acceleration*dt^2
    velocity(t+dt) ~= velocity + acceleration*dt
    acceleration follows a smooth random walk driven by white jerk.

Observation model:
    y_t = latent_level_t + noise
with a Student-t robustification implemented as an iteratively reweighted
Kalman update. Isolated PSA bumps therefore inflate their effective
observation variance rather than forcing the latent trajectory to follow them.

For each outer patient-level fold:
  1. Select observation/process noise hyperparameters using ONLY training
     patients and an unsupervised robust predictive likelihood.
  2. Filter every patient's history sequentially with those fold-specific
     hyperparameters.
  3. At each risk scan extract the current latent posterior:
       E[v], E[a], SD(v), SD(a), corr(v,a)
     and physics-style zero-crossing probabilities for 1 or 2 nominal scans.
  4. Train simple logistic hazards on outer-training patients only and predict
     outer-held-out patients.

Comparators:
  - time_only
  - robust_quadratic_best: current_slope + slope_change (previous best)
  - state_va: latent velocity + acceleration
  - state_va_unc: velocity + acceleration + posterior uncertainty
  - physics_raw: direct latent-slope zero-crossing probability
  - physics_calibrated: training-fold logistic calibration of physics_raw

Endpoint and censoring are EXACTLY inherited from
run_venice_sequential_nadir_risk.py.
"""
from __future__ import annotations
from rescue.runtime import workspace_path, allocated_threads
import argparse
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm, t as student_t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from run_venice_sequential_nadir_risk import load_inputs, build_person_period, make_patient_folds
EPS = 1e-09
CV_SEED = 20260924
BOOT_SEED = 20260925
OBS_GRID = [0.04, 0.07, 0.1, 0.15, 0.22, 0.32]
JERK_GRID = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16]
NU = 4.0

def transition(dt: float):
    dt = float(max(dt, 1e-06))
    return np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]], float)

def process_cov(dt: float, jerk_sigma: float):
    """Continuous white-jerk covariance for [position, velocity, acceleration]."""
    dt = float(max(dt, 1e-06))
    q = float(jerk_sigma) ** 2
    return q * np.array([[dt ** 5 / 20.0, dt ** 4 / 8.0, dt ** 3 / 6.0], [dt ** 4 / 8.0, dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 3 / 6.0, dt ** 2 / 2.0, dt]], float)

def robust_filter(times: np.ndarray, y: np.ndarray, obs_sigma: float, jerk_sigma: float, return_all=True):
    """Approximate Student-t robust Kalman filter.

    y should be log PSA relative to the first/anchor observation.
    Returns list of filtered (mean, cov, robust_weight) states and predictive loglik.
    """
    times = np.asarray(times, float)
    y = np.asarray(y, float)
    n = len(y)
    if n == 0:
        return ([], -np.inf)
    H = np.array([[1.0, 0.0, 0.0]])
    I = np.eye(3)
    R = float(obs_sigma) ** 2
    m = np.array([float(y[0]), 0.0, 0.0])
    P = np.diag([max(R * 4.0, 0.02 ** 2), 0.6 ** 2, 0.3 ** 2])
    states = [(m.copy(), P.copy(), 1.0)]
    ll = 0.0
    for k in range(1, n):
        dt = float(times[k] - times[k - 1])
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0
        F = transition(dt)
        Q = process_cov(dt, jerk_sigma)
        mp = F @ m
        Pp = F @ P @ F.T + Q
        Pp = 0.5 * (Pp + Pp.T)
        resid = float(y[k] - mp[0])
        S0 = float(max(Pp[0, 0] + R, EPS))
        z2 = resid * resid / S0
        w = float((NU + 1.0) / (NU + z2))
        w = float(np.clip(w, 0.05, 5.0))
        Reff = R / w
        S = float(max(Pp[0, 0] + Reff, EPS))
        K = Pp[:, 0] / S
        m = mp + K * resid
        KH = np.outer(K, H[0])
        P = (I - KH) @ Pp @ (I - KH).T + np.outer(K, K) * Reff
        P = 0.5 * (P + P.T)
        P += np.eye(3) * 1e-12
        ll += float(student_t.logpdf(resid / math.sqrt(S0), df=NU) - 0.5 * math.log(S0))
        states.append((m.copy(), P.copy(), w))
    return (states, ll)

def sequence_score(times, y, obs_sigma, jerk_sigma, burn=3):
    """Robust predictive score after a small initialization burn-in."""
    times = np.asarray(times, float)
    y = np.asarray(y, float)
    if len(y) <= burn:
        return (0.0, 0)
    H = np.array([[1.0, 0.0, 0.0]])
    I = np.eye(3)
    R = float(obs_sigma) ** 2
    m = np.array([float(y[0]), 0.0, 0.0])
    P = np.diag([max(R * 4.0, 0.02 ** 2), 0.6 ** 2, 0.3 ** 2])
    score = 0.0
    count = 0
    for k in range(1, len(y)):
        dt = float(times[k] - times[k - 1])
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0
        F = transition(dt)
        Q = process_cov(dt, jerk_sigma)
        mp = F @ m
        Pp = F @ P @ F.T + Q
        Pp = 0.5 * (Pp + Pp.T)
        resid = float(y[k] - mp[0])
        S0 = float(max(Pp[0, 0] + R, EPS))
        if k >= burn:
            score += float(student_t.logpdf(resid / math.sqrt(S0), df=NU) - 0.5 * math.log(S0))
            count += 1
        z2 = resid * resid / S0
        w = float(np.clip((NU + 1.0) / (NU + z2), 0.05, 5.0))
        Reff = R / w
        S = float(max(Pp[0, 0] + Reff, EPS))
        K = Pp[:, 0] / S
        m = mp + K * resid
        KH = np.outer(K, H[0])
        P = (I - KH) @ Pp @ (I - KH).T + np.outer(K, K) * Reff
        P = 0.5 * (P + P.T) + np.eye(3) * 1e-12
    return (score, count)

def build_patient_sequences(psa: pd.DataFrame, manifest: pd.DataFrame):
    """Anchor-normalized patient sequences used only for unsupervised filter tuning."""
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
        gg = g.iloc[ai:end + 1].copy()
        t = (gg['calendar_day'].to_numpy(float) - float(gg['calendar_day'].iloc[0])) / 21.0
        ly = np.log(gg['psa'].to_numpy(float))
        y = ly - ly[0]
        out[str(pid)] = (t, y)
    return out

def tune_hyperparams(train_ids, sequences, rows):
    """Fast empirical-Bayes hyperparameters from training histories only.

    Observation scale: median robust Huber residual MAD from training risk rows.
    Jerk scale: robust MAD of successive Huber-curvature changes per sqrt(cycle).
    No held-out outcome information is used.
    """
    tr = rows[rows['id'].astype(str).isin(set(train_ids))].copy()
    u = pd.to_numeric(tr['trend_uncertainty'], errors='coerce').to_numpy(float)
    u = u[np.isfinite(u) & (u > 1e-06)]
    obs = float(np.median(u)) if len(u) else 0.08
    obs = float(np.clip(obs, 0.03, 0.25))
    vals = []
    for pid, g in tr.groupby(tr['id'].astype(str)):
        g = g.sort_values('risk_calendar_day')
        if len(g) < 2:
            continue
        c = pd.to_numeric(g['curvature'], errors='coerce').to_numpy(float)
        t = pd.to_numeric(g['risk_calendar_day'], errors='coerce').to_numpy(float) / 21.0
        dc = np.diff(c)
        dt = np.diff(t)
        ok = np.isfinite(dc) & np.isfinite(dt) & (dt > 0)
        if np.any(ok):
            vals.extend((dc[ok] / np.sqrt(dt[ok])).tolist())
    if vals:
        a = np.asarray(vals, float)
        med = float(np.median(a))
        jerk = float(1.4826 * np.median(np.abs(a - med)))
    else:
        jerk = 0.02
    jerk = float(np.clip(jerk, 0.005, 0.15))
    return ({'obs_sigma': obs, 'jerk_sigma': jerk, 'mean_robust_logscore': np.nan, 'n_scored_obs': int(len(u))}, pd.DataFrame([{'obs_sigma': obs, 'jerk_sigma': jerk, 'mean_robust_logscore': np.nan, 'n_scored_obs': int(len(u)), 'method': 'training_empirical_bayes'}]))
_MC_RNG = np.random.default_rng(20260924)
_MC_Z0 = _MC_RNG.standard_normal((3, 512))
_MC_ZQ1 = _MC_RNG.standard_normal((3, 512))
_MC_ZQ2 = _MC_RNG.standard_normal((3, 512))

def _safe_chol(A):
    A = 0.5 * (A + A.T)
    for j in [1e-12, 1e-10, 1e-08, 1e-06]:
        try:
            return np.linalg.cholesky(A + np.eye(A.shape[0]) * j)
        except np.linalg.LinAlgError:
            pass
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, 1e-08)
    return V @ np.diag(np.sqrt(w))

def joint_cross_probability(m, P, horizon, jerk_sigma):
    """Monte-Carlo posterior probability P(v_now<0 and v_h>=0)."""
    h = float(horizon)
    F = transition(h)
    Q = process_cov(h, jerk_sigma)
    L0 = _safe_chol(P)
    Lq = _safe_chol(Q)
    Zq = _MC_ZQ1 if h <= 1.01 else _MC_ZQ2
    s0 = m[:, None] + L0 @ _MC_Z0
    sf = F @ s0 + Lq @ Zq
    return float(np.mean((s0[1] < 0.0) & (sf[1] >= 0.0)))

def state_features_for_fold(psa, manifest, rows, obs_sigma, jerk_sigma):
    """Compute causal filtered state features for every risk row under one fold's hyperparameters."""
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
        max_idx = int(rpid['risk_index'].max())
        max_idx = min(max_idx, len(g) - 1)
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
            sdv = math.sqrt(max(float(P[1, 1]), EPS))
            sda = math.sqrt(max(float(P[2, 2]), EPS))
            corr = float(P[1, 2] / max(sdv * sda, EPS))
            p1 = joint_cross_probability(m, P, 1.0, jerk_sigma)
            p2 = joint_cross_probability(m, P, 2.0, jerk_sigma)
            parts.append({'id': pid, 'risk_index': gi, 'state_level_mean': float(m[0]), 'state_velocity_mean': float(m[1]), 'state_acceleration_mean': float(m[2]), 'state_velocity_sd': sdv, 'state_acceleration_sd': sda, 'state_va_corr': corr, 'last_obs_robust_weight': float(w), 'physics_cross1': p1, 'physics_cross2': p2})
    return pd.DataFrame(parts)

def make_lr():
    return Pipeline([('scale', StandardScaler()), ('lr', LogisticRegression(C=1.0, solver='lbfgs', max_iter=5000))])

def safe_metrics(y, p):
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-08, 1 - 1e-08)
    out = {'n_rows': int(len(y)), 'positives': int(y.sum()), 'prevalence': float(y.mean()) if len(y) else np.nan, 'Brier': float(brier_score_loss(y, p)), 'log_loss': float(log_loss(y, p, labels=[0, 1])), 'AUROC': np.nan, 'average_precision': np.nan}
    if len(np.unique(y)) == 2:
        out['AUROC'] = float(roc_auc_score(y, p))
    if y.sum() > 0:
        out['average_precision'] = float(average_precision_score(y, p))
    return out

def cluster_bootstrap(pred, prob_col, n_boot):
    if n_boot <= 0:
        return {}
    ids = pred['id'].astype(str).drop_duplicates().to_numpy()
    by = {pid: pred[pred['id'].astype(str).eq(pid)] for pid in ids}
    rng = np.random.default_rng(BOOT_SEED)
    vals = {k: [] for k in ['Brier', 'log_loss', 'AUROC', 'average_precision']}
    for _ in range(n_boot):
        samp = rng.choice(ids, size=len(ids), replace=True)
        b = pd.concat([by[p] for p in samp], ignore_index=True)
        m = safe_metrics(b['y'].to_numpy(), b[prob_col].to_numpy())
        for k in vals:
            if np.isfinite(m[k]):
                vals[k].append(m[k])
    out = {}
    for k, v in vals.items():
        a = np.asarray(v, float)
        out[f'{k}_clusterboot_lo'] = float(np.quantile(a, 0.025)) if len(a) else np.nan
        out[f'{k}_clusterboot_hi'] = float(np.quantile(a, 0.975)) if len(a) else np.nan
    return out
