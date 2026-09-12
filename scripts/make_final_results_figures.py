#!/usr/bin/env python3
from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Paths: portable GitHub reproduction layout
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "data" / "cohort"
REF = ROOT / "data" / "reference"
ABC = NEW / "abc_hybrid10k_4scan"
REGRET = ABC / "regret_decomposition"

OUT = ROOT / "reproduced_results"
FIG = OUT / "figures"
DATA = OUT / "source_data"
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

SEED = 20260910
RNG = np.random.default_rng(SEED)
N_BOOT = 10000
N_REP = 10
TILE_END = 12

# Color-blind-friendly palette (Okabe-Ito-like)
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
BLACK = "#222222"
MIDGRAY = "#8A8A8A"
LIGHTGRAY = "#D9D9D9"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 120,
})


def save(fig, stem):
    for ext in ("pdf", "svg", "png"):
        path = FIG / f"{stem}.{ext}"
        fig.savefig(path, dpi=600 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
        print("WROTE", path)
    plt.close(fig)


def panel(ax, letter, x=-0.16, y=1.06):
    """Place panel letters outside the plotting area to avoid title collisions."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left",
            clip_on=False)


def bootstrap_mean_ci(values, seed_offset=0):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(SEED + seed_offset)
    means = np.empty(N_BOOT, dtype=float)
    # Chunking keeps memory small on Longleaf login/compute nodes.
    for i in range(0, N_BOOT, 1000):
        b = min(1000, N_BOOT - i)
        idx = rng.integers(0, len(x), size=(b, len(x)))
        means[i:i+b] = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def wilson_ci(p, n=N_REP, z=1.959963984540054):
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * np.sqrt((p*(1-p)/n) + z*z/(4*n*n)) / (1 + z*z/n)
    lo = np.clip(center-half, 0.0, 1.0)
    hi = np.clip(center+half, 0.0, 1.0)

    # Protect against tiny floating-point violations at p=0 or p=1.
    # Matplotlib requires every yerr value to be >= 0.
    lo = np.minimum(lo, p)
    hi = np.maximum(hi, p)
    return lo, hi


def extract_d1(df):
    x = df.copy()
    if "phase" in x.columns:
        phase = x["phase"].astype(str).str.lower()
        if (phase == "treatment").any():
            x = x[phase == "treatment"].copy()
    if "sim_stage" in x.columns:
        stage = x["sim_stage"].astype(str).str.lower()
        if (stage == "late").any():
            x = x[stage == "late"].copy()
    if "current_drug" in x.columns:
        drug = pd.to_numeric(x["current_drug"], errors="coerce")
        if (drug == 1).any():
            x = x[drug == 1].copy()

    if "N" not in x.columns:
        return None
    tcol = next((c for c in ("t_treatment", "t", "treatment_t", "cycle")
                 if c in x.columns), None)
    if tcol is None:
        return None

    z = pd.DataFrame({
        "t": pd.to_numeric(x[tcol], errors="coerce"),
        "N": pd.to_numeric(x["N"], errors="coerce"),
    }).dropna().sort_values("t")
    if z.empty:
        return None
    z["t"] = z["t"] - z["t"].min()
    z = z.groupby("t", as_index=False)["N"].median().sort_values("t")
    z["logN"] = np.log10(z["N"] + 1.0)
    return z


def find_prefix(pid):
    pdir = NEW / "patients" / f"patient_{int(pid):03d}"
    candidates = list(pdir.glob("**/*prefix*.csv"))
    if not candidates:
        candidates = [f for f in pdir.glob("**/*.csv")
                      if "outcome" not in f.name.lower()
                      and "meta" not in f.name.lower()]
    for f in sorted(candidates):
        try:
            h = pd.read_csv(f, nrows=5)
        except Exception:
            continue
        if "N" in h.columns and any(c in h.columns for c in
                                    ("t_treatment", "t", "treatment_t", "cycle")):
            return f
    return None


def evenly_spaced_schedule(n_scans, end=TILE_END):
    # n observations including t=0. For example:
    # n=5 -> [0,3,6,9,12]; n=13 -> every integer cycle 0..12.
    s = np.rint(np.linspace(0, end, int(n_scans))).astype(int)
    if len(np.unique(s)) != len(s):
        raise RuntimeError(f"Duplicate scan times for n={n_scans}: {s}")
    return s


# ============================================================
# Load authoritative result tables
# ============================================================
E = pd.read_csv(NEW / "extinction_by_lag_binary300.csv")
P = pd.read_csv(NEW / "patient_oracle_characterization.csv")
A = pd.read_csv(ABC / "abc_80patients_4scan_hybrid10k.csv")
D = pd.read_csv(REGRET / "abc_oracle_regret_by_patient.csv")
R = pd.read_csv(REF / "abc_reference_features_usable.csv")

for df in (E, P, A, D):
    if "patient_id" in df.columns:
        df["patient_id"] = pd.to_numeric(df["patient_id"], errors="raise").astype(int)

E["lag"] = pd.to_numeric(E["lag"], errors="raise").astype(int)
E["extinction_probability"] = pd.to_numeric(E["extinction_probability"], errors="raise")
P["tmin"] = pd.to_numeric(P["tmin"], errors="raise")
P["lag_at_tmin"] = pd.to_numeric(P["lag_at_tmin"], errors="raise").astype(int)

A_primary = A[A["method"] == "logN_plus_slope"].copy()
A_base = A[A["method"] == "baseline_logN"].copy()
D_primary = D[D["method"] == "logN_plus_slope"].copy()

if "tmin_boundary" in R.columns:
    R = R[pd.to_numeric(R["tmin_boundary"], errors="coerce") != 1].copy()

assert E["patient_id"].nunique() == 80
assert set(E["lag"].unique()) == set(range(1, 21))
assert len(E) == 80 * 20
assert P["patient_id"].nunique() == 80
assert len(A_primary) == 80
assert len(D_primary) == 80
assert len(R) == 9983, f"Expected 9,983 clean references, found {len(R)}"

# Use the minimal characterization columns to avoid conflicts with any
# pre-existing oracle columns in P.
CHAR = P[["patient_id", "tmin", "lag_at_tmin"]].copy()

# Recalculate earliest oracle lag directly from E for reproducibility.
oracle_rows = []
for pid, g in E.groupby("patient_id"):
    mx = float(g["extinction_probability"].max())
    best = g[np.isclose(g["extinction_probability"], mx)]
    oracle_rows.append({
        "patient_id": int(pid),
        "oracle_lag": int(best["lag"].min()),
        "oracle_extinction_recalc": mx,
        "min_extinction": float(g["extinction_probability"].min()),
        "timing_leverage": mx - float(g["extinction_probability"].min()),
    })
ORACLE = pd.DataFrame(oracle_rows)
CHAR = CHAR.merge(ORACLE, on="patient_id", how="left", validate="one_to_one")
CHAR["oracle_minus_tmin_lag"] = CHAR["oracle_lag"] - CHAR["lag_at_tmin"]
# No unique/meaningful oracle lag exists when no tested switch time produced extinction.
CHAR["untreatable_observed"] = CHAR["oracle_extinction_recalc"] <= 1e-12
n_untreatable = int(CHAR["untreatable_observed"].sum())

# Load all 80 Drug-1 trajectories.
trajs = {}
traj_rows = []
for pid in sorted(E["patient_id"].unique()):
    f = find_prefix(pid)
    if f is None:
        raise FileNotFoundError(f"No prefix CSV found for patient {pid}")
    z = extract_d1(pd.read_csv(f))
    if z is None or len(z) < 13:
        raise RuntimeError(f"Bad Drug-1 prefix for patient {pid}: {f}")
    trajs[int(pid)] = z
    q = z.copy()
    q.insert(0, "patient_id", int(pid))
    traj_rows.append(q)
TRAJ = pd.concat(traj_rows, ignore_index=True)
TRAJ.to_csv(DATA / "Fig2A_drug1_trajectories.csv", index=False)

# Convenience E lookup.
E_lookup = E.set_index(["patient_id", "lag"])["extinction_probability"]


# ============================================================
# Core numerical summaries used in Results
# ============================================================
tmin = CHAR["tmin"].to_numpy(float)
true = pd.to_numeric(A_primary["true_tmin"], errors="raise").to_numpy(float)
pred = pd.to_numeric(A_primary["pred_tmin_weighted"], errors="raise").to_numpy(float)
ae = np.abs(pred - true)

abc_mae = float(ae.mean())
abc_med_ae = float(np.median(ae))
abc_rmse = float(np.sqrt(np.mean((pred - true)**2)))
abc_bias = float(np.mean(pred - true))
abc_within05 = float(np.mean(ae <= 0.5))
abc_within1 = float(np.mean(ae <= 1.0))
abc_within2 = float(np.mean(ae <= 2.0))

abc_mean = float(D_primary["abc_extinction"].mean())
tmin_mean_ext = float(D_primary["tmin_extinction"].mean())
oracle_mean = float(D_primary["oracle_extinction"].mean())
regret_mean = float(D_primary["total_oracle_regret"].mean())
nadir_gap_mean = float(D_primary["oracle_minus_true_tmin"].mean())
pred_disp_mean = float(D_primary["true_tmin_minus_abc"].mean())

abc_gt_tmin = int((D_primary["abc_extinction"] > D_primary["tmin_extinction"] + 1e-12).sum())
abc_eq_tmin = int(np.isclose(D_primary["abc_extinction"], D_primary["tmin_extinction"]).sum())
abc_lt_tmin = int((D_primary["abc_extinction"] < D_primary["tmin_extinction"] - 1e-12).sum())
zero_regret = int(np.isclose(D_primary["total_oracle_regret"], 0).sum())

assert abs(abc_mean - 0.70375) < 1e-6
assert abs(tmin_mean_ext - 0.6975) < 1e-6
assert abs(oracle_mean - 0.77) < 1e-6
assert (abc_gt_tmin, abc_eq_tmin, abc_lt_tmin) == (9, 63, 8)
assert zero_regret == 58

# Raw finite-horizon outcome count if available; otherwise recover from E.
outcome_file = NEW / "all_outcomes_resolved_to300.csv"
if outcome_file.exists():
    OUTCOMES = pd.read_csv(outcome_file)
    n_outcomes = len(OUTCOMES)
    if "status" in OUTCOMES.columns:
        status_counts = OUTCOMES["status"].astype(str).value_counts().to_dict()
        n_extinct = int(status_counts.get("extinct", 0))
    else:
        status_counts = {}
        n_extinct = int(round(E["extinction_probability"].sum() * N_REP))
else:
    OUTCOMES = None
    status_counts = {}
    n_outcomes = 80 * 20 * N_REP
    n_extinct = int(round(E["extinction_probability"].sum() * N_REP))
overall_extinction = n_extinct / n_outcomes


# ============================================================
# Reactive monitoring: fixed 4-scan ABC benchmark versus 2..13 scans
# ============================================================
reactive_rows = []
for n_scans in range(2, TILE_END + 2):
    schedule = evenly_spaced_schedule(n_scans)
    for pid, z in trajs.items():
        obsN = np.interp(schedule, z["t"].to_numpy(float), z["N"].to_numpy(float))
        increases = np.where(np.diff(obsN) > 0)[0]
        if len(increases):
            j = int(increases[0] + 1)
        else:
            j = len(schedule) - 1
        switch_t = int(schedule[j])
        lag = int(np.clip(switch_t + 1, 1, 20))
        ext = float(E_lookup.loc[(pid, lag)])
        tm = float(CHAR.loc[CHAR["patient_id"] == pid, "tmin"].iloc[0])
        reactive_rows.append({
            "patient_id": pid,
            "n_scans": n_scans,
            "schedule": ",".join(map(str, schedule)),
            "switch_t": switch_t,
            "selected_lag": lag,
            "true_tmin": tm,
            "switch_minus_tmin": switch_t - tm,
            "extinction_probability": ext,
        })
REACTIVE = pd.DataFrame(reactive_rows)
REACTIVE.to_csv(DATA / "Fig6_reactive_patient_level.csv", index=False)

reactive_summary = []
for n_scans, g in REACTIVE.groupby("n_scans", sort=True):
    m, lo, hi = bootstrap_mean_ci(g["extinction_probability"], seed_offset=100+n_scans)
    dm, dlo, dhi = bootstrap_mean_ci(g["switch_minus_tmin"], seed_offset=200+n_scans)
    reactive_summary.append({
        "n_scans": int(n_scans),
        "schedule": g["schedule"].iloc[0],
        "mean_extinction": m,
        "extinction_ci_low": lo,
        "extinction_ci_high": hi,
        "mean_switch_minus_tmin": dm,
        "delay_ci_low": dlo,
        "delay_ci_high": dhi,
    })
REACTIVE_SUM = pd.DataFrame(reactive_summary)
REACTIVE_SUM.to_csv(DATA / "Fig6_reactive_scan_budget_summary.csv", index=False)

dense = REACTIVE[REACTIVE["n_scans"] == 13][["patient_id", "extinction_probability"]].rename(
    columns={"extinction_probability": "dense_extinction"})
PAIR_DENSE = D_primary[["patient_id", "abc_extinction"]].merge(
    dense, on="patient_id", how="inner", validate="one_to_one")
PAIR_DENSE.to_csv(DATA / "Fig6_dense_vs_abc_patient_level.csv", index=False)

dense_mean = float(PAIR_DENSE["dense_extinction"].mean())
abc_beats_dense = int((PAIR_DENSE["abc_extinction"] > PAIR_DENSE["dense_extinction"] + 1e-12).sum())
tie_dense = int(np.isclose(PAIR_DENSE["abc_extinction"], PAIR_DENSE["dense_extinction"]).sum())
dense_beats_abc = int((PAIR_DENSE["abc_extinction"] < PAIR_DENSE["dense_extinction"] - 1e-12).sum())

# Diagnostic: should reproduce the already established dense-vs-ABC result.
if abs(dense_mean - 0.703) > 0.015 or (abc_beats_dense, tie_dense, dense_beats_abc) != (11, 61, 8):
    warnings.warn(
        "Dense-monitoring recomputation does not exactly match the previously established "
        f"~0.701 and 11/61/8 result. Got mean={dense_mean:.4f}, "
        f"ABC/tie/dense={abc_beats_dense}/{tie_dense}/{dense_beats_abc}. "
        "Check the historical scan-schedule convention before manuscript use."
    )


# ============================================================
# FIGURE 2: heterogeneous trajectories and treatment windows
# ============================================================
fig = plt.figure(figsize=(7.25, 7.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.45], hspace=0.36, wspace=0.32)

# A: Drug-1 trajectories
ax = fig.add_subplot(gs[0, 0])
grid = np.arange(0, 16, dtype=float)
Q = []
for pid, z in trajs.items():
    q = z[z["t"] <= 15]
    ax.plot(q["t"], q["logN"], color=MIDGRAY, alpha=0.22, lw=0.65)
    if z["t"].max() >= 15:
        Q.append(np.interp(grid, z["t"], z["logN"]))
    tm = float(CHAR.loc[CHAR["patient_id"] == pid, "tmin"].iloc[0])
    ytm = float(np.interp(tm, z["t"], z["logN"]))
    ax.scatter(tm, ytm, s=7, color=ORANGE, alpha=0.55, zorder=3)
Q = np.vstack(Q)
ax.plot(grid, np.median(Q, axis=0), color=BLACK, lw=1.8, label="Median trajectory")
ax.axvline(3, color=BLUE, ls="--", lw=1.0, label="Last of 4 early scans")
ax.set_xlim(0, 15)
ax.set_xlabel("Drug-1 treatment cycle")
ax.set_ylabel(r"$\log_{10}(N+1)$")
ax.set_title("Drug-1 responses vary among patients")
ax.legend(frameon=False, loc="best")
panel(ax, "a")

# B: mean extinction aligned to each patient's true nadir
REL = E.merge(CHAR[["patient_id", "lag_at_tmin"]], on="patient_id", validate="many_to_one")
REL["lag_relative_to_tmin"] = REL["lag"] - REL["lag_at_tmin"]
rel_rows = []
for off, g in REL.groupby("lag_relative_to_tmin", sort=True):
    if len(g) < 20:
        continue
    m, lo, hi = bootstrap_mean_ci(g["extinction_probability"], seed_offset=300+int(off)+50)
    rel_rows.append({"relative_lag": int(off), "n_patients": len(g),
                     "mean_extinction": m, "ci_low": lo, "ci_high": hi})
REL_SUM = pd.DataFrame(rel_rows)
REL_SUM.to_csv(DATA / "Fig2B_extinction_relative_to_tmin.csv", index=False)
ax = fig.add_subplot(gs[0, 1])
x = REL_SUM["relative_lag"].to_numpy()
y = REL_SUM["mean_extinction"].to_numpy()
lo = REL_SUM["ci_low"].to_numpy()
hi = REL_SUM["ci_high"].to_numpy()
ax.fill_between(x, lo, hi, color=SKY, alpha=0.25, linewidth=0)
ax.plot(x, y, color=BLUE, marker="o", ms=3.5, lw=1.6)
ax.axvline(0, color=BLACK, ls="--", lw=1.0)
ax.set_ylim(0, 1.02)
ax.set_xlabel(r"Second-strike lag relative to true $t_{\min}$")
ax.set_ylabel("Mean extinction probability")
ax.set_title("High-extinction timing spans the nadir")
panel(ax, "b")

# C: patient-by-lag extinction landscape
order = CHAR.sort_values(["tmin", "patient_id"])["patient_id"].tolist()
mat = E.pivot(index="patient_id", columns="lag", values="extinction_probability").loc[order, range(1, 21)]
mat.to_csv(DATA / "Fig2C_extinction_landscape_matrix.csv")
ax = fig.add_subplot(gs[1, :])
im = ax.imshow(mat.to_numpy(), aspect="auto", interpolation="nearest",
               vmin=0, vmax=1, cmap="viridis")
char_i = CHAR.set_index("patient_id")
for yy, pid in enumerate(order):
    row = char_i.loc[pid]
    ax.scatter(row["lag_at_tmin"] - 1, yy,
               s=10, facecolors="white", edgecolors=BLACK,
               linewidths=0.45, zorder=6)

    # An all-zero extinction landscape has no biologically meaningful oracle lag.
    # Mark the patient separately rather than assigning the earliest tied lag (=1).
    if bool(row["untreatable_observed"]):
        ax.scatter(-0.72, yy, s=15, marker="s", color=BLACK,
                   edgecolor="white", linewidth=0.35, clip_on=False, zorder=5)
    else:
        ax.scatter(row["oracle_lag"] - 1, yy,
                   s=12, marker="x", color=ORANGE, linewidths=0.75)

ax.set_xlim(-1.05, 19.5)
ax.set_xticks(np.arange(20))
ax.set_xticklabels(np.arange(1, 21))
ax.set_xlabel("Second-strike lag")
ax.set_ylabel(r"Patients sorted by true $t_{\min}$")
ax.set_title("Treatment-timing landscapes are patient specific", pad=8)
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Extinction probability by cycle 300")
# Marker key.
ax.scatter([], [], s=22, facecolors="white", edgecolors=BLACK,
           linewidths=0.6, label=r"True $t_{\min}$ lag")
ax.scatter([], [], s=20, marker="x", color=ORANGE, label="Earliest max-extinction lag")
ax.scatter([], [], s=20, marker="s", color=BLACK, label="No extinction at any tested lag")
leg = ax.legend(frameon=True, loc="lower right", facecolor="white", framealpha=0.92,
                fontsize=7.2)
for txt in leg.get_texts():
    txt.set_color(BLACK)
panel(ax, "c")

save(fig, "Fig2_patient_timing_landscapes")


# ============================================================
# FIGURE 4: ABC prediction of the future nadir
# ============================================================
A_primary.to_csv(DATA / "Fig4_abc_predictions.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.35), gridspec_kw={"wspace": 0.38})

ax = axes[0]
loxy = min(true.min(), pred.min()) - 0.6
hixy = max(true.max(), pred.max()) + 0.6
xx = np.linspace(loxy, hixy, 200)
ax.fill_between(xx, xx-2, xx+2, color=LIGHTGRAY, alpha=0.55, linewidth=0,
                label=r"$\pm 2$ cycles")
ax.plot(xx, xx, color=BLACK, ls="--", lw=1.0)
ax.scatter(true, pred, s=22, color=BLUE, alpha=0.72, edgecolor="white", linewidth=0.3)
ax.set_xlim(loxy, hixy)
ax.set_ylim(loxy, hixy)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel(r"True $t_{\min}$")
ax.set_ylabel(r"Predicted $\widehat{t}_{\min}$")
ax.set_title("Four early scans predict the future nadir", pad=10)
ax.text(0.04, 0.96,
        f"MAE = {abc_mae:.3f} cycles\nRMSE = {abc_rmse:.3f}\nBias = {abc_bias:+.3f}",
        transform=ax.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=LIGHTGRAY, alpha=0.92))
ax.legend(frameon=False, loc="lower right")
panel(ax, "a", x=-0.24, y=1.06)

ax = axes[1]
sx = np.sort(ae)
sy = np.arange(1, len(sx)+1) / len(sx)
ax.step(sx, sy, where="post", color=BLUE, lw=1.8)
ax.axvline(1, color=BLACK, ls="--", lw=1.0)
ax.axvline(2, color=BLACK, ls=":", lw=1.0)
ax.scatter([1, 2], [abc_within1, abc_within2], color=ORANGE, s=28, zorder=5)
ax.text(1, abc_within1-0.07, f"{abc_within1:.1%}", ha="center", va="top")
ax.text(2, abc_within2-0.07, f"{abc_within2:.1%}", ha="center", va="top")
ax.set_xlim(left=0)
ax.set_ylim(0, 1.02)
ax.set_xlabel(r"Absolute $t_{\min}$ prediction error (cycles)")
ax.set_ylabel("Fraction of patients")
ax.set_title("Most predictions fall within two cycles", pad=10)
panel(ax, "b", x=-0.20, y=1.06)

save(fig, "Fig4_ABC_nadir_prediction")


# ============================================================
# FIGURE 5: policy performance and regret decomposition
# ============================================================
D_primary.to_csv(DATA / "Fig5_policy_regret_patient_level.csv", index=False)
fig, ax = plt.subplots(figsize=(5.35, 3.85))
cols = ["abc_extinction", "tmin_extinction", "oracle_extinction"]
labels = ["4-scan ABC", r"True $t_{\min}$", "Oracle"]
colors = [BLUE, GREEN, ORANGE]

# Patient-level paired outcomes.
for _, r in D_primary.iterrows():
    ax.plot([0, 1, 2], [r[c] for c in cols], color=MIDGRAY, alpha=0.13, lw=0.55)

# Cohort means with patient-bootstrap 95% CIs.
for j, (c, color) in enumerate(zip(cols, colors)):
    m, lo, hi = bootstrap_mean_ci(D_primary[c], seed_offset=400+j)
    ax.errorbar(j, m, yerr=[[max(0.0, m-lo)], [max(0.0, hi-m)]],
                fmt="o", color=color, ms=6.5, capsize=3, lw=1.4, zorder=6)
    ax.text(j, min(1.035, m+0.065), f"{m:.3f}", ha="center", va="bottom",
            color=color, fontsize=9)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(labels)
ax.set_xlim(-0.42, 2.42)
ax.set_ylim(-0.03, 1.08)
ax.set_ylabel("Extinction probability")
ax.set_title("Prediction error contributes little to remaining treatment regret", pad=10)

# Keep the decomposition visible without a redundant second panel.
ax.text(0.025, 0.055,
        f"Oracle - ABC = {regret_mean:.3f}\n"
        f"Oracle - true nadir = {nadir_gap_mean:.3f}\n"
        f"True nadir - ABC = {pred_disp_mean:+.3f}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=LIGHTGRAY, alpha=0.92))

fig.subplots_adjust(left=0.15, right=0.98, bottom=0.18, top=0.88)
save(fig, "Fig5_timing_regret")


# ============================================================
# FIGURE 6: prediction versus continued reactive monitoring
# ============================================================
fig = plt.figure(figsize=(7.25, 6.25), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12], wspace=0.20, hspace=0.18)

ax = fig.add_subplot(gs[0, 0])
x = REACTIVE_SUM["n_scans"].to_numpy()
y = REACTIVE_SUM["mean_extinction"].to_numpy()
lo = REACTIVE_SUM["extinction_ci_low"].to_numpy()
hi = REACTIVE_SUM["extinction_ci_high"].to_numpy()
ax.fill_between(x, lo, hi, color=LIGHTGRAY, alpha=0.75, linewidth=0)
ax.plot(x, y, color=GREEN, marker="o", ms=4, lw=1.7, label="Reactive monitoring")
ax.axhline(abc_mean, color=BLUE, ls="--", lw=1.5,
           label=f"4 early scans + ABC ({abc_mean:.3f})")
ax.set_xticks(np.arange(2, 14))
ax.set_ylim(0, 1.0)
ax.set_xlabel("Number of reactive observations")
ax.set_ylabel("Mean extinction probability")
ax.set_title("Reactive monitoring improves with scan density", pad=9)
ax.legend(frameon=False, loc="lower right", fontsize=7.5)
panel(ax, "a", x=-0.18, y=1.07)

ax = fig.add_subplot(gs[0, 1])
y = REACTIVE_SUM["mean_switch_minus_tmin"].to_numpy()
lo = REACTIVE_SUM["delay_ci_low"].to_numpy()
hi = REACTIVE_SUM["delay_ci_high"].to_numpy()
ax.fill_between(x, lo, hi, color=LIGHTGRAY, alpha=0.75, linewidth=0)
ax.plot(x, y, color=GREEN, marker="o", ms=4, lw=1.7)
ax.axhline(0, color=BLACK, ls="--", lw=0.8)
ax.set_xticks(np.arange(2, 14))
ax.set_xlabel("Number of reactive observations")
ax.set_ylabel(r"Switch delay from $t_{\min}$ (cycles)")
ax.set_title("Denser monitoring localizes the nadir sooner", pad=9)
panel(ax, "b", x=-0.18, y=1.07)

ax = fig.add_subplot(gs[1, :])
ax.plot([-0.03, 1.03], [-0.03, 1.03], color=BLACK, ls="--", lw=1.0)
ax.scatter(PAIR_DENSE["dense_extinction"], PAIR_DENSE["abc_extinction"],
           s=30, color=BLUE, alpha=0.60, edgecolor="white", linewidth=0.4)
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.03, 1.03)
# Do not force an equal aspect ratio: using the full-width axis prevents label collisions.
ax.set_xlabel("Dense reactive extinction probability")
ax.set_ylabel("4-scan ABC extinction probability")
ax.set_title("Four early scans match dense monitoring", pad=9)
ax.text(0.025, 0.97,
        f"Mean ABC = {abc_mean:.3f}\nMean dense = {dense_mean:.3f}\n"
        f"ABC / tie / dense = {abc_beats_dense} / {tie_dense} / {dense_beats_abc}",
        transform=ax.transAxes, ha="left", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=LIGHTGRAY, alpha=0.93))
panel(ax, "c", x=-0.075, y=1.06)

save(fig, "Fig6_prediction_vs_reactive_monitoring")


# ============================================================
# FIGURE 3: Amjad-style illustrative patient windows
# Uses Wilson 95% binomial CIs, as in Patil et al.'s simulation plots.
# ============================================================
aligned = CHAR[(CHAR["oracle_lag"] == CHAR["lag_at_tmin"]) &
               (CHAR["oracle_extinction_recalc"] >= 0.5)].copy()
if aligned.empty:
    aligned = CHAR[CHAR["oracle_lag"] == CHAR["lag_at_tmin"]].copy()
aligned = aligned.sort_values(["timing_leverage", "oracle_extinction_recalc"], ascending=False)
pid_aligned = int(aligned.iloc[0]["patient_id"])

shifted = CHAR[(CHAR["oracle_lag"] > CHAR["lag_at_tmin"]) &
               (CHAR["oracle_extinction_recalc"] >= 0.5)].copy()
if shifted.empty:
    shifted = CHAR[CHAR["oracle_lag"] != CHAR["lag_at_tmin"]].copy()
shifted["abs_shift"] = np.abs(shifted["oracle_minus_tmin_lag"])
shifted = shifted.sort_values(["timing_leverage", "abs_shift", "oracle_extinction_recalc"], ascending=False)
pid_shifted = int(shifted.iloc[0]["patient_id"])

example_ids = [pid_aligned, pid_shifted]
pd.DataFrame({"example": ["oracle_at_nadir", "oracle_after_nadir"],
              "patient_id": example_ids}).to_csv(DATA / "Fig3_example_patient_ids.csv", index=False)

fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.3), sharex="col",
                         gridspec_kw={"height_ratios": [1, 1.05], "hspace": 0.10, "wspace": 0.30})
for col, pid in enumerate(example_ids):
    row = CHAR.set_index("patient_id").loc[pid]
    z = trajs[pid]
    tm = float(row["tmin"])
    olag = int(row["oracle_lag"])
    ot = olag - 1

    ax = axes[0, col]
    q = z[z["t"] <= 19]
    ax.plot(q["t"], q["logN"], color=BLUE, lw=1.7)
    ax.axvline(tm, color=BLACK, ls="--", lw=1.0, label=r"True $t_{\min}$")
    ax.axvline(ot, color=ORANGE, ls=":", lw=1.4, label="Oracle switch")
    ax.set_ylabel(r"$\log_{10}(N+1)$")
    ax.set_title(("Oracle at the nadir" if col == 0 else "Oracle after the nadir") + f"\nPatient {pid}")
    if col == 0:
        ax.legend(frameon=False, loc="best")
    panel(ax, "a" if col == 0 else "c")

    ax = axes[1, col]
    g = E[E["patient_id"] == pid].sort_values("lag")
    p = g["extinction_probability"].to_numpy(float)
    lags = g["lag"].to_numpy(int)
    wlo, whi = wilson_ci(p)
    yerr_lo = np.maximum(0.0, p - wlo)
    yerr_hi = np.maximum(0.0, whi - p)
    ax.errorbar(lags, p, yerr=[yerr_lo, yerr_hi], fmt="o-", color=GREEN,
                ms=3.5, lw=1.2, capsize=2, ecolor=MIDGRAY)
    ax.axvline(tm+1, color=BLACK, ls="--", lw=1.0)
    ax.axvline(olag, color=ORANGE, ls=":", lw=1.4)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(1, 20)
    ax.set_xlabel("Second-strike lag")
    ax.set_ylabel("Extinction probability")
    panel(ax, "b" if col == 0 else "d")

save(fig, "Fig3_illustrative_patient_windows")


# ============================================================
# SUPPLEMENTARY FIGURE S1: ABC reference-library support
# ============================================================
# This is a support/QC figure rather than a main biological result.
sample_n = min(150, len(R))
sample_ids = RNG.choice(R["ref_id"].astype(int).to_numpy(), size=sample_n, replace=False)
sample_trajs = []
for rid in sample_ids:
    f = REF / "prefixes" / f"ref_{int(rid):05d}.csv"
    if not f.exists():
        continue
    z = extract_d1(pd.read_csv(f))
    if z is not None and z["t"].max() >= 15:
        sample_trajs.append(z)

fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.25), gridspec_kw={"wspace": 0.34})
ax = axes[0]
grid = np.arange(0, 16, dtype=float)
Qref = []
for z in sample_trajs:
    q = z[z["t"] <= 15]
    ax.plot(q["t"], q["logN"], color=MIDGRAY, alpha=0.10, lw=0.55)
    Qref.append(np.interp(grid, z["t"], z["logN"]))
if Qref:
    Qref = np.vstack(Qref)
    med = np.median(Qref, axis=0)
    q10 = np.quantile(Qref, 0.10, axis=0)
    q90 = np.quantile(Qref, 0.90, axis=0)
    ax.fill_between(grid, q10, q90, color=SKY, alpha=0.22, linewidth=0, label="10-90% interval")
    ax.plot(grid, med, color=BLACK, lw=1.8, label="Median")
ax.axvline(3, color=BLUE, ls="--", lw=1.0, label="Last of 4 early scans")
ax.set_xlim(0, 15)
ax.set_xlabel("Drug-1 treatment cycle")
ax.set_ylabel(r"$\log_{10}(N+1)$")
ax.set_title("Independent ABC reference trajectories", pad=9)
ax.legend(frameon=False, fontsize=7.2)
panel(ax, "a", x=-0.20, y=1.06)

ax = axes[1]
ref_t = pd.to_numeric(R["t_min_true"], errors="coerce").dropna().to_numpy(float)
ax.hist(ref_t, bins=np.arange(3.5, 20.6, 1), color=LIGHTGRAY, edgecolor="white",
        label=f"References (n={len(ref_t):,})")
# Held-out patients shown as a rug so support is visually obvious.
yrug = np.full(len(tmin), -0.015 * max(1, ax.get_ylim()[1]))
ax.scatter(tmin, yrug, marker="|", s=55, color=ORANGE, clip_on=False,
           label="Held-out patients")
ax.axvline(3, color=BLUE, ls="--", lw=1.0)
ax.set_xlabel(r"True $t_{\min}$")
ax.set_ylabel("Reference trajectories")
ax.set_title("Patient nadirs lie within reference support", pad=9)
ax.legend(frameon=False, fontsize=7.2)
panel(ax, "b", x=-0.20, y=1.06)

save(fig, "FigS1_ABC_reference_support")

# Compact numerical comparison of the two ABC representations; no main figure needed.
def prediction_metrics(df, label):
    tr = pd.to_numeric(df["true_tmin"], errors="raise").to_numpy(float)
    pr = pd.to_numeric(df["pred_tmin_weighted"], errors="raise").to_numpy(float)
    er = np.abs(pr-tr)
    return {
        "method": label,
        "n": len(df),
        "MAE": er.mean(),
        "median_AE": np.median(er),
        "RMSE": np.sqrt(np.mean((pr-tr)**2)),
        "bias": np.mean(pr-tr),
        "within_1_cycle": np.mean(er <= 1),
        "within_2_cycles": np.mean(er <= 2),
    }

pd.DataFrame([
    prediction_metrics(A_primary, "logN_plus_slope"),
    prediction_metrics(A_base, "baseline_logN"),
]).to_csv(DATA / "TableS1_ABC_representation_comparison.csv", index=False)


# ============================================================
# Results source-data tables and manuscript-safe numbers
# ============================================================
summary = [
    ("n_patients", 80),
    ("n_candidate_lags", 20),
    ("post_switch_replicates_per_lag", N_REP),
    ("n_counterfactual_outcome_runs", n_outcomes),
    ("n_extinct_by_cycle_300", n_extinct),
    ("n_no_extinction_at_any_tested_lag", n_untreatable),
    ("overall_extinction_frequency", overall_extinction),
    ("tmin_mean", float(np.mean(tmin))),
    ("tmin_sd", float(np.std(tmin, ddof=1))),
    ("tmin_median", float(np.median(tmin))),
    ("tmin_min", float(np.min(tmin))),
    ("tmin_max", float(np.max(tmin))),
    ("n_clean_abc_references", len(R)),
    ("reference_tmin_min", float(R["t_min_true"].min())),
    ("reference_tmin_max", float(R["t_min_true"].max())),
    ("abc_mae", abc_mae),
    ("abc_median_absolute_error", abc_med_ae),
    ("abc_rmse", abc_rmse),
    ("abc_bias", abc_bias),
    ("abc_within_0.5", abc_within05),
    ("abc_within_1", abc_within1),
    ("abc_within_2", abc_within2),
    ("abc_mean_extinction", abc_mean),
    ("true_tmin_mean_extinction", tmin_mean_ext),
    ("oracle_mean_extinction", oracle_mean),
    ("mean_total_oracle_regret", regret_mean),
    ("mean_nadir_policy_gap", nadir_gap_mean),
    ("mean_prediction_displacement", pred_disp_mean),
    ("abc_beats_true_tmin_n", abc_gt_tmin),
    ("abc_ties_true_tmin_n", abc_eq_tmin),
    ("abc_worse_true_tmin_n", abc_lt_tmin),
    ("zero_oracle_regret_n", zero_regret),
    ("dense_reactive_mean_extinction", dense_mean),
    ("abc_beats_dense_n", abc_beats_dense),
    ("abc_ties_dense_n", tie_dense),
    ("dense_beats_abc_n", dense_beats_abc),
]
SUMMARY = pd.DataFrame(summary, columns=["metric", "value"])
SUMMARY.to_csv(DATA / "results_summary.csv", index=False)

text = f"""FINAL SIMULATION RESULTS NUMBERS
================================
Patients: 80
Candidate second-strike lags: 20
Post-switch replicates per lag: {N_REP}
Counterfactual outcome runs: {n_outcomes:,}
Extinct by cycle 300: {n_extinct:,} ({overall_extinction:.1%})
Patients with no extinction at any tested lag: {n_untreatable}/80

Patient t_min: mean {np.mean(tmin):.3f}, SD {np.std(tmin, ddof=1):.3f}, median {np.median(tmin):.0f}, range {np.min(tmin):.0f}-{np.max(tmin):.0f}
ABC reference library: n={len(R):,}, t_min support {R['t_min_true'].min():.0f}-{R['t_min_true'].max():.0f}

4-scan ABC t_min prediction
MAE: {abc_mae:.6f}
Median absolute error: {abc_med_ae:.6f}
RMSE: {abc_rmse:.6f}
Bias: {abc_bias:+.6f}
Within 0.5 cycle: {abc_within05:.1%}
Within 1 cycle: {abc_within1:.1%}
Within 2 cycles: {abc_within2:.1%}

Treatment policies
ABC mean extinction: {abc_mean:.5f}
True-t_min mean extinction: {tmin_mean_ext:.5f}
Oracle mean extinction: {oracle_mean:.5f}
Mean oracle regret of ABC: {regret_mean:.5f}
Mean nadir-policy gap: {nadir_gap_mean:.5f}
Mean prediction displacement effect: {pred_disp_mean:+.5f}
ABC / tie / true-t_min: {abc_gt_tmin} / {abc_eq_tmin} / {abc_lt_tmin}
Zero estimated oracle regret: {zero_regret}/80

Dense reactive benchmark
Dense mean extinction: {dense_mean:.5f}
ABC / tie / dense: {abc_beats_dense} / {tie_dense} / {dense_beats_abc}
"""
(DATA / "results_numbers.txt").write_text(text)
print("\n" + text)

# Optional LaTeX macros to eliminate manual transcription errors.
macros = rf"""% Auto-generated by make_final_results_figures.py
\newcommand{{\NPatients}}{{80}}
\newcommand{{\NOutcomeRuns}}{{{n_outcomes:,}}}
\newcommand{{\NUntreatable}}{{{n_untreatable}}}
\newcommand{{\OverallExtinctionPct}}{{{100*overall_extinction:.1f}\%}}
\newcommand{{\TminMean}}{{{np.mean(tmin):.2f}}}
\newcommand{{\TminSD}}{{{np.std(tmin, ddof=1):.2f}}}
\newcommand{{\ABCMAE}}{{{abc_mae:.3f}}}
\newcommand{{\ABCRMSE}}{{{abc_rmse:.3f}}}
\newcommand{{\ABCWithinTwo}}{{{100*abc_within2:.1f}\%}}
\newcommand{{\ABCMeanExtinction}}{{{abc_mean:.3f}}}
\newcommand{{\TrueNadirMeanExtinction}}{{{tmin_mean_ext:.3f}}}
\newcommand{{\OracleMeanExtinction}}{{{oracle_mean:.3f}}}
\newcommand{{\ABCOracleRegret}}{{{regret_mean:.3f}}}
\newcommand{{\NadirPolicyGap}}{{{nadir_gap_mean:.3f}}}
\newcommand{{\PredictionDisplacement}}{{{pred_disp_mean:+.3f}}}
\newcommand{{\DenseMeanExtinction}}{{{dense_mean:.3f}}}
"""
(DATA / "results_macros.tex").write_text(macros)
print("WROTE", DATA / "results_summary.csv")
print("WROTE", DATA / "results_numbers.txt")
print("WROTE", DATA / "results_macros.tex")
print("\nDONE. Main figures are in:", FIG)
print("Source data are in:", DATA)
