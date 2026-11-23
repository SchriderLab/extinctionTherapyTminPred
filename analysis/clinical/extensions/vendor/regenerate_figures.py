#!/usr/bin/env python3
"""Render manuscript Figures 9-10 and S6-S20 from the frozen Treb reanalysis.

No simulations, fits, hyperparameter selection, or prediction changes are made.
Patient bootstrap intervals added for display condition on the saved models.
Run real computation on a compute node, not a Longleaf login node.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score

VERSION = "2026-10-29.3-dan-review"
PRIMARY = "ABC_noisy_joint"
BASE = "Constant_TRAIN"
DATASETS = ("venice", "mainsail")
LABEL = {
    PRIMARY: "ABC + TRAIN residuals", "ABC_clean_joint": "ABC, clean curves",
    "ABC_clean_d_only": "ABC, uniform shape weights", BASE: "TRAIN median",
    "ABC_GP": "ABC + GP correction", "XGBoost_absolute": "XGBoost, absolute loss",
    "XGBoost_squared": "XGBoost, squared loss", "Clinical_PSA": "Clinical PSA model",
    "Rejection_ABC": "Rejection ABC", "EB_Treb_joint": "Empirical joint prior",
    "EB_Treb_qg_shuffled": "Shuffled q-g prior", "MAINSAIL_to_VENICE": "MAINSAIL-trained ridge",
    "MAINSAIL_constant": "MAINSAIL median", "MAINSAIL_internal_CV": "MAINSAIL internal CV",
    "Covariates_PSA": "Early PSA", "Covariates_baseline": "Baseline variables",
    "Covariates_PSA_baseline": "PSA + baseline",
}
COLOR = {
    PRIMARY: "#007F86", "ABC_clean_joint": "#BF8A3B", "ABC_clean_d_only": "#BBA46A",
    BASE: "#646B73", "ABC_GP": "#7B5BA7", "XGBoost_absolute": "#C56528",
    "XGBoost_squared": "#B24562", "Clinical_PSA": "#367544", "Rejection_ABC": "#CB7642",
    "EB_Treb_joint": "#3269A8", "EB_Treb_qg_shuffled": "#B46F36",
    "MAINSAIL_to_VENICE": "#3269A8", "MAINSAIL_constant": "#646B73",
    "MAINSAIL_internal_CV": "#7B5BA7", "Covariates_PSA": "#3269A8",
    "Covariates_baseline": "#B46F36", "Covariates_PSA_baseline": "#7B5BA7",
}
TREB = r"$T_{\mathrm{reb}}$"
TFIT = r"$T_{\mathrm{fit}}$"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def boolean(x):
    return x.astype(str).str.lower().isin(["true", "1", "yes"])


def json_write(path, value):
    def default(x):
        if isinstance(x, (np.integer, np.floating)): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=default, allow_nan=False) + "\n")


def rho(y, p):
    return float(spearmanr(y, p).statistic) if np.std(y) > 0 and np.std(p) > 0 else np.nan


def metric(frame, name):
    if name in ("AUROC", "Brier", "balanced_accuracy", "early_recall", "late_recall"):
        z = frame.dropna(subset=["Treb", "prob_late8"])
        if z.empty: return np.nan
        y, p = z.Treb.to_numpy() > 8, np.clip(z.prob_late8.to_numpy(), 0, 1)
        if name == "Brier": return np.mean((y - p) ** 2)
        if len(np.unique(y)) < 2: return np.nan
        if name == "AUROC": return roc_auc_score(y, p)
        call = p >= z.threshold.to_numpy()
        if name == "balanced_accuracy": return balanced_accuracy_score(y, call)
        return np.mean(call[y]) if name == "late_recall" else np.mean(~call[~y])
    z = frame.dropna(subset=["Treb", "pred"])
    if z.empty: return np.nan
    err = z.pred.to_numpy() - z.Treb.to_numpy()
    if name == "MAE": return np.mean(np.abs(err))
    if name == "Spearman_rho": return rho(z.Treb, z.pred)
    if name == "within1": return np.mean(np.abs(err) <= 1)
    raise ValueError(name)


class Source:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.inputs = {}
        self.checks = []
        self.config = self.read_json("CONFIG_FROZEN.json")
        self.prepared_lock = self.read_json("PREPARED_INPUTS_LOCK.json")
        present = 0
        for relative, expected in self.prepared_lock.items():
            path = self.root / relative
            if path.exists():
                self.record(path)
                if digest(path) != expected: raise RuntimeError(f"Prepared input hash mismatch: {relative}")
                present += 1
        self.checks.append(f"All {present} available prepared inputs match their saved hashes")
        self.parent = self.read("cohort/PARENT_231_PRIVATE.csv")
        self.train = self.read("training/fits_PRIVATE.csv")
        self.test = self.read("evaluation/heldout_reference_PRIVATE.csv")
        self.main = self.read("evaluation/mainsail_reference_PRIVATE.csv")
        assert len(self.parent) == 231 and len(self.train) == 184 and len(self.test) == 47
        assert len(self.main) == 181
        assert not set(self.train.id) & set(self.test.id)
        assert set(self.train.id) | set(self.test.id) == set(self.parent.id)
        assert set(self.parent.loc[self.parent.split.eq("TRAIN"), "id"]) == set(self.train.id)
        assert set(self.parent.loc[self.parent.split.eq("TEST"), "id"]) == set(self.test.id)
        split = self.read("cohort/SPLIT_PRIVATE.csv").sort_values("id").reset_index(drop=True)
        pd.testing.assert_frame_equal(split, self.parent[["id", "split"]].sort_values("id").reset_index(drop=True))
        self.checks.append("231 parent patients; disjoint 184/47 split; MAINSAIL n=181")
        lock = self.read_json("predictions/ALL_PREDICTIONS_LOCK.json")
        for filename, expected in lock.items():
            path = self.root / "predictions" / filename
            self.record(path)
            if digest(path) != expected: raise RuntimeError(f"Prediction hash mismatch: {filename}")
        self.checks.append(f"All {len(lock)} prediction hashes match the saved lock")
        self.target = pd.concat([self.test.assign(dataset="venice"), self.main.assign(dataset="mainsail")], ignore_index=True)
        pieces = [self.read("predictions/" + f) for f in
                  ["fixed_PRIVATE.csv", "empirical_PRIVATE.csv", "covariates_PRIVATE.csv", "cross_trial_PRIVATE.csv"]]
        self.pred = pd.concat(pieces, ignore_index=True).merge(
            self.target[["dataset", "id", "Treb"]], on=["dataset", "id"], validate="many_to_one")
        assert not self.pred.duplicated(["dataset", "id", "scan_count", "model"]).any()
        self.fixed_ids = {}
        for dataset, expected_n in [("venice", 28), ("mainsail", 94)]:
            z = self.pred[(self.pred.dataset == dataset) & (self.pred.model == BASE) &
                          (self.pred.scan_count == 6) & (self.pred.Treb > 5)]
            self.fixed_ids[dataset] = set(z.id)
            assert len(z) == expected_n
        self.seq = self.read("predictions/sequential_PRIVATE.csv").merge(
            self.target[["dataset", "id", "Treb", "event_index", "confirm_cycle"]],
            on=["dataset", "id"], validate="many_to_one")
        self.seq["window"] = (abs(self.seq.risk_index - self.seq.event_index) <= 1).astype(int)
        self.seq["before_confirmation"] = self.seq.risk_index <= self.seq.event_index
        assert not (self.seq.risk_cycle > self.seq.confirm_cycle).any()
        self.checks.append("Sequential updates stop at confirmation; one-rise endpoint reused")
        self.scores = self.read("results/METRICS.csv")
        checked = 0
        for r in self.scores.itertuples():
            if r.model not in set(self.pred.model): continue
            z = self.select(r.dataset, r.scan_count, r.model, r.stratum)
            assert len(z) == r.n_target
            for key in ["MAE", "Spearman_rho", "AUROC", "Brier", "balanced_accuracy"]:
                a, b = metric(z, key), getattr(r, key)
                if not (pd.isna(a) and pd.isna(b)) and not np.isclose(a, b, rtol=1e-7, atol=1e-8, equal_nan=True):
                    raise RuntimeError(f"Saved metric mismatch: {r.dataset}/{r.scan_count}/{r.model}/{r.stratum}/{key}")
                checked += 1
        self.checks.append(f"Recomputed {checked} plotted-model metric values against saved results")
        self.meta = self.read_json("simulation/REFERENCE_META.json")
        for relative, expected in self.meta["sha256"].items():
            path = self.root / relative
            self.record(path)
            if digest(path) != expected: raise RuntimeError(f"Reference input hash mismatch: {relative}")
        self.checks.append("TRAIN fits, TRAIN histories and simulation design hashes verified")

    def record(self, path):
        self.inputs[str(Path(path).resolve())] = digest(path)

    def read(self, relative):
        path = self.root / relative
        self.record(path)
        return pd.read_csv(path, dtype={"id": str})

    def read_json(self, relative):
        path = self.root / relative
        self.record(path)
        return json.loads(path.read_text())

    def select(self, dataset, scans, model, stratum="fixed_future6"):
        z = self.pred[(self.pred.dataset == dataset) & (self.pred.scan_count == scans) & (self.pred.model == model)].copy()
        if stratum == "future_Treb": z = z[z.Treb > scans - 1]
        elif stratum == "fixed_future6": z = z[z.id.isin(self.fixed_ids[dataset])]
        elif stratum != "all_assessable": raise ValueError(stratum)
        return z.sort_values("id").reset_index(drop=True)

    def test_history(self, adsl=None, adlb=None):
        query = self.root / "queries/venice_long_PRIVATE.csv"
        if query.exists(): return self.read("queries/venice_long_PRIVATE.csv")
        if not adsl or not adlb:
            raise FileNotFoundError("S20 needs queries/venice_long_PRIVATE.csv. Use the full HPC output directory, "
                                    "or supply --venice-adsl and --venice-adlb from the original run.")
        original = self.read_json("cohort/COHORT_REPRODUCTION.json")["input_sha256"]
        for path in [Path(adsl), Path(adlb)]:
            self.record(path)
            expected = [v for k, v in original.items() if Path(k).name == path.name]
            if len(expected) != 1 or digest(path) != expected[0]:
                raise RuntimeError(f"Original raw-input hash not matched: {path.name}")
        a = pd.read_csv(adsl, low_memory=False)
        l = pd.read_csv(adlb, low_memory=False)
        for tab in [a, l]: tab["USUBJID"] = tab.USUBJID.astype(str).str.strip()
        a = a[a.RANDOM.astype(str).str.upper().eq("Y") & a.ARMCD.astype(str).str.upper().eq("PLACEBO")]
        baseline = pd.to_numeric(a.set_index("USUBJID").TMPSA, errors="coerce")
        baseline = baseline[baseline > 0]
        l = l[l.LBTESTCD.astype(str).str.upper().str.contains("PSA", na=False) &
              l.EPOCH.astype(str).str.strip().str.casefold().eq("treatment phase")].copy()
        l["cycle"] = pd.to_numeric(l.ANLVISN, errors="coerce")
        l["psa"] = pd.to_numeric(l.LBSTRESN, errors="coerce")
        l["day"] = pd.to_numeric(l.LBDY2, errors="coerce").fillna(pd.to_numeric(l.LBDY, errors="coerce"))
        l = l[l.cycle.notna() & (l.psa > 0) & l.USUBJID.isin(baseline.index)].copy()
        l["cycle"] = l.cycle.astype(int)
        l = l[l.cycle >= 1].copy()
        l["day_error"] = (l.day - 21 * l.cycle).abs().fillna(np.inf)
        l = l.sort_values(["USUBJID", "cycle", "day_error", "day"]).drop_duplicates(["USUBJID", "cycle"])
        l = l.rename(columns={"USUBJID": "id"})[["id", "cycle", "psa", "day"]]
        b = pd.DataFrame({"id": baseline.index, "cycle": 0, "psa": baseline.values, "day": 0.0})
        z = pd.concat([b, l]).sort_values(["id", "cycle"]).reset_index(drop=True)
        z = z[z.id.isin(self.test.id)]
        expected_query_hash = self.prepared_lock["queries/venice_long_PRIVATE.csv"]
        if hashlib.sha256(z.to_csv(index=False).encode()).hexdigest() != expected_query_hash:
            raise RuntimeError("Reconstructed VENICE history does not match the saved query hash")
        # Verify target extraction, including ties and adjacent available observations.
        for r in self.test.itertuples():
            g = z[z.id == r.id]
            t, y = g.cycle.to_numpy(), g.psa.to_numpy()
            found = [(t[j], t[j + 1]) for j in range(1, len(t) - 1)
                     if t[j] > 3 and y[j] <= y[j - 1] and y[j + 1] > y[j]]
            assert found and found[0] == (r.Treb, r.confirm_cycle)
        self.checks.append("S20 history reconstructed from SHA-verified VENICE raw files; query hash and all 47 targets match")
        return z


class Plotter:
    def __init__(self, src, out, bootstrap=2000):
        self.s, self.out, self.bootstrap = src, Path(out), bootstrap
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out / "plot_data").mkdir()
        (self.out / "private").mkdir()
        self.manifest, self.ci_rows = [], []
        self.ci_cache = {}
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                             "axes.titlesize": 10, "axes.labelsize": 9,
                             "axes.spines.top": False, "axes.spines.right": False,
                             "legend.frameon": False, "legend.fontsize": 8,
                             "xtick.labelsize": 8, "ytick.labelsize": 8,
                             "pdf.fonttype": 42, "ps.fonttype": 42,
                             "savefig.dpi": 220, "axes.axisbelow": True})

    def fig(self, rows=2, cols=2, size=(10.4, 7.1), title=None):
        fig, axs = plt.subplots(rows, cols, figsize=size, layout="constrained", squeeze=False)
        for i, ax in enumerate(axs.flat):
            ax.text(-0.13, 1.06, chr(65+i), transform=ax.transAxes, fontsize=12, weight="bold")
            ax.grid(axis="y", color="#E5E8EB", lw=.6)
        if title: fig.suptitle(title, fontsize=12, weight="bold")
        return fig, axs

    def save(self, fig, number, filename, label, description):
        for suffix in ["pdf", "png"]:
            fig.savefig(self.out / f"{filename}.{suffix}", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        self.manifest.append(dict(figure=number, filename=filename+".pdf", latex_label=label, description=description))
        print(f"Rendered {number}: {filename}.pdf", flush=True)

    def ci(self, z, metric_name="MAE", key=""):
        cols = ["id", "Treb", "prob_late8" if metric_name == "AUROC" else "pred"]
        z = z.dropna(subset=cols).sort_values("id")
        cachekey = hashlib.sha256((key + metric_name + z[cols].to_csv(index=False)).encode()).hexdigest()
        if cachekey in self.ci_cache: return self.ci_cache[cachekey]
        rng = np.random.default_rng(int(cachekey[:8], 16))
        point = metric(z, metric_name)
        if metric_name == "MAE":
            errors = abs(z.pred.to_numpy() - z.Treb.to_numpy())
            vals = np.mean(errors[rng.integers(len(z), size=(self.bootstrap, len(z)))], axis=1)
        else:
            y, p = z.Treb.to_numpy() > 8, z.prob_late8.to_numpy()
            vals = []
            for _ in range(self.bootstrap):
                ix = rng.integers(len(z), size=len(z))
                if len(np.unique(y[ix])) == 2: vals.append(roc_auc_score(y[ix], p[ix]))
        lo, hi = np.quantile(vals, [.025, .975])
        self.ci_cache[cachekey] = (point, lo, hi)
        self.ci_rows.append(dict(comparison=key, metric=metric_name, n=len(z), point=point,
                                 lo=lo, hi=hi, bootstrap_replicates=len(vals), scope="fixed-model patient bootstrap"))
        return point, lo, hi

    def curves(self, ax, dataset, models, measure="MAE", stratum="fixed_future6", scans=(2,3,4,5,6), legend=True):
        for model in models:
            xx, yy = [], []
            for n in scans:
                z = self.s.select(dataset, n, model, stratum)
                if z.empty: continue
                v = metric(z, measure)
                if not np.isfinite(v): continue
                xx.append(n); yy.append(v)
            ax.plot(xx, yy, marker="o", ms=4, lw=1.7, color=COLOR[model], label=LABEL[model],
                    ls="--" if model == BASE else "-")
        ax.set_xlabel("PSA measurements (baseline included)")
        ax.set_xticks(list(scans))
        ax.set_ylabel({"MAE": "MAE (cycles)", "Spearman_rho": "Spearman correlation",
                       "AUROC": "AUROC", "balanced_accuracy": "Balanced accuracy",
                       "early_recall": "Early-class recall", "late_recall": "Late-class recall"}[measure])
        if measure in ("AUROC", "balanced_accuracy"):
            ax.axhline(.5, color="#999999", ls=":", lw=1)
            ax.set_ylim(0, 1)
        if measure in ("early_recall", "late_recall"): ax.set_ylim(-.02,1.02)
        if legend: ax.legend()

    def ecdf(self, ax, values, label, color, weights=None, **kwargs):
        x = np.asarray(values, float)
        ok = np.isfinite(x)
        x = x[ok]
        w = np.ones(len(x)) if weights is None else np.asarray(weights, float)[ok]
        ix = np.argsort(x, kind="stable")
        ax.step(x[ix], np.cumsum(w[ix])/w.sum(), where="post", label=label, color=color, **kwargs)

    def roc(self, ax, y, p, label, color, **kwargs):
        y, p = np.asarray(y), np.asarray(p, float)
        ok = np.isfinite(p); y, p = y[ok], p[ok]
        if len(np.unique(y)) < 2: raise RuntimeError("ROC needs both classes")
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, color=color, label=f"{label}: {auc:.3f}", **kwargs)
        ax.plot([0,1],[0,1], color="#AAB0B4", ls=":", zorder=0)
        ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0,1), ylim=(0,1.02))

    def calibration(self, ax, y, p, label, color, bins=4, annotate_counts=False, count_offset=7):
        tab = pd.DataFrame({"y": np.asarray(y, int), "p": np.asarray(p, float)}).dropna()
        # Fixed probability bins; equal scores are never split arbitrarily.
        tab["bin"] = pd.cut(tab.p.clip(0,1), np.linspace(0,1,bins+1), include_lowest=True)
        agg = tab.groupby("bin", observed=True).agg(p=("p","mean"), y=("y","mean"), n=("y","size"))
        ax.plot(agg.p, agg.y, color=color, lw=1.3, label=label)
        ax.scatter(agg.p, agg.y, s=20 + 4*agg.n, color=color, alpha=.8, edgecolors="white", linewidths=.5)
        if annotate_counts:
            for r in agg.itertuples():
                offset=7 if r.y<.10 and count_offset<0 else count_offset
                ax.annotate(f"n={int(r.n)}", (r.p, r.y), xytext=(4,offset),
                            textcoords="offset points", fontsize=7, color=color,
                            ha="left", va="bottom" if offset>0 else "top")
        ax.plot([0,1],[0,1], color="#AAB0B4", ls=":", zorder=0)
        ax.set(xlabel="Mean predicted probability", ylabel="Observed positive fraction", xlim=(-.02,1.02), ylim=(-.03,1.03))

    def comparison_points(self, ax, dataset, models, scans=6, measure="MAE", baseline=None):
        for i, model in enumerate(models):
            z = self.s.select(dataset, scans, model)
            p, lo, hi = self.ci(z, measure, f"{dataset}/{model}/{scans}/fixed_future6")
            ax.plot([lo,hi],[i,i], color=COLOR[model], lw=2)
            ax.plot(p,i,"o",color=COLOR[model],ms=5)
        if baseline:
            value = metric(self.s.select(dataset, scans, baseline), measure)
            ax.axvline(value, ls="--", color=COLOR.get(baseline,"#555555"), label=LABEL.get(baseline,baseline))
        if measure == "AUROC": ax.axvline(.5,color="#999999",ls=":")
        ax.set_yticks(range(len(models)), [LABEL[m] for m in models])
        ax.invert_yaxis()
        ax.set_xlabel("MAE (cycles)" if measure == "MAE" else "AUROC")
        ax.set_title(f"{dataset.upper()}: six measurements, n={len(self.s.fixed_ids[dataset])}")

    def fig8(self):
        fig, a = self.fig(title="VENICE: forecasting the first observed rebound")
        models = [BASE, "ABC_clean_joint", PRIMARY]
        self.curves(a[0,0], "venice", models)
        a[0,0].legend(fontsize=7.5,loc="upper center",bbox_to_anchor=(.5,-.28),ncol=2)
        a[0,0].set_title("Prediction error by measurement count")
        for model in models[1:]:
            self.curves(a[0,1], "venice", [model], "Spearman_rho", legend=False)
        a[0,1].set_title("Prediction correlation by measurement count")
        a[0,1].set_ylim(-.5,1)
        a[0,1].axhline(0,color="#999999",ls=":")
        z = self.s.select("venice",6,PRIMARY)
        ax = a[1,0]
        ax.scatter(z.Treb,z.pred,color=COLOR[PRIMARY],s=30,alpha=.8)
        limit = max(z.Treb.max(),z.pred.max())+1
        ax.plot([3,limit],[3,limit],"--",color="#9AA1A6",lw=1)
        ax.set(xlabel=f"Observed {TREB} (cycles)",ylabel=f"Predicted {TREB} (cycles)",title=f"Observed and ABC-predicted {TREB}")
        ax.text(.04,.94,f"n={len(z)}\nMAE={metric(z,'MAE'):.2f}\nrho={metric(z,'Spearman_rho'):.2f}",
                transform=ax.transAxes,va="top",fontsize=9)
        # Dan's requested paired four-to-six measurement comparison. Each
        # observation is a patient's change in absolute error, not a per-patient MAE.
        z4=self.s.select("venice",4,PRIMARY)[["id","Treb","pred"]]
        pairs=z4.merge(z[["id","Treb","pred"]],on="id",suffixes=("_4","_6"),validate="one_to_one")
        assert len(pairs)==28 and np.array_equal(pairs.Treb_4,pairs.Treb_6)
        delta=(pairs.pred_4-pairs.Treb_4).abs()-(pairs.pred_6-pairs.Treb_6).abs()
        mean_delta=float(delta.mean())
        assert np.isclose(mean_delta,metric(z4.rename(columns={}),"MAE")-metric(z,"MAE"))
        ax=a[1,1]
        bound=max(.5,np.ceil(abs(delta).max()*2)/2)
        bins=np.linspace(-bound,bound,13)
        ax.hist(delta,bins=bins,color=COLOR[PRIMARY],alpha=.8,edgecolor="white")
        ax.axvline(0,color="#777777",ls=":",lw=1.2,label="No change")
        ax.axvline(mean_delta,color="#B27037",ls="--",lw=1.3,label=f"Mean reduction = {mean_delta:.2f}")
        ax.set(xlabel="Absolute-error reduction, 4 to 6 measurements (cycles)",
               ylabel="Patients",title="Change in error with two additional measurements")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True));ax.legend(fontsize=8)
        ax.text(.5,-.23,"Positive values favor six measurements; same 28 patients",transform=ax.transAxes,ha="center",fontsize=8)
        pairs["absolute_error_reduction"]=delta
        pairs.to_csv(self.out/"private/Fig9_four_to_six_pairs_PRIVATE.csv",index=False)
        counts,edges=np.histogram(delta,bins=bins)
        pd.DataFrame({"lower":edges[:-1],"upper":edges[1:],"patients":counts}).to_csv(self.out/"plot_data/Fig9_four_to_six_histogram.csv",index=False)
        json_write(self.out/"plot_data/Fig9_four_to_six_summary.json",dict(n=28,
            mean_absolute_error_reduction=mean_delta,
            improved=int((delta>1e-12).sum()),unchanged=int((abs(delta)<=1e-12).sum()),
            worsened=int((delta< -1e-12).sum()),
            MAE_four=float(abs(pairs.pred_4-pairs.Treb_4).mean()),MAE_six=float(abs(pairs.pred_6-pairs.Treb_6).mean())))
        self.save(fig,"9","E06_calibrated_ABC_scan_budget","fig:clinicalabc","All panels fixed TEST n=28 with complete cycles 0-5 and Treb>5; D paired four-to-six change in absolute error")

    def fig9(self):
        fig,a=self.fig(title=f"Early versus late first rebound: late = {TREB} > 8")
        for ax,dataset in zip(a[0],DATASETS):
            self.curves(ax,dataset,[PRIMARY,"Clinical_PSA"],"AUROC")
            z=self.s.select(dataset,6,PRIMARY)
            ax.set_title(f"{dataset.upper()}: fixed n={len(z)}, late={int((z.Treb>8).sum())}\nComplete cycles 0-5; {TREB} > 5")
        ax=a[1,0]
        for dataset,color in [("venice",COLOR[PRIMARY]),("mainsail","#3269A8")]:
            z=self.s.select(dataset,6,PRIMARY)
            self.roc(ax,z.Treb>8,z.prob_late8,dataset.upper(),color)
            self.calibration(a[1,1],z.Treb>8,z.prob_late8,dataset.upper(),color,
                             annotate_counts=True,count_offset=7 if dataset=="venice" else -10)
        ax.set_title("Six-measurement ABC ROC"); ax.legend()
        a[1,1].set_title("Six-measurement ABC calibration");a[1,1].legend()
        self.save(fig,"10","E07_binary_ABC_early_late","fig:binaryabc","Fixed complete-cycle-0-5, Treb>5 subsets; unified cutoff; calibration labels give patients per nonempty probability bin")

    def s6(self):
        fig,a=self.fig(title="Clinical ridge regression: exact cross-trial transfer")
        for col,(dataset,models,title) in enumerate([
            ("mainsail",[BASE,"Clinical_PSA","MAINSAIL_internal_CV"],"VENICE TRAIN to MAINSAIL"),
            ("venice",["MAINSAIL_constant","MAINSAIL_to_VENICE", "Clinical_PSA"],"MAINSAIL to VENICE TEST")]):
            self.curves(a[0,col],dataset,models,scans=(4,6))
            a[0,col].set_title(f"{title}\nComplete cycles 0-5; {TREB} > 5 (n={len(self.s.fixed_ids[dataset])})")
            handles,labels=a[0,col].get_legend_handles_labels()
            labels=["VENICE-trained ridge" if label==LABEL["Clinical_PSA"] else label for label in labels]
            a[0,col].legend(handles,labels,fontsize=7.5,loc="upper center",bbox_to_anchor=(.5,-.28),ncol=2)
            for model in models:
                z=self.s.select(dataset,6,model)
                name="VENICE-trained ridge" if model=="Clinical_PSA" else LABEL[model]
                self.ecdf(a[1,col],abs(z.pred-z.Treb),name,COLOR[model],lw=1.7)
            a[1,col].set(xlabel="Absolute error (cycles)",ylabel="Cumulative fraction",title="Six-measurement error distribution",ylim=(0,1.02))
            a[1,col].legend(fontsize=7.5)
        self.save(fig,"S6","E03c_exact_cross_trial_transfer","fig:exacttransfer","Fixed future6 n=94/28; source-specific median; reverse and internal CV labelled separately")

    def s7(self):
        fig,a=self.fig(title=f"Cross-trial classification of {TREB} > 8")
        panels=[("mainsail","MAINSAIL_internal_CV","MAINSAIL internal five-fold CV"),
                ("mainsail","Clinical_PSA","VENICE TRAIN to MAINSAIL"),
                ("venice","MAINSAIL_to_VENICE","MAINSAIL to VENICE TEST")]
        for ax,(dataset,model,title) in zip([a[0,0],a[0,1],a[1,0]],panels):
            for n,style in [(4,"--"),(6,"-")]:
                z=self.s.select(dataset,n,model)
                self.roc(ax,z.Treb>8,z.prob_late8,f"{n} measurements",COLOR[model],ls=style)
            ax.set_title(f"{title}\nComplete cycles 0-5; {TREB} > 5 (n={len(z)})");ax.legend()
        ax=a[1,1]
        for dataset,model,title in panels[1:]:
            z=self.s.select(dataset,6,model)
            self.calibration(ax,z.Treb>8,z.prob_late8,title,COLOR[model],
                             annotate_counts=True,count_offset=7 if dataset=="mainsail" else -27)
        ax.set_title("Six-measurement transfer calibration");ax.legend(fontsize=7.5)
        self.save(fig,"S7","E03d_binary_cross_trial_transfer","fig:binarytransfer","Cutoff 8; fixed future6; MAINSAIL internal CV is five folds, not repeated CV")

    def s8(self):
        train=self.s.train.sort_values(["rmse_log10","id"]).reset_index(drop=True)
        long=self.s.read("training/long_PRIVATE.csv")
        fig,a=self.fig(title="Completed VENICE trajectories: TRAIN examples")
        selection=[]
        for ax,q in zip(a.flat,[.1,.4,.7,.9]):
            r=train.iloc[int(q*(len(train)-1))]
            z=long[long.id==r.id].sort_values("cycle")
            t=np.linspace(0,z.cycle.max(),500)
            logp=np.logaddexp(np.log1p(-r.q_hat)-r.decline_rate_hat*t,np.log(r.q_hat)+r.growth_rate_hat*t)
            ax.plot(z.cycle,z.psa/z.psa.iloc[0],"o-",color="#53616F",ms=3,lw=.8,label="Observed PSA")
            ax.plot(t,np.exp(logp),color="#3269A8",lw=1.7,label="Completed-curve fit")
            ax.axvline(r.Treb,color=COLOR[PRIMARY],ls="--",label=TREB)
            if 0<=r.tmin_fit_hat<=z.cycle.max():
                ax.axvline(r.tmin_fit_hat,color="#C56528",ls=":",lw=2,label=TFIT)
            else:
                ax.text(.98,.95,f"Fitted nadir outside window: {r.tmin_fit_hat:.1f}",transform=ax.transAxes,ha="right",va="top",fontsize=8)
            ax.set(yscale="log",xlabel="Treatment cycle",ylabel="PSA / baseline",
                   title=f"Fit-error percentile {int(100*q)}; log10 RMSE={r.rmse_log10:.3f}")
            ax.legend(fontsize=7.5)
            selection.append(dict(figure="S8",id=r.id,selection_quantile=q))
        pd.DataFrame(selection).to_csv(self.out/"private/S8_example_selection_PRIVATE.csv",index=False)
        self.save(fig,"S8","E04a_biexponential_fit_examples","fig:fitexamples","TRAIN-only fit-error quantiles; both observed and fitted endpoints marked")

    def s9(self):
        fig,a=self.fig(title="Completed-curve fit and first-rebound timing are distinct")
        rows=[]
        for ax,name,col in zip(a[0],["TRAIN","TEST"],["#647A89","#007F86"]):
            g=self.s.parent[self.s.parent.split==name]
            inside=boolean(g.fit_tmin_in_observed_window)
            ax.scatter(g.loc[inside,"Treb"],g.loc[inside,"tmin_fit_hat"],s=22,color=col,alpha=.65,label=f"In window (n={inside.sum()})")
            ax.scatter(g.loc[~inside,"Treb"],g.loc[~inside,"tmin_fit_hat"],s=50,color="#BF573F",marker="x",lw=1.6,label=f"Outside window (n={(~inside).sum()})")
            high=max(g.Treb.max(),g.tmin_fit_hat.max())+2
            diagonal=np.linspace(0,high,400)
            ax.plot(diagonal,diagonal,color="#8B9298",ls=":",lw=1)
            ax.set_yscale("symlog",linthresh=20,linscale=1.3)
            ax.set(xlabel=f"Observed {TREB} (cycles)",ylabel=f"Full-curve {TFIT} (cycles)",title=f"{name}: all {len(g)} retained patients")
            ax.legend(fontsize=7.5)
            for scope,sub,style in [("All",g,"-"),("In window",g[inside],"--")]:
                err=abs(sub.tmin_fit_hat-sub.Treb)
                self.ecdf(a[1,0],err,f"{name}, {scope.lower()} (n={len(sub)})",col,ls=style,lw=1.6)
                rows.append(dict(split=name,scope=scope,n=len(sub),MAE=err.mean(),within1=(err<=1).mean(),within2=(err<=2).mean()))
            self.ecdf(a[1,1],g.rmse_log10,name,col,lw=1.7)
        a[1,0].set_xscale("symlog",linthresh=5)
        a[1,0].set(xlabel="Absolute nadir discrepancy (cycles; nonlinear axis)",ylabel="Cumulative fraction",ylim=(0,1.02),title="Full ranges retained, including extreme fitted nadirs")
        a[1,0].legend(fontsize=7.2,loc="lower right")
        a[1,1].set(xlabel="Full-curve RMSE in log10-relative PSA",ylabel="Cumulative fraction",ylim=(0,1.02),title="Overall trajectory-fit error")
        a[1,1].legend()
        pd.DataFrame(rows).to_csv(self.out/"plot_data/S9_fit_agreement.csv",index=False)
        self.save(fig,"S9","E04b_biexponential_nadir_diagnostics","fig:fitdiagnostics","All 231 retained; separate in-window sensitivity; nonlinear axes expose negative extremes")

    def s10(self):
        tab=self.s.read("training/parameter_information_PRIVATE.csv")
        fig,a=self.fig(1,2,size=(10.4,4.3),title="Early measurements constrain decline more than the growing component")
        specs=[("log_d",r"$\log d$","#007F86"),("logit_q",r"$\mathrm{logit}(q)$","#C56528"),("log_g",r"$\log g$","#7B5BA7")]
        rows=[]
        for ax,n in zip(a.flat,[4,6]):
            for i,(key,label,color) in enumerate(specs):
                z=tab[(tab.scan_count==n)&(tab.parameter==key)]
                vals=z.width_ratio.to_numpy()
                lo,med,hi=np.quantile(vals,[.25,.5,.75])
                ax.plot([i,i],[lo,hi],color=color,lw=5,solid_capstyle="round")
                ax.plot(i,med,"o",color="white",mec=color,mew=2,ms=7)
                ax.text(i,med+.07,f"{med:.3f}",ha="center",fontsize=9)
                rows.append(dict(scans=n,parameter=key,n=len(z),q25=lo,median=med,q75=hi))
            ax.axhline(1,color="#777777",ls=":",label="Prior width")
            ax.set_xticks(range(3),[v[1] for v in specs])
            ax.set(ylim=(0,1.18),ylabel="Posterior / prior 95% interval width",title=f"{n} measurements; TRAIN leave-one-patient-out (n={len(z)})")
            ax.text(.5,-.17,"Points: patient median; bars: interquartile range",transform=ax.transAxes,ha="center",fontsize=8)
        pd.DataFrame(rows).to_csv(self.out/"plot_data/S10_parameter_widths.csv",index=False)
        self.save(fig,"S10","E05a_early_parameter_information","fig:parameterinformation","Replaces old five-compartment associations; uncertainty widths are model/prior-dependent")

    def s11(self):
        s=self.s.read("simulation/full_fit_parameters.csv")
        s=s[boolean(s.fit_success)].sort_values("id",key=lambda z:z.astype(int)).reset_index(drop=True)
        t=self.s.train
        C=np.c_[t.decline_rate_hat,np.log10(t.q_hat),t.growth_rate_hat]
        S=np.c_[s.decline_rate_hat,np.log10(s.q_hat),s.growth_rate_hat]
        center=np.median(C,axis=0)
        scale=np.maximum(np.quantile(C,.75,axis=0)-np.quantile(C,.25,axis=0),1e-6)
        ds,ix=cKDTree((S-center)/scale).query((C-center)/scale,k=min(25,len(S)))
        w=np.zeros(len(S))
        for di,ii in zip(ds,ix):
            wi=1/np.maximum(di,1e-6); wi/=wi.sum(); np.add.at(w,ii,wi/len(C))
        w=.9*w+.1/len(S);w/=w.sum()
        ess=1/np.sum(w*w)
        assert np.isclose(ess,self.s.meta["weighted_effective_n"],rtol=1e-7)
        self.s.checks.append("Reconstructed TRAIN-guided weights reproduce saved effective sample size")
        fig,a=self.fig(title="Clinical and simulated biexponential parameter distributions")
        for ax,col,lab in zip([a[0,0],a[0,1],a[1,0]],
            ["decline_rate_hat","q_hat","growth_rate_hat"],[r"$\log_{10} d$",r"$\log_{10} q$",r"$\log_{10} g$"]):
            self.ecdf(ax,np.log10(s[col]),f"Unweighted simulations (n={len(s)})","#ACB3B8",lw=1.6)
            self.ecdf(ax,np.log10(s[col]),"TRAIN-weighted simulations",COLOR[PRIMARY],weights=w,lw=1.8)
            self.ecdf(ax,np.log10(t[col]),"VENICE TRAIN (n=184)","#3269A8",lw=1.5)
            self.ecdf(ax,np.log10(self.s.test[col]),"VENICE TEST (n=47)","#C56528",ls="--",lw=1.4)
            ax.set(xlabel=lab,ylabel="Cumulative fraction",ylim=(0,1.02))
        a[0,0].legend(fontsize=7.1)
        ax=a[1,1];ordered=np.sort(w)[::-1]
        ax.plot(np.arange(1,len(w)+1)/len(w),np.cumsum(ordered),color=COLOR[PRIMARY],lw=1.8)
        ax.plot([0,1],[0,1],color="#ACB3B8",ls=":",label="Uniform weights")
        ax.set(xlabel="Fraction of references (largest weights first)",ylabel="Cumulative reference weight",xlim=(0,1),ylim=(0,1.02),title="Weight concentration before endpoint filtering")
        ax.text(.98,.18,f"Effective reference count = {ess:.1f}\n{self.s.meta['clean_targets']} clean / {self.s.meta['noisy_targets']} perturbed targets",transform=ax.transAxes,ha="right",fontsize=8)
        ax.legend(loc="lower right")
        pd.DataFrame({"simulation_id":s.id,"weight":w}).to_csv(self.out/"plot_data/S11_reference_weights.csv",index=False)
        self.save(fig,"S11","E05b_training_guided_reference","fig:cloudsupport","Marginal shape coverage; TRAIN-only IQR weights; no historical support categories or new patient exclusion")

    def s12(self):
        fig,a=self.fig(size=(10.8,8.0),title="Predictor comparison using the same first-rebound target")
        pairs=[("ABC_GP",PRIMARY,"GP correction vs ABC"),
               ("XGBoost_absolute",PRIMARY,"XGBoost absolute vs ABC"),
               ("XGBoost_absolute","XGBoost_squared","Absolute vs squared loss")]
        ci=self.s.read("results/PAIRED_MAE_REDUCTION.csv")
        for col,dataset in enumerate(DATASETS):
            self.curves(a[0,col],dataset,[BASE,PRIMARY,"ABC_GP","XGBoost_absolute","XGBoost_squared"])
            a[0,col].set_title(f"{dataset.upper()}: fixed n={len(self.s.fixed_ids[dataset])}\nComplete cycles 0-5; {TREB} > 5")
            a[0,col].legend(fontsize=7.1,loc="upper center",bbox_to_anchor=(.5,-.28),ncol=2)
            ax=a[1,col]
            for i,(model,base,label) in enumerate(pairs):
                r=ci[(ci.dataset==dataset)&(ci.scan_count==6)&(ci.model==model)&(ci.comparator==base)].iloc[0]
                ax.plot([r.lo,r.hi],[i,i],color=COLOR[model],lw=2)
                ax.plot(r.MAE_reduction,i,"o",color=COLOR[model])
            ax.axvline(0,color="#888888",ls=":")
            ax.set_yticks(range(len(pairs)),[v[2] for v in pairs],fontsize=7.7)
            ax.invert_yaxis()
            ax.set(xlabel="Paired MAE reduction (cycles)",title="Six measurements: 95% patient-bootstrap intervals")
        self.save(fig,"S12","E06b_predictor_transfer_control","fig:xgb","No old XGBoost reference-CV panel; fixed future6; saved paired comparisons")

    def s13(self):
        tab=self.s.read("results/REFERENCE_SIZE.csv")
        fig,a=self.fig(title="Reference-library size sensitivity at fixed matching settings")
        for col,dataset in enumerate(DATASETS):
            z=tab[tab.dataset==dataset]
            for ax,key,ylab in [(a[0,col],"MAE","MAE (cycles)"),(a[1,col],"Spearman_rho","Spearman correlation")]:
                ag=z.groupby("reference_n")[key].agg(mean="mean",lo=lambda v:v.quantile(.025),hi=lambda v:v.quantile(.975),runs="size").reset_index()
                ax.plot(ag.reference_n,ag["mean"],"o-",color=COLOR[PRIMARY],lw=1.6)
                sub=ag[ag.runs>1]
                ax.fill_between(sub.reference_n,sub.lo,sub.hi,color=COLOR[PRIMARY],alpha=.15,label="Central 95% across reference subsets")
                ax.set(xlabel="Reference simulations",ylabel=ylab,title=f"{dataset.upper()}: six measurements, n={len(self.s.fixed_ids[dataset])}\nComplete cycles 0-5; {TREB} > 5")
                ax.set_xticks(ag.reference_n)
                if key=="MAE":
                    ax.axhline(metric(self.s.select(dataset,6,BASE),"MAE"),color=COLOR[BASE],ls="--",label="TRAIN median")
                    ax.legend(fontsize=7.2)
        self.save(fig,"S13","E06c_reference_library_size","fig:refsize","Twenty subsamples per reduced size; full bank one run; ranges describe reference sampling, not patient uncertainty")

    def s14(self):
        fig,a=self.fig(title=f"Weighted and rejection ABC: complete cycles 0-5; {TREB} > 5")
        keys=["AUROC","balanced_accuracy","early_recall","late_recall"]
        for ax,key in zip(a.flat,keys):
            for dataset,color in [("venice","#007F86"),("mainsail","#3269A8")]:
                for model,ls in [(PRIMARY,"-"),("Rejection_ABC","--")]:
                    vals=[metric(self.s.select(dataset,n,model),key) for n in range(2,7)]
                    name=f"{dataset.upper()} (n={len(self.s.fixed_ids[dataset])}), {'weighted' if model==PRIMARY else 'rejection'}"
                    ax.plot(range(2,7),vals,marker="o",color=color,ls=ls,ms=3.5,label=name)
            ax.set(xlabel="PSA measurements (baseline included)",ylabel={"AUROC":"AUROC","balanced_accuracy":"Balanced accuracy","early_recall":"Early-class recall","late_recall":"Late-class recall"}[key],ylim=(-.02,1.02))
            ax.set_xticks(range(2,7))
            if key in ["AUROC","balanced_accuracy"]:ax.axhline(.5,color="#999999",ls=":")
        a[0,0].legend(fontsize=7.3)
        self.save(fig,"S14","E07b_ABC_weighting_control","fig:rejection","Same 28/94 patients at every budget; separate reference-selected settings and thresholds")

    def s15(self):
        fig,a=self.fig(2,3,size=(12.2,7.1),title="Empirical clinical priors: prediction of observed first rebound")
        rows=[]
        for row,dataset in enumerate(DATASETS):
            self.curves(a[row,0],dataset,[BASE,PRIMARY,"EB_Treb_joint"])
            a[row,0].set_title(f"{dataset.upper()}: fixed n={len(self.s.fixed_ids[dataset])}\nComplete cycles 0-5; {TREB} > 5")
            cov,width=[],[]
            for n in range(2,7):
                z=self.s.select(dataset,n,"EB_Treb_joint")
                valid=z[["lo","hi"]].notna().all(axis=1)
                zz=z[valid]
                c=np.mean((zz.Treb>=zz.lo)&(zz.Treb<=zz.hi))
                w=np.median(zz.hi-zz.lo)
                cov.append(c);width.append(w)
                rows.append(dict(dataset=dataset,scans=n,n=len(z),interval_n=len(zz),coverage=c,median_width=w,
                    median_undefined_mass=z.undefined_mass.median(),max_undefined_mass=z.undefined_mass.max(),
                    undefined_mass_over_01=int((z.undefined_mass>.1).sum())))
            a[row,1].plot(range(2,7),cov,"o-",color=COLOR["EB_Treb_joint"])
            a[row,1].axhline(.95,color="#888888",ls=":",label="Nominal 95%")
            a[row,1].set(xlabel="PSA measurements",ylabel="Empirical predictive interval coverage",ylim=(0,1.02),title="Conditional 95% predictive intervals")
            a[row,1].set_xticks(range(2,7));a[row,1].legend()
            a[row,2].plot(range(2,7),width,"o-",color=COLOR["EB_Treb_joint"])
            a[row,2].set(xlabel="PSA measurements",ylabel="Median interval width (cycles)",title="Rebound by cycle 80 assumed in intervals")
            a[row,2].set_xticks(range(2,7))
            last=rows[-1]
            a[row,2].text(.04,.06,f"At six measurements:\nundefined mass > 0.1: {last['undefined_mass_over_01']}/{last['n']}\nmaximum undefined mass: {last['max_undefined_mass']:.3f}",
                          transform=a[row,2].transAxes,ha="left",va="bottom",fontsize=8)
        pd.DataFrame(rows).to_csv(self.out/"plot_data/S15_predictive_intervals.csv",index=False)
        self.save(fig,"S15","E08_empirical_Bayes_forecasting","fig:eb","Treb predictive intervals conditional on evaluable rebound by 80; undefined probability reported; no latent-Tfit relabelling")

    def s16(self):
        fig,a=self.fig(title="Sensitivity to the empirical q-g association")
        ci=self.s.read("results/PAIRED_MAE_REDUCTION.csv")
        for col,dataset in enumerate(DATASETS):
            self.curves(a[0,col],dataset,["EB_Treb_joint","EB_Treb_qg_shuffled"])
            a[0,col].set_title(f"{dataset.upper()}: fixed n={len(self.s.fixed_ids[dataset])}\nComplete cycles 0-5; {TREB} > 5")
            z=ci[(ci.dataset==dataset)&(ci.model=="EB_Treb_joint")&(ci.comparator=="EB_Treb_qg_shuffled")].sort_values("scan_count")
            ax=a[1,col]
            for i,r in enumerate(z.itertuples()):
                ax.plot([r.lo,r.hi],[i,i],color=COLOR["EB_Treb_joint"],lw=2)
                ax.plot(r.MAE_reduction,i,"o",color=COLOR["EB_Treb_joint"])
            ax.axvline(0,color="#888888",ls=":")
            ax.set_yticks(range(len(z)),[f"{int(r.scan_count)} measurements (n={int(r.n)})" for r in z.itertuples()])
            ax.set(xlabel="MAE reduction from preserving joint prior (cycles)",title="Future-target subsets; 95% patient-bootstrap intervals")
        self.save(fig,"S16","E08b_joint_qg_prior_ablation","fig:qg","g shuffled within d-rank bins; neutral title; target is Treb in both variants")

    def s17(self):
        models=["Covariates_PSA","Covariates_baseline","Covariates_PSA_baseline"]
        fig,a=self.fig(title=f"Clinical information: VENICE TEST, complete cycles 0-5; {TREB} > 5")
        self.comparison_points(a[0,0],"venice",models,measure="AUROC")
        a[0,0].set_title("AUROC with 95% patient-bootstrap intervals")
        a[0,0].set_xlim(0,1)
        for ax,key,ylab in [(a[0,1],"balanced_accuracy","Balanced accuracy"),(a[1,0],"Brier","Brier score")]:
            vals=[metric(self.s.select("venice",6,m),key) for m in models]
            ax.bar(range(3),vals,color=[COLOR[m] for m in models],width=.65)
            for i,v in enumerate(vals):ax.text(i,v+.02,f"{v:.3f}",ha="center",fontsize=9)
            ax.set_xticks(range(3),["Early PSA","Baseline\nvariables","PSA +\nbaseline"])
            ax.set(ylabel=ylab,ylim=(0,1 if key=="balanced_accuracy" else max(.4,max(vals)+.09)))
            if key=="balanced_accuracy":ax.axhline(.5,color="#888888",ls=":")
        ax=a[1,1]
        for model in models:
            z=self.s.select("venice",6,model)
            self.roc(ax,z.Treb>8,z.prob_late8,LABEL[model],COLOR[model])
        ax.set_title(f"n={len(z)} TEST patients; {(z.Treb>8).sum()} late events");ax.legend(fontsize=7.5)
        self.save(fig,"S17","E09a_clinical_covariate_test","fig:covariates","All preprocessing/tuning uses TRAIN; baseline includes absolute PSA; no broad claim that all labs are uninformative")

    def s18(self):
        tab=self.s.read("results/LEARNING_CURVE.csv")
        tab=tab[tab.stratum=="future_Treb"]
        g=tab.groupby("n_training").AUROC.agg(mean="mean",lo=lambda x:x.quantile(.025),hi=lambda x:x.quantile(.975),runs="size").reset_index()
        fig,a=self.fig(1,1,size=(6.5,4.6),title="Clinical training size: fixed VENICE TEST n=28")
        ax=a[0,0]
        for text in list(ax.texts): text.remove()
        ax.plot(g.n_training,g["mean"],"o-",color=COLOR["Clinical_PSA"])
        reduced=g[g.runs>1]
        ax.fill_between(reduced.n_training,reduced.lo,reduced.hi,color=COLOR["Clinical_PSA"],alpha=.18,label="Central 95% across training subsets")
        ax.axhline(.5,color="#888888",ls=":")
        ax.set(xlabel="Training patients",ylabel="AUROC",ylim=(0,1),title=f"Six-measurement PSA; complete cycles 0-5; {TREB} > 5")
        ax.set_xticks(g.n_training);ax.legend(fontsize=7.3)
        g.to_csv(self.out/"plot_data/S18_training_size_summary.csv",index=False)
        self.save(fig,"S18","E09b_clinical_training_size","fig:trainsize","Future-target TEST n=28 only; 20 reduced-training repetitions; 182 complete TRAIN patients at full size")

    def s19(self):
        fig=plt.figure(figsize=(12.4,7.3),layout="constrained")
        grid=fig.add_gridspec(2,3,width_ratios=(1,1,1.12))
        roc_axes=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1])]
        paired_ax=fig.add_subplot(grid[:,2])
        cal_axes=[fig.add_subplot(grid[1,0]),fig.add_subplot(grid[1,1])]
        for letter,ax in zip("ABCDE",[*roc_axes,paired_ax,*cal_axes]):
            ax.text(-.13,1.06,letter,transform=ax.transAxes,fontsize=12,weight="bold")
            ax.grid(axis="y",color="#E5E8EB",lw=.6)
        fig.suptitle("Sequential detection of the immediate first-rebound window",fontsize=12,weight="bold")
        seq=self.s.seq
        ci=self.s.read("results/SEQUENTIAL_PAIRED_AUC.csv")
        colors={"state":"#007F86","time":"#777C84"}
        for col,dataset in enumerate(DATASETS):
            z=seq[(seq.dataset==dataset)&seq.before_confirmation]
            for model in ["state","time"]:
                self.roc(roc_axes[col],z.window,z["p_window_"+model],"State" if model=="state" else "Elapsed time",colors[model])
                self.calibration(cal_axes[col],z.window,z["p_window_"+model],"State" if model=="state" else "Elapsed time",colors[model])
            roc_axes[col].set_title(f"{dataset.upper()}: before confirmation\n{z.id.nunique()} patients; {len(z)} updates")
            roc_axes[col].legend();cal_axes[col].legend()
            cal_axes[col].set_title(f"{dataset.upper()}: calibration\nbefore confirmation")
        ax=paired_ax;labels=[];rows=[]
        for i,(dataset,scope) in enumerate([(d,s) for d in DATASETS for s in ["before_confirmation","confirmation_inclusive"]]):
            z=seq[seq.dataset==dataset]
            if scope=="before_confirmation":z=z[z.before_confirmation]
            delta=roc_auc_score(z.window,z.p_window_state)-roc_auc_score(z.window,z.p_window_time)
            r=ci[(ci.dataset==dataset)&(ci.scope==scope)].iloc[0]
            # Saved 'mean' is a bootstrap mean, NOT the observed point difference.
            color="#007F86" if dataset=="venice" else "#3269A8"
            ax.plot([r.lo,r.hi],[i,i],color=color,lw=2)
            ax.plot(delta,i,"o",color=color,mfc=color if scope=="before_confirmation" else "white")
            labels.append(f"{dataset.upper()}\n{'Before confirmation' if scope=='before_confirmation' else 'Including confirmation'}")
            rows.append(dict(dataset=dataset,scope=scope,observed_AUROC_difference=delta,lo=r.lo,hi=r.hi,patients=z.id.nunique(),updates=len(z)))
        ax.axvline(0,color="#888888",ls=":")
        ax.set_yticks(range(4),labels,fontsize=7.5);ax.invert_yaxis()
        ax.set(xlabel="AUROC: state minus elapsed time",title="Paired differences\n95% patient-cluster bootstrap intervals")
        pd.DataFrame(rows).to_csv(self.out/"plot_data/S19_sequential_paired_AUROC.csv",index=False)
        self.save(fig,"S19","E11a_local_nadir_window","fig:window","A-B preconfirmation ROC; C paired observed AUROC differences and patient-cluster intervals; D-E preconfirmation calibration; aligned changing-membership means omitted")

    def s20(self,history):
        target=self.s.test.sort_values(["event_index","id"]).reset_index(drop=True)
        fig,a=self.fig(title="VENICE TEST: observed trajectories and sequential window probabilities")
        chosen=[]
        for col,q in enumerate([.25,.75]):
            r=target.iloc[int(q*(len(target)-1))]
            g=history[history.id==r.id].sort_values("cycle").reset_index(drop=True)
            z=self.s.seq[(self.s.seq.dataset=="venice")&(self.s.seq.id==r.id)].sort_values("risk_index")
            assert not z.empty
            j=int(r.event_index)
            assert g.cycle.iloc[j]==r.Treb and g.cycle.iloc[j+1]==r.confirm_cycle
            lower,upper=g.cycle.iloc[j-1],g.cycle.iloc[j+1]
            for ax in a[:,col]:
                ax.axvspan(lower,upper,color="#BFE1DC",alpha=.45,zorder=0)
                ax.axvline(r.Treb,color=COLOR[PRIMARY],ls="--",lw=1)
                ax.axvline(r.confirm_cycle,color="#B27037",ls=":",lw=1)
                ax.set_xlim(-.5,g.cycle.max()+.5)
            ax=a[0,col]
            ax.plot(g.cycle,g.psa/g.psa.iloc[0],"o-",color="#55616E",ms=3,lw=1,label="Recorded PSA")
            ax.scatter([r.Treb],[g.psa.iloc[j]/g.psa.iloc[0]],marker="v",s=60,color=COLOR[PRIMARY],zorder=5,label=TREB)
            ax.scatter([r.confirm_cycle],[g.psa.iloc[j+1]/g.psa.iloc[0]],s=45,facecolor="white",edgecolor="#B27037",zorder=5,label="Confirming rise")
            ax.set(yscale="log",xlabel="Treatment cycle",ylabel="PSA / baseline",title=f"Event-index percentile {int(q*100)}; {TREB} = {r.Treb:g}")
            ax.legend(fontsize=7.5)
            ax=a[1,col]
            for model,color,label in [("state","#007F86","State model"),("time","#777C84","Elapsed time")]:
                ax.plot(z.risk_cycle,z["p_window_"+model],color=color,lw=1.5,label=label)
                pre=z[z.before_confirmation];confirm=z[~z.before_confirmation]
                ax.scatter(pre.risk_cycle,pre["p_window_"+model],color=color,s=22,zorder=3)
                ax.scatter(confirm.risk_cycle,confirm["p_window_"+model],facecolor="white",edgecolor=color,s=36,zorder=4)
            ax.set(xlabel="Treatment cycle",ylabel="Probability current scan is\nin the nadir window",ylim=(0,1.02),title="Nadir-window probability from available PSA")
            ax.legend(fontsize=7.5)
            chosen.append(dict(figure="S20",id=r.id,event_index_quantile=q,Treb=r.Treb,confirm_cycle=r.confirm_cycle))
        pd.DataFrame(chosen).to_csv(self.out/"private/S20_example_selection_PRIVATE.csv",index=False)
        self.save(fig,"S20","E11b_heldout_window_examples","fig:windowexamples","TEST event-index quantiles selected without model scores; one confirming rise; confirmation probabilities hollow")

    def finish(self):
        pd.DataFrame(self.manifest).to_csv(self.out/"FIGURE_MAP.csv",index=False)
        pd.DataFrame(self.ci_rows).to_csv(self.out/"plot_data/ADDITIONAL_DISPLAY_INTERVALS.csv",index=False)
        keep=self.s.scores[self.s.scores.stratum.isin(["future_Treb","fixed_future6"])]
        keep.to_csv(self.out/"plot_data/FROZEN_METRICS.csv",index=False)
        outputs={p.name:digest(p) for p in sorted(self.out.glob("*.pdf"))}
        json_write(self.out/"PLOT_MANIFEST.json",dict(
            status="SUCCESS",version=VERSION,source=str(self.s.root),
            input_sha256=self.s.inputs,figure_sha256=outputs,checks=self.s.checks,
            figures=self.manifest,bootstrap_replicates=self.bootstrap,
            additional_interval_scope="Display-only patient bootstrap; fixed models and split; not adjusted for multiple comparisons.",
            uncertainty_note="Reference/training-size bands are variation across sampled libraries/training subsets, not patient confidence intervals.",
            software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__),
            source_code_sha256=digest(Path(__file__))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",required=True,type=Path,help="Completed VENICE_TREB_20261028_B directory, not the scripts directory")
    parser.add_argument("--out",required=True,type=Path,help="NEW output directory; existing directories are refused")
    parser.add_argument("--bootstrap",type=int,default=2000,help="Display-only patient bootstrap replicates; default 2000")
    parser.add_argument("--venice-adsl",type=Path,help="Optional original raw CSV, only needed if S20 query file was omitted")
    parser.add_argument("--venice-adlb",type=Path,help="Optional original raw CSV paired with --venice-adsl")
    parser.add_argument("--figures","--only",dest="figures",default="9,10,S6,S7,S11,S12,S13,S14,S15,S16,S17,S18,S19,S20",help="Comma-separated current-manuscript figure numbers; main figures are 9 and 10")
    args=parser.parse_args()
    if "login" in socket.gethostname().lower() and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Submit with sbatch/srun on a compute node; do not run on a Longleaf login node.")
    if args.bootstrap < 200: raise ValueError("Use at least 200 bootstrap draws; production default is 2000")
    if args.out.exists(): raise FileExistsError(f"Choose a new output directory: {args.out}")
    requested=args.figures.split(",")
    allowed=["9","10"]+[f"S{k}" for k in range(6,21)]
    if len(requested)!=len(set(requested)) or not set(requested)<=set(allowed):raise ValueError("Unknown or duplicate figure")
    src=Source(args.results)
    history=src.test_history(args.venice_adsl,args.venice_adlb) if "S20" in requested else None
    # Preflight reads complete before any output directory is created.
    plot=Plotter(src,args.out,args.bootstrap)
    for number in requested:
        if number=="S20":plot.s20(history)
        else:getattr(plot, {"9":"fig8","10":"fig9"}.get(number,number.lower()))()
    plot.finish()
    print(f"SUCCESS: {len(requested)} figures in {args.out.resolve()}",flush=True)


if __name__=="__main__":
    main()
