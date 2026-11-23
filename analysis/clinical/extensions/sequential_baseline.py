"""Compare saved sequential forecasts with TRAIN-only medians on identical updates.

The operational median conditions on the absence of a *confirmed* rebound at the
current nominal cycle. Retrospective T_reb > current cycle is used for scoring only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ("empirical_Bayes", "TRAIN_median", "TRAIN_unconfirmed_median")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predict_baselines(train, prefixes):
    """Use training event times and prefix cycle, never evaluation event times."""
    required = {"id", "Treb", "confirm_cycle"}
    if not required.issubset(train.columns):
        raise ValueError("TRAIN fit table must include id, Treb, and confirm_cycle")
    if train.id.duplicated().any() or not np.isfinite(train[["Treb", "confirm_cycle"]]).all().all():
        raise ValueError("TRAIN fit rows must have unique IDs and finite event times")
    if (train.confirm_cycle <= train.Treb).any():
        raise ValueError("Each TRAIN confirmation must follow the low")
    answer = prefixes[["dataset", "id", "risk_index", "risk_cycle", "pred"]].copy()
    answer = answer.rename(columns={"pred": "empirical_Bayes"})
    answer["TRAIN_median"] = float(train.Treb.median())
    answer["TRAIN_unconfirmed_median"] = np.nan
    answer["n_training_unconfirmed"] = 0
    for cycle in answer.risk_cycle.unique():
        # A low at or before this cycle can still be unconfirmed in the record.
        # The criterion uses the confirmation cycle, never future low status.
        eligible = train.loc[train.confirm_cycle > cycle, "Treb"]
        selection = answer.risk_cycle.eq(cycle)
        answer.loc[selection, "n_training_unconfirmed"] = len(eligible)
        if len(eligible):
            answer.loc[selection, "TRAIN_unconfirmed_median"] = float(eligible.median())
    return answer


def _cluster_metrics(frame, error_columns, bootstrap, rng):
    """One shared patient draw supplies every model and both averaging schemes."""
    errors = frame[["id", *error_columns]].groupby("id", sort=True)
    sums = errors[error_columns].sum().to_numpy(float)
    counts = errors.size().to_numpy(float)
    patient_means = sums / counts[:, None]
    indices = rng.integers(len(counts), size=(bootstrap, len(counts)))
    estimates = {
        "pooled_updates": sums.sum(axis=0) / counts.sum(),
        "equal_patients": patient_means.mean(axis=0),
    }
    samples = {
        "pooled_updates": sums[indices].sum(axis=1) / counts[indices].sum(axis=1)[:, None],
        "equal_patients": patient_means[indices].mean(axis=1),
    }
    return estimates, samples


def score_predictions(predictions, bootstrap=2000, seed=20261104):
    if bootstrap < 2:
        raise ValueError("Use at least two patient-bootstrap draws")
    rng = np.random.default_rng(seed)
    # Event times enter only here, after the baselines have been generated.
    future = predictions.loc[predictions.risk_cycle < predictions.Treb].copy()
    summary, paired = [], []
    for dataset, cohort in future.groupby("dataset", sort=True):
        for model in MODELS:
            available = cohort.loc[np.isfinite(cohort[model])].copy()
            if available.empty:
                continue
            available["error"] = abs(available[model] - available.Treb)
            estimates, samples = _cluster_metrics(available, ["error"], bootstrap, rng)
            for average in estimates:
                interval = np.quantile(samples[average][:, 0], [.025, .975])
                summary.append(dict(
                    dataset=dataset, model=model, averaging=average,
                    n_patients=available.id.nunique(), n_updates=len(available),
                    n_eligible_patients=cohort.id.nunique(), n_eligible_updates=len(cohort),
                    n_missing_predictions=len(cohort)-len(available),
                    MAE=float(estimates[average][0]), lo=float(interval[0]), hi=float(interval[1]),
                ))
        for comparator in MODELS[1:]:
            available = cohort.loc[
                np.isfinite(cohort.empirical_Bayes) & np.isfinite(cohort[comparator])
            ].copy()
            if available.empty:
                continue
            available["eb_error"] = abs(available.empirical_Bayes-available.Treb)
            available["baseline_error"] = abs(available[comparator]-available.Treb)
            estimates, samples = _cluster_metrics(
                available, ["eb_error", "baseline_error"], bootstrap, rng
            )
            for average in estimates:
                estimate = estimates[average]
                gain = samples[average][:, 1]-samples[average][:, 0]
                interval = np.quantile(gain, [.025, .975])
                paired.append(dict(
                    dataset=dataset, comparator=comparator, averaging=average,
                    n_patients=available.id.nunique(), n_updates=len(available),
                    n_eligible_updates=len(cohort), n_missing_pairs=len(cohort)-len(available),
                    EB_MAE=float(estimate[0]), baseline_MAE=float(estimate[1]),
                    MAE_reduction=float(estimate[1]-estimate[0]),
                    lo=float(interval[0]), hi=float(interval[1]),
                ))
    return future, pd.DataFrame(summary), pd.DataFrame(paired)


def run(source: Path, dest: Path, bootstrap: int = 2000):
    source, dest = Path(source), Path(dest)
    paths = {
        "training": source/"training/fits_PRIVATE.csv",
        "training_history": source/"training/long_PRIVATE.csv",
        "predictions": source/"predictions/sequential_PRIVATE.csv",
        "venice": source/"evaluation/heldout_reference_PRIVATE.csv",
        "mainsail": source/"evaluation/mainsail_reference_PRIVATE.csv",
    }
    tables = {key: pd.read_csv(path, dtype={"id": str}) for key, path in paths.items()}
    train, sequence = tables["training"], tables["predictions"]
    if len(train) != 184 or len(tables["venice"]) != 47 or len(tables["mainsail"]) != 181:
        raise ValueError("Unexpected frozen clinical cohort sizes")
    if set(train.id) & set(tables["venice"].id):
        raise ValueError("VENICE training and test identities overlap")
    # All training members had baseline plus cycles 1--3; this anchors the
    # baseline comparison at the same fourth-measurement starting point.
    history = tables["training_history"]
    early = history[history.cycle.isin([0, 1, 2, 3])]
    if not early.groupby("id").cycle.nunique().reindex(train.id).eq(4).all():
        raise ValueError("TRAIN records lack the required four starting measurements")
    if sequence.duplicated(["dataset", "id", "risk_index"]).any():
        raise ValueError("Duplicate saved sequential update")
    predictions = predict_baselines(train, sequence)
    targets = pd.concat([
        tables["venice"][["id", "Treb"]].assign(dataset="venice"),
        tables["mainsail"][["id", "Treb"]].assign(dataset="mainsail"),
    ], ignore_index=True)
    predictions = predictions.merge(targets, on=["dataset", "id"], validate="many_to_one")
    if len(predictions) != len(sequence) or predictions.Treb.isna().any():
        raise ValueError("A sequential prediction lacks a clinical endpoint")
    future, summary, paired = score_predictions(predictions, bootstrap)
    for group in ["results", "predictions"]:
        (dest/group).mkdir(parents=True, exist_ok=True)
    predictions.to_csv(dest/"predictions/sequential_baselines_PRIVATE.csv", index=False)
    summary.to_csv(dest/"results/SEQUENTIAL_BASELINE_METRICS.csv", index=False)
    paired.to_csv(dest/"results/SEQUENTIAL_BASELINE_PAIRED.csv", index=False)
    method = {
        "source_sha256": {str(path.relative_to(source)): _digest(path) for path in paths.values()},
        "target": "First observed local PSA low after cycle 3 followed by one strict rise (Treb).",
        "training_n": len(train),
        "constant": "Median Treb among all 184 VENICE TRAIN patients, fixed at every visit.",
        "unconfirmed_risk_set": "Median TRAIN Treb among records whose confirm_cycle exceeds the query risk_cycle.",
        "empty_risk_set": "Return a missing forecast and report its coverage; no fallback.",
        "prediction_inputs": "TRAIN endpoints/confirmation times and the current nominal cycle. Evaluation endpoints are joined after prediction.",
        "evaluation": "Same saved sequential prefixes as empirical Bayes. Score only risk_cycle < observed Treb, a retrospective evaluation restriction.",
        "averaging": "Pooled updates weight every scored forecast equally. Equal patients average each patient's mean absolute error.",
        "paired_comparison": "Each pair uses exactly the updates with both forecasts finite. Positive MAE reduction favors empirical Bayes.",
        "interval": "95% percentile patient-cluster bootstrap; each sampled patient contributes all scored prefixes, with repeats retained.",
        "bootstrap_draws": bootstrap,
        "seed": 20261104,
        "limitations": "The cohorts were selected retrospectively for an observed rebound. Intervals condition on the fitted model, training cohort, and observed follow-up.",
        "scored_updates": {str(k): int(v) for k,v in future.groupby("dataset").size().items()},
    }
    (dest/"results/SEQUENTIAL_BASELINE_METHOD.json").write_text(json.dumps(method, indent=2)+"\n")
    return summary, paired


def self_test():
    """Boundary, missing-risk-set, and aggregation tests independent of clinical data."""
    train = pd.DataFrame({"id": ["t1", "t2", "t3"], "Treb": [4., 6., 9.],
                          "confirm_cycle": [7., 7., 10.]})
    prefixes = pd.DataFrame({"dataset": ["x"]*3, "id": ["a"]*3,
                            "risk_index": [3, 4, 5], "risk_cycle": [5., 7., 10.],
                            "pred": [6., 8., 10.]})
    z = predict_baselines(train, prefixes)
    assert z.TRAIN_unconfirmed_median.iloc[0] == 6.  # includes low at 4, unconfirmed at 5
    assert z.TRAIN_unconfirmed_median.iloc[1] == 9.  # confirmation at current cycle excluded
    assert np.isnan(z.TRAIN_unconfirmed_median.iloc[2])
    assert z.n_training_unconfirmed.to_list() == [3, 1, 0]
    polluted = prefixes.assign(Treb=[100, 2, 0])
    pd.testing.assert_frame_equal(z, predict_baselines(train, polluted))
    scored = pd.DataFrame({"dataset": ["x"]*3, "id": ["a", "a", "b"],
                           "risk_cycle": [3, 4, 3], "Treb": [10., 10., 10.],
                           "empirical_Bayes": [9., 9., 1.],
                           "TRAIN_median": [6., 6., 6.],
                           "TRAIN_unconfirmed_median": [6., 6., 6.]})
    _, metrics, paired = score_predictions(scored, bootstrap=50)
    eb = metrics[metrics.model.eq("empirical_Bayes")].set_index("averaging")
    assert np.isclose(eb.loc["pooled_updates", "MAE"], 11./3)
    assert np.isclose(eb.loc["equal_patients", "MAE"], 5.)
    assert (paired.n_updates == 3).all() and (paired.n_patients == 2).all()
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("Sequential baseline tests passed")
    else:
        if args.source is None or args.out is None:
            parser.error("--source and --out are required")
        run(args.source, args.out, args.bootstrap)
