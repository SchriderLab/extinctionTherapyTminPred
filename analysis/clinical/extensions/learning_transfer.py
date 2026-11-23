"""Extend frozen VENICE learning-curve models to MAINSAIL.

The historical subsets, feature transform, tuning and random seeds are reused.
One fitted model predicts both trials. Evaluation targets are joined afterward.
The containing runner places the frozen package's scripts on sys.path.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, brier_score_loss
from sklearn.model_selection import StratifiedShuffleSplit

FRACTIONS = (0.25, 0.40, 0.55, 0.70, 0.85, 1.0)
KEYS = ["id", "scan_count", "fraction", "repeat"]
PROBABILITY_TOLERANCE = 1e-4


def training_subsets(features, targets, seed, repeats):
    """Reproduce predict.clinical_checks, including sklearn's rounded sizes."""
    for fraction in FRACTIONS:
        for repeat in range(1 if fraction == 1 else repeats):
            if fraction == 1:
                indices = np.arange(len(targets))
            else:
                splitter = StratifiedShuffleSplit(
                    1, train_size=fraction,
                    random_state=seed + repeat + int(fraction * 1000),
                )
                indices = next(splitter.split(features, targets > 8))[0]
            yield fraction, repeat, indices


def validate_reproduction(current, historical, tolerance=PROBABILITY_TOLERANCE):
    """Fail closed if rebuilding the historical models changes VENICE forecasts."""
    venice = current.loc[current.dataset.eq("venice")].copy()
    if historical.duplicated(KEYS).any() or venice.duplicated(KEYS).any():
        raise ValueError("Duplicate VENICE learning-curve prediction keys.")
    compare = venice.merge(
        historical[KEYS + ["prob_late8", "n_training"]],
        on=KEYS, how="outer", suffixes=("_new", "_old"),
        indicator=True, validate="one_to_one",
    )
    if not compare._merge.eq("both").all():
        counts = compare._merge.value_counts().to_dict()
        raise RuntimeError(f"VENICE learning-curve membership changed: {counts}")
    if not compare.n_training_new.eq(compare.n_training_old).all():
        raise RuntimeError("VENICE learning-curve training sizes changed.")
    delta = np.abs(compare.prob_late8_new - compare.prob_late8_old)
    if not np.isfinite(delta).all():
        raise RuntimeError("Missing or nonfinite VENICE learning-curve probability.")
    maximum = float(delta.max())
    if maximum > tolerance:
        raise RuntimeError(
            f"VENICE learning-curve replay differs from the frozen predictions "
            f"(maximum probability difference {maximum:.6g} > {tolerance}). "
            "Inspect the original environment and source before accepting this run."
        )
    return {"rows_compared": len(compare), "maximum_probability_difference": maximum,
            "absolute_tolerance": tolerance}


def score_fixed_patients(predictions, targets, fixed_ids):
    """Keep all training sizes on the same six-measurement future-event patients."""
    scored = predictions.merge(
        targets[["dataset", "id", "Treb"]],
        on=["dataset", "id"], validate="many_to_one", how="left",
    )
    if scored.Treb.isna().any():
        raise RuntimeError("Learning-curve forecast has no matching clinical target.")
    keep = [pid in fixed_ids[dataset] for dataset, pid in
            scored[["dataset", "id"]].itertuples(index=False, name=None)]
    scored = scored.loc[keep].copy()
    rows = []
    for (dataset, fraction, repeat, ntrain), group in scored.groupby(
            ["dataset", "fraction", "repeat", "n_training"], sort=True):
        if set(group.id) != fixed_ids[dataset]:
            raise RuntimeError(f"{dataset}: test cohort changed across training sizes.")
        labels = group.Treb.to_numpy() > 8
        prob = group.prob_late8.to_numpy()
        calls = prob >= 0.5
        both = np.unique(labels).size == 2
        rows.append({
            "dataset": dataset, "fraction": fraction, "repeat": repeat,
            "n_training": ntrain, "n_test": len(group),
            "n_early": int((~labels).sum()), "n_late": int(labels.sum()),
            "AUROC": roc_auc_score(labels, prob) if both else np.nan,
            "balanced_accuracy": balanced_accuracy_score(labels, calls) if both else np.nan,
            "early_recall": float((~calls[~labels]).mean()) if (~labels).any() else np.nan,
            "late_recall": float(calls[labels].mean()) if labels.any() else np.nan,
            "Brier": brier_score_loss(labels, prob),
        })
    return scored, pd.DataFrame(rows)


def plot_learning(summary, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4), layout="constrained",
                                sharey=True)
    for axis, dataset, color in zip(axes, ["venice", "mainsail"],
                                    ["#007F86", "#3269A8"]):
        data = summary.loc[summary.dataset.eq(dataset)]
        grouped = data.groupby("n_training").AUROC
        values = grouped.agg(mean="mean", low=lambda x: x.quantile(.025),
                             high=lambda x: x.quantile(.975), repeats="size")
        reduced = values[values.repeats > 1]
        axis.plot(values.index, values["mean"], "o-", color=color)
        axis.fill_between(reduced.index.to_numpy(), reduced.low.to_numpy(),
                          reduced.high.to_numpy(), color=color, alpha=.18)
        axis.axhline(.5, color="#888888", ls=":", lw=1)
        axis.set(xlabel="VENICE training patients", ylim=(0, 1),
                 title=f"{dataset.upper()} test patients (n={int(data.n_test.iloc[0])})")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("AUROC for rebound after cycle 8")
    figure.suptitle("The same VENICE-trained classifiers evaluated in both trials")
    for extension in ["pdf", "png"]:
        figure.savefig(destination / ("E09b_clinical_training_size." + extension),
                       dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(source: Path, dest: Path, seed: int):
    from common import features, matrix, csv, dump, digest
    from predict import clinical_fit, query_sets

    source, dest = Path(source), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for group in ["results", "predictions", "figures"]:
        (dest / group).mkdir(exist_ok=True)
    required = [
        "CONFIG_FROZEN.json", "training/fits_PRIVATE.csv",
        "training/long_PRIVATE.csv", "queries/venice_long_PRIVATE.csv",
        "queries/mainsail_long_PRIVATE.csv", "predictions/learning_PRIVATE.csv",
        "predictions/fixed_PRIVATE.csv", "evaluation/heldout_reference_PRIVATE.csv",
        "evaluation/mainsail_reference_PRIVATE.csv",
    ]
    missing = [relative for relative in required if not (source / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "Learning-curve transfer requires the complete Longleaf run directory. "
            "Missing " + ", ".join(missing)
        )
    config = json.loads((source / "CONFIG_FROZEN.json").read_text())
    if int(seed) != int(config["seed"]):
        raise ValueError("Learning-curve seed must match the frozen clinical run.")
    repeats = int(config["learning_repeats"])
    training = pd.read_csv(source / "training/fits_PRIVATE.csv", dtype={"id": str})
    long = pd.read_csv(source / "training/long_PRIVATE.csv", dtype={"id": str})
    if len(training) != 184 or training.id.duplicated().any():
        raise ValueError("Expected the frozen 184-patient VENICE training set.")
    values = matrix(long, training.id, 5)
    complete = np.isfinite(values).all(axis=1)
    training = training.loc[complete].copy()
    values = values[complete]
    if len(training) != 182:
        raise ValueError("Expected 182 VENICE training patients complete through cycle 5.")
    x_train = features(values)
    y_train = training.Treb.to_numpy()
    queries = query_sets(source, 5)
    query_features = {dataset: (ids, features(values))
                      for dataset, (ids, values) in queries.items()}
    if set(query_features) != {"venice", "mainsail"}:
        raise ValueError("Both VENICE and MAINSAIL queries are required.")

    forecast_rows, subset_rows = [], []
    for fraction, repeat, selected in training_subsets(
            x_train, y_train, seed, repeats):
        model = clinical_fit(x_train[selected], y_train[selected], True, seed + repeat)
        if model is None:
            raise RuntimeError("Historical learning-curve subset has insufficient classes.")
        subset_rows.append(pd.DataFrame({
            "fraction": fraction, "repeat": repeat, "id": training.id.to_numpy()[selected],
            "n_training": len(selected), "selected_C": model.best_params_["model__C"],
        }))
        for dataset, (ids, x_query) in query_features.items():
            forecast_rows.append(pd.DataFrame({
                "dataset": dataset, "id": ids, "scan_count": 6,
                "model": "Clinical_PSA_learning", "fraction": fraction,
                "repeat": repeat, "n_training": len(selected), "pred": np.nan,
                "prob_late8": model.predict_proba(x_query)[:, 1], "threshold": .5,
            }))
    predictions = pd.concat(forecast_rows, ignore_index=True)
    historical = pd.read_csv(source / "predictions/learning_PRIVATE.csv",
                             dtype={"id": str})
    comparison = validate_reproduction(predictions, historical)
    csv(dest / "predictions" / "LEARNING_TRANSFER_ALL_PREDICTIONS_PRIVATE.csv", predictions)
    csv(dest / "predictions" / "LEARNING_TRANSFER_TRAIN_SUBSETS_PRIVATE.csv",
        pd.concat(subset_rows, ignore_index=True))

    # Clinical evaluation outcomes enter only after all fitted models have predicted.
    targets = pd.concat([
        pd.read_csv(source / "evaluation/heldout_reference_PRIVATE.csv",
                    dtype={"id": str}).assign(dataset="venice"),
        pd.read_csv(source / "evaluation/mainsail_reference_PRIVATE.csv",
                    dtype={"id": str}).assign(dataset="mainsail"),
    ], ignore_index=True)
    fixed = pd.read_csv(source / "predictions/fixed_PRIVATE.csv", dtype={"id": str})
    fixed = fixed.loc[fixed.scan_count.eq(6) & fixed.model.eq("Constant_TRAIN")]
    fixed = fixed.merge(targets[["dataset", "id", "Treb"]],
                        on=["dataset", "id"], validate="one_to_one")
    fixed_ids = {name: set(fixed.loc[
        fixed.dataset.eq(name) & fixed.Treb.gt(5), "id"])
        for name in ["venice", "mainsail"]}
    if {name: len(ids) for name, ids in fixed_ids.items()} != {"venice": 28, "mainsail": 94}:
        raise RuntimeError("Expected the frozen matched 28 VENICE / 94 MAINSAIL test patients.")
    if set(training.id) & fixed_ids["venice"]:
        raise RuntimeError("Clinical training and VENICE test IDs overlap.")
    scored, summary = score_fixed_patients(predictions, targets, fixed_ids)
    csv(dest / "predictions" / "LEARNING_TRANSFER_SCORED_PRIVATE.csv", scored)
    csv(dest / "results" / "LEARNING_TRANSFER_REPETITIONS.csv", summary)
    plot_learning(summary, dest / "figures")
    manifest = {
        "source": str(source.resolve()),
        "inputs": {relative: digest(source / relative) for relative in required},
        "seed": int(seed), "repeats_per_reduced_size": repeats,
        "full_size_repetitions": 1, "training_sizes": sorted(summary.n_training.unique().tolist()),
        "fixed_test_patients": {name: len(ids) for name, ids in fixed_ids.items()},
        "cutoff": 8, "probability_threshold": .5,
        "venice_replay_check": comparison,
        "interval": "Central 95% range over repeated training subsets, conditional on fixed test patients. Full training is one fitted model.",
        "interpretation": "Each VENICE subset fits one model, then that same fitted model predicts both trial cohorts. All tuning uses its VENICE training subset.",
    }
    dump(dest / "results" / "LEARNING_TRANSFER_MANIFEST.json", manifest)
    return manifest


def _self_test():
    """Small checks of the changed cohort/replay contracts without clinical data."""
    import unittest

    class Contracts(unittest.TestCase):
        def test_original_stratified_sizes(self):
            x = np.arange(182 * 3).reshape(182, 3)
            y = np.where(np.arange(182) % 3 == 0, 10., 6.)
            sizes = [len(ix) for frac, rep, ix in training_subsets(x, y, 20261029, 1)]
            self.assertEqual(sizes, [45, 72, 100, 127, 154, 182])

        def test_replay_rejects_probability_or_membership_changes(self):
            historical = pd.DataFrame({
                "id": ["a", "b"], "scan_count": [6, 6], "fraction": [.25, .25],
                "repeat": [0, 0], "n_training": [45, 45], "prob_late8": [.2, .8],
            })
            current = historical.assign(dataset="venice")
            self.assertEqual(validate_reproduction(current, historical)["rows_compared"], 2)
            changed = current.copy()
            changed.loc[0, "prob_late8"] += .01
            with self.assertRaises(RuntimeError):
                validate_reproduction(changed, historical)
            with self.assertRaises(RuntimeError):
                validate_reproduction(current.iloc[:1], historical)

        def test_targets_do_not_change_fixed_membership_by_training_size(self):
            predictions = pd.DataFrame({
                "dataset": ["venice"] * 4, "id": ["a", "b"] * 2,
                "fraction": [.5, .5, 1., 1.], "repeat": [0] * 4,
                "n_training": [10, 10, 20, 20], "prob_late8": [.1, .9, .2, .8],
            })
            targets = pd.DataFrame({"dataset": ["venice"] * 2,
                                    "id": ["a", "b"], "Treb": [6., 10.]})
            scored, summary = score_fixed_patients(
                predictions, targets, {"venice": {"a", "b"}})
            self.assertEqual(summary.n_test.tolist(), [2, 2])
            self.assertEqual(summary.AUROC.tolist(), [1., 1.])
            with self.assertRaises(RuntimeError):
                score_fixed_patients(predictions.iloc[:-1], targets,
                                     {"venice": {"a", "b"}})

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    else:
        if args.source is None or args.out is None:
            parser.error("--source and --out are required unless --self-test is supplied")
        from common import require_compute
        require_compute()
        run(args.source, args.out, args.seed)

