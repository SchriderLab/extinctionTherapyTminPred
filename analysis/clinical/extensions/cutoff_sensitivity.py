"""TRAIN-only endpoint-cutoff sensitivity on a fixed six-measurement test cohort."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from common import matrix, features, csv, dump, digest
from predict import abc_predict, clinical_fit

CUTOFFS = (6, 7, 8, 9)
MODELS = ('ABC_noisy_joint', 'Clinical_PSA')


def probability_threshold(labels, probability):
    if np.unique(labels).size < 2:
        return 0.5
    grid = np.unique(np.r_[.01, np.quantile(probability, np.linspace(0, 1, 101)), .99])
    scores = [balanced_accuracy_score(labels, probability >= value) for value in grid]
    return float(grid[np.argmax(scores)])


def tune_cutoffs(R, y, weights, seed, cutoffs=CUTOFFS):
    """Original ABC candidate grid/folds, with endpoint labels recomputed per cutoff."""
    splits = list(KFold(5, shuffle=True, random_state=seed).split(R))
    ks = [k for k in (10, 25, 50, 100) if k < len(R) * .8]
    records = []
    for slope in (0., .5, 1., 2.):
        probabilities = {(c, k): np.full(len(R), np.nan) for c in cutoffs for k in ks}
        for tr, va in splits:
            sr = np.maximum(np.std(R[tr, 1:], axis=0, ddof=1), 1e-8)
            sd = np.maximum(np.std(np.diff(R[tr], axis=1), axis=0, ddof=1), 1e-8)
            def transform(values):
                return np.c_[values[:, 1:] / sr, slope * np.diff(values, axis=1) / sd]
            distance = cdist(transform(R[va]), transform(R[tr]))
            neighbors = np.argsort(distance, axis=1, kind='stable')[:, :max(ks)]
            nearest_distance = np.take_along_axis(distance, neighbors, axis=1)
            target = y[tr][neighbors]
            prior = weights[tr][neighbors]
            for k in ks:
                mass = prior[:, :k] / (nearest_distance[:, :k] + 1e-8)
                mass /= mass.sum(axis=1, keepdims=True)
                for cutoff in cutoffs:
                    probabilities[cutoff, k][va] = np.sum(mass * (target[:, :k] > cutoff), axis=1)
        for (cutoff, k), probability in probabilities.items():
            labels = y > cutoff
            auc = roc_auc_score(labels, probability) if np.unique(labels).size == 2 else np.nan
            records.append(dict(cutoff=cutoff, k=k, slope=slope, AUROC=auc,
                                probability_threshold=probability_threshold(labels, probability)))
    table = pd.DataFrame(records)
    best = {}
    for cutoff in cutoffs:
        candidates = [r for r in records if r['cutoff'] == cutoff and np.isfinite(r['AUROC'])]
        if not candidates:
            raise ValueError(f'No two-class reference tuning sample for cutoff {cutoff}')
        best[cutoff] = max(candidates, key=lambda r: (r['AUROC'], -r['k']))
    return best, table


def score(y, probability, threshold):
    y = np.asarray(y, bool)
    probability = np.asarray(probability, float)
    prediction = probability >= threshold
    result = dict(n=len(y), n_early=int((~y).sum()), n_late=int(y.sum()),
                  Brier=float(np.mean((probability - y) ** 2)))
    if np.unique(y).size == 2:
        result.update(AUROC=float(roc_auc_score(y, probability)),
                      balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
                      early_recall=float((~prediction[~y]).mean()),
                      late_recall=float(prediction[y].mean()))
    else:
        result.update(AUROC=np.nan, balanced_accuracy=np.nan,
                      early_recall=float((~prediction[~y]).mean()) if (~y).any() else np.nan,
                      late_recall=float(prediction[y].mean()) if y.any() else np.nan)
    return result


def summarize(predictions, bootstrap, seed):
    rows = []
    rng = np.random.default_rng(seed)
    for (dataset, n, cutoff), paired in predictions.groupby(['dataset', 'scan_count', 'cutoff']):
        models = {key: value.sort_values('id') for key, value in paired.groupby('model')}
        ids = next(iter(models.values())).id.tolist()
        if any(value.id.tolist() != ids for value in models.values()):
            raise RuntimeError('Different patients in the paired cutoff comparison')
        draws = rng.integers(len(ids), size=(bootstrap, len(ids)))
        for model, frame in models.items():
            y = frame.Treb.to_numpy() > cutoff
            p = frame.prob_late.to_numpy()
            threshold = frame.probability_threshold.to_numpy()
            record = dict(dataset=dataset, scan_count=n, cutoff=cutoff, model=model,
                          probability_threshold=float(threshold[0]), **score(y, p, threshold))
            values = pd.DataFrame([score(y[ix], p[ix], threshold[ix]) for ix in draws])
            for measure in ['AUROC', 'balanced_accuracy', 'early_recall', 'late_recall', 'Brier']:
                valid = values[measure].dropna()
                record[measure + '_lo'] = valid.quantile(.025) if len(valid) else np.nan
                record[measure + '_hi'] = valid.quantile(.975) if len(valid) else np.nan
                record[measure + '_bootstrap_n'] = len(valid)
            rows.append(record)
    return pd.DataFrame(rows)


def run(source, dest, bootstrap=2000):
    source, dest = Path(source), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    configuration = json.loads((source / 'CONFIG_FROZEN.json').read_text())
    seed = configuration['seed']
    bank_path = source / 'simulation/REFERENCE.npz'
    lock = json.loads((source / 'simulation/REFERENCE_LOCK.json').read_text())
    if digest(bank_path) != lock['sha256']:
        raise RuntimeError('Frozen clinical reference bank failed its SHA256 lock')
    with np.load(bank_path, allow_pickle=False) as bank:
        reference, targets, weights = bank['noisy'], bank['Treb_noisy'], bank['weights']
    training = pd.read_csv(source / 'training/fits_PRIVATE.csv', dtype={'id': str})
    training_long = pd.read_csv(source / 'training/long_PRIVATE.csv', dtype={'id': str})
    historical = pd.read_csv(source / 'predictions/fixed_PRIVATE.csv', dtype={'id': str})
    heldout = {}
    for dataset, expected in [('venice', 28), ('mainsail', 94)]:
        name = 'heldout' if dataset == 'venice' else 'mainsail'
        target = pd.read_csv(source / f'evaluation/{name}_reference_PRIVATE.csv', dtype={'id': str})
        long = pd.read_csv(source / f'queries/{dataset}_long_PRIVATE.csv', dtype={'id': str})
        Y = matrix(long, target.id, 5)
        eligible = np.isfinite(Y).all(axis=1) & (target.Treb.to_numpy() > 5)
        heldout[dataset] = (target.loc[eligible].copy(), Y[eligible])
        if eligible.sum() != expected:
            raise RuntimeError(f'{dataset} common cohort changed: {eligible.sum()}, expected {expected}')
    predictions, tuning = [], []
    for n in configuration['scan_counts']:
        R, yr = reference[:, :n], targets
        eligible = np.isfinite(R).all(axis=1) & np.isfinite(yr)
        R, yr, weight = R[eligible], yr[eligible], weights[eligible]
        weight = weight / weight.sum()
        settings, cv = tune_cutoffs(R, yr, weight, seed + n)
        tuning.append(cv.assign(scan_count=n))
        Y = matrix(training_long, training.id, n - 1)
        valid = np.isfinite(Y).all(axis=1)
        X, train_y = features(Y[valid]), training.loc[valid, 'Treb'].to_numpy()
        for cutoff in CUTOFFS:
            # Translating a continuous target preserves the original >8 clinical-fit
            # implementation exactly while placing the requested threshold at cutoff.
            classifier = clinical_fit(X, train_y - cutoff + 8, True, seed)
            if classifier is None:
                raise RuntimeError(f'Insufficient TRAIN classes at cutoff {cutoff}')
            setting = settings[cutoff]
            for dataset, (target, queries) in heldout.items():
                Q = queries[:, :n]
                probability = abc_predict(R, yr, weight, Q, setting['k'], setting['slope'], cutoff=cutoff)[2]
                clinical_probability = classifier.predict_proba(features(Q))[:, 1]
                for model, probs, threshold in [(MODELS[0], probability, setting['probability_threshold']),
                                                 (MODELS[1], clinical_probability, .5)]:
                    predictions.append(pd.DataFrame(dict(dataset=dataset, id=target.id.to_numpy(),
                        scan_count=n, cutoff=cutoff, model=model, Treb=target.Treb.to_numpy(),
                        prob_late=probs, probability_threshold=threshold)))
        print(f'Clinical cutoff sensitivity complete for {n} measurements', flush=True)
    result = pd.concat(predictions, ignore_index=True)
    anchor = result[result.cutoff.eq(8)].merge(historical[['dataset','id','scan_count','model','prob_late8','threshold']],
        on=['dataset','id','scan_count','model'], validate='one_to_one')
    if len(anchor) != len(result[result.cutoff.eq(8)]):
        raise RuntimeError('Missing original cutoff-8 anchor predictions')
    np.testing.assert_allclose(anchor.prob_late, anchor.prob_late8, rtol=1e-5, atol=1e-7,
                               err_msg='Cutoff-8 predictions differ from frozen original')
    np.testing.assert_allclose(anchor.probability_threshold, anchor.threshold, atol=1e-9)
    csv(dest / 'CUTOFF_PREDICTIONS_PRIVATE.csv', result)
    csv(dest / 'CUTOFF_TRAINING_CV.csv', pd.concat(tuning, ignore_index=True))
    scores = summarize(result, bootstrap, seed + 160)
    csv(dest / 'CUTOFF_METRICS.csv', scores)
    dump(dest / 'CUTOFF_DESIGN.json', dict(primary_cutoff=8, sensitivity_cutoffs=list(CUTOFFS),
        reference_sha256=lock['sha256'], cohort='Same 28 VENICE and 94 MAINSAIL patients at every cutoff and input count',
        abc_tuning='Reference fivefold CV AUROC; probability threshold maximizes reference CV balanced accuracy',
        logistic_tuning='VENICE TRAIN-only stratified CV; probability threshold 0.5',
        uncertainty='2000 patient bootstrap draws by default, fitted predictions fixed; single-class resamples omit AUROC/BA',
        original_cutoff8_match=True, bootstrap_replicates=bootstrap))
    plot(scores, dest)
    return scores


def plot(scores, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), layout='constrained')
    for column, dataset in enumerate(['venice', 'mainsail']):
        for row, metric in enumerate(['AUROC', 'balanced_accuracy']):
            ax = axes[row, column]
            for model, color, label in [(MODELS[0], '#007F86', 'ABC'), (MODELS[1], '#367544', 'Clinical logistic')]:
                part = scores[(scores.dataset == dataset) & (scores.scan_count == 6) & (scores.model == model)].sort_values('cutoff')
                ax.plot(part.cutoff, part[metric], 'o-', color=color, label=label)
                ax.fill_between(part.cutoff, part[metric+'_lo'], part[metric+'_hi'], color=color, alpha=.12)
            ax.axhline(.5, color='.6', linestyle=':')
            ax.axvline(8, color='.6', linewidth=.8)
            ax.set(xlabel='Rebound-cycle cutoff', ylabel=metric.replace('_', ' '),
                   title=dataset.upper(), ylim=(0,1), xticks=list(CUTOFFS))
            ax.legend(fontsize=8)
    fig.suptitle('Six-measurement classification on the same patients at each cutoff')
    fig.savefig(dest/'E07c_binary_cutoff_sensitivity.pdf')
    fig.savefig(dest/'E07c_binary_cutoff_sensitivity.png', dpi=160)
    plt.close(fig)
