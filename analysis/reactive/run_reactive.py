#!/usr/bin/env python3
"""Recompute trigger-only policies from frozen population histories and outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from abc_core import abc_noisy, score_at_switch

SCANS = (0, 4, 8, 12, 16, 19)
RULES = ('first_increase', 'progression_1.20', 'ned_10000')
SIGMAS = (0.0, 0.05, 0.10, 0.20, 0.30)
LEVELS = ['logN0', 'logN1', 'logN2', 'logN3']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_inputs(repo, source_tables=None):
    if source_tables is None:
        paths = {
            'patients': repo / 'data/cohort/patient_oracle_characterization.csv',
            'outcomes': repo / 'data/cohort/extinction_by_lag_binary300.csv',
            'abc': repo / 'data/cohort/abc_hybrid10k_4scan/abc_80patients_4scan_hybrid10k.csv',
            'reference': repo / 'data/reference/abc_reference_features_usable.csv',
        }
    else:
        paths = {name: source_tables / filename for name, filename in {
            'patients': 'patient_oracle_characterization.csv',
            'outcomes': 'extinction_by_lag_binary300.csv',
            'abc': 'Fig3_abc_predictions.csv',
            'reference': 'abc_reference_features_all.csv',
            'trajectories': 'Fig2A_drug1_trajectories.csv',
        }.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f'Missing frozen input: {path}')
    patients = pd.read_csv(paths['patients']).sort_values('patient_id')
    if patients.patient_id.duplicated().any() or len(patients) != 80:
        raise ValueError('Expected exactly 80 unique frozen patients')
    ids = patients.patient_id.to_numpy(int)
    ext = pd.read_csv(paths['outcomes'])
    if not np.array_equal(np.sort(ext.patient_id.unique()), ids):
        raise ValueError('Extinction input patient IDs differ from the frozen cohort')
    if (ext.n_runs != 10).any():
        raise ValueError('Expected ten post-switch replicates per patient and lag')
    if not np.allclose(ext.extinction_probability, ext.n_extinct / ext.n_runs):
        raise ValueError('Extinction fractions disagree with integer outcomes')
    surface_frame = ext.pivot(index='patient_id', columns='lag', values='extinction_probability').reindex(ids)
    if list(surface_frame.columns) != list(range(1, 21)) or surface_frame.isna().any().any():
        raise ValueError('Expected a complete 80 x 20 outcome surface (lag=t+1)')
    surface = surface_frame.to_numpy(float)
    if not np.isfinite(surface).all() or np.any((surface < 0) | (surface > 1)):
        raise ValueError('Invalid extinction probabilities')
    abc = pd.read_csv(paths['abc'])
    abc = abc[abc.method == 'logN_plus_slope'].sort_values('patient_id')
    if not np.array_equal(abc.patient_id.to_numpy(int), ids):
        raise ValueError('ABC IDs differ from patient IDs or contain duplicates')
    if (abc.k != 25).any() or (abc.slope_weight != 2).any():
        raise ValueError('Expected frozen ABC K=25 and slope weight 2')
    abc_t = abc.pred_switch_lag_round.to_numpy(float) - 1
    abc_ext = score_at_switch(surface, abc_t)
    population = np.full((len(ids), 20), np.nan)
    combined = pd.read_csv(paths['trajectories']) if source_tables else None
    for index, patient_id in enumerate(ids):
        if combined is None:
            path = repo / 'data/cohort/patients' / f'patient_{patient_id:03d}' / 'drug1_prefix.csv'
            paths[f'patient_{patient_id:03d}'] = path
            frame = pd.read_csv(path).rename(columns={'t_treatment': 't'})
        else:
            frame = combined[combined.patient_id == patient_id]
        times = pd.to_numeric(frame.t, errors='raise').to_numpy(float)
        counts = pd.to_numeric(frame.N, errors='raise').to_numpy(float)
        if not len(times) or not np.isfinite(times).all() or not np.allclose(times, np.rint(times)):
            raise ValueError(f'Invalid times for patient {patient_id}')
        if (np.diff(times) <= 0).any() or not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError(f'Unordered/duplicate times or invalid counts for patient {patient_id}')
        for t, n in zip(times.astype(int), counts):
            if 0 <= t <= 19:
                population[index, t] = n
        if np.isnan(population[index, :13]).any():
            raise ValueError(f'Patient {patient_id} lacks an exact cycle 0--12 observation')
        first_min = int(times[np.argmin(counts)])
        if first_min != patients.iloc[index].tmin or counts.min() != patients.iloc[index].N_min:
            raise ValueError(f'Saved nadir disagrees with trajectory for patient {patient_id}')
    reference = pd.read_csv(paths['reference'])
    if 'usable' in reference:
        reference = reference[reference.usable == 1]
    if 'tmin_boundary' in reference:
        reference = reference[reference.tmin_boundary != 1]
    rx = reference[LEVELS].to_numpy(float)
    ry = reference.t_min_true.to_numpy(float)
    if len(ry) != 9983 or not np.isfinite(rx).all() or not np.isfinite(ry).all():
        raise ValueError('Expected 9,983 complete frozen reference trajectories')
    if np.any((ry <= 3) | (ry >= 19)):
        raise ValueError('Reference target outside cycles 4--18')
    return {'ids': ids, 'population': population, 'tmin': patients.tmin.to_numpy(float),
            'surface': surface, 'abc_t': abc_t, 'abc_ext': abc_ext,
            'abc_pred': abc.pred_tmin_weighted.to_numpy(float),
            'ref_x': rx, 'ref_y': ry, 'input_paths': paths}


def trigger(counts, scans, rule):
    """Only actual observations may trigger. Missing observations remain missing."""
    counts = np.asarray(counts, float)
    scans = np.asarray(scans, int)
    if rule not in RULES or counts.shape[-1] != len(scans) or (np.diff(scans) <= 0).any():
        raise ValueError('Invalid rule, observation shape or schedule')
    if np.isinf(counts).any() or np.any(np.isfinite(counts) & (counts < 0)):
        raise ValueError('Population values must be nonnegative or missing')
    switch = np.full(counts.shape[:-1], np.nan)
    previous = np.full_like(switch, np.nan)
    minimum = np.full_like(switch, np.inf)
    entered_ned = np.zeros(switch.shape, bool)
    for index, cycle in enumerate(scans):
        value = counts[..., index]
        valid = np.isfinite(value)
        if rule == 'first_increase':
            new = valid & np.isfinite(previous) & (value > previous)
        elif rule == 'progression_1.20':
            new = valid & np.isfinite(minimum) & (value > minimum) & (value >= 1.20 * minimum)
        else:
            new = valid & entered_ned & (value > 10000)
        switch[np.isnan(switch) & new] = cycle
        entered_ned |= valid & (value < 10000)
        minimum = np.where(valid, np.minimum(minimum, value), minimum)
        previous = np.where(valid, value, previous)
    return switch, entered_ned, np.isfinite(counts).all(axis=-1)


def cluster_intervals(hit, values, bootstrap, seed):
    """Ratio-of-totals bootstrap over patients, retaining every noise draw."""
    n = hit.shape[1]
    samples = np.random.default_rng(seed).integers(0, n, (bootstrap, n))
    denominator = hit.sum(axis=0)[samples].sum(axis=1)
    valid = denominator > 0
    result = {'bootstrap_replicates': bootstrap, 'bootstrap_with_triggers': int(valid.sum())}
    for name, value in values.items():
        sums = np.where(hit, value, 0).sum(axis=0)[samples].sum(axis=1)
        interval = np.quantile(sums[valid] / denominator[valid], [.025, .975]) if valid.any() else [np.nan, np.nan]
        result[f'{name}_ci_low'], result[f'{name}_ci_high'] = map(float, interval)
    coverage = denominator / (hit.shape[0] * n)
    result['trigger_fraction_ci_low'], result['trigger_fraction_ci_high'] = map(float, np.quantile(coverage, [.025, .975]))
    return result


def evaluate_counts(data, counts, scans, rule, abc_ext, sigma, bootstrap, seed, family='main'):
    if counts.ndim == 2:
        counts = counts[None, ...]
    switches, entered, complete = trigger(counts, scans, rule)
    hit = np.isfinite(switches)
    reactive = score_at_switch(data['surface'], switches)
    abc_ext = np.broadcast_to(abc_ext, switches.shape)
    delay = switches - data['tmin'][None, :]
    values = {'reactive_extinction': reactive, 'abc_extinction_matched': abc_ext,
              'abc_minus_reactive': abc_ext - reactive, 'switch_delay': delay}
    n_draws, n_patients = hit.shape
    summary = {'family': family, 'rule': rule, 'n_scans': len(scans),
               'schedule': ','.join(map(str, scans)), 'horizon': int(scans[-1]), 'sigma': sigma,
               'n_patients': n_patients, 'n_draws': n_draws, 'n_instances': hit.size,
               'n_triggered': int(hit.sum()), 'n_unique_triggered_patients': int(hit.any(axis=0).sum()),
               'trigger_fraction_all': float(hit.mean()), 'n_missing_followup': int((~complete).sum()),
               'n_no_trigger_complete_followup': int((~hit & complete).sum()),
               'n_no_trigger_missing_followup': int((~hit & ~complete).sum()),
               'abc_extinction_full_cohort': float(abc_ext.mean())}
    if rule == 'ned_10000':
        summary.update(n_observed_below_threshold=int(entered.sum()),
                       n_never_observed_below_threshold=int((~entered).sum()),
                       n_observed_below_no_detected_return=int((entered & ~hit).sum()))
    for name, value in values.items():
        summary[name] = float(value[hit].mean()) if hit.any() else np.nan
    summary.update(cluster_intervals(hit, values, bootstrap, seed))
    status = np.full(hit.shape, 'no_trigger_with_complete_followup', dtype=object)
    if rule == 'ned_10000':
        status[~entered] = 'never_observed_below_threshold'
        status[entered] = 'observed_below_no_detected_return'
    status[~complete] = 'no_trigger_with_missing_followup'
    status[hit] = 'triggered'
    table = pd.DataFrame({'family': family, 'rule': rule, 'n_scans': len(scans),
                          'schedule': summary['schedule'], 'sigma': sigma,
                          'draw': np.repeat(np.arange(n_draws), n_patients),
                          'patient_id': np.tile(data['ids'], n_draws),
                          'triggered': hit.ravel(), 'status': status.ravel(),
                          'complete_followup': complete.ravel(), 'observed_below_threshold': entered.ravel(),
                          'switch_t': switches.ravel(), 'selected_lag': (switches + 1).ravel(),
                          'true_tmin': np.tile(data['tmin'], n_draws)})
    for name, value in values.items():
        table[name] = value.ravel()
    return summary, table


def check_frozen(data, summary):
    four = summary[summary.n_scans == 4].set_index('rule')
    expected = {'first_increase': (63, 0.607936507936508, 0.685714285714286),
                'progression_1.20': (57, 0.631578947368421, 0.714035087719298),
                'ned_10000': (11, 0.272727272727273, 0.472727272727273)}
    for rule, reference in expected.items():
        observed = four.loc[rule, ['n_triggered', 'reactive_extinction', 'abc_extinction_matched']].to_numpy(float)
        if not np.allclose(observed, reference, atol=1e-10, rtol=0):
            raise AssertionError(f'Four-scan frozen check failed for {rule}: {observed}')
    first5 = summary[(summary.n_scans == 5) & (summary.rule == 'first_increase')].iloc[0]
    if first5.n_triggered != 80 or not np.isclose(first5.reactive_extinction, .60125):
        raise AssertionError('Five-scan first-increase check failed')
    if not np.isclose(data['abc_ext'].mean(), .70375):
        raise AssertionError('Frozen full-cohort ABC result changed')
    if np.isnan(data['population'][:, 19]).sum() != 2:
        raise AssertionError('Expected two missing cycle-19 records')


def mask_ned(log_counts):
    """Zero is the shared nondetect code on the log10(count + 1) scale."""
    return np.where(log_counts < np.log10(10001.), 0., log_counts)


def ned_forecasts(data, query):
    # Mask first, so both the level and the slope features respect detection.
    query = mask_ned(query)
    unique, inverse = np.unique(query.reshape(-1, 4), axis=0, return_inverse=True)
    prediction = abc_noisy(mask_ned(data['ref_x']), data['ref_y'], unique)
    prediction = prediction[inverse].reshape(query.shape[:-1])
    switch = np.clip(np.rint(prediction), 0, 19)
    return prediction, switch, score_at_switch(data['surface'], switch)


def policy_pair(data, counts, scans, rule, raw_score, masked_score, sigma, bootstrap, seed, sensitivity):
    if rule == 'ned_10000':
        raw, _ = evaluate_counts(data, counts, scans, rule, raw_score, sigma, bootstrap, seed)
        raw['abc_inputs'] = 'unmasked_observations'
        sensitivity.append(raw)
        score, inputs = masked_score, 'below_10000_encoded_zero'
    else:
        score, inputs = raw_score, 'ordinary'
    summary, table = evaluate_counts(data, counts, scans, rule, score, sigma, bootstrap, seed)
    summary['abc_inputs'] = table['abc_inputs'] = inputs
    return summary, table


def run_noiseless(data, output, bootstrap, seed):
    early = np.log10(1 + data['population'][:, :4])
    prediction, switch, masked = ned_forecasts(data, early)
    sensitivity = []
    pd.DataFrame({'patient_id': data['ids'], 'any_early_nondetect': (early < np.log10(10001.)).any(axis=1),
                  'raw_prediction': data['abc_pred'], 'raw_switch_t': data['abc_t'],
                  'masked_prediction': prediction, 'masked_switch_t': switch,
                  'raw_extinction': data['abc_ext'], 'masked_extinction': masked}).to_csv(
                      output / 'ned_noiseless_forecasts.csv', index=False)
    summaries, frames = [], []
    for budget in (4, 5, 6):
        scans = SCANS[:budget]
        for rule_index, rule in enumerate(RULES):
            summary, frame = policy_pair(data, data['population'][:, scans], scans, rule,
                data['abc_ext'], masked, 0, bootstrap, seed + 100 * budget + rule_index, sensitivity)
            summaries.append(summary)
            frames.append(frame)
    summary = pd.DataFrame(summaries)
    pd.DataFrame(sensitivity).to_csv(output / 'ned_noiseless_unmasked_sensitivity.csv', index=False)
    check_frozen(data, summary)
    summary.to_csv(output / 'noiseless_summary.csv', index=False)
    pd.DataFrame({'patient_id': np.repeat(data['ids'], 20), 'cycle': np.tile(np.arange(20), len(data['ids'])),
                  'N': data['population'].ravel()}).to_csv(output / 'observed_population.csv', index=False)
    pd.concat(frames, ignore_index=True).to_csv(output / 'noiseless_patient_level.csv', index=False)
    dense_summaries, dense_frames = [], []
    for horizon in (12, 19):
        for budget in range(2, horizon + 2):
            scans = tuple(np.rint(np.linspace(0, horizon, budget)).astype(int))
            summary, frame = evaluate_counts(data, data['population'][:, scans], scans, 'first_increase',
                data['abc_ext'], 0, bootstrap, seed + horizon * 100 + budget, family='dense_sensitivity')
            dense_summaries.append(summary)
            dense_frames.append(frame)
    pd.DataFrame(dense_summaries).to_csv(output / 'dense_summary.csv', index=False)
    pd.concat(dense_frames, ignore_index=True).to_csv(output / 'dense_patient_level.csv', index=False)
    switch, _, _ = trigger(data['population'][:, SCANS[:4]], SCANS[:4], 'first_increase')
    late = np.isnan(switch)
    fallback = score_at_switch(data['surface'], np.where(late, 12, switch))
    wait = score_at_switch(data['surface'], np.where(late, 16, switch))
    pd.DataFrame([{'n_without_rise_at_12': int(late.sum()), 'all_rise_at_16': bool(np.all(data['population'][late, 16] > data['population'][late, 12])),
                   'extinction_17_switch12': float(fallback[late].mean()), 'extinction_17_switch16': float(wait[late].mean()),
                   'extinction_all_switch12_fallback': float(fallback.mean()), 'extinction_all_wait16': float(wait.mean())}]).to_csv(output / 'fallback_diagnostic.csv', index=False)
    return pd.DataFrame(summaries)


def noisy_measurements(data, sigma, sigma_index, rng, draws, seed):
    # Preserve the original 13-cycle, three-biomarker RNG shape and sigma order.
    truth = np.log10(data['population'][:, :13] + 1)
    noisy = truth[None, :, None, :] + rng.normal(0, sigma, (draws, len(data['ids']), 3, 13))
    observed = np.full((draws, len(data['ids']), 20), np.nan)
    observed[:, :, :13] = np.maximum(10 ** noisy[:, :, 0, :] - 1, 0)
    for cycle in (16, 19):
        future_rng = np.random.default_rng(np.random.SeedSequence([seed, cycle, sigma_index]))
        z = np.log10(data['population'][:, cycle] + 1)[None, :] + future_rng.normal(0, sigma, (draws, len(data['ids'])))
        observed[:, :, cycle] = np.maximum(10 ** z - 1, 0)
    if sigma == 0:
        observed[:, :, :13] = data['population'][:, :13]
        for cycle in (16, 19):
            observed[:, :, cycle] = data['population'][:, cycle]
    return noisy, observed


def run_noise(data, output, draws, bootstrap, seed):
    rng = np.random.default_rng(seed)
    summaries, abc_summaries, sample_records = [], [], []
    sensitivity = []
    noise_path = output / 'noise_patient_draws.csv.gz'
    first_write = True
    abc_path = output / 'abc_noise_patient_draws.csv.gz'
    abc_first_write = True
    for si, sigma in enumerate(SIGMAS):
        print(f'Reactive noise SD={sigma}, {draws} draws per patient', flush=True)
        noisy, observed = noisy_measurements(data, sigma, si, rng, draws, seed)
        abc_one = None
        for biomarkers in (1, 2, 3):
            query = noisy[:, :, :biomarkers, :4].mean(axis=2).reshape(-1, 4)
            if sigma == 0:
                prediction = np.tile(data['abc_pred'], (draws, 1))
            else:
                prediction = abc_noisy(data['ref_x'], data['ref_y'], query).reshape(draws, -1)
            switch = np.clip(np.rint(prediction), 0, 19)
            score = score_at_switch(data['surface'], switch)
            error = np.abs(prediction - data['tmin'][None, :])
            if sigma == 0 and not np.array_equal(score[0], data['abc_ext']):
                raise AssertionError('Noiseless noisy-pipeline ABC differs from frozen switch outcomes')
            if biomarkers == 1:
                abc_one = score
                early = noisy[:, :, 0, :4]
                masked_pred, masked_switch, masked_score = ned_forecasts(data, early)
                pd.DataFrame({'sigma': sigma, 'draw': np.repeat(np.arange(draws), len(data['ids'])),
                    'patient_id': np.tile(data['ids'], draws),
                    'any_early_nondetect': (early < np.log10(10001.)).any(axis=2).ravel(),
                    'raw_prediction': prediction.ravel(), 'raw_switch_t': switch.ravel(),
                    'masked_prediction': masked_pred.ravel(), 'masked_switch_t': masked_switch.ravel(),
                    'raw_extinction': score.ravel(), 'masked_extinction': masked_score.ravel()}).to_csv(
                    output / 'ned_noise_forecasts.csv.gz', mode='a' if si else 'w',
                    header=(si == 0), index=False, compression='gzip')
            abc_summaries.append({'sigma': sigma, 'n_biomarkers': biomarkers,
                                  'n_patients': len(data['ids']), 'n_draws': draws,
                                  'mean_extinction': float(score.mean()), 'tmin_mae': float(error.mean())})
            table = pd.DataFrame({'sigma': sigma, 'n_biomarkers': biomarkers,
                'draw': np.repeat(np.arange(draws), len(data['ids'])), 'patient_id': np.tile(data['ids'], draws),
                'pred_tmin': prediction.ravel(), 'switch_t': switch.ravel(),
                'extinction': score.ravel(), 'abs_error': error.ravel()})
            table.to_csv(abc_path, mode='w' if abc_first_write else 'a', header=abc_first_write, index=False, compression='gzip')
            abc_first_write = False
        for budget in (4, 5, 6):
            for pi, rule in enumerate(RULES):
                scans = SCANS[:budget]
                summary, table = policy_pair(data, observed[:, :, scans], scans, rule, abc_one, masked_score,
                    sigma, bootstrap, seed + 10000 + si * 100 + budget * 10 + pi, sensitivity)
                summaries.append(summary)
                table.to_csv(noise_path, mode='w' if first_write else 'a', header=first_write, index=False, compression='gzip')
                first_write = False
        # Fixed example selected before observation-noise draws; all 13 original times shown.
        for biomarker in range(3):
            for cycle in range(13):
                sample_records.append({'patient_id': int(data['ids'][0]), 'sigma': sigma, 'draw': 0,
                    'biomarker': biomarker + 1, 'cycle': cycle,
                    'observed_log10_N_plus1': float(noisy[0, 0, biomarker, cycle]),
                    'true_log10_N_plus1': float(np.log10(data['population'][0, cycle] + 1))})
    pd.DataFrame(summaries).to_csv(output / 'noise_summary.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(output / 'ned_noise_unmasked_sensitivity.csv', index=False)
    pd.DataFrame(abc_summaries).to_csv(output / 'abc_noise_summary.csv', index=False)
    pd.DataFrame(sample_records).to_csv(output / 'noise_example.csv', index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path('.'))
    parser.add_argument('--source-tables', type=Path, help='Optional verified exported frozen tables for local validation')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--noise-replicates', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260930)
    parser.add_argument('--preflight', action='store_true', help='Validate input schemas and noiseless anchors only')
    parser.add_argument('--no-noise', action='store_true', help='Noiseless and dense-sensitivity tables only')
    parser.add_argument('--overwrite', action='store_true', help='Explicitly replace a completed output directory')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    if args.bootstrap < 20 or args.noise_replicates < 1:
        parser.error('Use at least 20 bootstrap resamples and one noise replicate')
    data = read_inputs(args.repo.resolve(), args.source_tables.resolve() if args.source_tables else None)
    if args.preflight:
        anchors = []
        for budget in (4, 5, 6):
            for rule in RULES:
                row, _ = evaluate_counts(data, data['population'][:, SCANS[:budget]], SCANS[:budget],
                    rule, data['abc_ext'], 0, 20, args.seed)
                anchors.append(row)
        check_frozen(data, pd.DataFrame(anchors))
        print(json.dumps({'patients': len(data['ids']), 'references': len(data['ref_y']),
            'missing_cycle19_ids': data['ids'][np.isnan(data['population'][:, 19])].tolist(),
            'abc_full_cohort': float(data['abc_ext'].mean()), 'status': 'input validation passed'}))
        return
    output = args.output.resolve()
    if (output / 'COMPLETE.json').exists() and not args.overwrite:
        raise FileExistsError(f'Completed outputs already exist at {output}; use a fresh directory or --overwrite')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'COMPLETE.json').unlink(missing_ok=True)
    summary = run_noiseless(data, output, args.bootstrap, args.seed)
    print(summary[['rule', 'n_scans', 'n_triggered', 'reactive_extinction', 'abc_extinction_matched']].to_string(index=False), flush=True)
    if not args.no_noise:
        run_noise(data, output, args.noise_replicates, args.bootstrap, args.seed)
    if not args.no_plots:
        from plot_reactive import make_plots
        make_plots(output, with_noise=not args.no_noise)
    metadata = {'status': 'complete', 'schedules': {str(b): SCANS[:b] for b in (4, 5, 6)},
        'dense_sensitivity_horizons': [12, 19], 'fallback': False, 'missing_observations': 'remain missing',
        'switch_lookup': 't -> stored lag t+1', 'bootstrap_unit': 'patient; retain all noise draws',
        'bootstrap_replicates': args.bootstrap, 'noise_replicates': args.noise_replicates,
        'noise_seed': args.seed, 'noise_sigma_order': SIGMAS,
        'historical_early_noise_preserved': args.seed == 20260930 and args.noise_replicates == 100,
        'late_noise': 'independent SeedSequence([seed, cycle, sigma_index]) substreams for cycles 16 and 19',
        'NED_ABC': 'primary: below 10000 -> zero in both reference and query log10(N+1) prior to computing slopes; single measured biomarker; raw sensitivity on same triggers',
        'NED_early_below_limit_patients': int((data['population'][:, :4] < 10000).any(axis=1).sum()),
        'NED_early_below_limit_references': int((data['ref_x'] < np.log10(10001.)).any(axis=1).sum()),
        'noise_run': not args.no_noise,
        'input_sha256': {str(path): sha256(path) for path in data['input_paths'].values()},
        'code_sha256': {path.name: sha256(path) for path in Path(__file__).parent.glob('*.py')}}
    (output / 'COMPLETE.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
