"""Paper figures driven exclusively by the newly generated source tables."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RULES = ('first_increase', 'progression_1.20', 'ned_10000')
LABELS = ('First observed increase', '20% progression', 'Recurrence after NED')
COLORS = ('#177d87', '#bd602e', '#7e5ba5')


def save(fig, output, name):
    fig.savefig(output / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(output / f'{name}.png', dpi=160, bbox_inches='tight')
    if name == 'Reactive_scan_density_sensitivity':
        fig.savefig(output / f'{name}.svg', bbox_inches='tight')
    plt.close(fig)


def error_line(ax, frame, xcol, ycol, color, label, marker='o', offset=0):
    x = frame[xcol].to_numpy(float) + offset
    y = frame[ycol].to_numpy(float)
    low = frame[f'{ycol}_ci_low'].to_numpy(float)
    high = frame[f'{ycol}_ci_high'].to_numpy(float)
    # Draw endpoints explicitly. A bootstrap percentile interval need not contain its point estimate.
    ax.vlines(x, low, high, colors=color, alpha=.55, linewidth=1.2)
    ax.plot(x, y, marker=marker, color=color, label=label, linewidth=1.8)


def make_plots(output, with_noise=True):
    output = Path(output)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    main = pd.read_csv(output / 'noiseless_summary.csv')
    dense = pd.read_csv(output / 'dense_summary.csv')
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=True, layout='constrained')
    for index, (rule, label) in enumerate(zip(RULES, LABELS)):
        row = main[main.rule == rule].sort_values('n_scans')
        ax = axes[index]
        error_line(ax, row, 'n_scans', 'reactive_extinction', COLORS[index], 'Reactive', offset=-.04)
        error_line(ax, row, 'n_scans', 'abc_extinction_matched', '#26394e', 'ABC, same patients', marker='s', offset=.04)
        for r in row.itertuples():
            ax.text(r.n_scans, 1.04, f'{r.n_triggered}/80 triggered', ha='center', fontsize=8)
        ax.set(title=f'{chr(65+index)}  {label}' + (' (masked ABC)' if rule == 'ned_10000' else ''), xticks=[4, 5, 6], ylim=(0, 1.14),
               xlabel='Number of scheduled scans')
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xlim(3.65, 6.35)
    axes[0].set_ylabel('Mean estimated extinction probability')
    axes[0].legend(loc='lower left', frameon=False, fontsize=8)
    fig.suptitle('Trigger-only switching on common scan schedules', fontsize=13)
    save(fig, figures, 'Reactive_common_scan_budgets')

    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7), layout='constrained')
    for i, horizon in enumerate((19, 12)):
        frame = dense[dense.horizon == horizon].sort_values('n_scans')
        error_line(axes[i, 0], frame, 'n_scans', 'reactive_extinction', COLORS[0], 'Reactive')
        error_line(axes[i, 0], frame, 'n_scans', 'abc_extinction_matched', '#26394e', 'ABC, same patients', 's')
        error_line(axes[i, 1], frame, 'n_scans', 'switch_delay', COLORS[0], 'Switch delay')
        axes[i, 2].plot(frame.n_scans, frame.trigger_fraction_all, 'o-', color=COLORS[0])
        for j, metric in enumerate(('Extinction at an observed trigger', 'Delay after the true nadir', 'Fraction of patients triggering')):
            axes[i, j].set(title=f'{chr(65+i*3+j)}  {metric}\nScans spread over cycles 0-{horizon}', xlabel='Number of scans')
        for ax in axes[i]:
            ax.set_xticks(range(2, horizon + 2, 2))
        axes[i, 0].set(ylim=(0, 1), ylabel='Mean extinction probability')
        axes[i, 1].set_ylabel('Simulation cycles')
        axes[i, 2].set(ylim=(0, 1.05), ylabel='Fraction of all 80 patients')
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle('Scan-density sensitivity with switching only after an observed increase', fontsize=13)
    save(fig, figures, 'Reactive_scan_density_sensitivity')

    if not with_noise:
        return
    noise = pd.read_csv(output / 'noise_summary.csv')
    abc = pd.read_csv(output / 'abc_noise_summary.csv')
    curves = pd.read_csv(output / 'observed_population.csv')
    example = curves[curves.patient_id == curves.patient_id.min()].dropna()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    ax = axes[0, 0]
    ax.plot(example.cycle, example.N, color='#26394e', label='Drug 1 population')
    scans = example[example.cycle.isin([0, 4, 8, 12, 16, 19])]
    ax.scatter(scans.cycle, scans.N, color='#26394e', zorder=3, label='Scheduled scans')
    ax.step(scans.cycle, 1.2*np.minimum.accumulate(scans.N.to_numpy()), where='post',
            linestyle='--', color=COLORS[1], label='1.20 x running observed minimum')
    ax.axhline(10000, linestyle='--', color=COLORS[2], label='NED threshold')
    ax.set(yscale='log', xlabel='Simulation cycle', ylabel='Simulated cells', title='A  Observed scans and recurrence thresholds')
    ax.legend(frameon=False, fontsize=7.5)
    ax = axes[0, 1]
    six = main[main.n_scans == 6].set_index('rule').loc[list(RULES)].reset_index()
    six['position'] = np.arange(3)
    error_line(ax, six, 'position', 'reactive_extinction', '#177d87', 'Reactive', offset=-.08)
    error_line(ax, six, 'position', 'abc_extinction_matched', '#26394e', 'ABC, same patients', 's', offset=.08)
    ax.set(xticks=range(3), xticklabels=['First increase', '20% progression', 'NED recurrence\n(masked ABC)'], ylim=(0, 1.12),
           title='B  Second-strike outcomes with six scans', ylabel='Mean extinction probability')
    for i, r in six.iterrows():
        ax.text(i, 1.04, f'n={r.n_triggered}', ha='center')
    ax.legend(frameon=False, fontsize=8, loc='lower left')
    for rule, label, color in zip(RULES, LABELS, COLORS):
        frame = noise[(noise.rule == rule) & (noise.n_scans == 6)].sort_values('sigma')
        axes[1, 0].plot(frame.sigma, frame.trigger_fraction_all, 'o-', color=color, label=label)
        axes[1, 1].plot(frame.sigma, frame.switch_delay, 'o-', color=color, label=label)
    axes[1, 0].set(title='C  Detection in the presence of noise', ylabel='Fraction of patient-realizations triggering', ylim=(0, 1.05))
    axes[1, 1].set(title='D  Delay of switch in the presence of noise', ylabel='Mean switch cycle minus true nadir')
    for ax in axes[1]:
        ax.set_xlabel('Noise SD on log10(N + 1)')
    axes[1, 0].legend(frameon=False, fontsize=8)
    fig.suptitle('Reactive thresholds with scans at cycles 0, 4, 8, 12, 16 and 19', fontsize=13)
    save(fig, figures, 'Reactive_thresholds_six_scans')

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8), layout='constrained')
    examples = pd.read_csv(output / 'noise_example.csv')
    examples = examples[np.isclose(examples.sigma, .10)]
    for b, color in zip((1, 2, 3), COLORS):
        frame = examples[examples.biomarker == b]
        axes[0, 0].plot(frame.cycle, frame.observed_log10_N_plus1, 'o-', color=color, label=f'Biomarker {b}', markersize=3, alpha=.8)
        ab = abc[abc.n_biomarkers == b].sort_values('sigma')
        axes[0, 1].plot(ab.sigma, ab.tmin_mae, 'o-', color=color, label=f'{b} biomarker(s)')
        axes[1, 0].plot(ab.sigma, ab.mean_extinction, 'o-', color=color, label=f'{b} biomarker(s)')
    truth = examples[examples.biomarker == 1]
    axes[0, 0].plot(truth.cycle, truth.true_log10_N_plus1, '--', color='black', label='True population')
    axes[0, 0].set(title='A  Independent noisy measurements (SD 0.10)', xlabel='Simulation cycle', ylabel='log10(N + 1)')
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].set(title='B  Nadir forecasting with additional biomarkers', ylabel='Mean absolute error (cycles)')
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[1, 0].set(title='C  Extinction with forecast-based switching', ylabel='Full-cohort mean extinction probability')
    for rule, label, color in zip(RULES, LABELS, COLORS):
        frame = noise[(noise.rule == rule) & (noise.n_scans == 6)].sort_values('sigma')
        error_line(axes[1, 1], frame, 'sigma', 'abc_minus_reactive', color, label + (' (masked ABC)' if rule == 'ned_10000' else ''))
    axes[1, 1].axhline(0, color='gray', linestyle='--')
    axes[1, 1].set(title='D  ABC versus reactive switching with six scans', ylabel='Matched extinction probability difference')
    axes[1, 1].legend(frameon=False, fontsize=8)
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.set_xlabel('Noise SD on log10(N + 1)')
    save(fig, figures, 'Noise_biomarkers_and_matched_policies')

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    for index, (rule, label) in enumerate(zip(RULES, LABELS)):
        for budget, color in zip((4, 5, 6), COLORS):
            frame = noise[(noise.rule == rule) & (noise.n_scans == budget)].sort_values('sigma')
            axes[0, index].plot(frame.sigma, frame.trigger_fraction_all, 'o-', color=color, label=f'{budget} scans')
            error_line(axes[1, index], frame, 'sigma', 'abc_minus_reactive', color, f'{budget} scans')
        axes[0, index].set(title=label, ylim=(0, 1.05))
        axes[1, index].axhline(0, color='gray', linestyle='--')
        if rule == 'ned_10000':
            axes[1, index].set_title('ABC inputs below 10,000 encoded as zero', fontsize=9)
        axes[1, index].set_xlabel('Noise SD on log10(N + 1)')
    axes[0, 0].set_ylabel('Fraction of patient-realizations triggering')
    axes[1, 0].set_ylabel('Matched ABC minus reactive extinction')
    axes[0, 0].legend(frameon=False)
    fig.suptitle('Observation-budget sensitivity under noise', fontsize=13)
    save(fig, figures, 'Reactive_noise_budget_sensitivity')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no-noise', action='store_true')
    args = parser.parse_args()
    make_plots(args.output, with_noise=not args.no_noise)
