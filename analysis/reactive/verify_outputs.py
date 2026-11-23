"""Check the paper's locked run and write the Overleaf replacements."""
import argparse
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd

from run_reactive import sha256

HERE = Path(__file__).resolve().parent
CHECKS = HERE / 'checks'


def compare(actual, expected, keys, cols):
    a = actual.set_index(keys, verify_integrity=True).sort_index()
    e = expected.set_index(keys, verify_integrity=True).sort_index()
    if not a.index.equals(e.index):
        raise ValueError(f'Row keys changed: {keys}')
    for col in cols:
        np.testing.assert_allclose(a[col], e[col], rtol=0, atol=1e-8, equal_nan=True, err_msg=col)


def verify(out):
    meta = json.loads((out / 'COMPLETE.json').read_text())
    for key, target in dict(noise_seed=20260930, bootstrap_replicates=2000, noise_replicates=100,
                            NED_early_below_limit_patients=7, NED_early_below_limit_references=1267,
                            fallback=False, noise_run=True).items():
        if meta[key] != target:
            raise ValueError(f'Unexpected {key}: {meta[key]}')
    expected_hashes = json.loads((CHECKS/'input_sha256.json').read_text())
    computed = {}
    for key, value in meta['input_sha256'].items():
        p = Path(key)
        rel = next((name for name in expected_hashes if p.as_posix().endswith('/'+name)), None)
        if rel is None or rel in computed:
            raise ValueError(f'Unexpected or duplicate frozen input: {p}')
        computed[rel] = value
    if computed != expected_hashes:
        raise ValueError('Frozen input checksums differ from the audited 84 inputs')
    keys = ['sigma','rule','n_scans']
    for kind in ('noise','noiseless'):
        base = pd.read_csv(CHECKS/f'{kind}_summary.csv')
        primary = pd.read_csv(out/f'{kind}_summary.csv')
        exact = pd.read_csv(out/f'ned_{kind}_unmasked_sensitivity.csv')
        others = primary.rule != 'ned_10000'
        if not (primary.loc[~others,'abc_inputs']=='below_10000_encoded_zero').all():
            raise ValueError('NED primary variant is mislabeled')
        original = pd.concat([primary[others], exact])
        nums = base.select_dtypes('number').columns.difference(keys)
        compare(original, base, keys, nums)
        compare(primary, base, keys, [c for c in nums if not c.startswith('abc_')])
    ab = pd.read_csv(CHECKS/'abc_noise_summary.csv')
    compare(pd.read_csv(out/'abc_noise_summary.csv'), ab, ['sigma','n_biomarkers'], ['mean_extinction','tmin_mae'])
    replay = pd.read_csv(CHECKS/'ned_mask_replay.csv')
    primary = pd.read_csv(out/'noise_summary.csv').query("rule == 'ned_10000'").rename(columns={
        'n_scans':'budget','n_triggered':'triggered','abc_extinction_matched':'abc_masked','reactive_extinction':'reactive'})
    compare(primary,replay,['sigma','budget'],['triggered','abc_masked','reactive'])
    no = pd.read_csv(out/'noiseless_summary.csv')
    six = no[(no.rule=='ned_10000') & (no.n_scans==6)].iloc[0]
    np.testing.assert_allclose([six.abc_minus_reactive_ci_low,six.abc_minus_reactive_ci_high],
                               [.31963487332339796,.5232198660714286],atol=1e-10,rtol=0)
    return no, primary


def render(out, no, noise):
    def display(value):
        return format(Decimal(f"{value:.12f}").quantize(Decimal("0.001"), rounding=ROUND_HALF_UP), "f")

    files = []
    names = {'first_increase':'First increase','progression_1.20':r'20\% progression','ned_10000':'NED recurrence'}
    lines = [r'\begin{table}[p]',r'\centering\small\color{purple}',
             r'\caption{\textbf{Extinction under reactive and ABC-guided switching.} Four scans covered cycles 0, 4, 8, and 12. The fifth added cycle 16 and the sixth added cycle 19. Each row includes only patients who triggered that rule. For NED comparisons, ABC encoded early measurements below 10,000 cells as zero in patients and references. Intervals are paired 95\% patient-bootstrap intervals for ABC minus reactive extinction.}',
             r'\label{tab:reactivecommon}',r'\begin{tabular}{@{}lrrrrl@{}}',r'\toprule',
             r'Rule & Scans & Triggered & Reactive & Matched ABC & Difference [95\% interval]\\',r'\midrule']
    for rule, label in names.items():
        for r in no[no.rule==rule].sort_values('n_scans').itertuples():
            lines.append(f'{label} & {r.n_scans} & {r.n_triggered}/80 & {r.reactive_extinction:.3f} & '
                         f'{display(r.abc_extinction_matched)} & ${display(r.abc_minus_reactive)}\\ [{r.abc_minus_reactive_ci_low:.3f},{r.abc_minus_reactive_ci_high:.3f}]$'+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (out/'TableS2.tex').write_text('\n'.join(lines)+'\n')
    r=no[(no.rule=='ned_10000') & (no.n_scans==6)].iloc[0]
    z=noise[(noise.sigma==.1) & (noise.budget==6)].iloc[0]
    exact=pd.read_csv(out/'ned_noiseless_unmasked_sensitivity.csv').query('n_scans == 6').iloc[0]
    notes = [
        '% Methods, reactive and noise comparisons',
        r'For the NED comparison, we encoded each early count below 10,000 cells as zero in both the ABC reference trajectories and the patient observations before computing levels and slopes. With observation noise, we applied the threshold to the measured count. We repeated the comparison retaining the numerical values of subthreshold observations in the same patients and noise realizations.',
        '% Results, early detectability and six scheduled scans',
        'During cycles 0--3, seven of the 80 patients and 1,267 of the 9,983 reference trajectories had a count below 10,000 cells. In the noiseless data all such observations occurred at cycle 3.', 
        f'Sixty of 80 patients met the NED recurrence criterion. Among these patients, mean extinction probability was {r.abc_extinction_matched:.3f} for ABC and {r.reactive_extinction:.3f} for reactive switching (paired difference {r.abc_minus_reactive:.3f}, 95\\% interval [{r.abc_minus_reactive_ci_low:.3f}, {r.abc_minus_reactive_ci_high:.3f}]). Retaining exact subthreshold counts gave {exact.abc_extinction_matched:.3f} for ABC.',
        '% Noise results, six scheduled scans',
        f'At noise SD 0.10, NED recurrence was detected in {int(z.triggered):,} of 8,000 patient-realizations. Within these realizations, mean extinction was {z.abc_masked:.3f} for ABC and {z.reactive:.3f} for reactive switching (paired difference {z.abc_minus_reactive:.3f}, 95\\% interval [{z.abc_minus_reactive_ci_low:.3f}, {z.abc_minus_reactive_ci_high:.3f}]).',
        '% Add to captions of Figures 7, 10, S23 and S24 wherever matched NED outcomes appear.',
        'In the matched NED comparison, ABC encoded early observations below 10,000 cells as zero in both patients and references.']
    (out/'NED_changes.tex').write_text('\n\n'.join(notes)+'\n')
    figs = ['Reactive_thresholds_six_scans','Noise_biomarkers_and_matched_policies',
            'Reactive_common_scan_budgets','Reactive_noise_budget_sensitivity','Reactive_scan_density_sensitivity']
    files += [out/'figures'/f'{name}.pdf' for name in figs]
    files += [out/name for name in ('TableS2.tex','NED_changes.tex','noiseless_summary.csv','noise_summary.csv',
             'ned_noiseless_unmasked_sensitivity.csv','ned_noise_unmasked_sensitivity.csv','abc_noise_summary.csv',
             'fallback_diagnostic.csv','COMPLETE.json')]
    verified = out/'VERIFIED.json'
    verified.write_text(json.dumps({'status':'PASS','input_files':84,'masked_noise_comparisons':15,
        'noise_instances_per_setting':8000,'sha256':{str(p.relative_to(out)):sha256(p) for p in files}},indent=2)+'\n')
    with ZipFile(out/'paper_reactive_update.tmp','w',ZIP_DEFLATED) as bundle:
        for f in files+[verified]:
            bundle.write(f,f.relative_to(out))
    (out/'paper_reactive_update.tmp').replace(out/'paper_reactive_update.zip')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    root=parser.parse_args().output
    render(root,*verify(root))
    print('PASS:',root/'paper_reactive_update.zip')
