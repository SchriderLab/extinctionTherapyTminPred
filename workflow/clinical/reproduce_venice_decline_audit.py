"""Reconstruct the four-decline VENICE screen and audit (not optimize) it.

Uses only the uploaded/raw ADSL and ADLB tables. No ABC reference is inferred,
no forecasting model is trained, and no future-outcome filter is applied.
Four declines = FIVE measurements. --declines 3 instead uses four measurements.

Example on Longleaf:
  python reproduce_venice_decline_audit.py --data-dir data/venice_csv --outdir venice_decline_audit
Example with the uploaded archive:
  python reproduce_venice_decline_audit.py --zip 'venice_csv(1).zip' --outdir venice_decline_audit
"""
from __future__ import annotations
from rescue.runtime import workspace_path, allocated_threads
import argparse
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint
from scipy.stats import spearmanr

def num(x):
    return pd.to_numeric(x, errors='coerce')

def load_tables(data_dir: Path | None, archive: Path | None):
    cols = ['USUBJID', 'LBTESTCD', 'LBSTRESN', 'ANLVISN', 'EPOCH', 'LBDY', 'LBDY2', 'LBDT', 'LBDTC', 'LBBLFL']
    if archive is not None:
        with zipfile.ZipFile(archive) as z:
            names = {Path(n).name.lower(): n for n in z.namelist() if not n.endswith('/')}
            a = pd.read_csv(z.open(names['adsl.csv']), low_memory=False)
            l = pd.read_csv(z.open(names['adlb.csv']), usecols=lambda c: c in cols, low_memory=False)
    else:
        a = pd.read_csv(data_dir / 'adsl.csv', low_memory=False)
        l = pd.read_csv(data_dir / 'adlb.csv', usecols=lambda c: c in cols, low_memory=False)
    for d in (a, l):
        d['USUBJID'] = d['USUBJID'].astype(str).str.strip()
    return (a, l)

def extract_scheduled(a, l):
    arm = a[a.RANDOM.astype(str).str.strip().str.upper().eq('Y') & a.ARMCD.astype(str).str.strip().str.upper().eq('PLACEBO')].copy()
    if arm.USUBJID.duplicated().any():
        raise ValueError('Duplicate subjects in comparator ADSL; resolve before analysis.')
    p = l[l.USUBJID.isin(arm.USUBJID) & l.LBTESTCD.astype(str).str.strip().str.upper().eq('PSA')].copy()
    p['psa'] = num(p.LBSTRESN)
    p['cycle'] = num(p.ANLVISN)
    p['study_day'] = num(p.LBDY2).fillna(num(p.LBDY)) if 'LBDY2' in p else num(p.LBDY)
    date_col = 'LBDT' if 'LBDT' in p else 'LBDTC'
    p['date'] = pd.to_datetime(p[date_col], errors='coerce')
    start = pd.to_datetime(arm.set_index('USUBJID')['TRTSTDT'], errors='coerce')
    p['calendar_day'] = (p['date'] - p.USUBJID.map(start)).dt.total_seconds() / 86400
    tx = p[p.EPOCH.astype(str).str.strip().str.lower().eq('treatment phase') & p.cycle.ge(1) & p.psa.gt(0) & np.isfinite(p.psa)].copy()
    tx['nominal_deviation'] = (tx.study_day - 21 * tx.cycle).abs().fillna(np.inf)
    tx = tx.sort_values(['USUBJID', 'cycle', 'nominal_deviation', 'study_day']).drop_duplicates(['USUBJID', 'cycle']).copy()
    tx['is_baseline'] = False
    b = arm[['USUBJID', 'TMPSA']].rename(columns={'TMPSA': 'psa'}).copy()
    b['psa'] = num(b.psa)
    b = b[b.psa.gt(0) & np.isfinite(b.psa)].copy()
    b['cycle'] = 0.0
    b['study_day'] = 0.0
    b['is_baseline'] = True
    matched = []
    for r in b.itertuples(index=False):
        cand = p[p.USUBJID.eq(r.USUBJID) & p.calendar_day.le(0) & np.isclose(p.psa, r.psa, rtol=1e-06, atol=1e-08)].sort_values('calendar_day')
        matched.append(float(cand.calendar_day.iloc[-1]) if len(cand) else np.nan)
    b['calendar_day'] = matched
    columns = ['USUBJID', 'cycle', 'psa', 'study_day', 'calendar_day', 'is_baseline']
    long = pd.concat([b[columns], tx[columns]], ignore_index=True)
    long = long.sort_values(['USUBJID', 'cycle']).reset_index(drop=True)
    return (arm, long, tx)

def first_window(y, n_declines):
    m = n_declines + 1
    return next((s for s in range(len(y) - m + 1) if np.all(np.diff(y[s:s + m]) < 0)), None)

def project_logconvex(t, z):
    m = len(t)
    diff = np.zeros((m - 1, m))
    for j in range(m - 1):
        diff[j, j], diff[j, j + 1] = (-1.0, 1.0)
    secants = diff / np.diff(t)[:, None]
    convex = secants[1:] - secants[:-1]
    initial = z[0] + (z[-1] - z[0]) * (t - t[0]) / (t[-1] - t[0])
    opt = minimize(lambda v: 0.5 * np.sum((v - z) ** 2), initial, jac=lambda v: v - z, method='SLSQP', constraints=[LinearConstraint(diff, -np.inf, 0), LinearConstraint(convex, 0, np.inf)], options={'ftol': 1e-12, 'maxiter': 300})
    if not opt.success:
        raise RuntimeError('Log-convex projection failed: ' + opt.message)
    return opt.x

def describe_series(s):
    return {str(k): float(v) for k, v in s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).items()}

def audit_windows(long, n_declines):
    early_rows, outcome_rows, windows, full_rows = ([], [], [], [])
    for pid, g in long.groupby('USUBJID', sort=True):
        g = g.sort_values('cycle').reset_index(drop=True)
        y = g.psa.to_numpy(float)
        s = first_window(y, n_declines)
        if s is None:
            continue
        e = s + n_declines
        f = g.iloc[s:e + 1].copy()
        t = f.calendar_day.to_numpy(float)
        actual = np.isfinite(t).all() and np.all(np.diff(t) > 0)
        if not actual:
            t = f.cycle.to_numpy(float) * 21
        t = (t - t[0]) / 21
        z = np.log(f.psa.to_numpy(float) / f.psa.iloc[0])
        sl = np.diff(z) / np.diff(t)
        zz = project_logconvex(t, z)
        linear = np.polyval(np.polyfit(t, z, 1), t)
        row = dict(id=pid, anchor_index=s, forecast_index=e, n_input_scans=n_declines + 1, anchor_cycle=float(g.cycle.iloc[s]), forecast_cycle=float(g.cycle.iloc[e]), anchor_calendar_day=float(g.calendar_day.iloc[s]), forecast_calendar_day=float(g.calendar_day.iloc[e]), anchor_psa=float(y[s]), forecast_psa=float(y[e]), fold_decline=float(y[s] / y[e]), log_drop=float(z[0] - z[-1]), prior_record_low_psa=float(np.min(y[:e])), new_record_low_at_forecast=bool(y[e] < np.min(y[:e])), at_running_min_at_forecast=bool(y[e] <= np.min(y[:e])), cumulative_rebound_from_past_min_at_forecast=float(y[e] / np.min(y[:e])), baseline_in_window=bool(f.is_baseline.any()), skipped_analysis_cycles=bool(np.any(np.diff(f.cycle.to_numpy(float)) > 1)), nominal_span=float(f.cycle.iloc[-1] - f.cycle.iloc[0]), time_mode='actual' if actual else 'nominal_only', span_equiv=float(t[-1]), previous_log_slope=float(sl[-2]), recent_log_slope=float(sl[-1]), slope_change=float(sl[-1] - sl[-2]), logconvex_exact=bool(np.all(np.diff(sl) >= -1e-12)), logconvex_rmse=float(np.sqrt(np.mean((zz - z) ** 2))), linear_log_rmse=float(np.sqrt(np.mean((linear - z) ** 2))))
        early_rows.append(row)
        j = int(np.argmin(y))
        equal = g[g.psa.eq(y[j])]
        after = g.iloc[j + 1:]
        outcome_rows.append(dict(id=pid, n_total_records=len(g), n_after_forecast=len(g) - e - 1, global_min_cycle=float(g.cycle.iloc[j]), global_min_psa=float(y[j]), global_min_calendar_day=float(g.calendar_day.iloc[j]), global_min_after_forecast=bool(j > e), global_min_at_forecast=bool(j == e), global_min_before_forecast=bool(j < e), global_min_at_last_record=bool(j == len(g) - 1), global_min_remaining_nominal_cycles=float(g.cycle.iloc[j] - g.cycle.iloc[e]), global_min_remaining_days=float(g.calendar_day.iloc[j] - g.calendar_day.iloc[e]), global_min_remaining_cycle_equiv=float((g.calendar_day.iloc[j] - g.calendar_day.iloc[e]) / 21), n_after_global_min=len(after), two_higher_after_global_min=bool((after.psa > y[j]).sum() >= 2), n_global_min_ties=len(equal), global_min_first_cycle=float(equal.cycle.min()), global_min_last_cycle=float(equal.cycle.max()), global_min_tie_span=float(equal.cycle.max() - equal.cycle.min())))
        f['id'] = pid
        f['input_position'] = np.arange(n_declines + 1)
        f['t_since_anchor_cycle_equiv'] = t
        f['t_since_forecast_cycle_equiv'] = t - t[-1]
        f['relative_psa'] = f.psa / y[s]
        f['log_relative_psa'] = z
        f['logconvex_projection'] = zz
        windows.append(f)
        h = g.copy()
        h['id'] = pid
        h['in_selected_input_window'] = (h.index >= s) & (h.index <= e)
        h['nominal_cycles_since_forecast'] = h.cycle - g.cycle.iloc[e]
        h['actual_cycle_equiv_since_forecast'] = (h.calendar_day - g.calendar_day.iloc[e]) / 21
        h['relative_psa_to_anchor'] = h.psa / y[s]
        full_rows.append(h)
    if not early_rows:
        raise ValueError('No eligible windows were found.')
    return (pd.DataFrame(early_rows), pd.DataFrame(outcome_rows), pd.concat(windows), pd.concat(full_rows))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('data/venice_csv'))
    p.add_argument('--zip', type=Path, default=None)
    p.add_argument('--outdir', type=Path, default=Path('venice_decline_audit'))
    p.add_argument('--declines', type=int, default=4, choices=[3, 4])
    a = p.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    adsl, adlb = load_tables(a.data_dir, a.zip)
    arm, long, tx = extract_scheduled(adsl, adlb)
    early, outcomes, windows, full = audit_windows(long, a.declines)
    merged = early.merge(outcomes, on='id', validate='one_to_one')
    scan_counts = long.groupby('USUBJID').size().value_counts().sort_index()
    post_gaps = windows[~windows.is_baseline].sort_values(['id', 'calendar_day']).groupby('id').calendar_day.diff().dropna()
    summary = dict(randomized_comparator_n=len(arm), patients_with_plotted_psa_n=long.USUBJID.nunique(), n_declines=a.declines, n_input_scans=a.declines + 1, selected_patients=len(early), eligible_four_scan_three_decline_n=sum((first_window(g.psa.to_numpy(), 3) is not None for _, g in long.groupby('USUBJID'))), eligible_five_scan_four_decline_n=sum((first_window(g.psa.to_numpy(), 4) is not None for _, g in long.groupby('USUBJID'))), forecast_by_nominal_cycle6_n=int((early.forecast_cycle <= 6).sum()), new_record_low_at_forecast_n=int(early.new_record_low_at_forecast.sum()), not_new_record_low_at_forecast_n=int((~early.new_record_low_at_forecast).sum()), new_record_low_by_nominal_cycle6_n=int((early.new_record_low_at_forecast & early.forecast_cycle.le(6)).sum()), skipped_nominal_cycles_n=int(early.skipped_analysis_cycles.sum()), no_measurement_after_forecast_n=int((outcomes.n_after_forecast == 0).sum()), global_min_at_last_record_n=int(outcomes.global_min_at_last_record.sum()), global_min_at_forecast_n=int(outcomes.global_min_at_forecast.sum()), global_min_before_forecast_n=int(outcomes.global_min_before_forecast.sum()), global_min_future_interior_n=int((outcomes.global_min_after_forecast & ~outcomes.global_min_at_last_record).sum()), global_min_future_two_higher_later_n=int((outcomes.global_min_after_forecast & outcomes.two_higher_after_global_min).sum()), tied_global_minima_n=int((outcomes.n_global_min_ties > 1).sum()), exact_logconvex_n=int(early.logconvex_exact.sum()), nominal_time_only_n=int((early.time_mode == 'nominal_only').sum()), post_treatment_input_gaps_days=describe_series(post_gaps), post_treatment_input_gaps_fraction_18_24=float(post_gaps.between(18, 24).mean()), forecast_psa=describe_series(early.forecast_psa), fold_decline=describe_series(early.fold_decline), logconvex_projection_rmse=describe_series(early.logconvex_rmse), original_pdf_scan_count_frequencies={str(k): int(v) for k, v in scan_counts.items()}, caveat='Descriptive reconstruction of full comparator arm. Not a validated prediction filter. Original decline4 local outputs were not supplied. No ABC accuracy computed. Outcome audit fields must never be used to determine forecast-time inclusion.')
    for name, df in [('selected_early_features.csv', early), ('selected_outcome_audit.csv', outcomes), ('selected_input_windows.csv', windows), ('selected_full_trajectories.csv', full), ('all_scheduled_psa.csv', long)]:
        df.to_csv(a.outdir / name, index=False)
    (a.outdir / 'descriptive_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    interior = merged[merged.global_min_after_forecast & ~merged.global_min_at_last_record]
    corr = []
    for c in ['anchor_psa', 'forecast_psa', 'log_drop', 'logconvex_rmse', 'linear_log_rmse', 'slope_change']:
        r = spearmanr(interior[c], interior.global_min_remaining_cycle_equiv)
        corr.append(dict(feature=c, n=len(interior), Spearman_with_remaining_actual_cycle_equiv=float(r.statistic), p_unadjusted_exploratory=float(r.pvalue)))
    pd.DataFrame(corr).to_csv(a.outdir / 'exploratory_associations_in_future_interior_subset.csv', index=False)
    print(json.dumps(summary, indent=2))
    print('\nWROTE', a.outdir.resolve())
if __name__ == '__main__':
    main()
