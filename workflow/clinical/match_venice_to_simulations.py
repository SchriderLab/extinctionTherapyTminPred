from __future__ import annotations
from rescue.runtime import workspace_path, allocated_threads
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin({'true', '1', 't', 'yes'})

def clean_clinical(cloud: pd.DataFrame) -> pd.DataFrame:

    def pick(names):
        for n in names:
            if n in cloud.columns:
                return n
        return None
    dc = pick(['decline_rate_hat', 'd_hat', 'd'])
    qc = pick(['q_hat', 'growth_weight_hat', 'resistant_weight_hat', 'q'])
    gc = pick(['growth_rate_hat', 'g_hat', 'g'])
    if not (dc and qc and gc):
        raise ValueError('Could not identify clinical d/q/g columns.')
    out = cloud.copy().reset_index(drop=False).rename(columns={'index': 'clinical_source_row'})
    out['d'] = pd.to_numeric(out[dc], errors='coerce')
    out['q'] = pd.to_numeric(out[qc], errors='coerce')
    out['g'] = pd.to_numeric(out[gc], errors='coerce')
    keep = np.isfinite(out[['d', 'q', 'g']]).all(axis=1) & (out['q'] > 0)
    needed = {'fit_tmin_in_observed_window', 'growth_near_lower_bound', 'q_near_lower_bound'}
    if needed.issubset(out.columns):
        pathological = ~as_bool(out['fit_tmin_in_observed_window']) & (as_bool(out['growth_near_lower_bound']) | as_bool(out['q_near_lower_bound']))
        keep &= ~pathological
    out = out.loc[keep].copy().reset_index(drop=True)
    out.insert(0, 'clinical_match_id', np.arange(len(out), dtype=int))
    return out

def robust_space(df: pd.DataFrame, med: np.ndarray, scale: np.ndarray) -> np.ndarray:
    eps = 1e-12
    x = np.column_stack([pd.to_numeric(df['d'], errors='coerce').to_numpy(float), np.log10(np.maximum(pd.to_numeric(df['q'], errors='coerce').to_numpy(float), eps)), pd.to_numeric(df['g'], errors='coerce').to_numpy(float)])
    return (x - med) / scale

def weighted_summary(values: np.ndarray, distances: np.ndarray):
    values = np.asarray(values, float)
    distances = np.asarray(distances, float)
    ok = np.isfinite(values) & np.isfinite(distances)
    values = values[ok]
    distances = distances[ok]
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    w = 1.0 / np.maximum(distances, 1e-06)
    w /= w.sum()
    mean = float(np.sum(w * values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    return (mean, float(q25), float(q75))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clinical', type=Path, required=True)
    ap.add_argument('--sim-fits', type=Path, required=True)
    ap.add_argument('--outdir', type=Path, required=True)
    ap.add_argument('--scenario', default='broad_mix')
    ap.add_argument('--k', type=int, default=25)
    ap.add_argument('--internal-k', type=int, default=10)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    raw_clin = pd.read_csv(args.clinical)
    clin = clean_clinical(raw_clin)
    sim = pd.read_csv(args.sim_fits)
    required = {'patient_id', 'scenario', 'fit_success', 'd_hat', 'q_hat', 'g_hat', 'standing_frac', 'm1_effect_mean_target', 'm1_sdlog', 'patient_turnover', 'base_mut_rate', 'mean_age'}
    missing = required - set(sim.columns)
    if missing:
        raise ValueError(f'Simulation fits missing columns: {sorted(missing)}')
    sim = sim[(sim['scenario'] == args.scenario) & as_bool(sim['fit_success'])].copy()
    sim = sim[np.isfinite(sim[['d_hat', 'q_hat', 'g_hat']]).all(axis=1) & (sim['q_hat'] > 0)].copy()
    if len(sim) < max(args.k, args.internal_k + 1):
        raise ValueError(f'Too few usable simulations: {len(sim)}')
    sim = sim.rename(columns={'d_hat': 'd', 'q_hat': 'q', 'g_hat': 'g'}).reset_index(drop=True)
    eps = 1e-12
    C_raw = np.column_stack([clin['d'].to_numpy(float), np.log10(np.maximum(clin['q'].to_numpy(float), eps)), clin['g'].to_numpy(float)])
    med = np.median(C_raw, axis=0)
    scale = np.quantile(C_raw, 0.75, axis=0) - np.quantile(C_raw, 0.25, axis=0)
    scale = np.maximum(scale, 1e-06)
    C = robust_space(clin, med, scale)
    S = robust_space(sim, med, scale)
    tree_sim = cKDTree(S)
    kk_internal = min(args.internal_k + 1, len(sim))
    d_internal, _ = tree_sim.query(S, k=kk_internal)
    internal_radius = d_internal[:, -1]
    support_q90 = float(np.quantile(internal_radius, 0.9))
    support_q99 = float(np.quantile(internal_radius, 0.99))
    k = min(args.k, len(sim))
    dist, idx = tree_sim.query(C, k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    long_rows = []
    summary_rows = []
    param_cols = ['standing_frac', 'm1_effect_mean_target', 'm1_sdlog', 'patient_turnover', 'base_mut_rate', 'mean_age']
    for i in range(len(clin)):
        nearest_d = float(dist[i, 0])
        if nearest_d <= support_q90:
            support_class = 'well_supported'
        elif nearest_d <= support_q99:
            support_class = 'edge_of_support'
        else:
            support_class = 'outside_support'
        for rank in range(k):
            srow = sim.iloc[int(idx[i, rank])]
            long_rows.append({'clinical_match_id': int(clin.loc[i, 'clinical_match_id']), 'clinical_source_row': int(clin.loc[i, 'clinical_source_row']), 'clinical_d': float(clin.loc[i, 'd']), 'clinical_q': float(clin.loc[i, 'q']), 'clinical_g': float(clin.loc[i, 'g']), 'neighbor_rank': rank + 1, 'distance_scaled_d_log10q_g': float(dist[i, rank]), 'sim_patient_id': int(srow['patient_id']), 'sim_d': float(srow['d']), 'sim_q': float(srow['q']), 'sim_g': float(srow['g']), **{c: float(srow[c]) for c in param_cols}})
        near = sim.iloc[idx[i]].copy()
        row = {'clinical_match_id': int(clin.loc[i, 'clinical_match_id']), 'clinical_source_row': int(clin.loc[i, 'clinical_source_row']), 'clinical_d': float(clin.loc[i, 'd']), 'clinical_q': float(clin.loc[i, 'q']), 'clinical_g': float(clin.loc[i, 'g']), 'nearest_distance': nearest_d, 'support_class': support_class, 'nearest_sim_patient_id': int(near.iloc[0]['patient_id']), 'nearest_sim_d': float(near.iloc[0]['d']), 'nearest_sim_q': float(near.iloc[0]['q']), 'nearest_sim_g': float(near.iloc[0]['g'])}
        for c in param_cols:
            wmean, q25, q75 = weighted_summary(near[c].to_numpy(float), dist[i])
            row[f'{c}_local_weighted_mean'] = wmean
            row[f'{c}_local_q25'] = q25
            row[f'{c}_local_q75'] = q75
        summary_rows.append(row)
    matches = pd.DataFrame(long_rows)
    summary = pd.DataFrame(summary_rows)
    matches.to_csv(args.outdir / 'venice_topk_simulation_matches.csv', index=False)
    summary.to_csv(args.outdir / 'venice_patient_mechanistic_matches.csv', index=False)
    support_counts = summary['support_class'].value_counts().reindex(['well_supported', 'edge_of_support', 'outside_support'], fill_value=0).rename_axis('support_class').reset_index(name='n')
    support_counts['fraction'] = support_counts['n'] / len(summary)
    support_meta = pd.DataFrame([{'metric': 'n_clinical_patients', 'value': len(clin)}, {'metric': 'n_simulations_used', 'value': len(sim)}, {'metric': 'k_matches_per_patient', 'value': k}, {'metric': 'internal_k', 'value': args.internal_k}, {'metric': 'internal_radius_q90', 'value': support_q90}, {'metric': 'internal_radius_q99', 'value': support_q99}, {'metric': 'clinical_nearest_distance_median', 'value': float(summary['nearest_distance'].median())}, {'metric': 'clinical_nearest_distance_q90', 'value': float(summary['nearest_distance'].quantile(0.9))}, {'metric': 'clinical_nearest_distance_q95', 'value': float(summary['nearest_distance'].quantile(0.95))}, {'metric': 'clinical_nearest_distance_max', 'value': float(summary['nearest_distance'].max())}])
    support_counts.to_csv(args.outdir / 'support_class_summary.csv', index=False)
    support_meta.to_csv(args.outdir / 'support_thresholds_and_distances.csv', index=False)
    worst = summary.sort_values('nearest_distance', ascending=False).head(30)
    worst.to_csv(args.outdir / 'worst_supported_venice_patients.csv', index=False)
    print('\n=== MATCHING COMPLETE ===')
    print(f'Clinical patients retained: {len(clin)}')
    print(f'Broad-mix simulations used: {len(sim)}')
    print(f'Top-k matches per clinical patient: {k}')
    print('\n=== SUPPORT CLASSES ===')
    print(support_counts.to_string(index=False))
    print('\n=== CLINICAL -> SIM DISTANCE ===')
    print(summary['nearest_distance'].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).to_string())
    print('\n=== 15 WORST-SUPPORTED VENICE PATIENTS ===')
    cols = ['clinical_match_id', 'clinical_source_row', 'clinical_d', 'clinical_q', 'clinical_g', 'nearest_distance', 'support_class', 'nearest_sim_d', 'nearest_sim_q', 'nearest_sim_g']
    print(worst[cols].head(15).to_string(index=False))
if __name__ == '__main__':
    main()
