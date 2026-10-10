from __future__ import annotations
from rescue.runtime import workspace_path, allocated_threads
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
SCAN_TIMES = np.array([0.0, 1.0, 2.0, 3.0])
EPS = 1e-12

def extract_one(path: Path, manifest_row: pd.Series) -> dict | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        x = pd.read_csv(path)
    except Exception:
        return None
    if not {'t_treatment', 'N'}.issubset(x.columns) or len(x) < 4:
        return None
    z = pd.DataFrame({'t': pd.to_numeric(x['t_treatment'], errors='coerce'), 'N': pd.to_numeric(x['N'], errors='coerce')}).dropna().sort_values('t').drop_duplicates('t', keep='last')
    if len(z) < 4:
        return None
    vals = []
    for t in SCAN_TIMES:
        q = z[np.isclose(z['t'], t, atol=1e-09)]
        if len(q) != 1:
            return None
        vals.append(float(q.iloc[0]['N']))
    if vals[0] <= 0 or any((v < 0 for v in vals)):
        return None
    y = np.log((np.asarray(vals, float) + EPS) / (vals[0] + EPS))
    n_arr = z['N'].to_numpy(float)
    t_arr = z['t'].to_numpy(float)
    idx = int(np.argmin(n_arr))
    tmin = float(t_arr[idx])
    tmin_boundary = int(idx == len(z) - 1)
    final_status = str(x.iloc[-1].get('status', ''))
    out = {'ref_id': int(manifest_row.patient_id), 'patient_id': int(manifest_row.patient_id), 't_min_true': tmin, 'tmin_boundary': tmin_boundary, 'future_nadir_after_scan3': int(tmin > 3.0), 'final_status': final_status, 'final_t': float(t_arr[-1]), 'N0': vals[0], 'N1': vals[1], 'N2': vals[2], 'N3': vals[3], 'y0': float(y[0]), 'y1': float(y[1]), 'y2': float(y[2]), 'y3': float(y[3]), 'dy01': float(y[1] - y[0]), 'dy12': float(y[2] - y[1]), 'dy23': float(y[3] - y[2])}
    for c in ['target_decay_rate', 'base_mut_rate', 'dose_drug_1_mono', 'patient_turnover', 'standing_target', 'm1_effect_mean_target', 'm1_meanlog', 'm1_sdlog', 'single_age', 'n_founders', 'patient_seed_base']:
        if c in manifest_row.index:
            out[c] = manifest_row[c]
    if 'standing_realized' in x.columns:
        out['standing_realized'] = float(pd.to_numeric(x['standing_realized'], errors='coerce').iloc[0])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--library', type=Path, required=True)
    args = ap.parse_args()
    manifest = pd.read_csv(args.library / 'reference_manifest.csv')
    rows = []
    missing = 0
    for j, r in manifest.iterrows():
        pid = int(r.patient_id)
        path = args.library / 'raw' / f'patient_{pid:05d}' / 'trajectory.csv'
        q = extract_one(path, r)
        if q is None:
            missing += 1
        else:
            rows.append(q)
        if (j + 1) % 1000 == 0:
            print(f'processed {j + 1}/{len(manifest)} usable={len(rows)} missing={missing}', flush=True)
    out = pd.DataFrame(rows)
    out_path = args.library / 'abc_reference_features_usable.csv'
    out.to_csv(out_path, index=False)
    print(f'WROTE {out_path}')
    print(f'usable: {len(out):,}/{len(manifest):,}')
    if len(out):
        print('boundary nadirs:', int(out.tmin_boundary.sum()))
        clean = out[out.tmin_boundary != 1]
        print('clean interior nadirs:', len(clean))
        print('future nadir > scan3 among clean:', int((clean.t_min_true > 3).sum()))
        print('tmin q05/median/q95:', np.quantile(clean.t_min_true, [0.05, 0.5, 0.95]) if len(clean) else 'NA')
if __name__ == '__main__':
    main()
