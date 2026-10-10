from rescue.runtime import workspace_path, allocated_threads
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = workspace_path('')
DATA = ROOT / 'data'
VENICE = DATA / 'venice_csv'
OUT = DATA / 'source_data'
OUT.mkdir(parents=True, exist_ok=True)

def find_csv(required):
    hits = []
    for p in VENICE.rglob('*.csv'):
        try:
            h = pd.read_csv(p, nrows=3, low_memory=False)
        except Exception:
            continue
        if set(required).issubset(h.columns):
            hits.append(p)
    if not hits:
        raise FileNotFoundError(f'No CSV found containing columns: {required}')
    hits.sort(key=lambda p: ('adsl' not in p.name.lower() and 'adlb' not in p.name.lower(), len(str(p))))
    return hits[0]
ADSL_PATH = find_csv(['USUBJID', 'TMPSA'])
ADLB_PATH = find_csv(['USUBJID', 'LBTESTCD', 'LBSTRESN', 'ANLVISN'])
print('ADSL:', ADSL_PATH)
print('ADLB:', ADLB_PATH)
adsl = pd.read_csv(ADSL_PATH, low_memory=False)
adlb = pd.read_csv(ADLB_PATH, low_memory=False)
adsl['USUBJID'] = adsl['USUBJID'].astype(str).str.strip()
adlb['USUBJID'] = adlb['USUBJID'].astype(str).str.strip()
adsl['psa0'] = pd.to_numeric(adsl['TMPSA'], errors='coerce')
baseline = adsl.loc[adsl['psa0'].notna() & (adsl['psa0'] > 0), ['USUBJID', 'psa0']].drop_duplicates('USUBJID')
mask_psa = adlb['LBTESTCD'].astype(str).str.upper().str.contains('PSA', na=False)
lab = adlb.loc[mask_psa].copy()
if 'EPOCH' not in lab.columns:
    raise RuntimeError('EPOCH column missing from VENICE ADLB.')
lab = lab[lab['EPOCH'].astype(str).str.strip().str.casefold().eq('treatment phase')].copy()
lab['cycle'] = pd.to_numeric(lab['ANLVISN'], errors='coerce')
lab['psa'] = pd.to_numeric(lab['LBSTRESN'], errors='coerce')
if 'LBDY2' in lab.columns:
    d2 = pd.to_numeric(lab['LBDY2'], errors='coerce')
else:
    d2 = pd.Series(np.nan, index=lab.index)
if 'LBDY' in lab.columns:
    d1 = pd.to_numeric(lab['LBDY'], errors='coerce')
else:
    d1 = pd.Series(np.nan, index=lab.index)
lab['day'] = d2.where(d2.notna(), d1)
lab = lab.loc[lab['cycle'].notna() & lab['psa'].notna() & (lab['psa'] > 0)].copy()
lab['cycle'] = lab['cycle'].astype(int)
lab = lab.loc[lab['cycle'] >= 1].copy()
lab['nominal_day'] = 21.0 * lab['cycle']
lab['day_error'] = (lab['day'] - lab['nominal_day']).abs()
lab['day_error_sort'] = lab['day_error'].fillna(np.inf)
lab = lab.sort_values(['USUBJID', 'cycle', 'day_error_sort', 'day']).drop_duplicates(['USUBJID', 'cycle'], keep='first').sort_values(['USUBJID', 'cycle']).reset_index(drop=True)
wide = lab.pivot(index='USUBJID', columns='cycle', values='psa')
early_ids = wide.index[wide.reindex(columns=[1, 2, 3]).notna().all(axis=1)].tolist()
early_ids = sorted(set(early_ids) & set(baseline['USUBJID']))
print('\n' + '=' * 75)
print('EARLY WINDOW')
print('Complete baseline + cycles 1-3:', len(early_ids))
early_lab = lab[lab['USUBJID'].isin(early_ids)].copy()
post3 = early_lab[early_lab['cycle'] > 3]
has_post3 = set(post3['USUBJID'])
print('Complete early + any post-cycle3 PSA:', len(has_post3))
baseline_map = baseline.set_index('USUBJID')['psa0'].to_dict()
locked_rows = []
for pid in early_ids:
    g = early_lab[early_lab['USUBJID'] == pid].sort_values('cycle').reset_index(drop=True)
    if pid not in baseline_map:
        continue
    baseline_psa = float(baseline_map[pid])
    cycles = np.concatenate([[0.0], g['cycle'].to_numpy(float)])
    values = np.concatenate([[baseline_psa], g['psa'].to_numpy(float)])
    i_min = int(np.nanargmin(values))
    tmin = float(cycles[i_min])
    nadir_psa = float(values[i_min])
    has_later = bool(np.any(cycles > tmin))
    future_interior_nadir = bool(tmin > 3 and has_later)
    if not future_interior_nadir:
        continue
    locked_rows.append({'USUBJID': pid, 'observed_nadir_cycle': tmin, 'observed_nadir_psa': nadir_psa, 'has_measurement_after_nadir': has_later, 'last_observed_cycle': float(np.nanmax(cycles))})
locked = pd.DataFrame(locked_rows)
print('\n' + '=' * 75)
print('LOCKED FORECASTING COHORT')
print('Locked n:', len(locked))
target_rows = []
for pid in locked['USUBJID']:
    g = early_lab[early_lab['USUBJID'] == pid].sort_values('cycle').reset_index(drop=True)
    result = {'USUBJID': pid, 'target_available': False, 'target_t': np.nan, 'target_cycle': np.nan, 'target_psa': np.nan, 'previous_cycle': np.nan, 'previous_psa': np.nan, 'next_cycle': np.nan, 'next_psa': np.nan, 'target_day': np.nan, 'n_scheduled_psa': len(g)}
    for i in range(1, len(g) - 1):
        cyc = int(g.loc[i, 'cycle'])
        if cyc <= 3:
            continue
        prev_psa = float(g.loc[i - 1, 'psa'])
        this_psa = float(g.loc[i, 'psa'])
        next_psa = float(g.loc[i + 1, 'psa'])
        if this_psa <= prev_psa and next_psa > this_psa:
            result.update({'target_available': True, 'target_t': cyc, 'target_cycle': cyc, 'target_psa': this_psa, 'previous_cycle': int(g.loc[i - 1, 'cycle']), 'previous_psa': prev_psa, 'next_cycle': int(g.loc[i + 1, 'cycle']), 'next_psa': next_psa, 'target_day': float(g.loc[i, 'day']) if pd.notna(g.loc[i, 'day']) else np.nan})
            break
    target_rows.append(result)
targets = pd.DataFrame(target_rows)
avail = targets[targets['target_available']].copy()
print('\n' + '=' * 75)
print('TARGET')
print('Target available:', len(avail))
print('Target unavailable:', len(targets) - len(avail))
if len(avail):
    print('\nTarget distribution:')
    print(avail['target_t'].describe().to_string())
    print('\nHistogram:')
    print(avail['target_t'].astype(int).value_counts().sort_index().to_string())
    print('\nFraction target <= 12:', float((avail['target_t'] <= 12).mean()))
expected = {'early_complete': 506, 'any_post3': 482, 'locked': 268, 'target_available': 263}
observed = {'early_complete': len(early_ids), 'any_post3': len(has_post3), 'locked': len(locked), 'target_available': len(avail)}
print('\n' + '=' * 75)
print('FROZEN SANITY CHECK')
ok = True
for k in expected:
    status = 'OK' if observed[k] == expected[k] else 'FAIL'
    print(f'{k:20s} observed={observed[k]:4d} expected={expected[k]:4d} {status}')
    if observed[k] != expected[k]:
        ok = False
if not ok:
    print('\nSTOP: frozen VENICE counts were not reproduced.')
    print('Do NOT run MAINSAIL external validation yet.')
    if 'EPOCH' in adlb.columns:
        print('\nADLB EPOCH values for PSA rows:')
        print(adlb.loc[mask_psa, 'EPOCH'].value_counts(dropna=False).to_string())
    print('\nANLVISN counts among PSA rows:')
    print(pd.to_numeric(adlb.loc[mask_psa, 'ANLVISN'], errors='coerce').value_counts().sort_index().to_string())
    raise SystemExit(1)
locked[['USUBJID']].to_csv(OUT / 'venice_locked_268_ids.csv', index=False)
targets.to_csv(OUT / 'venice_first_rebound_nadirs.csv', index=False)
lab.to_csv(OUT / 'venice_psa_scheduled_long.csv', index=False)
print('\nALL FROZEN VENICE CHECKS PASSED.')
print('\nWrote:')
print(OUT / 'venice_locked_268_ids.csv')
print(OUT / 'venice_first_rebound_nadirs.csv')
print(OUT / 'venice_psa_scheduled_long.csv')
