from rescue.runtime import workspace_path, allocated_threads
from pathlib import Path
import re
import numpy as np
import pandas as pd
ROOT = workspace_path('data/mainsail_pc002')
OUT = workspace_path('data/source_data/mainsail')
OUT.mkdir(parents=True, exist_ok=True)
LAB = ROOT / 'lab.sas7bdat'
df = pd.read_sas(LAB, format='sas7bdat', encoding='latin1')
psa = df[df['TST_NAME'].astype(str).str.upper().eq('PSA')].copy()
print('Raw PSA rows:', len(psa))
print('Raw PSA patients:', psa['RPT'].nunique())

def parse_psa(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    if s.startswith('<'):
        try:
            return float(s[1:].strip())
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan
psa['psa'] = psa['US_RES'].apply(parse_psa)
print('\nNumeric PSA rows:', psa['psa'].notna().sum())
print('Dropped nonnumeric:', psa['psa'].isna().sum())
print('\nRaw nonnumeric values:')
print(psa.loc[psa['psa'].isna(), 'US_RES'].value_counts(dropna=False).head(20).to_string())
print('\nBelow-limit values:')
print(psa.loc[psa['US_RES'].astype(str).str.startswith('<', na=False), 'US_RES'].value_counts().to_string())
psa = psa[psa['psa'].notna() & (psa['psa'] > 0)].copy()
pat = re.compile('^Cycle\\s+(\\d+)\\s+Day\\s+1$', flags=re.IGNORECASE)

def get_cycle(x):
    m = pat.match(str(x).strip())
    if not m:
        return np.nan
    return int(m.group(1))
psa['cycle'] = psa['VISNAME'].apply(get_cycle)
sched = psa[psa['cycle'].notna()].copy()
sched['cycle'] = sched['cycle'].astype(int)
print('\nScheduled Cycle N Day 1 rows:', len(sched))
print('Scheduled patients:', sched['RPT'].nunique())
print('\nScheduled PSA counts by cycle:')
print(sched.groupby('cycle')['RPT'].nunique().sort_index().to_string())
dup = sched.groupby(['RPT', 'cycle']).size().reset_index(name='n')
print('\nPatient-cycle duplicates:')
print('groups with >1 row:', int((dup['n'] > 1).sum()))
print('max rows in a patient-cycle:', int(dup['n'].max()))
c1 = sched[sched['cycle'] == 1].copy()
c1 = c1.sort_values(['RPT', 'LABDAY']).drop_duplicates('RPT', keep='first')[['RPT', 'LABDAY']].rename(columns={'LABDAY': 'day0'})
sched = sched.merge(c1, on='RPT', how='left')
sched['expected_day'] = sched['day0'] + 21.0 * (sched['cycle'] - 1)
sched['day_error'] = (sched['LABDAY'] - sched['expected_day']).abs()
sched = sched.sort_values(['RPT', 'cycle', 'day_error', 'LABDAY']).drop_duplicates(['RPT', 'cycle'], keep='first').sort_values(['RPT', 'cycle'])
sched['t'] = sched['cycle'] - 1
sched['day_from_treatment'] = sched['LABDAY'] - sched['day0']
sched.to_csv(OUT / 'mainsail_psa_scheduled_long.csv', index=False)
wide = sched.pivot(index='RPT', columns='t', values='psa')
required = [0, 1, 2, 3]
early_complete_ids = wide.index[wide.reindex(columns=required).notna().all(axis=1)].tolist()
print('\n' + '=' * 72)
print('EARLY WINDOW')
print('Patients with C1D1 baseline:', wide.get(0).notna().sum())
print('Complete t=0,1,2,3:', len(early_complete_ids))
early_sched = sched[sched['RPT'].isin(early_complete_ids)].copy()
later_counts = early_sched[early_sched['t'] > 3].groupby('RPT').size()
forecast_ids = later_counts[later_counts >= 2].index.tolist()
print('Complete early window + >=2 later scheduled PSA:', len(forecast_ids))
rows = []
for rpt in forecast_ids:
    g = early_sched[early_sched['RPT'] == rpt].sort_values('t').reset_index(drop=True)
    target = None
    for i in range(1, len(g) - 1):
        t = int(g.loc[i, 't'])
        if t <= 3:
            continue
        prev_psa = float(g.loc[i - 1, 'psa'])
        this_psa = float(g.loc[i, 'psa'])
        next_psa = float(g.loc[i + 1, 'psa'])
        if this_psa <= prev_psa and next_psa > this_psa:
            target = {'RPT': rpt, 'target_available': True, 'target_t': t, 'target_cycle': int(g.loc[i, 'cycle']), 'target_day_from_treatment': float(g.loc[i, 'day_from_treatment']), 'target_psa': this_psa, 'previous_psa': prev_psa, 'next_psa': next_psa, 'n_scheduled_psa': len(g)}
            break
    if target is None:
        target = {'RPT': rpt, 'target_available': False, 'target_t': np.nan, 'target_cycle': np.nan, 'target_day_from_treatment': np.nan, 'target_psa': np.nan, 'previous_psa': np.nan, 'next_psa': np.nan, 'n_scheduled_psa': len(g)}
    rows.append(target)
targets = pd.DataFrame(rows)
targets.to_csv(OUT / 'mainsail_first_rebound_nadirs.csv', index=False)
pd.DataFrame({'RPT': early_complete_ids}).to_csv(OUT / 'mainsail_early_complete_ids.csv', index=False)
pd.DataFrame({'RPT': forecast_ids}).to_csv(OUT / 'mainsail_forecasting_ids.csv', index=False)
avail = targets[targets['target_available']].copy()
print('\n' + '=' * 72)
print('TARGET')
print('Forecasting cohort:', len(targets))
print('First rebound-associated nadir available:', len(avail))
if len(avail):
    x = avail['target_t']
    print('\nTarget t distribution:')
    print(x.describe().to_string())
    print('\nTarget histogram:')
    print(x.astype(int).value_counts().sort_index().to_string())
    print('\nFraction target t <= 12:', float((x <= 12).mean()))
    print('\nMedian target day from treatment:', float(avail['target_day_from_treatment'].median()))
ew = sched[sched['RPT'].isin(early_complete_ids) & sched['t'].isin([0, 1, 2, 3])].pivot(index='RPT', columns='t', values='psa').rename(columns={0: 'psa0', 1: 'psa1', 2: 'psa2', 3: 'psa3'})
for t in [1, 2, 3]:
    ew[f'z{t}'] = np.log10(ew[f'psa{t}'] / ew['psa0'])
xs = np.array([0.0, 1.0, 2.0, 3.0])

def slope(row):
    ys = np.array([0.0, row['z1'], row['z2'], row['z3']])
    return np.polyfit(xs, ys, 1)[0]
ew['slope'] = ew.apply(slope, axis=1)
ew.reset_index().to_csv(OUT / 'mainsail_early_features.csv', index=False)
print('\n' + '=' * 72)
print('FILES WRITTEN')
for p in sorted(OUT.glob('mainsail_*')):
    print(p)
