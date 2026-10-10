#!/usr/bin/env bash
# Hard guard for published/frozen-data reproducibility; refuses regeneration.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-config/public_frozen.yaml}"
TARGET="${2:-reproduce_frozen}"
case "$TARGET" in
  reproduce_frozen|paper) ;;
  *) echo "ERROR: Only reproduce_frozen or paper targets are permitted" >&2; exit 2 ;;
esac
if [[ "$CONFIG" != /* ]]; then CONFIG="$ROOT/$CONFIG"; fi
cd "$ROOT"
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python - "$CONFIG" "$TARGET" <<'PYCODE'
import sys
from pathlib import Path
from rescue.runtime import load_config
c = load_config(sys.argv[1]); target = sys.argv[2]
if c.get('mode') != 'frozen':
    raise SystemExit('REFUSING submission: configuration mode must be frozen')
if target == 'paper' and not c.get('clinical',{}).get('enabled', False):
    raise SystemExit('REFUSING paper target: clinical.enabled is false; use reproduce_frozen')
for k,files in {
    'cohort':['extinction_by_lag_binary300.csv','patient_oracle_characterization.csv',
              'all_outcomes_resolved_to300.csv'],
    'reference':['abc_reference_features_usable.csv']
}.items():
    p = Path(c['frozen'][k])
    if not p.is_dir(): raise SystemExit(f'Missing frozen {k} directory: {p}')
    for name in files:
        if not (p/name).is_file():
            raise SystemExit(f'Missing frozen input: {p/name}')
if target == 'paper':
    cc = Path(c['clinical']['config'])
    if not cc.is_file():
        raise SystemExit(f'Clinical config not found: {cc}')
print('Verified FROZEN mode, existing input tables, and target:', target)
print('No primary SLiM regeneration rules will be selected.')
PYCODE
bash "$ROOT/hpc/submit.sh" "$TARGET" "$CONFIG"
