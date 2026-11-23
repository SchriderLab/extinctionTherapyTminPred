#!/bin/bash
# Independent raw-data rerun for the revised training-guided clinical analyses.
set -euo pipefail
umask 077
repo=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
pkg="$repo/analysis/clinical/raw"
run=$(realpath -m -- "${1:?Supply a new private run directory}")
command -v sbatch >/dev/null
mkdir -p -- "$(dirname -- "$run")"
mkdir -m 700 -- "$run"
python "$repo/analysis/clinical/configure.py" "$run"
read -r count cap < <(python - "$run/config.json" <<'PY'
import math, json, sys
c=json.load(open(sys.argv[1]))
print(math.ceil(c['simulation_n']/c['simulation_batch_size']),c['simulation_concurrency'])
PY
)
submit() {
    local job
    job=$(sbatch --parsable --chdir="$pkg" --output="$run/%x-%A_%a.log" "$@")
    printf '%s\n' "${job%%;*}"
}
a=$(submit "$pkg/prepare.sbatch" "$run/config.json")
echo "Preparation: $a"
b=$(submit --dependency="afterok:$a" --array=0 "$pkg/simulate.sbatch" "$run/config.json")
echo "First batch: $b"
c=$(submit --dependency="afterok:$b" --array="1-$((count-1))%$cap" "$pkg/simulate.sbatch" "$run/config.json")
echo "Remaining simulations: $c"
d=$(submit --dependency="afterok:$c" "$pkg/analyse.sbatch" "$run/config.json")
echo "Refitting and scoring: $d"
e=$(submit --dependency="afterok:$d" "$repo/hpc/verify_clinical.sbatch" "$repo" "$run")
echo "Extensions and verification: $e"
echo "Aggregate check: $run/clinical_verification.json"
