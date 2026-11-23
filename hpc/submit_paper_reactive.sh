#!/bin/bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
out=$(realpath -m -- "${1:?Supply a new output directory}")
command -v sbatch >/dev/null
python -c 'import numpy, pandas, matplotlib, scipy, sklearn'
mkdir -p -- "$(dirname -- "$out")"
mkdir -- "$out"
job=$(sbatch --parsable --output="$out/slurm-%j.log" "$repo/hpc/paper_reactive.sbatch" "$repo" "$out")
printf 'Submitted reactive job %s\nLog: %s/slurm-%s.log\nAfter PASS: %s/reactive/paper_reactive_update.zip\n' "${job%%;*}" "$out" "${job%%;*}" "$out"
