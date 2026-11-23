#!/bin/bash
#SBATCH --job-name=paper-reactive
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
set -euo pipefail
repo=$1
out=$2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
cd "$repo"
python -m unittest discover -s analysis/reactive -p 'test_*.py'
python scripts/make_final_results_figures.py --output "$out"
test -s "$out/reactive/paper_reactive_update.zip"
printf 'Completed: %s/reactive/paper_reactive_update.zip\n' "$out"
