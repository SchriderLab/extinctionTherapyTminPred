#!/bin/bash
#SBATCH --job-name=venice-treb-sim
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sim-%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python scripts/run.py --config "${1:-config.json}" --stage simulate --batch "${SLURM_ARRAY_TASK_ID:-0}"
