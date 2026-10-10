#!/usr/bin/env bash
#SBATCH --job-name=rescue-parity
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
set -eo pipefail
source "$RESCUE_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$RESCUE_ENV"
set -u
cd "$RESCUE_ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m pytest -q tests
python scripts/check_model_parity.py \
  --source-archive "$RESCUE_SOURCE_ARCHIVE" --slim "$RESCUE_SLIM" \
  --out "results/runs/model_parity_${SLURM_JOB_ID}"
