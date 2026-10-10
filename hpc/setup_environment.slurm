#!/usr/bin/env bash
#SBATCH --job-name=extinction-env
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
set -eo pipefail
ROOT="${RESCUE_ROOT:-$SLURM_SUBMIT_DIR}"
CONDA_BASE="${RESCUE_CONDA_BASE:-$(conda info --base)}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
cd "$ROOT"
conda env create -f environment.yaml
conda activate extinction-paper
mkdir -p logs
conda env export > logs/environment-resolved.yaml
python -m pytest -q tests
