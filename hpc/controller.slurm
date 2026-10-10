#!/usr/bin/env bash
#SBATCH --job-name=extinction-paper
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=3-00:00:00
set -eo pipefail
source "$RESCUE_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$RESCUE_ENV"
set -u
cd "$RESCUE_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$RESCUE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q tests
EXTRA=()
if [[ "${RESCUE_DRY_RUN:-0}" == 1 ]]; then EXTRA+=(--dry-run); fi
snakemake "${EXTRA[@]}" --snakefile "$RESCUE_ROOT/Snakefile" \
    --directory "$RESCUE_ROOT" --profile "$RESCUE_ROOT/profiles/longleaf" \
    --config "config_path=$RESCUE_CONFIG" -- "$RESCUE_TARGET"
