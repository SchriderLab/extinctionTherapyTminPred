#!/bin/bash
#SBATCH --job-name=clinical-check
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -euo pipefail
umask 077
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
python "$1/analysis/clinical/extensions/run_clinical.py" --source-run "$2/raw_run" --output "$2/extensions" --bootstrap 2000
python "$1/analysis/clinical/verify.py" "$2/raw_run" "$2/extensions" "$2/clinical_verification.json"
