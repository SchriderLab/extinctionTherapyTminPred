#!/bin/bash
#SBATCH --job-name=regen80_hybrid
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=2-23:45:00
#SBATCH --output=/users/w/a/wang0207/cancer_rescue/cancer_sims/fixed80_hybrid_heterogeneous_regenerated_v1/full_%j.out
#SBATCH --error=/users/w/a/wang0207/cancer_rescue/cancer_sims/fixed80_hybrid_heterogeneous_regenerated_v1/full_%j.err

set -euo pipefail

# Prevent numpy/BLAS from creating extra threads inside each worker.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /users/w/a/wang0207/cancer_rescue/cancer_sims

python scripts/generate_regen80_manifest.py

python scripts/run_regen80_pipeline.py \
    --workers 12 \
    --n-reps 10

python scripts/sanity_check_regen80.py \
    --mode full \
    --n-reps 10
