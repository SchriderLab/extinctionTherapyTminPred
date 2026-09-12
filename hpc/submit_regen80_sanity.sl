#!/bin/bash
#SBATCH --job-name=regen80_sanity
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/users/w/a/wang0207/cancer_rescue/cancer_sims/fixed80_hybrid_heterogeneous_regenerated_v1/sanity_%j.out
#SBATCH --error=/users/w/a/wang0207/cancer_rescue/cancer_sims/fixed80_hybrid_heterogeneous_regenerated_v1/sanity_%j.err

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /users/w/a/wang0207/cancer_rescue/cancer_sims

python scripts/generate_regen80_manifest.py

python scripts/run_regen80_pipeline.py \
    --sanity \
    --workers 2

python scripts/sanity_check_regen80.py \
    --mode smoke
