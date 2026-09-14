# Predicting the Timing of Sequential Cancer Therapy from Sparse Early Tumor Dynamics

This repository contains the simulation code, frozen analysis inputs,
publication source data, and final figures for the predictive-timing extension
of the two-strike evolutionary-rescue cancer model.

## Overview

This study asks whether the future tumor-population nadir can be predicted
from a small number of early Drug-1 measurements, and whether such prediction
can recover the treatment benefit obtained from much denser reactive
longitudinal monitoring.

The final evaluation cohort contains 80 heterogeneous in-silico patients.
For each patient, the pretreatment and Drug-1 evolutionary history is fixed
across candidate second-strike times. Stochasticity is reintroduced only after
the second strike, allowing alternative treatment decisions to be evaluated
against the same underlying patient history.

The ABC population-nadir predictor is trained and tuned using an independent
reference library and does not use post-switch extinction outcomes during
training or hyperparameter selection.

## Repository structure

```text
extinctionTherapyTminPred/
├── README.md
├── requirements.txt
├── .gitignore
├── sims/                  Final SLiM simulation models
├── scripts/               Cohort generation, ABC, regret, and plotting code
├── hpc/                   Longleaf SLURM submission scripts
├── data/
│   ├── cohort/            Frozen 80-patient analysis inputs
│   ├── reference/         ABC reference features and sampled reference paths
│   └── source_data/       Data underlying manuscript figures and tables
└── figures/               Frozen manuscript figures
```

## Main simulation models

- `sims/hybrid_regen80_expand.slim`
- `sims/hybrid_regen80_prefix.slim`
- `sims/hybrid_regen80_outcome_extend300.slim`

## Main analysis scripts

- `scripts/generate_regen80_manifest.py`
- `scripts/run_regen80_pipeline.py`
- `scripts/sanity_check_regen80.py`
- `scripts/build_hybrid_abc_reference_library.py`
- `scripts/tune_and_run_hybrid_abc_80.py`
- `scripts/decompose_abc_oracle_regret.py`
- `scripts/make_final_results_figures.py`

## Final cohort design

- 80 in-silico patients
- 20 candidate second-strike lags
- 10 independent post-switch replicates per patient and lag
- 16,000 counterfactual outcome simulations
- extinction scored at cycle 300
- quantitative Drug-1 and Drug-2 resistance
- partial cross-resistance
- no resistance fitness cost

## ABC reference library

A total of 10,000 reference simulations were attempted. After excluding
unusable simulations and trajectories with boundary nadirs, the final
nadir-prediction reference set contains 9,983 trajectories.

The final predictor uses four early Drug-1 population measurements from
treatment cycles 0--3 to predict the future population nadir.

## Reproducing the manuscript figures

From the repository root, run:

```bash
python scripts/make_final_results_figures.py
```

Reproduced figures and source data are written to:

```text
reproduced_results/
├── figures/
└── source_data/
```

The `reproduced_results/` directory is not tracked by Git. The committed
`figures/` and `data/source_data/` directories contain the frozen manuscript
versions for comparison.

## Software

The simulations were performed with SLiM 5.1.

Python dependencies are listed in `requirements.txt`.

## Large intermediate files

Large SLiM population snapshots, scheduler logs, scratch analyses, and the
complete set of raw reference-library trajectories are intentionally not
tracked. The frozen tables required for the reported analyses are included.
A sampled subset of reference trajectories used for the supplementary
reference-support figure is also included.
