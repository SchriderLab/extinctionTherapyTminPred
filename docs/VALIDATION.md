# Validation performed for this refactor

## Executed successfully

**31 Python tests passed** with the supplied source archive available. The same
suite runs without the archive, with the three original-source comparisons
explicitly skipped. Tests cover:

- the primary Latin-hypercube design and mutation/treatment constraints;
- the reference-normalized distance and weighted ABC calculations;
- reference-only folds, hyperparameter selection and future/interior-nadir filtering;
- treatment time to stored lag conversion;
- reactive threshold detection, final-scan fallback and no-switch scoring;
- the noise driver's matrix-distance calculation;
- clinical task dependency/output consistency;
- a complete small table-based run through ABC, treatment evaluation, noise,
  numerical reporting and all eight simulation plots;
- equality of the shared ABC distance function to the uploaded implementation;
- equality of the K=5 broad-mixture and K=1 clinical parameter designs to the
  uploaded generators at fixed seeds;
- the command-line expansion dispatch without executing SLiM.

The fixture trajectories used in the integration test are mathematical software
fixtures, not SLiM outputs and not study results.

The clinical figure software self-test also passed **18/18 retained experimental
units**, producing **21 figures** from labelled synthetic tables. It checks source
schema handling and complete figure production. These fixtures and generated
figures are outside the delivered package; the 30 PDFs in `figures/` are the
frozen figures supplied with the manuscript.

## Checks using the available clinical extract

The ported VENICE reconstruction was executed on the available source extract.
The original expected selection counts were reproduced:

| Stage | Reconstructed count |
|---|---:|
| Positive PSA at baseline and cycles 1–3 | 506 |
| At least one later treatment-phase measurement | 482 |
| Eligible future interior observed nadir | 268 |
| First-rebound target available | 263 |

The decline-audit/event-row code was then executed. The E11 ±1-scan target
contained **1,414 evaluable updates from 266 patients**, with **500 positive
updates** (prevalence 0.3536067893). Only label/row construction was checked here;
these counts are not a rerun of the classifier, its AUROC or its bootstrap.
The raw clinical extract and patient-level test outputs are not included.

## Not executed in this environment

- SLiM 5.1 parsing/execution of the consolidated engine.
- Old-versus-new SLiM trajectory/RNG parity.
- Snakemake DAG parsing or scheduling, or a SLURM execution.
- A regenerated 80-patient/10,000-reference study.
- Full clinical model fitting, predictions, cross-validation or bootstrap.
- Reproduction of all manuscript numerical metrics from frozen simulation tables.

SLiM and Snakemake were unavailable in the preparation environment. Python AST
and shell syntax checks are not substitutes for those execution tests.

## Longleaf acceptance sequence

1. Configure the existing inputs and create `extinction-paper` through the
   supplied compute-node setup job.
2. Run `bash hpc/check_parity.sh SOURCE_ARCHIVE` and inspect its comparison report.
   Use the Python checker's `--full-size` option for the unmodified expansion
   exponent after the small comparison passes.
3. Run `RESCUE_DRY_RUN=1 bash hpc/submit.sh smoke`, then the actual smoke job.
4. Run `bash hpc/submit.sh reproduce_frozen config/local_frozen.yaml` and inspect
   `report/manuscript_comparison.csv`.
5. Run regenerated simulation/clinical targets only after these checks pass.
   Keep the frozen inputs and manuscript figures unchanged until then.

The workflow records numerical differences. It never samples repeatedly until a
reported result is obtained. The historical E02 reference table and E08b split
columns remain explicit frozen inputs, as documented in the main README.

## Preparation environment

```json
{
  "python": "3.13.5 (main, Jul 15 2026, 20:25:40) [GCC 14.2.0]",
  "platform": "Linux-6.18.44-x86_64-with-glibc2.41",
  "packages": {
    "numpy": "2.3.5",
    "pandas": "2.2.3",
    "scipy": "1.17.0",
    "scikit-learn": "1.8.0",
    "matplotlib": "3.10.8",
    "xgboost": "3.1.3",
    "pytest": "9.0.2",
    "PyYAML": "6.0.3"
  }
}
```

The Longleaf environment file uses version ranges to permit installation; the
setup job exports its exact resolved environment. New run records capture the
actual Python/package versions used there.
