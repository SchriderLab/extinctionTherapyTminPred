# Model and time conventions recovered from the supplied code

The maintained implementation is `models/quantitative_resistance.slim`.
Its `phase` argument selects an execution phase or the clinical initialization.
This document records what the uploaded implementations actually do. It does
not change the manuscript or convert simulation units into clinical units.

## Shared biological mechanism

Cells are diploid and reproduce clonally in a nonWF model with zero recombination.
The `m1` and `m2` mutation classes have independently drawn lognormal effects.
Their stacking policy is `s`; effects carried by a cell are summed and capped at
1 for the drug-specific resistance phenotype. `m3` has stacking policy `f` and
contributes an indicator for partial cross-resistance. Mutation-effect callbacks
return 1; selection is applied through explicit drug survival and turnover.

For the primary preset, each drug-specific effect is drawn with meanlog
-0.635825624 and sdlog 0.5. Its uncapped arithmetic mean is approximately 0.6.
For each active drug, the survival factor is

```
1 - dose * (1 - min(1, summed_drug_specific_effect))
         * (1 - cross_resistance * indicator_of_m3)
```

Primary turnover multiplies survival through subpopulation fitness scaling.
The clinical initialization applies the same active-cell turnover in the
individual fitness callback, so dormant founders remain alive until their
specified establishment cycle. No resistance cost is added. The supplied
mutation-rate input is multiplied by Q=1000 once.

## Primary expansion is a size-target stage

The original parameter named `expansion_time=20` is used in `2^expansion_time`.
Starting with one cell, the model waits until an early-event population census
reaches at least 90% of that target, then saves a binary checkpoint. Turnover
means the number of elapsed ticks need not equal 20. The refactor retains this
implementation, including the early-event checkpoint.

The manuscript currently describes 20 cycles of pretreatment expansion. That
wording should be checked against this recovered size-target convention before
submission. The refactor has not replaced size-target expansion with a fixed
20-tick expansion.

If expansion becomes extinct, patient parameters remain fixed and the expansion
seed is incremented by 10,000 per attempt, up to the configured attempt limit.
When the original reference seed-resolution table is supplied, its resolved
seeds are used directly.

## Primary Drug-1 history and switching

Both the prefix and outcome phases load the same pretreatment checkpoint and
set the same patient-specific Drug-1 seed. Drug 1 is activated in the reload
cycle. The first recorded late-event census has treatment index t=0 and follows
that cycle's treated survival. It is not an untreated clinical baseline.

The outcome phase schedules Drug 2 in an early event at reload_cycle + lag.
The last Drug-1 census immediately before that intervention has t=lag-1.
Accordingly, the stored lag selected from an observed/predicted time is
round(t)+1. A new independent treatment seed is set at the actual switch.
The implementation reruns the common Drug-1 prefix from the checkpoint for each
lag/replicate; identical seeds preserve the pre-switch history.

The configured `treatment_time=300` terminates after the late census at
zero-based t=299. This is 300 recorded treatment indices, t=0,...,299.
The old `N_at_original_horizon` column at t=119 is preserved for compatibility.
The manuscript's phrase "by cycle 300" should consistently describe this
300-observation convention, or the time labels should be clarified separately.
No off-by-one change was made during refactoring.

Pre-switch extinction and post-switch extinction are extinct outcomes.
Pre-switch relapse, later relapse and nonzero unresolved populations are
non-extinct for the finite endpoint. The primary relapse cap is
1.5 * 2^expansion_time.

## Clinical initialization presets

The clinical single-population preset starts with five founders and uses the
historically defined effective age. The five-compartment preset starts with one
tagged founder per compartment. Each founder is dormant before its establishment
cycle; descendants inherit the compartment tag. Treatment begins systemically.

Both clinical presets record the untreated baseline after the last pretreatment
survival step. Drug 1 begins in the next cycle, and the first treated observation
is t=1. This clinical-baseline convention differs from the primary prefix's t=0.

At baseline, standing Drug-1 resistance is seeded up to the rounded target
carrier count by choosing m1-negative cells and adding one drawn m1 mutation to
the first haplosome. Existing carriers are retained. The clinical priors use
m1 mutation supply only, variable m1 effects and no m3 contribution. The current
clinical preset horizon is 80 and its relapse cap is 4 times its recorded
baseline burden. Exact prior ranges are in `config/clinical.yaml`.

The broad-mixture calibration generator consumes the historical core-regime
random draws before generating the reported broad-mixture design. This preserves
the reported design's random stream without running the unreported core regime.
The K=1 and K=5 prior generators were compared numerically with the uploaded
source generators; those Python comparisons passed.

## Reference sets and historical inputs

The original primary feature builder and the later sameprior-v2 builder are
not interchangeable. Their data formats and filtering differ. The new primary
builder reproduces the original future-nadir / boundary rules and imports the
original parameter manifest for a manuscript realization. `config/fresh.yaml`
requests a newly sampled design with the same parameter ranges.

The source archive does not specify the complete original E02 expanded-reference
generation chain unambiguously. E02 therefore takes the exact saved expanded
trajectory table as an explicit input. The K=5 calibration and K=1 forecasting
libraries have supplied generators and can be regenerated with their presets.

## Required executable compatibility check

Python source and parameter checks cannot establish SLiM event/RNG equivalence.
Run `bash hpc/check_parity.sh SOURCE_ARCHIVE` with SLiM 5.1 before a large run.
The check compares original and consolidated expansion metadata, complete
Drug-1 paths, switch outcomes and both clinical initialization trajectories for
fixed seeds. The default small test lowers the expansion target in both compared
implementations only for testing; `--full-size` in the Python checker retains
20. Original model copies are temporary test inputs, not maintained public models.
