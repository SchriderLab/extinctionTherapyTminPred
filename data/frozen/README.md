# Frozen simulation inputs

The package links to the existing publication simulation tables through ignored
local configuration. It does not duplicate those tables or include clinical data.
Run `python scripts/configure_inputs.py --source ... --working-repo ...` to select
the source files. The `reproduce_frozen` target copies the selected simulation
tables into its new run directory and recomputes ABC, evaluation and figures.

Expected cohort inputs:
- extinction_by_lag_binary300.csv
- patient_oracle_characterization.csv
- all_outcomes_resolved_to300.csv
- patients/patient_000/drug1_prefix.csv through patient_079/drug1_prefix.csv

Expected reference inputs:
- abc_reference_features_usable.csv
- manifests/manifest_abc_ref_hybrid_regen80_v1_10k.csv for regeneration
- abc_reference_features_all.csv for resolved historical expansion seeds

Keep the original files unchanged. The paper's observed metrics are comparison
values in `src/rescue/report.py`, not conditions enforced by repeated simulation.
