# Manuscript workflow scope

The inclusion boundary is Table S1 of the supplied 51-page manuscript, together
with its primary simulation and observation experiments. Shared helper functions
are retained when an included analysis calls them. An old scheduler reference is
not sufficient grounds for inclusion.

| Workflow task | Manuscript experiment / figures |
|---|---|
| patient/reference design, simulation, reference CV, patient prediction | Methods 2.2–2.4; Figures 2–4; S1–S2 |
| policy evaluation and observation noise | Methods 2.5, 2.7, S1.6; Figures 5–6 and 10 |
| cohort | E01; Figure 7 |
| direct | E02; S3 |
| ridge, gp | E03a–b; S4–S5 |
| mainsail_cohort, cross_trial | E03c–d; S6–S7 |
| fit_cloud, fit_agreement | E04a–b; S8–S9 |
| calibration design/simulation, calibration_fits, matching | E05a–b; S10–S11 |
| reference_features, query, scan_budget | E06; Figure 8 |
| xgboost, reference_size | E06b–c; S12–S13 |
| binary, rejection | E07/E07b; Figure 9 and S14 |
| empirical_bayes, qg_ablation | E08/E08b; S15–S16 |
| covariates_training | E09a–b; S17–S18 |
| decline_audit, window | E11a–b; S19–S20 |

E10 is not an executable experiment or figure target. Its shared state/event
helpers are retained only for E11. Earlier clinical KNN/HGB comparisons, manifold
screens, oracle-future models, obsolete cutoffs and unreported model sweeps are
excluded. The cross-trial kernel retains the historical filename containing
"4step", but its entry point runs only the three manuscript-relevant steps.

`src/rescue/clinical.py` declares each clinical task's dependencies, outputs and
entry point. `docs/figure_map.csv` maps the 30 retained figure files to the
manuscript labels. Numerical settings for clinical inference are catalogued in
`docs/clinical_parameters.json` and the two simulation priors are configurable in
`config/clinical.yaml`.

The original clinical target definitions remain distinct: first post-cycle-3
rebound, full-history observed minimum, full-history fitted nadir, and the
confirmed local event used by E11. Clinical comparisons retain their original
90-, 133-, 131- and 266-patient populations; these are not merged into one cohort.
