# Frozen-first manuscript workflow: public integration

The public `data/cohort/` and `data/reference/` tables contain the definitive
80-patient results. Do **not** regenerate patients/references just to install
this workflow. The `simulation` / `paper_regenerate` targets are opt-in only.

## Public analysis from committed inputs (no clinical data)

From the repository root on Longleaf, once the environment is installed:

```bash
# This guard refuses non-frozen configurations and submits to a compute node.
bash hpc/submit_frozen.sh config/public_frozen.yaml reproduce_frozen
```

Outputs are created in the *ignored* `results/runs/public_frozen/` directory.
This reruns ABC, policy evaluation, noise analysis, plots, and reports **from
existing CSV tables**; it does not execute SLiM.

## Complete paper workflow with private clinical inputs

The original clinical data/saved K=1 and K=5 libraries are **not** public.
On Longleaf, configure ignored local paths once, then use the frozen paper target:

```bash
python scripts/configure_inputs.py \
  --source /absolute/path/to/cancer_sims \
  --working-repo "$PWD"
# Verify that required clinical inputs have resolved in config/clinical.local.yaml.
# If clinical input discovery reports MISSING, resolve it before submission.
bash hpc/submit_frozen.sh config/paper.yaml paper
```

`config/paper.yaml` explicitly inherits `local_frozen.yaml` and has `mode: frozen`.
`config/paper_regenerate.yaml` explicitly inherits `local.yaml` and has
`mode: regenerate` (only invoke that target for independent experiments).

**Keep raw clinical data and derived patient-level clinical files in ignored
`private/`; never add them to Git.** The paper target needs that local access,
so it cannot be fully reproduced from the public repository alone.

## Safe Git staging

Use `stage_public_repo.py` to create a new branch in a **clean** public Git
checkout. It copies only source/config/docs/figures, never `data/cohort/`,
`data/reference/`, raw clinical inputs, `private/`, run artifacts or logs.
It refuses to overwrite a pre-existing file with different content, stages
changes for review, and does **not** commit or push.

The historical `sims/`, `scripts/`, and `hpc/` files are preserved. Review their
relationships to the new modules before deleting any old source. Until that
review, the maintained new engine is `models/quantitative_resistance.slim`,
while older scripts remain as historical provenance.

The original source checksums in `docs/package_code_checksums.json` describe
the Pro-generated package *before* this integration patch. They should not be
mistaken for checksums of these modified integration files.
