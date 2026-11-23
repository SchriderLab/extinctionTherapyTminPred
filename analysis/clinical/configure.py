"""Validate private inputs and freeze a fresh Longleaf configuration."""
import hashlib
import json
import os
from pathlib import Path
import sys

raw=Path(__file__).resolve().parent/'raw'
run=Path(sys.argv[1]).resolve()
cfg=json.loads((raw/'config.template.json').read_text())
for rel, expected in json.loads((raw/'source_lock.json').read_text()).items():
    if hashlib.sha256((raw/rel).read_bytes()).hexdigest()!=expected:
        raise ValueError(f'Clinical source differs from the archived implementation: {rel}')
root=Path(os.environ.get('CLINICAL_HISTORICAL_ROOT',cfg['historical_root'])).resolve()
audit=Path(os.environ.get('CLINICAL_AUDIT_CASE', root/'venice_reanalysis_20261022/results/PAPER_CLINICAL_REPRO_20261027_B')).resolve()
flags=audit/'audit'/cfg['audit_pass']/'cohorts'
cfg.update(historical_root=str(root), audit_case=str(audit), output=str(run/'raw_run'),
           resume_from_output=None,slim=os.environ.get('CLINICAL_SLIM',cfg['slim']))
for key,name in [('adsl','venice_csv/adsl.csv'),('adlb','venice_csv/adlb.csv'),('mainsail_lab','mainsail_pc002/lab.sas7bdat')]:
    cfg[key]=str(Path(os.environ.get('CLINICAL_'+key.upper(),root/'data'/name)).resolve())
inputs=[Path(cfg[k]) for k in ('adsl','adlb','mainsail_lab')]
inputs.extend([flags/'manuscript_comparator_patient_flags_PRIVATE.csv',flags/'rerun_calibration231_membership_PRIVATE.csv'])
missing=[str(f) for f in inputs if not f.is_file() or not os.access(f,os.R_OK)]
if missing:
    sys.exit('Private Longleaf inputs missing or unreadable:\n'+'\n'.join(missing))
if not os.access(cfg['slim'],os.X_OK):
    sys.exit('SLiM executable unavailable: '+cfg['slim'])
if (run/'raw_run').exists() or (run/'config.json').exists():
    sys.exit('Choose a new run directory')
(run/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
