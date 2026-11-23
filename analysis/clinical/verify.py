"""Compare a fresh clinical run against the archived split, target rows and aggregate results."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

CHECKS=Path(__file__).resolve().parent/'checks'
TABLES={
    'metrics': ('results/METRICS.csv',['dataset','scan_count','model','stratum'],['n','n_binary','MAE','AUROC']),
    'cutoffs': ('cutoff_sensitivity/CUTOFF_METRICS.csv',['dataset','scan_count','cutoff','model'],['n','AUROC','balanced_accuracy']),
    'sequential': ('results/SEQUENTIAL_BASELINE_PAIRED.csv',['dataset','comparator','averaging'],['n_patients','n_updates','EB_MAE','baseline_MAE','MAE_reduction']),
    'learning': ('results/LEARNING_TRANSFER_REPETITIONS.csv',['dataset','fraction','repeat'],['n_training','n_test','AUROC','balanced_accuracy']),
}


def canonical(frame,cols):
    f=frame[cols].drop_duplicates().sort_values(cols).copy()
    if 'Treb' in cols:
        f['Treb']=f['Treb'].fillna(-1).astype(int)
    return hashlib.sha256(f.to_csv(index=False,lineterminator='\n').encode()).hexdigest()


def verify(raw,ext,output):
    errors=[]
    checks={}
    expected=json.loads((CHECKS/'cohort_hashes.json').read_text())
    split=pd.read_csv(raw/'cohort/SPLIT_PRIVATE.csv',dtype={'id':str})
    score=pd.read_csv(raw/'evaluation/scored_predictions_PRIVATE.csv',dtype={'id':str})
    for name,frame,cols in [('split',split,['id','split']),('targets',score,['dataset','id','Treb'])]:
        ok=expected[name]==canonical(frame,cols)
        checks[name]=ok
        if not ok:
            errors.append(f'{name} membership differs')
    for name,(path,keys,cols) in TABLES.items():
        current=pd.read_csv((raw if name=='metrics' else ext)/path).set_index(keys,verify_integrity=True).sort_index()
        ref=pd.read_csv(CHECKS/(name+'.csv')).set_index(keys,verify_integrity=True).sort_index()
        if not current.index.equals(ref.index):
            errors.append(f'{name} row keys differ')
            continue
        bad={}
        for col in cols:
            x,y=current[col].to_numpy(float),ref[col].to_numpy(float)
            neq=~np.isclose(x,y,rtol=0,atol=1e-5,equal_nan=True)
            if neq.any():
                bad[col]=int(neq.sum())
        checks[name]={'rows':len(ref),'changed_by_column':bad}
        if bad:
            errors.append(f'{name} differs from archived results')
    status='MISMATCH' if errors else 'PASS'
    output.write_text(json.dumps(dict(status=status,tolerance=1e-5,checks=checks,errors=errors),indent=2)+'\n')
    if errors:
        sys.exit('See aggregate comparison: '+str(output))
    print(status, output)

if __name__=='__main__':
    verify(*(Path(arg) for arg in sys.argv[1:]))
