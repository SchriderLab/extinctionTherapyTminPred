"""Numerical audit against manuscript values, kept separate from generation."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .runtime import write_csv,dump_json,execution_record

EXPECTED={'MAE_weighted':(.882708,.00005),'abc_extinction':(.70375,1e-8),
          'true_nadir_extinction':(.6975,1e-8),'empirical_best_lag':(.77,1e-8),'dense_reactive':(.70125,1e-8)}

def report(cfg: dict,root: Path) -> None:
    s=json.loads((root/'evaluation/summary.json').read_text());rows=[]
    for key,(expected,tol) in EXPECTED.items():
        val=float(s[key]);rows.append({'metric':key,'observed':val,'manuscript':expected,'absolute_difference':abs(val-expected),
                                      'tolerance':tol,'matches':abs(val-expected)<=tol})
    table=pd.DataFrame(rows);write_csv(table,root/'report/manuscript_comparison.csv')
    audit=execution_record(cfg);audit['manuscript_values_match']=bool(table.matches.all())
    audit['evaluation']='Reported metrics are comparisons, not stopping or resampling targets.'
    audit['mode']=cfg['mode'];dump_json(root/'report/run.json',audit)
    text=['# Simulation results','',f"Run mode: `{cfg['mode']}`",'','| Metric | This run | Manuscript |','|---|---:|---:|']
    text += [f'| {r.metric} | {r.observed:.6f} | {r.manuscript:.6f} |' for r in table.itertuples()]
    text+=['','The full table records tolerances and differences. Regenerated runs must be checked before replacing frozen manuscript results.',
           'Model compatibility tests and the recorded software versions accompany each release; accepted counts are never obtained by resampling until the paper values appear.']
    (root/'report/RESULTS.md').write_text('\n'.join(text)+'\n')
