#!/usr/bin/env python3
"""Generate experiment-specific figures from existing saved analysis outputs."""
from __future__ import annotations
import argparse, datetime, html, json, sys, time, traceback, platform
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, matplotlib, scipy, sklearn
from plot_core import Store, Blocked
from experiments import REGISTRY

def write_report(out, records, store):
    pd.DataFrame(records).to_csv(out/'run_report.csv',index=False)
    (out/'source_manifest.json').write_text(json.dumps(store.used,indent=2))
    (out/'source_headers.json').write_text(json.dumps(store.headers,indent=2))
    sections=[]
    for r in records:
        figs=json.loads(r['figures'])
        images=''.join(f'<figure><a href="{html.escape(f)}.pdf"><img src="{html.escape(f)}.png"></a><figcaption>{html.escape(f)}</figcaption></figure>' for f in figs)
        sections.append(f'<section><h2>{html.escape(r["experiment"])} — {html.escape(r["status"])}</h2><p>{html.escape(r["title"])}</p><pre>{html.escape(r["message"])}</pre>{images}</section>')
    page='''<!doctype html><meta charset="utf-8"><title>VENICE figure run</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.5 system-ui;}section{margin:30px 0;border-top:1px solid #bbb;}figure{margin:24px 0;}img{width:100%;}pre{white-space:pre-wrap;}h2{font-size:20px;}</style><h1>VENICE experiment-by-experiment figure run</h1><p>Click a preview for its vector PDF. Missing/ambiguous inputs block that experiment; they are not replaced with recalled metrics or schematic results.</p>'''+''.join(sections)
    (out/'index.html').write_text(page,encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.')
    parser.add_argument('--repo-root',default='.')
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('inputs.json'))
    parser.add_argument('--out',type=Path)
    parser.add_argument('--experiments',default='manuscript',help='manuscript, all, or comma-separated retained experimental IDs')
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--dpi',type=int,help='Override PNG dpi; 600 by default')
    args=parser.parse_args()
    if args.list:
        for code,(title,_) in REGISTRY.items():print(f'{code:6s} {title}')
        return 0
    cfg=json.loads(args.config.read_text())
    if args.dpi:cfg['dpi']=args.dpi
    codes=(sorted(REGISTRY) if args.experiments=='manuscript' else (sorted(REGISTRY) if args.experiments=='all' else [x.strip() for x in args.experiments.split(',') if x.strip()]))
    unknown=set(codes)-set(REGISTRY)
    if unknown:parser.error('Unknown experiment IDs: '+str(sorted(unknown)))
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out=args.out or Path(args.root)/'figures'/'VENICE_experiments_v3'/stamp
    out.mkdir(parents=True,exist_ok=True)
    if (out/'run_report.csv').exists():parser.error(f'{out} already contains a run. Use a new --out directory to avoid mixing stale figures.')
    (out/'inputs_used.json').write_text(json.dumps(cfg,indent=2))
    (out/'environment.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},indent=2))
    store=Store(args.root,args.repo_root,cfg,out);records=[]
    for i,code in enumerate(codes,1):
        title,fn=REGISTRY[code];start=time.monotonic();before={p.stem for p in out.glob('*.pdf')}
        print(f'[{i}/{len(codes)}] {code}: {title}',flush=True)
        status='SUCCESS';message='';trace=''
        try:fn(store)
        except Blocked as e:status='BLOCKED';message=str(e)
        except Exception as e:status='ERROR';message=f'{type(e).__name__}: {e}';trace=traceback.format_exc()
        finally:plt.close('all')
        figures=sorted({p.stem for p in out.glob('*.pdf')}-before)
        if status!='SUCCESS' and figures:status='PARTIAL_'+status
        if status=='SUCCESS' and not figures:status='ERROR';message='Experiment returned without a saved figure.'
        if trace:
            (out/'errors').mkdir(exist_ok=True);(out/'errors'/f'{code}.txt').write_text(trace)
        records.append(dict(experiment=code,title=title,status=status,seconds=round(time.monotonic()-start,2),figures=json.dumps(figures),message=message))
        print(f'  {status}: {len(figures)} figure(s). {message}',flush=True)
        write_report(out,records,store)
    failures=[r for r in records if r['status']!='SUCCESS']
    print(f'\nOutput: {out}\nReport: {out / "run_report.csv"}\nContact sheet: {out / "index.html"}',flush=True)
    print(f'{len(records)-len(failures)}/{len(records)} experiments complete; {len(failures)} need attention.',flush=True)
    return 2 if failures else 0

if __name__=='__main__':raise SystemExit(main())
