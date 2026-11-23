#!/usr/bin/env python3
"""Regenerate current clinical figures and new analyses from the frozen October run."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor/original/scripts'))
from common import csv, dump, digest, require_compute
import numpy as np
import pandas as pd


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_paths(config):
    root = Path(config['historical_root'])
    return {name: Path(config.get(name) or root/'data/venice_csv'/f'{name}.csv')
            for name in ['adsl', 'adlb']}


def validate(source):
    required = ['CONFIG_FROZEN.json','simulation/REFERENCE.npz','simulation/REFERENCE_LOCK.json',
                'queries/venice_long_PRIVATE.csv','queries/mainsail_long_PRIVATE.csv',
                'training/long_PRIVATE.csv','training/fits_PRIVATE.csv',
                'predictions/fixed_PRIVATE.csv','predictions/learning_PRIVATE.csv',
                'predictions/sequential_PRIVATE.csv','predictions/ALL_PREDICTIONS_LOCK.json']
    missing = [name for name in required if not (source/name).is_file()]
    if missing:
        raise FileNotFoundError('Use the COMPLETE Longleaf VENICE_TREB_20261028_B run directory. Missing: '+', '.join(missing))
    config = json.loads((source/'CONFIG_FROZEN.json').read_text())
    if config['simulation_n'] != 2000 or config['empirical_draws'] != 10000:
        raise ValueError('Expected original 2,000 SLiM references and 10,000 empirical-prior draws')
    lock = json.loads((source/'PREPARED_INPUTS_LOCK.json').read_text())
    for name, expected in lock.items():
        path = source/name
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError('Prepared-input lock mismatch: '+name)
    reference_lock = json.loads((source/'simulation/REFERENCE_LOCK.json').read_text())
    if digest(source/'simulation/REFERENCE.npz') != reference_lock['sha256']:
        raise RuntimeError('Frozen reference bank hash changed')
    raw_hashes = json.loads((source/'cohort/COHORT_REPRODUCTION.json').read_text())['input_sha256']
    for name, path in raw_paths(config).items():
        if not path.is_file():
            raise FileNotFoundError(f'Required cohort-figure input is missing: {path}. Restore the original raw {name}.csv before regeneration.')
        expected = [value for original, value in raw_hashes.items() if Path(original).name == path.name]
        if len(expected) != 1 or digest(path) != expected[0]:
            raise RuntimeError('Original raw clinical-file hash changed: '+str(path))
    return config


def recompute(source, dest, config, bootstrap):
    """Reuse all prepared inputs and references, rerun inference in a new workspace."""
    work = dest/'recomputed_run'
    work.mkdir()
    for directory in ['cohort','training','queries','evaluation']:
        shutil.copytree(source/directory, work/directory)
    (work/'simulation').mkdir()
    for path in (source/'simulation').iterdir():
        if path.is_file():
            shutil.copy2(path, work/'simulation'/path.name)
    for directory in ['models','predictions','results']:
        (work/directory).mkdir()
    config = dict(config, output=str(work), bootstrap_n=bootstrap)
    dump(work/'CONFIG_FROZEN.json',config)
    for name in ['PREPARED_INPUTS_LOCK.json','ENVIRONMENT.json','SOURCE_HASHES.json']:
        if (source/name).exists():shutil.copy2(source/name,work/name)
    # Retain historical auxiliary predictions whose methods are unchanged.
    for path in (source/'predictions').glob('early_fit_*_PRIVATE.csv'):
        shutil.copy2(path,work/'predictions'/path.name)
    import predict, empirical, sequential, report
    for name, action in [('fixed',predict.fixed),('empirical',empirical.run),
                         ('clinical_checks',predict.clinical_checks),('reference_size',predict.reference_size),
                         ('sequential',sequential.run),('cross_trial',predict.cross_trial)]:
        print('Recomputing clinical stage '+name,flush=True)
        action(config,work)
    dump(work/'predictions/ALL_PREDICTIONS_LOCK.json',
         {p.name:digest(p) for p in sorted((work/'predictions').glob('*.csv'))})
    report.run(config,work)
    dump(work/'RECOMPUTE_PROVENANCE.json',dict(source=str(source),
        explanation='Same frozen prepared cohort and reference bank; current vendored inference code; new output only'))
    return work


def manuscript_tables(source, dest, bootstrap, seed):
    """Explicit count/budget tables prevent mixing four-visit and six-visit results."""
    metrics = source.scores.copy()
    metrics['clinical_target'] = 'Treb'
    csv(dest/'results/CLINICAL_METRICS_BY_COHORT_AND_BUDGET.csv',metrics)
    table = []
    rng = np.random.default_rng(seed+200)
    for dataset in ['venice','mainsail']:
        for scope, n in [('future_Treb',4),('fixed_future6',4),('fixed_future6',6)]:
            a=source.select(dataset,n,'ABC_noisy_joint',scope)
            for comparator in ['ABC_clean_joint','Constant_TRAIN','ABC_GP','Clinical_PSA']:
                b=source.select(dataset,n,comparator,scope)
                joined=a[['id','Treb','pred']].merge(b[['id','pred']],on='id',validate='one_to_one',suffixes=('_abc','_comparison'))
                if len(joined)!=len(a) or len(joined)!=len(b):raise ValueError('Unmatched clinical comparison')
                gain=np.abs(joined.pred_comparison-joined.Treb)-np.abs(joined.pred_abc-joined.Treb)
                draws=rng.integers(len(gain),size=(bootstrap,len(gain)))
                values=gain.to_numpy()[draws].mean(axis=1)
                table.append(dict(dataset=dataset,scope=scope,scan_count=n,n=len(joined),
                    model='ABC_noisy_joint',comparator=comparator,MAE_ABC=float(np.mean(abs(joined.pred_abc-joined.Treb))),
                    MAE_comparator=float(np.mean(abs(joined.pred_comparison-joined.Treb))),
                    MAE_reduction=float(gain.mean()),lo=float(np.quantile(values,.025)),hi=float(np.quantile(values,.975))))
    csv(dest/'results/CLINICAL_PAIRED_SCOPE_CHECKS.csv',pd.DataFrame(table))
    import matplotlib.pyplot as plt
    rows=pd.DataFrame(table)
    figure,axes=plt.subplots(1,2,figsize=(9.2,3.8),layout='constrained')
    for ax,dataset in zip(axes,['venice','mainsail']):
        group=rows[(rows.dataset==dataset)&(rows.scope=='future_Treb')&rows.comparator.eq('ABC_clean_joint')].iloc[0]
        median=rows[(rows.dataset==dataset)&(rows.scope=='future_Treb')&rows.comparator.eq('Constant_TRAIN')].iloc[0]
        ax.bar(['Original\nreferences','Residual-adjusted\nreferences','TRAIN\nmedian'],[group.MAE_comparator,group.MAE_ABC,median.MAE_comparator],color=['#BF8A3B','#007F86','#777777'])
        ax.set(ylabel='MAE (treatment cycles)',title=f'{dataset.upper()}: four PSA measurements, n={group.n}')
    figure.savefig(dest/'figures/E06d_four_measurement_transfer.pdf')
    figure.savefig(dest/'figures/E06d_four_measurement_transfer.png',dpi=160)
    plt.close(figure)


def sequential_plot(dest):
    import matplotlib.pyplot as plt
    paired=pd.read_csv(dest/'results/SEQUENTIAL_BASELINE_PAIRED.csv')
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.3),layout='constrained')
    for ax,averaging,title in zip(axes,['equal_patients','pooled_updates'],['Each patient weighted equally','Each available update\nweighted equally']):
        rows=paired[paired.averaging.eq(averaging)].sort_values(['dataset','comparator'])
        for i,row in enumerate(rows.itertuples()):
            color='#007F86' if row.comparator=='TRAIN_unconfirmed_median' else '#777777'
            ax.hlines(i,row.lo,row.hi,color=color,lw=1.8)
            ax.plot(row.MAE_reduction,i,'o',color=color)
        labels=[r.dataset.upper()+' / '+('unconfirmed median' if r.comparator=='TRAIN_unconfirmed_median' else 'fixed median') for r in rows.itertuples()]
        ax.set_yticks(np.arange(len(rows)),labels,fontsize=8)
        ax.axvline(0,color='.6',ls=':')
        ax.set(xlabel='MAE reduction by empirical Bayes (cycles)',title=title)
    fig.suptitle('Sequential forecasts and TRAIN medians on the same available prefixes', fontsize=11)
    fig.savefig(dest/'figures/E08c_sequential_matched_baselines.pdf')
    fig.savefig(dest/'figures/E08c_sequential_matched_baselines.png',dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--recompute',action='store_true',help='Rerun all clinical inference with frozen cohorts/reference bank in a new output copy')
    parser.add_argument('--preflight',action='store_true',help='Validate required inputs and cohorts without fitting')
    args=parser.parse_args()
    if args.bootstrap<200:raise ValueError('Use at least 200 resamples, production default 2000')
    source=args.source_run.resolve()
    config=validate(source)
    renderer=module_from_file('publication_renderer',HERE/'vendor/regenerate_figures.py')
    data=renderer.Source(source)
    if args.preflight:
        print('Clinical preflight passed: frozen reference, inputs, 184/47 split, matched28/94, MAINSAIL181')
        return
    require_compute()
    dest=args.output.resolve()
    if dest.exists():raise FileExistsError('Choose a new output directory: '+str(dest))
    dest.mkdir(parents=True)
    (dest/'results').mkdir()
    if args.recompute:
        source=recompute(source,dest,config,args.bootstrap)
        data=renderer.Source(source)
    plot=renderer.Plotter(data,dest/'figures',args.bootstrap)
    for number in ['8','9']+[f'S{i}' for i in range(6,21) if i!=18]:
        if number=='S20':plot.s20(data.test_history())
        else:getattr(plot,'fig'+number if number in ['8','9'] else number.lower())()
    plot.finish()
    manuscript_tables(data,dest,args.bootstrap,config['seed'])
    import cutoff_sensitivity, learning_transfer, sequential_baseline
    cutoff_sensitivity.run(source,dest/'cutoff_sensitivity',args.bootstrap)
    learning_transfer.run(source,dest,config['seed'])
    sequential_baseline.run(source,dest,args.bootstrap)
    sequential_plot(dest)
    raw = raw_paths(config)
    cohort=module_from_file('cohort_renderer',HERE/'vendor/plot_cohort_figure8.py')
    cohort.plot(raw['adsl'],raw['adlb'],dest/'figures',source/'cohort/SPLIT_PRIVATE.csv')
    cohort_status='regenerated E01 with one panel-A title and later TRAIN/TEST groups labeled in B'
    dump(dest/'CLINICAL_REGENERATION.json',dict(status='SUCCESS',source_run=str(source),
        recomputed=args.recompute,clinical_target='Treb',primary_cutoff=8,bootstrap_replicates=args.bootstrap,
        sections={'3.6':'guided references with/without residuals and measurement count',
                  '3.7':'binary cutoff sensitivity, same-model MAINSAIL learning curve, baseline labs',
                  '3.8':'sequential EB with matched-prefix constant and operational medians',
                  '3.9':'state vs elapsed-time proximity'},
        cohort_figure=cohort_status,checks=data.checks,
        input_hashes=data.inputs,code_sha256={str(p.relative_to(HERE)):digest(p) for p in HERE.rglob('*.py') if 'tests' not in p.parts},
        uncertainty='Conditional on fitted models; bootstrap by patient; learning curves use subset variation'))
    print('Clinical regeneration complete: '+str(dest),flush=True)


if __name__=='__main__':
    main()
