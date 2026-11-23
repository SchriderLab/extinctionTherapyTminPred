"""Reuse only the verified VENICE fit cache and split from the failed original run."""
from common import *
import shutil

def recover_venice(c,out):
    source=c.get('resume_from_output')
    if not source:return
    source=Path(source);out=Path(out)
    if source.resolve()==out.resolve():raise RuntimeError('Recovery requires a new output directory')
    oldconfig=json.loads((source/'CONFIG_FROZEN.json').read_text())
    comparable=lambda d:{k:v for k,v in d.items() if k not in ['output','resume_from_output']}
    if comparable(oldconfig)!=comparable(c):
        raise RuntimeError('Recovery configuration differs beyond output location; refusing to reuse cached fits/split')
    expected=json.loads((PACKAGE/'validation/PREVIOUS_SOURCE_HASHES.json').read_text())
    if json.loads((source/'SOURCE_HASHES.json').read_text())!=expected:
        raise RuntimeError('Recovery source is not the original verified code version')
    venice=json.loads((source/'steps/prepare_venice.json').read_text())
    mainsail=json.loads((source/'steps/prepare_mainsail.json').read_text())
    if venice.get('status')!='SUCCESS' or mainsail.get('status')!='FAILED' or mainsail.get('error')!='MAINSAIL membership differs from historical audit':
        raise RuntimeError('Recovery source does not have the expected successful VENICE / failed MAINSAIL preparation')
    if (source/'PREPARED_INPUTS_LOCK.json').exists():
        raise RuntimeError('Recovery source already completed preparation; this repair is scoped to the reported pre-simulation failure')
    cached_inputs=json.loads((source/'cohort/FIT_INPUTS.json').read_text())
    current_inputs={str(Path(c[k])):digest(c[k]) for k in ['adsl','adlb']}
    if cached_inputs!=current_inputs:raise RuntimeError('Raw VENICE inputs differ from the completed fit cache')
    fit=pd.read_csv(source/'cohort/fit263_PRIVATE.csv',dtype={'id':str})
    split=pd.read_csv(source/'cohort/SPLIT_PRIVATE.csv',dtype={'id':str})
    if len(fit)!=263 or fit.id.nunique()!=263 or not bools(fit.fit_success).all():
        raise RuntimeError('Invalid recovered 263-patient fit cache')
    if len(split)!=231 or split.id.nunique()!=231 or split.split.value_counts().to_dict()!={'TRAIN':184,'TEST':47}:
        raise RuntimeError('Invalid recovered 184/47 patient split')
    assert_ids(fit.loc[bools(fit.retained),'id'],split.id,'Recovered fit membership and split disagree')
    paths=['cohort/fit263_PRIVATE.csv','cohort/FIT_INPUTS.json','cohort/SPLIT_PRIVATE.csv']
    sources={relative:digest(source/relative) for relative in paths}
    for relative,sha in sources.items():
        dest=out/relative
        if dest.exists() and digest(dest)!=sha:
            raise RuntimeError('Recovery destination already contains a different file: '+relative)
    for relative,sha in sources.items():
        dest=out/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copyfile(source/relative,dest)
        if digest(dest)!=sha:raise RuntimeError('Recovery copy failed checksum: '+relative)
    dump(out/'cohort/RECOVERY.json',dict(source_output=str(source),files=sources,
        policy='Byte-identical original VENICE point fits and split copied; old run and its locks unchanged. Raw cohort/membership/targets are rechecked by prepare.venice.'))
    print('Reused completed VENICE point-fit cache and the frozen 184/47 split; no nonlinear refit needed.',flush=True)
