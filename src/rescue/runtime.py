"""Configuration, paths and atomic output shared by workflow entry points."""
from __future__ import annotations
import hashlib, json, os, platform, subprocess, sys, tempfile
from pathlib import Path
from typing import Any
import yaml

PACKAGE = Path(__file__).resolve().parents[2]

def allocated_threads() -> int:
    return max(1, int(os.environ.get('RESCUE_THREADS', os.environ.get('SLURM_CPUS_PER_TASK', '1'))))

def workspace_path(suffix: str = '') -> Path:
    root = os.environ.get('RESCUE_WORKSPACE')
    if not root:
        raise RuntimeError('Clinical kernels must be run by the workflow; RESCUE_WORKSPACE is unset.')
    return Path(root) / suffix

def deep_update(base: dict, other: dict) -> dict:
    for key, val in other.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], val)
        else:
            base[key] = val
    return base

def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    cfg: dict = {}
    for parent in raw.pop('extends', []):
        deep_update(cfg, load_config(path.parent / parent))
    return deep_update(cfg, raw)

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def config_digest(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

def dump_json(path: str | Path, obj: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as f:
        json.dump(obj, f, indent=2, allow_nan=False, default=str); f.write('\n'); tmp = f.name
    os.replace(tmp, path)

def write_csv(df, path: str | Path, **kwargs) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False, newline='') as f:
        df.to_csv(f, index=False, **kwargs); tmp = f.name
    os.replace(tmp, path)

def run_command(cmd: list[str], log: Path, *, cwd: Path | None = None, env: dict | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as f:
        f.write('COMMAND: '+repr(cmd)+'\n'); f.flush()
        p = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode:
        tail = '\n'.join(log.read_text(errors='replace').splitlines()[-15:])
        raise RuntimeError(f'Exit {p.returncode}; see {log}\n{tail}')

def execution_record(cfg: dict) -> dict:
    import importlib.metadata as md
    versions = {}
    for name in ['numpy','pandas','scipy','scikit-learn','matplotlib','xgboost','snakemake','snakemake-executor-plugin-slurm']:
        try: versions[name] = md.version(name)
        except md.PackageNotFoundError: versions[name] = None
    return {'config_sha256':config_digest(cfg), 'python':sys.version, 'platform':platform.platform(), 'packages':versions}
