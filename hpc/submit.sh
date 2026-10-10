#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-simulation}"
CONFIG="${2:-config/local.yaml}"
case "$TARGET" in simulation|clinical|paper|reproduce_frozen|smoke) ;; *) echo "Unknown target: $TARGET" >&2;exit 2;; esac
if [[ "$TARGET" == smoke ]];then
    TARGET=simulation
    CONFIG=config/smoke.yaml
    [[ ! -f "$ROOT/config/local_smoke.yaml" ]] || CONFIG=config/local_smoke.yaml
fi
if [[ "$CONFIG" != /* ]];then CONFIG="$ROOT/$CONFIG";fi
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2;exit 2; }
mkdir -p "$ROOT/logs"
export RESCUE_ROOT="$ROOT" RESCUE_CONFIG="$CONFIG" RESCUE_TARGET="$TARGET"
export RESCUE_ENV="${RESCUE_ENV:-extinction-paper}"
export RESCUE_CONDA_BASE="${RESCUE_CONDA_BASE:-$(conda info --base)}"
sbatch --chdir="$ROOT" --output="$ROOT/logs/controller_%j.out" --error="$ROOT/logs/controller_%j.err" "$ROOT/hpc/controller.slurm"
