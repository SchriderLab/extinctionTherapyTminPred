#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# -ge 1 ]] || { echo "Usage: bash hpc/check_parity.sh SOURCE_ARCHIVE [SLIM_EXECUTABLE]" >&2;exit 2; }
export RESCUE_ROOT="$ROOT" RESCUE_SOURCE_ARCHIVE="$(realpath "$1")"
export RESCUE_SLIM="${2:-/nas/longleaf/rhel9/apps/slim/5.1/bin/slim}"
export RESCUE_ENV="${RESCUE_ENV:-extinction-paper}" RESCUE_CONDA_BASE="${RESCUE_CONDA_BASE:-$(conda info --base)}"
mkdir -p "$ROOT/logs"
sbatch --chdir="$ROOT" --output="$ROOT/logs/parity_%j.out" --error="$ROOT/logs/parity_%j.err" "$ROOT/hpc/parity.slurm"
