#!/usr/bin/env bash
#SBATCH --job-name=tinystories-control
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=./logs/tinystories_controlled_%j.out
#SBATCH --error=./logs/tinystories_controlled_%j.err

set -euo pipefail

usage() {
  cat <<'USAGE'
Run one controlled TinyStories job through the normal config entrypoint.

Usage:
  sbatch scripts/slurm_tinystories_controlled.sh [options] [train.py arguments]

Options:
  --repo-root PATH            Repository root; defaults to the submit directory.
  --python-bin PATH           Python executable; defaults to python.
  --final-holdout-only PATH   Evaluate a completed selected run without training.
  -h, --help                  Show this message.

All other arguments are forwarded to train.py. If --config is omitted, the
frozen controlled TinyStories comparison config is used.
USAGE
}

REPO_ROOT_ARG=""
PYTHON_BIN="${PYTHON_BIN:-python}"
FINAL_HOLDOUT_RUN_DIR=""
FORWARDED_ARGS=()
HAS_CONFIG=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --repo-root" >&2
        exit 2
      fi
      REPO_ROOT_ARG="$2"
      shift 2
      ;;
    --python-bin)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --python-bin" >&2
        exit 2
      fi
      PYTHON_BIN="$2"
      shift 2
      ;;
    --final-holdout-only)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --final-holdout-only" >&2
        exit 2
      fi
      FINAL_HOLDOUT_RUN_DIR="$2"
      shift 2
      ;;
    --config)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --config" >&2
        exit 2
      fi
      HAS_CONFIG=true
      FORWARDED_ARGS+=("$1" "$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      FORWARDED_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$REPO_ROOT_ARG" ]]; then
  ROOT_DIR="$REPO_ROOT_ARG"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="$SLURM_SUBMIT_DIR"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT_DIR"

if [[ -n "$FINAL_HOLDOUT_RUN_DIR" ]]; then
  if [[ ${#FORWARDED_ARGS[@]} -ne 0 ]]; then
    echo "--final-holdout-only cannot be combined with training arguments" >&2
    exit 2
  fi
  exec "$PYTHON_BIN" scripts/evaluate_final_holdout.py \
    --run-dir "$FINAL_HOLDOUT_RUN_DIR"
fi

if [[ "$HAS_CONFIG" == false ]]; then
  FORWARDED_ARGS=(
    --config configs/controlled_exps/tinystories_controlled_convergence.yaml
    "${FORWARDED_ARGS[@]}"
  )
fi

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
exec "$PYTHON_BIN" train.py "${FORWARDED_ARGS[@]}"
