#!/usr/bin/env bash
#SBATCH --job-name=matformer-dmodel256-queue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=./logs/matformer_dmodel256_queue_%j.out
#SBATCH --error=./logs/matformer_dmodel256_queue_%j.err

set -euo pipefail

usage() {
  cat <<'USAGE'
Queue the d_model=256 pilot matrix through Slurm.

Usage:
  sbatch scripts/slurm_queue_dmodel256_pilot.sh --output-root /mnt/experiments/matformer [options]

Options:
  --repo-root PATH            Repository root; defaults to the sbatch submit directory.
  --output-root PATH          Root for run artifacts.
  --config PATH               Pilot config path.
  --slurm-script PATH         Slurm launcher used for the actual training jobs.
  --python-bin PATH           Python executable used for the queue helper.
  -h, --help                  Show this message.

Any remaining args are forwarded to the queue helper, for example:
  --token-budget 200000000 --learning-rate 0.001 --learning-rate-scale-rule none

The helper inspects OUTPUT_ROOT first and only submits unfinished runs.
USAGE
}

REPO_ROOT_ARG=""
OUTPUT_ROOT_ARG=""
FORWARDED_ARGS=()

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
    --output-root)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --output-root" >&2
        exit 2
      fi
      OUTPUT_ROOT_ARG="$2"
      shift 2
      ;;
    --python-bin)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --python-bin" >&2
        exit 2
      fi
      export PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      FORWARDED_ARGS+=("$@")
      break
      ;;
    *)
      FORWARDED_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${ALLOW_LOCAL_SLURM_WRAPPER:-0}" != "1" ]] \
  && [[ -z "${SLURM_SUBMIT_DIR:-}" || -z "${SLURM_JOB_NAME:-}" ]]; then
  echo "This launcher is intended for sbatch, not direct execution on the current node." >&2
  echo "Use: sbatch scripts/slurm_queue_dmodel256_pilot.sh --output-root /mnt/experiments/matformer" >&2
  exit 2
fi

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "$REPO_ROOT_ARG" ]]; then
  ROOT_DIR="$REPO_ROOT_ARG"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="$SLURM_SUBMIT_DIR"
else
  ROOT_DIR="$SCRIPT_ROOT"
fi

cd "$ROOT_DIR"
if [[ ! -f scripts/queue_dmodel256_pilot.py ]]; then
  echo "Could not find scripts/queue_dmodel256_pilot.py under repo root: $ROOT_DIR" >&2
  echo "Submit from the repository root or pass --repo-root /path/to/matformer." >&2
  exit 2
fi

if [[ -n "$OUTPUT_ROOT_ARG" ]]; then
  export OUTPUT_ROOT="$OUTPUT_ROOT_ARG"
fi
export OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-elasticnn}"
DEFAULT_CONDA_PYTHON="$HOME/.conda/envs/$CONDA_ENV_NAME/bin/python"
if [[ -z "${PYTHON_BIN:-}" && -x "$DEFAULT_CONDA_PYTHON" ]]; then
  export PYTHON_BIN="$DEFAULT_CONDA_PYTHON"
fi
export PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$OUTPUT_ROOT" logs

printf 'Slurm job id: %s\n' "${SLURM_JOB_ID:-local-shell}"
printf 'Python: %s\n' "$PYTHON_BIN"
printf 'Output root: %s\n' "$OUTPUT_ROOT"

exec "$PYTHON_BIN" scripts/queue_dmodel256_pilot.py --output-root "$OUTPUT_ROOT" "${FORWARDED_ARGS[@]}"
