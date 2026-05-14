#!/usr/bin/env bash
#SBATCH --job-name=matformer-78m
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -p cscc-gpu-p
#SBATCH --output=./logs/matformer_78m_%j.out
#SBATCH --error=./logs/matformer_78m_%j.err

set -euo pipefail

usage() {
  cat <<'USAGE'
Submit the Phase 4.5 78M reduced-token pilot to Slurm.

Usage:
  sbatch scripts/slurm_78m_pilot.sh --output-root /mnt/experiments/matformer [options] [-- runner args]

Options:
  --repo-root PATH            Repository root; defaults to the sbatch submit directory.
  --output-root PATH          Root for run artifacts; forwarded as OUTPUT_ROOT.
  --run-id RUN_ID             Run id from configs/78m_reduced_pilot.yaml.
  --config PATH               Pilot config path.
  --python-bin PATH           Python executable to use inside the job.
  -h, --help                  Show this message.

Any remaining args are forwarded to scripts/run_78m_pilot.sh, for example:
  --override training.max_steps_cap=1

Resource requests can be overridden at submission time, for example:
  sbatch --time=01:00:00 --mem=32G scripts/slurm_78m_pilot.sh --output-root /mnt/experiments/matformer --override training.max_steps_cap=1
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
    --run-id)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --run-id" >&2
        exit 2
      fi
      export RUN_ID="$2"
      shift 2
      ;;
    --config)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --config" >&2
        exit 2
      fi
      export CONFIG_PATH="$2"
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
  echo "Use: sbatch scripts/slurm_78m_pilot.sh --output-root /mnt/experiments/matformer" >&2
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
if [[ ! -f scripts/run_78m_pilot.sh ]]; then
  echo "Could not find scripts/run_78m_pilot.sh under repo root: $ROOT_DIR" >&2
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
printf 'Config: %s\n' "${CONFIG_PATH:-configs/78m_reduced_pilot.yaml}"
printf 'Run id: %s\n' "${RUN_ID:-78m-reduced-pilot-001}"

exec bash scripts/run_78m_pilot.sh "${FORWARDED_ARGS[@]}"
