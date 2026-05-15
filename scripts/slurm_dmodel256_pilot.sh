#!/usr/bin/env bash
#SBATCH --job-name=matformer-dmodel256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -p cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --output=./logs/matformer_dmodel256_%j.out
#SBATCH --error=./logs/matformer_dmodel256_%j.err

set -euo pipefail

usage() {
  cat <<'USAGE'
Submit the Phase 4.6 d_model=256 reduced-token pilot comparison to Slurm.

Usage:
  sbatch scripts/slurm_dmodel256_pilot.sh --output-root /mnt/experiments/matformer [options] [-- runner args]

Options:
  --repo-root PATH            Repository root; defaults to the sbatch submit directory.
  --output-root PATH          Root for run artifacts; forwarded as OUTPUT_ROOT.
  --run-id RUN_ID             Run id from configs/dmodel256_pilot_comparison.yaml.
  --config PATH               Pilot config path.
  --python-bin PATH           Python executable to use inside the job.
  -h, --help                  Show this message.

Any remaining args are forwarded to scripts/run_dmodel256_pilot.sh, for example:
  --override training.max_steps_cap=1

For multi-GPU allocations, the launcher starts one training process per GPU
with python -m torch.distributed.run. On clusters that expose allocations
through CUDA_VISIBLE_DEVICES, that variable is used to choose --nproc_per_node
when Slurm GPU count variables are unavailable.

Resource requests can be overridden at submission time, for example:
  sbatch --gres=gpu:2 --time=01:00:00 --mem=32G scripts/slurm_dmodel256_pilot.sh --output-root /mnt/experiments/matformer --override training.max_steps_cap=1
USAGE
}

REPO_ROOT_ARG=""
OUTPUT_ROOT_ARG=""
FORWARDED_ARGS=()
DEFAULT_RUN_ID="dmodel256-pilot-comparison-001"
if [[ -n "${RUN_ID:-}" ]]; then
  RUN_ID_EXPLICIT=true
else
  RUN_ID="$DEFAULT_RUN_ID"
  RUN_ID_EXPLICIT=false
fi

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
      RUN_ID_EXPLICIT=true
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
  echo "Use: sbatch scripts/slurm_dmodel256_pilot.sh --output-root /mnt/experiments/matformer" >&2
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
if [[ ! -f scripts/run_dmodel256_pilot.sh ]]; then
  echo "Could not find scripts/run_dmodel256_pilot.sh under repo root: $ROOT_DIR" >&2
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

gpu_count_from_value() {
  local raw_value="$1"
  raw_value="${raw_value// /}"
  if [[ "$raw_value" == *,* ]]; then
    local without_commas="${raw_value//,/}"
    printf '%s\n' $(( ${#raw_value} - ${#without_commas} + 1 ))
    return 0
  fi
  if [[ "$raw_value" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$raw_value"
    return 0
  fi
  if [[ "$raw_value" =~ ([0-9]+)$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

visible_cuda_device_count() {
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    printf '0\n'
    return 0
  fi
  local visible_devices="$CUDA_VISIBLE_DEVICES"
  visible_devices="${visible_devices// /}"
  if [[ -z "$visible_devices" || "$visible_devices" == "NoDevFiles" ]]; then
    printf '0\n'
    return 0
  fi
  local without_commas="${visible_devices//,/}"
  printf '%s\n' $(( ${#visible_devices} - ${#without_commas} + 1 ))
}

resolve_gpus_per_node() {
  if [[ -n "${GPUS_PER_NODE:-}" ]]; then
    gpu_count_from_value "$GPUS_PER_NODE"
    return 0
  fi
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    gpu_count_from_value "$SLURM_GPUS_ON_NODE"
    return 0
  fi
  if [[ -n "${SLURM_GPUS_PER_NODE:-}" ]]; then
    gpu_count_from_value "$SLURM_GPUS_PER_NODE"
    return 0
  fi
  if [[ -n "${SLURM_GPUS:-}" ]]; then
    gpu_count_from_value "$SLURM_GPUS"
    return 0
  fi
  if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    gpu_count_from_value "$SLURM_JOB_GPUS"
    return 0
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    visible_cuda_device_count
    return 0
  fi
  if [[ "${ALLOW_LOCAL_SLURM_WRAPPER:-0}" == "1" ]]; then
    printf '1\n'
    return 0
  fi
  printf '4\n'
}

GPUS_PER_NODE="$(resolve_gpus_per_node)"
export GPUS_PER_NODE

printf 'Slurm job id: %s\n' "${SLURM_JOB_ID:-local-shell}"
printf 'Python: %s\n' "$PYTHON_BIN"
printf 'Output root: %s\n' "$OUTPUT_ROOT"
printf 'Config: %s\n' "${CONFIG_PATH:-configs/dmodel256_pilot_comparison.yaml}"
printf 'Run id: %s\n' "$RUN_ID"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
printf 'SLURM_GPUS_ON_NODE: %s\n' "${SLURM_GPUS_ON_NODE:-unset}"
printf 'SLURM_GPUS_PER_NODE: %s\n' "${SLURM_GPUS_PER_NODE:-unset}"
printf 'SLURM_GPUS: %s\n' "${SLURM_GPUS:-unset}"
printf 'SLURM_JOB_GPUS: %s\n' "${SLURM_JOB_GPUS:-unset}"
printf 'GPUs per node: %s\n' "$GPUS_PER_NODE"

if [[ "$GPUS_PER_NODE" -gt 1 ]]; then
  OUTPUT_ARGS=(--output-root "$OUTPUT_ROOT")
  TRAIN_ARGS=(
    train.py
    --config "${CONFIG_PATH:-configs/dmodel256_pilot_comparison.yaml}"
    "${OUTPUT_ARGS[@]}"
    "${FORWARDED_ARGS[@]}"
  )
  if [[ "$RUN_ID_EXPLICIT" == "true" ]]; then
    TRAIN_ARGS+=(--override "run.run_id=$RUN_ID")
  fi
  exec "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$GPUS_PER_NODE" \
    "${TRAIN_ARGS[@]}"
fi

exec bash scripts/run_dmodel256_pilot.sh "${FORWARDED_ARGS[@]}"
