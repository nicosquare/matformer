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
                              Repeat to evaluate several runs sequentially.
  --final-holdout-manifest PATH
                              Evaluate all checkpoints in a portfolio manifest.
                              Repeat to evaluate several manifests sequentially;
                              failures are reported after all manifests run.
  -h, --help                  Show this message.

All other arguments are forwarded to train.py. If --config is omitted, the
frozen controlled TinyStories comparison config is used.

Finalize several completed runs in one allocation:
  sbatch scripts/slurm_tinystories_controlled.sh --final-holdout-only RUN_A --final-holdout-only RUN_B

Finalize several portfolio arms in one allocation:
  sbatch scripts/slurm_tinystories_controlled.sh --final-holdout-manifest MANIFEST_A --final-holdout-manifest MANIFEST_B
USAGE
}

REPO_ROOT_ARG=""
PYTHON_BIN="${PYTHON_BIN:-python}"
FINAL_HOLDOUT_RUN_DIRS=()
FINAL_HOLDOUT_MANIFESTS=()
FORWARDED_ARGS=()
HAS_CONFIG=false

forwarded_option_value() {
  local option_name="$1"
  local index
  for ((index = 0; index < ${#FORWARDED_ARGS[@]}; index++)); do
    if [[ "${FORWARDED_ARGS[$index]}" == "$option_name" ]]; then
      if ((index + 1 < ${#FORWARDED_ARGS[@]})); then
        printf '%s\n' "${FORWARDED_ARGS[$((index + 1))]}"
      fi
      return 0
    fi
  done
  return 1
}

forwarded_override_value() {
  local field_name="$1"
  local index override
  for ((index = 0; index < ${#FORWARDED_ARGS[@]}; index++)); do
    if [[ "${FORWARDED_ARGS[$index]}" != "--override" ]]; then
      continue
    fi
    if ((index + 1 >= ${#FORWARDED_ARGS[@]})); then
      continue
    fi
    override="${FORWARDED_ARGS[$((index + 1))]}"
    if [[ "$override" == "$field_name="* ]]; then
      printf '%s\n' "${override#*=}"
      return 0
    fi
  done
  return 1
}

print_gpu_inventory() {
  local gpu_inventory gpu_line
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'Detected GPU: nvidia-smi unavailable\n'
    return
  fi
  if ! gpu_inventory="$(
    nvidia-smi \
      --query-gpu=index,uuid,name,memory.total \
      --format=csv,noheader 2>&1
  )"; then
    printf 'Detected GPU: unable to query (%s)\n' "$gpu_inventory"
    return
  fi
  while IFS= read -r gpu_line; do
    [[ -n "$gpu_line" ]] && printf 'Detected GPU: %s\n' "$gpu_line"
  done <<< "$gpu_inventory"
}

print_job_banner() {
  local workload_kind="training"
  local config_path run_id comparison_arm comparison_group comparison_role
  local experiment_id
  config_path="$(forwarded_option_value --config || true)"
  run_id="$(forwarded_option_value --run-id || true)"
  if [[ -z "$run_id" ]]; then
    run_id="$(forwarded_override_value run.run_id || true)"
  fi
  comparison_arm="$(
    forwarded_override_value controlled_experiment.comparison_arm_id || true
  )"
  comparison_group="$(
    forwarded_override_value controlled_experiment.comparison_group_id || true
  )"
  comparison_role="$(
    forwarded_override_value controlled_experiment.comparison_role || true
  )"
  if [[ -z "$config_path" && ${#FINAL_HOLDOUT_MANIFESTS[@]} -eq 0 && \
    ${#FINAL_HOLDOUT_RUN_DIRS[@]} -eq 0 ]]; then
    config_path="configs/controlled_exps/tinystories_controlled_convergence.yaml"
  fi

  if [[ ${#FINAL_HOLDOUT_MANIFESTS[@]} -gt 0 ]]; then
    workload_kind="final-holdout-manifests"
    if [[ ${#FINAL_HOLDOUT_MANIFESTS[@]} -eq 1 ]]; then
      experiment_id="$(basename "$(dirname "${FINAL_HOLDOUT_MANIFESTS[0]}")")"
    else
      experiment_id="${SLURM_JOB_NAME:-portfolio-holdout-bundle}"
    fi
  elif [[ ${#FINAL_HOLDOUT_RUN_DIRS[@]} -gt 0 ]]; then
    workload_kind="final-holdout-runs"
    experiment_id="$(basename "${FINAL_HOLDOUT_RUN_DIRS[0]}")"
  else
    experiment_id="${comparison_arm:-${run_id:-${SLURM_JOB_NAME:-unresolved}}}"
  fi

  printf '%s\n' '=== TinyStories controlled job ==='
  printf 'Experiment id: %s\n' "$experiment_id"
  printf 'Workload: %s\n' "$workload_kind"
  printf 'Holdout manifest count: %s\n' "${#FINAL_HOLDOUT_MANIFESTS[@]}"
  printf 'Run id: %s\n' "${run_id:-not-supplied}"
  printf 'Comparison arm: %s\n' "${comparison_arm:-not-supplied}"
  printf 'Comparison group: %s\n' "${comparison_group:-from-config}"
  printf 'Comparison role: %s\n' "${comparison_role:-from-config}"
  printf 'Config: %s\n' "${config_path:-not-applicable}"
  printf 'Slurm job id: %s\n' "${SLURM_JOB_ID:-local-shell}"
  printf 'Slurm job name: %s\n' "${SLURM_JOB_NAME:-unset}"
  printf 'Partition/QOS: %s / %s\n' \
    "${SLURM_JOB_PARTITION:-unset}" "${SLURM_JOB_QOS:-unset}"
  printf 'Node list: %s\n' "${SLURM_JOB_NODELIST:-unset}"
  printf 'Node hostname: %s\n' "$(hostname)"
  printf 'CUDA_VISIBLE_DEVICES: %s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  printf 'SLURM_JOB_GPUS: %s\n' "${SLURM_JOB_GPUS:-unset}"
  printf 'SLURM_GPUS_ON_NODE: %s\n' "${SLURM_GPUS_ON_NODE:-unset}"
  printf 'SLURM_GPUS_PER_NODE: %s\n' "${SLURM_GPUS_PER_NODE:-unset}"
  printf 'CPUs per task: %s\n' "${SLURM_CPUS_PER_TASK:-unset}"
  printf 'Memory per node: %s\n' "${SLURM_MEM_PER_NODE:-unset}"
  printf 'Memory per CPU: %s\n' "${SLURM_MEM_PER_CPU:-unset}"
  printf 'Python: %s\n' "$PYTHON_BIN"
  print_gpu_inventory
  printf '%s\n' '=== End job allocation ==='
}

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
      FINAL_HOLDOUT_RUN_DIRS+=("$2")
      shift 2
      ;;
    --final-holdout-manifest)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --final-holdout-manifest" >&2
        exit 2
      fi
      FINAL_HOLDOUT_MANIFESTS+=("$2")
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

print_job_banner

if [[ ${#FINAL_HOLDOUT_MANIFESTS[@]} -gt 0 ]]; then
  if [[ ${#FINAL_HOLDOUT_RUN_DIRS[@]} -ne 0 || ${#FORWARDED_ARGS[@]} -ne 0 ]]; then
    echo "--final-holdout-manifest cannot be combined with other work" >&2
    exit 2
  fi
  FINAL_HOLDOUT_FAILURES=()
  for manifest in "${FINAL_HOLDOUT_MANIFESTS[@]}"; do
    printf 'Finalizing portfolio holdout manifest: %s\n' "$manifest"
    if "$PYTHON_BIN" scripts/evaluate_final_holdout.py \
      --selection-manifest "$manifest" \
      --device cuda \
      --skip-existing; then
      printf 'Portfolio holdout manifest completed: %s\n' "$manifest"
    else
      exit_code=$?
      FINAL_HOLDOUT_FAILURES+=("exit=$exit_code manifest=$manifest")
      printf 'Portfolio holdout manifest failed (exit %s): %s\n' \
        "$exit_code" "$manifest" >&2
    fi
  done
  if [[ ${#FINAL_HOLDOUT_FAILURES[@]} -gt 0 ]]; then
    printf 'Portfolio holdout bundle completed with %s failure(s):\n' \
      "${#FINAL_HOLDOUT_FAILURES[@]}" >&2
    for failure in "${FINAL_HOLDOUT_FAILURES[@]}"; do
      printf '  %s\n' "$failure" >&2
    done
    exit 1
  fi
  printf 'Portfolio holdout bundle completed: %s manifest(s) succeeded\n' \
    "${#FINAL_HOLDOUT_MANIFESTS[@]}"
  exit 0
fi

if [[ ${#FINAL_HOLDOUT_RUN_DIRS[@]} -gt 0 ]]; then
  if [[ ${#FORWARDED_ARGS[@]} -ne 0 ]]; then
    echo "--final-holdout-only cannot be combined with training arguments" >&2
    exit 2
  fi
  for run_dir in "${FINAL_HOLDOUT_RUN_DIRS[@]}"; do
    printf 'Finalizing completed run: %s\n' "$run_dir"
    "$PYTHON_BIN" scripts/evaluate_final_holdout.py \
      --run-dir "$run_dir" \
      --device cuda \
      --skip-existing
  done
  exit 0
fi

if [[ "$HAS_CONFIG" == false ]]; then
  FORWARDED_ARGS=(
    --config configs/controlled_exps/tinystories_controlled_convergence.yaml
    "${FORWARDED_ARGS[@]}"
  )
fi

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
exec "$PYTHON_BIN" train.py "${FORWARDED_ARGS[@]}"
