#!/usr/bin/env bash
set -euo pipefail

# Phase 4 runner: d_model=256 MatFormer-Llama/SwiGLU pilot comparison path.
# Default comparison scope: nested-random, nested-all, and standalone S/M/L/XL.
# Standalone rows may be emitted as run_status=omitted when compute is capped.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
CONFIG_PATH="${CONFIG_PATH:-configs/dmodel256_pilot_comparison.yaml}"
DEFAULT_RUN_ID="dmodel256-pilot-comparison-001"
MODE="${MODE:-comparison}"
GRANULARITY="${GRANULARITY:-}"
RUN_STANDALONE_BASELINES="${RUN_STANDALONE_BASELINES:-0}"
COMPARISON_ID="${COMPARISON_ID:-dmodel256-pilot-comparison-001}"
if [[ -n "${RUN_ID:-}" ]]; then
  RUN_ID_EXPLICIT=true
else
  RUN_ID="$DEFAULT_RUN_ID"
  RUN_ID_EXPLICIT=false
fi
OUTPUT_ARGS=()
FORWARDED_ARGS=()
HAS_OUTPUT_ROOT_ARG=false
OUTPUT_DIR_EXPLICIT=false
OUTPUT_ROOT_VALUE="${OUTPUT_ROOT:-outputs}"

usage() {
  cat <<'USAGE'
Run the Phase 4.7 d_model=256 pilot comparison.

Usage:
  bash scripts/run_dmodel256_pilot.sh [options] [-- train.py args]

Options:
  --config PATH             Pilot config path.
  --mode MODE               comparison, nested-random, nested-all, or standalone.
  --granularity NAME        Standalone granularity: s, m, l, or xl.
  --run-id RUN_ID           Run id to write through a config override.
  --output-root PATH        Root for run artifacts.
  --output-dir PATH         Explicit run output directory; implies a single run.
  -h, --help                Show this message.

The default comparison runs nested-random and nested-all. Standalone S/M/L/XL
rows are emitted as omitted unless RUN_STANDALONE_BASELINES=1 is set.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --config" >&2
        exit 2
      fi
      CONFIG_PATH="$2"
      shift 2
      ;;
    --mode)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --mode" >&2
        exit 2
      fi
      MODE="$2"
      shift 2
      ;;
    --granularity)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --granularity" >&2
        exit 2
      fi
      GRANULARITY="$2"
      shift 2
      ;;
    --run-id)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --run-id" >&2
        exit 2
      fi
      RUN_ID="$2"
      RUN_ID_EXPLICIT=true
      shift 2
      ;;
    --output-root)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --output-root" >&2
        exit 2
      fi
      OUTPUT_ARGS+=(--output-root "$2")
      OUTPUT_ROOT_VALUE="$2"
      HAS_OUTPUT_ROOT_ARG=true
      shift 2
      ;;
    --output-dir)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --output-dir" >&2
        exit 2
      fi
      OUTPUT_ARGS+=(--output-dir "$2")
      OUTPUT_DIR_EXPLICIT=true
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

if [[ -n "${OUTPUT_ROOT:-}" && "$HAS_OUTPUT_ROOT_ARG" != "true" ]]; then
  OUTPUT_ARGS+=(--output-root "$OUTPUT_ROOT")
  OUTPUT_ROOT_VALUE="$OUTPUT_ROOT"
fi

# Explicit single-run targets should not fan out into multiple output dirs.
if [[ "$MODE" == "comparison" ]] \
  && [[ "$RUN_ID_EXPLICIT" == "true" || "$OUTPUT_DIR_EXPLICIT" == "true" ]]; then
  MODE="nested-random"
fi

printf 'd_model=256 pilot target: MatFormer-Llama/SwiGLU reduced-token comparison\n'
printf 'Config: %s\n' "$CONFIG_PATH"
printf 'Mode: %s\n' "$MODE"
if [[ "$MODE" == "standalone" ]]; then
  printf 'Granularity: %s\n' "$GRANULARITY"
fi
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  printf 'Output root: %s\n' "$OUTPUT_ROOT"
fi

python_command() {
  local -a command_parts
  # Allow the Slurm wrapper to provide a launcher prefix such as
  # "python -m torch.distributed.run --nproc_per_node 4".
  read -r -a command_parts <<< "$PYTHON_BIN"
  printf '%s\n' "${command_parts[@]}"
}

mode_run_id() {
  local mode="$1"
  local granularity="${2:-}"
  case "$mode" in
    nested-random) printf 'dmodel256-nested-random-001\n' ;;
    nested-all) printf 'dmodel256-nested-all-001\n' ;;
    standalone) printf 'dmodel256-standalone-%s-001\n' "$granularity" ;;
    *)
      echo "Unknown mode: $mode" >&2
      exit 2
      ;;
  esac
}

mode_overrides() {
  local mode="$1"
  local granularity="${2:-}"
  case "$mode" in
    nested-random)
      printf '%s\n' \
        "run.model_family=nested" \
        "run.sampling_mode=nested-random"
      ;;
    nested-all)
      printf '%s\n' \
        "run.model_family=nested" \
        "run.sampling_mode=nested-all"
      ;;
    standalone)
      if [[ "$granularity" != "s" && "$granularity" != "m" \
        && "$granularity" != "l" && "$granularity" != "xl" ]]; then
        echo "Standalone mode requires --granularity s, m, l, or xl" >&2
        exit 2
      fi
      printf '%s\n' \
        "run.model_family=standalone" \
        "run.sampling_mode=standalone" \
        "run.granularity=$granularity"
      ;;
    *)
      echo "Unknown mode: $mode" >&2
      exit 2
      ;;
  esac
}

run_training_mode() {
  local mode="$1"
  local granularity="${2:-}"
  local run_id="$3"
  local -a python_cmd
  local -a train_args
  mapfile -t python_cmd < <(python_command)

  train_args=(
    train.py
    --config "$CONFIG_PATH"
    "${OUTPUT_ARGS[@]}"
    "${FORWARDED_ARGS[@]}"
    --override "run.run_id=$run_id"
  )

  while IFS= read -r override; do
    train_args+=(--override "$override")
  done < <(mode_overrides "$mode" "$granularity")

  printf 'Launching %s run_id=%s\n' "$mode" "$run_id"
  "${python_cmd[@]}" "${train_args[@]}"
}

emit_omitted_standalone_row() {
  local granularity="$1"
  local output_dir="$OUTPUT_ROOT_VALUE/dmodel256-pilot-comparison"
  local rows_path="$output_dir/pilot_comparison_rows.jsonl"
  local run_id
  run_id="$(mode_run_id standalone "$granularity")"

  mkdir -p "$output_dir"
  printf '{"comparison_id":"%s","run_id":"%s","run_status":"omitted","omit_reason":"standalone baseline not scheduled for capped pilot comparison","model_family":"standalone","granularity":"%s","sampling_mode":"standalone","model_shape_label":"dmodel256","table_reference_label":"matlm_78m","completion_label":"reduced-token-pilot","token_budget":100000000,"effective_world_size":null,"checkpoint_status":"unavailable","checkpoint_path":null,"mismatch_notes":["Standalone %s baseline omitted from this capped pilot comparison."]}\n' \
    "$COMPARISON_ID" "$run_id" "$granularity" "$granularity" >> "$rows_path"
}

run_comparison() {
  run_training_mode nested-random "" "$(mode_run_id nested-random)"
  run_training_mode nested-all "" "$(mode_run_id nested-all)"

  for granularity in s m l xl; do
    if [[ "$RUN_STANDALONE_BASELINES" == "1" ]]; then
      run_training_mode standalone "$granularity" "$(mode_run_id standalone "$granularity")"
    else
      emit_omitted_standalone_row "$granularity"
    fi
  done
}

case "$MODE" in
  comparison)
    run_comparison
    ;;
  nested-random|nested-all)
    if [[ "$RUN_ID_EXPLICIT" == "true" ]]; then
      SELECTED_RUN_ID="$RUN_ID"
    else
      SELECTED_RUN_ID="$(mode_run_id "$MODE")"
    fi
    run_training_mode "$MODE" "" "$SELECTED_RUN_ID"
    ;;
  standalone)
    if [[ "$RUN_ID_EXPLICIT" == "true" ]]; then
      SELECTED_RUN_ID="$RUN_ID"
    else
      SELECTED_RUN_ID="$(mode_run_id standalone "$GRANULARITY")"
    fi
    run_training_mode standalone "$GRANULARITY" "$SELECTED_RUN_ID"
    ;;
  *)
    echo "Unknown --mode: $MODE" >&2
    exit 2
    ;;
esac
