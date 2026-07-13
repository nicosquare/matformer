#!/usr/bin/env bash
set -euo pipefail

# Phase 4 runner: d_model=256 MatFormer-Llama/SwiGLU pilot comparison path.
# Default comparison scope: nested-random, nested-all, and resolved standalone labels.
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
OUTPUT_ROOT_VALUE="${OUT:-${OUTPUT_ROOT:-outputs}}"

usage() {
  cat <<'USAGE'
Run the Phase 4.7 d_model=256 pilot comparison.

Usage:
  bash scripts/run_dmodel256_pilot.sh [options] [-- train.py args]

Options:
  --config PATH             Pilot config path.
  --mode MODE               comparison, nested-random, nested-all, or standalone.
  --granularity NAME        Standalone granularity from the resolved config.
  --run-id RUN_ID           Run id to write through a config override.
  --output-root PATH        Root for run artifacts.
  --output-dir PATH         Explicit run output directory; implies a single run.
  -h, --help                Show this message.

The default comparison runs nested-random and nested-all. Standalone rows for
the resolved model.granularities are emitted as omitted unless
RUN_STANDALONE_BASELINES=1 is set.
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

if [[ -n "${OUT:-}" && "$HAS_OUTPUT_ROOT_ARG" != "true" ]]; then
  OUTPUT_ARGS+=(--output-root "$OUT")
  OUTPUT_ROOT_VALUE="$OUT"
elif [[ -n "${OUTPUT_ROOT:-}" && "$HAS_OUTPUT_ROOT_ARG" != "true" ]]; then
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
printf 'Output root: %s\n' "$OUTPUT_ROOT_VALUE"

python_command() {
  local -a command_parts
  # Allow the Slurm wrapper to provide a launcher prefix such as
  # "python -m torch.distributed.run --nproc_per_node 4".
  read -r -a command_parts <<< "$PYTHON_BIN"
  printf '%s\n' "${command_parts[@]}"
}

resolved_granularities() {
  local resolver_bin="${PYTHON_CONFIG_BIN:-${PYTHON:-python}}"
  local -a resolver_overrides=()
  local -a resolver_command
  local index

  for ((index = 0; index < ${#FORWARDED_ARGS[@]}; index++)); do
    if [[ "${FORWARDED_ARGS[$index]}" == "--override" ]]; then
      if [[ $((index + 1)) -ge ${#FORWARDED_ARGS[@]} ]]; then
        echo "Missing value for --override" >&2
        exit 2
      fi
      resolver_overrides+=("${FORWARDED_ARGS[$((index + 1))]}")
      index=$((index + 1))
    fi
  done

  read -r -a resolver_command <<< "$resolver_bin"
  local resolved_output=""
  if command -v "${resolver_command[0]}" >/dev/null 2>&1 \
    && resolved_output=$("${resolver_command[@]}" - "$CONFIG_PATH" "${resolver_overrides[@]}" <<'PY'
import sys

from src.utils.config import resolve_run_config


config_path = sys.argv[1]
overrides = sys.argv[2:]
resolved = resolve_run_config(config_path, overrides=overrides)
for granularity in resolved["model"]["granularities"]:
    print(granularity)
PY
  ); then
    if [[ -n "$resolved_output" ]]; then
      printf '%s\n' "$resolved_output"
      return 0
    fi
  fi

  awk '
    /^[[:space:]]+granularities:[[:space:]]*\[/ {
      labels = $0
      sub(/.*\[/, "", labels)
      sub(/\].*/, "", labels)
      gsub(/,/, "", labels)
      count = split(labels, values, /[[:space:]]+/)
      for (position = 1; position <= count; position++) {
        if (values[position] != "") print values[position]
      }
      exit
    }
  ' "$CONFIG_PATH"
}

granularity_is_resolved() {
  local requested="$1"
  local resolved
  while IFS= read -r resolved; do
    if [[ "$resolved" == "$requested" ]]; then
      return 0
    fi
  done < <(resolved_granularities)
  return 1
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
      if [[ -z "$granularity" ]] || ! granularity_is_resolved "$granularity"; then
        echo "Standalone mode requires --granularity from the resolved model.granularities" >&2
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
  local mode_override_output
  mapfile -t python_cmd < <(python_command)

  train_args=(
    train.py
    --config "$CONFIG_PATH"
    "${OUTPUT_ARGS[@]}"
    "${FORWARDED_ARGS[@]}"
    --override "run.run_id=$run_id"
  )

  if ! mode_override_output="$(mode_overrides "$mode" "$granularity")"; then
    return 2
  fi
  while IFS= read -r override; do
    train_args+=(--override "$override")
  done <<< "$mode_override_output"

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
  printf '{"comparison_id":"%s","run_id":"%s","run_status":"omitted","omit_reason":"standalone baseline not scheduled for capped pilot comparison","model_family":"standalone","granularity":"%s","sampling_mode":"standalone","model_shape_label":"dmodel256","completion_label":"run","token_budget":100000000,"effective_world_size":null,"checkpoint_status":"unavailable","checkpoint_path":null}\n' \
    "$COMPARISON_ID" "$run_id" "$granularity" >> "$rows_path"
}

run_comparison() {
  run_training_mode nested-random "" "$(mode_run_id nested-random)"
  run_training_mode nested-all "" "$(mode_run_id nested-all)"

  while IFS= read -r granularity; do
    if [[ "$RUN_STANDALONE_BASELINES" == "1" ]]; then
      run_training_mode standalone "$granularity" "$(mode_run_id standalone "$granularity")"
    else
      emit_omitted_standalone_row "$granularity"
    fi
  done < <(resolved_granularities)
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
