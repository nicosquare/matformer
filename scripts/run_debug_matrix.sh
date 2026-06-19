#!/usr/bin/env bash
set -euo pipefail

# Phase 4 runner: one nested debug run plus matched S/M/L/XL standalone baselines.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
CONFIG_PATH="${CONFIG_PATH:-configs/debug_matrix.yaml}"
NESTED_RUN_ID="${NESTED_RUN_ID:-debug-nested-001}"
BASELINE_ARGS=()
if [[ -n "${BASELINE_GRANULARITIES:-}" ]]; then
  for granularity in ${BASELINE_GRANULARITIES//,/ }; do
    BASELINE_ARGS+=(--granularity "$granularity")
  done
elif [[ -n "${BASELINE_GRANULARITY:-}" ]]; then
  BASELINE_ARGS+=(--granularity "$BASELINE_GRANULARITY")
fi
OUTPUT_ARGS=()
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  OUTPUT_ARGS+=(--output-root "$OUTPUT_ROOT")
fi

if [[ ${#BASELINE_ARGS[@]} -gt 0 ]]; then
  printf 'Debug matrix target: nested MatFormer plus selected standalone baselines\n'
else
  printf 'Debug matrix target: nested MatFormer plus standalone S/M/L/XL baselines\n'
fi
printf 'Config: %s\n' "$CONFIG_PATH"
printf 'Nested run id: %s\n' "$NESTED_RUN_ID"
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  printf 'Output root: %s\n' "$OUTPUT_ROOT"
fi

exec "$PYTHON_BIN" -m src.training.baselines \
  --config "$CONFIG_PATH" \
  --nested-run-id "$NESTED_RUN_ID" \
  "${BASELINE_ARGS[@]}" \
  "${OUTPUT_ARGS[@]}" \
  "$@"
