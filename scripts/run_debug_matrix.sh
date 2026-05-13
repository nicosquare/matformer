#!/usr/bin/env bash
set -euo pipefail

# Phase 3 runner: one nested debug run plus one matched standalone baseline.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
CONFIG_PATH="${CONFIG_PATH:-configs/debug_matrix.yaml}"
NESTED_RUN_ID="${NESTED_RUN_ID:-debug-nested-001}"
BASELINE_GRANULARITY="${BASELINE_GRANULARITY:-s}"
OUTPUT_ARGS=()
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  OUTPUT_ARGS+=(--output-root "$OUTPUT_ROOT")
fi

printf 'Debug matrix target: nested MatFormer plus standalone %s baseline\n' "$BASELINE_GRANULARITY"
printf 'Config: %s\n' "$CONFIG_PATH"
printf 'Nested run id: %s\n' "$NESTED_RUN_ID"
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  printf 'Output root: %s\n' "$OUTPUT_ROOT"
fi

exec "$PYTHON_BIN" -m training.baselines \
  --config "$CONFIG_PATH" \
  --nested-run-id "$NESTED_RUN_ID" \
  --granularity "$BASELINE_GRANULARITY" \
  "${OUTPUT_ARGS[@]}" \
  "$@"
