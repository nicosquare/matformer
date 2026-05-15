#!/usr/bin/env bash
set -euo pipefail

# Phase 4 runner: first paper-aligned 78M reduced-token pilot path.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
CONFIG_PATH="${CONFIG_PATH:-configs/78m_reduced_pilot.yaml}"
DEFAULT_RUN_ID="78m-reduced-pilot-001"
if [[ -n "${RUN_ID:-}" ]]; then
  RUN_ID_EXPLICIT=true
else
  RUN_ID="$DEFAULT_RUN_ID"
  RUN_ID_EXPLICIT=false
fi
OUTPUT_ARGS=()
HAS_OUTPUT_ROOT_ARG=false

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
      HAS_OUTPUT_ROOT_ARG=true
      shift 2
      ;;
    --output-dir)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --output-dir" >&2
        exit 2
      fi
      OUTPUT_ARGS+=(--output-dir "$2")
      shift 2
      ;;
    *)
      OUTPUT_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "${OUTPUT_ROOT:-}" && "$HAS_OUTPUT_ROOT_ARG" != "true" ]]; then
  OUTPUT_ARGS+=(--output-root "$OUTPUT_ROOT")
fi

printf '78M pilot target: paper-aligned reduced-token pilot\n'
printf 'Config: %s\n' "$CONFIG_PATH"
printf 'Run id: %s\n' "$RUN_ID"
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  printf 'Output root: %s\n' "$OUTPUT_ROOT"
fi

if [[ "$RUN_ID_EXPLICIT" == "true" ]]; then
  OUTPUT_ARGS+=(--override "run.run_id=$RUN_ID")
fi

exec "$PYTHON_BIN" train.py \
  --config "$CONFIG_PATH" \
  "${OUTPUT_ARGS[@]}"
