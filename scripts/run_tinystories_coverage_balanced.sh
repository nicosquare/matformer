#!/usr/bin/env bash
set -euo pipefail

campaign_root="$1"
source_root="$campaign_root/source"
python_bin='/home/ivo.navarrete/.conda/envs/elasticnn/bin/python'

export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export SLURM_CONF="$campaign_root/launchers/slurm-client.conf"
cd "$source_root"

"$python_bin" - "$campaign_root/campaign/source_manifest.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
for name, expected in manifest.items():
    actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"Snapshot source changed: {name}")
print(f"Verified {len(manifest)} source files", flush=True)
PY

sha256sum -c "$campaign_root/campaign/inputs.sha256"
"$python_bin" scripts/train_cuda_required.py \
  --config "$campaign_root/campaign/S1-coverage.yaml" \
  --entry-evidence "$campaign_root/launchers/cuda-entry.json"
