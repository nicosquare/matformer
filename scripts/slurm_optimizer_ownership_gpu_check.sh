#!/usr/bin/env bash
#SBATCH --job-name=ownership-gpu-check
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-[05,50,51]
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/ivo.navarrete/ElasticNN/matformer
#SBATCH --output=/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/diagnostics/gpu-check-%j.out
#SBATCH --error=/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/diagnostics/gpu-check-%j.err

# Create the shared diagnostics directory before submitting with sbatch.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
OO_PY=/home/ivo.navarrete/.conda/envs/elasticnn/bin/python
OO_BASE=/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/diagnostics

hostname
git log -1 --oneline
sha256sum tests/test_optimizer_ownership.py tests/test_optimizer_ownership_resume.py
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
printf 'SLURM_JOB_GPUS=%s\n' "${SLURM_JOB_GPUS:-unset}"
nvidia-smi --query-gpu=name,driver_version --format=csv || true
"$OO_PY" - <<'PY'
import torch

assert torch.cuda.is_available(), 'No CUDA GPU visible in the allocation'
assert torch.cuda.is_bf16_supported(), 'Allocated GPU lacks bf16 support'
print('GPU:', torch.cuda.get_device_name(0), flush=True)
print('PyTorch:', torch.__version__, flush=True)
PY

exec "$OO_PY" -m pytest \
  tests/test_optimizer_ownership.py \
  tests/test_optimizer_ownership_resume.py \
  -k cuda -q -rs --tb=short \
  --basetemp="$OO_BASE/gpu-tests-$SLURM_JOB_ID" \
  --junitxml="$OO_BASE/gpu-check-$SLURM_JOB_ID.xml"
