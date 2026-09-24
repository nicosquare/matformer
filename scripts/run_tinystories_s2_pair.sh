#!/usr/bin/env bash

# Slurm array worker for the six S2 warmup/peak-LR runs. Each array task
# executes two independent runs sequentially on its one allocated GPU.
set -u -o pipefail

if [[ $# -ne 1 || -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo 'Usage: sbatch --array=0-2%2 ... run_tinystories_s2_pair.sh CAMPAIGN_ROOT' >&2
  exit 2
fi

campaign_root="$1"
source_root='/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-s1-warmup-v1/source'
python_bin='/home/ivo.navarrete/.conda/envs/elasticnn/bin/python'
case "$SLURM_ARRAY_TASK_ID" in
  0) arms=(S2-linear-w256 S2-linear-lr0004) ;;
  1) arms=(S2-geometric-w256 S2-geometric-lr0004) ;;
  2) arms=(S2-linear-lr0012 S2-geometric-lr0012) ;;
  *) echo "Unexpected Slurm array task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

export SLURM_CONF="$campaign_root/launchers/slurm-client.conf"
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg

cd "$campaign_root" || exit 2
sha256sum -c campaign/configs.sha256 || exit 2
status_path="$campaign_root/launchers/pair-${SLURM_ARRAY_TASK_ID}-${SLURM_JOB_ID}.txt"
printf 'job=%s task=%s node=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$(hostname)" > "$status_path"

failed=0
for arm in "${arms[@]}"; do
  echo "Starting $arm in Slurm job $SLURM_JOB_ID on $(hostname)"
  printf 'start=%s arm=%s\n' "$(date -u +%FT%TZ)" "$arm" >> "$status_path"
  "$python_bin" "$source_root/scripts/train_cuda_required.py" \
    --config "$campaign_root/campaign/configs/$arm.yaml" \
    --entry-evidence "$campaign_root/launchers/cuda-entry-$arm.json"
  result=$?
  printf 'finish=%s arm=%s returncode=%s\n' "$(date -u +%FT%TZ)" "$arm" "$result" >> "$status_path"
  echo "Finished $arm with return code $result"
  if [[ "$result" -ne 0 ]]; then failed=1; fi
done
exit "$failed"
