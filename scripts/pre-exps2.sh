set -euo pipefail

export OUT=/nfs-stor/nicolas.avila/results/elasticnn
repo=/l/users/nicolas.avila/dev/references/matformer
script="$repo/scripts/slurm_dmodel256_pilot.sh"
labels='[micro,small,medium,large,full]'
prefixes='{micro: 0.125, small: 0.25, medium: 0.5, large: 0.75, full: 1.0}'
output_group=matformer_llama_148m_20m_tokens
status_python="${PYTHON_BIN:-/home/nicolas.avila/.conda/envs/elasticnn/bin/python}"
max_in_flight=4
heartbeat_active_seconds=900
submitted=0
slurm_user=$(id -un)
active_job_names=$(squeue -h -u "$slurm_user" -o '%j')

cd "$repo"
mkdir -p logs

run_is_completed() {
  local run_dir=$1
  local summary_path="$run_dir/run_summary.json"

  [[ -f "$summary_path" ]] || return 1
  "$status_python" -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as summary_file:
    summary = json.load(summary_file)

status = summary.get("status")
if status is None:
    status = summary.get("continuation_state", {}).get("status")
raise SystemExit(0 if status == "completed" else 1)
' "$summary_path"
}

run_is_active() {
  local run_id=$1
  local run_dir=$2
  local active_job_name

  while IFS= read -r active_job_name
  do
    if [[ "$active_job_name" == "$run_id" ]]; then
      return 0
    fi
  done <<< "$active_job_names"

  local heartbeat_path="$run_dir/heartbeats.jsonl"
  [[ -s "$heartbeat_path" ]] || return 1
  "$status_python" -c '
import json
import sys
from datetime import datetime, timezone

heartbeat_path = sys.argv[1]
maximum_age_seconds = float(sys.argv[2])
with open(heartbeat_path, encoding="utf-8") as heartbeat_file:
    records = [line for line in heartbeat_file if line.strip()]
if not records:
    raise SystemExit(1)

timestamp = json.loads(records[-1]).get("timestamp")
if not timestamp:
    raise SystemExit(1)
observed_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
age_seconds = (datetime.now(timezone.utc) - observed_at).total_seconds()
raise SystemExit(0 if 0 <= age_seconds <= maximum_age_seconds else 1)
' "$heartbeat_path" "$heartbeat_active_seconds"
}

submit_if_needed() {
  local run_id=$1
  shift
  local run_dir="$OUT/$output_group/$run_id"

  if run_is_completed "$run_dir"; then
    printf 'SKIP completed: %s\n' "$run_id"
    return 0
  fi

  if run_is_active "$run_id" "$run_dir"; then
    printf 'SKIP queued/running: %s\n' "$run_id"
    return 0
  fi

  if (( active_count + submitted >= max_in_flight )); then
    printf 'DEFER batch limit reached: %s\n' "$run_id"
    return 0
  fi

  sbatch --job-name="$run_id" --gres=gpu:1 --time=04:00:00 "$script" \
    --repo-root "$repo" --output-root "$OUT" \
    --mode nested-random \
    --run-id "$run_id" \
    "${common_args[@]}" \
    "$@"
  submitted=$((submitted + 1))
  active_job_names+=$'\n'"$run_id"
}

candidate_run_ids=(
  d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-noreset-pc1p0em6-s43
  d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-k1000-pc1p0em6-s43
  d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-k2000-pc1p0em6-s43
  d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-noreset-pc1p0em6-s44
  d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-k1000-pc1p0em6-s44
  d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-k2000-pc1p0em6-s44
)

active_count=0
for candidate_run_id in "${candidate_run_ids[@]}"
do
  candidate_run_dir="$OUT/$output_group/$candidate_run_id"
  if ! run_is_completed "$candidate_run_dir" \
    && run_is_active "$candidate_run_id" "$candidate_run_dir"
  then
    active_count=$((active_count + 1))
  fi
done
available_slots=$((max_in_flight - active_count))
if (( available_slots < 0 )); then
  available_slots=0
fi
printf 'Recognized %d queued/running experiment jobs; %d slots available.\n' \
  "$active_count" "$available_slots"

for seed in 43 44
do
  common_args=(
    --override "run.seed=$seed"
    --override training.token_budget=20000000
    --override training.warmup_steps=400
    --override training.pre_nested_warmup.enabled=true
    --override training.pre_nested_warmup.duration=500
    --override training.pre_nested_warmup.unit=steps
    --override training.pre_nested_warmup.policy=balanced_global
    --override training.pre_nested_warmup.action_interval_steps=50
    --override training.optimizer.preset=adam
    --override training.learning_rate=0.001
    --override training.learning_rate_scale_rule=none
    --override dataset.sample_limit=100000
    --override model.variant=concat
    --override model.correction_mode=gmc
    --override model.membership_correction=true
    --override model.granularity_mode=explicit
    --override "model.granularities=$labels"
    --override "model.granularity_prefixes=$prefixes"
    --override model.granularity_sampling_mode=adaptive_global
    --override model.adaptive_sampler_strategy=thompson
    --override model.adaptive_controller.preset=bayesian_thompson
    --override model.adaptive_controller.decision_interval_steps=50
    --override 'model.adaptive_controller.prior_mean=[0.0,0.0,0.0,0.0,0.0]'
    --override 'model.adaptive_controller.prior_covariance=[1.0e-4,1.0e-6,1.0e-6,1.0e-6,1.0e-6]'
    --override model.adaptive_controller.observation_noise_variance=1.0e-7
    --override model.adaptive_controller.process_noise_covariance=0.0
  )

  submit_if_needed \
    "d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-noreset-pc1p0em6-s${seed}" \
    --override model.adaptive_controller.reset.enabled=false

  submit_if_needed \
    "d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-k1000-pc1p0em6-s${seed}" \
    --override model.adaptive_controller.reset.enabled=true \
    --override model.adaptive_controller.reset.interval_steps=1000 \
    --override model.adaptive_controller.reset.policy=full_prior \
    --override model.adaptive_controller.reset.acquisition_policy=balanced_global \
    --override model.adaptive_controller.reset.acquisition_passes=1

  submit_if_needed \
    "d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-k2000-pc1p0em6-s${seed}" \
    --override model.adaptive_controller.reset.enabled=true \
    --override model.adaptive_controller.reset.interval_steps=2000 \
    --override model.adaptive_controller.reset.policy=full_prior \
    --override model.adaptive_controller.reset.acquisition_policy=balanced_global \
    --override model.adaptive_controller.reset.acquisition_passes=1
done
