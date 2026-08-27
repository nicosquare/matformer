#!/usr/bin/env bash

# Source this file so the selected TinyStories data contract remains in the
# current shell. TINYSTORIES_PROFILE is intentionally required: an implicit
# default could send a campaign to the wrong corpus.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this script: export TINYSTORIES_PROFILE=...; source ${BASH_SOURCE[0]}" >&2
  exit 2
fi

if ! PYTHON_BIN="$(command -v python)" || [[ ! -x "$PYTHON_BIN" ]]; then
  echo "The active environment does not provide python" >&2
  return 2
fi

export PYTHON_BIN
export NFS_USER_ROOT="${NFS_USER_ROOT:-/nfs-stor/$USER}"
export MATFORMER_TOKENIZER_ROOT="${MATFORMER_TOKENIZER_ROOT:-$NFS_USER_ROOT/matformer-tokenizers}"
export MATFORMER_CORPUS_ROOT="${MATFORMER_CORPUS_ROOT:-$NFS_USER_ROOT/matformer-corpora}"
export MATFORMER_EXPERIMENT_ROOT="${MATFORMER_EXPERIMENT_ROOT:-$NFS_USER_ROOT/results/elasticnn}"
export HF_HOME="${HF_HOME:-$NFS_USER_ROOT/huggingface}"
export BASE=configs/controlled_exps/tinystories_controlled_convergence.yaml

case "${TINYSTORIES_PROFILE:-}" in
  stories)
    export PROFILE_SLUG=tinystories
    export RUN_PREFIX=tiny
    export TOKENIZER_NAME=tinystories-sentencepiece-bpe-2k-v1
    export CORPUS_NAME=tinystories-packed-full-v1
    export DATASET_NAME=roneneldan/TinyStories
    export DATASET_CONFIG_NAME=default
    export DATASET_SPLIT=train+validation
    export DATASET_PHASE=tinystories_controlled
    export PREPROCESSING_NOTES=immutable_nonempty_huggingface_rows_contiguous_eos_uint32_v1
    export EXPERIMENT_NAME=tinystories-frozen-elastic-v2
    export EXPERIMENT_PHASE=tinystories_frozen_elastic
    export PROFILE_RECIPE_STATUS=frozen
    export CAPACITY_CONFIG=configs/controlled_exps/tinystories_capacity_converged.yaml
    ;;
  instruct)
    export PROFILE_SLUG=tinystories-instruct
    export RUN_PREFIX=tiny-instruct
    export TOKENIZER_NAME=tinystories-instruct-sentencepiece-bpe-2k-v1
    export CORPUS_NAME=tinystories-instruct-packed-full-v1
    export DATASET_NAME=roneneldan/TinyStoriesInstruct
    export DATASET_CONFIG_NAME=default
    export DATASET_SPLIT=train+validation
    export DATASET_PHASE=tinystories_instruct_controlled
    export PREPROCESSING_NOTES=complete_record_lf_join_preserve_fields_and_internal_newlines_v1
    export EXPERIMENT_NAME=tinystories-instruct-recipe-selection-v1
    export EXPERIMENT_PHASE=tinystories_instruct_recipe_selection
    export PROFILE_RECIPE_STATUS=calibration
    export CAPACITY_CONFIG=configs/controlled_exps/tinystories_instruct_capacity_converged.yaml
    ;;
  *)
    echo "TINYSTORIES_PROFILE must be stories or instruct" >&2
    return 2
    ;;
esac

export TOKENIZER="$MATFORMER_TOKENIZER_ROOT/$TOKENIZER_NAME"
export CORPUS="$MATFORMER_CORPUS_ROOT/$CORPUS_NAME"

printf 'profile=%s dataset=%s tokenizer=%s corpus=%s\n' \
  "$TINYSTORIES_PROFILE" "$DATASET_NAME" "$TOKENIZER" "$CORPUS"
