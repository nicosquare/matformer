"""Model construction and config-sync helpers for training runs."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from src.models.ffn import CatLlamaMLP
from src.models.wiring import (
    ModifiedLlamaForCausalLM,
    prime_standalone_granularity_state,
)
from src.training.distributed import should_write_shared_artifact
from src.utils.config import resolve_training_length_for_world_size
from src.utils.metrics import build_parameter_counts_by_granularity

__all__ = [
    "build_artifact_parameter_counts",
    "build_llama_config",
    "build_model",
    "distributed_summary_fields",
    "load_tokenizer",
    "sync_config_with_distributed_context",
]


def build_artifact_parameter_counts(
    config: dict[str, Any],
    model,
    distributed_context,
) -> dict[str, dict[str, Any]]:
    if not should_write_shared_artifact(distributed_context):
        return {}
    return build_parameter_counts_by_granularity(
        model,
        config["model"]["granularities"],
    )


def build_model(config: dict[str, Any]):
    llama_config = build_llama_config(config)
    model_config = config["model"]
    membership_correction_enabled = model_config.get(
        "membership_correction",
        model_config.get(
            "gradient_membership_correction",
            model_config["variant"] == "concat",
        ),
    )
    mlp_kwargs = {
        "trained_granularities": tuple(model_config["granularities"]),
        "gradient_membership_correction_enabled": config["model"].get(
            "membership_correction",
            membership_correction_enabled,
        ),
    }
    if config["run"]["model_family"] == "standalone":
        model = LlamaForCausalLM(llama_config)
        prime_standalone_granularity_state(
            model,
            config["run"]["granularity"],
            run_id=config["run"].get("run_id"),
        )
        return model

    if config["model"]["variant"] == "concat":
        return ModifiedLlamaForCausalLM(
            llama_config,
            mlp_cls=CatLlamaMLP,
            mlp_kwargs=mlp_kwargs,
        )

    return ModifiedLlamaForCausalLM(llama_config, mlp_kwargs=mlp_kwargs)


def build_llama_config(config: dict[str, Any]) -> LlamaConfig:
    model = config["model"]
    llama_config = LlamaConfig(
        vocab_size=model["vocab_size_assumption"],
        hidden_size=model.get("d_model", model.get("hidden_size")),
        intermediate_size=model["intermediate_size"],
        num_hidden_layers=model["num_layers"],
        num_attention_heads=model["num_attention_heads"],
        num_key_value_heads=model["num_attention_heads"],
        max_position_embeddings=model["context_length"],
        tie_word_embeddings=False,
        use_cache=False,
    )
    if "d_model" in model or "hidden_size" in model:
        llama_config.d_model = model.get("d_model", model.get("hidden_size"))
    if "granularities" in model:
        llama_config.granularities = list(model["granularities"])
    if "granularity_prefixes" in model:
        llama_config.granularity_prefixes = copy.deepcopy(
            model["granularity_prefixes"]
        )
    if "matformer_source_granularity_prefixes" in model:
        llama_config.matformer_source_granularity_prefixes = copy.deepcopy(
            model["matformer_source_granularity_prefixes"]
        )
    if "ffn_prefix_metadata" in model:
        llama_config.ffn_prefix_metadata = copy.deepcopy(model["ffn_prefix_metadata"])
    if "ffn_concat_block_metadata" in model:
        llama_config.ffn_concat_block_metadata = copy.deepcopy(
            model["ffn_concat_block_metadata"]
        )
    if "matformer_source_intermediate_size" in model:
        llama_config.matformer_source_intermediate_size = model[
            "matformer_source_intermediate_size"
        ]
    return llama_config


def load_tokenizer(config: dict[str, Any]):
    model = config["model"]
    dataset = config["dataset"]
    tokenizer_name = dataset.get("tokenizer_name") or model.get("tokenizer_name")
    tokenizer_name = tokenizer_name or model["base_model_name"]
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except OSError:
        # When the hub is unreachable, use the locally cached snapshot if one exists.
        cached_snapshot = snapshot_download(
            repo_id=tokenizer_name,
            local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(cached_snapshot)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def sync_config_with_distributed_context(config: dict[str, Any], context) -> None:
    fsdp_config = copy.deepcopy(getattr(context, "fsdp_config", {}) or {})
    resolve_training_length_for_world_size(
        config,
        effective_world_size=context.world_size,
        world_size_source="distributed_context" if context.enabled else "single_process",
    )
    distributed_config = config["training"].setdefault("distributed", {})
    distributed_config.update(
        {
            "enabled": bool(context.enabled),
            "strategy": context.strategy,
            "rank": context.rank,
            "local_rank": context.local_rank,
            "world_size": context.world_size,
            "fsdp": fsdp_config,
        }
    )


def distributed_summary_fields(context) -> dict[str, Any]:
    fsdp_config = copy.deepcopy(getattr(context, "fsdp_config", {}) or {})
    return {
        "distributed_strategy": context.strategy,
        "distributed_rank": context.rank,
        "distributed_local_rank": context.local_rank,
        "distributed_world_size": context.world_size,
        "distributed_fsdp_config": fsdp_config,
    }
