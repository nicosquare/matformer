"""Consistency metrics for nested and standalone comparison runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import yaml

from src.training.data import load_and_tokenize_dataset
from src.training.run import build_dataloaders, build_model, load_tokenizer
from src.evaluation.validation import configure_model_granularity
from src.utils.metrics import write_consistency_results_csv


TOKEN_LEVEL_AGREEMENT = "token_level_agreement"
TOP_K_OVERLAP = "top_k_overlap"
KL_DIVERGENCE = "kl_divergence"
DEFAULT_TOP_K_VALUES = (5,)
KL_DIVERGENCE_DEFERRED_REASON = (
    "KL divergence is deferred for a later phase because it requires more "
    "careful probability-space handling and numerical stabilization."
)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    output_root = Path(
        args.output_root
        or config.get("run", {}).get("output_root")
        or os.environ.get("OUTPUT_ROOT")
        or "outputs"
    )
    rows = run_consistency_evaluation(config, output_root=output_root)
    output_dir = Path(
        args.output_dir
        or config.get("run", {}).get("output_dir")
        or (output_root / config.get("run", {}).get("run_id", "consistency-001"))
    )
    artifact_path = write_consistency_results_csv(output_dir, rows)
    if artifact_path is not None:
        print(artifact_path)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as config_file:
        return json.load(config_file) if str(path).endswith(".json") else yaml.safe_load(config_file)


def run_consistency_evaluation(
    config: Mapping[str, Any],
    output_root: str | Path,
) -> list[dict[str, Any]]:
    output_root = Path(output_root)
    run_ids = _load_consistency_run_ids(config)
    nested_artifact = load_run_artifact(output_root / run_ids["nested"])
    nested_config = nested_artifact["config"]
    standalone_artifacts = {
        granularity: load_run_artifact(output_root / run_id)
        for granularity, run_id in run_ids["standalone"].items()
    }

    nested_model = build_model(nested_config)
    _load_model_checkpoint(nested_model, nested_artifact["checkpoint_path"])

    tokenizer = load_tokenizer(nested_config)
    tokenized_dataset = load_and_tokenize_dataset(
        nested_config,
        tokenizer,
        num_proc=nested_config["training"].get("preprocess_num_proc", 1),
    )
    _, eval_dataloader = build_dataloaders(
        nested_config,
        tokenized_dataset,
        device=torch.device("cpu"),
        distributed_context=None,
    )
    eval_batches = list(eval_dataloader)
    if not eval_batches:
        raise ValueError("consistency evaluation dataloader produced no batches")

    rows = []
    for pair in config["consistency"]["pairs"]:
        small_granularity = pair["small_granularity"]
        large_granularity = pair["large_granularity"]
        comparison_id = f"{config['run']['run_id']}__{small_granularity}__{large_granularity}"
        small_run_artifact = standalone_artifacts.get(small_granularity)
        large_run_artifact = standalone_artifacts.get(large_granularity)
        if small_run_artifact is None or large_run_artifact is None:
            raise ValueError(
                "consistency config refers to a standalone granularity "
                f"missing from inputs: {small_granularity}, {large_granularity}"
            )

        small_model = build_model(nested_config)
        large_model = build_model(large_run_artifact["config"])
        _load_model_checkpoint(small_model, nested_artifact["checkpoint_path"])
        _load_model_checkpoint(large_model, large_run_artifact["checkpoint_path"])
        rows.extend(
            _build_rows_for_pair(
                comparison_id,
                nested_config,
                small_model,
                large_model,
                eval_batches,
                small_granularity,
                large_granularity,
                config["consistency"],
                nested_run_id=run_ids["nested"],
                large_run_id=run_ids["standalone"][large_granularity],
            )
        )
    return rows


def _build_rows_for_pair(
    comparison_id: str,
    config: Mapping[str, Any],
    small_model,
    large_model,
    eval_batches,
    small_granularity: str,
    large_granularity: str,
    consistency_config: Mapping[str, Any],
    nested_run_id: str,
    large_run_id: str,
) -> list[dict[str, Any]]:
    top_k_values = consistency_config.get("top_k_values", list(DEFAULT_TOP_K_VALUES))
    batch = eval_batches[0]
    with torch.no_grad():
        configure_model_granularity(small_model, small_granularity)
        configure_model_granularity(large_model, large_granularity)
        small_logits = small_model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        ).logits
        large_logits = large_model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        ).logits
    rows = build_consistency_rows(
        comparison_id=comparison_id,
        small_run_id=nested_run_id,
        large_run_id=large_run_id,
        small_granularity=small_granularity,
        large_granularity=large_granularity,
        small_logits=small_logits,
        large_logits=large_logits,
        attention_mask=batch.get("attention_mask"),
        top_k_values=top_k_values,
        include_deferred_kl_note="kl_divergence" in consistency_config.get("deferred_metrics", []),
    )
    return rows


def _load_consistency_run_ids(config: Mapping[str, Any]) -> dict[str, Any]:
    inputs = config.get("inputs", {})
    return {
        "nested": str(inputs["nested_run_id"]),
        "standalone": {
            str(granularity): str(run_id)
            for granularity, run_id in (inputs.get("standalone_run_ids") or {}).items()
        },
    }


def load_run_artifact(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    checkpoint_path = _select_checkpoint_path(run_dir / "checkpoints")
    return {
        "config": load_resolved_config(config_path),
        "checkpoint_path": checkpoint_path,
    }


def _select_checkpoint_path(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    preferred_names = [
        "best_eval.pt",
        "final.pt",
    ]
    for preferred_name in preferred_names:
        preferred_path = checkpoint_dir / preferred_name
        if preferred_path.exists():
            return preferred_path
    checkpoint_candidates = sorted(checkpoint_dir.glob("*.pt"))
    return checkpoint_candidates[0] if checkpoint_candidates else None


def load_resolved_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Resolved config must be a JSON object: {path}")
    return config


def _load_model_checkpoint(model, checkpoint_path: str | Path | None) -> None:
    if checkpoint_path is None:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)


def token_level_agreement(
    small_logits: torch.Tensor,
    large_logits: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    small_tokens, large_tokens = _masked_argmax_predictions(
        small_logits,
        large_logits,
        attention_mask=attention_mask,
    )
    sample_count = int(small_tokens.numel())
    if sample_count == 0:
        agreement = 0.0
    else:
        agreement = float((small_tokens == large_tokens).float().mean().item())
    return {
        "metric_name": TOKEN_LEVEL_AGREEMENT,
        "metric_value": agreement,
        "sample_count": sample_count,
    }


def top_k_overlap(
    small_logits: torch.Tensor,
    large_logits: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    k: int = 5,
) -> dict[str, Any]:
    _validate_logits_shape(small_logits, large_logits)
    if k <= 0:
        raise ValueError("top-k overlap requires k > 0")

    active_indices = _active_token_indices(small_logits, attention_mask=attention_mask)
    sample_count = int(active_indices.numel())
    if sample_count == 0:
        return {
            "metric_name": TOP_K_OVERLAP,
            "metric_value": 0.0,
            "sample_count": 0,
            "top_k": k,
        }

    vocab_size = int(small_logits.shape[-1])
    effective_k = min(k, vocab_size)
    small_topk = torch.topk(
        small_logits.reshape(-1, vocab_size).index_select(0, active_indices),
        k=effective_k,
        dim=-1,
    ).indices
    large_topk = torch.topk(
        large_logits.reshape(-1, vocab_size).index_select(0, active_indices),
        k=effective_k,
        dim=-1,
    ).indices

    overlaps = []
    for small_ids, large_ids in zip(small_topk, large_topk):
        overlap_count = len(set(small_ids.tolist()) & set(large_ids.tolist()))
        overlaps.append(overlap_count / effective_k)

    return {
        "metric_name": TOP_K_OVERLAP,
        "metric_value": float(sum(overlaps) / len(overlaps)),
        "sample_count": sample_count,
        "top_k": effective_k,
    }


def deferred_kl_divergence_note(
    sample_count: int,
) -> dict[str, Any]:
    return {
        "metric_name": KL_DIVERGENCE,
        "metric_value": None,
        "sample_count": int(sample_count),
        "deferred": True,
        "deferred_reason": KL_DIVERGENCE_DEFERRED_REASON,
    }


def build_consistency_rows(
    comparison_id: str,
    small_run_id: str,
    large_run_id: str,
    small_granularity: str,
    large_granularity: str,
    small_logits: torch.Tensor,
    large_logits: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    top_k_values: Sequence[int] = DEFAULT_TOP_K_VALUES,
    include_deferred_kl_note: bool = True,
) -> list[dict[str, Any]]:
    base_fields = {
        "comparison_id": comparison_id,
        "small_run_id": small_run_id,
        "large_run_id": large_run_id,
        "small_granularity": small_granularity,
        "large_granularity": large_granularity,
    }

    rows = [base_fields | token_level_agreement(small_logits, large_logits, attention_mask)]
    sample_count = rows[0]["sample_count"]

    for k in top_k_values:
        rows.append(base_fields | top_k_overlap(small_logits, large_logits, attention_mask, k=k))

    if include_deferred_kl_note:
        rows.append(base_fields | deferred_kl_divergence_note(sample_count))

    return rows


def summarize_consistency_suite(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(row["metric_name"]): dict(row)
        for row in rows
    }


def _masked_argmax_predictions(
    small_logits: torch.Tensor,
    large_logits: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_logits_shape(small_logits, large_logits)
    active_indices = _active_token_indices(small_logits, attention_mask=attention_mask)
    vocab_size = int(small_logits.shape[-1])
    flat_small = small_logits.reshape(-1, vocab_size).index_select(0, active_indices)
    flat_large = large_logits.reshape(-1, vocab_size).index_select(0, active_indices)
    return flat_small.argmax(dim=-1), flat_large.argmax(dim=-1)


def _active_token_indices(
    logits: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("consistency logits must have shape [batch, seq, vocab]")

    batch_size, sequence_length, _ = logits.shape
    total_positions = batch_size * sequence_length
    if attention_mask is None:
        return torch.arange(total_positions, device=logits.device)

    if attention_mask.shape != logits.shape[:2]:
        raise ValueError(
            "attention_mask must match logits batch and sequence dimensions"
        )
    flat_mask = attention_mask.reshape(-1).to(dtype=torch.bool, device=logits.device)
    return flat_mask.nonzero(as_tuple=False).flatten()


def _validate_logits_shape(
    small_logits: torch.Tensor,
    large_logits: torch.Tensor,
) -> None:
    if small_logits.shape != large_logits.shape:
        raise ValueError(
            f"consistency logits must match exactly: {small_logits.shape} vs {large_logits.shape}"
        )
    if small_logits.ndim != 3:
        raise ValueError("consistency logits must have shape [batch, seq, vocab]")


if __name__ == "__main__":
    main()
