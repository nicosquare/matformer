"""Speculative decoding pair loading and alignment metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import torch
from transformers import AutoTokenizer

from evaluation.validation import configure_model_granularity
from training.run import build_model
from utils.config import load_yaml_config
from utils.metrics import build_speculative_task_rows, write_task_results_csv

DEFAULT_SPECULATIVE_DECODING = {
    "max_draft_tokens": 4,
    "max_new_tokens": 32,
    "batch_size": 1,
    "temperature": 0.0,
    "do_sample": False,
}
SPECULATIVE_SUITE_ID = "speculative-alignment"


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = resolve_speculative_config(
        args.config,
        run_id=args.run_id,
        output_dir=args.output_dir,
    )
    result = run_speculative_evaluation(config)
    task_results_path = result.get("task_results_path")
    if task_results_path is not None:
        print(task_results_path)


def resolve_speculative_config(
    config_path: str | Path,
    run_id: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    run = dict(config.get("run") or {})
    if run_id is not None and run.get("run_id") != run_id:
        raise ValueError(
            f"Requested run_id={run_id}, but config defines run_id={run.get('run_id')}"
        )

    output_root = os.environ.get("OUTPUT_ROOT", run.get("output_root", "outputs"))
    run["output_root"] = str(output_root)
    if output_dir is None:
        run["output_dir"] = str(Path(str(output_root)) / str(run["run_id"]))
    else:
        run["output_dir"] = str(output_dir)
    config["run"] = run

    decoding = dict(DEFAULT_SPECULATIVE_DECODING)
    decoding.update(config.get("decoding") or {})
    config["decoding"] = decoding

    outputs = dict(config.get("outputs") or {})
    outputs.setdefault("task_results_csv", "task_results.csv")
    outputs.setdefault("run_summary_json", "run_summary.json")
    config["outputs"] = outputs
    return config


def run_speculative_evaluation(
    config: Mapping[str, Any],
    pair_loader: Callable[[Mapping[str, Any]], dict[str, dict[str, Any]]] | None = None,
    prompt_loader: Callable[[Mapping[str, Any]], list[str]] | None = None,
    pair_evaluator: Callable[[dict[str, Any], list[str], Mapping[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    pair_loader = pair_loader or load_speculative_model_pairs
    prompt_loader = prompt_loader or load_prompt_texts
    pair_evaluator = pair_evaluator or evaluate_speculative_pair

    prompts = prompt_loader(config)
    decoding = dict(config["decoding"])
    decoding.setdefault(
        "max_prompt_length",
        int(config.get("prompt_set", {}).get("max_prompt_length", 128)),
    )
    pairs = pair_loader(config)
    pair_results = [
        pair_evaluator(pair, prompts, decoding)
        for pair in pairs.values()
    ]
    rows = build_speculative_task_rows(config, pair_results)
    task_results_path = write_task_results_csv(config["run"]["output_dir"], rows)
    return {
        "prompts": prompts,
        "pair_results": pair_results,
        "rows": rows,
        "task_results_path": task_results_path,
    }


def load_speculative_model_pairs(
    config: Mapping[str, Any],
    run_loader: Callable[[str, str | Path], dict[str, Any]] | None = None,
    model_loader: Callable[[Mapping[str, Any], str | None], Any] | None = None,
) -> dict[str, dict[str, Any]]:
    run_loader = run_loader or load_run_artifacts
    model_loader = model_loader or load_model_from_artifacts
    output_root = config["run"]["output_root"]
    pair_config = config["pairs"]
    artifact_cache: dict[str, dict[str, Any]] = {}

    def get_artifact(run_id: str) -> dict[str, Any]:
        if run_id not in artifact_cache:
            artifact_cache[run_id] = run_loader(run_id, output_root)
        return artifact_cache[run_id]

    nested = pair_config["nested"]
    standalone = pair_config["standalone"]
    nested_draft_artifact = get_artifact(nested["draft_run_id"])
    nested_verifier_artifact = get_artifact(nested["verifier_run_id"])
    standalone_draft_artifact = get_artifact(standalone["draft_run_id"])
    standalone_verifier_artifact = get_artifact(standalone["verifier_run_id"])

    return {
        "nested": {
            "pair_type": "nested",
            "pair_id": nested.get("pair_id")
            or (
                f"nested:{nested['draft_run_id']}[{nested['draft_granularity']}]"
                f"->{nested['verifier_run_id']}[{nested['verifier_granularity']}]"
            ),
            "draft": {
                "run_id": nested["draft_run_id"],
                "granularity": nested["draft_granularity"],
                "artifact": nested_draft_artifact,
                "model": model_loader(
                    nested_draft_artifact,
                    nested["draft_granularity"],
                ),
            },
            "verifier": {
                "run_id": nested["verifier_run_id"],
                "granularity": nested["verifier_granularity"],
                "artifact": nested_verifier_artifact,
                "model": model_loader(
                    nested_verifier_artifact,
                    nested["verifier_granularity"],
                ),
            },
        },
        "standalone": {
            "pair_type": "standalone",
            "pair_id": standalone.get("pair_id")
            or (
                f"standalone:{standalone['draft_run_id']}"
                f"->{standalone['verifier_run_id']}"
            ),
            "draft": {
                "run_id": standalone["draft_run_id"],
                "granularity": standalone.get("draft_granularity")
                or standalone_draft_artifact["config"].get("run", {}).get("granularity"),
                "artifact": standalone_draft_artifact,
                "model": model_loader(
                    standalone_draft_artifact,
                    standalone.get("draft_granularity")
                    or standalone_draft_artifact["config"].get("run", {}).get(
                        "granularity"
                    ),
                ),
            },
            "verifier": {
                "run_id": standalone["verifier_run_id"],
                "granularity": standalone.get("verifier_granularity")
                or standalone_verifier_artifact["config"]
                .get("run", {})
                .get("granularity"),
                "artifact": standalone_verifier_artifact,
                "model": model_loader(
                    standalone_verifier_artifact,
                    standalone.get("verifier_granularity")
                    or standalone_verifier_artifact["config"].get("run", {}).get(
                        "granularity"
                    ),
                ),
            },
        },
    }


def load_run_artifacts(run_id: str, output_root: str | Path) -> dict[str, Any]:
    output_root = Path(output_root)
    matches = sorted(output_root.rglob(f"{run_id}/config.json"))
    if not matches:
        raise FileNotFoundError(
            f"Speculative evaluation missing config for run_id={run_id} under {output_root}"
        )
    run_dir = matches[0].parent
    config_path = run_dir / "config.json"
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Speculative evaluation missing run summary: {summary_path}"
        )

    with config_path.open("r", encoding="utf-8") as config_file:
        run_config = json.load(config_file)
    with summary_path.open("r", encoding="utf-8") as summary_file:
        run_summary = json.load(summary_file)

    checkpoint_path = (
        run_summary.get("best_checkpoint_path")
        or run_summary.get("final_checkpoint_path")
        or run_summary.get("checkpoint_path")
    )
    if not checkpoint_path:
        raise ValueError(
            "Speculative evaluation requires an available checkpoint for "
            f"run_id={run_id}"
        )

    return {
        "run_id": run_id,
        "output_dir": str(run_dir),
        "config": run_config,
        "summary": run_summary,
        "checkpoint_path": str(checkpoint_path),
    }


def load_model_from_artifacts(
    artifact: Mapping[str, Any],
    granularity: str | None = None,
) -> Any:
    checkpoint_path = Path(str(artifact["checkpoint_path"]))
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Speculative evaluation missing checkpoint: {checkpoint_path}"
        )

    model = build_model(dict(artifact["config"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError(
            f"Checkpoint for run_id={artifact['run_id']} is missing model_state_dict"
        )
    model.load_state_dict(state_dict)
    configure_model_granularity(model, granularity)
    model.eval()
    return model


def measure_acceptance_and_rollback(
    draft_tokens: torch.Tensor,
    verifier_tokens: torch.Tensor,
) -> dict[str, float | int]:
    if draft_tokens.shape != verifier_tokens.shape:
        raise ValueError(
            "speculative draft and verifier token tensors must match exactly: "
            f"{tuple(draft_tokens.shape)} vs {tuple(verifier_tokens.shape)}"
        )
    if draft_tokens.ndim != 2:
        raise ValueError("speculative token tensors must have shape [batch, proposed]")

    batch_size, proposed_tokens = draft_tokens.shape
    accepted_tokens = 0
    rollback_count = 0

    for draft_row, verifier_row in zip(draft_tokens, verifier_tokens):
        accepted_prefix = accepted_prefix_length(draft_row, verifier_row)
        accepted_tokens += accepted_prefix
        if accepted_prefix < proposed_tokens:
            rollback_count += 1

    total_proposed_tokens = batch_size * proposed_tokens
    acceptance_rate = (
        accepted_tokens / total_proposed_tokens if total_proposed_tokens else 0.0
    )
    rollback_frequency = rollback_count / batch_size if batch_size else 0.0

    return {
        "accepted_tokens": accepted_tokens,
        "proposed_tokens": total_proposed_tokens,
        "rollback_count": rollback_count,
        "sample_count": batch_size,
        "acceptance_rate": float(acceptance_rate),
        "rollback_frequency": float(rollback_frequency),
    }


def measure_throughput_and_latency(
    token_count: int,
    elapsed_seconds: float,
    sample_count: int,
) -> dict[str, float | int]:
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be > 0")
    if sample_count < 0:
        raise ValueError("sample_count must be >= 0")
    if token_count < 0:
        raise ValueError("token_count must be >= 0")

    throughput = token_count / elapsed_seconds
    latency = elapsed_seconds / sample_count if sample_count else 0.0
    return {
        "generated_tokens": int(token_count),
        "elapsed_seconds": float(elapsed_seconds),
        "throughput": float(throughput),
        "latency": float(latency),
    }


def measure_speculative_metrics(
    draft_tokens: torch.Tensor,
    verifier_tokens: torch.Tensor,
    elapsed_seconds: float,
) -> dict[str, float | int]:
    alignment_metrics = measure_acceptance_and_rollback(
        draft_tokens,
        verifier_tokens,
    )
    timing_metrics = measure_throughput_and_latency(
        token_count=int(alignment_metrics["accepted_tokens"]),
        elapsed_seconds=elapsed_seconds,
        sample_count=int(alignment_metrics["sample_count"]),
    )
    return alignment_metrics | timing_metrics


def evaluate_speculative_pair(
    pair: dict[str, Any],
    prompts: list[str],
    decoding: Mapping[str, Any],
    tokenizer_loader: Callable[[Mapping[str, Any]], Any] | None = None,
    generator: Callable[[Any, Any, list[str], Mapping[str, Any]], tuple[torch.Tensor, float]] | None = None,
) -> dict[str, Any]:
    tokenizer_loader = tokenizer_loader or load_tokenizer_from_artifact
    generator = generator or generate_token_blocks

    tokenizer = tokenizer_loader(pair["draft"]["artifact"])
    draft_tokens, draft_elapsed = generator(
        pair["draft"]["model"],
        tokenizer,
        prompts,
        decoding,
    )
    verifier_tokens, verifier_elapsed = generator(
        pair["verifier"]["model"],
        tokenizer,
        prompts,
        decoding,
    )
    metrics = measure_speculative_metrics(
        draft_tokens,
        verifier_tokens,
        elapsed_seconds=draft_elapsed + verifier_elapsed,
    )

    draft_artifact = pair["draft"]["artifact"]
    draft_run = draft_artifact["config"].get("run", {})
    draft_summary = draft_artifact.get("summary", {})
    return {
        "pair_id": pair["pair_id"],
        "pair_type": pair["pair_type"],
        "draft_run_id": pair["draft"]["run_id"],
        "draft_granularity": pair["draft"]["granularity"],
        "verifier_run_id": pair["verifier"]["run_id"],
        "verifier_granularity": pair["verifier"]["granularity"],
        "sampling_mode": draft_summary.get("sampling_mode") or draft_run.get("sampling_mode"),
        "model_shape_label": draft_summary.get("model_shape_label")
        or draft_run.get("model_shape_label")
        or draft_run.get("model_size_label"),
        "model_family_slug": draft_run.get("model_family_slug"),
        "model_size_slug": draft_run.get("model_size_slug"),
        "token_budget_slug": draft_run.get("token_budget_slug"),
        "output_group": draft_run.get("output_group"),
        **metrics,
    }


def load_prompt_texts(config: Mapping[str, Any]) -> list[str]:
    prompt_set = config["prompt_set"]
    sample_count = int(prompt_set["sample_count"])
    prompts = prompt_set.get("prompts")
    if prompts:
        return [str(prompt) for prompt in prompts[:sample_count]]

    path = prompt_set.get("path")
    if not path:
        raise ValueError("speculative prompt_set requires prompts or path")
    prompt_path = Path(str(path))
    if not prompt_path.exists():
        raise FileNotFoundError(f"Speculative prompt set path not found: {prompt_path}")

    text_field = str(prompt_set.get("text_field") or "prompt")
    if prompt_path.suffix == ".json":
        with prompt_path.open("r", encoding="utf-8") as prompt_file:
            payload = json.load(prompt_file)
        if isinstance(payload, list):
            items = payload
        else:
            raise ValueError("Speculative JSON prompt file must contain a list")
        return [
            str(item[text_field] if isinstance(item, Mapping) else item)
            for item in items[:sample_count]
        ]

    prompts = []
    with prompt_path.open("r", encoding="utf-8") as prompt_file:
        for line in prompt_file:
            line = line.strip()
            if not line:
                continue
            if prompt_path.suffix == ".jsonl":
                payload = json.loads(line)
                prompts.append(str(payload[text_field]))
            else:
                prompts.append(line)
            if len(prompts) >= sample_count:
                break
    return prompts


def load_tokenizer_from_artifact(artifact: Mapping[str, Any]):
    model_config = artifact["config"].get("model", {})
    tokenizer_name = model_config.get("tokenizer_name") or model_config.get(
        "base_model_name"
    )
    if not tokenizer_name:
        raise ValueError(
            f"Speculative evaluation requires tokenizer_name or base_model_name for run_id={artifact['run_id']}"
        )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def generate_token_blocks(
    model,
    tokenizer,
    prompts: list[str],
    decoding: Mapping[str, Any],
) -> tuple[torch.Tensor, float]:
    if not prompts:
        raise ValueError("speculative evaluation requires at least one prompt")

    batch_size = int(decoding["batch_size"])
    max_draft_tokens = int(decoding["max_draft_tokens"])
    max_prompt_length = int(decoding.get("max_prompt_length", 128))
    device = next(model.parameters()).device
    generated_batches = []
    total_elapsed = 0.0

    for start in range(0, len(prompts), batch_size):
        prompt_batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        )
        attention_mask = encoded["attention_mask"]
        input_ids = encoded["input_ids"].to(device)
        attention_mask = attention_mask.to(device)
        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_draft_tokens,
            "do_sample": bool(decoding.get("do_sample", False)),
            "pad_token_id": tokenizer.pad_token_id,
        }
        if bool(decoding.get("do_sample", False)):
            generate_kwargs["temperature"] = float(decoding.get("temperature", 1.0))

        start_time = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(**generate_kwargs)
        total_elapsed += time.perf_counter() - start_time
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        generated_batches.append(
            extract_generated_token_blocks(
                generated.cpu(),
                prompt_lengths=prompt_lengths,
                max_new_tokens=max_draft_tokens,
                pad_token_id=int(tokenizer.pad_token_id),
            )
        )

    return torch.cat(generated_batches, dim=0), total_elapsed


def extract_generated_token_blocks(
    generated_sequences: torch.Tensor,
    prompt_lengths: Sequence[int],
    max_new_tokens: int,
    pad_token_id: int,
) -> torch.Tensor:
    rows = []
    for row_index, prompt_length in enumerate(prompt_lengths):
        generated_tail = generated_sequences[row_index, int(prompt_length) :]
        generated_tail = generated_tail[:max_new_tokens]
        if generated_tail.numel() < max_new_tokens:
            padding = torch.full(
                (max_new_tokens - generated_tail.numel(),),
                pad_token_id,
                dtype=generated_sequences.dtype,
            )
            generated_tail = torch.cat([generated_tail, padding], dim=0)
        rows.append(generated_tail)
    return torch.stack(rows, dim=0)


def accepted_prefix_length(
    draft_tokens: torch.Tensor,
    verifier_tokens: torch.Tensor,
) -> int:
    if draft_tokens.shape != verifier_tokens.shape:
        raise ValueError(
            "speculative draft and verifier token rows must match exactly: "
            f"{tuple(draft_tokens.shape)} vs {tuple(verifier_tokens.shape)}"
        )
    for index, (draft_token, verifier_token) in enumerate(
        zip(draft_tokens.tolist(), verifier_tokens.tolist())
    ):
        if draft_token != verifier_token:
            return index
    return int(draft_tokens.numel())


if __name__ == "__main__":
    main()
