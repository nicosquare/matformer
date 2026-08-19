"""Run a small, matched PanelGrad versus uniform-global comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset

import src.training.data as training_data
import src.training.modeling as training_modeling
from src.evaluation.validation import evaluate_validation_per_granularity
from src.training.run import prepare_controller_data_roles, run_training
from src.utils.config import resolve_run_config
from src.utils.metrics import write_json_artifact
from src.utils.reproducibility import seed_model_initialization


RUNS = (
    (
        "panelgrad",
        "configs/opt-in_exps/panelgrad_smoke.yaml",
        "panelgrad-opt-in-smoke-001",
    ),
    (
        "uniform_global",
        "configs/opt-in_exps/panelgrad_uniform_baseline.yaml",
        "panelgrad-uniform-baseline-001",
    ),
)


def _comparison_overrides() -> list[str]:
    return [
        "model.d_model=8",
        "model.num_layers=1",
        "model.num_attention_heads=1",
        "model.vocab_size=32",
        "model.context_length=4",
        "training.max_steps=3",
        "training.batch_size_per_process=128",
        "training.eval_batches=1",
        "training.eval_interval=0",
        "evaluation.validation.enabled=true",
        "evaluation.validation.interval_steps=0",
        "evaluation.validation.holdout.examples=128",
        "evaluation.validation.run_at_completion=true",
        "run.continuation.enabled=false",
        "outputs.save_checkpoints=false",
    ]


def _synthetic_dataset() -> Dataset:
    examples = 900
    return Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4] for _ in range(examples)],
            "attention_mask": [[1, 1, 1, 1] for _ in range(examples)],
        }
    )


def _model_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _final_holdout_results(config, model, dataset):
    _, _, _, partition = prepare_controller_data_roles(
        config,
        dataset,
        torch.device("cpu"),
    )
    final_loader = training_data.build_language_model_dataloader(
        partition["datasets"]["final_holdout"],
        batch_size=int(config["training"]["batch_size_per_process"]),
        num_workers=0,
        pin_memory=False,
    )
    return evaluate_validation_per_granularity(
        model,
        final_loader,
        list(config["model"]["granularities"]),
        device="cpu",
        config=config,
    )


def _compact_evaluation_rows(rows):
    return [
        {
            "granularity": row["granularity"],
            "loss": row["loss"],
            "perplexity": row["perplexity"],
            "evaluation_target_tokens": row["evaluation_target_tokens"],
        }
        for row in rows
    ]


def _measurement_cost(result, controller_summary):
    if controller_summary is None:
        return {
            "refresh_count": 0,
            "backward_evaluations": 0,
            "controller_target_tokens": 0,
            "controller_examples": 0,
            "duration_seconds": 0.0,
        }
    refresh_events = []
    metrics_path = controller_summary.get("controller_metrics_path")
    if metrics_path is not None:
        for line in Path(metrics_path).read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("event_type") == "panelgrad_refresh_completed":
                refresh_events.append(event)
    return {
        "refresh_count": int(controller_summary.get("refresh_count", 0)),
        "backward_evaluations": int(
            controller_summary.get("cumulative_backward_evaluations", 0)
        ),
        "controller_target_tokens": sum(
            int(
                event.get(
                    "controller_target_evaluation_count",
                    int(event["controller_target_count"])
                    * len(event["measurements"]),
                )
            )
            for event in refresh_events
        ),
        "controller_examples": sum(
            int(
                event.get(
                    "controller_packed_sequence_evaluation_count",
                    int(
                        event.get(
                            "controller_packed_sequence_count",
                            event["controller_example_count"],
                        )
                    )
                    * len(event["measurements"]),
                )
            )
            for event in refresh_events
        ),
        "duration_seconds": float(
            controller_summary.get("cumulative_measurement_duration_seconds", 0.0)
        ),
    }


def run_comparison(output_root: Path) -> Path:
    dataset = _synthetic_dataset()
    records = {}
    manifest_reference = None
    initial_model_reference = None
    for method, config_path, run_id in RUNS:
        output_dir = output_root / run_id
        if output_dir.exists():
            # Comparison runs are disposable and must not append to an earlier journal.
            shutil.rmtree(output_dir)
        config = resolve_run_config(
            config_path,
            output_dir=output_dir,
            overrides=_comparison_overrides(),
        )
        seed_model_initialization(config)
        model = training_modeling.build_model(config)
        initial_model_hash = _model_state_hash(model)
        if initial_model_reference is None:
            initial_model_reference = initial_model_hash
        elif initial_model_hash != initial_model_reference:
            raise RuntimeError("comparison runs started from different model parameters")
        started = time.perf_counter()
        result = run_training(
            config,
            model=model,
            tokenized_dataset=dataset,
            device="cpu",
        )
        training_seconds = time.perf_counter() - started
        final_results = _final_holdout_results(config, model, dataset)
        validation_rows = [
            row for row in result["metrics_rows"] if row["split"] == "validation"
        ]
        training_rows = [
            row for row in result["metrics_rows"] if row["split"] == "train"
        ]
        exposures = Counter(str(row["granularity"]) for row in training_rows)
        controller_summary = None
        if result["controller_summary_path"] is not None:
            controller_summary = json.loads(
                result["controller_summary_path"].read_text(encoding="utf-8")
            )
            exposures = Counter(controller_summary.get("exposure_counts", exposures))
        manifests = {
            field: config.get(field)
            for field in (
                "data_roles_manifest_hash",
                "optimizer_training_manifest_hash",
                "controller_manifest_hash",
                "validation_manifest_hash",
                "final_holdout_manifest_hash",
            )
        }
        if manifest_reference is None:
            manifest_reference = manifests
        elif manifests != manifest_reference:
            raise RuntimeError("comparison runs resolved different data-role manifests")
        records[method] = {
            "run_id": run_id,
            "initial_model_hash": initial_model_hash,
            "optimizer_steps": len(training_rows),
            "training_tokens": max(
                (int(row["tokens_seen"]) for row in training_rows), default=0
            ),
            "target_token_budget": config["training"]["token_budget"],
            "exposure_counts": dict(exposures),
            "sampling_distribution": (
                {
                    label: probability
                    for label, probability in zip(
                        controller_summary["ordered_granularities"],
                        controller_summary["final_p"],
                    )
                }
                if controller_summary is not None
                else {
                    label: 1.0 / len(config["model"]["granularities"])
                    for label in config["model"]["granularities"]
                }
            ),
            "validation_metrics": _compact_evaluation_rows(validation_rows),
            "final_holdout_metrics": _compact_evaluation_rows(final_results),
            "training_wall_clock_seconds": training_seconds,
            "measurement_cost": _measurement_cost(result, controller_summary),
            "manifests": manifests,
        }

    payload = {
        "schema_version": 1,
        "comparison": "panelgrad_vs_uniform_global",
        "matching_rule": "same model, data-role manifests, root seed, optimizer, scheduler, validation, and exactly three completed optimizer steps",
        "selection_rule": "use the final in-memory model after exactly three optimizer steps; evaluate the untouched final holdout once after training",
        "matched_compute_claim": False,
        "measurement_cost_reported_separately": True,
        "runs": records,
    }
    output_path = output_root / "comparison.json"
    write_json_artifact(output_path, payload)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/panelgrad-comparison"),
    )
    args = parser.parse_args()
    print(run_comparison(args.output_root))


if __name__ == "__main__":
    main()
