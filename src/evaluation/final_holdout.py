"""Post-training-only evaluation of the fixed probabilistic final holdout."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from src.evaluation.validation import evaluate_validation_per_granularity
from src.training.data import build_language_model_dataloader, load_and_tokenize_dataset
from src.training.packed_corpus import PackedMMapDataset
from src.training.modeling import build_model, load_tokenizer
from src.utils.metrics import write_json_artifact
from src.utils.reproducibility import stable_hash


FINAL_HOLDOUT_RESULT_VERSION = 1
FINAL_HOLDOUT_EXAMPLE_COUNT = 512
FINAL_HOLDOUT_AGGREGATION = "target_token_weighted_causal_shift_float64"
BAYESIAN_METHOD_FAMILY = "bayesian_gaussian_linear_thompson"


class FinalHoldoutError(RuntimeError):
    """Raised when a final comparison would violate saved run provenance."""


def resolve_existing_final_holdout_result(
    run_dir: str | Path,
) -> dict[str, Any] | None:
    """Return an existing complete result, rejecting malformed artifacts."""

    run_directory = _require_run_directory(run_dir)
    result_path = run_directory / "final_holdout_results.json"
    if not result_path.exists():
        return None

    result = _read_json(result_path)
    summary = _read_json(run_directory / "run_summary.json")
    configured_result_path = result.get("result_path")
    if (
        result.get("schema_version") != FINAL_HOLDOUT_RESULT_VERSION
        or summary.get("status") != "completed"
        or result.get("run_id") != summary.get("run_id")
        or configured_result_path in (None, "")
        or not _paths_equal(result_path, Path(str(configured_result_path)))
    ):
        raise FinalHoldoutError(
            "Existing final holdout result is incompatible or incomplete"
        )

    result_without_hash = dict(result)
    result_hash = result_without_hash.pop("result_hash", None)
    if result_hash != stable_hash(result_without_hash):
        raise FinalHoldoutError("Existing final holdout result hash mismatch")
    return result


def resolve_final_holdout_checkpoint(
    run_dir: str | Path,
    checkpoint_path: str | Path | None = None,
) -> Path:
    """Require a completed run and resolve an explicit or validation-selected checkpoint."""

    run_directory = _require_run_directory(run_dir)
    summary = _read_json(run_directory / "run_summary.json")
    if summary.get("status") != "completed":
        raise FinalHoldoutError(
            "Final holdout evaluation requires a completed training run"
        )

    if checkpoint_path is not None:
        return _resolve_existing_checkpoint(run_directory, checkpoint_path)

    selected_path = summary.get("best_checkpoint_path")
    if summary.get("checkpoint_status") != "best_eval" or not selected_path:
        raise FinalHoldoutError(
            "No ordinary-validation-selected checkpoint is available; provide an "
            "explicit checkpoint"
        )
    return _resolve_existing_checkpoint(run_directory, selected_path)


def validate_final_holdout_provenance(
    run_dir: str | Path,
    checkpoint_path: str | Path,
    *,
    checkpoint_was_explicit: bool = False,
) -> dict[str, Any]:
    """Validate immutable run, manifest, method, and checkpoint provenance."""

    run_directory = _require_run_directory(run_dir)
    config = _read_json(run_directory / "config.json")
    summary = _read_json(run_directory / "run_summary.json")
    manifest = _read_json(run_directory / "final_holdout_manifest.json")
    controller_summary_path = run_directory / "controller_summary.json"
    controller_summary = (
        _read_json(controller_summary_path)
        if controller_summary_path.is_file()
        else {}
    )
    resolved_checkpoint = _resolve_existing_checkpoint(
        run_directory,
        checkpoint_path,
    )
    checkpoint = _load_checkpoint(resolved_checkpoint)

    if summary.get("status") != "completed":
        raise FinalHoldoutError(
            "Final holdout evaluation requires a completed training run"
        )

    artifact_hashes = _validate_training_artifact_hashes(
        run_directory,
        summary,
        controller_summary,
    )

    run_id = _config_run_id(config)
    for artifact_name, artifact_run_id in (
        ("run summary", summary.get("run_id")),
        ("checkpoint", checkpoint.get("run_id")),
    ):
        if artifact_run_id not in (None, run_id):
            raise FinalHoldoutError(f"Final holdout {artifact_name} run ID mismatch")

    _validate_manifest(manifest)
    final_manifest_hash = str(manifest["manifest_hash"])
    config_hashes = _role_hashes(config)
    summary_hashes = _role_hashes(summary)
    checkpoint_hashes = _checkpoint_role_hashes(checkpoint)
    if config_hashes.get("final_holdout") != final_manifest_hash:
        raise FinalHoldoutError("Final holdout manifest hash mismatch in config")
    for artifact_name, role_hashes in (
        ("run summary", summary_hashes),
        ("checkpoint", checkpoint_hashes),
    ):
        for role_name, configured_hash in config_hashes.items():
            artifact_hash = role_hashes.get(role_name)
            if (
                configured_hash not in (None, "")
                and artifact_hash not in (None, "")
                and artifact_hash != configured_hash
            ):
                raise FinalHoldoutError(
                    f"Final holdout {role_name} manifest provenance mismatch "
                    f"between config and {artifact_name}"
                )
        artifact_final_hash = role_hashes.get("final_holdout")
        if artifact_final_hash != final_manifest_hash:
            raise FinalHoldoutError(
                f"Final holdout manifest hash mismatch in {artifact_name}"
            )

    final_contract = config.get("evaluation", {}).get("final_holdout", {})
    if not isinstance(final_contract, Mapping) or (
        final_contract.get("enabled") is not True
        or int(final_contract.get("examples", -1)) != FINAL_HOLDOUT_EXAMPLE_COUNT
        or final_contract.get("fixed_manifest") is not True
        or final_contract.get("evaluate_during_training") is not False
    ):
        raise FinalHoldoutError("Saved final holdout contract is incompatible")

    controller_config = config.get("model", {}).get("adaptive_controller", {})
    checkpoint_controller = checkpoint.get("probabilistic_controller_state", {})
    if not isinstance(checkpoint_controller, Mapping):
        checkpoint_controller = {}
    config_family = controller_config.get("method_family")
    checkpoint_family = checkpoint_controller.get(
        "method_family",
        checkpoint.get("method_family"),
    )
    config_version = controller_config.get("method_version")
    checkpoint_version = checkpoint_controller.get(
        "method_version",
        checkpoint.get("method_version"),
    )
    has_controller = bool(config_family or checkpoint_family)
    if has_controller:
        if config_family != BAYESIAN_METHOD_FAMILY or checkpoint_family != config_family:
            raise FinalHoldoutError("Final holdout checkpoint method family mismatch")
        if checkpoint_version != config_version:
            raise FinalHoldoutError("Final holdout checkpoint method version mismatch")

    selected_checkpoint = summary.get("best_checkpoint_path")
    is_validation_selected = (
        summary.get("checkpoint_status") == "best_eval"
        and selected_checkpoint not in (None, "")
        and _paths_equal(
            resolved_checkpoint,
            _resolve_checkpoint_candidate(run_directory, selected_checkpoint),
        )
    )
    if not checkpoint_was_explicit and not is_validation_selected:
        raise FinalHoldoutError(
            "Final holdout checkpoint was not selected by ordinary validation; "
            "provide it explicitly"
        )
    selection_source = (
        "explicit_checkpoint" if checkpoint_was_explicit else "ordinary_validation"
    )
    return {
        "run_id": run_id,
        "checkpoint_path": str(resolved_checkpoint),
        "checkpoint_selection_provenance": {
            "source": selection_source,
            "checkpoint_status": checkpoint.get("checkpoint_status"),
            "selection_step": checkpoint.get(
                "checkpoint_selection_step",
                summary.get("checkpoint_selection_step"),
            ),
            "metric": checkpoint.get(
                "checkpoint_metric",
                summary.get("checkpoint_metric"),
            ),
            "metric_value": checkpoint.get(
                "checkpoint_metric_value",
                summary.get("checkpoint_metric_value"),
            ),
        },
        "final_holdout_manifest_hash": final_manifest_hash,
        **artifact_hashes,
        "config": config,
        "manifest": manifest,
        "checkpoint": checkpoint,
    }


def evaluate_final_holdout(
    run_dir: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    model=None,
    tokenized_dataset=None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Evaluate every saved granularity and write one separately hashed result."""

    run_directory = _require_run_directory(run_dir)
    explicit_checkpoint = checkpoint_path is not None
    resolved_checkpoint = resolve_final_holdout_checkpoint(
        run_directory,
        checkpoint_path=checkpoint_path,
    )
    provenance = validate_final_holdout_provenance(
        run_directory,
        resolved_checkpoint,
        checkpoint_was_explicit=explicit_checkpoint,
    )
    config = provenance.pop("config")
    manifest = provenance.pop("manifest")
    checkpoint = provenance.pop("checkpoint")
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if model is None:
        model = build_model(config)
    model.load_state_dict(dict(checkpoint["model_state_dict"]))
    model = model.to(resolved_device)

    if tokenized_dataset is None and config.get("dataset", {}).get("mode") == "packed_mmap":
        final_dataset = PackedMMapDataset(
            config["dataset"]["prepared_corpus_dir"],
            "final_holdout",
        )
    elif tokenized_dataset is None:
        tokenizer = load_tokenizer(config)
        tokenized_dataset = load_and_tokenize_dataset(
            config,
            tokenizer,
            num_proc=int(config.get("training", {}).get("preprocess_num_proc", 1)),
        )
        final_dataset = _select_manifest_dataset(tokenized_dataset, manifest)
    else:
        final_dataset = _select_manifest_dataset(tokenized_dataset, manifest)
    batch_size = int(
        config.get("evaluation", {})
        .get("validation", {})
        .get(
            "batch_size_per_process",
            config.get("training", {}).get("batch_size_per_process", 32),
        )
    )
    dataloader = build_language_model_dataloader(
        final_dataset,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=int(config.get("training", {}).get("dataloader_num_workers", 0)),
        pin_memory=resolved_device.type == "cuda",
    )
    granularities = [str(value) for value in config["model"]["granularities"]]
    if not granularities or len(set(granularities)) != len(granularities):
        raise FinalHoldoutError("Saved granularity order is invalid")
    component_results = evaluate_validation_per_granularity(
        model,
        dataloader,
        granularities,
        resolved_device,
        distributed=False,
        config=config,
    )
    ordered_losses: list[dict[str, Any]] = []
    for expected_granularity, component in zip(
        granularities,
        component_results,
        strict=True,
    ):
        loss = float(component["loss"])
        if component.get("granularity") != expected_granularity or not math.isfinite(
            loss
        ):
            raise FinalHoldoutError(
                "Final holdout evaluation returned invalid or reordered losses"
            )
        ordered_losses.append(
            {
                "granularity": expected_granularity,
                "loss": loss,
                "perplexity": float(component["perplexity"]),
                "evaluation_examples": int(component["evaluation_examples"]),
                "evaluation_target_tokens": int(component["evaluation_target_tokens"]),
            }
        )

    result_path = run_directory / "final_holdout_results.json"
    result = {
        "schema_version": FINAL_HOLDOUT_RESULT_VERSION,
        "run_id": provenance["run_id"],
        "checkpoint_path": provenance["checkpoint_path"],
        "checkpoint_selection_provenance": provenance[
            "checkpoint_selection_provenance"
        ],
        "final_holdout_manifest_hash": provenance["final_holdout_manifest_hash"],
        "run_summary_hash": provenance["run_summary_hash"],
        "controller_summary_hash": provenance["controller_summary_hash"],
        "controller_metrics_hash": provenance["controller_metrics_hash"],
        "ordered_granularities": granularities,
        "ordered_per_granularity_losses": ordered_losses,
        "uniform_average_loss": math.fsum(row["loss"] for row in ordered_losses)
        / len(ordered_losses),
        "aggregation_method": FINAL_HOLDOUT_AGGREGATION,
        "evaluation_example_count": int(manifest["example_count"]),
        "evaluation_invoked_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_path": str(result_path),
    }
    result["result_hash"] = stable_hash(result)
    written_path = write_json_artifact(
        result_path,
        result,
    )
    if written_path is None:
        raise FinalHoldoutError("Final holdout result was not written")
    return result


def _validate_training_artifact_hashes(
    run_directory: Path,
    run_summary: Mapping[str, Any],
    controller_summary: Mapping[str, Any],
) -> dict[str, str]:
    """Verify the standalone controller audit chain recorded by new runs."""

    controller_summary_hash = stable_hash(controller_summary)
    expected_summary_hash = run_summary.get("controller_summary_hash")
    if expected_summary_hash not in (None, ""):
        if expected_summary_hash != controller_summary_hash:
            raise FinalHoldoutError("Controller summary artifact hash mismatch")

        embedded_summary = run_summary.get("controller_summary")
        if not isinstance(embedded_summary, Mapping) or (
            stable_hash(embedded_summary) != controller_summary_hash
        ):
            raise FinalHoldoutError(
                "Run summary controller summary does not match the standalone artifact"
            )

    configured_metrics_hash = run_summary.get("controller_metrics_hash")
    summary_metrics_hash = controller_summary.get("controller_metrics_hash")
    if (
        configured_metrics_hash not in (None, "")
        and configured_metrics_hash != summary_metrics_hash
    ):
        raise FinalHoldoutError("Controller journal hash mismatch between summaries")

    metrics_path_value = controller_summary.get(
        "controller_metrics_path",
        run_summary.get("controller_metrics_path", "controller_metrics.jsonl"),
    )
    metrics_path = _resolve_artifact_candidate(run_directory, metrics_path_value)
    actual_metrics_hash = (
        hashlib.sha256(metrics_path.read_bytes()).hexdigest()
        if metrics_path.is_file()
        else None
    )
    if expected_summary_hash not in (None, ""):
        if actual_metrics_hash is None:
            raise FinalHoldoutError("Controller journal artifact is missing")
        if summary_metrics_hash != actual_metrics_hash:
            raise FinalHoldoutError("Controller journal artifact hash mismatch")

    return {
        "run_summary_hash": stable_hash(run_summary),
        "controller_summary_hash": controller_summary_hash,
        "controller_metrics_hash": str(
            actual_metrics_hash or summary_metrics_hash or ""
        ),
    }


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("role") != "final_holdout":
        raise FinalHoldoutError("Final holdout manifest role mismatch")
    identities = manifest.get("ordered_example_identities")
    if (
        not isinstance(identities, list)
        or len(identities) != FINAL_HOLDOUT_EXAMPLE_COUNT
        or int(manifest.get("example_count", -1)) != FINAL_HOLDOUT_EXAMPLE_COUNT
    ):
        raise FinalHoldoutError("Final holdout manifest example count mismatch")
    expected_identity_hash = stable_hash(identities)
    if manifest.get("example_identity_hash") != expected_identity_hash:
        raise FinalHoldoutError("Final holdout manifest identity hash mismatch")
    payload = copy.deepcopy(dict(manifest))
    saved_hash = payload.pop("manifest_hash", None)
    if saved_hash != stable_hash(payload):
        raise FinalHoldoutError("Final holdout manifest hash mismatch")


def _select_manifest_dataset(tokenized_dataset, manifest: Mapping[str, Any]):
    identities = manifest["ordered_example_identities"]
    column_names = set(getattr(tokenized_dataset, "column_names", []))
    if "source_row_identity" in column_names:
        source_rows = list(tokenized_dataset["source_row_identity"])
        index_by_identity = {
            str(source_identity): index
            for index, source_identity in enumerate(source_rows)
        }
    else:
        index_by_identity = {
            str(index): index for index in range(len(tokenized_dataset))
        }
    selected_indices = []
    for identity in identities:
        if not isinstance(identity, Mapping):
            raise FinalHoldoutError("Final holdout manifest identity is invalid")
        source_identity = str(identity.get("source_row_identity"))
        if source_identity not in index_by_identity:
            raise FinalHoldoutError(
                "Final holdout manifest example is missing from the reconstructed dataset"
            )
        selected_indices.append(index_by_identity[source_identity])
    if len(set(selected_indices)) != len(selected_indices):
        raise FinalHoldoutError("Final holdout manifest contains duplicate examples")
    return tokenized_dataset.select(selected_indices)


def _role_hashes(artifact: Mapping[str, Any]) -> dict[str, Any]:
    model = artifact.get("model", {})
    controller = (
        model.get("adaptive_controller", {}) if isinstance(model, Mapping) else {}
    )
    if not isinstance(controller, Mapping):
        controller = {}
    return {
        "data_roles": _first_value(
            artifact,
            controller,
            names=("data_roles_manifest_hash",),
        ),
        "optimizer_training": _first_value(
            artifact,
            controller,
            names=("optimizer_training_manifest_hash",),
        ),
        "controller": _first_value(
            artifact,
            controller,
            names=("controller_manifest_hash",),
        ),
        "ordinary_validation": _first_value(
            artifact,
            controller,
            names=("ordinary_validation_manifest_hash", "validation_manifest_hash"),
        ),
        "final_holdout": _first_value(
            artifact,
            controller,
            names=("final_holdout_manifest_hash",),
        ),
    }


def _checkpoint_role_hashes(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    controller_state = checkpoint.get("probabilistic_controller_state", {})
    manifests = (
        controller_state.get("manifest_hashes", {})
        if isinstance(controller_state, Mapping)
        else {}
    )
    if not isinstance(manifests, Mapping):
        manifests = {}
    return {
        "data_roles": _first_value(
            checkpoint,
            manifests,
            names=("data_roles_manifest_hash",),
        ),
        "optimizer_training": _first_value(
            checkpoint,
            manifests,
            names=("optimizer_training_manifest_hash",),
        ),
        "controller": _first_value(
            checkpoint,
            manifests,
            names=("controller_manifest_hash",),
        ),
        "ordinary_validation": _first_value(
            checkpoint,
            manifests,
            names=("ordinary_validation_manifest_hash", "validation_manifest_hash"),
        ),
        "final_holdout": _first_value(
            checkpoint,
            manifests,
            names=("final_holdout_manifest_hash",),
        ),
    }


def _first_value(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    names: tuple[str, ...],
) -> Any:
    for mapping in (first, second):
        for name in names:
            value = mapping.get(name)
            if value not in (None, ""):
                return value
    return None


def _config_run_id(config: Mapping[str, Any]) -> str:
    run = config.get("run", {})
    run_id = run.get("run_id") if isinstance(run, Mapping) else None
    if not isinstance(run_id, str) or not run_id:
        raise FinalHoldoutError("Saved config is missing the run ID")
    return run_id


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FinalHoldoutError(f"Required final holdout artifact is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FinalHoldoutError(
            f"Could not read final holdout artifact: {path}"
        ) from error
    if not isinstance(value, dict):
        raise FinalHoldoutError(f"Final holdout artifact must be a mapping: {path}")
    return value


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise FinalHoldoutError(
            f"Could not load final holdout checkpoint: {path}"
        ) from error
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise FinalHoldoutError("Final holdout checkpoint is missing model state")
    return checkpoint


def _require_run_directory(run_dir: str | Path) -> Path:
    path = Path(run_dir).expanduser().resolve()
    if not path.is_dir():
        raise FinalHoldoutError(f"Final holdout run directory does not exist: {path}")
    return path


def _resolve_existing_checkpoint(run_dir: Path, checkpoint_path: str | Path) -> Path:
    path = _resolve_checkpoint_candidate(run_dir, checkpoint_path)
    if not path.is_file():
        raise FinalHoldoutError(f"Final holdout checkpoint does not exist: {path}")
    return path


def _resolve_checkpoint_candidate(run_dir: Path, checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        if path.is_absolute():
            path = run_dir / "checkpoints" / path.name
        else:
            path = run_dir / path
    return path.resolve()


def _resolve_artifact_candidate(run_dir: Path, artifact_path: Any) -> Path:
    if not isinstance(artifact_path, (str, Path)) or not str(artifact_path):
        return (run_dir / "controller_metrics.jsonl").resolve()
    path = Path(artifact_path).expanduser()
    if not path.exists():
        path = run_dir / path.name if path.is_absolute() else run_dir / path
    return path.resolve()


def _paths_equal(first: Path, second: Path) -> bool:
    return first.resolve() == second.resolve()


__all__ = [
    "FinalHoldoutError",
    "evaluate_final_holdout",
    "resolve_existing_final_holdout_result",
    "resolve_final_holdout_checkpoint",
    "validate_final_holdout_provenance",
]
