"""Fixed TinyStories seed-42 campaign definitions and plain artifact records.

Preflight resolves the fixed matrix, audits pinned inputs, inspects fresh CPU
models and hashes complete deterministic traces. It publishes staged artifacts
without training or model evaluation. Record types describe JSON-compatible
dictionaries; scientific serialization lives in src.utils.reproducibility.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

from src.utils.metrics import write_json_artifact
from src.utils.reproducibility import SCIENTIFIC_CONTRACT_SCHEMA_VERSION, stable_hash


CAMPAIGN_SCHEMA_VERSION = 1
CLIPPING_SCHEMA_VERSION = 1
OPTIMIZER_OWNERSHIP_CHECKPOINT_SCHEMA_VERSION = 1
RESOURCE_ATTEMPTS_SCHEMA_VERSION = 1
TERMINAL_VALIDATION_SCHEMA_VERSION = 1
FROZEN_MANIFEST_SCHEMA_VERSION = 1
COMPARISON_REPORT_SCHEMA_VERSION = 1

SEED = 42
UPDATES_PER_EPOCH = 87_132
TOKENS_PER_UPDATE = 8_192
TOKENS_PER_EPOCH = 713_785_344
ENDPOINT_COUNT = 24
PARAMETER_COUNT_FIELD = "non_embedding_parameters"
PARAMETER_COUNT_CONVENTION = "active parameters excluding input embeddings and LM head"
EVALUATION_ROLE = "ordinary_validation"
VALIDATION_AGGREGATION = "target_token_weighted_causal_shift_float64"
QUARTER_IDS = ("A", "B", "C", "D")
OWNER_IDS = ("O-A", "O-B", "O-C", "O-D", "O-common")

Representation = Literal["dense", "slicing", "concat"]
StateScope = Literal["shared", "per_granularity", "per_ffn_block"]
ClippingMode = Literal["global", "per_owner"]
RunStatus = Literal["defined", "preflight_passed", "running", "completed", "frozen"]


class Width(TypedDict):
    label: str
    source_fraction: float
    active_ffn_dimension: int
    non_embedding_parameters: int
    active_quarters: tuple[str, ...]


# Tuple order is scientific metadata: narrowest to widest, quarter prefix order.
# Dense source widths retain these labels but use local active fraction 1.0.
WIDTHS: tuple[Width, ...] = (
    {
        "label": "g250",
        "source_fraction": 0.25,
        "active_ffn_dimension": 64,
        "non_embedding_parameters": 115_264,
        "active_quarters": ("A",),
    },
    {
        "label": "g500",
        "source_fraction": 0.50,
        "active_ffn_dimension": 128,
        "non_embedding_parameters": 164_416,
        "active_quarters": ("A", "B"),
    },
    {
        "label": "g750",
        "source_fraction": 0.75,
        "active_ffn_dimension": 192,
        "non_embedding_parameters": 213_568,
        "active_quarters": ("A", "B", "C"),
    },
    {
        "label": "g1000",
        "source_fraction": 1.00,
        "active_ffn_dimension": 256,
        "non_embedding_parameters": 262_720,
        "active_quarters": QUARTER_IDS,
    },
)
WIDTH_LABELS = tuple(width["label"] for width in WIDTHS)


class ArmDefinition(TypedDict):
    arm_id: str
    model_family: Literal["standalone", "nested"]
    representation: Representation
    state_scope: StateScope
    clipping_mode: ClippingMode
    physical_ffn_dimension: int
    source_width: str | None
    endpoint_widths: tuple[str, ...]
    assigned_epochs: int
    assigned_updates: int
    assigned_tokens: int


STANDALONE_ARMS: tuple[ArmDefinition, ...] = tuple(
    {
        "arm_id": f"ST-{width['label']}",
        "model_family": "standalone",
        "representation": "dense",
        "state_scope": "shared",
        "clipping_mode": "global",
        "physical_ffn_dimension": width["active_ffn_dimension"],
        "source_width": width["label"],
        "endpoint_widths": (width["label"],),
        "assigned_epochs": 1,
        "assigned_updates": UPDATES_PER_EPOCH,
        "assigned_tokens": TOKENS_PER_EPOCH,
    }
    for width in WIDTHS
)
ELASTIC_ARMS: tuple[ArmDefinition, ...] = tuple(
    {
        "arm_id": arm_id,
        "model_family": "nested",
        "representation": representation,
        "state_scope": state_scope,
        "clipping_mode": clipping_mode,
        "physical_ffn_dimension": 256,
        "source_width": None,
        "endpoint_widths": WIDTH_LABELS,
        "assigned_epochs": 4,
        "assigned_updates": 4 * UPDATES_PER_EPOCH,
        "assigned_tokens": 4 * TOKENS_PER_EPOCH,
    }
    for arm_id, representation, state_scope, clipping_mode in (
        ("S1", "slicing", "shared", "global"),
        ("S2", "slicing", "per_granularity", "global"),
        ("C1", "concat", "shared", "global"),
        ("C2", "concat", "per_granularity", "global"),
        ("C3", "concat", "per_ffn_block", "per_owner"),
    )
)
ARMS = STANDALONE_ARMS + ELASTIC_ARMS
ARM_IDS = tuple(arm["arm_id"] for arm in ARMS)


class CampaignContract(TypedDict):
    """Resolved preflight record; expected_data includes all pinned role hashes."""

    schema_version: int
    campaign_id: str
    seed: int
    common: dict[str, Any]
    arms: list[ArmDefinition]
    expected_data: dict[str, Any]
    evaluation_protocol: dict[str, Any]
    count_convention: str
    run_contract_hashes: dict[str, str]


class CampaignRun(TypedDict):
    """One fresh identity; failed attempts continue this same durable run."""

    campaign_id: str
    run_id: str
    arm_id: str
    status: RunStatus
    resolved_config: dict[str, Any]
    representation: Representation
    state_scope: StateScope
    clipping: dict[str, Any]
    physical_ffn_dimension: int
    source_width: str | None
    initialization: dict[str, Any]
    assigned_updates: int
    assigned_tokens: int
    assigned_epochs: int
    output_path: str
    optimizer_ownership_contract: dict[str, Any]
    contract_hash: str


class Endpoint(TypedDict):
    """Key: (campaign_id, arm_id, run_id, width); terminal ordinary validation.

    Evaluation records carry role, manifest/protocol, aggregation, examples,
    targets and sidecar hash. Budgets carry updates/tokens/epochs. Resources and
    exposure describe measured values; missing device observations remain null.
    """

    campaign_id: str
    arm_id: str
    run_id: str
    width: str
    seed: int
    representation: Representation
    state_scope: StateScope
    clipping: dict[str, Any]
    source_fraction: float
    active_ffn_dimension: int
    non_embedding_parameters: int
    count_convention: str
    loss: float
    perplexity: float
    checkpoint_path: str
    checkpoint_sha256: str
    contract_hash: str
    evaluation: dict[str, Any]
    actual_budget: dict[str, int]
    assigned_budget: dict[str, int]
    initialization: dict[str, Any]
    data: dict[str, Any]
    exposure: dict[str, Any]
    resources: dict[str, Any]
    # Omitted until a frozen report assigns artifact locations.
    artifact_paths: NotRequired[dict[str, str]]


# Frozen scientific recipe: validate inputs against this protocol, not against
# another arm or a hash supplied by the caller. Paths and campaign ID are bound later.
PINNED_COMMON = {
    "run": {
        "phase_id": "tinystories_optimizer_ownership",
        "model_shape_label": "tinystories-instruct-d64-l4",
        "seed": 42,
        "reproducibility": {
            "mode": "strict",
            "seed_stream_version": 1,
            "data_split_version": 1,
        },
        "continuation": {
            "enabled": True,
            "latest_checkpoint_save_interval_steps": 0,
            "latest_checkpoint_save_on_validation": True,
            "latest_checkpoint_save_on_completion": True,
        },
    },
    "model": {
        "base_model_name": "tinystories-instruct-optimizer-ownership-llama",
        "tokenizer_dir": "__PREPARED_TOKENIZER_DIR__",
        "correction_mode": "none",
        "granularity_mode": "explicit",
        "d_model": 64,
        "num_layers": 4,
        "num_attention_heads": 4,
        "context_length": 128,
        "vocab_size": 2048,
        "initializer_range": 0.02,
        "granularities": ["g250", "g500", "g750", "g1000"],
        "granularity_prefixes": {"g250": 0.25, "g500": 0.5, "g750": 0.75, "g1000": 1.0},
    },
    "training": {
        "max_steps_cap": None,
        "batch_size_per_process": 64,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.008,
        "learning_rate_scale_rule": "none",
        "warmup_steps": 64,
        "pre_nested_warmup": {"enabled": False, "duration": 0, "unit": "steps"},
        "gradient_clip_norm": 1.0,
        "mixed_precision": "bf16",
        "activation_checkpointing": False,
        "dataloader_num_workers": 0,
        "optimizer": {
            "name": "adamw",
            "scheduler_clock": "global_step",
            "kwargs": {"betas": [0.9, 0.95], "eps": 1e-08, "weight_decay": 0.1},
        },
        "scheduler": {"name": "cosine"},
        "distributed": {"strategy": "none", "expected_world_size": 1},
        "gradient_clipping": {"mode": "global", "norm_type": 2},
    },
    "dataset": {
        "mode": "packed_mmap",
        "prepared_corpus_dir": "__PREPARED_CORPUS_DIR__",
        "optimizer_iteration": {
            "mode": "repeat_epochs",
            "epoch_order": "deterministic_per_epoch",
        },
        "fixed_four_role_partition": True,
        "data_seed": 42,
        "dataset_name": "roneneldan/TinyStoriesInstruct",
        "dataset_config_name": "default",
        "dataset_split": "train+validation",
        "dataset_phase": "tinystories_instruct_controlled",
        "sample_limit": None,
        "tokenization_keep_in_memory": False,
        "preprocessing_notes": "complete_record_lf_join_preserve_fields_and_internal_newlines_v1",
    },
    "outputs": {
        "save_config": True,
        "save_metrics_csv": True,
        "save_run_summary_json": True,
        "save_checkpoints": True,
        "make_plots": False,
        "metrics_flush_interval_steps": 32,
        "best_eval_retention_count": 1,
        "artifact_io": {"checkpoint_staging": "auto"},
    },
    "monitoring": {
        "enabled": False,
        "backend": "wandb",
        "project": "tinystories_optimizer_ownership",
        "job_type": "train",
        "tags": ["controlled-experiment", "optimizer-ownership"],
        "notes": "Seed-42 nine-arm ownership and clipping comparison",
        "log_loss_by_granularity": True,
        "log_validation_loss": True,
        "log_stage_events": True,
    },
    "evaluation": {
        "validation": {
            "enabled": True,
            "interval_steps": 64,
            "interval_tokens": 0,
            "run_at_completion": True,
            "holdout": {"source": "configured_dataset_split", "examples": 128},
            "trailing_summary_evaluations": 5,
        },
        "adaptive_controller": {
            "enabled": True,
            "source": "configured_dataset_split",
            "examples": 128,
            "objective_weights": "uniform",
            "fixed_manifest": True,
        },
        "final_holdout": {
            "enabled": True,
            "source": "configured_dataset_split",
            "examples": 512,
            "fixed_manifest": True,
            "evaluate_during_training": False,
        },
        "test": {"enabled": False},
        "downstream_suite": [],
        "consistency": False,
        "speculative": False,
    },
}
PINNED_DATA = {
    "corpus_hash": "e2eff35bc7078f4f4c4d618638988b3b82e96e7c20e40175ec8d272c5370efc9",
    "optimizer_training_manifest_hash": "c06270adebde4c5456517a5bcf266b2e4520587fb928541333009735ee2e60dd",
    "training_order_sha256": "d94ed9514a314a404fc420385a4b0f3a317774707ca3a212100aa7d4f35badbc",
    "tokenizer_manifest_hash": "93e6bc7a94df82ca043519a71d5178bb645846ecd553f2302244b72c954e3819",
    "tokenizer_model_sha256": "6ae42a09bb5dc007267f11bfd7b7b4fefd006c960c5954a4a45ce7b89ace7713",
    "ordinary_validation_manifest_hash": "0c1beea552f54941e397d2442de736b1586e0292f6b1271b62d27ad782627856",
    "controller_manifest_hash": "69b039d6f6cec565e9efb576080b3e91d652c63290ebdea6f376fd133b6bd987",
    "final_holdout_manifest_hash": "e20160ab1d7781b4fe7f24a78ed547fce0ef3b7b1f0cd66eaf2f84dd89113076",
}


def _merge(common, override):
    import copy

    result = copy.deepcopy(common)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _require_equal(actual, expected, field):
    """Name the first changed control, including missing or extra nested fields."""
    from src.utils.config import ConfigError

    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(actual) | set(expected)):
            if key not in actual or key not in expected:
                raise ConfigError(f"{field}.{key}: missing or unexpected control")
            _require_equal(actual[key], expected[key], f"{field}.{key}")
    elif actual != expected:
        raise ConfigError(f"{field}: expected {expected!r}, found {actual!r}")


def _arm_overrides(arm):
    run = {"model_family": arm["model_family"]}
    model = {
        "variant": "slicing"
        if arm["representation"] == "dense"
        else arm["representation"]
    }
    training = {
        "token_budget": arm["assigned_tokens"],
        "optimizer": {"state_scope": arm["state_scope"]},
    }
    if arm["source_width"]:
        run["granularity"] = arm["source_width"]
    else:
        run["sampling_mode"] = "nested-random"
        model.update(
            granularity_sampling_mode="global",
            global_sampling_schedule="random_with_replacement",
            global_sampling_interval_steps=1,
        )
    if arm["clipping_mode"] == "per_owner":
        training["gradient_clipping"] = {
            "mode": "per_owner",
            "norm_type": 2,
            "owner_max_norms": dict.fromkeys(OWNER_IDS, 1.0),
        }
    return {"run": run, "model": model, "training": training}


def _reservation_path(run_output_root):
    from pathlib import Path

    root = Path(run_output_root).expanduser().resolve()
    return root.parent / f".{root.name}.optimizer-ownership-reservation.json"


def _check_unoccupied(run_output_root):
    from pathlib import Path
    from src.utils.config import ConfigError

    root = Path(run_output_root).expanduser().resolve()
    for path in [_reservation_path(root), *(root / arm for arm in ARM_IDS)]:
        if path.exists() or path.is_symlink():
            raise ConfigError(f"occupied campaign run identity: {path}")


def _provenance():
    import importlib.metadata
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    # Working changes matter when preflight precedes an implementation commit.
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--", "src", "train.py", "scripts"],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout
    import hashlib

    source_paths = sorted((root / "src").rglob("*.py")) + [
        root / "train.py",
        root / "scripts/analyze_tinystories_optimizer_ownership.py",
    ]
    source_hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    return {
        "source_files_sha256": source_hashes,
        "code_revision": revision,
        "working_tree_dirty": bool(dirty),
        "tracked_code_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "dependency_versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "datasets",
                "PyYAML",
                "numpy",
                "pandas",
                "matplotlib",
            )
        },
    }


def expand_campaign(recipe, *, prepared_corpus_dir, tokenizer_dir, run_output_root):
    """Resolve exactly nine fresh normal trainer configs, without run-directory IO."""
    import copy
    import re
    import tempfile
    from pathlib import Path
    import yaml
    from src.utils.config import ConfigError, resolve_run_config
    from src.utils.reproducibility import build_optimizer_ownership_signature, seed_for

    if not isinstance(recipe, dict):
        raise ConfigError("campaign must be a mapping")
    _require_equal(
        set(recipe),
        {"schema_version", "campaign_id", "common", "arms", "expected_data"},
        "campaign fields",
    )
    _require_equal(recipe["schema_version"], CAMPAIGN_SCHEMA_VERSION, "schema_version")
    campaign_id = recipe["campaign_id"]
    if not isinstance(campaign_id, str) or not re.fullmatch(
        r"tinystories-optimizer-ownership-[A-Za-z0-9_-]+", campaign_id
    ):
        raise ConfigError(
            "campaign_id must be a fresh tinystories-optimizer-ownership-* identity"
        )
    _require_equal(recipe["expected_data"], PINNED_DATA, "expected_data")
    if not isinstance(recipe["arms"], dict) or not isinstance(recipe["common"], dict):
        raise ConfigError("campaign arms and common must be mappings")
    _require_equal(set(recipe["arms"]), set(ARM_IDS), "arms")
    _check_unoccupied(run_output_root)
    provenance = _provenance()
    runs = []
    with tempfile.TemporaryDirectory(prefix="ownership-resolve-") as staging:
        for arm in ARMS:
            arm_id = arm["arm_id"]
            if not isinstance(recipe["arms"][arm_id], dict):
                raise ConfigError(f"arms.{arm_id} must be a mapping")
            raw = _merge(recipe["common"], recipe["arms"][arm_id])
            _require_equal(raw, _merge(PINNED_COMMON, _arm_overrides(arm)), arm_id)
            run_id = f"{campaign_id}-{arm_id}-s{SEED}"
            output = str(Path(run_output_root).expanduser().resolve() / arm_id)
            raw["run"].update(
                run_id=run_id, campaign_id=campaign_id, arm_id=arm_id, output_dir=output
            )
            raw["model"]["tokenizer_dir"] = str(
                Path(tokenizer_dir).expanduser().resolve()
            )
            raw["dataset"]["prepared_corpus_dir"] = str(
                Path(prepared_corpus_dir).expanduser().resolve()
            )
            config_path = Path(staging) / f"{arm_id}.yaml"
            config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
            resolved = resolve_run_config(config_path, create_output_dirs=False)
            validate_run_budget(resolved, arm)
            initialization = {
                "method": "fresh_normal_constructor",
                "seed": SEED,
                "seed_stream": "model_initialization",
                "derived_seed": seed_for(resolved, "model_initialization"),
                "initializer_range": 0.02,
                "cross_model_tensor_equality_required": False,
                "constructor": "LlamaForCausalLM"
                if arm["source_width"]
                else "ModifiedLlamaForCausalLM",
                **provenance,
            }
            clipping = copy.deepcopy(resolved["training"]["gradient_clipping"])
            model_controls = {
                k: v for k, v in resolved["model"].items() if k != "tokenizer_dir"
            }
            data_controls = {
                k: v
                for k, v in resolved["dataset"].items()
                if k != "prepared_corpus_dir"
            }
            contract = {
                "schema_version": SCIENTIFIC_CONTRACT_SCHEMA_VERSION,
                "campaign_id": campaign_id,
                "run_id": run_id,
                "arm_id": arm_id,
                "representation": arm["representation"],
                "state_scope": arm["state_scope"],
                "clipping": clipping,
                "initialization": initialization,
                "model": model_controls,
                "optimizer": copy.deepcopy(resolved["training"]),
                "sampling": {
                    "mode": resolved["run"]["sampling_mode"],
                    "probabilities": None if arm["source_width"] else [0.25] * 4,
                    "action_seed": None
                    if arm["source_width"]
                    else seed_for(resolved, "granularity_selection"),
                },
                "data": data_controls,
                "budget": {
                    k: arm[k]
                    for k in ("assigned_epochs", "assigned_updates", "assigned_tokens")
                },
                "evaluation": copy.deepcopy(resolved["evaluation"]),
                "count_convention": PARAMETER_COUNT_CONVENTION,
            }
            contract_hash, contract = build_optimizer_ownership_signature(contract)
            for config in (raw, resolved):
                config["optimizer_ownership_contract"] = copy.deepcopy(contract)
                config["optimizer_ownership_contract_hash"] = contract_hash
            runs.append(
                {
                    **copy.deepcopy(arm),
                    "campaign_id": campaign_id,
                    "run_id": run_id,
                    "status": "preflight_passed",
                    "resolved_config": resolved,
                    "executable_config": raw,
                    "clipping": clipping,
                    "initialization": initialization,
                    "output_path": output,
                    "optimizer_ownership_contract": contract,
                    "contract_hash": contract_hash,
                }
            )
    return runs


def validate_run_budget(config, arm):
    training, dataset = config["training"], config["dataset"]
    for key, expected in {
        "max_steps": arm["assigned_updates"],
        "derived_max_steps": arm["assigned_updates"],
        "token_budget": arm["assigned_tokens"],
        "expected_tokens_per_step": TOKENS_PER_UPDATE,
        "max_steps_cap": None,
        "resolved_warmup_steps": 64,
    }.items():
        _require_equal(training[key], expected, f"{arm['arm_id']}.training.{key}")
    expected = {
        "aligned_epoch_samples": 5576448,
        "aligned_epoch_tokens": TOKENS_PER_EPOCH,
        "excluded_tail_samples": 43,
        "excluded_tail_tokens": 5504,
        "complete_epochs": arm["assigned_epochs"],
        "partial_final_epoch_samples": 0,
        "partial_final_epoch_tokens": 0,
        "planned_samples": arm["assigned_tokens"] // 128,
        "mode": "repeat_epochs",
        "epoch_order": "deterministic_per_epoch",
    }
    for key, value in expected.items():
        _require_equal(
            dataset["optimizer_iteration"][key],
            value,
            f"{arm['arm_id']}.optimizer_iteration.{key}",
        )
    _require_equal(
        config["model"]["intermediate_size"],
        arm["physical_ffn_dimension"],
        f"{arm['arm_id']}.intermediate_size",
    )


def validate_corpus_audit(audit, expected_data):
    from src.utils.config import ConfigError

    _require_equal(expected_data, PINNED_DATA, "expected_data")
    expected = {
        "status": "passed",
        "schema_version": 3,
        "source_exhausted": True,
        "training_sequence_count": 5576491,
        "training_token_count": 713790848,
        "verified_training_order_count": 5576491,
        "verified_shard_count": 89,
        "tokenizer_vocab_size": 2048,
        "source_document_count": 2477172,
        "role_source_document_counts": {
            "optimizer_training": 2476404,
            "ordinary_validation": 128,
            "controller": 128,
            "final_holdout": 512,
        },
        "reserved_role_counts": {
            "ordinary_validation": 128,
            "controller": 128,
            "final_holdout": 512,
        },
    }
    for key, value in expected.items():
        _require_equal(audit.get(key), value, f"audit.{key}")
    for key, value in expected_data.items():
        role = key.removesuffix("_manifest_hash")
        actual = (
            audit.get("role_manifest_hashes", {}).get(role)
            if role
            in (
                "optimizer_training",
                "ordinary_validation",
                "controller",
                "final_holdout",
            )
            else audit.get(key)
        )
        _require_equal(actual, value, f"audit.{key}")
    intersections = audit.get("reserved_pairwise_intersection_counts", {})
    if len(intersections) != 3 or any(value != 0 for value in intersections.values()):
        raise ConfigError(
            "audit: reserved roles overlap or role separation evidence is missing"
        )
    return audit


def inspect_campaign_models(runs):
    """Normal seeded construction, static physical tensors and exact active counts."""
    import random
    import numpy as np
    import torch
    from src.training.modeling import build_model
    from src.training.optimizer_state import (
        build_concat_parameter_partition,
        build_parameter_descriptors,
    )
    from src.utils.model_size import model_parameter_counts
    from src.utils.reproducibility import seed_for
    from src.utils.config import ConfigError
    from src.models.ffn import CatLlamaMLP, ModifiedLlamaMLP

    checks, models = [], []
    # Retain models until coverage is checked: object IDs cannot be recycled and
    # accidentally treated as cross-run parameter sharing.
    all_parameter_ids = set()
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            for run in runs:
                config = run["resolved_config"]
                seed = seed_for(config, "model_initialization")
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.default_generator.manual_seed(seed)
                model = build_model(config).cpu()
                models.append(model)
                ids = {id(p) for p in model.parameters()}
                if ids & all_parameter_ids:
                    raise ConfigError(
                        f"{run['arm_id']}: models share parameter objects"
                    )
                all_parameter_ids.update(ids)
                counts = {}
                for width in WIDTHS:
                    if width["label"] not in run["endpoint_widths"]:
                        continue
                    counts[width["label"]] = model_parameter_counts(
                        model, granularity=width["label"]
                    )[PARAMETER_COUNT_FIELD]
                    _require_equal(
                        counts[width["label"]],
                        width[PARAMETER_COUNT_FIELD],
                        f"{run['arm_id']}.counts.{width['label']}",
                    )
                owners = []
                for layer in model.model.layers:
                    mlp = layer.mlp
                    if run["representation"] == "concat":
                        if not isinstance(mlp, CatLlamaMLP):
                            raise ConfigError(f"{run['arm_id']}: expected concat FFN")
                    else:
                        if (
                            run["representation"] == "dense"
                            and isinstance(mlp, ModifiedLlamaMLP)
                        ) or (
                            run["representation"] == "slicing"
                            and not isinstance(mlp, ModifiedLlamaMLP)
                        ):
                            raise ConfigError(
                                f"{run['arm_id']}: wrong dense/slicing FFN representation"
                            )
                        size = run["physical_ffn_dimension"]
                        for name, shape in [
                            ("gate_proj", (size, 64)),
                            ("up_proj", (size, 64)),
                            ("down_proj", (64, size)),
                        ]:
                            _require_equal(
                                tuple(getattr(mlp, name).weight.shape),
                                shape,
                                f"{run['arm_id']}.{name}.shape",
                            )
                if run["representation"] == "concat":
                    for owner in build_concat_parameter_partition(
                        model, ordered_widths=WIDTH_LABELS
                    ):
                        owners.append(
                            {
                                "owner_id": owner.owner_id,
                                "active_widths": list(owner.active_widths),
                                "parameter_elements": sum(
                                    p.numel() for p in owner.parameters
                                ),
                                "parameters": list(owner.descriptors),
                            }
                        )
                descriptors = build_parameter_descriptors(
                    model, ordered_widths=config["model"]["granularities"]
                )
                checks.append(
                    {
                        "arm_id": run["arm_id"],
                        "counts": counts,
                        "owners": owners,
                        "parameter_descriptors": list(descriptors),
                        "fresh_parameter_identity": True,
                    }
                )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    return checks


def expected_action_trace(config):
    """Same isolated randrange stream as select_random_granularity_index, H=1."""
    import hashlib
    import random
    from src.utils.reproducibility import seed_for

    generator = random.Random(seed_for(config, "granularity_selection"))
    digest = hashlib.sha256()
    counts = dict.fromkeys(WIDTH_LABELS, 0)
    for _ in range(config["training"]["max_steps"]):
        width = WIDTH_LABELS[generator.randrange(len(WIDTH_LABELS))]
        digest.update((width + "\n").encode("ascii"))
        counts[width] += 1
    return {
        "sha256": digest.hexdigest(),
        "encoding": "ASCII width label plus LF per update",
        "updates": sum(counts.values()),
        "counts": counts,
        "expected_counts": dict.fromkeys(
            WIDTH_LABELS, config["training"]["max_steps"] / 4
        ),
        "sampling": "uniform_random_with_replacement",
        "forced_balance": False,
    }


def expected_epoch_traces(sampler, *, epochs):
    import hashlib
    import numpy as np

    traces = []
    for epoch in range(epochs):
        # Use the runtime sampler's exact ordering implementation, vectorized;
        # fixed batch size and flattened order uniquely determine every batch.
        order = np.asarray(sampler._epoch_indices(epoch), dtype="<u8")
        traces.append(
            {
                "epoch_index": epoch,
                "sha256": hashlib.sha256(order.tobytes()).hexdigest(),
                "encoding": "ordered sequence IDs as little-endian uint64",
                "sequences": len(order),
                "updates": len(order) // sampler.global_batch_size,
                "batch_size": sampler.global_batch_size,
                "fixed_epoch_set_hash": sampler.fixed_epoch_set_hash,
            }
        )
    return traces


def build_expected_traces(runs, corpus_dir, corpus_manifest):
    from pathlib import Path
    from src.training.packed_corpus import RepeatingNoPaddingDistributedBatchSampler

    result = {}
    for run in runs:
        config = run["resolved_config"]
        iteration = config["dataset"]["optimizer_iteration"]
        sampler = RepeatingNoPaddingDistributedBatchSampler(
            dataset_size=5576491,
            batch_size_per_rank=config["training"]["batch_size_per_process"],
            rank=0,
            world_size=1,
            planned_sample_count=iteration["planned_samples"],
            epoch_sample_count=iteration["aligned_epoch_samples"],
            data_seed=config["dataset"]["data_seed"],
            permutation_path=Path(corpus_dir)
            / corpus_manifest["training_order"]["path"],
            permutation_hash_expected=iteration["permutation_hash"],
            permutation_version=iteration["permutation_version"],
            corpus_hash=config["dataset"]["corpus_hash"],
            optimizer_training_manifest_hash=iteration[
                "optimizer_training_manifest_hash"
            ],
        )
        result[run["arm_id"]] = {
            "epochs": expected_epoch_traces(sampler, epochs=run["assigned_epochs"]),
            "actions": None if run["source_width"] else expected_action_trace(config),
        }
    first = result[ARM_IDS[0]]["epochs"][0]
    elastic = result["S1"]
    for run in runs:
        trace = result[run["arm_id"]]
        _require_equal(trace["epochs"][0], first, f"{run['arm_id']}.first_epoch")
        if not run["source_width"]:
            _require_equal(trace, elastic, f"{run['arm_id']}.elastic_trace")
    return result


def preflight_campaign(
    *, campaign_path, prepared_corpus_dir, tokenizer_dir, output_dir, run_output_root
):
    """Audit and stage everything before publishing a manifest or reserving runs."""
    import json
    import os
    import shutil
    import tempfile
    from pathlib import Path
    import yaml
    from src.training.packed_corpus import audit_packed_corpus, load_corpus_manifest
    from src.utils.config import ConfigError

    output = Path(output_dir).expanduser().resolve()
    root = Path(run_output_root).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ConfigError(f"occupied preflight output: {output}")
    if output == root or output in root.parents or root in output.parents:
        raise ConfigError(
            "Preflight output and run-output-root must be separate directory trees"
        )
    recipe = yaml.safe_load(Path(campaign_path).read_text())
    runs = expand_campaign(
        recipe,
        prepared_corpus_dir=prepared_corpus_dir,
        tokenizer_dir=tokenizer_dir,
        run_output_root=root,
    )
    audit = audit_packed_corpus(
        prepared_corpus_dir,
        prepared_tokenizer_dir=tokenizer_dir,
        required_vocab_size=2048,
        minimum_training_tokens=TOKENS_PER_EPOCH,
    )
    validate_corpus_audit(audit, recipe["expected_data"])
    models = inspect_campaign_models(runs)
    from src.utils.reproducibility import build_optimizer_ownership_signature

    for run, check in zip(runs, models):
        topology_hash = stable_hash(
            {"parameters": check["parameter_descriptors"], "owners": check["owners"]}
        )
        contract = run["optimizer_ownership_contract"]
        contract["parameter_topology_hash"] = topology_hash
        contract["clipping"] = {
            **run["clipping"],
            "schema_version": CLIPPING_SCHEMA_VERSION,
            "topology_hash": topology_hash,
        }
        run["clipping"] = contract["clipping"]
        run["contract_hash"], _ = build_optimizer_ownership_signature(contract)
        for config in (run["executable_config"], run["resolved_config"]):
            config["optimizer_ownership_contract"] = contract
            config["optimizer_ownership_contract_hash"] = run["contract_hash"]
    corpus_manifest = load_corpus_manifest(prepared_corpus_dir, verify_shards=False)
    traces = build_expected_traces(runs, prepared_corpus_dir, corpus_manifest)
    manifest = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": recipe["campaign_id"],
        "seed": SEED,
        "common": recipe["common"],
        "allowed_difference_matrix": list(ARMS),
        "expected_data": recipe["expected_data"],
        "count_convention": PARAMETER_COUNT_CONVENTION,
        "evaluation_role": EVALUATION_ROLE,
        "runs": runs,
        "run_contract_hashes": {r["arm_id"]: r["contract_hash"] for r in runs},
        "expected_traces": traces,
        "model_checks": models,
        "run_output_root": str(root),
        "runtime_ownership_verified": False,
        "training_started": False,
        "holdout_evaluated": False,
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    report = {
        "schema_version": 1,
        "status": "passed",
        "kind": "campaign_audit_model_trace",
        "campaign_id": recipe["campaign_id"],
        "audit": audit,
        "run_count": len(runs),
        "manifest_hash": manifest["manifest_hash"],
        "manifest_path": str(output / "campaign_manifest.json"),
        "model_counts": {m["arm_id"]: m["counts"] for m in models},
        "expected_traces": traces,
        "training_started": False,
        "holdout_evaluated": False,
        "runtime_ownership_verified": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    reservation = _reservation_path(root)
    reserved = False
    try:
        (stage / "configs").mkdir()
        for run in runs:
            (stage / "configs" / f"{run['arm_id']}.yaml").write_text(
                yaml.safe_dump(run["executable_config"], sort_keys=False)
            )
        write_json_artifact(stage / "preflight.json", report)
        write_json_artifact(stage / "campaign_manifest.json", manifest)
        _check_unoccupied(root)
        reservation.parent.mkdir(parents=True, exist_ok=True)
        with reservation.open("x") as handle:
            reserved = True
            json.dump(
                {
                    "campaign_id": recipe["campaign_id"],
                    "manifest_path": str(output / "campaign_manifest.json"),
                    "manifest_hash": manifest["manifest_hash"],
                    "run_ids": [r["run_id"] for r in runs],
                },
                handle,
                indent=2,
            )
            handle.flush()
            os.fsync(handle.fileno())
        # The entire staged directory becomes visible at once; no partial success.
        if output.exists():
            raise ConfigError(f"occupied preflight output: {output}")
        stage.rename(output)
    except BaseException:
        if reserved:
            reservation.unlink(missing_ok=True)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return report


def validate_materialized_config(config):
    """Reject stale scientific metadata when an ordinary campaign YAML is read."""
    from src.utils.config import ConfigError
    from src.utils.reproducibility import build_optimizer_ownership_signature, seed_for

    contract = config.get("optimizer_ownership_contract")
    if not isinstance(contract, dict):
        raise ConfigError("optimizer_ownership_contract must be a mapping")
    try:
        actual_hash, _ = build_optimizer_ownership_signature(contract)
    except ValueError as error:
        raise ConfigError(str(error)) from error
    _require_equal(
        config.get("optimizer_ownership_contract_hash"),
        actual_hash,
        "optimizer_ownership_contract_hash",
    )
    arm = next((a for a in ARMS if a["arm_id"] == config["run"].get("arm_id")), None)
    if arm is None:
        raise ConfigError("optimizer_ownership_contract: unknown arm_id")
    for key in ("campaign_id", "run_id", "arm_id"):
        _require_equal(config["run"].get(key), contract[key], f"run.{key}")
    _require_equal(config["run"]["seed"], SEED, "run.seed")
    _require_equal(contract["initialization"]["seed"], SEED, "initialization.seed")
    _require_equal(
        contract["initialization"]["derived_seed"],
        seed_for(config, "model_initialization"),
        "initialization.derived_seed",
    )
    _require_equal(contract["representation"], arm["representation"], "representation")
    _require_equal(contract["state_scope"], arm["state_scope"], "state_scope")
    validate_run_budget(config, arm)
    controls = {
        "model": {k: v for k, v in config["model"].items() if k != "tokenizer_dir"},
        "data": {
            k: v for k, v in config["dataset"].items() if k != "prepared_corpus_dir"
        },
        "optimizer": config["training"],
        "clipping": config["training"]["gradient_clipping"],
        "evaluation": config["evaluation"],
        "sampling": {
            "mode": config["run"]["sampling_mode"],
            "probabilities": None if arm["source_width"] else [0.25] * 4,
            "action_seed": None
            if arm["source_width"]
            else seed_for(config, "granularity_selection"),
        },
        "budget": {
            k: arm[k]
            for k in ("assigned_epochs", "assigned_updates", "assigned_tokens")
        },
        "count_convention": PARAMETER_COUNT_CONVENTION,
    }
    for field, values in controls.items():
        expected = contract[field]
        if field == "clipping":
            expected = {
                k: v
                for k, v in expected.items()
                if k not in {"topology_hash", "schema_version"}
            }
        _require_equal(values, expected, f"optimizer_ownership_contract.{field}")
