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
# The pinned ordinary-validation manifest packs 128 source documents into 285
# context-128 sequences. Evaluation counts sequences, not source documents.
VALIDATION_SEQUENCES = 285
VALIDATION_TARGET_TOKENS = VALIDATION_SEQUENCES * 127
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

CORRECTION_CAMPAIGN_SCHEMA_VERSION = 2
CORRECTION_ARMS = tuple(
    {**arm, "arm_id": arm["arm_id"] + "-" + mode.upper(),
     "reference_arm_id": arm["arm_id"], "correction_mode": mode}
    for arm in ELASTIC_ARMS if arm["representation"] == "concat"
    for mode in ("gmc", "lmc")
)
CORRECTION_ARM_IDS = tuple(arm["arm_id"] for arm in CORRECTION_ARMS)


INVERSE_MEMBERSHIP_CAMPAIGN_SCHEMA_VERSION = 3
IM_PROBABILITIES = (.12, .16, .24, .48)
INVERSE_MEMBERSHIP_ARMS = tuple(
    {**arm, "arm_id": arm["arm_id"] + "-IM", "reference_arm_id": arm["arm_id"],
     "sampling_policy": "fixed_inverse_membership"}
    for arm in ELASTIC_ARMS
)
INVERSE_MEMBERSHIP_ARM_IDS = tuple(a["arm_id"] for a in INVERSE_MEMBERSHIP_ARMS)


def inverse_membership_sampling_contract(config):
    from src.utils.reproducibility import seed_for
    return {
        "schema_version": 1, "policy": "fixed_inverse_membership",
        "mode": "fixed_global", "ordered_widths": list(WIDTH_LABELS),
        "membership_counts": [4, 3, 2, 1], "probabilities": list(IM_PROBABILITIES),
        "scope": "global", "replacement": True, "interval_steps": 1,
        "inverse_probability_loss_weighting": False,
        "action_seed": seed_for(config, "granularity_selection"),
    }


def _sampling_contract(config, arm):
    from src.utils.reproducibility import seed_for
    if arm.get("sampling_policy") == "fixed_inverse_membership":
        return inverse_membership_sampling_contract(config)
    return {
        "mode": config["run"]["sampling_mode"],
        "probabilities": None if arm["source_width"] else [0.25] * 4,
        "action_seed": None if arm["source_width"] else seed_for(config, "granularity_selection"),
    }


def campaign_arms(schema_version):
    from src.utils.config import ConfigError
    if type(schema_version) is int and schema_version == 1:
        return ARMS
    if type(schema_version) is int and schema_version == 2:
        return CORRECTION_ARMS
    if type(schema_version) is int and schema_version == 3:
        return INVERSE_MEMBERSHIP_ARMS
    raise ConfigError(f"Unsupported campaign schema_version: {schema_version}")


def membership_correction_contract(mode):
    """Explicit new-campaign semantics; never added to original contracts."""
    from src.utils.config import ConfigError
    if mode not in ("gmc", "lmc"):
        raise ConfigError("Corrected campaign requires gmc or lmc")
    return {
        "schema_version": 1, "mode": mode,
        "trained_widths": list(WIDTH_LABELS), "membership_counts": [4, 3, 2, 1],
        "factors": [1., 4/3, 2., 4.], "gradient_correction": True,
        "parameter_change_correction": mode == "lmc",
        "scales_weight_decay": mode == "lmc", "common_factor": 1.,
        "parameter_scope": "FFN block weights and block-local gate/up biases",
        "moment_scaling": False,
        "order": ["gradient_correction", "clipping", "adamw"]
                 + (["parameter_change_correction"] if mode == "lmc" else [])
                 + ["global_scheduler", "accounting"],
    }


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
    if "correction_mode" in arm:
        model["correction_mode"] = arm["correction_mode"]
    if arm["source_width"]:
        run["granularity"] = arm["source_width"]
    else:
        run["sampling_mode"] = "nested-random"
        model.update(
            granularity_sampling_mode="global",
            global_sampling_schedule="random_with_replacement",
            global_sampling_interval_steps=1,
        )
    if arm.get("sampling_policy") == "fixed_inverse_membership":
        model["granularity_sampling_mode"] = "fixed_global"
        model["global_sampling_distribution"] = dict(zip(WIDTH_LABELS, IM_PROBABILITIES))
        model.pop("global_sampling_schedule")
        model.pop("global_sampling_interval_steps")
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


def _check_unoccupied(run_output_root, arms=ARMS):
    from pathlib import Path
    from src.utils.config import ConfigError

    root = Path(run_output_root).expanduser().resolve()
    for path in [_reservation_path(root), *(root / arm["arm_id"] for arm in arms)]:
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
    arms = campaign_arms(recipe["schema_version"])
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
    _require_equal(set(recipe["arms"]), {a["arm_id"] for a in arms}, "arms")
    _check_unoccupied(run_output_root, arms)
    provenance = _provenance()
    runs = []
    with tempfile.TemporaryDirectory(prefix="ownership-resolve-") as staging:
        for arm in arms:
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
                "sampling": _sampling_contract(resolved, arm),
                "data": data_controls,
                "budget": {
                    k: arm[k]
                    for k in ("assigned_epochs", "assigned_updates", "assigned_tokens")
                },
                "evaluation": copy.deepcopy(resolved["evaluation"]),
                "count_convention": PARAMETER_COUNT_CONVENTION,
            }
            if "correction_mode" in arm:
                contract["correction"] = membership_correction_contract(arm["correction_mode"])
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
                        if "correction_mode" in run:
                            _require_equal(list(mlp.gradient_membership_counts), [4, 3, 2, 1], "membership counts")
                            _require_equal(list(mlp.gradient_membership_correction_scales), [1., 4/3, 2., 4.], "membership factors")
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
    """Production categorical primitive and isolated action seed, H=1."""
    import hashlib
    import random
    from src.utils.reproducibility import seed_for

    generator = random.Random(seed_for(config, "granularity_selection"))
    digest = hashlib.sha256()
    counts = dict.fromkeys(WIDTH_LABELS, 0)
    fixed = config["model"].get("granularity_sampling_mode") == "fixed_global"
    probabilities = [config["model"]["global_sampling_distribution"][w] for w in WIDTH_LABELS] if fixed else [.25] * 4
    for _ in range(config["training"]["max_steps"]):
        width = (generator.choices(WIDTH_LABELS, weights=probabilities, k=1)[0] if fixed
                 else WIDTH_LABELS[generator.randrange(len(WIDTH_LABELS))])
        digest.update((width + "\n").encode("ascii"))
        counts[width] += 1
    return {
        "sha256": digest.hexdigest(),
        "encoding": "ASCII width label plus LF per update",
        "updates": sum(counts.values()),
        "counts": counts,
        "expected_counts": {w: config["training"]["max_steps"] * p for w, p in zip(WIDTH_LABELS, probabilities)},
        "sampling": "fixed_inverse_membership" if fixed else "uniform_random_with_replacement",
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
    first = result[runs[0]["arm_id"]]["epochs"][0]
    elastic = next(result[r["arm_id"]] for r in runs if not r["source_width"])
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
    arms = campaign_arms(recipe["schema_version"])
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
        "schema_version": recipe["schema_version"],
        "campaign_id": recipe["campaign_id"],
        "seed": SEED,
        "common": recipe["common"],
        "allowed_difference_matrix": list(arms),
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
        _check_unoccupied(root, arms)
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
    arm = next((a for a in (*ARMS, *CORRECTION_ARMS, *INVERSE_MEMBERSHIP_ARMS) if a["arm_id"] == config["run"].get("arm_id")), None)
    if arm is None:
        raise ConfigError("optimizer_ownership_contract: unknown arm_id")
    if arm.get("sampling_policy") == "fixed_inverse_membership":
        _require_equal(config["model"]["granularity_sampling_mode"], "fixed_global", "IM sampling mode")
        _require_equal(config["model"]["granularities"], list(WIDTH_LABELS), "IM ordered widths")
        _require_equal(config["model"]["global_sampling_distribution"], dict(zip(WIDTH_LABELS, IM_PROBABILITIES)), "IM probabilities")
        _require_equal(config["model"]["global_sampling_interval_steps"], 1, "IM cadence")
    expected_mode = arm.get("correction_mode", "none")
    _require_equal(config["model"]["correction_mode"], expected_mode, "model.correction_mode")
    if "correction_mode" in arm:
        _require_equal(contract.get("correction"), membership_correction_contract(expected_mode), "correction contract")
    elif "correction" in contract:
        raise ConfigError("Original campaign cannot contain a correction extension")
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
        "sampling": _sampling_contract(config, arm),
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


def inspect_run_observations(run_dir, summary):
    """Stream saved commits and clipping records, validating their accounting."""
    import hashlib
    import json
    import math
    from pathlib import Path
    import numpy as np
    from src.utils.config import ConfigError

    root = Path(run_dir)
    audit = summary['optimizer_ownership']
    if (summary.get('optimizer_ownership_schema_version') != 1 or audit.get('schema_version') != 1
            or stable_hash(audit['contract']) != audit['contract_hash']
            or summary.get('run_id') != audit['run_id']):
        raise ConfigError('Run summary schema/contract identity mismatch')
    scope = audit['state_scope']
    counts = dict.fromkeys(audit['width_selection_counts'], 0)
    action_digest = hashlib.sha256()
    epoch_digests = {}
    clipping_summary = {}
    clip_path = root / audit['clipping_path'] if audit.get('clipping_path') else None
    clip_stream = clip_path.open() if clip_path else None
    last = None
    try:
        with (root / audit['trace_path']).open() as stream:
            for ordinal, line in enumerate(stream, 1):
                row = json.loads(line)
                for key in ('run_id', 'campaign_id', 'arm_id', 'contract_hash'):
                    if row.get(key) != audit.get(key):
                        raise ConfigError(f'Trace provenance mismatch: {key}')
                if row.get('schema_version') != 1 or row['step'] != ordinal or row['action_ordinal'] != ordinal or row['scheduler_position'] != ordinal:
                    raise ConfigError('Trace committed step/action/clock mismatch')
                width = row['width']
                if width not in counts:
                    raise ConfigError('Trace selected width mismatch')
                counts[width] += 1
                quarters = {f'O-{q}': sum(counts.get(w, 0) for w in WIDTH_LABELS[i:]) for i, q in enumerate('ABCD')} if len(counts) == 4 else {}
                calls = {**quarters, 'O-common': ordinal} if scope == 'per_ffn_block' else counts if scope == 'per_granularity' else {'shared': ordinal}
                active = list(OWNER_IDS[:WIDTH_LABELS.index(width) + 1]) + ['O-common'] if scope == 'per_ffn_block' else [width] if scope == 'per_granularity' else ['shared']
                if row['width_selection_counts'] != counts or row['quarter_activation_counts'] != quarters or row['owner_call_counts'] != calls or row['active_owners'] != active:
                    raise ConfigError('Trace exposure/owner accounting mismatch')
                if row['packed_tokens'] != audit['packed_tokens_per_update'] or row['tokens_seen'] != ordinal * audit['packed_tokens_per_update']:
                    raise ConfigError('Trace packed tokens mismatch')
                encoded_action = (width + '\n').encode('ascii')
                if hashlib.sha256(encoded_action).hexdigest() != row['action_sha256']:
                    raise ConfigError('Trace action digest mismatch')
                action_digest.update(encoded_action)
                sample_ids = row.get('sample_ids')
                if sample_ids is not None:
                    encoded = np.asarray(sample_ids, dtype='<u8').tobytes()
                    if hashlib.sha256(encoded).hexdigest() != row['batch_sha256']:
                        raise ConfigError('Trace batch digest mismatch')
                    provenance = row['batch_provenance']
                    epoch = provenance['epoch_index']
                    epoch_digests.setdefault(epoch, hashlib.sha256()).update(encoded)
                    sampler = audit.get('sampler_state')
                    if sampler:
                        batch_size = sampler['distributed_batch_geometry']['global_batch_size']
                        cursor = ordinal * batch_size
                        expected_epoch, within = divmod(cursor, sampler['epoch_sample_count'])
                        if (len(sample_ids) != batch_size or provenance['total_cursor'] != cursor - batch_size
                                or row['epoch'] != expected_epoch or row['batch_index'] != within // batch_size
                                or provenance['fixed_epoch_set_hash'] != sampler['fixed_epoch_set_hash']
                                or provenance['permutation_hash'] != sampler['permutation_hash']):
                            raise ConfigError('Trace epoch/cursor/order mismatch')
                elif audit.get('sampler_state'):
                    raise ConfigError('Packed trace sample identities are missing')
                if clip_stream is not None:
                    clip_line = clip_stream.readline()
                    if not clip_line:
                        raise ConfigError('Missing committed clipping observation')
                    clip = json.loads(clip_line)
                    if (clip.get('schema_version') != 1 or clip['step'] != ordinal or clip['width'] != width
                            or any(clip.get(k) != row[k] for k in ('run_id', 'contract_hash', 'campaign_id', 'arm_id', 'attempt_id'))):
                        raise ConfigError('Clipping provenance/trace mismatch')
                    _accumulate_clipping(clipping_summary, clip, audit['clipping_contract'])
                last = row
        if clip_stream is not None and clip_stream.read().strip():
            raise ConfigError('Extra non-committed clipping observations')
    finally:
        if clip_stream is not None:
            clip_stream.close()
    if last is None or last['step'] != audit['steps']:
        raise ConfigError('Trace/summary committed step mismatch')
    for key in ('tokens_seen', 'epoch', 'batch_index', 'scheduler_position', 'width_selection_counts', 'quarter_activation_counts', 'owner_call_counts'):
        if last[key] != audit[key]:
            raise ConfigError(f'Trace/summary mismatch: {key}')
    if not audit['accounting_reconciled']:
        raise ConfigError('Summary accounting is not reconciled')
    for groups in clipping_summary.values():
        for row in groups.values():
            n = row['active_observations']
            row['frequency'] = row['clipped_observations'] / n if n else None
            pre, post = row.pop('sum_pre_norm'), row.pop('sum_post_norm')
            row['mean_pre_norm'] = pre / n if n else None
            row['mean_post_norm'] = post / n if n else None
    return {'clipping_by_width': clipping_summary, 'action_sha256': action_digest.hexdigest(),
            'epoch_order_sha256': {str(k): v.hexdigest() for k, v in epoch_digests.items()},
            'committed_updates': last['step']}


def _accumulate_clipping(summary, clip, contract):
    import math
    from src.utils.config import ConfigError

    width = clip['width']
    if clip['mode'] != contract['mode'] or set(clip['groups']) != set(OWNER_IDS):
        raise ConfigError('Clipping mode/group topology mismatch')
    groups = summary.setdefault(width, {})
    active_owners = (*OWNER_IDS[:WIDTH_LABELS.index(width) + 1], 'O-common')
    for name in OWNER_IDS:
        item = clip['groups'][name]
        active = name in active_owners
        if item['active'] is not active:
            raise ConfigError('Clipping active flag mismatch')
        aggregate = groups.setdefault(name, {'active_observations': 0, 'clipped_observations': 0, 'sum_pre_norm': 0., 'sum_post_norm': 0.})
        if not active:
            if any(item[key] is not None for key in ('pre_norm', 'post_norm', 'coefficient', 'max_norm')):
                raise ConfigError('Inactive clipping values must be null')
            continue
        pre, post, coefficient = (item[k] for k in ('pre_norm', 'post_norm', 'coefficient'))
        if any(not math.isfinite(v) or v < 0 for v in (pre, post, coefficient)) or coefficient > 1:
            raise ConfigError('Invalid clipping norm/coefficient')
        cap = contract['owner_max_norms'][name] if clip['mode'] == 'per_owner' else contract['max_norm']
        expected = min(1., cap / ((pre if clip['mode'] == 'per_owner' else clip['combined_pre_norm']) + 1e-6))
        if (not math.isclose(coefficient, expected, rel_tol=2e-5, abs_tol=1e-7)
                or not math.isclose(post, pre * coefficient, rel_tol=2e-5, abs_tol=1e-7)):
            raise ConfigError('Clipping applied coefficient/norm mismatch')
        if clip['mode'] == 'per_owner':
            if item['max_norm'] != cap or clip['global_coefficient'] is not None or clip['global_max_norm'] is not None:
                raise ConfigError('Per-owner clipping cap/global rescale mismatch')
        elif item['max_norm'] is not None or coefficient != clip['global_coefficient'] or clip['global_max_norm'] != cap:
            raise ConfigError('Global clipping coefficient/cap mismatch')
        aggregate['active_observations'] += 1
        aggregate['clipped_observations'] += coefficient < 1
        aggregate['sum_pre_norm'] += pre
        aggregate['sum_post_norm'] += post
    for field in ('pre_norm', 'post_norm'):
        expected = math.sqrt(sum(item[field] ** 2 for item in clip['groups'].values() if item['active']))
        if not math.isclose(expected, clip['combined_' + field], rel_tol=2e-5, abs_tol=1e-7):
            raise ConfigError('Combined clipping norm mismatch')


def report_run_artifacts(run_dir, output_dir, *, partial=False):
    """Render one run from saved ordinary metrics and audited trace sidecars."""
    import csv
    import json
    import math
    from pathlib import Path
    from matplotlib.figure import Figure
    from src.utils.config import ConfigError
    from src.training.run import ResourceAttemptLedger

    root, destination = Path(run_dir), Path(output_dir)
    summary = json.loads((root / 'run_summary.json').read_text())
    audit = summary['optimizer_ownership']
    observations = inspect_run_observations(root, summary)
    trajectories = {}
    with (root / 'metrics.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['split'] not in ('train', 'validation'):
                continue
            if row.get('optimizer_step_committed', '').lower() == 'false':
                continue
            step, loss, perplexity = int(row['step']), float(row['loss']), float(row['perplexity'])
            if step > audit['steps'] or not math.isfinite(loss) or not math.isfinite(perplexity):
                raise ConfigError('Non-durable or nonfinite metric trajectory')
            series = trajectories.setdefault(row['split'] + ':' + row['granularity'], {'step': [], 'loss': [], 'perplexity': []})
            series['step'].append(step); series['loss'].append(loss); series['perplexity'].append(perplexity)
    resources = dict(audit['resources'])
    if (root / 'resource_attempts.json').exists():
        ledger = ResourceAttemptLedger(root, run_id=audit['run_id'])
        resources.update(ledger.summary())
        seconds = resources['elapsed_seconds']
        resources['useful_committed_tokens_per_second'] = audit['tokens_seen'] / seconds if seconds else None
        resources['attempted_tokens_per_second'] = resources['attempted_steps'] * audit['packed_tokens_per_update'] / seconds if seconds else None
    label = 'complete measurements' if resources['measurement_complete'] else 'incomplete measurements'
    prefix = 'PARTIAL campaign diagnostic — ' if partial else ''
    report = {'schema_version': 1, 'run_id': audit['run_id'], 'arm_id': audit['arm_id'],
              'contract_hash': audit['contract_hash'], 'trajectories': trajectories,
              'resources': resources, 'resource_label': label, **observations,
              'expected_exposure': audit['expected_exposure'], 'figures': []}
    destination.mkdir(parents=True, exist_ok=True)
    for metric in ('loss', 'perplexity'):
        figure = Figure(figsize=(8, 5)); ax = figure.subplots()
        for name, series in trajectories.items():
            ax.plot(series['step'], series[metric], label=name,
                    marker='o' if name.startswith('validation:') else None, markersize=3)
        ax.set(xlabel='Committed optimizer updates', ylabel=metric.capitalize(), title=f"{prefix}{audit['arm_id']} — ordinary validation and training")
        if trajectories: ax.legend()
        figure.tight_layout()
        for suffix in ('png', 'pdf'):
            path = destination / f'{audit["arm_id"]}_{metric}_trajectory.{suffix}'
            figure.savefig(path); report['figures'].append(str(path.resolve()))
    figure = Figure(figsize=(12, 8)); axes = figure.subplots(2, 3).flat
    owners = audit['storage']['owners']
    owner_bytes = [sum(r['bytes'] for r in audit['storage']['components'] if r['owner_id'] == owner['owner_id']) for owner in owners]
    axes[0].bar([r['owner_id'] for r in owners], owner_bytes)
    axes[0].set(title='Allocated persistent optimizer tensors', ylabel='Bytes')
    axes[0].tick_params(axis='x', labelrotation=25)
    widths = [w for w in WIDTH_LABELS if w in audit['width_selection_counts']]
    axes[1].bar(widths, [audit['width_selection_counts'][w] for w in widths], label='Realized selections')
    expectations = audit['expected_exposure']['width_selections']
    if expectations: axes[1].scatter(widths, [expectations[w] for w in widths], marker='_', color='black', label='Uniform expectation')
    axes[1].set(title='Width exposure', ylabel='Committed selections'); axes[1].legend()
    peaks = [(k, resources.get(k)) for k in ('peak_allocated_bytes', 'peak_reserved_bytes') if resources.get(k) is not None]
    if peaks: axes[2].bar([k.replace('peak_', '').replace('_bytes', '') for k, _ in peaks], [v for _, v in peaks])
    else:
        axes[2].text(.5, .5, 'GPU peaks unavailable', ha='center', transform=axes[2].transAxes)
        axes[2].set_axis_off()
    axes[2].set(title='CUDA allocator peaks', ylabel='Bytes')
    quarters = audit['quarter_activation_counts']
    if quarters:
        axes[3].bar(list(quarters), list(quarters.values()), label='Realized activations')
        expected = audit['expected_exposure']['quarter_activations']
        axes[3].scatter(list(quarters), [expected[q] for q in quarters], marker='_', color='black', label='Uniform expectation')
        axes[3].legend()
    else:
        axes[3].text(.5, .5, 'Dense standalone', ha='center', transform=axes[3].transAxes)
        axes[3].set_axis_off()
    axes[3].set(title='Quarter activation', ylabel='Committed activations')
    if resources['elapsed_seconds'] is not None:
        axes[4].bar(['All attempts'], [resources['elapsed_seconds']])
    else:
        axes[4].text(.5, .5, 'Time unavailable', ha='center', transform=axes[4].transAxes)
        axes[4].set_axis_off()
    axes[4].set(title='Cumulative runtime', ylabel='Seconds (queue downtime excluded)')
    rates = [resources.get(k) for k in ('useful_committed_tokens_per_second', 'attempted_tokens_per_second')]
    if all(value is not None for value in rates):
        axes[5].bar(['Useful committed', 'Attempted'], rates)
    else:
        axes[5].text(.5, .5, 'Throughput unavailable', ha='center', transform=axes[5].transAxes)
        axes[5].set_axis_off()
    axes[5].set(title='Throughput including replay costs', ylabel='Packed tokens / second')
    figure.suptitle(f"{prefix}{audit['arm_id']} resources — {label}"); figure.tight_layout()
    for suffix in ('png', 'pdf'):
        path = destination / f'{audit["arm_id"]}_resources.{suffix}'
        figure.savefig(path); report['figures'].append(str(path.resolve()))
    write_json_artifact(destination / 'run_diagnostics.json', report)
    return report


def _read_json(path):
    import json
    from pathlib import Path
    from src.utils.config import ConfigError

    try:
        value = json.loads(Path(path).read_text())
    except (ValueError, OSError) as error:
        raise ConfigError(f'{path}: {error}') from error
    if not isinstance(value, dict):
        raise ConfigError(f'{path}: expected an object')
    return value


def _check_content_hash(record, field, label):
    _require_equal(record.get(field), stable_hash({k: v for k, v in record.items() if k != field}), f'{label}.{field}')


def _source_record(path):
    from pathlib import Path
    from src.training.packed_corpus import sha256_file

    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256_file(path)}


def _check_sources(sources):
    from pathlib import Path
    from src.training.packed_corpus import sha256_file
    from src.utils.config import ConfigError

    for source in sources:
        path = Path(source['path'])
        if not path.is_file() or sha256_file(path) != source['sha256']:
            raise ConfigError(f'Frozen source hash mismatch: {path}')


def _read_preflight_manifest(path):
    """Validate saved controls without opening the corpus or constructing a model."""
    import copy
    from src.utils.config import ConfigError

    manifest = _read_json(path)
    _check_content_hash(manifest, 'manifest_hash', 'preflight')
    arms = campaign_arms(manifest.get('schema_version'))
    arm_ids = [a['arm_id'] for a in arms]
    for key, expected in {
        'schema_version': manifest['schema_version'], 'seed': SEED,
        'common': PINNED_COMMON, 'allowed_difference_matrix': [
            {**a, 'endpoint_widths': list(a['endpoint_widths'])} for a in arms],
        'expected_data': PINNED_DATA, 'count_convention': PARAMETER_COUNT_CONVENTION,
        'evaluation_role': EVALUATION_ROLE,
    }.items():
        _require_equal(manifest.get(key), expected, f'preflight.{key}')
    _require_equal([r['arm_id'] for r in manifest['runs']], arm_ids, 'preflight.arms')
    run_ids = set()
    def check_resolved_controls(expected, actual, field):
        # Resolution may add defaults, but every explicit scientific input must
        # survive unchanged except the documented dense-width/scope normalization.
        for key, value in expected.items():
            if key not in actual:
                raise ConfigError(f'{field}.{key}: missing resolved control')
            if isinstance(value, dict):
                check_resolved_controls(value, actual[key], field + '.' + key)
            else:
                _require_equal(actual[key], value, field + '.' + key)
    first_epoch = manifest['expected_traces'][arm_ids[0]]['epochs'][0]
    for arm, run in zip(arms, manifest['runs'], strict=True):
        label = arm['arm_id']
        for key, value in arm.items():
            # JSON converts tuple width lists to lists.
            _require_equal(run.get(key), list(value) if isinstance(value, tuple) else value, f'{label}.{key}')
        _require_equal(run['campaign_id'], manifest['campaign_id'], f'{label}.campaign_id')
        _require_equal(run['run_id'], f"{manifest['campaign_id']}-{label}-s{SEED}", f'{label}.run_id')
        if run['run_id'] in run_ids:
            raise ConfigError('Duplicate campaign run identity')
        run_ids.add(run['run_id'])
        config = run['resolved_config']
        validate_materialized_config(config)
        contract = config['optimizer_ownership_contract']
        _require_equal(run['optimizer_ownership_contract'], contract, f'{label}.contract')
        _require_equal(run['contract_hash'], stable_hash(contract), f'{label}.contract_hash')
        _require_equal(manifest['run_contract_hashes'][label], run['contract_hash'], f'{label}.preflight contract_hash')
        _require_equal(run['initialization'], contract['initialization'], f'{label}.initialization')
        _require_equal(run['clipping'], contract['clipping'], f'{label}.clipping')
        raw = copy.deepcopy(run['executable_config'])
        _require_equal(raw.pop('optimizer_ownership_contract'), contract, f'{label}.executable contract')
        _require_equal(raw.pop('optimizer_ownership_contract_hash'), run['contract_hash'], f'{label}.executable hash')
        for key in ('run_id', 'campaign_id', 'arm_id'):
            _require_equal(raw['run'].pop(key), run[key], f'{label}.{key}')
        _require_equal(raw['run'].pop('output_dir'), run['output_path'], f'{label}.output_path')
        raw['model']['tokenizer_dir'] = PINNED_COMMON['model']['tokenizer_dir']
        raw['dataset']['prepared_corpus_dir'] = PINNED_COMMON['dataset']['prepared_corpus_dir']
        _require_equal(raw, _merge(PINNED_COMMON, _arm_overrides(arm)), f'{label}.scientific controls')
        scientific = copy.deepcopy(run['executable_config'])
        scientific['training']['optimizer'].pop('state_scope')
        scientific['training']['optimizer'].pop('scheduler_clock')
        if arm['source_width']:
            scientific['model']['granularities'] = [arm['source_width']]
            scientific['model']['granularity_prefixes'] = {arm['source_width']: 1.0}
        check_resolved_controls({k: scientific[k] for k in ('run', 'model', 'training', 'dataset', 'evaluation')}, config, f'{label}.resolved')
        traces = manifest['expected_traces'][label]
        _require_equal(len(traces['epochs']), arm['assigned_epochs'], f'{label}.expected epochs')
        _require_equal(traces['epochs'][0], first_epoch, f'{label}.first epoch')
        if not arm['source_width']:
            _require_equal(traces, manifest['expected_traces'][next(a['arm_id'] for a in arms if not a['source_width'])], f'{label}.elastic traces')
    return manifest


def _terminal_endpoints(sidecar, run, *, allow_partial):
    import math
    from src.utils.config import ConfigError

    label = run['arm_id']
    contract = run['optimizer_ownership_contract']
    _check_content_hash(sidecar, 'content_hash', f'{label}.terminal')
    protocol = dict(contract['evaluation']['validation'])
    # The runtime binds the pinned manifest when data loaders are constructed.
    if 'manifest_hash' in sidecar.get('evaluation_protocol', {}):
        protocol['manifest_hash'] = PINNED_DATA['ordinary_validation_manifest_hash']
    identity = {
        'schema_version': TERMINAL_VALIDATION_SCHEMA_VERSION,
        **{k: run[k] for k in ('campaign_id', 'arm_id', 'run_id', 'contract_hash', 'representation', 'state_scope', 'clipping', 'initialization')},
        'contract': contract, 'global_step': run['assigned_updates'],
        'evaluation_role': EVALUATION_ROLE,
        'validation_manifest_hash': PINNED_DATA['ordinary_validation_manifest_hash'],
        'evaluation_protocol': protocol, 'evaluation_protocol_hash': stable_hash(protocol),
        'validation_loss_aggregation': VALIDATION_AGGREGATION,
        'count_convention': PARAMETER_COUNT_CONVENTION,
        'evaluation_examples': VALIDATION_SEQUENCES,
        # Packed validation has fixed context, no padding, all causal targets valid.
        'evaluation_target_tokens': VALIDATION_TARGET_TOKENS,
    }
    for unit in ('updates', 'tokens', 'epochs'):
        identity['assigned_' + unit] = identity['actual_' + unit] = run['assigned_' + unit]
    for key, value in identity.items():
        _require_equal(sidecar.get(key), value, f'{label}.terminal.{key}')
    rows = sidecar.get('endpoints')
    if not isinstance(rows, list):
        raise ConfigError(f'{label}.endpoints must be a list')
    seen = set()
    for row in rows:
        width = row.get('width')
        if width not in run['endpoint_widths'] or width in seen:
            raise ConfigError(f'{label}.duplicate or unexpected endpoint: {width}')
        seen.add(width)
        for key in ('evaluation_role', 'validation_manifest_hash', 'evaluation_protocol_hash', 'validation_loss_aggregation',
                    'count_convention', 'evaluation_examples', 'evaluation_target_tokens'):
            _require_equal(row.get(key), identity[key], f'{label}.{width}.{key}')
        count = next(w['non_embedding_parameters'] for w in WIDTHS if w['label'] == width)
        _require_equal(row.get('non_embedding_parameters'), count, f'{label}.{width}.non_embedding_parameters')
        loss, perplexity = row['loss'], row['perplexity']
        if (isinstance(loss, bool) or isinstance(perplexity, bool) or not math.isfinite(loss)
                or not math.isfinite(perplexity) or perplexity <= 0 or loss > 709
                or not math.isclose(perplexity, math.exp(loss), rel_tol=1e-12)):
            raise ConfigError(f'{label}.{width}.nonfinite or inconsistent loss/perplexity')
    if not allow_partial and seen != set(run['endpoint_widths']):
        raise ConfigError(f'{label}.missing endpoints: {sorted(set(run["endpoint_widths"]) - seen)}')
    return sorted(rows, key=lambda r: WIDTH_LABELS.index(r['width']))


def _inspect_terminal_run(root, run, expected_traces, *, allow_partial):
    """Read only saved terminal evidence; never choose a checkpoint from metrics."""
    from pathlib import Path
    import torch
    from src.utils.config import ConfigError

    root = Path(root).resolve()
    label = run['arm_id']
    summary_path = root / 'run_summary.json'
    sidecar_path = root / 'terminal_validation_results.json'
    summary, sidecar = _read_json(summary_path), _read_json(sidecar_path)
    endpoints = _terminal_endpoints(sidecar, run, allow_partial=allow_partial)
    _require_equal(summary.get('status'), 'completed', f'{label}.summary.status')
    _require_equal(summary.get('run_id'), run['run_id'], f'{label}.summary.run_id')
    audit = summary['optimizer_ownership']
    _require_equal(summary.get('optimizer_ownership_schema_version'), 1, f'{label}.summary schema')
    for key, value in {
        'schema_version': 1, 'contract': run['optimizer_ownership_contract'],
        **{k: run[k] for k in ('run_id', 'campaign_id', 'arm_id', 'contract_hash', 'state_scope')},
        'clipping_contract': run['resolved_config']['training']['gradient_clipping'],
        'steps': run['assigned_updates'], 'tokens_seen': run['assigned_tokens'],
        'epoch': run['assigned_epochs'], 'batch_index': 0,
        'packed_tokens_per_update': TOKENS_PER_UPDATE,
        'scheduler_position': run['assigned_updates'], 'accounting_reconciled': True,
        'trace_path': 'optimizer_ownership_trace.jsonl',
        'clipping_path': 'optimizer_ownership_clipping.jsonl' if run['representation'] == 'concat' and run['state_scope'] != 'per_granularity' else None,
    }.items():
        _require_equal(audit.get(key), value, f'{label}.summary.{key}')
    checkpoint = Path(sidecar['checkpoint_path'])
    if not checkpoint.is_absolute():
        raise ConfigError(f'{label}.checkpoint_path must be absolute')
    sources = [_source_record(p) for p in (summary_path, sidecar_path, checkpoint, root / audit['trace_path'], root / 'metrics.csv')]
    if audit['clipping_path']:
        sources.append(_source_record(root / audit['clipping_path']))
    # Bind ledger absence too: a later recovery attempt invalidates the freeze.
    optional_sources = {str(root / 'resource_attempts.json'): (root / 'resource_attempts.json').exists()}
    if optional_sources[str(root / 'resource_attempts.json')]:
        sources.append(_source_record(root / 'resource_attempts.json'))
    _require_equal(sources[2]['sha256'], sidecar['checkpoint_sha256'], f'{label}.checkpoint_sha256')
    _require_equal(checkpoint.stat().st_size, sidecar['checkpoint_bytes'], f'{label}.checkpoint_bytes')
    for key in ('path', 'sha256', 'bytes'):
        _require_equal(audit['checkpoint'].get('terminal_checkpoint_' + key), sidecar['checkpoint_' + key], f'{label}.checkpoint_{key}')
    _require_equal(audit['checkpoint'].get('terminal_checkpoint_purpose'), 'resumable_training', f'{label}.checkpoint purpose')
    import pickle
    try:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        raise ConfigError(f'{label}.checkpoint cannot be read: {error}') from error
    for key, value in {
        'checkpoint_kind': 'resumable_training', 'checkpoint_schema_version': 1,
        'optimizer_ownership_checkpoint_schema_version': 1, 'run_id': run['run_id'],
        'optimizer_ownership_contract': run['optimizer_ownership_contract'],
        'optimizer_ownership_contract_hash': run['contract_hash'],
        'step': audit['steps'], 'tokens_seen': audit['tokens_seen'], 'epoch': audit['epoch'], 'batch_index': 0,
        'global_scheduler_position': audit['scheduler_position'],
        'optimizer_width_selection_counts': audit['width_selection_counts'],
        'optimizer_quarter_activation_counts': audit['quarter_activation_counts'],
        'optimizer_update_counts': audit['owner_call_counts'],
    }.items():
        _require_equal(saved.get(key), value, f'{label}.checkpoint.{key}')
    optimizer_key = 'optimizer_state_dict' if run['state_scope'] == 'shared' else 'optimizer_state_collection'
    for key in ('model_state_dict', optimizer_key, 'scheduler_state_dict'):
        if not isinstance(saved.get(key), dict) or not saved[key]:
            raise ConfigError(f'{label}.checkpoint missing {key}')
    if optimizer_key == 'optimizer_state_collection':
        _require_equal(saved['optimizer_state_collection'].get('total_successful_updates'), audit['steps'], f'{label}.checkpoint collection updates')
        _require_equal(saved['scheduler_state_dict'].get('position'), audit['steps'], f'{label}.checkpoint clock position')
    del saved
    observations = inspect_run_observations(root, summary)
    _require_equal(observations['committed_updates'], run['assigned_updates'], f'{label}.trace updates')
    _require_equal(observations['epoch_order_sha256'], {str(e['epoch_index']): e['sha256'] for e in expected_traces['epochs']}, f'{label}.epoch trace digest')
    if not run['source_width']:
        _require_equal(observations['action_sha256'], expected_traces['actions']['sha256'], f'{label}.action trace digest')
    _check_sources(sources)
    return {'arm_id': label, 'run_id': run['run_id'], 'run_dir': str(root), 'sources': sources,
            'optional_sources': optional_sources, 'checkpoint_sha256': sidecar['checkpoint_sha256'],
            'terminal_content_hash': sidecar['content_hash'], 'endpoints': endpoints, 'observations': observations}


def _missing_endpoints(runs, arms=ARMS):
    present = {(r['arm_id'], e['width']) for r in runs for e in r['endpoints']}
    return [{'arm_id': a['arm_id'], 'width': w} for a in arms for w in a['endpoint_widths'] if (a['arm_id'], w) not in present]


def _publish_directory(output_dir, write):
    """Expose a success directory only after every export and final source check."""
    import shutil
    import tempfile
    from pathlib import Path
    from src.utils.config import ConfigError

    output = Path(output_dir).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ConfigError(f'occupied output: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    try:
        result = write(stage, output)
        if output.exists():
            raise ConfigError(f'occupied output: {output}')
        stage.rename(output)
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def freeze_campaign(*, campaign_manifest, output_dir, run_root=None, run_dirs=None, allow_partial=False):
    """Freeze the exact preflight-bound terminal sources, including diagnostics."""
    from pathlib import Path
    from src.utils.config import ConfigError

    if (run_root is None) == (run_dirs is None):
        raise ConfigError('Specify exactly one of run-root or run-dir')
    try:
        preflight_source = _source_record(campaign_manifest)
        manifest = _read_preflight_manifest(campaign_manifest)
        arms = campaign_arms(manifest['schema_version'])
        arm_ids = [a['arm_id'] for a in arms]
        by_id = {r['run_id']: r for r in manifest['runs']}
        if run_root is not None:
            # Directory names locate candidates only; saved run identity is authoritative.
            paths = [Path(run_root) / Path(r['output_path']).name for r in manifest['runs']]
            if allow_partial:
                paths = [p for p in paths if p.exists()]
        else:
            paths = [Path(p) for p in run_dirs]
        runs, seen = [], set()
        for path in paths:
            summary = _read_json(path / 'run_summary.json')
            run_id = summary.get('run_id')
            if run_id not in by_id or run_id in seen:
                raise ConfigError(f'Duplicate or non-preflight run identity: {run_id}')
            seen.add(run_id)
            run = by_id[run_id]
            runs.append(_inspect_terminal_run(path, run, manifest['expected_traces'][run['arm_id']], allow_partial=allow_partial))
        runs.sort(key=lambda r: arm_ids.index(r['arm_id']))
        missing = _missing_endpoints(runs, arms)
        if missing and not allow_partial:
            raise ConfigError(f'missing campaign endpoints: {missing}')
        if not any(r['endpoints'] for r in runs):
            raise ConfigError('missing all campaign endpoints')
        frozen = {'schema_version': FROZEN_MANIFEST_SCHEMA_VERSION, 'campaign_id': manifest['campaign_id'],
                  'status': 'partial' if missing else 'complete', 'missing_endpoints': missing,
                  'missing_arms': [a for a in arm_ids if a not in {r['arm_id'] for r in runs}],
                  'preflight_source': preflight_source, 'preflight_manifest_hash': manifest['manifest_hash'],
                  'runs': runs, 'evaluation_role': EVALUATION_ROLE, 'holdout_evaluated': False}
        frozen['content_hash'] = stable_hash(frozen)
        def publish(stage, output):
            _check_sources([preflight_source] + [s for r in runs for s in r['sources']])
            write_json_artifact(stage / 'frozen_manifest.json', frozen)
            return frozen
        return _publish_directory(output_dir, publish)
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ConfigError(f'Malformed campaign terminal evidence: {error}') from error


def endpoint_csv_value(value):
    """One lossless scalar/JSON-cell convention shared by both table exports."""
    import json
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) if not isinstance(value, str) else value


def _endpoint_table(frozen, preflight):
    rows = []
    definitions = {r['arm_id']: r for r in preflight['runs']}
    for saved in frozen['runs']:
        run = definitions[saved['arm_id']]
        summary = _read_json(saved['run_dir'] + '/run_summary.json')['optimizer_ownership']
        sidecar = _read_json(saved['run_dir'] + '/terminal_validation_results.json')
        for endpoint in saved['endpoints']:
            width = next(w for w in WIDTHS if w['label'] == endpoint['width'])
            row = {'status': frozen['status'], 'campaign_id': frozen['campaign_id'], 'arm_id': run['arm_id'],
                   'run_id': run['run_id'], 'seed': SEED, 'representation': run['representation'],
                   'state_scope': run['state_scope'], 'clipping': run['clipping'],
                   'sampling_policy': run.get('sampling_policy', 'standalone' if run['source_width'] else 'uniform'),
                   'sampling_contract': run['optimizer_ownership_contract']['sampling'],
                   'historical_reference': False,
                   'checkpoint_bytes': sidecar['checkpoint_bytes'],
                   'correction_mode': run.get('correction_mode', 'none'),
                   'correction': run['optimizer_ownership_contract'].get('correction'),
                   'width_fraction': width['source_fraction'], 'ffn_dimension': width['active_ffn_dimension'], **endpoint,
                   'contract_hash': run['contract_hash'], 'initialization': run['initialization'],
                   'checkpoint_path': sidecar['checkpoint_path'], 'checkpoint_sha256': sidecar['checkpoint_sha256'],
                   'terminal_content_hash': sidecar['content_hash'],
                   'width_selection_counts': summary['width_selection_counts'],
                   'quarter_activation_counts': summary['quarter_activation_counts'], 'owner_call_counts': summary['owner_call_counts'],
                   'optimizer_storage': summary['storage'], 'resources': summary['resources'],
                   'temporary_concat_storage': summary['temporary_concat_storage']}
            for unit in ('updates', 'tokens', 'epochs'):
                for kind in ('actual', 'assigned'):
                    row[kind + '_' + unit] = sidecar[kind + '_' + unit]
            rows.append(row)
    return rows


def endpoint_figure(rows, *, metric, partial):
    """Retain separate series even when all measured points coincide."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11, 7))
    ax = figure.subplots()
    colors = dict(zip(('S1', 'S2', 'C1', 'C2', 'C3'), ('#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00')))
    standalone_labeled = False
    for arm in (*ELASTIC_ARMS, *CORRECTION_ARMS, *INVERSE_MEMBERSHIP_ARMS, *STANDALONE_ARMS):
        values = sorted((r for r in rows if r['arm_id'] == arm['arm_id']), key=lambda r: r['non_embedding_parameters'])
        if not values:
            continue
        elastic = not arm['source_width']
        mode = arm.get('correction_mode', 'none')
        label = arm['arm_id'] if elastic else ('Standalone' if not standalone_labeled else '_nolegend_')
        if elastic and any(r['arm_id'] in CORRECTION_ARM_IDS for r in rows):
            label = f"{arm.get('reference_arm_id', arm['arm_id'])} ({mode.upper() if mode != 'none' else mode})"
        im_view = any(r['arm_id'] in INVERSE_MEMBERSHIP_ARM_IDS for r in rows)
        if elastic and im_view:
            label = f"{arm.get('reference_arm_id', arm['arm_id'])} ({'IM' if arm.get('sampling_policy') else 'uniform'})"
        if not elastic:
            standalone_labeled = True
        ax.plot([r['non_embedding_parameters'] for r in values], [r[metric] for r in values],
                label=label, color=colors.get(arm.get('reference_arm_id', arm['arm_id']), '#8B4513'),
                linestyle=('--' if arm.get('sampling_policy') else {'none': '-', 'gmc': '--', 'lmc': ':'}[mode]) if elastic else 'None', marker='o' if elastic else '^',
                markersize=5 if elastic else 9, fillstyle='none' if elastic else 'full')
    ax.set(xlabel='Active non-embedding parameters',
           ylabel='Perplexity' if metric == 'perplexity' else 'Loss')
    ax.set_xticks([w['non_embedding_parameters'] for w in WIDTHS])
    ax.ticklabel_format(axis='x', style='plain'); ax.grid(alpha=.2)
    ax.legend(ncol=3)
    title = ('PARTIAL diagnostic — ' if partial else '') + 'TinyStories-Instruct · seed 42 · ordinary validation\nExact terminal checkpoint endpoints'
    figure.suptitle(title)
    figure.text(.5, .025, 'Standalone: 1 epoch / 713,785,344 tokens per run; elastic: 4 epochs / 2,855,141,376 tokens per run.', ha='center', fontsize=9)
    figure.tight_layout(rect=(0, .09, 1, .9))
    return figure


def _comparison_interpretations(rows):
    explanations = (
        ('S1/S2', 'S1', 'S2', 'Within slicing: shared versus per-width AdamW histories.'),
        ('C1/C2', 'C1', 'C2', 'Within concat: shared versus per-width AdamW histories.'),
        ('S1/C1', 'S1', 'C1', 'Representation changes inactive-tail momentum/decay and parameter counters.'),
        ('S2/C2', 'S2', 'C2', 'Representation changes inactive-tail behavior, counters and lazy history allocation.'),
        ('C1/C3', 'C1', 'C3', 'Global cap 1 versus independent owner caps 1; combined gradient bounds sqrt(2) through sqrt(5). Bounds do not describe AdamW update norms; block histories are shared across activating widths.'),
    )
    im_only = bool(rows) and all(r['arm_id'] in INVERSE_MEMBERSHIP_ARM_IDS for r in rows)
    by_key = {(r['arm_id'], r['width']): r for r in rows}
    def comparison(left, right, width):
        if im_only:
            left, right = left + "-IM", right + "-IM"
        a, b = by_key.get((left, width)), by_key.get((right, width))
        if a is None or b is None:
            return {'left': left, 'right': right, 'width': width, 'status': 'missing endpoints'}
        return {'left': left, 'right': right, 'width': width,
                'loss_right_minus_left': b['loss'] - a['loss'], 'perplexity_right_minus_left': b['perplexity'] - a['perplexity'],
                'left_resources': a['resources'], 'right_resources': b['resources'],
                'left_optimizer_storage': a['optimizer_storage'], 'right_optimizer_storage': b['optimizer_storage']}
    result = [{'comparison': name, 'interpretation': explanation,
               'measurements': [comparison(left, right, w) for w in WIDTH_LABELS]} for name, left, right, explanation in explanations]
    if not im_only:
        result.append({'comparison': 'elastic/standalone', 'interpretation': 'Each elastic width versus its matching fresh dense standalone at the same active count; per-run token budgets differ.',
                       'measurements': [comparison(a['arm_id'], 'ST-' + w, w) for a in ELASTIC_ARMS for w in WIDTH_LABELS]})
    return result


def report_campaign(*, manifest, output_dir, allow_partial=False):
    """Revalidate the freeze and publish tables/figures from saved artifacts only."""
    import csv
    from pathlib import Path
    from src.utils.config import ConfigError

    try:
        frozen_source = _source_record(manifest)
        frozen = _read_json(manifest)
        _check_content_hash(frozen, 'content_hash', 'frozen')
        _require_equal(frozen.get('schema_version'), FROZEN_MANIFEST_SCHEMA_VERSION, 'frozen schema')
        if frozen['status'] == 'partial' and not allow_partial:
            raise ConfigError('Partial manifest requires --allow-partial again for report')
        sources = [frozen_source, frozen['preflight_source']] + [s for r in frozen['runs'] for s in r['sources']]
        _check_sources(sources)
        preflight = _read_preflight_manifest(frozen['preflight_source']['path'])
        _require_equal(frozen['preflight_manifest_hash'], preflight['manifest_hash'], 'frozen preflight hash')
        _require_equal(frozen['campaign_id'], preflight['campaign_id'], 'frozen campaign_id')
        by_arm = {r['arm_id']: r for r in preflight['runs']}
        seen = set()
        for saved in frozen['runs']:
            arm = saved['arm_id']
            if arm not in by_arm or arm in seen:
                raise ConfigError('Duplicate or unexpected frozen arm')
            seen.add(arm)
            for path, present in saved['optional_sources'].items():
                _require_equal(Path(path).exists(), present, f'Frozen source presence: {path}')
            actual = _inspect_terminal_run(saved['run_dir'], by_arm[arm], preflight['expected_traces'][arm], allow_partial=allow_partial)
            _require_equal(actual, saved, f'{arm}.frozen terminal sources')
        missing = _missing_endpoints(frozen['runs'], campaign_arms(preflight['schema_version']))
        _require_equal(frozen['missing_endpoints'], missing, 'frozen missing endpoints')
        _require_equal(frozen['status'], 'partial' if missing else 'complete', 'frozen status')
        if missing and not allow_partial:
            raise ConfigError(f'missing endpoints: {missing}')
        rows = _endpoint_table(frozen, preflight)
        if not rows:
            raise ConfigError('missing all endpoints')
        def publish(stage, output):
            report = {'schema_version': COMPARISON_REPORT_SCHEMA_VERSION, 'campaign_id': frozen['campaign_id'],
                      'status': frozen['status'], 'missing_endpoints': missing, 'missing_arms': frozen['missing_arms'],
                      'endpoints': rows, 'figures': [], 'individual_reports': [],
                      'comparisons': _comparison_interpretations(rows), 'holdout_evaluated': False,
                      'interpretation_scope': 'Descriptive seed-42 observations only; no multi-seed robustness or uncertainty claims. Initialization matches the normal constructor seed, not exact cross-model tensors. One elastic run matches aggregate tokens of four standalones, not per-run compute, time or realized width coverage. Resource claims retain measurement-completeness flags.'}
            for saved in frozen['runs']:
                relative = Path('individual') / saved['arm_id']
                diagnostic = report_run_artifacts(saved['run_dir'], stage / relative, partial=bool(missing))
                # The reader uses the latest ledger; propagate the same costs to tables.
                for row in rows:
                    if row['arm_id'] == saved['arm_id']:
                        row['resources'] = diagnostic['resources']
                diagnostic['status'] = frozen['status']
                diagnostic['missing_endpoints'] = missing
                diagnostic['figures'] = [str(output / relative / Path(p).name) for p in diagnostic['figures']]
                write_json_artifact(stage / relative / 'run_diagnostics.json', diagnostic)
                report['individual_reports'].append(str(output / relative / 'run_diagnostics.json'))
            report['comparisons'] = _comparison_interpretations(rows)
            write_json_artifact(stage / 'optimizer_ownership_endpoints.json', {'schema_version': 1, 'status': frozen['status'], 'missing_endpoints': missing, 'endpoints': rows})
            with (stage / 'optimizer_ownership_endpoints.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
                writer.writerows({k: endpoint_csv_value(v) for k, v in r.items()} for r in rows)
            for metric in ('perplexity', 'loss'):
                figure = endpoint_figure(rows, metric=metric, partial=bool(missing))
                for suffix in ('png', 'pdf'):
                    name = f'optimizer_ownership_{metric}_vs_non_embedding_parameters.{suffix}'
                    figure.savefig(stage / name); report['figures'].append(str(output / name))
            _check_sources(sources)
            write_json_artifact(stage / 'comparison_report.json', report)
            return report
        return _publish_directory(output_dir, publish)
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ConfigError(f'Malformed frozen comparison evidence: {error}') from error


def _validated_comparison_sources(manifest_path, *, correction=False, schema_version=None):
    """Read both campaigns under their own immutable contracts before combining."""
    from pathlib import Path
    frozen = _read_json(manifest_path)
    _check_content_hash(frozen, 'content_hash', 'frozen')
    _require_equal(frozen['status'], 'complete', 'comparison requires complete freeze')
    _require_equal(frozen.get('holdout_evaluated'), False, 'holdout_evaluated')
    sources = [_source_record(manifest_path), frozen['preflight_source']]
    sources.extend(s for run in frozen['runs'] for s in run['sources'])
    _check_sources(sources)
    preflight = _read_preflight_manifest(frozen['preflight_source']['path'])
    _require_equal(preflight['schema_version'], schema_version if schema_version is not None else (2 if correction else 1), 'comparison campaign schema')
    _require_equal(frozen['preflight_manifest_hash'], preflight['manifest_hash'], 'frozen preflight hash')
    _require_equal(frozen['campaign_id'], preflight['campaign_id'], 'frozen campaign identity')
    definitions = {r['arm_id']: r for r in preflight['runs']}
    _require_equal([r['arm_id'] for r in frozen['runs']], list(definitions), 'complete frozen arms')
    for saved in frozen['runs']:
        for path, present in saved['optional_sources'].items():
            _require_equal(Path(path).exists(), present, f'Frozen source presence: {path}')
        arm = saved['arm_id']
        actual = _inspect_terminal_run(saved['run_dir'], definitions[arm], preflight['expected_traces'][arm], allow_partial=False)
        _require_equal(actual, saved, f'{arm}.frozen terminal sources')
    _require_equal(_missing_endpoints(frozen['runs'], campaign_arms(preflight['schema_version'])), [], 'complete endpoints')
    return frozen, preflight, sources, _endpoint_table(frozen, preflight)


def loss_progress_figure(runs, *, include_slicing=False):
    """Stream only ordinary-validation rows; retain every recorded update."""
    import csv
    import math
    from pathlib import Path
    from matplotlib.figure import Figure
    from src.utils.config import ConfigError

    figure = Figure(figsize=(14, 10))
    axes = figure.subplots(2, 2, sharex=True)
    colors = {'S1': '#0072B2', 'S2': '#E69F00', 'C1': '#009E73', 'C2': '#CC79A7', 'C3': '#D55E00'}
    data = {}
    for run in runs:
        arm = run['arm_id']
        if arm.startswith('ST-') or (arm.startswith('S') and not include_slicing):
            continue
        curves = {w: ([], []) for w in WIDTH_LABELS}
        with (Path(run['run_dir']) / 'metrics.csv').open() as handle:
            for row in csv.DictReader(handle):
                if row['split'] != 'validation':
                    continue
                width = row['granularity']
                step, loss = int(row['step']), float(row['loss'])
                if width not in curves or not math.isfinite(loss) or not 0 <= step <= 4 * UPDATES_PER_EPOCH:
                    raise ConfigError(f'{arm}: invalid ordinary-validation progress')
                x, y = curves[width]
                if x and step < x[-1]:
                    raise ConfigError(f'{arm}: validation progress is not ordered')
                x.append(step); y.append(loss)
        data[arm] = curves
        mode = arm.split('-')[1] if '-' in arm else 'none'
        for ax, width in zip(axes.flat, WIDTH_LABELS):
            x, y = curves[width]
            if not x:
                raise ConfigError(f'{arm}: missing validation progress for {width}')
            ax.plot(x, y, color=colors[arm[:2]], linestyle={'none': '-', 'GMC': '--', 'LMC': ':', 'IM': '--'}[mode],
                    linewidth=1.1, label=f'{arm[:2]} ({"uniform" if include_slicing and mode == "none" else mode})')
            ax.set(title=width, xlabel='Committed optimizer updates', ylabel='Loss')
            ax.grid(alpha=.2)
            ax.set_xlim(0, 4 * UPDATES_PER_EPOCH)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=3)
    figure.suptitle('TinyStories-Instruct · seed 42 · ordinary-validation loss progress')
    figure.tight_layout(rect=(0, .1, 1, .95))
    return figure, {arm: {w: {'observations': len(x), 'first_update': x[0], 'last_update': x[-1]}
                         for w, (x, _) in curves.items()} for arm, curves in data.items()}


def report_correction_comparison(*, manifest, reference_manifest, output_dir):
    """40 terminal endpoints plus full progress, without model/holdout evaluation."""
    import csv
    corrected, current, sources, rows = _validated_comparison_sources(manifest, correction=True)
    original, reference, old_sources, baseline = _validated_comparison_sources(reference_manifest, correction=False)
    sources += old_sources
    rows += [r for r in baseline if r['arm_id'].startswith(('C', 'ST-'))]
    _require_equal(len(rows), 40, 'comparison endpoint count')
    for arm in CORRECTION_ARMS:
        _require_equal(current['expected_traces'][arm['arm_id']], reference['expected_traces'][arm['reference_arm_id']], 'paired action/batch traces')
    comparisons = []
    by_key = {(r['arm_id'], r['width']): r for r in rows}
    for arm in CORRECTION_ARMS:
        for width in WIDTH_LABELS:
            row, baseline_row = by_key[arm['arm_id'], width], by_key[arm['reference_arm_id'], width]
            comparisons.append({'arm_id': arm['arm_id'], 'reference_arm_id': arm['reference_arm_id'], 'width': width,
                                'loss_delta': row['loss'] - baseline_row['loss'],
                                'perplexity_delta': row['perplexity'] - baseline_row['perplexity']})
    all_runs = corrected['runs'] + [r for r in original['runs'] if r['arm_id'].startswith(('C', 'ST-'))]
    def publish(stage, output):
        report = {'schema_version': 1, 'status': 'complete', 'endpoints': rows,
                  'comparisons': comparisons, 'sources': sources, 'holdout_evaluated': False,
                  'figures': [], 'paired_traces_verified': True,
                  'interpretation_scope': 'Paired seed-42 observations only. LMC combines GMC before clipping with whole AdamW change scaling, including decay. No direct moment/counter scaling and no extra data exposure. C3 independently clips each active owner at 1; C1/C2 globally clip at 1. C2 has width-specific histories over shared weights. Runtime/resource totals are measured per attempt and retain replay/completeness caveats; no across-seed significance.'}
        write_json_artifact(stage / 'optimizer_ownership_endpoints.json', {'status': 'complete', 'endpoints': rows})
        with (stage / 'optimizer_ownership_endpoints.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader()
            writer.writerows({k: endpoint_csv_value(v) for k, v in row.items()} for row in rows)
        for metric in ('loss', 'perplexity'):
            figure = endpoint_figure(rows, metric=metric, partial=False)
            for suffix in ('png', 'pdf'):
                name = f'optimizer_ownership_{metric}_vs_non_embedding_parameters.{suffix}'
                figure.savefig(stage / name); report['figures'].append(str(output / name))
        figure, coverage = loss_progress_figure(all_runs)
        report['progress_coverage'] = coverage
        for suffix in ('png', 'pdf'):
            name = f'optimizer_ownership_validation_loss_progress.{suffix}'
            figure.savefig(stage / name); report['figures'].append(str(output / name))
        report['clipping'] = {r['arm_id']: r['observations']['clipping_by_width'] for r in all_runs}
        _check_sources(sources)
        write_json_artifact(stage / 'comparison_report.json', report)
        return report
    return _publish_directory(output_dir, publish)


def report_inverse_membership_comparison(*, manifest, reference_manifest, output_dir):
    """Validate 20 fixed-IM + 24 historical endpoints under their own contracts."""
    import csv
    current_frozen, current, sources, rows = _validated_comparison_sources(manifest, schema_version=3)
    old_frozen, reference, old_sources, baseline = _validated_comparison_sources(reference_manifest, schema_version=1)
    sources += old_sources
    for row in baseline:
        row['historical_reference'] = True
    rows += baseline
    _require_equal(len(rows), 44, 'IM comparison endpoint count')
    comparisons, exposures = [], []
    by_key = {(r['arm_id'], r['width']): r for r in rows}
    _require_equal(len(by_key), 44, 'IM unique endpoint count')
    for arm in INVERSE_MEMBERSHIP_ARMS:
        name, ref = arm['arm_id'], arm['reference_arm_id']
        _require_equal(current['expected_traces'][name]['epochs'], reference['expected_traces'][ref]['epochs'], 'paired epoch/batch traces')
        # Action streams intentionally differ between policies; each freeze has
        # already validated equality against its own expected sequence.
        for width in WIDTH_LABELS:
            a, b = by_key[name, width], by_key[ref, width]
            for field in ('evaluation_role', 'validation_manifest_hash', 'evaluation_protocol_hash',
                          'validation_loss_aggregation', 'evaluation_target_tokens', 'non_embedding_parameters',
                          'assigned_updates', 'assigned_tokens', 'actual_updates', 'actual_tokens'):
                _require_equal(a[field], b[field], f'{name}/{ref}.{field}')
            comparisons.append({'arm_id': name, 'reference_arm_id': ref, 'width': width,
                                'loss_delta': a['loss']-b['loss'], 'perplexity_delta': a['perplexity']-b['perplexity']})
        exposures.append({'arm_id': name, 'reference_arm_id': ref,
            'fixed_im_width_selections': a['width_selection_counts'], 'uniform_width_selections': b['width_selection_counts'],
            'fixed_im_block_activations': a['quarter_activation_counts'], 'uniform_block_activations': b['quarter_activation_counts'],
            'fixed_im_expected_block_probabilities': [1., .88, .72, .48],
            'uniform_expected_block_probabilities': [1., .75, .5, .25],
            'fixed_im_resources': a['resources'], 'uniform_resources': b['resources'],
            'fixed_im_optimizer_storage': a['optimizer_storage'], 'uniform_optimizer_storage': b['optimizer_storage'],
            'fixed_im_checkpoint_bytes': a['checkpoint_bytes'], 'uniform_checkpoint_bytes': b['checkpoint_bytes']})
    all_runs = current_frozen['runs'] + old_frozen['runs']
    def publish(stage, output):
        report = {'schema_version': 1, 'status': 'complete', 'endpoints': rows, 'comparisons': comparisons,
                  'exposure_comparisons': exposures, 'sources': sources, 'figures': [],
                  'paired_epoch_traces_verified': True, 'holdout_evaluated': False,
                  'interpretation_scope': 'Descriptive seed-42 observations. IM minus uniform isolates the sampling change within each arm. Between arms, representation changes inactive-tail momentum/decay, histories and clipping differ. C3 uses independent owner caps, C1/C2 global clipping. Equal training tokens do not imply equal FLOPs or runtime: IM selects wider models more often. Gradient-support exposure is not actual optimizer mutation count in slicing. Resource totals retain continuation/replay and completeness caveats; no across-seed significance.'}
        write_json_artifact(stage/'optimizer_ownership_endpoints.json', {'status':'complete','endpoints':rows})
        for name, values in (('optimizer_ownership_endpoints', rows), ('inverse_membership_deltas', comparisons)):
            with (stage/(name+'.csv')).open('w', newline='') as handle:
                writer=csv.DictWriter(handle, fieldnames=list(values[0])); writer.writeheader()
                writer.writerows({k:endpoint_csv_value(v) for k,v in r.items()} for r in values)
        write_json_artifact(stage/'inverse_membership_deltas.json', comparisons)
        for metric in ('loss','perplexity'):
            figure=endpoint_figure(rows,metric=metric,partial=False)
            for suffix in ('png','pdf'):
                name=f'optimizer_ownership_{metric}_vs_non_embedding_parameters.{suffix}'
                figure.savefig(stage/name); report['figures'].append(str(output/name))
        figure, report['progress_coverage']=loss_progress_figure(all_runs,include_slicing=True)
        for suffix in ('png','pdf'):
            name=f'optimizer_ownership_validation_loss_progress.{suffix}'
            figure.savefig(stage/name); report['figures'].append(str(output/name))
        report['clipping']={r['arm_id']:r['observations']['clipping_by_width'] for r in all_runs}
        _check_sources(sources)
        write_json_artifact(stage/'comparison_report.json',report)
        return report
    return _publish_directory(output_dir,publish)
