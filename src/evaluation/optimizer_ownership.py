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


MATFORMER_CAMPAIGN_SCHEMA_VERSION = 4
MATFORMER_CAMPAIGN_ID = "tinystories-optimizer-ownership-matformer-widths-v1"
MATFORMER_WIDTHS: tuple[Width, ...] = tuple(
    {
        "label": label,
        "source_fraction": fraction,
        "active_ffn_dimension": dimension,
        "non_embedding_parameters": count,
        # Legacy field names denote incremental blocks, not equal-size quarters.
        "active_quarters": QUARTER_IDS[:index + 1],
    }
    for index, (label, fraction, dimension, count) in enumerate((
        ("g125", 0.125, 32, 90_688),
        ("g250", 0.25, 64, 115_264),
        ("g500", 0.5, 128, 164_416),
        ("g1000", 1.0, 256, 262_720),
    ))
)
MATFORMER_WIDTH_LABELS = tuple(width["label"] for width in MATFORMER_WIDTHS)
MATFORMER_STANDALONE_ARMS: tuple[ArmDefinition, ...] = tuple(
    {
        **arm,
        "arm_id": f"ST-{width['label']}",
        "physical_ffn_dimension": width["active_ffn_dimension"],
        "source_width": width["label"],
        "endpoint_widths": (width["label"],),
    }
    for arm, width in zip(STANDALONE_ARMS, MATFORMER_WIDTHS)
)
MATFORMER_ELASTIC_ARMS: tuple[ArmDefinition, ...] = tuple(
    {**arm, "endpoint_widths": MATFORMER_WIDTH_LABELS} for arm in ELASTIC_ARMS
)
MATFORMER_ARMS = MATFORMER_STANDALONE_ARMS + MATFORMER_ELASTIC_ARMS
MATFORMER_BLOCK_BOUNDARIES = tuple(
    {
        "id": block_id,
        "start": start,
        "end": end,
        "dimension": end - start,
        "supported_widths": MATFORMER_WIDTH_LABELS[index:],
    }
    for index, (block_id, start, end) in enumerate((
        ("A", 0, 32), ("B", 32, 64), ("C", 64, 128), ("D", 128, 256),
    ))
)


WARMUP_CAMPAIGN_SCHEMA_VERSION = 5
WARMUP_CAMPAIGN_ID = "tinystories-optimizer-ownership-s1-warmup-v1"
WARMUP_ARMS = tuple(
    {**ELASTIC_ARMS[0], "arm_id": f"S1-{grid}-w256", "grid_id": grid,
     "endpoint_widths": tuple(w["label"] for w in widths)}
    for grid, widths in (("linear", WIDTHS), ("geometric", MATFORMER_WIDTHS))
)
WARMUP_REFERENCES = {
    "linear": {
        "campaign_id": "tinystories-optimizer-ownership-v1", "schema_version": 1,
        "s1_config_sha256": "ef16f8a6cb883abe6782b9201f32b6d137b324f4d0cd5ac112f66b6fb57ac3fa",
        "selected_arms": ["ST-g250", "ST-g500", "ST-g750", "ST-g1000", "S1"],
    },
    "geometric": {
        "campaign_id": MATFORMER_CAMPAIGN_ID, "schema_version": 4,
        "s1_config_sha256": "781254c7bbd8a7fdc48ba4c09bdc940dd44e132c73d1fe9d8b9209796d94a6c8",
        "selected_arms": ["ST-g125", "ST-g250", "ST-g500", "ST-g1000", "S1"],
    },
}


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


def campaign_arms(schema_version=CAMPAIGN_SCHEMA_VERSION):
    from src.utils.config import ConfigError
    if type(schema_version) is int and schema_version == 1:
        return ARMS
    if type(schema_version) is int and schema_version == 2:
        return CORRECTION_ARMS
    if type(schema_version) is int and schema_version == 3:
        return INVERSE_MEMBERSHIP_ARMS
    if type(schema_version) is int and schema_version == MATFORMER_CAMPAIGN_SCHEMA_VERSION:
        return MATFORMER_ARMS
    if type(schema_version) is int and schema_version == 5:
        return WARMUP_ARMS
    raise ConfigError(f"Unsupported campaign schema_version: {schema_version}")


def campaign_arm(schema_version, arm_id):
    """Short labels are meaningful only within their campaign schema."""
    from src.utils.config import ConfigError

    for arm in campaign_arms(schema_version):
        if arm["arm_id"] == arm_id:
            return arm
    raise ConfigError(f"Unknown arm_id {arm_id!r} for campaign schema_version {schema_version}")


def campaign_widths(schema_version=CAMPAIGN_SCHEMA_VERSION, arm_id=None):
    campaign_arms(schema_version)  # Reject unknown versions before choosing a grid.
    if schema_version == 5:
        return MATFORMER_WIDTHS if campaign_arm(5, arm_id)["grid_id"] == "geometric" else WIDTHS
    return MATFORMER_WIDTHS if schema_version == MATFORMER_CAMPAIGN_SCHEMA_VERSION else WIDTHS


def campaign_common(schema_version=CAMPAIGN_SCHEMA_VERSION, arm_id=None):
    """Return detached controls; historical inputs receive no new defaults."""
    import copy

    widths = campaign_widths(schema_version, arm_id)
    common = copy.deepcopy(PINNED_COMMON)
    if schema_version == 5:
        common["training"]["warmup_steps"] = 256
    if schema_version in (4, 5):
        common["model"]["granularities"] = [width["label"] for width in widths]
        common["model"]["granularity_prefixes"] = {
            width["label"]: width["source_fraction"] for width in widths
        }
    return common


def campaign_topology(schema_version=CAMPAIGN_SCHEMA_VERSION, arm_id=None):
    """New-only identity fields shared by campaign and run contracts."""
    import copy

    campaign_arms(schema_version)
    if schema_version == 5:
        arm = campaign_arm(5, arm_id)
        widths = campaign_widths(5, arm_id)
        ends = [0] + [w["active_ffn_dimension"] for w in widths]
        return copy.deepcopy({
            "campaign_schema_version": 5, "grid_id": arm["grid_id"],
            "width_grid": list(widths),
            "block_boundaries": [dict(id=q, start=ends[i], end=ends[i+1],
                dimension=ends[i+1]-ends[i], supported_widths=arm["endpoint_widths"][i:])
                for i, q in enumerate(QUARTER_IDS)],
        })
    if schema_version != MATFORMER_CAMPAIGN_SCHEMA_VERSION:
        return {}
    return copy.deepcopy({
        "campaign_schema_version": MATFORMER_CAMPAIGN_SCHEMA_VERSION,
        "width_grid": list(MATFORMER_WIDTHS),
        "block_boundaries": list(MATFORMER_BLOCK_BOUNDARIES),
    })


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
    campaign_schema_version: NotRequired[int]
    width_grid: NotRequired[list[Width]]
    block_boundaries: NotRequired[list[dict[str, Any]]]


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
    snapshot_provenance = root / 'source-provenance.json'
    if snapshot_provenance.exists():
        import json
        import hashlib
        from src.utils.config import ConfigError
        provenance = json.loads(snapshot_provenance.read_text())
        for relative, expected in provenance['source_files_sha256'].items():
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
                raise ConfigError(f'Snapshot provenance changed: {relative}')
        return provenance
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


def build_run_scientific_contract(
    config, initialization, *, campaign_schema_version=CAMPAIGN_SCHEMA_VERSION
):
    """Bind resolved controls to a schema-qualified arm using the existing serializer.

    Resolution and topology eligibility remain the caller's responsibility. This
    constructor keeps the historical signature inputs exactly as they were.
    """
    import copy
    from src.utils.reproducibility import build_optimizer_ownership_signature

    arm = campaign_arm(campaign_schema_version, config["run"]["arm_id"])
    if campaign_schema_version == MATFORMER_CAMPAIGN_SCHEMA_VERSION:
        _require_equal(config["run"]["campaign_id"], MATFORMER_CAMPAIGN_ID, "campaign_id")
    contract = {
        "schema_version": SCIENTIFIC_CONTRACT_SCHEMA_VERSION,
        "campaign_id": config["run"]["campaign_id"],
        "run_id": config["run"]["run_id"],
        "arm_id": arm["arm_id"],
        "representation": arm["representation"],
        "state_scope": arm["state_scope"],
        "clipping": copy.deepcopy(config["training"]["gradient_clipping"]),
        "initialization": initialization,
        "model": {k: v for k, v in config["model"].items() if k != "tokenizer_dir"},
        "optimizer": copy.deepcopy(config["training"]),
        "sampling": _sampling_contract(config, arm),
        "data": {k: v for k, v in config["dataset"].items() if k != "prepared_corpus_dir"},
        "budget": {
            k: arm[k]
            for k in ("assigned_epochs", "assigned_updates", "assigned_tokens")
        },
        "evaluation": copy.deepcopy(config["evaluation"]),
        "count_convention": PARAMETER_COUNT_CONVENTION,
        **campaign_topology(campaign_schema_version, arm["arm_id"]),
    }
    if campaign_schema_version == 5:
        contract["intervention"] = warmup_intervention(arm["arm_id"])
    if "correction_mode" in arm:
        contract["correction"] = membership_correction_contract(arm["correction_mode"])
    return build_optimizer_ownership_signature(contract)


def expand_campaign(recipe, *, prepared_corpus_dir, tokenizer_dir, run_output_root):
    """Resolve the exact declared fresh trainer configs, without run-directory IO."""
    import copy
    import re
    import tempfile
    from pathlib import Path
    import yaml
    from src.utils.config import ConfigError, resolve_run_config
    from src.utils.reproducibility import seed_for

    if not isinstance(recipe, dict):
        raise ConfigError("campaign must be a mapping")
    _require_equal(
        set(recipe),
        {"schema_version", "campaign_id", "common", "arms", "expected_data"}
        | ({"references"} if recipe.get("schema_version") == 5 else set()),
        "campaign fields",
    )
    arms = campaign_arms(recipe["schema_version"])
    if recipe["schema_version"] == 5:
        _require_equal(recipe["campaign_id"], WARMUP_CAMPAIGN_ID, "campaign_id")
        _require_equal(recipe["references"], WARMUP_REFERENCES, "references")
        _require_equal(recipe["common"], campaign_common(5, WARMUP_ARMS[0]["arm_id"]), "common")
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
            override = copy.deepcopy(recipe["arms"][arm_id])
            if recipe["schema_version"] == 5:
                _require_equal(override.pop("grid_id", None), arm["grid_id"], f"{arm_id}.grid_id")
            raw = _merge(recipe["common"], override)
            # Prefix maps are a complete grid, not an additive override.
            if recipe["schema_version"] == 5 and "granularity_prefixes" in override.get("model", {}):
                raw["model"]["granularity_prefixes"] = override["model"]["granularity_prefixes"]
            _require_equal(raw, _merge(campaign_common(recipe["schema_version"], arm_id), _arm_overrides(arm)), arm_id)
            run_id = f"{campaign_id}-{arm_id}-s{SEED}"
            output = str(Path(run_output_root).expanduser().resolve() / arm_id)
            raw["run"].update(
                run_id=run_id, campaign_id=campaign_id, arm_id=arm_id, output_dir=output
            )
            if recipe["schema_version"] in (4, 5):
                raw["run"]["campaign_schema_version"] = recipe["schema_version"]
            if recipe["schema_version"] == 5:
                raw["run"]["grid_id"] = arm["grid_id"]
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
            contract_hash, contract = build_run_scientific_contract(
                resolved, initialization, campaign_schema_version=recipe["schema_version"]
            )
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
    warmup = 64
    if config["run"].get("campaign_schema_version") == 5:
        validate_warmup_controls(config)
        _require_equal(arm, campaign_arm(5, config["run"]["arm_id"]), "budget arm")
        warmup = 256
    for key, expected in {
        "max_steps": arm["assigned_updates"],
        "derived_max_steps": arm["assigned_updates"],
        "token_budget": arm["assigned_tokens"],
        "expected_tokens_per_step": TOKENS_PER_UPDATE,
        "max_steps_cap": None,
        "resolved_warmup_steps": warmup,
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
                for width in campaign_widths(config["run"].get("campaign_schema_version", 1), config["run"].get("arm_id")):
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
                        model, ordered_widths=config["model"]["granularities"],
                        topology=config["training"].get("optimizer_state_topology"),
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

    widths = tuple(config["model"]["granularities"])
    generator = random.Random(seed_for(config, "granularity_selection"))
    digest = hashlib.sha256()
    counts = dict.fromkeys(widths, 0)
    fixed = config["model"].get("granularity_sampling_mode") == "fixed_global"
    probabilities = [config["model"]["global_sampling_distribution"][w] for w in widths] if fixed else [.25] * 4
    for _ in range(config["training"]["max_steps"]):
        width = (generator.choices(widths, weights=probabilities, k=1)[0] if fixed
                 else widths[generator.randrange(len(widths))])
        digest.update((width + "\n").encode("ascii"))
        counts[width] += 1
    return {
        "sha256": digest.hexdigest(),
        "encoding": "ASCII width label plus LF per update",
        "updates": sum(counts.values()),
        "counts": counts,
        "expected_counts": {w: config["training"]["max_steps"] * p for w, p in zip(widths, probabilities)},
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
        if not run["source_width"] and run["resolved_config"]["run"].get("campaign_schema_version") != 5:
            _require_equal(trace, elastic, f"{run['arm_id']}.elastic_trace")
    return result


def preflight_campaign(
    *, campaign_path, prepared_corpus_dir, tokenizer_dir, output_dir, run_output_root,
    linear_reference_root=None, geometric_reference_root=None
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
    recipe = yaml.safe_load(Path(campaign_path).read_text())
    warmup = recipe.get("schema_version") == 5
    if warmup and (linear_reference_root is None or geometric_reference_root is None):
        raise ConfigError("schema 5 requires --linear-reference-root and --geometric-reference-root")
    if output.is_symlink():
        raise ConfigError(f"occupied preflight output: {output}")
    if output.exists() and warmup:
        return verify_prepared_warmup(output, root, recipe, prepared_corpus_dir, tokenizer_dir,
                                      linear_reference_root, geometric_reference_root)
    if output.exists() or output.is_symlink():
        raise ConfigError(f"occupied preflight output: {output}")
    if warmup:
        if output.parent != root.parent or output.name != 'campaign' or root.name != 'runs':
            raise ConfigError('schema 5 requires sibling ROOT/campaign and ROOT/runs')
        if output.parent.exists() and any(output.parent.iterdir()):
            raise ConfigError(f'occupied warmup campaign root: {output.parent}')
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
    extra = {}
    references = None
    if warmup:
        references = inspect_warmup_references(linear_reference_root=linear_reference_root,
                                               geometric_reference_root=geometric_reference_root)
        audits = {}
        for run in runs:
            selected = references['grids'][run['grid_id']]
            audits[run['arm_id']] = audit_warmup_counterpart(
                run['resolved_config'], selected['counterpart']['resolved_config'])
            audits[run['arm_id']]['old_config_source'] = selected['s1_config_source']
            _require_equal(traces[run['arm_id']], selected['expected_traces'], f"{run['arm_id']}.counterpart traces")
        extra = dict(arm_grids={r['arm_id']: campaign_topology(5, r['arm_id']) for r in runs},
                     references=recipe['references'], reference_selection=references,
                     control_audits=audits, recipe_source=_source_record(campaign_path))
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
        **(extra if warmup else campaign_topology(recipe["schema_version"])),
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
        if warmup:
            (stage / 'schedules').mkdir()
            schedules = {}
            for run in runs:
                filename = f"schedules/{run['arm_id']}.csv"
                schedules[run['arm_id']] = export_expected_schedule(run['resolved_config'], stage / filename)
                schedules[run['arm_id']]['path'] = filename
                config_source = _source_record(stage / 'configs' / f"{run['arm_id']}.yaml")
                manifest['control_audits'][run['arm_id']]['new_config_source'] = {
                    **config_source, 'path': str(output / 'configs' / f"{run['arm_id']}.yaml")}
            _require_equal(len({v['array_sha256'] for v in schedules.values()}), 1, 'identical new schedules')
            manifest['expected_schedules'] = schedules
            (stage / 'references').mkdir()
            write_json_artifact(stage / 'references/selection.json', references)
            write_json_artifact(stage / 'control_differences.json', manifest['control_audits'])
            write_json_artifact(stage / 'expected_traces.json', traces)
            write_json_artifact(stage / 'expected_schedules.json', schedules)
            manifest['artifact_sha256'] = {str(p.relative_to(stage)): _source_record(p)['sha256']
                for p in sorted(stage.rglob('*')) if p.is_file()}
            manifest['manifest_hash'] = stable_hash({k:v for k,v in manifest.items() if k != 'manifest_hash'})
            report['manifest_hash'] = manifest['manifest_hash']
            _check_sources(references['sources'] + [manifest['recipe_source']])
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
    schema = config["run"].get("campaign_schema_version")
    if schema is not None or "campaign_schema_version" in contract:
        if schema not in (4, 5):
            raise ConfigError("Unsupported run.campaign_schema_version")
        campaign_id = WARMUP_CAMPAIGN_ID if schema == 5 else MATFORMER_CAMPAIGN_ID
        _require_equal(config["run"].get("campaign_id"), campaign_id, "campaign_id")
        for key, expected in campaign_topology(schema, config["run"].get("arm_id")).items():
            _require_equal(stable_hash(contract.get(key)), stable_hash(expected), key)
        arm = campaign_arm(schema, config["run"].get("arm_id"))
        _require_equal(contract["initialization"].get("method"), "fresh_normal_constructor", "initialization.method")
        _require_equal(config["run"].get("run_id"), f"{campaign_id}-{arm['arm_id']}-s{SEED}", "run_id")
    else:
        arm = next((a for a in (*ARMS, *CORRECTION_ARMS, *INVERSE_MEMBERSHIP_ARMS) if a["arm_id"] == config["run"].get("arm_id")), None)
    if arm is None:
        raise ConfigError("optimizer_ownership_contract: unknown arm_id")
    if schema == 5:
        _require_equal(contract.get("intervention"), warmup_intervention(arm["arm_id"]), "intervention")
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
    widths = tuple(w['label'] for w in campaign_widths(audit['contract'].get('campaign_schema_version', 1), audit['arm_id']))
    scope = audit['state_scope']
    counts = dict.fromkeys(audit['width_selection_counts'], 0)
    if audit['contract'].get('campaign_schema_version') in (4, 5):
        schema = audit['contract']['campaign_schema_version']
        arm = campaign_arm(schema, audit['contract']['arm_id'])
        for key, expected in campaign_topology(schema, arm['arm_id']).items():
            _require_equal(stable_hash(audit['contract'].get(key)), stable_hash(expected), key)
        _require_equal(audit['contract'].get('representation'), arm['representation'], 'observation representation')
        _require_equal(scope, arm['state_scope'], 'observation state_scope')
        _require_equal(set(counts), set(arm['endpoint_widths']), 'observation physical widths')
    action_digest = hashlib.sha256()
    epoch_digests = {}
    clipping_summary = {}
    semantic_clipping = (audit['contract'].get('representation') == 'concat' and scope != 'per_granularity')
    if semantic_clipping and not audit.get('clipping_path'):
        raise ConfigError('Missing required committed clipping sidecar reference')
    clip_path = root / audit['clipping_path'] if audit.get('clipping_path') else None
    if clip_path is not None and not clip_path.is_file():
        raise ConfigError('Missing committed clipping sidecar')
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
                quarters = {f'O-{q}': sum(counts.get(w, 0) for w in widths[i:]) for i, q in enumerate('ABCD')} if len(counts) == 4 else {}
                calls = {**quarters, 'O-common': ordinal} if scope == 'per_ffn_block' else counts if scope == 'per_granularity' else {'shared': ordinal}
                active = list(OWNER_IDS[:widths.index(width) + 1]) + ['O-common'] if scope == 'per_ffn_block' else [width] if scope == 'per_granularity' else ['shared']
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
                    _accumulate_clipping(clipping_summary, clip, audit['clipping_contract'], widths=widths)
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
    if audit['contract'].get('campaign_schema_version') in (4, 5):
        import csv
        metrics_path = root / 'metrics.csv'
        if not metrics_path.is_file():
            raise ConfigError('Missing campaign scalar metrics')
        committed = 0
        with metrics_path.open() as stream:
            for metric in csv.DictReader(stream):
                if metric.get('split') != 'train' or metric.get('optimizer_step_committed') not in ('True', 'true', '1'):
                    continue
                committed += 1
                if (metric.get('run_id') != audit['run_id']
                        or metric.get('optimizer_ownership_contract_hash') != audit['contract_hash']
                        or metric.get('optimizer_ownership_trace_path') != audit['trace_path']
                        or metric.get('optimizer_ownership_clipping_path', '') != (audit.get('clipping_path') or '')
                        or int(metric['step']) != committed):
                    raise ConfigError('Scalar metrics identity/sidecar reference mismatch')
        if committed != audit['steps']:
            raise ConfigError('Scalar metrics committed count mismatch')
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


def _accumulate_clipping(summary, clip, contract, *, widths=WIDTH_LABELS):
    import math
    from src.utils.config import ConfigError

    width = clip['width']
    if clip['mode'] != contract['mode'] or set(clip['groups']) != set(OWNER_IDS):
        raise ConfigError('Clipping mode/group topology mismatch')
    groups = summary.setdefault(width, {})
    active_owners = (*OWNER_IDS[:widths.index(width) + 1], 'O-common')
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
    widths = [w['label'] for w in campaign_widths(audit['contract'].get('campaign_schema_version', 1)) if w['label'] in audit['width_selection_counts']]
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
        'common': campaign_common(manifest['schema_version'], arm_ids[0]), 'allowed_difference_matrix': [
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
        common = campaign_common(manifest['schema_version'], label)
        if manifest['schema_version'] == 5:
            _require_equal(raw['run'].pop('campaign_schema_version'), 5, f'{label}.schema')
            _require_equal(raw['run'].pop('grid_id'), arm['grid_id'], f'{label}.grid_id')
            _require_equal(stable_hash(manifest['arm_grids'][label]), stable_hash(campaign_topology(5, label)), f'{label}.grid')
            _require_equal(manifest['references'], WARMUP_REFERENCES, 'preflight.references')
        if manifest['schema_version'] == 4:
            _require_equal(raw['run'].pop('campaign_schema_version'), 4, f'{label}.schema')
            for field, value in campaign_topology(4).items():
                _require_equal(stable_hash(manifest.get(field)), stable_hash(value), f'preflight.{field}')
        raw['model']['tokenizer_dir'] = common['model']['tokenizer_dir']
        raw['dataset']['prepared_corpus_dir'] = common['dataset']['prepared_corpus_dir']
        _require_equal(raw, _merge(common, _arm_overrides(arm)), f'{label}.scientific controls')
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
        if not arm['source_width'] and manifest['schema_version'] != 5:
            _require_equal(traces, manifest['expected_traces'][next(a['arm_id'] for a in arms if not a['source_width'])], f'{label}.elastic traces')
    return manifest


def _terminal_endpoints(sidecar, run, *, allow_partial):
    import math
    from src.utils.config import ConfigError

    label = run['arm_id']
    contract = run['optimizer_ownership_contract']
    widths = campaign_widths(contract.get('campaign_schema_version', 1), contract.get('arm_id'))
    labels = [w['label'] for w in widths]
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
        physical = next(w for w in widths if w['label'] == width)
        count = physical['non_embedding_parameters']
        if contract.get('campaign_schema_version') in (4, 5):
            for key, value in {'width_fraction': physical['source_fraction'], 'ffn_dimension': physical['active_ffn_dimension']}.items():
                if key in row:
                    _require_equal(row[key], value, f'{label}.{width}.{key}')
        _require_equal(row.get('non_embedding_parameters'), count, f'{label}.{width}.non_embedding_parameters')
        loss, perplexity = row['loss'], row['perplexity']
        if (isinstance(loss, bool) or isinstance(perplexity, bool) or not math.isfinite(loss)
                or not math.isfinite(perplexity) or perplexity <= 0 or loss > 709
                or not math.isclose(perplexity, math.exp(loss), rel_tol=1e-12)):
            raise ConfigError(f'{label}.{width}.nonfinite or inconsistent loss/perplexity')
    if not allow_partial and seen != set(run['endpoint_widths']):
        raise ConfigError(f'{label}.missing endpoints: {sorted(set(run["endpoint_widths"]) - seen)}')
    return sorted(rows, key=lambda r: labels.index(r['width']))


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
    if run['optimizer_ownership_contract'].get('campaign_schema_version') in (4, 5):
        import math
        from src.training.run import ResourceAttemptLedger
        resources = audit.get('resources')
        if not isinstance(resources, dict) or type(resources.get('measurement_complete')) is not bool:
            raise ConfigError(f'{label}.missing resource completeness disclosure')
        for field in ('elapsed_seconds', 'attempted_steps', 'peak_allocated_bytes', 'peak_reserved_bytes'):
            value = resources.get(field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0 or (field != 'elapsed_seconds' and type(value) is not int)):
                raise ConfigError(f'{label}.invalid resource measurement: {field}')
        ledger_path = root / 'resource_attempts.json'
        if ledger_path.exists():
            measured = ResourceAttemptLedger(root, run_id=run['run_id']).summary()
            for field in ('elapsed_seconds', 'attempted_steps', 'peak_allocated_bytes', 'peak_reserved_bytes', 'measurement_complete'):
                _require_equal(resources.get(field), measured[field], f'{label}.resources.{field}')
        if resources['measurement_complete'] and (
                not ledger_path.exists() or resources.get('elapsed_seconds') is None
                or resources.get('attempted_steps') is None or resources['attempted_steps'] < run['assigned_updates']):
            raise ConfigError(f'{label}.complete resource claim lacks measured attempts')
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
            terminal = _inspect_terminal_run(path, run, manifest['expected_traces'][run['arm_id']], allow_partial=allow_partial)
            if manifest['schema_version'] == 5:
                terminal['execution_evidence'] = inspect_warmup_execution(Path(campaign_manifest).resolve().parent.parent, run, manifest)
                terminal['sources'].extend(terminal['execution_evidence']['sources'])
            runs.append(terminal)
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
            if manifest['schema_version'] == 5:
                rows = warmup_endpoint_rows(frozen, manifest)
                _write_warmup_table(stage, 'endpoints', 'endpoints', rows)
                _check_sources([preflight_source] + [s for r in runs for s in r['sources']])
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
        widths = campaign_widths(preflight['schema_version'], run['arm_id'])
        summary = _read_json(saved['run_dir'] + '/run_summary.json')['optimizer_ownership']
        sidecar = _read_json(saved['run_dir'] + '/terminal_validation_results.json')
        for endpoint in saved['endpoints']:
            width = next(w for w in widths if w['label'] == endpoint['width'])
            row = {**endpoint, 'status': frozen['status'], 'campaign_id': frozen['campaign_id'], 'arm_id': run['arm_id'],
                   'run_id': run['run_id'], 'seed': SEED, 'representation': run['representation'],
                   'state_scope': run['state_scope'], 'clipping': run['clipping'],
                   'sampling_policy': run.get('sampling_policy', 'standalone' if run['source_width'] else 'uniform'),
                   'sampling_contract': run['optimizer_ownership_contract']['sampling'],
                   'historical_reference': False,
                   'checkpoint_bytes': sidecar['checkpoint_bytes'],
                   'correction_mode': run.get('correction_mode', 'none'),
                   'correction': run['optimizer_ownership_contract'].get('correction'),
                   'width_fraction': width['source_fraction'], 'ffn_dimension': width['active_ffn_dimension'],
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
            if preflight['schema_version'] == 4:
                row.update(group='matformer', canonical_arm=run['arm_id'],
                    endpoint_identity=[run['campaign_id'], run['run_id'], width['source_fraction'], width['active_ffn_dimension']],
                    source_records=saved['sources'], clipping_observations=saved['observations'].get('clipping_by_width') or None)
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
    selected_widths = tuple(dict.fromkeys(r['width'] for r in rows))
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
               'measurements': [comparison(left, right, w) for w in selected_widths]} for name, left, right, explanation in explanations]
    if not im_only:
        result.append({'comparison': 'elastic/standalone', 'interpretation': 'Each elastic width versus its matching fresh dense standalone at the same active count; per-run token budgets differ.',
                       'measurements': [comparison(a['arm_id'], 'ST-' + w, w) for a in ELASTIC_ARMS for w in selected_widths]})
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
            stem = 'endpoints' if preflight['schema_version'] == 4 else 'optimizer_ownership_endpoints'
            write_json_artifact(stage / (stem + '.json'), {'schema_version': 1, 'status': frozen['status'], 'missing_endpoints': missing, 'endpoints': rows})
            with (stage / (stem + '.csv')).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
                writer.writerows({k: endpoint_csv_value(v) for k, v in r.items()} for r in rows)
            for metric in ('perplexity', 'loss'):
                figure = (matformer_endpoint_figure(rows, metric=metric, partial=bool(missing)) if preflight['schema_version'] == 4 else endpoint_figure(rows, metric=metric, partial=bool(missing)))
                for suffix in ('png', 'pdf'):
                    name = f'optimizer_ownership_{metric}_vs_non_embedding_parameters.{suffix}'
                    figure.savefig(stage / name); report['figures'].append(str(output / name))
            _check_sources(sources)
            if preflight['schema_version'] == 4:
                report['endpoint_count'] = len(rows)
                report['input_sources'] = sources
                report['output_sha256'] = {str(p.relative_to(stage)): _source_record(p)['sha256'] for p in sorted(stage.rglob('*')) if p.is_file()}
                report['content_hash'] = stable_hash(report)
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
        if preflight['schema_version'] == 5:
            root = Path(frozen['preflight_source']['path']).parent.parent
            actual['execution_evidence'] = inspect_warmup_execution(root, definitions[arm], preflight)
            actual['sources'].extend(actual['execution_evidence']['sources'])
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


def inspect_selected_terminals(campaign_manifest, arm_ids, *, run_root=None):
    """The same strict saved-terminal reader serves the barrier and full freeze."""
    from pathlib import Path
    from src.utils.config import ConfigError
    manifest = _read_preflight_manifest(campaign_manifest)
    definitions = {r['arm_id']: r for r in manifest['runs']}
    if len(set(arm_ids)) != len(arm_ids) or not set(arm_ids) <= set(definitions):
        raise ConfigError('Duplicate or unknown selected terminal arms')
    return [_inspect_terminal_run(
        Path(run_root) / Path(definitions[arm]['output_path']).name if run_root else definitions[arm]['output_path'],
        definitions[arm], manifest['expected_traces'][arm], allow_partial=False) for arm in arm_ids]


def matformer_endpoint_figure(rows, *, metric, partial=False):
    """Keep repeated baseline measurements at their exact physical coordinates."""
    from matplotlib.figure import Figure
    figure = Figure(figsize=(11, 7)); ax = figure.subplots()
    colors = ('#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00')
    for arm, color in zip(('S1', 'S2', 'C1', 'C2', 'C3'), colors):
        values = sorted((r for r in rows if r['arm_id'] == arm and not r['historical_reference']), key=lambda r: r['ffn_dimension'])
        ax.plot([r['non_embedding_parameters'] for r in values], [r[metric] for r in values], label=arm, color=color, marker='o')
    for historical, label, size, fill in ((True, 'Standalone — historical grid', 13, 'none'), (False, 'Standalone — MatFormer grid', 7, 'full')):
        values = [r for r in rows if r['arm_id'].startswith('ST-') and r['historical_reference'] == historical]
        for index, row in enumerate(values):
            ax.plot([row['non_embedding_parameters']], [row[metric]], linestyle='None', marker='o',
                markersize=size, fillstyle=fill, color='#654321', label=label if index == 0 else '_nolegend_', zorder=5)
    ax.set(xlabel='Active non-embedding parameters (embeddings and LM head excluded)', ylabel=metric.capitalize())
    ax.set_xticks(sorted({r['non_embedding_parameters'] for r in rows}))
    ax.ticklabel_format(axis='x', style='plain'); ax.grid(alpha=.2); ax.legend(ncol=2)
    figure.suptitle(('PARTIAL diagnostic — ' if partial else '') + 'TinyStories-Instruct · seed 42 · ordinary validation\nExact terminal checkpoint endpoints')
    figure.text(.5, .025, 'Standalone: 1 epoch / 713,785,344 tokens; elastic: 4 epochs / 2,855,141,376 tokens per run.', ha='center', fontsize=9)
    figure.tight_layout(rect=(0, .09, 1, .9))
    return figure


def report_matformer_widths_comparison(*, manifest, reference_manifest, output_dir):
    """Validate all new runs but only the four selected original standalones."""
    import copy
    import csv
    from pathlib import Path
    frozen, preflight, sources, rows = _validated_comparison_sources(manifest, schema_version=4)
    old = _read_json(reference_manifest)
    _check_content_hash(old, 'content_hash', 'historical frozen')
    _require_equal(old.get('schema_version'), FROZEN_MANIFEST_SCHEMA_VERSION, 'historical frozen schema')
    _require_equal(old.get('status'), 'complete', 'historical status')
    _require_equal(old.get('holdout_evaluated'), False, 'historical holdout')
    selected_sources = [_source_record(reference_manifest), old['preflight_source']]
    _check_sources(selected_sources)
    old_preflight = _read_preflight_manifest(old['preflight_source']['path'])
    _require_equal(old_preflight['schema_version'], 1, 'historical schema')
    _require_equal(old['campaign_id'], old_preflight['campaign_id'], 'historical campaign')
    _require_equal(old['preflight_manifest_hash'], old_preflight['manifest_hash'], 'historical preflight')
    labels = [a['arm_id'] for a in STANDALONE_ARMS]
    _require_equal([r['arm_id'] for r in old['runs']], [a['arm_id'] for a in ARMS], 'historical frozen arms')
    definitions = {r['arm_id']: r for r in old_preflight['runs']}
    selected = [r for r in old['runs'] if r['arm_id'] in labels]
    _require_equal([r['arm_id'] for r in selected], labels, 'historical selected standalones')
    for saved in selected:
        arm = saved['arm_id']
        _check_sources(saved['sources'])
        for path, present in saved['optional_sources'].items():
            _require_equal(Path(path).exists(), present, f'Historical source presence: {path}')
        actual = _inspect_terminal_run(saved['run_dir'], definitions[arm], old_preflight['expected_traces'][arm], allow_partial=False)
        _require_equal(actual, saved, f'{arm}.historical terminal')
        selected_sources.extend(saved['sources'])
    history = _endpoint_table({**old, 'runs': selected}, old_preflight)
    for row in history:
        row.update(historical_reference=True, group='historical', canonical_arm=row['arm_id'],
            endpoint_identity=[row['campaign_id'], row['run_id'], row['width_fraction'], row['ffn_dimension']],
            source_records=next(r['sources'] for r in selected if r['arm_id'] == row['arm_id']), clipping_observations=None)
    _require_equal(len(rows), 24, 'new endpoint count')
    _require_equal(len(history), 4, 'historical endpoint count')
    rows = copy.deepcopy(rows + history)
    _require_equal(len({tuple(r['endpoint_identity']) for r in rows}), 28, 'combined endpoint identities')
    sources += selected_sources
    def publish(stage, output):
        write_json_artifact(stage / 'combined_endpoints.json', {'schema_version': 1, 'status': 'complete', 'endpoints': rows})
        with (stage / 'combined_endpoints.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
            writer.writerows({k: endpoint_csv_value(v) for k, v in r.items()} for r in rows)
        figures = []
        for metric in ('loss', 'perplexity'):
            figure = matformer_endpoint_figure(rows, metric=metric)
            for suffix in ('png', 'pdf'):
                name = f'{metric}_vs_parameters.{suffix}'
                figure.savefig(stage / name); figures.append(str(output / name))
        report = dict(schema_version=1, status='complete', endpoint_count=28, run_count=13,
            input_sources=sources, figures=figures, holdout_evaluated=False,
            interpretation_scope='Descriptive seed-42 comparison primarily against fresh dense baselines. Historical standalones are repeat measurements, not replacements. Representation/history effects differ from independent C3 clipping over changed block sizes. Equal assigned tokens do not imply equal compute or direct width exposure; no across-seed inference.',
            comparisons=_comparison_interpretations(rows[:24]),
            output_sha256={str(p.relative_to(stage)): _source_record(p)['sha256'] for p in sorted(stage.iterdir())})
        report['content_hash'] = stable_hash(report)
        _check_sources(sources)
        write_json_artifact(stage / 'comparison_report.json', report)
        return report
    return _publish_directory(output_dir, publish)


def warmup_intervention(arm_id):
    import copy
    arm = campaign_arm(5, arm_id)
    reference = copy.deepcopy(WARMUP_REFERENCES[arm['grid_id']])
    return dict(kind='s1_lr_warmup', original_warmup_steps=64, warmup_steps=256,
                total_updates=4 * UPDATES_PER_EPOCH,
                reference={**reference, 'arm_id': 'S1',
                           'run_id': reference['campaign_id'] + '-S1-s42'})


def _check_resolved_subset(expected, actual, path):
    """Resolver defaults may add fields; explicit pinned controls cannot change."""
    from src.utils.config import ConfigError
    for key, value in expected.items():
        if key not in actual:
            raise ConfigError(f'{path}.{key}: missing resolved control')
        if isinstance(value, dict):
            _check_resolved_subset(value, actual[key], f'{path}.{key}')
        else:
            _require_equal(actual[key], value, f'{path}.{key}')


def validate_warmup_controls(config):
    """Warmup eligibility follows the complete fixed protocol, never a parameter."""
    run = config['run']
    arm = campaign_arm(5, run.get('arm_id'))
    _require_equal(run.get('campaign_schema_version'), 5, 'run.campaign_schema_version')
    _require_equal(run.get('campaign_id'), WARMUP_CAMPAIGN_ID, 'run.campaign_id')
    _require_equal(run.get('run_id'), f"{WARMUP_CAMPAIGN_ID}-{arm['arm_id']}-s42", 'run.run_id')
    _require_equal(run.get('grid_id'), arm['grid_id'], 'run.grid_id')
    expected = _merge(campaign_common(5, arm['arm_id']), _arm_overrides(arm))
    expected['model'].pop('tokenizer_dir')
    expected['dataset'].pop('prepared_corpus_dir')
    expected['training']['optimizer'].pop('state_scope')
    expected['training']['optimizer'].pop('scheduler_clock')
    _check_resolved_subset(expected, config, 'protocol')
    _require_equal(config['training'].get('resolved_warmup_steps'), 256, 'training.resolved_warmup_steps')
    for field in ('corpus_hash', 'training_order_sha256'):
        _require_equal(config['dataset'].get(field), PINNED_DATA[field], f'dataset.{field}')
    for role in ('optimizer_training', 'ordinary_validation', 'controller', 'final_holdout'):
        _require_equal(config['dataset']['role_manifest_hashes'].get(role), PINNED_DATA[role+'_manifest_hash'], f'dataset.role_manifest_hashes.{role}')
    for field in ('tokenizer_manifest_hash', 'tokenizer_model_sha256'):
        _require_equal(config['model'].get(field), PINNED_DATA[field], f'model.{field}')
    from src.utils.config import _resolve_gradient_interference_defaults, _resolve_sign_dynamics_defaults
    import copy
    defaults = copy.deepcopy(config)
    for name in ('gradient_interference', 'sign_dynamics'):
        defaults['evaluation'].pop(name, None)
    _resolve_gradient_interference_defaults(defaults)
    _resolve_sign_dynamics_defaults(defaults)
    for name in ('gradient_interference', 'sign_dynamics'):
        _require_equal(config['evaluation'].get(name), defaults['evaluation'][name], f'evaluation.{name}')
    for schedule in (config['training']['scheduler'], config['training']['optimizer_state_contract']['scheduler_contract']):
        _require_equal(schedule['resolved_warmup_steps'], 256, 'scheduler.resolved_warmup_steps')
        _require_equal(schedule['kwargs']['warmup_steps'], 256, 'scheduler.kwargs.warmup_steps')
    for key, value in dict(resolved_warmup_steps=256, resolved_learning_rate=.008,
                           optimizer_state_scope='shared', optimizer_scheduler_clock='global_step').items():
        _require_equal(config['training'].get(key), value, f'training.{key}')


def _control_leaves(value, path=''):
    if isinstance(value, dict) and value:
        return {p: v for key, item in value.items()
                for p, v in _control_leaves(item, f'{path}.{key}' if path else key).items()}
    return {path: value}


def audit_warmup_counterpart(config, counterpart):
    """A closed leaf-path audit, retaining full resolved scientific projections."""
    import copy
    from src.utils.config import ConfigError
    validate_materialized_config(config)
    validate_materialized_config(counterpart)
    grid = config['run']['grid_id']
    reference = WARMUP_REFERENCES[grid]
    _require_equal(counterpart['run']['campaign_id'], reference['campaign_id'], 'reference.campaign_id')
    _require_equal(counterpart['run']['arm_id'], 'S1', 'reference.arm_id')
    _require_equal(counterpart['training']['resolved_warmup_steps'], 64, 'reference.warmup')
    # Contract contents are audited through their resolved projections and the
    # materialized validator; provenance is kept separately with both hashes.
    def projection(c):
        result = copy.deepcopy({k: v for k, v in c.items()
                                if k not in ('optimizer_ownership_contract', 'optimizer_ownership_contract_hash')})
        provenance_fields = {'code_revision', 'working_tree_dirty', 'tracked_code_diff_sha256',
                             'source_files_sha256', 'dependency_versions'}
        result['initialization'] = {k:v for k,v in c['optimizer_ownership_contract']['initialization'].items()
                                    if k not in provenance_fields}
        return result
    old, new = projection(counterpart), projection(config)
    allowed = {
        'training.warmup_steps', 'training.resolved_warmup_steps',
        'run.campaign_id', 'run.run_id', 'run.arm_id', 'run.grid_id',
        'run.campaign_schema_version', 'run.output_dir', 'run.output_root',
        'monitoring.name',
        'training.scheduler.kwargs.warmup_steps', 'training.scheduler.resolved_warmup_steps',
        'training.optimizer_state_contract.scheduler_contract.kwargs.warmup_steps',
        'training.optimizer_state_contract.scheduler_contract.resolved_warmup_steps',
        'evaluation.gradient_interference.diagnostic_contract_hash',
        'evaluation.gradient_interference.milestone_reasons.64',
        'evaluation.gradient_interference.milestone_reasons.256',
        'evaluation.gradient_interference.resolved_milestones',
        'evaluation.gradient_interference.resolved_steps',
        'evaluation.sign_dynamics.resolved_snapshot_milestones',
        'evaluation.sign_dynamics.resolved_snapshot_steps',
        'evaluation.sign_dynamics.snapshot_milestone_reasons.64',
        'evaluation.sign_dynamics.snapshot_milestone_reasons.256',
    }
    # Topology is new identity metadata; enumerate every leaf, so changes in the
    # model, optimizer, evaluation or stream controls are never hidden by a prefix.
    topology = _control_leaves(campaign_topology(5, config['run']['arm_id']), 'training.optimizer_state_topology')
    allowed.update(topology)
    # Linear predates sign-dynamics defaults. Permit only the exact disabled
    # default record, already checked above, enumerated leaf by leaf in evidence.
    if 'sign_dynamics' not in counterpart['evaluation']:
        allowed.update(_control_leaves(config['evaluation']['sign_dynamics'], 'evaluation.sign_dynamics'))
    differences = []
    missing = {'absent': True}
    old_leaves, new_leaves = _control_leaves(old), _control_leaves(new)
    for path in sorted(old_leaves.keys() | new_leaves.keys()):
        a, b = old_leaves.get(path, missing), new_leaves.get(path, missing)
        if a != b:
            differences.append(dict(path=path, old=a, new=b, allowed=path in allowed))
    failures = [row['path'] for row in differences if not row['allowed']]
    if failures:
        raise ConfigError('counterpart control differences: ' + ', '.join(failures))
    return dict(status='passed', grid_id=grid, allowed_paths=sorted(allowed),
                differences=differences, failures=[], old_projection=old, new_projection=new,
                old_projection_hash=stable_hash(old), new_projection_hash=stable_hash(new),
                old_contract_hash=counterpart['optimizer_ownership_contract_hash'],
                new_contract_hash=config['optimizer_ownership_contract_hash'])


def export_expected_schedule(config, path):
    """Stream the full inherited cosine schedule; memory stays independent of T."""
    import csv
    import hashlib
    import importlib.metadata
    import inspect
    import math
    import struct
    from pathlib import Path
    from transformers.optimization import _get_cosine_schedule_with_warmup_lr_lambda
    validate_warmup_controls(config)
    horizon, warmup, peak = 4 * UPDATES_PER_EPOCH, 256, .008
    digest = hashlib.sha256()
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['scheduler_position', 'learning_rate', 'applied_update', 'warmup_steps', 'total_updates'])
        writer.writeheader()
        for position in range(horizon + 1):
            lr = peak * position / warmup if position < warmup else peak * .5 * (1 + math.cos(math.pi * (position-warmup)/(horizon-warmup)))
            inherited = peak * _get_cosine_schedule_with_warmup_lr_lambda(position, num_warmup_steps=warmup, num_training_steps=horizon, num_cycles=.5)
            if not math.isclose(lr, inherited, rel_tol=1e-14, abs_tol=1e-18):
                raise ValueError(f'Installed cosine schedule mismatch at {position}')
            digest.update(struct.pack('<d', lr))
            writer.writerow(dict(scheduler_position=position, learning_rate=lr,
                applied_update=position+1 if position < horizon else None,
                warmup_steps=warmup, total_updates=horizon))
    return dict(schema_version=1, run_id=config['run']['run_id'], grid_id=config['run']['grid_id'],
        positions=horizon+1, warmup_steps=warmup, total_updates=horizon, peak_learning_rate=peak,
        array_sha256=digest.hexdigest(), array_encoding='ordered little-endian float64 learning rates',
        indexing='update u applies position u-1; terminal position has no subsequent update',
        kind='expected_schedule_not_measured_execution',
        dependency_versions={name: importlib.metadata.version(name) for name in ('torch', 'transformers')},
        scheduler_source_sha256=hashlib.sha256(inspect.getsource(_get_cosine_schedule_with_warmup_lr_lambda).encode()).hexdigest(),
        source_files_sha256=_provenance().get('source_files_sha256', {}),
        csv_sha256=_source_record(path)['sha256'])


def _historical_job_accounting(job_id):
    """Read-only Slurm accounting; no queue reconciliation or cancellation."""
    import subprocess
    from src.utils.config import ConfigError
    result = subprocess.run(['sacct', '-X', '--noheader', '--parsable2', '--jobs='+str(job_id),
        '--format=JobIDRaw,State,ExitCode'], check=True, capture_output=True, text=True, timeout=45)
    rows = [line.split('|') for line in result.stdout.strip().splitlines()]
    if len(rows) != 1 or rows[0][:3] != [str(job_id), 'COMPLETED', '0:0']:
        raise ConfigError(f'Historical job {job_id}: missing successful scheduler evidence')
    return dict(job_id=str(job_id), state='COMPLETED', exit_code='0:0')


def inspect_historical_device(root, run, manifest):
    """Check evidence available to that generation, including measured allocation.

    Linear predates worker/CUDA-entry records; geometric standalones predate the
    CUDA entry point. Geometric S1 must have the replacement attempt's entry.
    """
    from pathlib import Path
    from src.utils.config import ConfigError
    root = Path(root)
    arm, run_id = run['arm_id'], run['run_id']
    modern = manifest['schema_version'] == 4
    submissions_path = root/'launchers'/('submissions.json' if modern else 'training-submissions.json')
    config_path = root/'runs'/arm/'config.json'
    ledger_path = root/'runs'/arm/'resource_attempts.json'
    paths = [submissions_path, config_path, ledger_path]
    sources = [_source_record(p) for p in paths]
    config, ledger = _read_json(config_path), _read_json(ledger_path)
    _require_equal(config.get('optimizer_ownership_contract_hash'), run['contract_hash'], f'{arm}.saved config contract')
    # Runtime adds observations (distributed rank, role bindings, continuation)
    # to config.json. Every preflight control must still match its saved value.
    import copy
    controls = copy.deepcopy(run['resolved_config'])
    source = config['training'].get('effective_world_size_source')
    if source not in (controls['training']['effective_world_size_source'], 'single_process'):
        raise ConfigError(f'{arm}: incompatible runtime world size source')
    controls['training']['effective_world_size_source'] = source
    _check_resolved_subset(controls, config, f'{arm}.runtime config')
    _require_equal(config['training'].get('resolved_mixed_precision'), 'bf16', f'{arm}.actual precision')
    _require_equal(ledger.get('run_id'), run_id, f'{arm}.resource run')
    jobs = [j for j in _read_json(submissions_path)['jobs'] if j['arm_id'] == arm]
    if not jobs:
        raise ConfigError(f'{arm}: missing historical submission')
    intent = max(jobs, key=lambda j: j.get('attempt_id', 0))
    if intent.get('status') in ('cancelled', 'invalid', 'retryable', 'failed'):
        raise ConfigError(f'{arm}: invalid selected submission')
    job = str(intent['job_id'])
    accounting = _historical_job_accounting(job)
    for key, value in dict(job_id=job, state='COMPLETED', exit_code='0:0').items():
        _require_equal(accounting.get(key), value, f'{arm}.scheduler.{key}')
    if modern:
        attempt = intent['attempt_id']
        worker_path = root/'launchers'/f'worker-{arm}-{attempt}.json'
        sources.append(_source_record(worker_path))
        worker = _read_json(worker_path)
        for key, value in dict(job_id=job, arm_id=arm, attempt_id=attempt, status='completed', returncode=0).items():
            _require_equal(worker.get(key), value, f'{arm}.worker.{key}')
        bindings = intent.get('bindings', {})
        _require_equal(bindings.get('manifest_hash'), manifest['manifest_hash'], f'{arm}.submission manifest')
        _require_equal(bindings.get('config_sha256', {}).get(arm),
                       _source_record(root/'campaign/configs'/f'{arm}.yaml')['sha256'], f'{arm}.submission config')
        measured = ledger['attempts'].get(worker['process_uuid'])
        if not measured:
            raise ConfigError(f'{arm}: missing selected attempt resources')
        _require_equal(measured.get('slurm_job_id'), job, f'{arm}.resource job')
        _require_equal(measured.get('launch_attempt_id'), attempt, f'{arm}.resource attempt')
        if arm == 'S1':
            entry_path = root/'launchers'/f'cuda-entry-{arm}-{attempt}.json'
            sources.append(_source_record(entry_path))
            entry = _read_json(entry_path)
            for key, value in dict(job_id=job, requested_device='cuda:0', required_precision='bf16',
                                  config=str(root/'campaign/configs'/f'{arm}.yaml')).items():
                _require_equal(entry.get(key), value, f'{arm}.CUDA entry.{key}')
    else:
        _require_equal(intent.get('run_id'), run_id, f'{arm}.submission run')
        command = intent.get('command', [])
        if 'sbatch' not in command or '--gres=gpu:1' not in command or str(root/'campaign/configs'/f'{arm}.yaml') not in command:
            raise ConfigError(f'{arm}: missing GPU launcher/config evidence')
        completed = [a for a in ledger['attempts'].values() if a.get('status') == 'completed']
        if len(completed) != 1:
            raise ConfigError(f'{arm}: ambiguous completed resource attempt')
        measured = completed[0]
    _require_equal(measured.get('status'), 'completed', f'{arm}.resource status')
    _require_equal(measured.get('measurement_complete'), True, f'{arm}.resource completeness')
    import math
    for key in ('peak_allocated_bytes', 'peak_reserved_bytes', 'elapsed_seconds'):
        value = measured.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ConfigError(f'{arm}: missing actual CUDA measurement {key}')
    start = (measured.get('source_checkpoint') or {}).get('step', 0)
    if measured.get('attempted_steps', -1) < run['assigned_updates'] - start:
        raise ConfigError(f'{arm}: incomplete measured terminal attempt')
    _check_sources(sources)
    return dict(generation='worker' if modern else 'legacy_launcher', job_id=job,
                accounting=accounting, intent=intent, resources=measured, sources=sources)


def inspect_warmup_references(*, linear_reference_root, geometric_reference_root):
    """Select exactly five per grid from immutable, possibly closed campaigns."""
    from pathlib import Path
    import yaml
    from src.utils.config import ConfigError
    roots = dict(linear=linear_reference_root, geometric=geometric_reference_root)
    grids, sources = {}, []
    for grid, location in roots.items():
        if location is None:
            raise ConfigError(f'schema 5 requires --{grid}-reference-root')
        root = Path(location).expanduser().resolve()
        reference = WARMUP_REFERENCES[grid]
        manifest_path = root/'campaign/campaign_manifest.json'
        manifest_source = _source_record(manifest_path)
        manifest = _read_preflight_manifest(manifest_path)
        for key in ('schema_version', 'campaign_id'):
            _require_equal(manifest[key], reference[key], f'{grid}.reference.{key}')
        selected = reference['selected_arms']
        expected = [a['arm_id'] for a in campaign_arms(reference['schema_version']) if a['source_width']] + ['S1']
        _require_equal(selected, expected, f'{grid}.selected arms')
        sources.append(manifest_source)
        definitions = {r['arm_id']: r for r in manifest['runs']}
        for arm in selected:
            path = root/'campaign/configs'/f'{arm}.yaml'
            source = _source_record(path); sources.append(source)
            if arm == 'S1':
                _require_equal(source['sha256'], reference['s1_config_sha256'], f'{grid}.S1 config hash')
            _require_equal(yaml.safe_load(path.read_text()), definitions[arm]['executable_config'], f'{grid}.{arm}.saved config')
            _require_equal(str(Path(definitions[arm]['output_path']).resolve()), str(root/'runs'/arm), f'{grid}.{arm}.run location')
        # Capture before parsing terminal sources as well as after, so a file
        # replaced during inspection cannot be certified under its new hash.
        for arm in selected:
            run_dir = root/'runs'/arm
            for filename in ('run_summary.json', 'terminal_validation_results.json',
                             'optimizer_ownership_trace.jsonl', 'metrics.csv', 'resource_attempts.json'):
                sources.append(_source_record(run_dir/filename))
        terminals = inspect_selected_terminals(manifest_path, selected, run_root=root/'runs')
        for terminal in terminals:
            arm = terminal['arm_id']
            widths = {w['label']: w for w in campaign_widths(reference['schema_version'])}
            for endpoint in terminal['endpoints']:
                width = widths[endpoint['width']]
                for field, expected_value in [('width_fraction', width['source_fraction']),
                                              ('ffn_dimension', width['active_ffn_dimension'])]:
                    if field in endpoint:
                        _require_equal(endpoint[field], expected_value, f'{grid}.{arm}.{field}')
            device = inspect_historical_device(root, definitions[arm], manifest)
            terminal['execution_evidence'] = device
            sources.extend(terminal['sources']); sources.extend(device['sources'])
        grids[grid] = dict(root=str(root), manifest_source=manifest_source, terminals=terminals,
            counterpart=definitions['S1'], expected_traces=manifest['expected_traces']['S1'],
            s1_config_source=_source_record(root/'campaign/configs/S1.yaml'))
    _require_equal(sum(len(g['terminals']) for g in grids.values()), 10, 'selected terminal count')
    _require_equal(sum(len(t['endpoints']) for g in grids.values() for t in g['terminals']), 16, 'historical endpoint count')
    _check_sources(sources)
    return dict(schema_version=1, status='passed', grids=grids, sources=sources)


def verify_prepared_warmup(output, root, recipe, corpus, tokenizer, linear, geometric):
    """An identical untouched preparation is verifiable, never adopted or rewritten."""
    from pathlib import Path
    import yaml
    from src.utils.config import ConfigError
    manifest = _read_preflight_manifest(output/'campaign_manifest.json')
    _require_equal(manifest['schema_version'], 5, 'prepared schema')
    _require_equal(manifest['run_output_root'], str(root), 'prepared run root')
    _require_equal(yaml.safe_load(Path(manifest['recipe_source']['path']).read_text()), recipe, 'prepared recipe')
    _check_sources([manifest['recipe_source']])
    for grid, location in [('linear', linear), ('geometric', geometric)]:
        _require_equal(manifest['reference_selection']['grids'][grid]['root'], str(Path(location).resolve()), f'{grid}.prepared root')
    for run in manifest['runs']:
        if (root/run['arm_id']).exists() or (root/run['arm_id']).is_symlink():
            raise ConfigError(f"occupied campaign run identity: {root/run['arm_id']}")
        config = run['resolved_config']
        _require_equal(config['model']['tokenizer_dir'], str(Path(tokenizer).resolve()), 'prepared tokenizer')
        _require_equal(config['dataset']['prepared_corpus_dir'], str(Path(corpus).resolve()), 'prepared corpus')
        for key, value in _provenance().items():
            _require_equal(run['initialization'].get(key), value, f'prepared source.{key}')
    reservation = _read_json(_reservation_path(root))
    _require_equal(reservation['manifest_hash'], manifest['manifest_hash'], 'prepared reservation')
    for relative, digest in manifest['artifact_sha256'].items():
        _require_equal(_source_record(output/relative)['sha256'], digest, f'prepared artifact.{relative}')
    _check_sources(manifest['reference_selection']['sources'])
    report = _read_json(output/'preflight.json')
    _require_equal(report['manifest_hash'], manifest['manifest_hash'], 'prepared report')
    return report


def inspect_warmup_execution(root, run, manifest):
    """Revalidate saved successful execution without requiring historical inputs."""
    from pathlib import Path
    import copy
    from scripts import run_tinystories_s1_warmup as ops
    from src.utils.config import ConfigError

    root = Path(root).resolve()
    plan = ops.report_source_plan(root)
    _require_equal(plan['bindings']['manifest_hash'], manifest['manifest_hash'], 'execution manifest')
    submissions = root/'launchers/submissions.json'
    jobs = _read_json(submissions)['jobs']
    selected = [j for j in jobs if j['arm_id'] == run['arm_id']]
    if not selected:
        raise ConfigError(f"{run['arm_id']}: missing execution attempt")
    intent = max(selected, key=lambda j: j['attempt_id'])
    if intent.get('status') != 'completed' or intent.get('bindings') != plan['bindings']:
        raise ConfigError('Latest execution attempt is not complete or has stale bindings')
    account = intent.get('scheduler_accounting', {})
    _require_equal(account.get('job_id'), intent.get('job_id'), 'execution accounting job')
    evidence = ops.execution_evidence(root, intent, plan)
    config = _read_json(root/'runs'/run['arm_id']/'config.json')
    controls = copy.deepcopy(run['resolved_config'])
    source = config['training'].get('effective_world_size_source')
    if source not in (controls['training']['effective_world_size_source'], 'single_process'):
        raise ConfigError('Incompatible execution world size source')
    controls['training']['effective_world_size_source'] = source
    _check_resolved_subset(controls, config, 'executed configuration')
    sources = [*evidence['sources'], _source_record(submissions), _source_record(root/'launchers/plan.json'),
               _source_record(root/'diagnostics/source-manifest.json')]
    sources.append(_source_record(root/'campaign/configs'/f"{run['arm_id']}.yaml"))
    snapshot = _read_json(root/'diagnostics/source-manifest.json')
    sources.extend(dict(path=str(root/'source'/name), sha256=digest) for name,digest in snapshot['files'].items())
    for mode in ('cpu','gpu'):
        gate_path = root/'diagnostics'/f'{mode}-gate.json'
        gate = ops.verify_gate(root, mode, expected=plan['bindings'])
        _require_equal(gate.get('status'), 'passed', f'{mode} execution gate')
        _require_equal(gate.get('bindings'), plan['bindings'], f'{mode} execution bindings')
        expected_hash = plan['cpu_gate_hash'] if mode == 'cpu' else intent['gpu_gate_hash']
        _require_equal(gate['content_hash'], expected_hash, f'{mode} execution gate hash')
        sources.append(_source_record(gate_path))
        for check in gate['checks']:
            if check.get('returncode') or check.get('failures') or check.get('errors') or not check.get('tests'):
                raise ConfigError('Execution gate lacks passing tests')
            sources.extend(check['artifacts'])
    _check_sources(sources)
    return dict(status='passed', job_id=intent['job_id'], attempt_id=intent['attempt_id'],
                source_sha256=plan['bindings']['source_sha256'], sources=sources)


def warmup_endpoint_rows(frozen, preflight, *, grid=None):
    """Add grid-qualified identities to the existing lossless endpoint projection."""
    from pathlib import Path
    definitions = {r['arm_id']:r for r in preflight['runs']}
    terminals = {r['arm_id']:r for r in frozen['runs']}
    rows = _endpoint_table(frozen, preflight)
    for row in rows:
        run = definitions[row['arm_id']]; saved = terminals[row['arm_id']]
        run_dir = Path(saved['run_dir'])
        sidecar = _read_json(run_dir/'terminal_validation_results.json')
        summary = _read_json(run_dir/'run_summary.json')['optimizer_ownership']
        grid_id = grid or run['grid_id']
        reference_campaign = WARMUP_REFERENCES[grid_id]['campaign_id']
        original_identity = [grid_id, reference_campaign, f'{reference_campaign}-S1-s42', row['ffn_dimension']]
        standalone_identity = [grid_id, reference_campaign, f"{reference_campaign}-ST-{row['width']}-s42", row['ffn_dimension']]
        role = 'standalone' if run['source_width'] else ('s1_warmup256' if preflight['schema_version']==5 else 's1_warmup64')
        config_path = Path(frozen['preflight_source']['path']).parent/'configs'/f"{run['arm_id']}.yaml"
        row.update(grid_id=grid_id, grid_label=grid_id.title(), role=role,
            warmup_updates=run['resolved_config']['training']['scheduler']['resolved_warmup_steps'],
            global_step=sidecar['global_step'], endpoint_identity=[grid_id,run['campaign_id'],run['run_id'],row['ffn_dimension']],
            evaluation_path=str(run_dir/'terminal_validation_results.json'),
            evaluation_sha256=_source_record(run_dir/'terminal_validation_results.json')['sha256'],
            config_path=str(config_path), config_sha256=_source_record(config_path)['sha256'],
            source_records=saved['sources'], source_identity=run['initialization']['source_files_sha256'],
            execution_evidence=saved['execution_evidence'], expected_exposure=summary['expected_exposure'],
            historical_reference=preflight['schema_version']!=5,
            matching_original_s1=original_identity, matching_standalone=standalone_identity)
        # Keep the same column set for all historical schema generations.
        for key in ('group','canonical_arm','clipping_observations'):
            row.pop(key, None)
    return rows


def _write_warmup_table(stage, stem, key, rows):
    import csv
    write_json_artifact(stage/(stem+'.json'), dict(schema_version=1, **{key:rows}))
    with (stage/(stem+'.csv')).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k:endpoint_csv_value(v) for k,v in r.items()} for r in rows)


def pair_warmup_endpoints(rows):
    from src.utils.config import ConfigError
    if len(rows)!=24 or len({tuple(r['endpoint_identity']) for r in rows})!=24:
        raise ConfigError('Comparison requires exactly 24 unique grid-qualified endpoints')
    deltas = []
    for grid in ('linear','geometric'):
        selected = [r for r in rows if r['grid_id']==grid]
        _require_equal(len(selected), 12, f'{grid}.endpoint count')
        for width in campaign_widths(5, f'S1-{grid}-w256'):
            dim = width['active_ffn_dimension']
            matches = {}
            for role in ('standalone','s1_warmup64','s1_warmup256'):
                candidates = [r for r in selected if r['role']==role and r['ffn_dimension']==dim]
                _require_equal(len(candidates), 1, f'{grid}.{dim}.{role}')
                matches[role] = candidates[0]
            old,new,st = (matches[r] for r in ('s1_warmup64','s1_warmup256','standalone'))
            for row in matches.values():
                row.update(matching_original_s1=old['endpoint_identity'], matching_standalone=st['endpoint_identity'])
            delta = dict(grid_id=grid, ffn_dimension=dim, non_embedding_parameters=width['non_embedding_parameters'],
                original_endpoint=old['endpoint_identity'], new_endpoint=new['endpoint_identity'], standalone_endpoint=st['endpoint_identity'])
            for metric in ('loss','perplexity'):
                delta.update({f'delta_{metric}':new[metric]-old[metric],
                    f'old_standalone_gap_{metric}':old[metric]-st[metric], f'new_standalone_gap_{metric}':new[metric]-st[metric]})
            deltas.append(delta)
    return deltas


def extract_warmup_early(run, root, *, end_step=1024):
    """Read bounded raw scalars, accepting only unambiguous committed observations."""
    import csv
    import json
    import math
    from pathlib import Path
    from src.utils.config import ConfigError

    if type(end_step) is not int or not 1024 <= end_step <= run['assigned_updates']:
        raise ConfigError('early-end-step must be between 1024 and the terminal update')
    root = Path(root); metric_source = _source_record(root/'metrics.csv')
    trace_source = _source_record(root/'optimizer_ownership_trace.jsonl')
    config_source = _source_record(root/'config.json')
    schema = run['optimizer_ownership_contract'].get('campaign_schema_version',1)
    grid = run.get('grid_id', 'geometric' if schema==4 else 'linear')
    widths = {w['label']:w['active_ffn_dimension'] for w in campaign_widths(schema,run['arm_id'])}
    commits = {}
    with (root/'optimizer_ownership_trace.jsonl').open() as stream:
        for line in stream:
            commit = json.loads(line); step = commit['step']
            if step>end_step: break
            if step in commits or commit.get('run_id')!=run['run_id']:
                raise ConfigError('Ambiguous committed early trace provenance')
            commits[step]=commit
    candidates = {}
    with (root/'metrics.csv').open() as stream:
        for ordinal,row in enumerate(csv.DictReader(stream),2):
            if row.get('split') not in ('train','validation'): continue
            step = int(row['step'])
            if step<1 or step>end_step: continue
            if row.get('optimizer_step_committed','').lower() in ('false','0'): continue
            if row.get('run_id') and row['run_id']!=run['run_id']:
                raise ConfigError('Early metric run identity mismatch')
            if row.get('optimizer_ownership_contract_hash') and row['optimizer_ownership_contract_hash']!=run['contract_hash']:
                raise ConfigError('Early metric contract identity mismatch')
            if row.get('evaluation_role') and row['evaluation_role']!=EVALUATION_ROLE:
                raise ConfigError('Early metric evaluation role mismatch')
            if step not in commits: raise ConfigError('Early metric lacks committed trace')
            width = row['granularity']
            if width not in widths: raise ConfigError('Early metric physical width mismatch')
            attempt = row.get('attempt_id') or row.get('process_uuid')
            if attempt and attempt!=commits[step].get('attempt_id'): continue
            if row['split']=='train' and commits[step].get('width') and width!=commits[step]['width']:
                raise ConfigError('Early training width differs from committed action')
            if row['split']=='train' and row.get('optimizer_batch_provenance'):
                try:
                    batch_provenance = json.loads(row['optimizer_batch_provenance'])
                except json.JSONDecodeError:
                    # Historical CSV writers serialized dictionaries with repr.
                    import ast
                    batch_provenance = ast.literal_eval(row['optimizer_batch_provenance'])
                _require_equal(batch_provenance, commits[step]['batch_provenance'], 'early committed batch')
            if row.get('optimizer_action_id') and commits[step].get('action_id') and row['optimizer_action_id']!=commits[step]['action_id']: continue
            key=(step,row['split'],width if row['split']=='validation' else None)
            if key in candidates: raise ConfigError(f'Ambiguous duplicate early observation: {key}')
            candidates[key]=(ordinal,row)
    observations=[]
    def append(step, split, width, metric, value, ordinal, row, kind='recorded'):
        value=float(value)
        if not math.isfinite(value) or (metric=='learning_rate' and value<0):
            raise ConfigError('Nonfinite/invalid early scalar')
        observations.append(dict(grid_id=grid,campaign_id=run['campaign_id'],run_id=run['run_id'],
            warmup_updates=run['resolved_config']['training']['scheduler']['resolved_warmup_steps'],step=step,split=split,
            width=width,ffn_dimension=widths.get(width),metric=metric,value=value,source_kind=kind,
            source_path=metric_source['path'] if kind=='recorded' else str(root/'config.json'),
            source_sha256=metric_source['sha256'] if kind=='recorded' else config_source['sha256'],
            source_row=ordinal if kind=='recorded' else None,attempt_id=commits.get(step,{}).get('attempt_id'),
            action_id=row.get('optimizer_action_id') or None,trace_source=trace_source,
            schedule_position=step-1 if metric=='learning_rate' else None,
            schedule_convention='update u applies position u-1; stored position u',smoothing='none'))
    for (step,split,_), (ordinal,row) in sorted(candidates.items()):
        if row.get('loss') not in ('',None): append(step,split,row['granularity'],'loss',row['loss'],ordinal,row)
        if split=='train' and row.get('learning_rate') not in ('',None):
            append(step,split,None,'learning_rate',row['learning_rate'],ordinal,row)
    training = {r['step'] for r in observations if r['split']=='train' and r['metric']=='loss'}
    missing = []
    if not training or max(training)<1024: missing.append('required training loss through update 1024')
    validation_steps={w:sorted(r['step'] for r in observations if r['split']=='validation' and r['width']==w) for w in widths}
    for width,steps in validation_steps.items():
        if not steps or max(steps)<1024: missing.append(f'required ordinary-validation loss through update 1024: {width}')
    if missing: raise ConfigError('Missing early evidence: '+', '.join(missing))
    lr_steps={r['step'] for r in observations if r['metric']=='learning_rate'}
    scheduler=run['resolved_config']['training']['scheduler']
    warmup=scheduler['resolved_warmup_steps']; horizon=run['assigned_updates']
    # Controls were validated by the terminal reader; reconstruction is never
    # labeled as measured execution, and losses are never filled in.
    for step in range(1,end_step+1):
        if step in lr_steps: continue
        position=step-1
        factor=position/warmup if position<warmup else .5*(1+math.cos(math.pi*(position-warmup)/(horizon-warmup)))
        append(step,'train',None,'learning_rate',.008*factor,None,{},'reconstructed_schedule')
    observations.sort(key=lambda r:(r['step'],r['split'],r['width'] or '',r['metric']))
    notes=dict(grid_id=grid,run_id=run['run_id'],window=[0,end_step],smoothing='none',
        training_missing_steps=sorted(set(range(1,end_step+1))-training),validation_steps=validation_steps,
        lr_reconstructed_steps=sorted(set(range(1,end_step+1))-lr_steps),
        explanation='Raw minibatch and ordinary-validation losses are distinct; no measured step-zero loss. Missing training steps are gaps; validation retains its recorded cadence.')
    _check_sources([metric_source,trace_source,config_source])
    return observations,notes


def warmup_endpoint_figure(rows, *, metric):
    from matplotlib.figure import Figure
    figure=Figure(figsize=(10,7)); ax=figure.subplots()
    for role,label,color,marker in [('standalone','Standalone (1 epoch)','#555555','^'),
            ('s1_warmup64','64-update warmup','#0072B2','o'),('s1_warmup256','256-update warmup','#D55E00','s')]:
        selected=sorted((r for r in rows if r['role']==role),key=lambda r:r['ffn_dimension'])
        ax.plot([r['non_embedding_parameters'] for r in selected],[r[metric] for r in selected],
            label=label,color=color,marker=marker,linestyle='None' if role=='standalone' else '-',
            markersize=10 if role=='standalone' else 5, fillstyle='none', zorder=4 if role=='standalone' else 2)
    ax.set(xlabel='Active non-embedding parameters (input embedding and LM head excluded)',ylabel=metric.capitalize())
    ax.set_xticks(sorted({r['non_embedding_parameters'] for r in rows})); ax.ticklabel_format(axis='x',style='plain')
    ax.legend();ax.grid(alpha=.2)
    figure.suptitle(f"{rows[0]['grid_label']} · TinyStories-Instruct · seed 42\nOrdinary validation · exact terminal checkpoints")
    figure.text(.5,.025,'Standalone: 1 epoch / 713,785,344 tokens; each S1: 4 epochs / 2,855,141,376 tokens.',ha='center',fontsize=9)
    figure.tight_layout(rect=(0,.07,1,.92))
    return figure


def warmup_early_figure(rows, *, grid, metric, end_step):
    from matplotlib.figure import Figure
    import math
    figure=Figure(figsize=(11,8 if metric=='loss' else 5))
    axes=list(figure.subplots(2,1)) if metric=='loss' else [figure.subplots()]
    selected=[r for r in rows if r['grid_id']==grid and r['metric']==metric]
    for warmup,color in ((64,'#0072B2'),(256,'#D55E00')):
        for index,split in enumerate(('train','validation') if metric=='loss' else ('train',)):
            values=[r for r in selected if r['warmup_updates']==warmup and r['split']==split]
            dimensions=sorted({r['ffn_dimension'] for r in values}) if split=='validation' else [None]
            for dimension in dimensions:
                series=sorted((r for r in values if split=='train' or r['ffn_dimension']==dimension),key=lambda r:r['step'])
                label=f'{warmup}-update warmup'+(f' · FFN {dimension}' if dimension else '')
                if metric=='learning_rate' and any(r['source_kind']=='reconstructed_schedule' for r in series):label+=' (includes reconstructed schedule)'
                if split=='validation':
                    # Disconnected markers preserve observed cadence without
                    # suggesting interpolation over missing evaluations.
                    axes[index].plot([r['step'] for r in series],[r['value'] for r in series],marker={32:'v',64:'o',128:'s',192:'D',256:'^'}[dimension],linestyle='None',color=color,label=label,markersize=4)
                else:
                    by_step={r['step']:r['value'] for r in series}
                    axes[index].plot(range(1,end_step+1),[by_step.get(s,math.nan) for s in range(1,end_step+1)],
                        color=color,label=label,linestyle='-' if warmup==64 else '--')
    for index,ax in enumerate(axes):
        ax.axvline(64,color='grey',linestyle=':',label='update 64');ax.axvline(256,color='grey',linestyle='--',label='update 256')
        ax.set(xlim=(0,end_step),xlabel='Committed absolute optimizer update',ylabel='Applied LR' if metric!='loss' else ('Training minibatch loss' if index==0 else 'Ordinary-validation loss'))
        ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2)
    figure.suptitle(f'{grid.title()} · TinyStories-Instruct · seed 42 · raw early observations')
    figure.text(.5,.015,'No smoothing or step-zero loss. Training gaps are breaks; validation markers retain recorded cadence.\nLR reconstruction, if present, is labeled and is not execution evidence. See early_metrics.json and report cadence/gaps.',ha='center',fontsize=8)
    figure.tight_layout(rect=(0,.065,1,.94))
    return figure


def report_s1_warmup(*, manifest, linear_reference_root, geometric_reference_root, output_dir, early_end_step=1024):
    """Publish a complete comparison atomically, or a separate incomplete record."""
    from pathlib import Path
    from src.utils.config import ConfigError
    output=Path(output_dir).resolve()
    endpoint_status=early_status='incomplete'
    sources=[]
    if type(early_end_step) is not int or early_end_step<1024:
        raise ConfigError('early-end-step must be at least 1024')
    # A diagnostic can be replaced on recovery; complete artifacts are immutable.
    if output.exists():
        existing=_read_json(output/'comparison_report.json')
        if existing.get('status')=='complete':
            _check_content_hash(existing, 'content_hash', 'existing comparison')
            if _source_record(manifest) not in existing['input_sources']:
                raise ConfigError('Existing comparison uses a different frozen manifest')
            _check_sources(existing['input_sources'])
            for relative,digest in existing['output_sha256'].items():
                _check_sources([dict(path=str(output/relative),sha256=digest)])
            _require_equal(existing['early_end_step'],early_end_step,'existing early window')
            _require_equal(existing['reference_roots'],dict(linear=str(Path(linear_reference_root).resolve()),geometric=str(Path(geometric_reference_root).resolve())),'existing reference roots')
            return existing
        if set(p.name for p in output.iterdir())!={'comparison_report.json'}:
            raise ConfigError(f'Occupied comparison output: {output}')
    try:
        frozen,preflight,sources,_=_validated_comparison_sources(manifest,schema_version=5)
        rows=warmup_endpoint_rows(frozen,preflight)
        references=inspect_warmup_references(linear_reference_root=linear_reference_root,geometric_reference_root=geometric_reference_root)
        sources.extend(references['sources'])
        early_runs=[(r,Path(s['run_dir'])) for r,s in zip(preflight['runs'],frozen['runs'],strict=True)]
        for grid,selection in references['grids'].items():
            historical=_read_preflight_manifest(selection['manifest_source']['path'])
            old_frozen=dict(status='complete',campaign_id=historical['campaign_id'],runs=selection['terminals'],preflight_source=selection['manifest_source'])
            rows.extend(warmup_endpoint_rows(old_frozen,historical,grid=grid))
            early_runs.append((selection['counterpart'],Path(selection['root'])/'runs/S1'))
        deltas=pair_warmup_endpoints(rows);endpoint_status='complete'
        early,notes=[],[]
        for run,root in early_runs:
            observations,note=extract_warmup_early(run,root,end_step=early_end_step)
            early.extend(observations);notes.append(note)
            sources.append(_source_record(root/'config.json'))
        early_status='complete'
        def publish(stage,destination):
            _write_warmup_table(stage,'endpoints','endpoints',rows)
            _write_warmup_table(stage,'paired_deltas','paired_deltas',deltas)
            _write_warmup_table(stage,'early_metrics','observations',early)
            figures=[]
            for grid in ('linear','geometric'):
                for metric in ('loss','perplexity'):
                    figure=warmup_endpoint_figure([r for r in rows if r['grid_id']==grid],metric=metric)
                    for suffix in ('png','pdf'):
                        name=f'{grid}_{metric}_vs_parameters.{suffix}';figure.savefig(stage/name);figures.append(str(destination/name))
                    figure.clear()
                for metric,label in (('learning_rate','lr'),('loss','loss')):
                    figure=warmup_early_figure(early,grid=grid,metric=metric,end_step=early_end_step)
                    for suffix in ('png','pdf'):
                        name=f'{grid}_early_{label}.{suffix}';figure.savefig(stage/name);figures.append(str(destination/name))
                    figure.clear()
            findings=['# Warmup comparison','', 'Descriptive seed-42 observations; no significance or causal diagnosis. Standalones use one epoch and S1 uses four; these are not equal per-run compute or runtime budgets.','']
            for d in deltas:
                direction='improved' if d['delta_loss']<0 else 'worsened' if d['delta_loss']>0 else 'unchanged'
                findings.append(f"- {d['grid_id'].title()} FFN {d['ffn_dimension']}: loss {direction}; new-minus-old loss {d['delta_loss']:+.8g}, perplexity {d['delta_perplexity']:+.8g}; standalone loss gaps old/new {d['old_standalone_gap_loss']:+.8g}/{d['new_standalone_gap_loss']:+.8g}, perplexity gaps {d['old_standalone_gap_perplexity']:+.8g}/{d['new_standalone_gap_perplexity']:+.8g}.")
            findings+=['','Early observations are raw minibatch loss, recorded or explicitly reconstructed applied LR, and separate per-width ordinary validation. Warmup boundaries are marked at 64 and 256; no step-zero loss is invented.']
            for run,root in early_runs:
                loss=[r for r in early if r['run_id']==run['run_id'] and r['split']=='train' and r['metric']=='loss']
                findings.append(f"- {run['run_id']}: observed minibatch loss {loss[0]['value']:.8g} at update {loss[0]['step']} to {loss[-1]['value']:.8g} at update {loss[-1]['step']}; this is not a validation estimate.")
            incomplete=sorted({r['run_id'] for r in rows if not r['resources']['measurement_complete']})
            findings+=['',f'Resource measurements marked incomplete for: {incomplete or "none"}. Attempted/replayed work and allocator measurements retain their saved completeness flags; no equal-runtime or resource-improvement claim is made.']
            (stage/'findings.md').write_text('\n'.join(findings)+'\n')
            _check_sources(sources)
            report=dict(schema_version=1,status='complete',endpoint_status='complete',early_status='complete',new_terminal_status='complete',
                endpoint_count=len(rows),paired_delta_count=len(deltas),early_observation_count=len(early),early_end_step=early_end_step,
                reference_roots={g:s['root'] for g,s in references['grids'].items()},selection_status=references['status'],
                figures=figures,input_sources=sources,early_notes=notes,reasons=[],holdout_evaluated=False,
                output_sha256={str(p.relative_to(stage)):_source_record(p)['sha256'] for p in stage.iterdir() if p.is_file()})
            report['content_hash']=stable_hash(report);write_json_artifact(stage/'comparison_report.json',report)
            return report
        if output.exists():
            (output/'comparison_report.json').unlink();output.rmdir()
        return _publish_directory(output,publish)
    except (ValueError,OSError,RuntimeError,KeyError,TypeError,IndexError,OverflowError) as error:
        try:
            _check_sources(sources)
        except (ValueError, OSError):
            endpoint_status='incomplete'
        record=dict(schema_version=1,status='incomplete',endpoint_status=endpoint_status,early_status='incomplete',
            reasons=[str(error)],input_sources=sources,holdout_evaluated=False)
        output.mkdir(parents=True,exist_ok=True)
        write_json_artifact(output/'comparison_report.json',record)
        raise ConfigError(f'Incomplete warmup comparison: {error}') from error
