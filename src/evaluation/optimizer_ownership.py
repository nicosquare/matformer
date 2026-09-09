"""Fixed TinyStories seed-42 campaign definitions and plain artifact records.

This setup module does not resolve configs, construct models, or run experiments.
Record types describe JSON-compatible dictionaries, not runtime validators.
Use the imported stable_hash and write_json_artifact utilities directly as later
phases add scientific contract serialization and atomic artifact publication.
Scientific hashes must cover resolved controls; hashing an arm definition alone
does not establish a valid campaign identity. Paths in saved records are strings.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

from src.utils.metrics import write_json_artifact
from src.utils.reproducibility import stable_hash


CAMPAIGN_SCHEMA_VERSION = 1
SCIENTIFIC_CONTRACT_SCHEMA_VERSION = 1
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
    {"label": "g250", "source_fraction": 0.25, "active_ffn_dimension": 64,
     "non_embedding_parameters": 115_264, "active_quarters": ("A",)},
    {"label": "g500", "source_fraction": 0.50, "active_ffn_dimension": 128,
     "non_embedding_parameters": 164_416, "active_quarters": ("A", "B")},
    {"label": "g750", "source_fraction": 0.75, "active_ffn_dimension": 192,
     "non_embedding_parameters": 213_568, "active_quarters": ("A", "B", "C")},
    {"label": "g1000", "source_fraction": 1.00, "active_ffn_dimension": 256,
     "non_embedding_parameters": 262_720, "active_quarters": QUARTER_IDS},
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
