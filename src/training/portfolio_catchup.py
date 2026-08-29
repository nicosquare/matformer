"""Online state for the amortized four-granularity catch-up experiment."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.utils.reproducibility import stable_hash


PORTFOLIO_SCHEMA_VERSION = 2
PORTFOLIO_GRANULARITIES = ("g250", "g500", "g750", "g1000")


class PortfolioCatchupError(ValueError):
    """Raised when catch-up state or immutable experiment inputs are invalid."""


def uses_portfolio_catchup(config: Mapping[str, Any]) -> bool:
    controlled = config.get("controlled_experiment", {})
    catchup = (
        controlled.get("portfolio_catchup", {})
        if isinstance(controlled, Mapping)
        else {}
    )
    return bool(
        isinstance(catchup, Mapping)
        and catchup.get("enabled") is True
        and controlled.get("comparison_role") == "elastic_candidate"
    )


def manifest_hash(payload: Mapping[str, Any]) -> str:
    body = copy.deepcopy(dict(payload))
    body.pop("manifest_hash", None)
    return stable_hash(body)


def load_immutable_manifest(
    path: str | Path,
    expected_hash: str,
    *,
    artifact_name: str,
) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PortfolioCatchupError(
            f"Cannot read {artifact_name}: {manifest_path}"
        ) from error
    if not isinstance(payload, dict):
        raise PortfolioCatchupError(f"{artifact_name} must be a JSON mapping")
    actual_hash = manifest_hash(payload)
    if payload.get("manifest_hash") != actual_hash:
        raise PortfolioCatchupError(f"{artifact_name} embedded hash mismatch")
    if str(expected_hash) != actual_hash:
        raise PortfolioCatchupError(
            f"{artifact_name} does not match its configured immutable hash"
        )
    return payload


def build_portfolio_catchup_state(config: Mapping[str, Any]) -> dict[str, Any] | None:
    if not uses_portfolio_catchup(config):
        return None

    controlled = config["controlled_experiment"]
    contract = controlled["portfolio_catchup"]
    target_manifest = load_immutable_manifest(
        contract["target_manifest_path"],
        contract["target_manifest_hash"],
        artifact_name="standalone portfolio target manifest",
    )
    seed = str(int(config["run"]["seed"]))
    raw_targets = target_manifest.get("targets", {}).get(seed)
    if not isinstance(raw_targets, Mapping):
        raise PortfolioCatchupError(
            f"Target manifest has no standalone targets for seed {seed}"
        )
    ordered_granularities = [str(value) for value in contract["granularities"]]
    targets: dict[str, dict[str, float]] = {}
    for granularity in ordered_granularities:
        raw_target = raw_targets.get(granularity)
        if not isinstance(raw_target, Mapping):
            raise PortfolioCatchupError(
                f"Target manifest has no {granularity} target for seed {seed}"
            )
        loss = _finite_float(raw_target.get("target_loss"), "target_loss")
        perplexity = _finite_float(
            raw_target.get("target_perplexity", math.exp(loss)),
            "target_perplexity",
        )
        targets[granularity] = {
            "target_loss": loss,
            "target_perplexity": perplexity,
        }

    tolerance = _finite_float(contract["perplexity_tolerance"], "perplexity_tolerance")
    state = {
        "schema_version": PORTFOLIO_SCHEMA_VERSION,
        "comparison_group_id": controlled["comparison_group_id"],
        "comparison_role": controlled["comparison_role"],
        "target_manifest_hash": target_manifest["manifest_hash"],
        "learning_rate": _finite_float(
            config["training"]["resolved_learning_rate"],
            "learning_rate",
        ),
        "seed": int(seed),
        "ordered_granularities": ordered_granularities,
        "targets": targets,
        "perplexity_tolerance": tolerance,
        "loss_tolerance": math.log1p(tolerance),
        "required_consecutive_evaluations": int(
            contract["required_consecutive_evaluations"]
        ),
        "evaluation_count": 0,
        "streak_length": 0,
        "streak_onset_step": None,
        "streak_onset_tokens": None,
        "last_validation_step": None,
        "last_validation_tokens": None,
        "last_per_width": {},
        "last_joint_max_loss_gap": None,
        "last_joint_qualifies": False,
        "confirmed": False,
        "confirmation_step": None,
        "confirmation_tokens": None,
        "confirmation_checkpoint_path": None,
        "confirmation_checkpoint_sha256": None,
        "confirmation_checkpoint_saved": False,
    }
    state["contract_hash"] = _state_contract_hash(state)
    return state


def validate_portfolio_catchup_state(
    state: Mapping[str, Any] | None,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not uses_portfolio_catchup(config):
        return None
    if not isinstance(state, Mapping):
        raise PortfolioCatchupError(
            "Elastic candidate checkpoint lacks portfolio_catchup_state"
        )
    normalized = copy.deepcopy(dict(state))
    expected = build_portfolio_catchup_state(config)
    if expected is None:
        raise AssertionError("portfolio catch-up configuration was not active")
    if normalized.get("schema_version") != PORTFOLIO_SCHEMA_VERSION:
        raise PortfolioCatchupError("Unsupported portfolio catch-up state schema")
    if normalized.get("contract_hash") != _state_contract_hash(normalized):
        raise PortfolioCatchupError(
            "Portfolio catch-up checkpoint contract hash is internally inconsistent"
        )
    if normalized.get("contract_hash") != expected["contract_hash"]:
        raise PortfolioCatchupError(
            "Portfolio catch-up checkpoint contract does not match the current run"
        )

    integer_fields = (
        "evaluation_count",
        "streak_length",
    )
    for field in integer_fields:
        value = normalized.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PortfolioCatchupError(f"portfolio catch-up {field} is invalid")
    required = int(normalized["required_consecutive_evaluations"])
    streak = int(normalized["streak_length"])
    if required <= 0 or streak > required:
        raise PortfolioCatchupError("Portfolio catch-up streak state is invalid")
    if normalized.get("confirmed") is True:
        if streak < required:
            raise PortfolioCatchupError(
                "Confirmed portfolio catch-up has an incomplete streak"
            )
        if (
            normalized.get("confirmation_step") is None
            or normalized.get("confirmation_tokens") is None
        ):
            raise PortfolioCatchupError(
                "Confirmed portfolio catch-up lacks its confirmation boundary"
            )
        if normalized.get("confirmation_checkpoint_saved") is True:
            checkpoint_path = normalized.get("confirmation_checkpoint_path")
            checksum = normalized.get("confirmation_checkpoint_sha256")
            if not checkpoint_path or not checksum:
                raise PortfolioCatchupError(
                    "Confirmed portfolio catch-up lacks checkpoint provenance"
                )
            path = Path(str(checkpoint_path)).expanduser()
            if not path.is_file():
                raise PortfolioCatchupError(
                    "Portfolio confirmation checkpoint artifact is missing"
                )
    return normalized


def update_portfolio_catchup_state(
    state: dict[str, Any],
    validation_results: Sequence[Mapping[str, Any]],
    *,
    step: int,
    tokens_seen: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    """Commit one simultaneous all-width validation observation."""

    if state.get("confirmed") is True:
        # Continue recording diagnostics after confirmation without moving t*.
        was_confirmed = True
    else:
        was_confirmed = False

    by_width: dict[str, Mapping[str, Any]] = {}
    for result in validation_results:
        granularity = str(result.get("granularity"))
        if granularity in by_width:
            raise PortfolioCatchupError(
                f"Duplicate validation result for {granularity}"
            )
        by_width[granularity] = result
    expected = list(state["ordered_granularities"])
    if set(by_width) != set(expected):
        raise PortfolioCatchupError(
            "Portfolio validation must contain all configured granularities together"
        )
    if state.get("last_validation_step") is not None and int(step) <= int(
        state["last_validation_step"]
    ):
        raise PortfolioCatchupError(
            "Portfolio validation boundaries must be strictly increasing"
        )

    loss_tolerance = float(state["loss_tolerance"])
    decorated: list[dict[str, Any]] = []
    per_width: dict[str, dict[str, Any]] = {}
    for granularity in expected:
        result = dict(by_width[granularity])
        loss = _finite_float(result.get("loss"), f"{granularity} validation loss")
        target_loss = float(state["targets"][granularity]["target_loss"])
        loss_gap = loss - target_loss
        qualifies = loss_gap <= loss_tolerance
        diagnostic = {
            "target_loss": target_loss,
            "target_perplexity": float(
                state["targets"][granularity]["target_perplexity"]
            ),
            "loss": loss,
            "loss_gap": loss_gap,
            "perplexity_deficit": math.expm1(loss_gap),
            "qualifies": qualifies,
        }
        per_width[granularity] = diagnostic
        result.update(
            {
                "portfolio_target_loss": target_loss,
                "portfolio_loss_gap": loss_gap,
                "portfolio_qualifies": qualifies,
            }
        )
        decorated.append(result)

    joint_max_gap = max(row["loss_gap"] for row in per_width.values())
    joint_qualifies = all(row["qualifies"] for row in per_width.values())
    if was_confirmed:
        streak_length = int(state["streak_length"])
    elif joint_qualifies:
        if int(state["streak_length"]) == 0:
            state["streak_onset_step"] = int(step)
            state["streak_onset_tokens"] = int(tokens_seen)
        streak_length = int(state["streak_length"]) + 1
    else:
        streak_length = 0
        state["streak_onset_step"] = None
        state["streak_onset_tokens"] = None

    state.update(
        {
            "evaluation_count": int(state["evaluation_count"]) + 1,
            "streak_length": streak_length,
            "last_validation_step": int(step),
            "last_validation_tokens": int(tokens_seen),
            "last_per_width": per_width,
            "last_joint_max_loss_gap": joint_max_gap,
            "last_joint_qualifies": joint_qualifies,
        }
    )
    newly_confirmed = False
    required = int(state["required_consecutive_evaluations"])
    if not was_confirmed and streak_length >= required:
        state["streak_length"] = required
        state["confirmed"] = True
        state["confirmation_step"] = int(step)
        state["confirmation_tokens"] = int(tokens_seen)
        newly_confirmed = True
    return state, decorated, newly_confirmed


def portfolio_metric_fields(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "portfolio_joint_max_loss_gap": state.get("last_joint_max_loss_gap"),
        "portfolio_joint_qualifies": state.get("last_joint_qualifies"),
        "portfolio_streak_length": state.get("streak_length"),
        "portfolio_onset_step": state.get("streak_onset_step"),
        "portfolio_onset_tokens": state.get("streak_onset_tokens"),
        "portfolio_confirmation_step": state.get("confirmation_step"),
        "portfolio_confirmation_tokens": state.get("confirmation_tokens"),
        "portfolio_target_manifest_hash": state.get("target_manifest_hash"),
        "portfolio_confirmation_checkpoint_path": state.get(
            "confirmation_checkpoint_path"
        ),
        "portfolio_confirmation_checkpoint_sha256": state.get(
            "confirmation_checkpoint_sha256"
        ),
    }


def _state_contract_hash(state: Mapping[str, Any]) -> str:
    fields = {
        key: state[key]
        for key in (
            "schema_version",
            "comparison_group_id",
            "comparison_role",
            "target_manifest_hash",
            "learning_rate",
            "seed",
            "ordered_granularities",
            "targets",
            "perplexity_tolerance",
            "loss_tolerance",
            "required_consecutive_evaluations",
        )
    }
    return stable_hash(fields)


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PortfolioCatchupError(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise PortfolioCatchupError(f"{field} must be finite")
    return result
