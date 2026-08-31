"""Per-granularity optimizer ownership with one global scheduler clock."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from transformers import get_scheduler

from src.utils.config import ConfigError, resolve_optimizer_kwargs


OPTIMIZER_COLLECTION_SCHEMA_VERSION = 1


def _require_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{field} must be a nonnegative integer")
    return int(value)


def _validate_finite_values(value: Any, field: str) -> None:
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise ConfigError(f"{field} contains non-finite tensor state")
        return
    if isinstance(value, Mapping):
        for key, component in value.items():
            _validate_finite_values(component, f"{field}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, component in enumerate(value):
            _validate_finite_values(component, f"{field}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ConfigError(f"{field} contains a non-finite scalar")


def _ordered_model_parameters(model: torch.nn.Module) -> tuple[torch.nn.Parameter, ...]:
    # Match the historical optimizer construction exactly: every registered
    # parameter is present, while ``grad is None`` remains the inactivity gate.
    parameters = tuple(model.parameters())
    if not parameters:
        raise ConfigError("Per-granularity optimizer state requires trainable parameters")
    return parameters


def _build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    training: Mapping[str, Any],
) -> torch.optim.Optimizer:
    optimizer_name = str(training.get("optimizer_name", "adamw"))
    optimizer_kwargs = resolve_optimizer_kwargs(
        optimizer_name,
        training.get("optimizer_kwargs", {}),
    )
    learning_rate = training.get("resolved_learning_rate", training.get("learning_rate"))
    if learning_rate is None:
        raise ConfigError("training must include learning_rate or resolved_learning_rate")

    if optimizer_name == "adamw":
        if isinstance(optimizer_kwargs.get("betas"), list):
            optimizer_kwargs["betas"] = tuple(optimizer_kwargs["betas"])
        return torch.optim.AdamW(
            parameters,
            lr=float(learning_rate),
            **optimizer_kwargs,
        )
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=float(learning_rate),
            **optimizer_kwargs,
        )
    raise ConfigError(f"Unsupported optimizer name: {optimizer_name}")


@dataclass(frozen=True)
class WidthOptimizerEntry:
    """One optimizer state machine associated with one ordered width label."""

    granularity: str
    optimizer: torch.optim.Optimizer


class PerGranularityOptimizerCollection:
    """Ordered width-local optimizer states over one shared parameter set."""

    schema_version = OPTIMIZER_COLLECTION_SCHEMA_VERSION

    def __init__(
        self,
        entries: Sequence[WidthOptimizerEntry],
        *,
        parameters: Sequence[torch.nn.Parameter],
    ) -> None:
        if len(entries) < 2:
            raise ConfigError(
                "Per-granularity optimizer state requires at least two optimizers"
            )
        labels = [str(entry.granularity) for entry in entries]
        if len(set(labels)) != len(labels):
            raise ConfigError("Per-granularity optimizer labels must be unique")

        self.entries = tuple(entries)
        self._entries_by_granularity = {
            entry.granularity: entry for entry in self.entries
        }
        self._parameters = tuple(parameters)
        self.successful_update_counts = {label: 0 for label in labels}
        self.total_successful_updates = 0
        self.last_active_granularity: str | None = None

        expected_parameter_ids = tuple(id(parameter) for parameter in self._parameters)
        for entry in self.entries:
            actual_parameter_ids = tuple(
                id(parameter)
                for group in entry.optimizer.param_groups
                for parameter in group["params"]
            )
            if actual_parameter_ids != expected_parameter_ids:
                raise ConfigError(
                    "Every width optimizer must reference the same ordered model parameters"
                )
        self.validate_synchronized_learning_rates()

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        training: Mapping[str, Any],
    ) -> "PerGranularityOptimizerCollection":
        contract = training.get("optimizer_state_contract")
        if not isinstance(contract, Mapping):
            raise ConfigError("training.optimizer_state_contract must be resolved")
        labels = contract.get("ordered_granularities")
        if not isinstance(labels, list):
            raise ConfigError(
                "training.optimizer_state_contract.ordered_granularities must be a list"
            )
        canonical_labels = [str(label) for label in labels]
        parameters = _ordered_model_parameters(model)
        entries = [
            WidthOptimizerEntry(
                granularity=label,
                optimizer=_build_optimizer(parameters, training),
            )
            for label in canonical_labels
        ]
        return cls(entries, parameters=parameters)

    @property
    def ordered_granularities(self) -> tuple[str, ...]:
        return tuple(entry.granularity for entry in self.entries)

    @property
    def current_learning_rates(self) -> tuple[float, ...]:
        first = self.entries[0].optimizer
        return tuple(float(group["lr"]) for group in first.param_groups)

    def optimizer_for(self, granularity: str) -> torch.optim.Optimizer:
        label = str(granularity)
        entry = self._entries_by_granularity.get(label)
        if entry is None:
            raise ConfigError(
                "Per-granularity optimizer owner must be one of "
                f"{list(self.ordered_granularities)}; received {label!r}"
            )
        return entry.optimizer

    def owner_from_action(self, action: Mapping[str, Any]) -> str:
        selected = action.get("granularities")
        if not isinstance(selected, list) or len(selected) != 1:
            raise ConfigError(
                "Per-granularity optimizer commits require exactly one global width"
            )
        if str(action.get("kind")) != "global":
            raise ConfigError(
                "Per-granularity optimizer commits require a global-width action"
            )
        label = str(selected[0])
        self.optimizer_for(label)
        return label

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        # Every entry owns the same parameter objects. Clearing through one entry
        # clears the shared model gradients for all owners without touching state.
        self.entries[0].optimizer.zero_grad(set_to_none=set_to_none)

    def record_successful_update(self, granularity: str) -> None:
        label = str(granularity)
        self.optimizer_for(label)
        self.successful_update_counts[label] += 1
        self.total_successful_updates += 1
        self.last_active_granularity = label

    def synchronize_learning_rates(self, learning_rates: Sequence[float]) -> None:
        rates = tuple(float(rate) for rate in learning_rates)
        if not rates or any(not math.isfinite(rate) for rate in rates):
            raise RuntimeError("Global scheduler learning rates must be finite")
        for entry in self.entries:
            if len(entry.optimizer.param_groups) != len(rates):
                raise RuntimeError(
                    "Width optimizer parameter groups do not match the global clock"
                )
            for group, rate in zip(entry.optimizer.param_groups, rates, strict=True):
                group["lr"] = rate
        self.validate_synchronized_learning_rates()

    def validate_synchronized_learning_rates(self) -> tuple[float, ...]:
        expected = self.current_learning_rates
        if not expected or any(not math.isfinite(rate) for rate in expected):
            raise RuntimeError("Width optimizer learning rate is missing or non-finite")
        for entry in self.entries[1:]:
            actual = tuple(float(group["lr"]) for group in entry.optimizer.param_groups)
            if actual != expected:
                raise RuntimeError(
                    "All width optimizers must share the global learning rate"
                )
        return expected

    def state_dict(self) -> dict[str, Any]:
        """Return the complete ordered state required for exact continuation."""

        return {
            "schema_version": self.schema_version,
            "ordered_entries": [
                {
                    "granularity": entry.granularity,
                    "state_dict": entry.optimizer.state_dict(),
                }
                for entry in self.entries
            ],
            "successful_update_counts": copy.deepcopy(
                self.successful_update_counts
            ),
            "total_successful_updates": self.total_successful_updates,
            "last_active_granularity": self.last_active_granularity,
            "current_learning_rates": list(self.current_learning_rates),
        }

    def validate_state_dict(self, state_dict: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and stage a collection payload without changing runtime state."""

        if not isinstance(state_dict, Mapping):
            raise ConfigError("Optimizer collection checkpoint must be a mapping")
        if state_dict.get("schema_version") != self.schema_version:
            raise ConfigError(
                "Optimizer collection checkpoint schema version does not match"
            )
        ordered_entries = state_dict.get("ordered_entries")
        if not isinstance(ordered_entries, list):
            raise ConfigError("Optimizer collection checkpoint entries are missing")
        if any(not isinstance(entry, Mapping) for entry in ordered_entries):
            raise ConfigError("Optimizer collection checkpoint entry is malformed")
        labels = [str(entry.get("granularity")) for entry in ordered_entries]
        if labels != list(self.ordered_granularities):
            raise ConfigError("Optimizer collection checkpoint order does not match")

        for entry_index, (runtime_entry, saved_entry) in enumerate(
            zip(self.entries, ordered_entries, strict=True)
        ):
            saved_optimizer = saved_entry.get("state_dict")
            if not isinstance(saved_optimizer, Mapping):
                raise ConfigError(
                    f"Optimizer collection entry {entry_index} state is missing"
                )
            saved_groups = saved_optimizer.get("param_groups")
            saved_parameter_state = saved_optimizer.get("state")
            runtime_groups = runtime_entry.optimizer.state_dict()["param_groups"]
            if not isinstance(saved_groups, list) or len(saved_groups) != len(
                runtime_groups
            ):
                raise ConfigError(
                    f"Optimizer collection entry {entry_index} parameter groups do not match"
                )
            if not isinstance(saved_parameter_state, Mapping):
                raise ConfigError(
                    f"Optimizer collection entry {entry_index} parameter state is malformed"
                )

            saved_ids: list[Any] = []
            parameters: list[torch.nn.Parameter] = []
            for group_index, (saved_group, runtime_group, live_group) in enumerate(
                zip(
                    saved_groups,
                    runtime_groups,
                    runtime_entry.optimizer.param_groups,
                    strict=True,
                )
            ):
                if not isinstance(saved_group, Mapping):
                    raise ConfigError(
                        f"Optimizer collection entry {entry_index} group {group_index} is malformed"
                    )
                saved_params = saved_group.get("params")
                runtime_params = runtime_group.get("params")
                if not isinstance(saved_params, list) or len(saved_params) != len(
                    runtime_params
                ):
                    raise ConfigError(
                        f"Optimizer collection entry {entry_index} parameter layout does not match"
                    )
                for key, runtime_value in runtime_group.items():
                    if key in {"params", "lr"}:
                        continue
                    if saved_group.get(key) != runtime_value:
                        raise ConfigError(
                            f"Optimizer collection entry {entry_index} group contract does not match"
                        )
                saved_ids.extend(saved_params)
                parameters.extend(live_group["params"])

            if len(set(saved_ids)) != len(saved_ids) or any(
                parameter_id not in saved_ids for parameter_id in saved_parameter_state
            ):
                raise ConfigError(
                    f"Optimizer collection entry {entry_index} parameter IDs are malformed"
                )
            parameter_by_id = dict(zip(saved_ids, parameters, strict=True))
            for parameter_id, parameter_state in saved_parameter_state.items():
                if not isinstance(parameter_state, Mapping):
                    raise ConfigError(
                        f"Optimizer collection entry {entry_index} parameter state is malformed"
                    )
                parameter = parameter_by_id[parameter_id]
                _validate_finite_values(
                    parameter_state,
                    f"optimizer_collection.ordered_entries[{entry_index}].state",
                )
                for value in parameter_state.values():
                    if (
                        torch.is_tensor(value)
                        and value.ndim > 0
                        and tuple(value.shape) != tuple(parameter.shape)
                    ):
                        raise ConfigError(
                            f"Optimizer collection entry {entry_index} tensor shape does not match"
                        )

        counts = state_dict.get("successful_update_counts")
        if not isinstance(counts, Mapping) or list(counts) != list(
            self.ordered_granularities
        ):
            raise ConfigError("Optimizer collection update-count labels do not match")
        normalized_counts = {
            label: _require_nonnegative_int(
                counts[label], f"successful_update_counts.{label}"
            )
            for label in self.ordered_granularities
        }
        total = _require_nonnegative_int(
            state_dict.get("total_successful_updates"),
            "total_successful_updates",
        )
        if total != sum(normalized_counts.values()):
            raise ConfigError("Optimizer collection successful-update totals do not match")
        last_owner = state_dict.get("last_active_granularity")
        if last_owner is not None and str(last_owner) not in self.ordered_granularities:
            raise ConfigError("Optimizer collection last active owner is unknown")
        if total == 0 and last_owner is not None:
            raise ConfigError("Optimizer collection cannot have an owner before a commit")
        if total > 0 and last_owner is None:
            raise ConfigError("Optimizer collection last active owner is missing")

        raw_rates = state_dict.get("current_learning_rates")
        if not isinstance(raw_rates, (list, tuple)):
            raise ConfigError("Optimizer collection current learning rates are missing")
        rates = tuple(float(rate) for rate in raw_rates)
        expected_group_count = len(self.entries[0].optimizer.param_groups)
        if len(rates) != expected_group_count or any(
            not math.isfinite(rate) for rate in rates
        ):
            raise ConfigError("Optimizer collection current learning rates are invalid")
        for entry_index, saved_entry in enumerate(ordered_entries):
            entry_rates = tuple(
                float(group["lr"])
                for group in saved_entry["state_dict"]["param_groups"]
            )
            if entry_rates != rates:
                raise ConfigError(
                    f"Optimizer collection entry {entry_index} learning rates are not synchronized"
                )

        normalized = copy.deepcopy(dict(state_dict))
        normalized["successful_update_counts"] = normalized_counts
        normalized["total_successful_updates"] = total
        normalized["last_active_granularity"] = (
            str(last_owner) if last_owner is not None else None
        )
        normalized["current_learning_rates"] = list(rates)
        return normalized

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore a fully validated collection, rolling back on load failure."""

        staged = self.validate_state_dict(state_dict)
        ordered_entries = staged["ordered_entries"]
        snapshots = [copy.deepcopy(entry.optimizer.state_dict()) for entry in self.entries]
        metadata_snapshot = (
            copy.deepcopy(self.successful_update_counts),
            self.total_successful_updates,
            self.last_active_granularity,
        )
        try:
            for runtime_entry, saved_entry in zip(
                self.entries, ordered_entries, strict=True
            ):
                runtime_entry.optimizer.load_state_dict(
                    copy.deepcopy(dict(saved_entry["state_dict"]))
                )
            self.successful_update_counts = copy.deepcopy(
                staged["successful_update_counts"]
            )
            self.total_successful_updates = staged["total_successful_updates"]
            self.last_active_granularity = staged["last_active_granularity"]
            self.synchronize_learning_rates(staged["current_learning_rates"])
        except Exception:
            for runtime_entry, snapshot in zip(
                self.entries, snapshots, strict=True
            ):
                runtime_entry.optimizer.load_state_dict(snapshot)
            (
                self.successful_update_counts,
                self.total_successful_updates,
                self.last_active_granularity,
            ) = metadata_snapshot
            raise


class GlobalSchedulerClock:
    """One scheduler position whose rates are fanned out to every width."""

    def __init__(
        self,
        *,
        carrier_optimizer: torch.optim.Optimizer,
        scheduler,
    ) -> None:
        self._carrier_optimizer = carrier_optimizer
        self._scheduler = scheduler
        self.position = 0
        self.last_committed_learning_rates: tuple[float, ...] | None = None

    @classmethod
    def from_training(cls, training: Mapping[str, Any]) -> "GlobalSchedulerClock":
        learning_rate = training.get(
            "resolved_learning_rate", training.get("learning_rate")
        )
        if learning_rate is None:
            raise ConfigError(
                "training must include learning_rate or resolved_learning_rate"
            )
        scalar = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        carrier_optimizer = torch.optim.SGD([scalar], lr=float(learning_rate))
        scheduler_name = str(training.get("scheduler_name", "cosine"))
        scheduler_kwargs = dict(training.get("scheduler_kwargs", {}))
        warmup_steps = int(
            training.get(
                "resolved_warmup_steps",
                training.get("scheduler", {}).get("resolved_warmup_steps", 0),
            )
        )
        scheduler = get_scheduler(
            scheduler_name,
            optimizer=carrier_optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=(
                None
                if scheduler_name == "warmup_stable_decay"
                and "num_stable_steps" in scheduler_kwargs
                else int(training["max_steps"])
            ),
            scheduler_specific_kwargs=scheduler_kwargs,
        )
        return cls(carrier_optimizer=carrier_optimizer, scheduler=scheduler)

    @property
    def current_learning_rates(self) -> tuple[float, ...]:
        return tuple(float(group["lr"]) for group in self._carrier_optimizer.param_groups)

    def synchronize(self, collection: PerGranularityOptimizerCollection) -> None:
        collection.synchronize_learning_rates(self.current_learning_rates)

    def step(self) -> None:
        self.last_committed_learning_rates = self.current_learning_rates
        # Advancing the scalar optimizer satisfies scheduler call ordering while
        # owning no model parameter or optimizer history used by the experiment.
        self._carrier_optimizer.step()
        self._scheduler.step()
        self.position += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "scheduler_state_dict": self._scheduler.state_dict(),
            "carrier_optimizer_state_dict": self._carrier_optimizer.state_dict(),
            "position": self.position,
            "last_committed_learning_rates": (
                list(self.last_committed_learning_rates)
                if self.last_committed_learning_rates is not None
                else None
            ),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        staged = self.validate_state_dict(state_dict)
        snapshot = copy.deepcopy(self.state_dict())
        try:
            self._load_validated_state_dict(staged)
        except Exception:
            self._load_validated_state_dict(snapshot)
            raise

    def validate_state_dict(self, state_dict: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(state_dict, Mapping):
            raise ConfigError("Global scheduler clock checkpoint must be a mapping")
        scheduler_state = state_dict.get("scheduler_state_dict")
        carrier_state = state_dict.get("carrier_optimizer_state_dict")
        if not isinstance(scheduler_state, Mapping) or not isinstance(
            carrier_state, Mapping
        ):
            raise ConfigError("Global scheduler clock checkpoint state is incomplete")
        position = _require_nonnegative_int(
            state_dict.get("position"), "global scheduler position"
        )
        _validate_finite_values(scheduler_state, "global_scheduler.scheduler_state")
        _validate_finite_values(carrier_state, "global_scheduler.carrier_state")
        last_epoch = scheduler_state.get("last_epoch")
        if (
            isinstance(last_epoch, bool)
            or not isinstance(last_epoch, int)
            or last_epoch != position
        ):
            raise ConfigError("Global scheduler state position does not match")
        groups = carrier_state.get("param_groups")
        if not isinstance(groups, list) or len(groups) != len(
            self._carrier_optimizer.param_groups
        ):
            raise ConfigError("Global scheduler carrier parameter groups do not match")
        current_rates = tuple(float(group.get("lr")) for group in groups)
        if not current_rates or any(not math.isfinite(rate) for rate in current_rates):
            raise ConfigError("Global scheduler current learning rates are invalid")
        saved_rates = state_dict.get("last_committed_learning_rates")
        if saved_rates is None:
            if position != 0:
                raise ConfigError("Global scheduler last committed rates are missing")
            normalized_last_rates = None
        else:
            if not isinstance(saved_rates, (list, tuple)):
                raise ConfigError("Global scheduler last committed rates are malformed")
            normalized_last_rates = tuple(float(rate) for rate in saved_rates)
            if len(normalized_last_rates) != len(current_rates) or any(
                not math.isfinite(rate) for rate in normalized_last_rates
            ):
                raise ConfigError("Global scheduler last committed rates are invalid")
        normalized = copy.deepcopy(dict(state_dict))
        normalized["position"] = position
        normalized["last_committed_learning_rates"] = (
            list(normalized_last_rates) if normalized_last_rates is not None else None
        )
        return normalized

    def _load_validated_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self._carrier_optimizer.load_state_dict(
            copy.deepcopy(dict(state_dict["carrier_optimizer_state_dict"]))
        )
        self._scheduler.load_state_dict(
            copy.deepcopy(dict(state_dict["scheduler_state_dict"]))
        )
        self.position = int(state_dict["position"])
        saved_rates = state_dict.get("last_committed_learning_rates")
        self.last_committed_learning_rates = (
            tuple(float(rate) for rate in saved_rates)
            if saved_rates is not None
            else None
        )


def build_per_granularity_optimizer_runtime(
    model: torch.nn.Module,
    training: Mapping[str, Any],
) -> tuple[PerGranularityOptimizerCollection, GlobalSchedulerClock]:
    collection = PerGranularityOptimizerCollection.from_model(model, training)
    clock = GlobalSchedulerClock.from_training(training)
    clock.synchronize(collection)
    return collection, clock
