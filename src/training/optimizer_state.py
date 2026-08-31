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
        """Return the minimal Phase-3 runtime state used by ordinary checkpoints."""

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

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore collection state; Phase 6 adds full staged resume validation."""

        ordered_entries = state_dict.get("ordered_entries")
        if not isinstance(ordered_entries, list):
            raise ConfigError("Optimizer collection checkpoint entries are missing")
        labels = [entry.get("granularity") for entry in ordered_entries]
        if labels != list(self.ordered_granularities):
            raise ConfigError("Optimizer collection checkpoint order does not match")
        for runtime_entry, saved_entry in zip(
            self.entries, ordered_entries, strict=True
        ):
            runtime_entry.optimizer.load_state_dict(dict(saved_entry["state_dict"]))
        counts = state_dict.get("successful_update_counts", {})
        self.successful_update_counts = {
            label: int(counts[label]) for label in self.ordered_granularities
        }
        self.total_successful_updates = int(
            state_dict.get(
                "total_successful_updates",
                sum(self.successful_update_counts.values()),
            )
        )
        self.last_active_granularity = state_dict.get("last_active_granularity")
        self.synchronize_learning_rates(state_dict["current_learning_rates"])


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
        self._carrier_optimizer.load_state_dict(
            dict(state_dict["carrier_optimizer_state_dict"])
        )
        self._scheduler.load_state_dict(dict(state_dict["scheduler_state_dict"]))
        self.position = int(state_dict.get("position", 0))
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
