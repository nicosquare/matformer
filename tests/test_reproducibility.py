import copy
import random
from types import SimpleNamespace

import pytest
import torch
import yaml
from datasets import Dataset

from src.evaluation.validation import evaluate_validation_loss
from src.training.data import (
    DistributedValidationSampler,
    build_dataloaders,
    build_language_model_dataloader,
    collate_language_model_batch,
)
from src.training.checkpointing import _validate_reproducibility_payload
from src.training.run import run_training
from src.utils.config import ConfigError, resolve_run_config
from src.utils.metrics import build_scaling_result_rows
from src.utils.reproducibility import (
    build_balanced_warmup_schedule,
    build_controller_reset_schedule,
    build_comparison_control_signature,
    derive_seed,
    seed_for,
    seed_model_initialization,
)


def test_named_seed_derivation_is_stable_and_independent():
    assert derive_seed(42, "model_initialization") == derive_seed(
        42, "model_initialization"
    )
    assert derive_seed(42, "model_initialization") != derive_seed(
        42, "training_sampler"
    )
    assert derive_seed(42, "training_sampler") != derive_seed(
        43, "training_sampler"
    )


PROBABILISTIC_SEED_STREAMS = (
    "controller_panel",
    "final_holdout",
    "posterior_sampling",
    "pre_nested_warmup_schedule",
    "controller_reset_schedule",
)


def test_balanced_warmup_schedule_is_exact_deterministic_and_hash_stable():
    labels = ["micro", "small", "medium", "large", "full"]
    seed = derive_seed(42, "pre_nested_warmup_schedule")
    schedule, schedule_hash = build_balanced_warmup_schedule(
        labels,
        passes=2,
        seed=seed,
        action_interval_steps=50,
        duration_steps=500,
    )
    repeated, repeated_hash = build_balanced_warmup_schedule(
        labels,
        passes=2,
        seed=seed,
        action_interval_steps=50,
        duration_steps=500,
    )
    other, other_hash = build_balanced_warmup_schedule(
        labels,
        passes=2,
        seed=derive_seed(43, "pre_nested_warmup_schedule"),
        action_interval_steps=50,
        duration_steps=500,
    )

    assert schedule == repeated
    assert schedule_hash == repeated_hash
    assert schedule != other
    assert schedule_hash != other_hash
    assert len(schedule) == 10
    assert all(schedule.count(label) == 2 for label in labels)
    assert set(schedule[:5]) == set(labels)
    assert set(schedule[5:]) == set(labels)


def test_probabilistic_seed_streams_are_stable_distinct_and_root_seeded():
    first = {
        stream_name: derive_seed(42, stream_name)
        for stream_name in PROBABILISTIC_SEED_STREAMS
    }
    second = {
        stream_name: derive_seed(42, stream_name)
        for stream_name in PROBABILISTIC_SEED_STREAMS
    }

    assert first == second
    assert len(set(first.values())) == len(PROBABILISTIC_SEED_STREAMS)
    assert all(
        first[stream_name] != derive_seed(43, stream_name)
        for stream_name in PROBABILISTIC_SEED_STREAMS
    )


def test_controller_reset_episode_schedules_are_indexed_and_rng_independent():
    labels = ["micro", "medium", "full"]
    root_seed = derive_seed(42, "controller_reset_schedule")
    first, first_seed, first_hash = build_controller_reset_schedule(
        labels,
        acquisition_passes=2,
        root_seed=root_seed,
        episode_index=0,
    )
    repeated, repeated_seed, repeated_hash = build_controller_reset_schedule(
        labels,
        acquisition_passes=2,
        root_seed=root_seed,
        episode_index=0,
    )
    second, second_seed, second_hash = build_controller_reset_schedule(
        labels,
        acquisition_passes=2,
        root_seed=root_seed,
        episode_index=1,
    )

    assert (first, first_seed, first_hash) == (
        repeated,
        repeated_seed,
        repeated_hash,
    )
    assert first_seed != second_seed
    assert first_hash != second_hash
    assert len(first) == len(second) == 6
    assert set(first[:3]) == set(first[3:]) == set(labels)
    assert set(second[:3]) == set(second[3:]) == set(labels)


def test_probabilistic_split_and_sampling_streams_reproduce_independently():
    controller_seed = derive_seed(42, "controller_panel")
    final_seed = derive_seed(42, "final_holdout")
    posterior_seed = derive_seed(42, "posterior_sampling")

    expected_controller = random.Random(controller_seed).sample(range(2048), 128)
    expected_final = random.Random(final_seed).sample(range(2048), 512)
    expected_posterior_generator = torch.Generator().manual_seed(posterior_seed)
    expected_posterior = torch.randn(16, generator=expected_posterior_generator)

    controller_generator = random.Random(controller_seed)
    assert controller_generator.sample(range(2048), 128) == expected_controller
    for _ in range(100):
        controller_generator.random()

    assert random.Random(final_seed).sample(range(2048), 512) == expected_final
    posterior_generator = torch.Generator().manual_seed(posterior_seed)
    assert torch.equal(
        torch.randn(16, generator=posterior_generator),
        expected_posterior,
    )


def test_probabilistic_seed_provenance_is_in_comparison_signature():
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
    )
    config["model"]["granularity_sampling_mode"] = "adaptive_global"
    config["model"]["adaptive_sampler_strategy"] = "thompson"

    signature, inputs = build_comparison_control_signature(config)

    expected_provenance = {
        stream_name: {
            "stream_name": stream_name,
            "seed_stream_version": config["run"]["reproducibility"][
                "seed_stream_version"
            ],
            "resolved_seed": seed_for(config, stream_name),
        }
        for stream_name in PROBABILISTIC_SEED_STREAMS
    }
    assert inputs["probabilistic_seed_streams"] == expected_provenance
    assert len(signature) == 64


def test_config_migrates_legacy_validation_and_emits_only_canonical_fields():
    resolved = resolve_run_config(
        "configs/debug_matrix.yaml", run_id="debug-nested-001"
    )
    assert resolved["evaluation"]["validation"]["enabled"] is True
    assert resolved["evaluation"]["validation"]["interval_steps"] == 2
    assert resolved["evaluation"]["validation"]["holdout"]["examples"] == 2
    assert resolved["evaluation"]["test"] == {"enabled": False}
    assert "eval_interval" not in resolved["training"]
    assert "eval_batches" not in resolved["training"]
    assert "final_validation" not in resolved["evaluation"]


def test_canonical_and_legacy_validation_conflict_is_rejected():
    with pytest.raises(ConfigError, match="Conflicting training.eval_interval"):
        resolve_run_config(
            "configs/dmodel256_pilot_comparison.yaml",
            overrides=["training.eval_interval=1"],
        )


def test_seed_is_required(tmp_path):
    raw = yaml.safe_load(open("configs/debug_matrix.yaml", encoding="utf-8"))
    raw["run"].pop("seed")
    config_path = tmp_path / "missing-seed.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="explicit nonnegative integer"):
        resolve_run_config(config_path, run_id="debug-nested-001")


def _tokenized_dataset(size=520):
    return Dataset.from_dict(
        {
            "input_ids": [[index + 1, index + 2, 0] for index in range(size)],
            "attention_mask": [[1, 1, 0] for _ in range(size)],
        }
    )


def test_exact_disjoint_512_example_holdout_and_model_independent_train_order(tmp_path):
    first = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        output_dir=tmp_path / "dmodel256-pilot-comparison-001",
    )
    second = copy.deepcopy(first)
    second["model"]["d_model"] = 512
    dataset = _tokenized_dataset()

    first_train, first_validation = build_dataloaders(first, dataset, torch.device("cpu"))
    second_train, second_validation = build_dataloaders(
        second, dataset, torch.device("cpu")
    )

    assert len(first_validation.dataset) == 512
    assert len(first_train.dataset) == 8
    validation_indices = first["_validation_manifest"]["validation_indices"]
    assert len(validation_indices) == len(set(validation_indices)) == 512
    assert validation_indices == second["_validation_manifest"]["validation_indices"]
    assert list(first_train.sampler) == list(second_train.sampler)


def test_padding_labels_are_ignored_after_causal_shift():
    batch = collate_language_model_batch(
        [{"input_ids": [7, 8, 0], "attention_mask": [1, 1, 0]}]
    )
    assert batch["labels"].tolist() == [[7, 8, -100]]


class SequencedLossModel(torch.nn.Module):
    def __init__(self, losses):
        super().__init__()
        self.losses = iter(losses)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return SimpleNamespace(loss=torch.tensor(next(self.losses)))


def test_validation_uses_target_weighted_loss_and_reports_skipped_batches():
    examples = [
        {"input_ids": [1, 0, 0], "attention_mask": [1, 0, 0]},
        {"input_ids": [1, 2, 0], "attention_mask": [1, 1, 0]},
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
    ]
    dataloader = build_language_model_dataloader(examples, batch_size=1)
    result = evaluate_validation_loss(
        SequencedLossModel([1.0, 3.0]), dataloader, "cpu"
    )
    assert result["loss"] == pytest.approx((1.0 * 1 + 3.0 * 2) / 3)
    assert result["evaluation_examples"] == 3
    assert result["evaluation_batches"] == 3
    assert result["evaluation_target_tokens"] == 3
    assert result["evaluation_skipped_batches"] == 1


def test_validation_fails_when_complete_holdout_has_no_targets():
    dataloader = build_language_model_dataloader(
        [{"input_ids": [1, 0], "attention_mask": [1, 0]}], batch_size=1
    )
    with pytest.raises(ValueError, match="zero valid causal"):
        evaluate_validation_loss(SequencedLossModel([]), dataloader, "cpu")


def test_distributed_validation_partitions_without_duplicates():
    dataset = list(range(11))
    partitions = [
        list(DistributedValidationSampler(dataset, rank, 3)) for rank in range(3)
    ]
    flattened = [index for partition in partitions for index in partition]
    assert sorted(flattened) == list(range(11))
    assert len(flattened) == len(set(flattened))


def test_scaling_rows_label_final_best_and_trailing_statistics(tmp_path):
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=tmp_path / "debug-nested-001",
    )
    config["validation_manifest_hash"] = "manifest"
    config["comparison_control_signature"] = "controls"
    rows = []
    for granularity in config["model"]["granularities"]:
        for step, loss in enumerate([4.0, 2.0, 3.0], start=1):
            rows.append(
                {
                    "split": "validation",
                    "step": step,
                    "granularity": granularity,
                    "loss": loss,
                    "perplexity": loss + 1,
                    "evaluation_target_tokens": 17,
                }
            )
    counts = {
        granularity: {
            "total_parameters": 10,
            "embedding_parameters": 1,
            "lm_head_parameters": 1,
            "non_embedding_parameters": 8,
        }
        for granularity in config["model"]["granularities"]
    }
    result = build_scaling_result_rows(config, rows, counts)[0]
    assert result["final_validation_loss"] == 3.0
    assert result["best_validation_loss"] == 2.0
    assert result["best_validation_step"] == 2
    assert result["trailing_validation_mean"] == 3.0
    assert result["trailing_validation_count"] == 3
    assert result["final_minus_best_loss"] == 1.0
    assert result["evaluation_target_tokens"] == 17


def test_corrected_run_rejects_checkpoint_without_reproducibility_payload(tmp_path):
    config = resolve_run_config(
        "configs/debug_matrix.yaml", run_id="debug-nested-001"
    )
    config["validation_manifest_hash"] = "manifest"
    with pytest.raises(ConfigError, match="lacks the reproducibility payload"):
        _validate_reproducibility_payload(
            {}, config=config, checkpoint_path=tmp_path / "legacy.pt"
        )


class ReplicationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(()))
        self.current_granularity = None

    def configure_subnetwork(self, granularity):
        self.current_granularity = granularity

    def forward(self, input_ids, attention_mask=None, labels=None):
        return SimpleNamespace(loss=self.weight.square() + input_ids.float().mean() * 0)


def _assert_nested_state_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if torch.is_tensor(left[key]):
            assert torch.equal(left[key], right[key])
        elif isinstance(left[key], dict):
            _assert_nested_state_equal(left[key], right[key])
        elif isinstance(left[key], list):
            assert len(left[key]) == len(right[key])
            for left_item, right_item in zip(left[key], right[key]):
                if isinstance(left_item, dict):
                    _assert_nested_state_equal(left_item, right_item)
                elif torch.is_tensor(left_item):
                    assert torch.equal(left_item, right_item)
                else:
                    assert left_item == right_item
        else:
            assert left[key] == right[key]


def test_two_same_seed_cpu_runs_match_metrics_weights_optimizer_and_scheduler(tmp_path):
    outcomes = []
    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "attention_mask": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        }
    )
    for replica in ("a", "b"):
        output_dir = tmp_path / replica / "debug-nested-001"
        config = resolve_run_config(
            "configs/debug_matrix.yaml",
            run_id="debug-nested-001",
            output_dir=output_dir,
            overrides=[
                "training.max_steps=2",
                "training.eval_interval=0",
                "training.batch_size_per_process=1",
                "training.learning_rate=0.01",
                "training.scheduler.kwargs.warmup_steps=0",
                "evaluation.validation=false",
                "evaluation.final_validation=false",
                "outputs.save_checkpoints=true",
            ],
        )
        seed_model_initialization(config)
        model = ReplicationModel()
        result = run_training(
            config,
            model=model,
            tokenized_dataset=dataset,
            device="cpu",
        )
        checkpoint = torch.load(
            output_dir / "checkpoints" / "final.pt", map_location="cpu"
        )
        stable_metrics = [
            {
                key: row.get(key)
                for key in (
                    "step",
                    "split",
                    "granularity",
                    "loss",
                    "perplexity",
                    "tokens_seen",
                    "content_tokens_seen",
                )
            }
            for row in result["metrics_rows"]
        ]
        outcomes.append((stable_metrics, checkpoint))

    assert outcomes[0][0] == outcomes[1][0]
    for state_name in (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
    ):
        _assert_nested_state_equal(outcomes[0][1][state_name], outcomes[1][1][state_name])
