import torch
import pytest
from contextlib import nullcontext

import src.training.distributed as distributed
from src.training.distributed import (
    DistributedContext,
    broadcast_object,
    destroy_distributed_process_group,
    prepare_distributed_context,
    sum_int,
)
from src.utils.config import ConfigError


def test_cpu_resolves_requested_bf16_and_checkpointing_to_none():
    config = {
        "training": {
            "mixed_precision": "bf16",
            "activation_checkpointing": True,
        }
    }

    context = prepare_distributed_context(config, device="cpu")

    assert context.mixed_precision == "none"
    assert context.activation_checkpointing is False
    assert config["training"]["requested_mixed_precision"] == "bf16"
    assert config["training"]["resolved_mixed_precision"] == "none"
    assert config["training"]["requested_activation_checkpointing"] is True
    assert config["training"]["resolved_activation_checkpointing"] is False


def test_single_process_fp16_is_rejected_explicitly():
    with pytest.raises(ConfigError, match="fp16 is unsupported for single-process"):
        distributed.resolve_runtime_settings(
            {"mixed_precision": "fp16"},
            "cuda:0",
            single_process=True,
        )


def test_bf16_requires_native_cuda_support(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    with pytest.raises(ConfigError, match="native CUDA BF16 support"):
        distributed.resolve_runtime_settings(
            {"mixed_precision": "bf16"},
            "cuda:0",
            single_process=True,
        )


def test_activation_checkpointing_applies_without_fsdp(monkeypatch):
    calls = []
    monkeypatch.setattr(
        distributed,
        "apply_model_activation_checkpointing",
        lambda model: calls.append(model),
    )
    model = torch.nn.Linear(2, 2)
    context = DistributedContext(
        enabled=False,
        world_size=1,
        device="cuda:0",
        activation_checkpointing=True,
    )

    assert distributed.wrap_model_for_distributed(model, context) is model
    assert calls == [model]


def test_cuda_bf16_autocast_uses_resolved_runtime_setting(monkeypatch):
    calls = []

    def fake_autocast(*, device_type, dtype):
        calls.append((device_type, dtype))
        return nullcontext()

    monkeypatch.setattr(torch, "autocast", fake_autocast)

    with distributed.autocast_context(
        {"training": {"resolved_mixed_precision": "bf16"}},
        "cuda:0",
    ):
        pass

    assert calls == [("cuda", torch.bfloat16)]


def test_prepare_distributed_context_initializes_nccl_with_local_cuda_device(
    monkeypatch,
):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    set_device_calls = []
    init_calls = []
    monkeypatch.setattr(torch.cuda, "set_device", set_device_calls.append)
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda **kwargs: init_calls.append(kwargs),
    )

    context = prepare_distributed_context(
        {"training": {"distributed": {"enabled": True}}},
    )

    expected_device = torch.device("cuda", 1)
    assert set_device_calls == [expected_device]
    assert init_calls == [{"backend": "nccl", "device_id": expected_device}]
    assert context.enabled is True
    assert context.rank == 1
    assert context.local_rank == 1
    assert context.world_size == 2
    assert context.device == expected_device


def test_prepare_distributed_context_uses_gloo_without_device_id_on_cpu(
    monkeypatch,
):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    init_calls = []
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda **kwargs: init_calls.append(kwargs),
    )

    context = prepare_distributed_context(
        {"training": {"distributed": {"enabled": True}}},
    )

    assert init_calls == [{"backend": "gloo"}]
    assert context.enabled is True
    assert context.device == torch.device("cpu")


def test_prepare_distributed_context_ignores_config_enabled_flag(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    init_calls = []
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda **kwargs: init_calls.append(kwargs),
    )

    context = prepare_distributed_context(
        {
            "training": {
                "distributed": {
                    "enabled": False,
                    "strategy": "fsdp",
                }
            }
        },
    )

    assert init_calls == [{"backend": "gloo"}]
    assert context.enabled is True
    assert context.strategy == "fsdp"


def test_broadcast_object_receives_rank_zero_payload(monkeypatch):
    context = DistributedContext(enabled=True, rank=1, world_size=2)

    monkeypatch.setattr(distributed, "distributed_is_initialized", lambda: True)
    monkeypatch.setattr(distributed, "get_rank", lambda default=0: 1)

    def fake_broadcast_object_list(object_list, src):
        assert src == 0
        assert object_list == [None]
        object_list[0] = {"checkpoint_status": "best_eval"}

    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        fake_broadcast_object_list,
    )

    assert broadcast_object(None, context) == {"checkpoint_status": "best_eval"}


def test_destroy_distributed_process_group_when_initialized(monkeypatch):
    context = DistributedContext(enabled=True, rank=0, world_size=2)
    destroy_calls = []

    monkeypatch.setattr(distributed, "distributed_is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "destroy_process_group",
        lambda: destroy_calls.append("destroy"),
    )

    destroy_distributed_process_group(context)

    assert destroy_calls == ["destroy"]


def test_sum_int_all_reduces_across_ranks(monkeypatch):
    context = DistributedContext(enabled=True, rank=0, world_size=4)

    monkeypatch.setattr(distributed, "distributed_is_initialized", lambda: True)

    def fake_all_reduce(tensor, op):
        assert op == torch.distributed.ReduceOp.SUM
        tensor.fill_(12)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    assert sum_int(3, device="cpu", context=context) == 12


def test_checkpoint_state_dict_uses_distributed_checkpoint_api_for_fsdp(
    monkeypatch,
):
    from torch.distributed.checkpoint import state_dict as state_dict_module

    import src.training.checkpointing as training_checkpointing

    context = DistributedContext(
        enabled=True,
        rank=0,
        local_rank=0,
        world_size=2,
        strategy="fsdp",
        device="cpu",
    )
    model = torch.nn.Linear(1, 1)
    calls = []

    def fake_get_state_dict(model_arg, optimizers, *, options):
        calls.append(
            {
                "model": model_arg,
                "optimizers": optimizers,
                "options": options,
            }
        )
        return {"weight": torch.tensor([1.0])}, {}

    monkeypatch.setattr(state_dict_module, "get_state_dict", fake_get_state_dict)

    state_dict = training_checkpointing.checkpoint_state_dict(model, context)

    assert list(state_dict) == ["weight"]
    assert torch.equal(state_dict["weight"], torch.tensor([1.0]))
    assert calls[0]["model"] is model
    assert calls[0]["optimizers"] == []
    assert calls[0]["options"].full_state_dict is True
    assert calls[0]["options"].cpu_offload is True


def test_wrap_model_for_distributed_uses_hf_style_fsdp_recipe(monkeypatch):
    import src.training.distributed as distributed

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    context = DistributedContext(
        enabled=True,
        rank=0,
        local_rank=0,
        world_size=2,
        strategy="fsdp",
        device="cuda:0",
        mixed_precision="bf16",
        activation_checkpointing=True,
        fsdp_config={
            "sharding_strategy": "full_shard",
            "auto_wrap_policy": "transformer_based_wrap",
            "transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
            "backward_prefetch": "backward_pre",
            "use_orig_params": True,
            "sync_module_states": True,
            "forward_prefetch": False,
            "limit_all_gathers": False,
        },
    )

    model = torch.nn.Linear(2, 2)
    activation_calls = []
    fsdp_calls = []

    def fake_apply_activation_checkpointing(
        model_arg,
        *,
        checkpoint_wrapper_fn,
        check_fn,
    ):
        activation_calls.append(
            {
                "model": model_arg,
                "checkpoint_wrapper_fn": checkpoint_wrapper_fn,
                "check_fn": check_fn,
            }
        )

    def fake_fsdp(model_arg, **kwargs):
        fsdp_calls.append({"model": model_arg, "kwargs": kwargs})
        return model_arg

    monkeypatch.setattr(
        distributed,
        "apply_activation_checkpointing",
        fake_apply_activation_checkpointing,
    )
    monkeypatch.setattr(distributed, "FSDP", fake_fsdp)

    wrapped = distributed.wrap_model_for_distributed(model, context)

    assert wrapped is model
    assert len(activation_calls) == 1
    assert activation_calls[0]["model"] is model
    llama_layer = distributed.LlamaDecoderLayer.__new__(
        distributed.LlamaDecoderLayer
    )
    assert activation_calls[0]["check_fn"](llama_layer)
    assert len(fsdp_calls) == 1
    fsdp_kwargs = fsdp_calls[0]["kwargs"]
    assert fsdp_kwargs["device_id"] == torch.device("cuda:0")
    assert fsdp_kwargs["use_orig_params"] is True
    assert fsdp_kwargs["sync_module_states"] is True
    assert fsdp_kwargs["forward_prefetch"] is False
    assert fsdp_kwargs["limit_all_gathers"] is False
    assert fsdp_kwargs["sharding_strategy"].name == "FULL_SHARD"
    assert fsdp_kwargs["backward_prefetch"].name == "BACKWARD_PRE"
    assert fsdp_kwargs["mixed_precision"].param_dtype == torch.bfloat16
    assert (
        fsdp_kwargs["auto_wrap_policy"].keywords["transformer_layer_cls"]
        == {distributed.LlamaDecoderLayer}
    )


def test_probabilistic_controller_panel_partitions_and_reduces_like_single_process():
    from types import SimpleNamespace

    from datasets import Dataset

    from src.evaluation.validation import evaluate_fixed_panel_objective
    from src.training.data import (
        DistributedValidationSampler,
        build_language_model_dataloader,
    )

    class ControlledPanelModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.current_granularity = None

        def configure_subnetwork(self, granularity):
            self.current_granularity = granularity

        def forward(self, input_ids, attention_mask=None, labels=None):
            offsets = {"micro": 0.25, "medium": 0.5, "full": 1.0}
            example_values = input_ids[:, 0].double() / 100.0
            loss = example_values.mean() + offsets[self.current_granularity]
            return SimpleNamespace(loss=loss)

    panel = Dataset.from_dict(
        {
            "input_ids": [[index + 1, index + 2, index + 3] for index in range(128)],
            "attention_mask": [[1, 1, 1] for _ in range(128)],
        }
    )
    granularities = ["micro", "medium", "full"]
    single_process = evaluate_fixed_panel_objective(
        ControlledPanelModel(),
        build_language_model_dataloader(panel, batch_size=8),
        granularities,
        device="cpu",
        controller_manifest_hash="fixed-controller-panel",
        boundary_step=4,
    )

    world_size = 4
    samplers = [
        DistributedValidationSampler(panel, rank=rank, world_size=world_size)
        for rank in range(world_size)
    ]
    partitions = [list(sampler) for sampler in samplers]
    flattened_indices = [index for partition in partitions for index in partition]
    assert sorted(flattened_indices) == list(range(128))
    assert len(flattened_indices) == len(set(flattened_indices)) == 128

    local_results = [
        evaluate_fixed_panel_objective(
            ControlledPanelModel(),
            build_language_model_dataloader(
                panel,
                batch_size=8,
                sampler=sampler,
            ),
            granularities,
            device="cpu",
            controller_manifest_hash="fixed-controller-panel",
            boundary_step=4,
        )
        for sampler in samplers
    ]
    reduced_component_losses = []
    for component_index in range(len(granularities)):
        total_nll = sum(
            result["component_results"][component_index]["loss"]
            * result["component_results"][component_index][
                "evaluation_target_tokens"
            ]
            for result in local_results
        )
        total_targets = sum(
            result["component_results"][component_index][
                "evaluation_target_tokens"
            ]
            for result in local_results
        )
        assert total_targets == single_process["evaluation_target_tokens"] == 256
        reduced_component_losses.append(total_nll / total_targets)

    reduced_objective = sum(reduced_component_losses) / len(
        reduced_component_losses
    )
    assert single_process["evaluation_example_count"] == sum(
        result["evaluation_example_count"] for result in local_results
    )
    assert single_process["ordered_granularities"] == granularities
    assert reduced_component_losses == pytest.approx(
        single_process["ordered_component_losses"],
        rel=1e-6,
        abs=1e-8,
    )
    assert reduced_objective == pytest.approx(
        single_process["uniform_objective"],
        rel=1e-6,
        abs=1e-8,
    )


def test_probabilistic_controller_rank_zero_owns_sampling_update_and_shared_outputs(
    monkeypatch,
    tmp_path,
):
    rank_zero = DistributedContext(enabled=True, rank=0, world_size=2)
    nonzero = DistributedContext(enabled=True, rank=1, world_size=2)
    transition_calls = []
    lifecycle_logs = []
    artifact_writes = []

    def sample_and_update():
        transition_calls.append(0)
        return {
            "controller_state": {
                "method_version": 1,
                "scope": "global",
                "belief": {"round_index": 2},
                "sampling": {"sample_count": 3},
            },
            "action": {"global_granularity": "medium"},
        }

    authoritative_payload = distributed.rank_zero_only(rank_zero, sample_and_update)
    assert distributed.rank_zero_only(nonzero, sample_and_update) is None
    assert transition_calls == [0]

    broadcasts = []

    def fake_broadcast(value, context, src=0):
        assert src == 0
        broadcasts.append((context.rank, value))
        return authoritative_payload

    monkeypatch.setattr(distributed, "broadcast_object", fake_broadcast)

    rank_zero_payload = distributed.broadcast_probabilistic_controller_state(
        controller_state=authoritative_payload["controller_state"],
        action=authoritative_payload["action"],
        context=rank_zero,
    )
    nonzero_payload = distributed.broadcast_probabilistic_controller_state(
        controller_state=None,
        action=None,
        context=nonzero,
    )

    assert rank_zero_payload == nonzero_payload == authoritative_payload
    assert broadcasts == [(0, authoritative_payload), (1, None)]
    distributed.rank_zero_only(rank_zero, lifecycle_logs.append, "completed-window")
    distributed.rank_zero_only(nonzero, lifecycle_logs.append, "completed-window")
    distributed.rank_zero_only(rank_zero, artifact_writes.append, "controller.jsonl")
    distributed.rank_zero_only(nonzero, artifact_writes.append, "controller.jsonl")
    assert lifecycle_logs == ["completed-window"]
    assert artifact_writes == ["controller.jsonl"]

    from src.utils.metrics import append_controller_event

    warmup_event = {
        "schema_version": 1,
        "event_type": "warmup_window_completed",
        "phase": "warmup",
        "schedule_hash": "shared-balanced-schedule",
        "boundary_step": 2,
        "window_index": 0,
        "posterior_updated": False,
    }
    journal_path = tmp_path / "controller_metrics.jsonl"
    append_controller_event(
        journal_path,
        warmup_event,
        distributed_context=rank_zero,
    )
    append_controller_event(
        journal_path,
        warmup_event,
        distributed_context=nonzero,
    )
    assert len(journal_path.read_text(encoding="utf-8").splitlines()) == 1

    from src.utils.reproducibility import build_balanced_warmup_schedule

    rank_zero_schedule = build_balanced_warmup_schedule(
        ["micro", "medium", "full"],
        passes=2,
        seed=123,
        action_interval_steps=2,
        duration_steps=12,
    )
    nonzero_schedule = build_balanced_warmup_schedule(
        ["micro", "medium", "full"],
        passes=2,
        seed=123,
        action_interval_steps=2,
        duration_steps=12,
    )
    assert rank_zero_schedule == nonzero_schedule


def test_probabilistic_controller_rank_zero_broadcast_requires_complete_state():
    context = DistributedContext(enabled=True, rank=0, world_size=2)

    with pytest.raises(ConfigError, match="rank zero.*controller state"):
        distributed.broadcast_probabilistic_controller_state(
            controller_state=None,
            action={"global_granularity": "micro"},
            context=context,
        )
