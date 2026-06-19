import torch

import src.training.distributed as distributed
from src.training.distributed import (
    DistributedContext,
    broadcast_object,
    destroy_distributed_process_group,
    prepare_distributed_context,
    sum_int,
)


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

    import src.training.run as training_run

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

    state_dict = training_run.checkpoint_state_dict(model, context)

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
