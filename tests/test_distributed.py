import torch

from training.distributed import prepare_distributed_context


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
