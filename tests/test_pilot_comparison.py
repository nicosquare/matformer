from pathlib import Path


def _pilot_comparison_builder():
    import utils.metrics as metrics

    builder = getattr(metrics, "build_pilot_comparison_rows", None)
    assert builder is not None, (
        "utils.metrics.build_pilot_comparison_rows is required"
    )
    return builder


def _pilot_summary(
    run_id,
    model_family,
    sampling_mode,
    *,
    granularity=None,
    checkpoint_status="best_eval",
    checkpoint_path="/tmp/pilot/checkpoints/best_eval_step_10.pt",
    effective_world_size=2,
):
    return {
        "run_id": run_id,
        "phase_id": "dmodel256_pilot_comparison",
        "model_family": model_family,
        "sampling_mode": sampling_mode,
        "model_shape_label": "dmodel256",
        "completion_label": "run",
        "model_family_slug": "matformer_llama",
        "model_size_slug": "148m",
        "token_budget_slug": "100m_tokens",
        "output_group": "matformer_llama_148m_100m_tokens",
        "granularity": granularity,
        "granularities": (
            ["s", "m", "l", "xl"]
            if model_family == "nested"
            else [granularity]
        ),
        "d_model": 256,
        "num_layers": 16,
        "num_attention_heads": 16,
        "context_length": 1024,
        "vocab_size_assumption": 256000,
        "token_budget": 100_000_000,
        "effective_world_size": effective_world_size,
        "checkpoint_status": checkpoint_status,
        "best_checkpoint_path": (
            checkpoint_path if checkpoint_status == "best_eval" else None
        ),
        "final_checkpoint_path": (
            checkpoint_path if checkpoint_status == "final" else None
        ),
        "checkpoint_metric": "validation_loss" if checkpoint_status == "best_eval" else None,
        "parameter_counts_by_granularity": {
            "s": {
                "total_parameters": 12,
                "embedding_parameters": 2,
                "lm_head_parameters": 2,
                "non_embedding_parameters": 8,
                "ffn_parameters": 4,
                "attention_parameters": 2,
                "other_non_embedding_parameters": 2,
                "lm_head_counting": "separately_counted",
            },
            "m": {
                "total_parameters": 16,
                "embedding_parameters": 2,
                "lm_head_parameters": 2,
                "non_embedding_parameters": 12,
                "ffn_parameters": 8,
                "attention_parameters": 2,
                "other_non_embedding_parameters": 2,
                "lm_head_counting": "separately_counted",
            },
            "l": {
                "total_parameters": 24,
                "embedding_parameters": 2,
                "lm_head_parameters": 2,
                "non_embedding_parameters": 20,
                "ffn_parameters": 16,
                "attention_parameters": 2,
                "other_non_embedding_parameters": 2,
                "lm_head_counting": "separately_counted",
            },
            "xl": {
                "total_parameters": 40,
                "embedding_parameters": 2,
                "lm_head_parameters": 2,
                "non_embedding_parameters": 36,
                "ffn_parameters": 32,
                "attention_parameters": 2,
                "other_non_embedding_parameters": 2,
                "lm_head_counting": "separately_counted",
            },
        },
    }


def test_pilot_comparison_rows_cover_nested_and_standalone_modes(tmp_path):
    rows = _pilot_comparison_builder()(
        comparison_id="dmodel256-pilot-comparison-001",
        run_summaries=[
            _pilot_summary(
                "dmodel256-nested-random-001",
                "nested",
                "nested-random",
            ),
            _pilot_summary(
                "dmodel256-nested-all-001",
                "nested",
                "nested-all",
            ),
            _pilot_summary(
                "dmodel256-standalone-s-001",
                "standalone",
                "standalone",
                granularity="s",
                checkpoint_status="final",
                checkpoint_path=str(
                    tmp_path / "standalone-s" / "checkpoints" / "final.pt"
                ),
            ),
        ],
        omitted_rows=[
            {
                "run_id": "dmodel256-standalone-m-001",
                "model_family": "standalone",
                "sampling_mode": "standalone",
                "granularity": "m",
                "token_budget": 100_000_000,
                "omit_reason": "not scheduled for capped pilot smoke",
            }
        ],
    )

    by_key = {
        (row["run_id"], row["sampling_mode"], row["granularity"]): row
        for row in rows
    }

    assert (
        "dmodel256-nested-random-001",
        "nested-random",
        "xl",
    ) in by_key
    assert (
        "dmodel256-nested-all-001",
        "nested-all",
        "s",
    ) in by_key
    standalone = by_key[
        ("dmodel256-standalone-s-001", "standalone", "s")
    ]
    omitted = by_key[
        ("dmodel256-standalone-m-001", "standalone", "m")
    ]

    for row in rows:
        assert row["comparison_id"] == "dmodel256-pilot-comparison-001"
        assert row["model_shape_label"] == "dmodel256"
        assert row["completion_label"] == "run"
        assert row["token_budget"] == 100_000_000
        if row["run_status"] == "omitted":
            assert row["model_family_slug"] is None
            assert row["output_group"] is None
        else:
            assert row["model_family_slug"] == "matformer_llama"
            assert row["output_group"] == "matformer_llama_148m_100m_tokens"

    assert standalone["run_status"] == "completed"
    assert standalone["model_family"] == "standalone"
    assert standalone["sampling_mode"] == "standalone"
    assert standalone["checkpoint_status"] == "final"
    assert standalone["checkpoint_path"] == str(
        Path(tmp_path) / "standalone-s" / "checkpoints" / "final.pt"
    )
    assert standalone["effective_world_size"] == 2

    assert omitted["run_status"] == "omitted"
    assert omitted["omit_reason"] == "not scheduled for capped pilot smoke"
    assert omitted["model_family"] == "standalone"
    assert omitted["granularity"] == "m"
    assert omitted["sampling_mode"] == "standalone"
    assert omitted["effective_world_size"] is None
    assert omitted["checkpoint_status"] == "unavailable"
    assert omitted["checkpoint_path"] is None
