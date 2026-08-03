from src.utils.monitoring import group_loss_rows_by_series


def test_group_loss_rows_by_series_uses_run_local_granularity_labels():
    nested_rows = [
        {"split": "train", "granularity": "s", "metric_name": "train_loss"},
        {"split": "train", "granularity": "m", "metric_name": "validation_loss"},
        {"split": "validation", "granularity": "xl", "metric_name": "loss"},
    ]
    standalone_rows = [
        {"split": "train", "granularity": "s", "metric_name": "loss"},
    ]

    nested_grouped = group_loss_rows_by_series(nested_rows)
    standalone_grouped = group_loss_rows_by_series(standalone_rows)

    assert set(nested_grouped) == {
        "train/loss/s",
        "train/loss/m",
        "validation/loss/xl",
    }
    assert nested_grouped["train/loss/s"][0]["metric_name"] == "train_loss"
    assert nested_grouped["train/loss/m"][0]["metric_name"] == "validation_loss"
    assert nested_grouped["validation/loss/xl"][0]["granularity"] == "xl"
    assert set(standalone_grouped) == {"train/loss/s"}
    assert standalone_grouped["train/loss/s"][0]["granularity"] == "s"


def test_per_block_rows_use_one_stable_series_without_parsing_pattern_labels():
    rows = [
        {
            "step": 1,
            "split": "train",
            "granularity_sampling_mode": "per_block",
            "granularity": "small,large,small,medium",
            "loss": 2.0,
        },
        {
            "step": 2,
            "split": "train",
            "granularity_sampling_mode": "per_block",
            "granularity": "large,small,medium,small",
            "loss": 1.5,
        },
        {
            "step": 2,
            "split": "train",
            "granularity_sampling_mode": "adaptive_per_block",
            "granularity": "small,small,large,medium",
            "loss": 1.4,
        },
    ]

    grouped = group_loss_rows_by_series(rows)

    assert list(grouped) == [
        "train/loss/per_block",
        "train/loss/adaptive_per_block",
    ]
    assert [row["granularity"] for row in grouped["train/loss/per_block"]] == [
        "small,large,small,medium",
        "large,small,medium,small",
    ]
