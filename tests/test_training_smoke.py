import csv
from types import SimpleNamespace

import torch
from datasets import Dataset

from training.run import run_training
from utils.config import resolve_run_config


class TinyNestedTrainingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.current_granularity = None
        self.train_forward_granularities = []

    def configure_subnetwork(self, granularity):
        self.current_granularity = granularity

    def forward(self, input_ids, attention_mask=None, labels=None):
        if self.training:
            self.train_forward_granularities.append(self.current_granularity)

        loss = self.weight.pow(2) + input_ids.float().mean() * 0.0
        return SimpleNamespace(loss=loss)


def test_tiny_nested_training_accumulates_all_granularities_per_batch(tmp_path):
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "training.max_steps=1",
            "training.eval_interval=0",
            "training.batch_size_per_process=1",
            "training.learning_rate=0.01",
            "training.warmup_steps=0",
        ],
    )
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 0], [3, 4, 5]],
            "attention_mask": [[1, 1, 0], [1, 1, 1]],
        }
    )
    model = TinyNestedTrainingModel()

    result = run_training(
        config,
        model=model,
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    assert model.train_forward_granularities == ["s", "m", "l", "xl"]
    assert result["metrics_path"] == output_dir / "metrics.csv"

    with result["metrics_path"].open("r", encoding="utf-8", newline="") as metrics_file:
        train_rows = [
            row
            for row in csv.DictReader(metrics_file)
            if row["split"] == "train" and row["step"] == "1"
        ]

    assert [row["granularity"] for row in train_rows] == ["s", "m", "l", "xl"]
