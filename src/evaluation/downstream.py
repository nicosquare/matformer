"""Downstream evaluation adapter built around EleutherAI lm-eval."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from src.utils.config import resolve_run_config
from src.utils.metrics import write_task_results_csv


MINIMAL_DOWNSTREAM_SUITE_ID = "minimal-downstream"
MINIMAL_DOWNSTREAM_TASKS = [
    "hellaswag",
    "piqa",
    "arc_challenge",
    "boolq",
    "winogrande",
    "openbookqa",
]
DEFAULT_LM_EVAL_METRIC_CANDIDATES = [
    "acc_norm,none",
    "acc,none",
    "exact_match,none",
    "acc_norm",
    "acc",
    "accuracy",
]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = resolve_run_config(
        args.config,
        run_id=args.run_id,
        output_dir=args.output_dir,
    )
    suite = downstream_suite_from_config(config)

    if args.results_json:
        with Path(args.results_json).open("r", encoding="utf-8") as results_file:
            lm_eval_results = json.load(results_file)
        rows = lm_eval_results_to_task_rows(
            lm_eval_results,
            config,
            suite=suite,
            granularity=args.granularity,
        )
    else:
        output_path = Path(config["run"]["output_dir"]) / "lm_eval_results.json"
        rows = run_lm_eval_downstream(
            config,
            checkpoint_path=args.checkpoint_path,
            output_path=output_path,
            suite=suite,
            granularity=args.granularity,
        )

    task_results_path = write_task_results_csv(config["run"]["output_dir"], rows)
    if task_results_path is not None:
        print(task_results_path)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--granularity")
    parser.add_argument(
        "--results-json",
        help="Convert an existing lm-eval JSON output instead of launching lm_eval.",
    )
    return parser.parse_args(argv)


def downstream_suite_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    suite_config = config.get("evaluation", {}).get("downstream_suite")

    if not suite_config:
        return default_downstream_suite()
    if isinstance(suite_config, list):
        if not suite_config:
            return default_downstream_suite()
        first_suite = suite_config[0]
        if not isinstance(first_suite, Mapping):
            raise ValueError("evaluation.downstream_suite entries must be mappings")
        return normalize_downstream_suite(first_suite)
    if isinstance(suite_config, Mapping):
        return normalize_downstream_suite(suite_config)

    raise ValueError("evaluation.downstream_suite must be a mapping or list")


def default_downstream_suite() -> dict[str, Any]:
    return normalize_downstream_suite(
        {
            "suite_id": MINIMAL_DOWNSTREAM_SUITE_ID,
            "suite_type": "downstream",
            "tool": "lm-eval",
            "tasks": MINIMAL_DOWNSTREAM_TASKS,
            "metric_name": "accuracy",
            "metric_candidates": DEFAULT_LM_EVAL_METRIC_CANDIDATES,
            "model": "hf",
            "model_args": {},
            "batch_size": "auto",
            "num_fewshot": 0,
            "limit": None,
        }
    )


def normalize_downstream_suite(suite: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(suite)
    normalized.setdefault("suite_id", MINIMAL_DOWNSTREAM_SUITE_ID)
    normalized.setdefault("suite_type", "downstream")
    normalized.setdefault("tool", "lm-eval")
    normalized.setdefault("tasks", MINIMAL_DOWNSTREAM_TASKS)
    normalized.setdefault("metric_name", "accuracy")
    normalized.setdefault("metric_candidates", DEFAULT_LM_EVAL_METRIC_CANDIDATES)
    normalized.setdefault("model", "hf")
    normalized.setdefault("model_args", {})
    normalized.setdefault("batch_size", "auto")
    normalized.setdefault("num_fewshot", 0)
    normalized.setdefault("limit", None)

    tasks = normalized["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("downstream suite tasks must be a non-empty list")
    if normalized.get("tool") != "lm-eval":
        raise ValueError("downstream suite tool must be lm-eval")

    return normalized


def build_lm_eval_command(
    config: Mapping[str, Any],
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
    suite: Mapping[str, Any] | None = None,
    executable: str = "lm_eval",
) -> list[str]:
    suite = normalize_downstream_suite(suite or downstream_suite_from_config(config))
    command = [
        executable,
        "--model",
        str(suite["model"]),
        "--model_args",
        build_lm_eval_model_args(config, suite=suite, checkpoint_path=checkpoint_path),
        "--tasks",
        ",".join(str(task) for task in suite["tasks"]),
        "--batch_size",
        str(suite["batch_size"]),
        "--num_fewshot",
        str(suite["num_fewshot"]),
    ]

    if suite.get("limit") is not None:
        command.extend(["--limit", str(suite["limit"])])
    if suite.get("device"):
        command.extend(["--device", str(suite["device"])])
    if output_path is not None:
        command.extend(["--output_path", str(output_path)])

    return command


def build_lm_eval_model_args(
    config: Mapping[str, Any],
    suite: Mapping[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
) -> str:
    suite = normalize_downstream_suite(suite or downstream_suite_from_config(config))
    model_args = dict(suite.get("model_args") or {})
    model = config["model"]

    if checkpoint_path is not None:
        model_args.setdefault("pretrained", str(checkpoint_path))
    else:
        model_args.setdefault(
            "pretrained",
            model.get("base_model_name") or config["run"]["run_id"],
        )
    tokenizer_name = model.get("tokenizer_name") or model.get("base_model_name")
    if tokenizer_name:
        model_args.setdefault("tokenizer", tokenizer_name)

    return ",".join(f"{key}={value}" for key, value in model_args.items())


def run_lm_eval_downstream(
    config: Mapping[str, Any],
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
    suite: Mapping[str, Any] | None = None,
    granularity: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> list[dict[str, Any]]:
    suite = normalize_downstream_suite(suite or downstream_suite_from_config(config))
    command = build_lm_eval_command(
        config,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        suite=suite,
    )
    completed = runner(command, check=True, capture_output=True, text=True)

    lm_eval_results = None
    if output_path is not None and Path(output_path).exists():
        with Path(output_path).open("r", encoding="utf-8") as results_file:
            lm_eval_results = json.load(results_file)
    elif completed.stdout:
        lm_eval_results = json.loads(completed.stdout)

    if lm_eval_results is None:
        raise ValueError("lm_eval did not produce JSON results")

    return lm_eval_results_to_task_rows(
        lm_eval_results,
        config,
        suite=suite,
        granularity=granularity,
    )


def lm_eval_results_to_task_rows(
    lm_eval_results: Mapping[str, Any],
    config: Mapping[str, Any],
    suite: Mapping[str, Any] | None = None,
    granularity: str | None = None,
) -> list[dict[str, Any]]:
    suite = normalize_downstream_suite(suite or downstream_suite_from_config(config))
    results = lm_eval_results.get("results", lm_eval_results)
    if not isinstance(results, Mapping):
        raise ValueError("lm_eval results must contain a results mapping")

    run = config["run"]
    rows = []
    for task_name in suite["tasks"]:
        task_results = results.get(task_name)
        if not isinstance(task_results, Mapping):
            raise ValueError(f"lm_eval results missing task: {task_name}")
        metric_value = select_lm_eval_metric(
            task_results,
            suite["metric_candidates"],
        )
        rows.append(
            {
                "run_id": run["run_id"],
                "suite_id": suite["suite_id"],
                "task": task_name,
                "model_family": run["model_family"],
                "model_size_label": _model_shape_label(run),
                "model_shape_label": _model_shape_label(run),
                "sampling_mode": _sampling_mode(run, config.get("training", {})),
                "model_family_slug": run.get("model_family_slug"),
                "model_size_slug": run.get("model_size_slug"),
                "token_budget_slug": run.get("token_budget_slug"),
                "output_group": run.get("output_group"),
                "granularity": granularity or run.get("granularity"),
                "metric_name": suite["metric_name"],
                "metric_value": metric_value,
            }
        )

    return rows


def select_lm_eval_metric(
    task_results: Mapping[str, Any],
    metric_candidates: Iterable[str],
) -> float:
    for metric_name in metric_candidates:
        if metric_name in task_results:
            return float(task_results[metric_name])

    for value in task_results.values():
        if isinstance(value, (int, float)):
            return float(value)

    raise ValueError("lm_eval task result does not include a numeric metric")


def _model_shape_label(run: Mapping[str, Any]) -> Any:
    return run.get("model_shape_label", run.get("model_size_label"))


def _sampling_mode(run: Mapping[str, Any], training: Mapping[str, Any]) -> Any:
    if run.get("sampling_mode") is not None:
        return run["sampling_mode"]
    if run.get("model_family") == "standalone":
        return "standalone"

    granularity_sampling = training.get("granularity_sampling")
    if granularity_sampling == "random":
        return "nested-random"
    if granularity_sampling == "all":
        return "nested-all"
    return granularity_sampling


if __name__ == "__main__":
    main()
