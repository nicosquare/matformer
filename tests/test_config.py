import copy
import json
import math
import textwrap

import pytest

from utils.config import (
    ConfigError,
    resolve_all_run_configs,
    resolve_run_config,
    validate_run_config,
    write_resolved_config,
)
from models.correction import correction_context_from_config
from models.granularity import build_granularity_pattern


def _write_single_run_config(tmp_path):
    config_path = tmp_path / "single_run.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            run:
              run_id: single-output-root-001
              phase_id: debug_matrix
              model_family: nested
              model_size_label: debug
              seed: 42

            model:
              base_model_name: debug-llama
              d_model: 128
              num_layers: 2
              num_attention_heads: 4
              context_length: 64
              vocab_size_assumption: 32000
              granularities: [s, m, l, xl]

            training:
              token_budget: 8192
              max_steps: 1
              batch_size_per_process: 1
              learning_rate: 0.0003
              eval_interval: 0
              scheduler:
                name: cosine
                kwargs:
                  warmup_steps: 0

            dataset:
              dataset_name: roneneldan/TinyStories
              dataset_split: train
              dataset_phase: debug
              sample_limit: 2
              preprocessing_notes: debug_output_root_resolution

            outputs:
              save_config: true
              save_metrics_csv: true
              save_run_summary_json: true
              save_checkpoints: false
              make_plots: false

            evaluation:
              validation: true
              downstream_suite: []
              consistency: false
              speculative: false
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return config_path


def test_resolve_debug_matrix_expands_nested_and_standalone_runs():
    resolved_runs = resolve_all_run_configs("configs/debug_matrix.yaml")

    run_ids = [config["run"]["run_id"] for config in resolved_runs]
    assert run_ids == [
        "debug-nested-001",
        "debug-standalone-s-001",
        "debug-standalone-m-001",
        "debug-standalone-l-001",
        "debug-standalone-xl-001",
    ]

    nested = resolved_runs[0]
    assert nested["run"]["completion_label"] == "debug"
    assert nested["run"]["model_family_slug"] == "matformer_llama"
    assert nested["monitoring"]["project"] == "debug_matrix"
    assert nested["monitoring"]["job_type"] == "train"
    assert nested["monitoring"]["tags"] == ["debug", "matrix"]
    assert nested["monitoring"]["notes"] == "debug matrix smoke and warmup validation"
    assert nested["run"]["output_dir"] == (
        f"outputs/{nested['run']['output_group']}/debug-nested-001"
    )
    assert nested["model"]["d_model"] == 128
    assert nested["model"]["intermediate_size"] == 512
    assert nested["model"]["granularities"] == ["s", "m", "l", "xl"]
    assert nested["model"]["granularity_prefixes"] == {
        "s": 0.125,
        "m": 0.25,
        "l": 0.5,
        "xl": 1.0,
    }
    assert [entry["prefix_width"] for entry in nested["model"]["ffn_prefix_metadata"]] == [
        64,
        128,
        256,
        512,
    ]
    assert nested["training"]["granularity_sampling"] == "all"

    standalone_s = resolved_runs[1]
    assert standalone_s["run"]["model_family"] == "standalone"
    assert standalone_s["run"]["granularity"] == "s"
    assert standalone_s["run"]["completion_label"] == "debug"
    assert standalone_s["run"]["output_group"].startswith("matformer_llama_")
    assert standalone_s["model"]["granularities"] == ["s"]
    assert standalone_s["model"]["granularity_prefixes"] == {"s": 1.0}
    assert standalone_s["model"]["matformer_source_intermediate_size"] == 512
    assert [entry["prefix_width"] for entry in standalone_s["model"]["ffn_prefix_metadata"]] == [
        64,
    ]

    for resolved in resolved_runs:
        validate_run_config(resolved)


def test_cli_overrides_are_parsed_and_applied():
    resolved = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        overrides=[
            "training.max_steps=7",
            "run.seed=123",
            "outputs.save_checkpoints=false",
            "model.variant=concat",
            "model.correction_mode=none",
            "model.membership_correction=false",
            "training.scheduler.name=constant",
        ],
    )

    assert resolved["training"]["max_steps"] == 7
    assert resolved["run"]["seed"] == 123
    assert resolved["outputs"]["save_checkpoints"] is False
    assert resolved["model"]["variant"] == "concat"
    assert resolved["model"]["membership_correction"] is False
    assert resolved["training"]["scheduler_name"] == "constant"


def test_granularity_sampling_mode_validation():
    random_sampling = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        overrides=["training.granularity_sampling=random"],
    )
    assert random_sampling["training"]["granularity_sampling"] == "random"

    with pytest.raises(ConfigError, match="granularity_sampling"):
        resolve_run_config(
            "configs/debug_matrix.yaml",
            run_id="debug-nested-001",
            overrides=["training.granularity_sampling=cyclic"],
        )


def test_requested_run_sampling_mode_does_not_force_per_block_model_mode():
    resolved = resolve_run_config("configs/dmodel256_pilot_comparison.yaml")

    assert resolved["run"]["sampling_mode"] == "nested-random"
    assert resolved["training"]["granularity_sampling"] == "random"
    assert resolved["model"]["granularity_sampling_mode"] == "global"


def test_explicit_nested_random_mode_keeps_legacy_alias_stable_for_adaptive():
    resolved = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        overrides=[
            "run.sampling_mode=nested-random",
            "training.granularity_sampling=random",
            "model.granularity_sampling_mode=adaptive_per_block",
        ],
    )

    assert resolved["run"]["sampling_mode"] == "nested-random"
    assert resolved["training"]["granularity_sampling"] == "random"
    assert resolved["model"]["granularity_sampling_mode"] == "adaptive_per_block"
    assert resolved["model"]["requested_granularity_sampling_alias"] == "random"
    assert resolved["model"]["resolved_sampling_mode"] == "adaptive_per_block"
    assert resolved["model"]["granularity_pattern_provenance"] == {
        "pattern_type": "per_block",
        "scope": "model",
        "source": "model.granularity_sampling_mode",
        "requested_alias": "random",
        "layer_count": resolved["model"]["num_layers"],
        "available_granularities": ["s", "m", "l", "xl"],
    }


@pytest.mark.parametrize(
    "sampling_mode, expected_pattern_type, selected_granularities, expected_local_correction_active",
    [
        ("global", "single", ("m",), False),
        ("per_block", "per_block", ("s", "m"), True),
        ("adaptive_per_block", "per_block", ("s", "m"), True),
    ],
)
def test_explicit_model_sampling_modes_preserve_nested_random_run_mode(
    sampling_mode,
    expected_pattern_type,
    selected_granularities,
    expected_local_correction_active,
):
    resolved = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        overrides=[f"model.granularity_sampling_mode={sampling_mode}"],
    )

    assert resolved["run"]["sampling_mode"] == "nested-random"
    assert resolved["training"]["granularity_sampling"] == "random"
    assert resolved["model"]["granularity_sampling_mode"] == sampling_mode
    assert resolved["model"]["granularity_pattern_provenance"] == {
        "pattern_type": expected_pattern_type,
        "scope": "model",
        "source": "model.granularity_sampling_mode",
        "requested_alias": None,
        "layer_count": resolved["model"]["num_layers"],
        "available_granularities": ["s", "m", "l", "xl"],
    }

    runtime_pattern = build_granularity_pattern(
        pattern_type=expected_pattern_type,
        selected_granularities=selected_granularities,
        layer_count=resolved["model"]["num_layers"],
        repeatable_source=(
            "dmodel256-pilot-comparison-001",
            f"model.granularity_sampling_mode={sampling_mode}",
        ),
    )
    context = correction_context_from_config(
        resolved,
        granularity_pattern=runtime_pattern,
    )

    assert context.sampling_mode == sampling_mode
    assert context.local_correction_active is expected_local_correction_active
    if expected_local_correction_active:
        assert context.derived_membership_pattern == selected_granularities
    else:
        assert context.derived_membership_pattern == ()
    if sampling_mode == "adaptive_per_block":
        assert resolved["model"]["adaptive_sampler_strategy"] == "thompson"
        assert resolved["model"]["adaptive_sampler_exploration_scale"] == 1.0
        assert resolved["model"]["adaptive_sampler_decay_rate"] == 0.0
        assert resolved["model"]["adaptive_sampler_reward_penalty_weight"] == 1.0


def test_adaptive_sampler_controls_override_and_validate():
    resolved = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        overrides=[
            "model.granularity_sampling_mode=adaptive_per_block",
            "model.adaptive_sampler_strategy=ucb",
            "model.adaptive_sampler_exploration_scale=2.5",
            "model.adaptive_sampler_decay_rate=0.125",
            "model.adaptive_sampler_reward_penalty_weight=0.75",
        ],
    )

    assert resolved["run"]["sampling_mode"] == "nested-random"
    assert resolved["model"]["granularity_sampling_mode"] == "adaptive_per_block"
    assert resolved["model"]["adaptive_sampler_strategy"] == "ucb"
    assert resolved["model"]["adaptive_sampler_exploration_scale"] == 2.5
    assert resolved["model"]["adaptive_sampler_decay_rate"] == 0.125
    assert resolved["model"]["adaptive_sampler_reward_penalty_weight"] == 0.75


@pytest.mark.parametrize(
    "alias, expected_mode, expected_sampling_mode",
    [
        ("all", "global", "nested-all"),
        ("random", "per_block", "nested-random"),
    ],
)
def test_legacy_granularity_sampling_alias_resolves_to_canonical_model_mode(
    alias,
    expected_mode,
    expected_sampling_mode,
):
    resolved = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        overrides=[f"training.granularity_sampling={alias}"],
    )

    assert resolved["training"]["granularity_sampling"] == alias
    assert resolved["model"]["granularity_sampling_mode"] == expected_mode
    assert resolved["model"]["requested_granularity_sampling_alias"] == alias
    assert resolved["run"]["sampling_mode"] == expected_sampling_mode
    assert resolved["model"]["granularity_pattern_provenance"] == {
        "pattern_type": (
            "all_granularities" if expected_sampling_mode == "nested-all" else "per_block"
        ),
        "scope": "model",
        "source": "model.granularity_sampling_mode",
        "requested_alias": alias,
        "layer_count": resolved["model"]["num_layers"],
        "available_granularities": ["s", "m", "l", "xl"],
        "active_granularity": None,
    }


@pytest.mark.parametrize(
    "overrides, expected_active",
    [
        (["model.granularity_sampling_mode=global", "model.correction_mode=gmc"], False),
        (["model.granularity_sampling_mode=per_block", "model.correction_mode=gmc"], True),
        (
            [
                "model.granularity_sampling_mode=per_block",
                "model.correction_mode=none",
                "model.membership_correction=false",
            ],
            False,
        ),
    ],
)
def test_per_block_sampling_controls_local_correction_activation(
    overrides,
    expected_active,
):
    resolved = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        overrides=overrides,
    )

    pattern = build_granularity_pattern(
        pattern_type="per_block",
        selected_granularities=("s", "m", "l", "xl"),
        layer_count=resolved["model"]["num_layers"],
    )
    context = correction_context_from_config(resolved, granularity_pattern=pattern)

    assert context.sampling_mode == resolved["model"]["granularity_sampling_mode"]
    assert context.correction_mode == resolved["model"]["correction_mode"]
    assert context.local_correction_active is expected_active
    if expected_active:
        assert context.derived_membership_pattern == ("s", "m", "l", "xl")
    else:
        assert context.derived_membership_pattern == ()


def test_write_resolved_config(tmp_path):
    output_dir = tmp_path / "dmodel256-pilot-comparison-001"
    resolved = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        output_dir=output_dir,
    )

    config_path = write_resolved_config(resolved)

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["run"]["run_id"] == "dmodel256-pilot-comparison-001"
    assert saved["run"]["model_shape_label"] == "dmodel256"
    assert saved["run"]["sampling_mode"] == "nested-random"
    assert saved["run"]["completion_label"] == "run"
    assert saved["run"]["model_family_slug"] == "matformer_llama"
    assert saved["run"]["output_group"] == (
        f"matformer_llama_{saved['run']['model_size_slug']}"
        f"_{saved['run']['token_budget_slug']}"
    )
    assert saved["training"]["warmup_ratio"] == 0.01635
    assert saved["training"]["warmup_steps"] == 2000
    assert saved["training"]["resolved_warmup_steps"] == 2000
    assert saved["training"]["gradient_clip_norm"] == 1.0
    assert saved["training"]["scheduler"]["kwargs"]["warmup_steps"] == 2000
    assert saved["training"]["scheduler_name"] == "cosine"
    assert saved["training"]["scheduler"]["resolved_warmup_steps"] == 2000
    assert saved["run"]["continuation"] == {
        "enabled": True,
        "latest_checkpoint_save_interval_steps": 0,
        "latest_checkpoint_save_on_validation": True,
        "latest_checkpoint_save_on_completion": True,
    }
    assert saved["monitoring"]["project"] == "dmodel256_pilot_comparison"
    assert saved["monitoring"]["job_type"] == "train"
    assert saved["monitoring"]["tags"] == ["pilot", "dmodel256"]
    assert saved["monitoring"]["notes"] == "d_model=256 pilot comparison"
    assert saved["training"]["optimizer_name"] == "adamw"
    assert saved["training"]["optimizer_kwargs"] == {
        "betas": [0.9, 0.95],
        "eps": 1e-08,
        "weight_decay": 0.1,
    }
    assert config_path == output_dir / "config.json"


def test_resolve_minimal_config_includes_long_run_defaults(tmp_path):
    config_path = _write_single_run_config(tmp_path)
    output_dir = tmp_path / "single-output-root-001"

    resolved = resolve_run_config(config_path, output_dir=output_dir)

    assert resolved["model"]["d_model"] == 128
    assert resolved["run"]["continuation"] == {"enabled": False}
    assert resolved["monitoring"] == {
        "enabled": False,
        "backend": "wandb",
        "project": "debug_matrix",
        "entity": None,
        "group": resolved["run"]["output_group"],
        "job_type": "train",
        "name": "single-output-root-001",
        "tags": [],
        "notes": None,
        "mode": None,
        "log_loss_by_granularity": True,
        "log_validation_loss": True,
        "log_stage_events": True,
    }
    assert resolved["training"]["pre_nested_warmup"] == {
        "enabled": False,
        "duration": 0,
        "unit": "epochs",
        "active": False,
        "completed": False,
        "completion_step": None,
        "transition_reason": None,
    }


def test_pre_nested_warmup_validation_rules(tmp_path):
    config_path = _write_single_run_config(tmp_path)

    resolved = resolve_run_config(
        config_path,
        overrides=[
            "training.pre_nested_warmup.enabled=true",
            "training.pre_nested_warmup.duration=3",
            "training.pre_nested_warmup.unit=steps",
        ],
    )
    assert resolved["training"]["pre_nested_warmup"] == {
        "enabled": True,
        "duration": 3,
        "unit": "steps",
        "active": True,
        "completed": False,
        "completion_step": None,
        "transition_reason": None,
    }

    standalone_resolved = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-standalone-s-001",
        overrides=[
            "training.pre_nested_warmup.enabled=true",
            "training.pre_nested_warmup.duration=1",
        ],
    )
    assert standalone_resolved["training"]["pre_nested_warmup"]["active"] is False

    with pytest.raises(
        ConfigError,
        match="training.pre_nested_warmup.duration must be positive",
    ):
        resolve_run_config(
            config_path,
            overrides=[
                "training.pre_nested_warmup.enabled=true",
                "training.pre_nested_warmup.duration=0",
                "training.pre_nested_warmup.unit=steps",
            ],
        )

    with pytest.raises(
        ConfigError,
        match="training.pre_nested_warmup.unit must be one of",
    ):
        resolve_run_config(
            config_path,
            overrides=[
                "training.pre_nested_warmup.enabled=true",
                "training.pre_nested_warmup.duration=1",
                "training.pre_nested_warmup.unit=minutes",
            ],
        )


def test_dmodel256_completion_label_validation():
    resolved = resolve_run_config("configs/dmodel256_pilot_comparison.yaml")
    validate_run_config(resolved)

    mislabeled = copy.deepcopy(resolved)
    mislabeled["run"]["completion_label"] = "full-token-budget"

    with pytest.raises(ConfigError, match="Unknown completion label"):
        validate_run_config(mislabeled)

    validate_run_config(resolved)


def test_standalone_requires_one_granularity():
    resolved = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-standalone-s-001",
    )

    invalid = copy.deepcopy(resolved)
    invalid["model"]["granularities"] = ["s", "m"]

    with pytest.raises(ConfigError, match="exactly one"):
        validate_run_config(invalid)


def test_debug_matrix_resolves_all_standalone_granularities():
    expected = {
        "debug-standalone-s-001": ("s", 64),
        "debug-standalone-m-001": ("m", 128),
        "debug-standalone-l-001": ("l", 256),
        "debug-standalone-xl-001": ("xl", 512),
    }

    resolved_runs = resolve_all_run_configs("configs/debug_matrix.yaml")
    by_run_id = {config["run"]["run_id"]: config for config in resolved_runs}

    assert set(expected).issubset(by_run_id)
    for run_id, (granularity, intermediate_size) in expected.items():
        resolved = by_run_id[run_id]
        assert resolved["run"]["model_family"] == "standalone"
        assert resolved["run"]["sampling_mode"] == "standalone"
        assert resolved["run"]["resolved_run_mode"] == "standalone"
        assert resolved["run"]["granularity"] == granularity
        assert resolved["training"]["granularity_sampling"] == "all"
        assert resolved["model"]["granularity_sampling_mode"] == "global"
        assert resolved["model"]["resolved_sampling_mode"] == "global"
        assert resolved["model"]["granularities"] == [granularity]
        assert resolved["model"]["intermediate_size"] == intermediate_size
        assert resolved["model"]["matformer_source_intermediate_size"] == 512
        assert resolved["run"]["completion_label"] == "debug"
        assert resolved["run"]["output_dir"] == (
            f"outputs/{resolved['run']['output_group']}/{run_id}"
        )
        validate_run_config(resolved)


def test_debug_matrix_standalone_runs_share_family_folder_key():
    resolved_runs = {
        run_id: resolve_run_config("configs/debug_matrix.yaml", run_id=run_id)
        for run_id in [
            "debug-standalone-s-001",
            "debug-standalone-m-001",
            "debug-standalone-l-001",
        ]
    }

    output_groups = {resolved["run"]["output_group"] for resolved in resolved_runs.values()}
    family_size_slugs = {
        resolved["run"]["family_size_slug"] for resolved in resolved_runs.values()
    }

    assert len(output_groups) == 1
    assert len(family_size_slugs) == 1
    assert {
        resolved["run"]["active_size_label"]
        for resolved in resolved_runs.values()
    } == {"s", "m", "l"}
    for resolved in resolved_runs.values():
        assert (
            resolved["run"]["family_resolution_rule"]
            == "output_group is keyed from the largest configured family size"
        )
        assert resolved["run"]["output_group"] == next(iter(output_groups))
        assert resolved["run"]["family_size_slug"] == next(iter(family_size_slugs))
        validate_run_config(resolved)


def test_debug_standalone_granularity_must_match_model_granularities():
    resolved = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-standalone-m-001",
    )

    invalid = copy.deepcopy(resolved)
    invalid["model"]["granularities"] = ["s"]

    with pytest.raises(ConfigError, match="exactly one matching granularity"):
        validate_run_config(invalid)


@pytest.mark.parametrize(
    "overrides, error_message",
    [
        (
            ["training.granularity_sampling=random"],
            "model.granularity_sampling_mode=per_block requires nested runs",
        ),
        (
            ["model.granularity_sampling_mode=per_block"],
            "model.granularity_sampling_mode=per_block requires nested runs",
        ),
        (
            ["model.granularity_sampling_mode=adaptive_per_block"],
            "model.granularity_sampling_mode=adaptive_per_block requires nested-random runs",
        ),
    ],
)
def test_standalone_rejects_nested_sampling_submodes(overrides, error_message):
    with pytest.raises(ConfigError, match=error_message):
        resolve_run_config(
            "configs/debug_matrix.yaml",
            run_id="debug-standalone-m-001",
            overrides=overrides,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        [
            "run.sampling_mode=nested-all",
            "model.granularity_sampling_mode=adaptive_per_block",
        ],
        [
            "training.granularity_sampling=all",
            "model.granularity_sampling_mode=adaptive_per_block",
        ],
    ],
)
def test_adaptive_per_block_rejects_non_nested_random_pairings(overrides):
    with pytest.raises(
        ConfigError,
        match="model.granularity_sampling_mode=adaptive_per_block requires nested-random runs",
    ):
        resolve_run_config(
            "configs/dmodel256_pilot_comparison.yaml",
            overrides=overrides,
        )


def test_matrix_output_root_override_derives_each_run_directory(tmp_path):
    output_root = tmp_path / "matrix-output"

    resolved_runs = resolve_all_run_configs(
        "configs/debug_matrix.yaml",
        overrides=[f"run.output_root={output_root}"],
    )

    assert output_root.is_dir()
    for resolved in resolved_runs:
        run = resolved["run"]
        assert run["output_root"] == str(output_root)
        assert run["output_dir"] == str(
            output_root / run["output_group"] / run["run_id"]
        )


def test_single_run_defaults_to_outputs_root(tmp_path):
    config_path = _write_single_run_config(tmp_path)

    resolved = resolve_run_config(config_path)

    assert resolved["model"]["variant"] == "slicing"
    assert resolved["model"]["membership_correction"] is True
    assert resolved["run"]["output_root"] == "outputs"
    assert resolved["training"]["gradient_clip_norm"] == 1.0
    assert resolved["training"]["optimizer_kwargs"] == {
        "betas": [0.9, 0.95],
        "eps": 1e-08,
        "weight_decay": 0.1,
    }
    assert resolved["run"]["output_dir"] == (
        f"outputs/{resolved['run']['output_group']}/single-output-root-001"
    )


def test_shared_configs_resolve_default_model_variant():
    debug_resolved = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
    )
    pilot_resolved = resolve_run_config("configs/dmodel256_pilot_comparison.yaml")

    assert debug_resolved["model"]["variant"] == "slicing"
    assert debug_resolved["model"]["membership_correction"] is True
    assert pilot_resolved["model"]["variant"] == "slicing"
    assert pilot_resolved["model"]["membership_correction"] is True


@pytest.mark.parametrize(
    "overrides, expected_mode, expected_membership_correction",
    [
        (["model.correction_mode=none", "model.membership_correction=false"], "none", False),
        (["model.correction_mode=gmc"], "gmc", True),
        (
            ["model.variant=concat", "model.correction_mode=lmc"],
            "lmc",
            True,
        ),
    ],
)
def test_explicit_correction_modes_resolve_and_validate(
    overrides,
    expected_mode,
    expected_membership_correction,
):
    resolved = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        overrides=overrides,
    )

    assert resolved["model"]["requested_correction_mode"] == expected_mode
    assert resolved["model"]["correction_mode"] == expected_mode
    assert resolved["model"]["membership_correction"] is expected_membership_correction


@pytest.mark.parametrize(
    "overrides",
    [
        ["model.correction_mode=gmc", "model.membership_correction=false"],
        ["model.correction_mode=none", "model.membership_correction=true"],
    ],
)
def test_membership_correction_conflicts_fail_fast(overrides):
    with pytest.raises(
        ConfigError,
        match="model.correction_mode and model.membership_correction must not disagree",
    ):
        resolve_run_config(
            "configs/debug_matrix.yaml",
            run_id="debug-nested-001",
            overrides=overrides,
        )


def test_lmc_is_rejected_for_non_concat_runs():
    with pytest.raises(
        ConfigError,
        match="model.correction_mode=lmc is only valid for concat runs",
    ):
        resolve_run_config(
            "configs/debug_matrix.yaml",
            run_id="debug-standalone-s-001",
            overrides=["model.correction_mode=lmc"],
        )


def test_concat_defaults_membership_correction_on():
    resolved = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        overrides=["model.variant=concat"],
    )

    assert resolved["model"]["variant"] == "concat"
    assert resolved["model"]["membership_correction"] is True


def test_slicing_allows_disabling_membership_correction():
    resolved = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        overrides=[
            "model.correction_mode=none",
            "model.membership_correction=false",
        ],
    )

    assert resolved["model"]["variant"] == "slicing"
    assert resolved["model"]["membership_correction"] is False


def test_invalid_model_variant_override_fails_fast_before_output_setup(tmp_path):
    output_root = tmp_path / "should-not-exist"

    with pytest.raises(
        ConfigError,
        match=r"Unsupported model\.variant='dog_llama'",
    ):
        resolve_run_config(
            "configs/debug_matrix.yaml",
            run_id="debug-nested-001",
            overrides=[
                f"run.output_root={output_root}",
                "model.variant=dog_llama",
            ],
        )

    assert not output_root.exists()


def test_single_run_output_root_override_derives_output_dir(tmp_path):
    config_path = _write_single_run_config(tmp_path)
    output_root = tmp_path / "single-output"

    resolved = resolve_run_config(
        config_path,
        overrides=[f"run.output_root={output_root}"],
    )

    assert output_root.is_dir()
    assert resolved["run"]["output_root"] == str(output_root)
    assert resolved["run"]["output_dir"] == str(
        output_root / resolved["run"]["output_group"] / "single-output-root-001"
    )


def test_explicit_output_dir_override_wins_over_output_root(tmp_path):
    output_root = tmp_path / "matrix-output"
    explicit_output_dir = tmp_path / "explicit-output" / "debug-nested-001"

    resolved = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        overrides=[f"run.output_root={output_root}"],
        output_dir=explicit_output_dir,
    )

    assert resolved["run"]["output_root"] == str(output_root)
    assert resolved["run"]["output_dir"] == str(explicit_output_dir)


def test_unwritable_output_root_fails_before_training(tmp_path):
    output_root = tmp_path / "blocked-output"
    output_root.mkdir()
    output_root.chmod(0o555)

    try:
        with pytest.raises(ConfigError, match="writ|permission|output root"):
            resolve_run_config(
                "configs/debug_matrix.yaml",
                run_id="debug-nested-001",
                overrides=[f"run.output_root={output_root}"],
            )
    finally:
        output_root.chmod(0o755)


def test_dmodel256_pilot_config_preserves_clarified_terms_and_shape_fields():
    resolved = resolve_run_config("configs/dmodel256_pilot_comparison.yaml")

    run = resolved["run"]
    model = resolved["model"]

    assert run["model_shape_label"] == "dmodel256"
    assert run["sampling_mode"] == "nested-random"
    assert run["completion_label"] == "run"
    assert run["model_family_slug"] == "matformer_llama"
    assert run["family_size_slug"] == run["model_size_slug"]
    assert run["output_group"].startswith("matformer_llama_")
    assert run["output_dir"] == f"outputs/{run['output_group']}/{run['run_id']}"
    assert "model_size_label" not in run

    assert model["d_model"] == 256
    assert model["num_layers"] == 16
    assert model["num_attention_heads"] == 16
    assert model["context_length"] == 1024
    assert model["vocab_size_assumption"] == 256000
    assert model["granularity_prefixes"] == {
        "s": 0.125,
        "m": 0.25,
        "l": 0.5,
        "xl": 1.0,
    }
    assert [entry["prefix_width"] for entry in model["ffn_prefix_metadata"]] == [
        128,
        256,
        512,
        1024,
    ]

    assert resolved["training"]["token_budget"] < 10_000_000_000
    assert resolved["training"]["granularity_sampling"] == "random"
    validate_run_config(resolved)


def test_granularity_prefix_validation_rejects_non_monotonic_widths():
    with pytest.raises(
        ConfigError,
        match="strictly nested|strictly increasing",
    ):
        resolve_run_config(
            "configs/dmodel256_pilot_comparison.yaml",
            overrides=["model.granularity_prefixes.m=0.1"],
        )


def test_granularity_prefix_validation_rejects_extra_keys():
    with pytest.raises(ConfigError, match="must match model.granularities"):
        resolve_run_config(
            "configs/dmodel256_pilot_comparison.yaml",
            overrides=["model.granularity_prefixes.extra=0.01"],
        )


def test_intermediate_size_is_derived_from_d_model_and_rejects_mismatch():
    resolved = resolve_run_config("configs/dmodel256_pilot_comparison.yaml")
    assert resolved["model"]["intermediate_size"] == 1024

    with pytest.raises(ConfigError, match="model.intermediate_size must equal"):
        resolve_run_config(
            "configs/dmodel256_pilot_comparison.yaml",
            overrides=["model.intermediate_size=2048"],
        )


def test_dmodel256_rejects_old_completion_label_strings():
    resolved = resolve_run_config("configs/dmodel256_pilot_comparison.yaml")

    mislabeled = copy.deepcopy(resolved)
    mislabeled["run"]["completion_label"] = "full-token-budget"

    with pytest.raises(ConfigError, match="Unknown completion label"):
        validate_run_config(mislabeled)


def test_dmodel256_sampling_mode_derives_granularity_sampling():
    random_sampling = resolve_run_config("configs/dmodel256_pilot_comparison.yaml")
    assert random_sampling["run"]["sampling_mode"] == "nested-random"
    assert random_sampling["training"]["granularity_sampling"] == "random"

    nested_all = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        overrides=[
            "run.run_id=dmodel256-pilot-nested-all-001",
            "run.sampling_mode=nested-all",
        ],
    )
    assert nested_all["run"]["sampling_mode"] == "nested-all"
    assert nested_all["training"]["granularity_sampling"] == "all"

    standalone = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        overrides=[
            "run.run_id=dmodel256-standalone-m-001",
            "run.model_family=standalone",
            "run.sampling_mode=standalone",
            "run.granularity=m",
        ],
    )
    assert standalone["run"]["sampling_mode"] == "standalone"
    assert standalone["training"]["granularity_sampling"] == "all"
    assert standalone["model"]["granularities"] == ["m"]

    with pytest.raises(ConfigError, match="conflicts"):
        resolve_run_config(
            "configs/dmodel256_pilot_comparison.yaml",
            overrides=[
                "run.sampling_mode=nested-all",
                "training.granularity_sampling=random",
            ],
        )


def test_dmodel256_pilot_derives_training_length_with_default_world_size(tmp_path, monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    output_root = tmp_path / "pilot-output"

    resolved = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        overrides=[f"run.output_root={output_root}"],
    )

    training = resolved["training"]
    expected_tokens_per_step = (
        training["batch_size_per_process"]
        * resolved["model"]["context_length"]
    )
    expected_steps = math.ceil(training["token_budget"] / expected_tokens_per_step)

    for field_name in [
        "effective_world_size",
        "expected_tokens_per_step",
        "derived_max_steps",
    ]:
        assert field_name in training
    assert training["effective_world_size"] == 1
    assert training["expected_tokens_per_step"] == expected_tokens_per_step
    assert training["derived_max_steps"] == expected_steps
    assert training["max_steps"] == expected_steps


def test_dmodel256_pilot_derives_training_length_from_distributed_world_size(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("WORLD_SIZE", "4")
    output_root = tmp_path / "dmodel256-pilot-comparison-001"

    resolved = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        overrides=[f"run.output_root={output_root}"],
    )

    training = resolved["training"]
    expected_tokens_per_step = (
        training["batch_size_per_process"]
        * resolved["model"]["context_length"]
        * 4
    )
    expected_steps = math.ceil(training["token_budget"] / expected_tokens_per_step)

    for field_name in [
        "effective_world_size",
        "expected_tokens_per_step",
        "derived_max_steps",
    ]:
        assert field_name in training
    assert training["effective_world_size"] == 4
    assert training["expected_tokens_per_step"] == expected_tokens_per_step
    assert training["derived_max_steps"] == expected_steps
    assert training["max_steps"] == expected_steps


def test_dmodel256_pilot_resolves_scaled_learning_rate_warmup_precedence_and_optimizer_controls(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("WORLD_SIZE", "4")
    output_root = tmp_path / "dmodel256-pilot-comparison-001"

    resolved = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        output_dir=output_root,
        overrides=[
            "training.warmup_ratio=0.9",
            "training.warmup_steps=7",
            "training.optimizer.preset=null",
            "training.optimizer.name=sgd",
            "training.optimizer.kwargs.momentum=0.8",
            "training.optimizer.kwargs.dampening=0.1",
            "training.optimizer.kwargs.nesterov=true",
        ],
    )

    training = resolved["training"]
    assert training["learning_rate_scale_rule"] == "linear"
    assert training["learning_rate_scale_factor"] == 4.0
    assert training["resolved_learning_rate"] == 0.0012
    assert training["warmup_ratio"] == 0.9
    assert training["warmup_steps"] == 7
    assert training["resolved_warmup_steps"] == 7
    assert training["scheduler"]["kwargs"]["warmup_steps"] == 7
    assert training["scheduler"]["resolved_warmup_steps"] == 7
    assert training["gradient_clip_norm"] == 1.0
    assert training["optimizer_name"] == "sgd"
    assert training["optimizer_kwargs"] == {
        "momentum": 0.8,
        "dampening": 0.1,
        "nesterov": True,
        "weight_decay": 0.0,
    }


def test_dmodel256_pilot_resolves_schedule_and_optimizer_defaults(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("WORLD_SIZE", "4")
    output_root = tmp_path / "pilot-output"

    resolved = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        overrides=[f"run.output_root={output_root}"],
    )

    training = resolved["training"]
    assert training["base_learning_rate"] == 0.0003
    assert training["learning_rate_scale_rule"] == "linear"
    assert training["learning_rate_scale_factor"] == 4.0
    assert training["resolved_learning_rate"] == 0.0012
    assert training["warmup_ratio"] == 0.01635
    assert training["warmup_steps"] == 2000
    assert training["resolved_warmup_steps"] == 2000
    assert training["scheduler"]["kwargs"]["warmup_steps"] == 2000
    assert training["scheduler"]["resolved_warmup_steps"] == 2000
    assert training["gradient_clip_norm"] == 1.0
    assert training["optimizer_name"] == "adamw"
    assert training["optimizer_kwargs"] == {
        "betas": [0.9, 0.95],
        "eps": 1e-08,
        "weight_decay": 0.1,
    }
    assert training["optimizer"] == {
        "name": "adamw",
        "kwargs": {
            "betas": [0.9, 0.95],
            "eps": 1e-08,
            "weight_decay": 0.1,
        },
    }


def test_optimizer_preset_resolution_merges_registry_defaults_and_partial_overrides():
    resolved = resolve_run_config(
        "tests/fixtures/experiment_config_resolution.yaml",
        overrides=["training.optimizer.kwargs.weight_decay=0.05"],
    )

    training = resolved["training"]
    expected_optimizer_kwargs = {
        "betas": [0.9, 0.95],
        "eps": 1e-08,
        "weight_decay": 0.05,
    }

    assert resolved["run"]["sampling_mode"] == "nested-all"
    assert resolved["model"]["granularity_sampling_mode"] == "global"
    assert training["optimizer_name"] == "adamw"
    assert training["optimizer_kwargs"] == expected_optimizer_kwargs
    assert training["optimizer"] == {
        "name": "adamw",
        "kwargs": expected_optimizer_kwargs,
    }
    assert training["preset_selections"] == {"optimizer": "adam"}
    assert set(training["preset_registry_paths"]) == {"optimizer"}
    assert training["preset_registry_paths"]["optimizer"].endswith(
        "configs/presets/optimizer/adam.yaml"
    )


def test_invalid_optimizer_preset_names_fail_before_training_starts():
    with pytest.raises(
        ConfigError,
        match=r"Unknown training\.optimizer\.preset='missing'",
    ):
        resolve_run_config(
            "tests/fixtures/experiment_config_resolution.yaml",
            overrides=["training.optimizer.preset=missing"],
        )


def test_optimizer_preset_rejects_name_overrides():
    with pytest.raises(
        ConfigError,
        match=(
            r"training\.optimizer\.name cannot be overridden when "
            r"training\.optimizer\.preset is set"
        ),
    ):
        resolve_run_config(
            "tests/fixtures/experiment_config_resolution.yaml",
            overrides=["training.optimizer.name=sgd"],
        )


def test_single_run_resolves_explicit_schedule_and_optimizer_overrides(tmp_path):
    config_path = _write_single_run_config(tmp_path)

    resolved = resolve_run_config(
        config_path,
        overrides=[
            "training.warmup_ratio=0.25",
            "training.optimizer.preset=null",
            "training.optimizer.name=sgd",
            "training.optimizer.kwargs.momentum=0.9",
            "training.optimizer.kwargs.nesterov=true",
            "training.warmup_steps=null",
            "training.scheduler.kwargs.warmup_steps=null",
        ],
    )

    training = resolved["training"]
    assert training["learning_rate_scale_rule"] == "none"
    assert training["learning_rate_scale_factor"] == 1.0
    assert training["resolved_learning_rate"] == 0.0003
    assert training["warmup_ratio"] == 0.25
    assert training["warmup_steps"] is None
    assert training["resolved_warmup_steps"] == 1
    assert training["scheduler"]["kwargs"]["warmup_steps"] == 1
    assert training["scheduler"]["resolved_warmup_steps"] == 1
    assert training["gradient_clip_norm"] == 1.0
    assert training["optimizer_name"] == "sgd"
    assert training["optimizer_kwargs"] == {
        "momentum": 0.9,
        "dampening": 0.0,
        "nesterov": True,
        "weight_decay": 0.0,
    }
    assert training["optimizer"] == {
        "name": "sgd",
        "kwargs": {
            "momentum": 0.9,
            "dampening": 0.0,
            "nesterov": True,
            "weight_decay": 0.0,
        },
    }
