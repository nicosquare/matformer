"""Preflight acceptance with audited metadata fixtures and real CPU models."""
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import src.evaluation.optimizer_ownership as campaign
from src.utils.config import ConfigError, resolve_run_config
from src.utils.reproducibility import seed_for

RECIPE = Path("configs/controlled_exps/tinystories_instruct_optimizer_ownership.yaml")


@pytest.fixture
def audited_inputs(tmp_path, monkeypatch):
    """Mock external integrity IO only; retain the real resolver/model checks."""
    from src.training import packed_corpus, fineweb_tokenizer

    recipe = yaml.safe_load(RECIPE.read_text())
    expected = recipe["expected_data"]
    roles = {
        role: {
            "manifest_hash": expected[f"{role}_manifest_hash"],
            "token_count": 713790848 if role == "optimizer_training" else 16384,
            "sequence_count": 5576491 if role == "optimizer_training" else 128,
            "source_document_count": 2476404 if role == "optimizer_training" else 128,
            "shards": [],
        }
        for role in (
            "optimizer_training",
            "ordinary_validation",
            "controller",
            "final_holdout",
        )
    }
    tokenizer = {
        "tokenizer_name": "fixture-tokenizer",
        "name": "fixture-tokenizer",
        "manifest_hash": expected["tokenizer_manifest_hash"],
        "revision": expected["tokenizer_manifest_hash"],
        "sentencepiece_model_sha256": expected["tokenizer_model_sha256"],
        "vocab_size": 2048,
    }
    manifest = {
        "schema_version": 3,
        "corpus_hash": expected["corpus_hash"],
        "context_length": 128,
        "data_seed": 42,
        "roles": roles,
        "role_manifest_hashes": {r: v["manifest_hash"] for r, v in roles.items()},
        "tokenizer": tokenizer,
        "training_order": {
            "path": "order.bin",
            "sha256": expected["training_order_sha256"],
            "count": 5576491,
            "permutation_version": packed_corpus.PERMUTATION_VERSION,
        },
        "source": {
            "dataset_name": "roneneldan/TinyStoriesInstruct",
            "dataset_config_name": "default",
            "split": "train+validation",
        },
    }
    audit = {
        **expected,
        "status": "passed",
        "schema_version": 3,
        "source_exhausted": True,
        "training_sequence_count": 5576491,
        "training_token_count": 713790848,
        "verified_training_order_count": 5576491,
        "verified_shard_count": 89,
        "tokenizer_vocab_size": 2048,
        "role_manifest_hashes": manifest["role_manifest_hashes"],
        "source_document_count": 2477172,
        "role_source_document_counts": {
            "optimizer_training": 2476404,
            "ordinary_validation": 128,
            "controller": 128,
            "final_holdout": 512,
        },
        "reserved_role_counts": {
            "ordinary_validation": 128,
            "controller": 128,
            "final_holdout": 512,
        },
        "reserved_pairwise_intersection_counts": {
            "validation_controller": 0,
            "validation_holdout": 0,
            "controller_holdout": 0,
        },
    }
    monkeypatch.setattr(
        packed_corpus, "load_corpus_manifest", lambda *a, **kw: copy.deepcopy(manifest)
    )
    monkeypatch.setattr(
        packed_corpus, "audit_packed_corpus", lambda *a, **kw: copy.deepcopy(audit)
    )
    monkeypatch.setattr(
        fineweb_tokenizer,
        "load_tokenizer_manifest",
        lambda *a, **kw: copy.deepcopy(tokenizer),
    )
    return recipe, manifest, audit


def expand(tmp_path, recipe):
    return campaign.expand_campaign(
        recipe,
        prepared_corpus_dir=tmp_path / "corpus",
        tokenizer_dir=tmp_path / "tokenizer",
        run_output_root=tmp_path / "runs",
    )


def test_nine_arm_expansion_and_real_model_checks(tmp_path, audited_inputs):
    recipe, _, _ = audited_inputs
    runs = expand(tmp_path, recipe)
    assert [r["arm_id"] for r in runs] == list(campaign.ARM_IDS)
    assert len({r["run_id"] for r in runs}) == 9
    assert not (tmp_path / "runs").exists()
    checks = campaign.inspect_campaign_models(runs)
    for run, check in zip(runs, checks):
        arm = next(a for a in campaign.ARMS if a["arm_id"] == run["arm_id"])
        assert run["assigned_updates"] == arm["assigned_updates"]
        assert (
            run["resolved_config"]["model"]["intermediate_size"]
            == arm["physical_ffn_dimension"]
        )
        assert check["counts"] == {
            w["label"]: w["non_embedding_parameters"]
            for w in campaign.WIDTHS
            if w["label"] in arm["endpoint_widths"]
        }
        assert run["initialization"]["seed"] == 42
        assert run["initialization"]["method"] == "fresh_normal_constructor"
        assert (
            run["contract_hash"]
            == run["resolved_config"]["optimizer_ownership_contract_hash"]
        )
        assert (
            run["resolved_config"]["dataset"]["optimizer_iteration"][
                "excluded_tail_samples"
            ]
            == 43
        )
    assert [o["owner_id"] for o in checks[-1]["owners"]] == list(campaign.OWNER_IDS)
    assert sum(o["parameter_elements"] for o in checks[-1]["owners"]) == 524864


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("training", "learning_rate", 0.007),
        ("training", "max_steps", 100),
        ("training", "gradient_clip_norm", 2),
        ("training", "mixed_precision", "none"),
        ("model", "initializer_range", 0.03),
        ("model", "d_model", 128),
        ("run", "seed", 43),
        ("dataset", "data_seed", 43),
        ("dataset", "optimizer_iteration", {"mode": "single_pass"}),
        ("evaluation", "final_holdout", {"evaluate_during_training": True}),
    ],
)
def test_changed_common_controls_rejected(
    tmp_path, audited_inputs, section, key, value
):
    recipe, _, _ = audited_inputs
    recipe["common"][section][key] = value
    with pytest.raises(ConfigError, match=key):
        expand(tmp_path, recipe)


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "scope", "horizon", "historical", "clip"]
)
def test_changed_arm_matrix_rejected(tmp_path, audited_inputs, mutation):
    recipe, _, _ = audited_inputs
    if mutation == "missing":
        del recipe["arms"]["S1"]
    elif mutation == "extra":
        recipe["arms"]["old"] = {}
    elif mutation == "scope":
        recipe["arms"]["C2"]["training"]["optimizer"]["state_scope"] = "shared"
    elif mutation == "horizon":
        recipe["arms"]["S1"]["training"]["token_budget"] = 713785344
    elif mutation == "historical":
        recipe["arms"]["S1"]["run"][
            "run_id"
        ] = "tinystories-instruct-optimizer-state-shared-s42"
    else:
        recipe["arms"]["C3"]["training"]["gradient_clipping"]["mode"] = "global"
    with pytest.raises(ConfigError):
        expand(tmp_path, recipe)


def test_occupied_and_pinned_identity_rejections(tmp_path, audited_inputs):
    recipe, _, audit = audited_inputs
    path = tmp_path / "runs" / "S1"
    path.mkdir(parents=True)
    with pytest.raises(ConfigError, match="occupied"):
        expand(tmp_path, recipe)
    path.rmdir()
    for key in recipe["expected_data"]:
        altered = copy.deepcopy(recipe)
        altered["expected_data"][key] = "changed"
        with pytest.raises(ConfigError, match=key):
            expand(tmp_path, altered)
    for key in [
        "training_sequence_count",
        "training_token_count",
        "verified_training_order_count",
        "source_exhausted",
    ]:
        bad = copy.deepcopy(audit)
        bad[key] = 0
        with pytest.raises(ConfigError, match=key):
            campaign.validate_corpus_audit(bad, recipe["expected_data"])
    audit["reserved_pairwise_intersection_counts"]["validation_holdout"] = 1
    with pytest.raises(ConfigError, match="overlap"):
        campaign.validate_corpus_audit(audit, recipe["expected_data"])


def test_trace_digests_match_runtime_rng_and_repeat_sampler(tmp_path, audited_inputs):
    from src.training.packed_corpus import RepeatingNoPaddingDistributedBatchSampler
    from src.training import steps

    recipe, _, _ = audited_inputs
    config = expand(tmp_path, recipe)[-1]["resolved_config"]
    original = random.getstate()
    result = campaign.expected_action_trace(config)
    generator = random.Random(seed_for(config, "granularity_selection"))
    digest = hashlib.sha256()
    counts = dict.fromkeys(campaign.WIDTH_LABELS, 0)
    for _ in range(348528):
        width = campaign.WIDTH_LABELS[generator.randrange(4)]
        digest.update((width + "\n").encode())
        counts[width] += 1
    assert result["sha256"] == digest.hexdigest()
    assert result["counts"] == counts
    assert sum(counts.values()) == 348528
    assert len(set(counts.values())) > 1
    assert random.getstate() == original
    # Independent existing trainer selector, isolated from its live dedicated RNG.
    old = steps.dedicated_random
    try:
        rng = random.Random(seed_for(config, "granularity_selection"))
        steps.dedicated_random = lambda *_: rng
        expected = random.Random(seed_for(config, "granularity_selection"))
        for _ in range(100):
            assert steps.select_training_granularities(
                config, list(campaign.WIDTH_LABELS), torch.device("cpu")
            ) == [campaign.WIDTH_LABELS[expected.randrange(4)]]
    finally:
        steps.dedicated_random = old
    sampler = RepeatingNoPaddingDistributedBatchSampler(
        19,
        4,
        0,
        1,
        64,
        16,
        corpus_hash="fixture",
        optimizer_training_manifest_hash="fixture",
    )
    traces = campaign.expected_epoch_traces(sampler, epochs=4)
    batches = list(sampler)
    for epoch in range(4):
        values = np.asarray(batches[epoch * 4 : (epoch + 1) * 4], dtype="<u8").reshape(
            -1
        )
        assert traces[epoch]["sha256"] == hashlib.sha256(values.tobytes()).hexdigest()
        assert set(values) == set(np.asarray(batches[:4]).reshape(-1))


def test_preflight_publishes_only_after_all_checks(
    tmp_path, audited_inputs, monkeypatch
):
    recipe, _, _ = audited_inputs
    source = tmp_path / "campaign.yaml"
    source.write_text(yaml.safe_dump(recipe))
    monkeypatch.setattr(
        campaign,
        "build_expected_traces",
        lambda *a: {
            r["arm_id"]: {
                "epochs": [{"sha256": "same"}] * r["assigned_epochs"],
                "actions": None,
            }
            for r in a[0]
        },
    )
    import src.training.modeling as modeling
    from transformers import LlamaForCausalLM

    monkeypatch.setattr(
        LlamaForCausalLM,
        "forward",
        lambda *a, **kw: pytest.fail("preflight ran forward"),
    )
    original = modeling.build_model
    calls = []

    def fail_last(config):
        calls.append(config["run"]["run_id"])
        if len(calls) == 9:
            raise RuntimeError("C3 injected failure")
        return original(config)

    monkeypatch.setattr(modeling, "build_model", fail_last)
    kwargs = dict(
        campaign_path=source,
        prepared_corpus_dir=tmp_path / "corpus",
        tokenizer_dir=tmp_path / "tokenizer",
        output_dir=tmp_path / "published",
        run_output_root=tmp_path / "runs",
    )
    with pytest.raises(RuntimeError, match="injected"):
        campaign.preflight_campaign(**kwargs)
    assert not (tmp_path / "published").exists()
    assert not (tmp_path / "runs").exists()
    monkeypatch.setattr(modeling, "build_model", original)
    result = campaign.preflight_campaign(**kwargs)
    assert result["status"] == "passed"
    assert len(list((tmp_path / "published" / "configs").glob("*.yaml"))) == 9
    manifest = json.loads(
        (tmp_path / "published" / "campaign_manifest.json").read_text()
    )
    assert len(manifest["runs"]) == 9
    assert not (tmp_path / "runs").exists()
    resolved = resolve_run_config(
        tmp_path / "published" / "configs" / "C3.yaml", create_output_dirs=False
    )
    assert (
        resolved["optimizer_ownership_contract_hash"]
        == manifest["runs"][-1]["contract_hash"]
    )
    with pytest.raises(ConfigError, match="occupied"):
        campaign.preflight_campaign(**kwargs)


def test_materialized_config_rejects_stale_controls_and_hash(tmp_path, audited_inputs):
    runs = expand(tmp_path, audited_inputs[0])
    for run in runs:
        path = tmp_path / f"{run['arm_id']}.yaml"
        path.write_text(yaml.safe_dump(run["executable_config"]))
        resolved = resolve_run_config(path, create_output_dirs=False)
        assert resolved["model"]["intermediate_size"] == run["physical_ffn_dimension"]
    path = tmp_path / "S1.yaml"
    with pytest.raises(ConfigError, match="learning_rate"):
        resolve_run_config(
            path, overrides=["training.learning_rate=0.007"], create_output_dirs=False
        )
    with pytest.raises(ConfigError, match="hash"):
        resolve_run_config(
            path,
            overrides=["optimizer_ownership_contract_hash=stale"],
            create_output_dirs=False,
        )
    with pytest.raises(ConfigError, match="run.output_dir"):
        resolve_run_config(path, overrides=["run.seed=43"], create_output_dirs=False)


def test_model_construction_uses_normal_seed_and_restores_rng(
    tmp_path, audited_inputs, monkeypatch
):
    import src.training.modeling as modeling
    from src.utils.reproducibility import seed_model_initialization

    runs = expand(tmp_path, audited_inputs[0])
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    original = modeling.build_model
    constructions = []

    def observe(config):
        model = original(config)
        constructions.append(
            {k: v.detach().clone() for k, v in model.state_dict().items()}
        )
        return model

    monkeypatch.setattr(modeling, "build_model", observe)
    campaign.inspect_campaign_models(runs)
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    try:
        for run, actual in zip(runs, constructions):
            seed_model_initialization(run["resolved_config"])
            expected = original(run["resolved_config"]).state_dict()
            assert all(torch.equal(actual[k], v) for k, v in expected.items())
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def test_cli_requires_five_options_and_names_bad_control(
    tmp_path, audited_inputs, capsys
):
    from scripts.analyze_tinystories_optimizer_ownership import main

    with pytest.raises(SystemExit) as missing:
        main(["preflight"])
    assert missing.value.code == 2
    recipe = audited_inputs[0]
    recipe["common"]["training"]["learning_rate"] = 0.1
    source = tmp_path / "bad.yaml"
    source.write_text(yaml.safe_dump(recipe))
    with pytest.raises(SystemExit) as invalid:
        main(
            [
                "preflight",
                "--campaign",
                str(source),
                "--prepared-corpus-dir",
                str(tmp_path / "corpus"),
                "--tokenizer-dir",
                str(tmp_path / "tokenizer"),
                "--output-dir",
                str(tmp_path / "published"),
                "--run-output-root",
                str(tmp_path / "runs"),
            ]
        )
    assert invalid.value.code == 1
    assert "learning_rate" in capsys.readouterr().err
    assert not (tmp_path / "published").exists()


def test_failed_publication_removes_reservation(tmp_path, audited_inputs, monkeypatch):
    recipe, _, _ = audited_inputs
    source = tmp_path / "campaign.yaml"
    source.write_text(yaml.safe_dump(recipe))
    monkeypatch.setattr(campaign, "build_expected_traces", lambda *a: {})
    original_rename = Path.rename

    def fail_publication(path, target):
        if Path(target) == tmp_path / "published":
            raise OSError("injected publication failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_publication)
    with pytest.raises(OSError, match="publication failure"):
        campaign.preflight_campaign(
            campaign_path=source,
            prepared_corpus_dir=tmp_path / "corpus",
            tokenizer_dir=tmp_path / "tokenizer",
            output_dir=tmp_path / "published",
            run_output_root=tmp_path / "runs",
        )
    assert not (tmp_path / "published").exists()
    assert not (tmp_path / "runs").exists()
    assert not campaign._reservation_path(tmp_path / "runs").exists()
    assert not list(tmp_path.glob(".published-*"))


@pytest.mark.parametrize(
    "key,value",
    [
        ("excluded_tail_samples", 42),
        ("aligned_epoch_samples", 5576449),
        ("complete_epochs", 3),
        ("partial_final_epoch_samples", 1),
    ],
)
def test_changed_resolved_alignment_rejected(tmp_path, audited_inputs, key, value):
    run = expand(tmp_path, audited_inputs[0])[-1]
    run["resolved_config"]["dataset"]["optimizer_iteration"][key] = value
    with pytest.raises(ConfigError, match=key):
        campaign.validate_run_budget(run["resolved_config"], campaign.ARMS[-1])
