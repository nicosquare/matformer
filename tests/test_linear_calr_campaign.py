"""Compatibility boundary for the new campaign; no schema-6 behavior yet."""
import copy
import math

import pytest
import yaml

from src.evaluation import optimizer_ownership as campaign
from src.training import steps
from src.utils.reproducibility import build_optimizer_ownership_signature, stable_hash
from test_optimizer_ownership_campaign import audited_inputs, expand
from test_s1_warmup_campaign import (
    historical_campaign, WARMUP_RECIPE,
    test_historical_resolved_signatures_and_serialized_fields as check_legacy,
)
from test_optimizer_ownership_resume import (
    fixture, train, save, load, assert_state_equal,
)
from test_c4_separate_corrections import (
    test_schedule_boundary_resume_and_incompatible_metadata as check_c4_restore,
)


def test_schema1_through4_frozen_resolution(historical_campaign, tmp_path):
    # Reuse the existing immutable schema 1–3 signatures and schema-4 resolved
    # snapshot, including serialized field sets and JSON/YAML round trips.
    check_legacy(historical_campaign, tmp_path)


SCHEMA5 = {
    'S1-linear-w256': (
        '4aa6eb318119374a569d5716b06104f3de62ca863e798019ae3f8a3c4d6c2766',
        '8bf1b1ffb62094fb66649422f6c55f4d566fe3be6222d021b28ac63b959d293b'),
    'S1-geometric-w256': (
        '38523a3105de4b721560f1416e481d37e3b1ab495368e89ebd471e95e29ccba2',
        '26f594e9f1bd864f41c420b92771bef6e1b4c93381cca553d3ac790ce29897fb'),
}


def test_schema5_frozen_resolution(tmp_path, monkeypatch, audited_inputs):
    monkeypatch.setattr(campaign, '_provenance', lambda: {
        'code_revision': 'linear-calr-phase1-schema5-fixture',
        'working_tree_dirty': False,
    })
    runs = expand(tmp_path, yaml.safe_load(WARMUP_RECIPE.read_text()))
    assert {r['arm_id'] for r in runs} == set(SCHEMA5)
    for run in runs:
        config = copy.deepcopy(run['resolved_config'])
        for section, key in [('dataset', 'prepared_corpus_dir'),
                             ('model', 'tokenizer_dir'), ('run', 'output_dir'),
                             ('run', 'output_root')]:
            config[section][key] = config[section][key].replace(str(tmp_path), '<fixture>')
        expected_contract, expected_config = SCHEMA5[run['arm_id']]
        assert run['contract_hash'] == expected_contract
        assert stable_hash(config) == expected_config
        contract = run['optimizer_ownership_contract']
        for restored in [yaml.safe_load(yaml.safe_dump(contract)), copy.deepcopy(contract)]:
            assert build_optimizer_ownership_signature(restored) == (expected_contract, contract)
        campaign.validate_materialized_config(run['resolved_config'])
    assert not (tmp_path / 'runs').exists()


@pytest.mark.parametrize('arm', ['S1', 'S2', 'C3'])
@pytest.mark.parametrize('position', [0, 63, 64, 65, 87132, 174264, 348527, 348528])
def test_legacy_cosine_schedule_and_clock_restore(tmp_path, arm, position):
    config, model, _, _, _, _ = fixture(tmp_path, arm)
    training = copy.deepcopy(config['training'])
    training.update(scheduler_name='cosine', scheduler_kwargs={},
                    resolved_warmup_steps=64, max_steps=348528)
    optimizer, clock = steps.build_optimizer_and_scheduler(model, training)
    carrier = getattr(clock, '_scheduler', clock)
    expected = carrier.base_lrs[0] * (
        position / 64 if position < 64 else
        0.5 * (1 + math.cos(math.pi * (position - 64) / (348528 - 64))))
    assert carrier.base_lrs[0] * carrier.lr_lambdas[0](position) == pytest.approx(expected, abs=1e-18)
    # A state-seeded clock probe checks global, rather than owner-local time.
    state = clock.state_dict()
    scheduler_state = state.get('scheduler_state_dict', state)
    scheduler_state.update(last_epoch=position, _step_count=position + 1, _last_lr=[expected])
    if hasattr(clock, 'synchronize'):
        state['position'] = position
        state['last_committed_learning_rates'] = (
            [carrier.base_lrs[0] * carrier.lr_lambdas[0](position - 1)] if position else None)
        for group in state.get('carrier_optimizer_state_dict', {}).get('param_groups', []):
            group['lr'] = expected
    clock.load_state_dict(state)
    if hasattr(clock, 'synchronize'):
        clock.synchronize(optimizer)
    else:
        for group in optimizer.param_groups:
            group['lr'] = expected
    clone_optimizer, clone_clock = steps.build_optimizer_and_scheduler(model, training)
    clone_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    clone_clock.load_state_dict(copy.deepcopy(clock.state_dict()))
    if hasattr(clone_clock, 'synchronize'):
        clone_clock.synchronize(clone_optimizer)
    assert_state_equal(optimizer.state_dict(), clone_optimizer.state_dict())
    assert_state_equal(clock.state_dict(), clone_clock.state_dict())


@pytest.mark.parametrize('arm', ['S1', 'S2', 'C3'])
def test_legacy_committed_bundle_resume(tmp_path, arm):
    source = fixture(tmp_path, arm)
    train(source, stop=2)
    path = tmp_path / 'latest.pt'
    save(source, path)
    train(source)
    resumed = fixture(tmp_path, arm)
    load(resumed, path)
    train(resumed)
    for left, right in zip(source[1:4], resumed[1:4]):
        assert_state_equal(left.state_dict(), right.state_dict())


@pytest.mark.parametrize('grid', ['linear', 'geometric'])
@pytest.mark.parametrize('kind', ['GMC', 'LMC-only'])
def test_legacy_c4_schedule_and_restore(grid, kind):
    # Populated moments, mixed owners, exact continuation and rejection before
    # mutation are checked by the existing independent C4 regression helper.
    check_c4_restore(grid, kind, 64)
