"""Bounded campaign accounting, legacy migration, and real checkpoint recovery."""
import copy
import json

import pytest
import torch

from src.training.steps import snapshot_run_state
from src.utils.metrics import MetricsJournal, StreamingMetricsAccumulator


def row(ordinal, *, committed=True, width='g250'):
    return dict(split='train', step=ordinal + 1, granularity=width,
                tokens_seen=(ordinal + 1) * 8, content_tokens_seen=(ordinal + 1) * 8,
                optimizer_action_id=f'global:{ordinal}:-:-:{width}',
                optimizer_step_attempted=True, optimizer_step_committed=committed)


def test_ordered_accounting_matches_legacy_with_duplicate_and_failed_rows():
    legacy = StreamingMetricsAccumulator()
    compact = StreamingMetricsAccumulator(ordered_attempts=True)
    for i in range(1000):
        rows = [row(i, committed=i % 3 != 0)] * 2
        legacy.update(rows)
        compact.update(rows)
        if i == 499:
            compact = StreamingMetricsAccumulator(compact.state_dict())
    old, new = legacy.state_dict(), compact.state_dict()
    for key in old.keys() - {'schema_version', 'optimizer_attempt_ids'}:
        assert new[key] == old[key]
    assert new['schema_version'] == 2
    assert new['optimizer_last_attempt_id'] == 'global:999:-:-:g250'
    assert 'optimizer_attempt_ids' not in new


def test_campaign_bookkeeping_and_rollback_do_not_retain_history():
    compact = StreamingMetricsAccumulator(ordered_attempts=True)
    for i in range(20000):
        compact.update([row(i)])
        # Exercise the actual journal publication + training rollback path.
        state = {'metrics_accumulator_state': compact.state_dict()}
        saved = snapshot_run_state(state)
    assert compact.optimizer_attempt_ids == set()
    assert compact._sorted_optimizer_attempt_ids == []
    assert len(json.dumps(saved)) < 1000
    compact.update([row(20000)])
    assert saved['metrics_accumulator_state']['attempted_optimizer_steps'] == 20000
    assert saved['metrics_accumulator_state']['optimizer_last_attempt_id'] == 'global:19999:-:-:g250'


@pytest.mark.parametrize('count', [0, 1, 10000])
def test_legacy_history_is_converted_once_to_a_compact_marker(count):
    legacy = StreamingMetricsAccumulator()
    legacy.update(row(i, width=['g250', 'g500', 'g750', 'g1000'][i % 4]) for i in range(count))
    state = legacy.state_dict()
    original = copy.deepcopy(state)
    migrated = StreamingMetricsAccumulator(state, ordered_attempts=True)
    assert state == original
    assert migrated.attempted_optimizer_steps == count
    assert len(json.dumps(migrated.state_dict())) < 1000
    migrated.update([row(count)])
    assert migrated.attempted_optimizer_steps == count + 1


@pytest.mark.parametrize('damage', ['gap', 'duplicate', 'conflicting_width', 'count', 'format'])
def test_migration_rejects_incomplete_or_ambiguous_legacy_history(damage):
    legacy = StreamingMetricsAccumulator()
    legacy.update([row(i) for i in range(3)])
    state = legacy.state_dict()
    if damage == 'gap': state['optimizer_attempt_ids'][1] = 'global:3:-:-:g250'
    elif damage == 'duplicate': state['optimizer_attempt_ids'][1] = state['optimizer_attempt_ids'][0]
    elif damage == 'conflicting_width': state['optimizer_attempt_ids'][1] = 'global:0:-:-:g500'
    elif damage == 'count': state['attempted_optimizer_steps'] += 1
    else: state['optimizer_attempt_ids'][1] = 'unrecognized-id'
    with pytest.raises(ValueError):
        StreamingMetricsAccumulator(state, ordered_attempts=True)


@pytest.mark.parametrize('invalid', [row(0), row(1, width='g500'), row(3)])
def test_out_of_order_gap_and_conflicting_replay_fail_without_changing_state(invalid):
    compact = StreamingMetricsAccumulator(ordered_attempts=True)
    compact.update([row(0), row(1)])
    before = copy.deepcopy(compact.state_dict())
    with pytest.raises(ValueError, match='in order'):
        compact.update([invalid])
    assert compact.state_dict() == before


def test_repeated_failure_after_commit_is_not_counted_twice():
    compact = StreamingMetricsAccumulator(ordered_attempts=True)
    compact.update([row(0)])
    compact.update([dict(row(0), step=2, optimizer_failure_stage='checkpoint')])
    assert compact.attempted_optimizer_steps == compact.committed_optimizer_steps == 1
    assert compact.failed_optimizer_attempts == 0


def train_with_journal(bundle, *, stop=None, legacy=False):
    from src.training import steps
    config, model, optimizer, scheduler, batches, state = bundle
    journal = MetricsJournal(config['run']['output_dir'], artifact_state=state,
                             artifact_io_config=config,
                             checkpoint_step=state.get('last_completed_step', 0))
    if legacy:
        journal.accumulator = StreamingMetricsAccumulator()
    limited = {**config, 'training': {**config['training'],
                                    'max_steps': stop or config['training']['max_steps']}}
    steps.train_for_steps(limited, model, batches, [], optimizer, scheduler,
                          torch.device('cpu'), run_state=state, metrics_journal=journal)
    return journal


@pytest.mark.parametrize('arm', ['ST-g250', 'S1', 'S2', 'C1', 'C2', 'C3'])
@pytest.mark.parametrize('legacy', [False, True])
def test_real_model_checkpoint_resume_with_metrics_at_epoch_boundary(tmp_path, arm, legacy):
    from test_optimizer_ownership_resume import packed_fixture, save, load, assert_state_equal
    from src.utils.reproducibility import capture_rng_state
    full = packed_fixture(tmp_path / 'full', arm)
    train_with_journal(full)
    final_rng = capture_rng_state()
    source = packed_fixture(tmp_path / 'resumed', arm)
    train_with_journal(source, stop=2, legacy=legacy)
    path = tmp_path / 'resume.pt'
    save(source, path)
    restored = packed_fixture(tmp_path / 'resumed', arm)
    load(restored, path)
    train_with_journal(restored)
    assert restored[-1]['metrics_accumulator_state']['schema_version'] == 2
    for left, right in zip(full[1:4], restored[1:4]):
        assert_state_equal(left.state_dict(), right.state_dict())
    assert_state_equal(capture_rng_state(), final_rng)
    for field in ('sampler_state', 'optimizer_width_selection_counts', 'metrics_accumulator_state'):
        assert_state_equal(full[-1][field], restored[-1][field])


@pytest.mark.parametrize('damage', ['marker', 'missing_marker', 'history', 'version'])
def test_corrupt_compact_checkpoint_is_rejected_before_model_mutation(tmp_path, damage):
    from src.utils.config import ConfigError
    from test_optimizer_ownership_resume import packed_fixture, save, load, assert_state_equal
    bundle = packed_fixture(tmp_path)
    train_with_journal(bundle, stop=2)
    path = tmp_path / 'bad.pt'
    save(bundle, path)
    saved = torch.load(path, weights_only=False)
    metrics = saved['metrics_accumulator_state']
    if damage == 'marker': metrics['optimizer_last_attempt_id'] = 'global:9:-:-:g250'
    elif damage == 'missing_marker': del metrics['optimizer_last_attempt_id']
    elif damage == 'history': metrics['optimizer_attempt_ids'] = []
    else: metrics['schema_version'] = 3
    torch.save(saved, path)
    target = packed_fixture(tmp_path)
    before = [copy.deepcopy(x.state_dict()) for x in target[1:4]]
    with pytest.raises(ConfigError, match='metrics'):
        load(target, path)
    for value, expected in zip(target[1:4], before):
        assert_state_equal(value.state_dict(), expected)


def test_pre_mutation_failure_rolls_back_marker_and_retry_repairs_failed_row(tmp_path):
    from test_optimizer_ownership_resume import packed_fixture, save, load, assert_state_equal
    full = packed_fixture(tmp_path / 'full', 'S1')
    train_with_journal(full)
    source = packed_fixture(tmp_path / 'failed', 'S1')
    train_with_journal(source, stop=2)
    path = tmp_path / 'retry.pt'
    save(source, path)
    durable_bytes = path.read_bytes()
    def fail(*args, **kwargs):
        raise RuntimeError('injected forward failure')
    hook = source[1].register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match='injected forward failure'):
        train_with_journal(source)
    hook.remove()
    metrics = source[-1]['metrics_accumulator_state']
    assert metrics['attempted_optimizer_steps'] == 2
    assert metrics['optimizer_last_attempt_id'].startswith('global:1:')
    assert path.read_bytes() == durable_bytes
    restored = packed_fixture(tmp_path / 'failed', 'S1')
    load(restored, path)
    journal = train_with_journal(restored)
    assert len([r for r in journal.iter_rows() if r['split'] == 'train']) == 8
    assert_state_equal(full[-1]['metrics_accumulator_state'], restored[-1]['metrics_accumulator_state'])
    assert_state_equal(full[1].state_dict(), restored[1].state_dict())
