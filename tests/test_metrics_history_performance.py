"""Long-history bookkeeping retains exact checkpoint and rollback semantics."""
import copy

import pytest
import torch

from src.training.steps import snapshot_run_state
from src.utils.metrics import StreamingMetricsAccumulator


def test_step_scoped_bounded_accounting_and_resume():
    accumulator = StreamingMetricsAccumulator()
    for attempt in range(500):
        # Held actions must still account for each optimizer update.
        key = f'global:{attempt // 4}:-:-:g250'
        record = dict(split='train', step=attempt + 1,
                      optimizer_action_id=key, optimizer_step_attempted=True,
                      optimizer_step_committed=attempt % 3 != 0)
        accumulator.update([record, record])
        saved = accumulator.state_dict()
        assert 'optimizer_attempt_ids' not in saved
        assert saved['attempted_optimizer_steps'] == attempt + 1
        if attempt == 475:
            accumulator = StreamingMetricsAccumulator(saved)
    assert accumulator.committed_optimizer_steps == sum(i % 3 != 0 for i in range(500))
    assert accumulator.failed_optimizer_attempts == sum(i % 3 == 0 for i in range(500))


def test_rollback_snapshot_retains_live_sign_state_without_copying():
    sign_state = {'coordinates': torch.ones(8)}
    state = {'sign_dynamics_state': sign_state, 'nested': {'values': [1]}}
    snapshot = snapshot_run_state(state)
    assert snapshot['sign_dynamics_state'] is sign_state
    state['nested']['values'].append(2)
    assert snapshot['nested']['values'] == [1]


@pytest.mark.parametrize('ids', [['a', 'b'], [], [{'mutable': [1]}]])
def test_rollback_snapshot_matches_deepcopy_and_preserves_aliases(ids):
    state = {'metrics_accumulator_state': {'optimizer_attempt_ids': ids,
             'trailing': [{'loss': 2.0}]}, 'same_ids': ids, 'nested': {'values': [1]}}
    snapshot = snapshot_run_state(state)
    assert snapshot == copy.deepcopy(state)
    assert snapshot['same_ids'] is snapshot['metrics_accumulator_state']['optimizer_attempt_ids']
    assert snapshot['same_ids'] is not ids
    state['nested']['values'].append(2)
    ids.append('new')
    assert snapshot['nested']['values'] == [1]
    assert 'new' not in snapshot['same_ids']
    if ids and isinstance(ids[0], dict):
        ids[0]['mutable'].append(2)
        assert snapshot['same_ids'][0]['mutable'] == [1]


@pytest.mark.parametrize('arm', ['ST-g250', 'S1', 'S2', 'C1', 'C2', 'C3'])
def test_campaign_model_only_best_checkpoint(tmp_path, arm):
    from test_optimizer_ownership_resume import packed_fixture, train, assert_state_equal
    from src.training.checkpointing import save_model_checkpoint
    bundle = packed_fixture(tmp_path, arm)
    config, model, optimizer, scheduler, batches, state = bundle
    train(bundle, stop=1)
    path = tmp_path / 'best.pt'
    fields = dict(checkpoint_status='best_eval', checkpoint_metric='validation_loss',
                  checkpoint_metric_value=2.0, checkpoint_selection_step=1)
    save_model_checkpoint(config, model, None, None, path, fields, state)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['optimizer_storage'] is None
    assert saved['optimizer_state_dict'] is None
    assert saved['scheduler_state_dict'] is None
    assert_state_equal(saved['model_state_dict'], model.state_dict())
