"""Long-history bookkeeping retains exact checkpoint and rollback semantics."""
import copy
import random

import pytest
import torch

from src.training.steps import snapshot_run_state
from src.utils.metrics import StreamingMetricsAccumulator


def test_incremental_attempt_order_and_resume():
    accumulator = StreamingMetricsAccumulator()
    attempts = list(range(500)) * 2
    random.Random(42).shuffle(attempts)
    seen = set()
    for index, attempt in enumerate(attempts):
        key = f'global:{attempt}:-:-:g250'
        accumulator.update([dict(split='train', step=attempt + 1,
            optimizer_action_id=key, optimizer_step_attempted=True,
            optimizer_step_committed=attempt % 3 != 0)])
        seen.add(key)
        saved = accumulator.state_dict()
        assert saved['optimizer_attempt_ids'] == sorted(seen)
        assert saved['attempted_optimizer_steps'] == len(seen)
        if index == 475:
            accumulator = StreamingMetricsAccumulator(saved)
    saved['optimizer_attempt_ids'].clear()
    assert accumulator.state_dict()['optimizer_attempt_ids'] == sorted(seen)
    assert accumulator.committed_optimizer_steps == sum(i % 3 != 0 for i in range(500))
    assert accumulator.failed_optimizer_attempts == sum(i % 3 == 0 for i in range(500))


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
