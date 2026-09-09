"""Saved phase-6 diagnostics, measured histories, and ordinary-validation evidence."""
import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from test_optimizer_ownership_resume import ARMS, packed_fixture, train
from test_optimizer_ownership import assert_state_equal
from src.training import optimizer_state
from src.utils.config import ConfigError


@pytest.mark.parametrize('arm', ARMS)
def test_storage_reads_actual_tensors_without_allocating_lazy_state(tmp_path, arm):
    bundle = packed_fixture(tmp_path, arm)
    opt = bundle[2]
    before = copy.deepcopy(opt.state_dict())
    empty = optimizer_state.measure_optimizer_storage(opt, step=0)
    assert empty['moment_elements'] == empty['counter_elements'] == 0
    assert_state_equal(before, opt.state_dict())
    train(bundle, stop=1)
    before = copy.deepcopy(opt.state_dict())
    measured = optimizer_state.measure_optimizer_storage(opt, step=1)
    optimizers = [entry.optimizer for entry in opt.entries] if hasattr(opt, 'entries') else [opt]
    tensors = [value for item in optimizers for state in item.state.values() for value in state.values() if torch.is_tensor(value)]
    assert measured['total_bytes'] == sum(t.numel() * t.element_size() for t in tensors)
    assert measured['moment_elements'] == sum(t.numel() for item in optimizers for state in item.state.values() for key, t in state.items() if key != 'step' and torch.is_tensor(t))
    assert sum(row['bytes'] for row in measured['components']) == measured['total_bytes']
    assert all(row['dtype'] == 'torch.float32' for row in measured['components'])
    if arm == 'C3':
        assert len(measured['owners']) == 5
        assert all(row['allocated_histories'] == 0 for row in empty['owners'])
        active = opt.active_owner_ids(bundle[-1]['optimizer_last_active_granularity'])
        assert all((row['allocated_histories'] > 0) == (row['owner_id'] in active) for row in measured['owners'])
    assert_state_equal(before, opt.state_dict())


@pytest.mark.parametrize('arm', ARMS)
def test_committed_saved_trace_and_summary_reconcile(tmp_path, arm):
    from src.utils.metrics import append_optimizer_ownership_observation
    from src.training.run import build_ownership_run_summary
    bundle = packed_fixture(tmp_path, arm)
    config, model, opt, clock, batches, state = bundle
    opt._ownership_observer = lambda: append_optimizer_ownership_observation(config, state, train_dataloader=batches)
    train(bundle)
    root = Path(config['run']['output_dir'])
    rows = [json.loads(line) for line in (root / 'optimizer_ownership_trace.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == list(range(1, 9))
    for row in rows:
        assert row['schema_version'] == 1
        assert row['contract_hash'] == config['optimizer_ownership_contract_hash']
        assert row['scheduler_position'] == row['step']
        assert sum(row['width_selection_counts'].values()) == row['step']
        assert row['tokens_seen'] == row['step'] * 8
        assert row['batch_sha256'] == hashlib.sha256(__import__('numpy').asarray(row['sample_ids'], dtype='<u8').tobytes()).hexdigest()
    assert rows[-1]['epoch'] == 4 and rows[-1]['batch_index'] == 0
    fields = build_ownership_run_summary(config, model, opt, state)
    assert fields['optimizer_ownership_schema_version'] == 1
    assert fields['optimizer_ownership']['accounting_reconciled']
    assert fields['optimizer_ownership']['expected_exposure']['label'] == 'uniform replacement expectation; realized counts are random'
    assert fields['optimizer_ownership']['storage']['total_bytes'] > 0
    assert fields['optimizer_ownership']['resources']['peak_allocated_bytes'] is None
    assert fields['optimizer_ownership']['resources']['measurement_complete'] is False
    assert (root / 'optimizer_ownership_clipping.jsonl').exists() == (arm in ('C1', 'C3'))


@pytest.mark.parametrize('arm', ['C1', 'C3'])
def test_clipping_active_denominators_and_saved_plot_series(tmp_path, arm):
    from src.utils.metrics import append_optimizer_ownership_observation, write_json_artifact
    from src.training.run import build_ownership_run_summary
    from src.evaluation.optimizer_ownership import report_run_artifacts
    bundle = packed_fixture(tmp_path, arm)
    config, model, opt, clock, batches, state = bundle
    opt._ownership_observer = lambda: append_optimizer_ownership_observation(config, state, train_dataloader=batches)
    train(bundle)
    root = Path(config['run']['output_dir'])
    summary = {'run_id': config['run']['run_id'], **build_ownership_run_summary(config, model, opt, state)}
    write_json_artifact(root / 'run_summary.json', summary)
    import csv
    with (root / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'split', 'granularity', 'loss', 'perplexity'])
        writer.writeheader()
        for step, loss in ((4, 2.), (8, 1.)):
            writer.writerow(dict(step=step, split='validation', granularity='g250', loss=loss, perplexity=math.exp(loss)))
    report = report_run_artifacts(root, tmp_path / 'plots')
    assert report['trajectories']['validation:g250']['loss'] == [2., 1.]
    assert report['resources']['measurement_complete'] is False
    assert report['resource_label'] == 'incomplete measurements'
    clipping = [json.loads(line) for line in (root / 'optimizer_ownership_clipping.jsonl').read_text().splitlines()]
    for width, groups in report['clipping_by_width'].items():
        for owner, result in groups.items():
            values = [r['groups'][owner]['coefficient'] for r in clipping if r['width'] == width and r['groups'][owner]['active']]
            assert result['active_observations'] == len(values)
            assert result['frequency'] == (sum(v < 1 for v in values) / len(values) if values else None)
    assert len(report['figures']) == 6
    assert all(Path(path).stat().st_size > 100 for path in report['figures'])
    rows = (root / 'optimizer_ownership_trace.jsonl').read_text().splitlines()
    bad = json.loads(rows[-1]); bad['tokens_seen'] += 8
    rows[-1] = json.dumps(bad)
    (root / 'optimizer_ownership_trace.jsonl').write_text('\n'.join(rows) + '\n')
    with pytest.raises(ConfigError, match='tokens'):
        report_run_artifacts(root, tmp_path / 'bad-plots')


@pytest.mark.parametrize('arm,expected', [('S1', 1049728), ('S2', 4198912), ('C1', 1049728), ('C2', 3609088), ('C3', 1049728)])
def test_post_exposure_pinned_model_moment_elements(arm, expected):
    from test_optimizer_ownership import _config, training, WIDTHS
    from src.models.ffn import CatLlamaMLP, ModifiedLlamaMLP
    from src.models.wiring import ModifiedLlamaForCausalLM
    from src.training.steps import build_optimizer_and_scheduler
    config = _config(intermediate_size=256, hidden_size=64)
    config.vocab_size = 2048
    config.num_hidden_layers = 4
    model = ModifiedLlamaForCausalLM(config, mlp_cls=ModifiedLlamaMLP if arm.startswith('S') else CatLlamaMLP,
                                   mlp_kwargs={'gradient_membership_correction_enabled': False})
    assert sum(p.numel() for p in model.parameters()) == 524864
    scope = 'per_ffn_block' if arm == 'C3' else 'per_granularity' if arm in ('S2', 'C2') else 'shared'
    opt, clock = build_optimizer_and_scheduler(model, training(scope))
    for width in WIDTHS:
        opt.zero_grad(set_to_none=True)
        model.configure_subnetwork(width)
        tokens = torch.arange(1, 9).reshape(1, 8)
        model(input_ids=tokens, labels=tokens).loss.backward()
        if scope == 'per_ffn_block':
            for owner in opt.active_owner_ids(width): opt.optimizer_for(owner).step()
        elif scope == 'per_granularity': opt.optimizer_for(width).step()
        else: opt.step()
    measurement = optimizer_state.measure_optimizer_storage(opt, step=4)
    assert measurement['moment_elements'] == expected
    assert measurement['moment_bytes'] == expected * 4
