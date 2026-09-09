"""Campaign committed-boundary restore and continuation cost regression tests."""
import copy
import hashlib
import json

import pytest
import torch
from transformers import LlamaForCausalLM

from src.training import checkpointing as cp, steps
from src.training.modeling import build_model
from src.utils.config import ConfigError
from src.utils.reproducibility import capture_rng_state, restore_rng_state, stable_hash
from test_optimizer_ownership import runtime_fixture, assert_state_equal, WIDTHS

ARMS = ('ST-g250', 'ST-g500', 'ST-g750', 'ST-g1000', 'S1', 'S2', 'C1', 'C2', 'C3')


def fixture(tmp_path, arm='C3'):
    scope = 'per_ffn_block' if arm == 'C3' else 'per_granularity' if arm in ('S2', 'C2') else 'shared'
    config, model, opt, clock, batches, state = runtime_fixture(tmp_path, scope=scope)
    from src.utils.reproducibility import configure_strict_determinism
    configure_strict_determinism(config)
    if arm.startswith('S'):
        config['model']['variant'] = 'slicing'
        model = build_model(config)
    if arm.startswith('ST-'):
        model_config = copy.deepcopy(model.config)
        model_config.intermediate_size = 16 * (WIDTHS.index(arm[3:]) + 1)
        model = LlamaForCausalLM(model_config)
        config['run']['model_family'] = 'standalone'
        config['run']['sampling_mode'] = 'standalone'
        config['model']['granularities'] = ['g1000']
        config['model']['granularity_sampling_mode'] = None
        config['training']['optimizer_state_contract']['ordered_granularities'] = ['g1000']
        state = cp.build_initial_continuation_state(config)
    opt, clock = steps.build_optimizer_and_scheduler(model, config['training'])
    # Synthetic diagnostic identity; production materialization validates the full protocol.
    config['optimizer_ownership_contract'] = {'schema_version': 1, 'arm_id': arm, 'campaign_id': 'resume-test'}
    config['optimizer_ownership_contract_hash'] = stable_hash(config['optimizer_ownership_contract'])
    from src.utils.reproducibility import seed_training_randomness
    seed_training_randomness(config)
    return config, model, opt, clock, batches, state


def train(bundle, stop=None, trace=None):
    config, model, opt, clock, batches, state = bundle
    from src.training.data import restore_packed_sampler_state
    if state.get('sampler_state') is not None:
        restore_packed_sampler_state(batches, state['sampler_state'])
    def committed(**kw):
        if trace is not None:
            trace.append((kw['step'], copy.deepcopy(state.get('global_sampling_state')), state['epoch'], state['batch_index']))
        if kw['step'] == stop:
            raise StopIteration('diagnostic boundary')
    try:
        steps.train_for_steps(config, model, batches, [], opt, clock, torch.device('cpu'), run_state=state, successful_step_callback=committed)
    except StopIteration:
        pass


def save(bundle, path):
    config, model, opt, clock, _, state = bundle
    cp.save_model_checkpoint(config, model, opt, clock, path,
        {'checkpoint_status': 'latest', 'checkpoint_metric': None, 'checkpoint_metric_value': None, 'checkpoint_selection_step': None}, state)


def load(bundle, path):
    config, model, opt, clock, _, state = bundle
    restored = cp.load_checkpoint_state(path, model, opt, clock, config=config, train_dataloader=bundle[4])
    state.clear()
    state.update(restored)


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('boundary', [1, 2, 3])
def test_all_arm_exact_resume_across_epoch_boundary(tmp_path, arm, boundary):
    source = fixture(tmp_path, arm)
    train(source, stop=boundary)
    path = tmp_path / 'latest.pt'
    save(source, path)
    trace = []
    train(source, trace=trace)
    restored = fixture(tmp_path, arm)
    load(restored, path)
    resumed_trace = []
    train(restored, trace=resumed_trace)
    assert trace == resumed_trace
    for left, right in zip(source[1:4], restored[1:4]):
        assert_state_equal(left.state_dict(), right.state_dict())
    for key in ('step', 'tokens_seen', 'content_tokens_seen', 'microstep', 'epoch', 'batch_index', 'global_sampling_state'):
        assert_state_equal(source[-1][key], restored[-1][key])


def optimizer_payload(payload):
    collection = payload.get('optimizer_state_collection')
    if collection:
        return (collection.get('ordered_owners') or collection['ordered_entries'])[0]['state_dict']
    return payload['optimizer_state_dict']


@pytest.mark.parametrize('arm', ['S1', 'S2', 'C1', 'C2', 'C3'])
@pytest.mark.parametrize('damage', ['missing', 'shape', 'dtype', 'negative', 'counter', 'kwargs', 'mapping', 'model', 'rng', 'clock', 'tokens', 'identity', 'purpose', 'schema', 'metrics'])
def test_rejects_corruption_without_mutating_live_bundle(tmp_path, arm, damage):
    source = fixture(tmp_path, arm)
    train(source, stop=3)
    path = tmp_path / 'latest.pt'
    save(source, path)
    payload = torch.load(path, weights_only=False)
    op = optimizer_payload(payload)
    # The first width may still be unexposed; choose a populated history.
    if not op['state']:
        collection = payload['optimizer_state_collection']
        op = next(e['state_dict'] for e in collection['ordered_entries'] if e['state_dict']['state'])
    history = next(iter(op['state'].values()))
    if damage == 'missing': op['state'].pop(next(iter(op['state'])))
    elif damage == 'shape': history['exp_avg'] = torch.zeros(7)
    elif damage == 'dtype': history['exp_avg'] = history['exp_avg'].double()
    elif damage == 'negative': history['exp_avg_sq'].fill_(-1)
    elif damage == 'counter': history['step'].fill_(1.5)
    elif damage == 'kwargs': op['param_groups'][0]['eps'] = 0.3
    elif damage == 'mapping': op['param_groups'][0]['params'].reverse()
    elif damage == 'model': next(iter(payload['model_state_dict'].values())).fill_(float('nan'))
    elif damage == 'rng': payload['reproducibility']['rng_states_by_rank'][0]['python'] = ()
    elif damage == 'clock': payload['scheduler_state_dict']['bad_field'] = True
    elif damage == 'tokens': payload['tokens_seen'] += 1
    elif damage == 'identity': payload['optimizer_ownership_contract']['arm_id'] = 'other'
    elif damage == 'purpose': payload['checkpoint_kind'] = 'model_only_evaluation'
    elif damage == 'schema': payload['optimizer_ownership_checkpoint_schema_version'] = 99
    elif damage == 'metrics': payload['metrics_accumulator_state'] = {'bad': True}
    torch.save(payload, path)
    restored = fixture(tmp_path, arm)
    before = [copy.deepcopy(x.state_dict()) for x in restored[1:4]]
    rng = capture_rng_state()
    with pytest.raises(ConfigError): load(restored, path)
    for value, expected in zip(restored[1:4], before): assert_state_equal(value.state_dict(), expected)
    assert_state_equal(capture_rng_state(), rng)


def test_install_failure_rolls_back_entire_bundle(tmp_path, monkeypatch):
    source = fixture(tmp_path)
    train(source, stop=3)
    path = tmp_path / 'latest.pt'
    save(source, path)
    restored = fixture(tmp_path)
    before = [copy.deepcopy(x.state_dict()) for x in restored[1:4]]
    rng = capture_rng_state()
    original = restored[3].load_state_dict
    calls = 0
    def fail_once(value):
        nonlocal calls
        calls += 1
        original(value)
        if calls == 1: raise RuntimeError('installation failed after mutation')
    monkeypatch.setattr(restored[3], 'load_state_dict', fail_once)
    with pytest.raises(RuntimeError, match='installation failed'): load(restored, path)
    for value, expected in zip(restored[1:4], before): assert_state_equal(value.state_dict(), expected)
    assert_state_equal(capture_rng_state(), rng)


def test_attempt_ledger_preserves_replay_and_incomplete_costs(tmp_path):
    from src.training.run import ResourceAttemptLedger
    ledger = ResourceAttemptLedger(tmp_path, run_id='run')
    ledger.observe('first', sequence=1, elapsed_seconds=3, peak_allocated_bytes=20, peak_reserved_bytes=30, attempted_steps=2, status='running', source_checkpoint=None)
    ledger.observe('first', sequence=2, elapsed_seconds=5, peak_allocated_bytes=40, peak_reserved_bytes=50, attempted_steps=4, status='failed', source_checkpoint=None)
    ledger.observe('first', sequence=1, elapsed_seconds=3, peak_allocated_bytes=20, peak_reserved_bytes=30, attempted_steps=2, status='running', source_checkpoint=None)
    ledger.observe('replay', sequence=1, elapsed_seconds=7, peak_allocated_bytes=30, peak_reserved_bytes=45, attempted_steps=3, status='running', source_checkpoint={'step': 1, 'sha256': 'old'})
    restored = ResourceAttemptLedger(tmp_path, run_id='run')
    totals = restored.summary()
    assert totals['elapsed_seconds'] == 12
    assert totals['peak_allocated_bytes'] == 40
    assert totals['peak_reserved_bytes'] == 50
    assert totals['measurement_complete'] is False
    assert totals['attempted_steps'] == 7
    restored.validate_watermark({'first': 1})
    assert restored.summary() == totals
    with pytest.raises(ConfigError): restored.validate_watermark({'first': 3})


def packed_fixture(tmp_path, arm='C3'):
    from src.training.packed_corpus import RepeatingNoPaddingDistributedBatchSampler, REPEATED_EPOCH_ORDER_VERSION
    bundle = list(fixture(tmp_path, arm))
    config = bundle[0]
    config['dataset'].update(mode='packed_mmap', data_seed=42, optimizer_iteration={
        'mode': 'repeat_epochs', 'epoch_order': 'deterministic_per_epoch',
        'ordering_policy_version': REPEATED_EPOCH_ORDER_VERSION,
        'aligned_epoch_samples': 2, 'aligned_epoch_tokens': 16,
        'excluded_tail_samples': 1, 'excluded_tail_tokens': 8,
    })
    sampler = RepeatingNoPaddingDistributedBatchSampler(3, 1, 0, 1, planned_sample_count=8,
        epoch_sample_count=2, corpus_hash='fixture-corpus', optimizer_training_manifest_hash='fixture-role')
    dataset = [{'input_ids': torch.arange(1+i, 9+i), 'labels': torch.arange(1+i, 9+i)} for i in range(3)]
    bundle[4] = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return bundle


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('boundary', [1, 2, 3])
def test_repeating_packed_sampler_exact_batches_actions_rng_and_state(tmp_path, arm, boundary):
    full = packed_fixture(tmp_path, arm)
    full_trace, full_batches = [], []
    full[1].register_forward_pre_hook(lambda module, args, kwargs: full_batches.append(kwargs['input_ids'].clone()), with_kwargs=True)
    train(full, trace=full_trace)
    final_rng = capture_rng_state()
    source = packed_fixture(tmp_path, arm)
    prefix_trace = []
    train(source, stop=boundary, trace=prefix_trace)
    path = tmp_path / 'packed.pt'
    save(source, path)
    restored = packed_fixture(tmp_path, arm)
    before_rng = capture_rng_state()
    load(restored, path)
    suffix_trace, suffix_batches = [], []
    restored[1].register_forward_pre_hook(lambda module, args, kwargs: suffix_batches.append(kwargs['input_ids'].clone()), with_kwargs=True)
    train(restored, trace=suffix_trace)
    assert prefix_trace + suffix_trace == full_trace
    assert_state_equal(suffix_batches, full_batches[boundary:])
    for left, right in zip(full[1:4], restored[1:4]): assert_state_equal(left.state_dict(), right.state_dict())
    assert_state_equal(capture_rng_state(), final_rng)
    assert restored[-1]['sampler_state'] == full[-1]['sampler_state']
    assert restored[-1]['epoch'] == 4 and restored[-1]['batch_index'] == 0


@pytest.mark.parametrize('field', ['total_cursor', 'fixed_epoch_set_hash', 'permutation_hash', 'epoch_index', 'epoch_sample_count', 'planned_sample_count'])
def test_packed_sampler_corruption_is_rejected_before_model_or_sampler_mutation(tmp_path, field):
    source = packed_fixture(tmp_path)
    train(source, stop=2)
    path = tmp_path / 'packed.pt'
    save(source, path)
    payload = torch.load(path, weights_only=False)
    value = payload['sampler_state'][field]
    payload['sampler_state'][field] = value + 1 if isinstance(value, int) else 'bad'
    torch.save(payload, path)
    restored = packed_fixture(tmp_path)
    before = [copy.deepcopy(x.state_dict()) for x in restored[1:4]]
    sampler_before = restored[4].batch_sampler.state_dict()
    with pytest.raises(ConfigError): load(restored, path)
    for item, expected in zip(restored[1:4], before): assert_state_equal(item.state_dict(), expected)
    assert restored[4].batch_sampler.state_dict() == sampler_before


def test_scientific_rows_rewind_but_resource_costs_survive(tmp_path):
    from src.training.run import ResourceAttemptLedger
    from src.utils.metrics import MetricsJournal
    ledger = ResourceAttemptLedger(tmp_path, run_id='run')
    ledger.observe('old', sequence=1, elapsed_seconds=5, peak_allocated_bytes=None, peak_reserved_bytes=None, attempted_steps=5, status='failed', source_checkpoint=None)
    original_ledger = ledger.path.read_bytes()
    path = tmp_path / 'optimizer_ownership_trace.jsonl'
    path.write_text(''.join(json.dumps({'step': n}) + '\n' for n in (1, 2, 3, 4)) + '{partial')
    cp.reconcile_ownership_scientific_rows(tmp_path, 2)
    assert [json.loads(line)['step'] for line in path.read_text().splitlines()] == [1, 2]
    assert len(list(tmp_path.glob('*.non_durable.*'))) == 1
    assert ledger.path.read_bytes() == original_ledger
    journal = MetricsJournal(tmp_path)
    journal.append([{'run_id': 'run', 'model_family': 'nested', 'granularity': 'g250', 'perplexity': 2.718, 'wall_clock_seconds': 1., 'tokens_per_second': 8., 'peak_memory_bytes': 0, 'step': i, 'split': 'train', 'loss': 1., 'tokens_seen': i * 8} for i in (1, 2, 3)], force=True)
    resumed = MetricsJournal(tmp_path, checkpoint_step=2)
    assert max(int(row['step']) for row in resumed.read_all()) == 2


@pytest.mark.parametrize('arm', ['ST-g250', 'S1', 'S2', 'C1', 'C2', 'C3'])
def test_fresh_checkpoint_retains_true_lazy_absence(tmp_path, arm):
    bundle = fixture(tmp_path, arm)
    bundle[-1].update(cp.build_initial_continuation_state(bundle[0]))
    path = tmp_path / 'fresh.pt'
    save(bundle, path)
    restored = fixture(tmp_path, arm)
    load(restored, path)
    assert_state_equal(bundle[2].state_dict(), restored[2].state_dict())


@pytest.mark.parametrize('damage', ['extra_history', 'owner_order', 'descriptor', 'clock_rate', 'action_rng', 'action_count', 'model_dtype', 'scope', 'representation', 'model_only', 'resource', 'epoch', 'negative_counter'])
def test_additional_campaign_rejections(tmp_path, damage):
    source = packed_fixture(tmp_path)
    train(source, stop=1)
    path = tmp_path / 'bad.pt'
    save(source, path)
    payload = torch.load(path, weights_only=False)
    owners = payload['optimizer_state_collection']['ordered_owners']
    if damage == 'extra_history':
        state = owners[0]['state_dict']['state']
        state[max(state) + 1] = copy.deepcopy(next(iter(state.values())))
    elif damage == 'owner_order': owners.reverse()
    elif damage == 'descriptor': payload['optimizer_parameter_descriptors'][0]['canonical_name'] = 'bad'
    elif damage == 'clock_rate': payload['scheduler_state_dict']['scheduler_state_dict']['_last_lr'] = [0.8]
    elif damage == 'action_rng':
        import random
        for rng in [payload['reproducibility']['rng_state'], *payload['reproducibility']['rng_states_by_rank']]:
            rng['dedicated']['granularity_selection'] = random.Random(99).getstate()
    elif damage == 'action_count': payload['global_sampling_state']['exposure_counts']['g250'] += 1
    elif damage == 'model_dtype':
        key = next(iter(payload['model_state_dict']))
        payload['model_state_dict'][key] = payload['model_state_dict'][key].double()
    elif damage == 'scope': payload['optimizer_state_contract']['state_scope'] = 'shared'
    elif damage == 'representation': payload['optimizer_ownership_contract']['representation'] = 'slicing'
    elif damage == 'model_only': payload['optimizer_state_collection'] = None
    elif damage == 'resource': payload['resource_ledger_watermark'] = {'missing': 1}
    elif damage == 'epoch': payload['epoch'] += 1
    elif damage == 'negative_counter': next(iter(owners[0]['state_dict']['state'].values()))['step'].fill_(-1)
    torch.save(payload, path)
    restored = packed_fixture(tmp_path)
    before = [copy.deepcopy(x.state_dict()) for x in restored[1:4]]
    rng = capture_rng_state()
    with pytest.raises(ConfigError): load(restored, path)
    for value, expected in zip(restored[1:4], before): assert_state_equal(value.state_dict(), expected)
    assert_state_equal(capture_rng_state(), rng)


def test_runtime_resource_ledger_checkpoint_and_completed_reentry(tmp_path, monkeypatch):
    import src.training.run as run
    import src.evaluation.optimizer_ownership as campaign
    # Deliberately short budget/identity fixture; production fixed-matrix validation
    # is exercised separately by the campaign suite.
    monkeypatch.setattr(campaign, 'validate_materialized_config', lambda config: None)
    bundle = fixture(tmp_path)
    config, model = bundle[:2]
    config['run']['continuation']['enabled'] = True
    config['outputs']['metrics_flush_interval_steps'] = 1
    def dataloaders(config, *args, **kwargs):
        config['_validation_manifest'] = {'fixture': True}
        config['validation_manifest_hash'] = 'fixture-validation'
        return bundle[4], bundle[4]
    monkeypatch.setattr(run.training_data, 'build_dataloaders', dataloaders)
    run.run_training(config, model=model, tokenizer=object(), tokenized_dataset=[{}], device='cpu')
    ledger = run.ResourceAttemptLedger(config['run']['output_dir'], run_id=config['run']['run_id'])
    first = ledger.summary()
    assert first['attempted_steps'] == 8 and first['measurement_complete']
    assert first['peak_allocated_bytes'] is None
    checkpoint = torch.load(ledger.path.parent / 'checkpoints/latest.pt', weights_only=False)
    ledger.validate_watermark(checkpoint['resource_ledger_watermark'])
    run.run_training(config, model=fixture(tmp_path)[1], tokenizer=object(), tokenized_dataset=[{}], device='cpu')
    resumed = run.ResourceAttemptLedger(config['run']['output_dir'], run_id=config['run']['run_id'])
    assert resumed.summary()['attempt_count'] == 2
    assert resumed.summary()['attempted_steps'] == 8
    assert resumed.summary()['elapsed_seconds'] >= first['elapsed_seconds']


@pytest.mark.parametrize('scope', ['shared', 'per_granularity', 'per_ffn_block'])
def test_exposure_validation_rejects_allocated_inactive_histories(scope):
    from test_optimizer_ownership import RealFFNModel, training, backward, commit
    from src.training.optimizer_state import validate_campaign_optimizer
    model = RealFFNModel()
    optimizer, scheduler = steps.build_optimizer_and_scheduler(model, training(scope))
    backward(model, optimizer, 'g250')
    commit(optimizer, scheduler, 'g250')
    state = copy.deepcopy(optimizer.state_dict())
    counts = dict.fromkeys(WIDTHS, 0)
    counts['g250'] = 1
    kwargs = dict(widths=WIDTHS, width_counts=counts, learning_rates=[0.01])
    validate_campaign_optimizer(model, optimizer, state, **kwargs)
    if scope == 'per_ffn_block':
        absent = state['ordered_owners'][1]['state_dict']
        present = state['ordered_owners'][0]['state_dict']
        absent['state'][0] = copy.deepcopy(next(iter(present['state'].values())))
    else:
        saved = state['ordered_entries'][0]['state_dict'] if scope == 'per_granularity' else state
        pid = next(i for i in saved['param_groups'][0]['params'] if i not in saved['state'])
        saved['state'][pid] = {'step': torch.tensor(1.), 'exp_avg': torch.zeros_like(tuple(model.parameters())[pid]), 'exp_avg_sq': torch.zeros_like(tuple(model.parameters())[pid])}
    with pytest.raises(ConfigError, match='Impossible'):
        validate_campaign_optimizer(model, optimizer, state, **kwargs)


def test_pre_mutation_failure_restores_actions_sampler_and_rng(tmp_path, monkeypatch):
    source = packed_fixture(tmp_path)
    train(source, stop=1)
    model_before = copy.deepcopy(source[1].state_dict())
    opt_before = copy.deepcopy(source[2].state_dict())
    sampler_before = copy.deepcopy(source[-1]['sampler_state'])
    actions_before = copy.deepcopy(source[-1]['global_sampling_state'])
    rng_before = capture_rng_state()
    def reject(*args, **kwargs): raise RuntimeError('before mutation')
    monkeypatch.setattr(steps, 'clip_optimizer_gradients', reject)
    with pytest.raises(RuntimeError, match='before mutation'): train(source)
    assert not source[-1].get('optimizer_poisoned') and not source[-1]['update_in_flight']
    assert source[-1]['global_sampling_state'] == actions_before
    assert source[4].batch_sampler.state_dict() == sampler_before
    assert_state_equal(model_before, source[1].state_dict())
    assert_state_equal(opt_before, source[2].state_dict())
    assert_state_equal(capture_rng_state(), rng_before)
    assert all(p.grad is None for p in source[1].parameters())


def test_trainer_failure_records_cost_and_preserves_last_durable_checkpoint(tmp_path, monkeypatch):
    import src.training.run as run
    import src.evaluation.optimizer_ownership as campaign
    monkeypatch.setattr(campaign, 'validate_materialized_config', lambda config: None)
    bundle = fixture(tmp_path)
    config, model = bundle[:2]
    config['run']['continuation']['enabled'] = True
    config['outputs']['metrics_flush_interval_steps'] = 1
    def dataloaders(config, *args, **kwargs):
        config['_validation_manifest'] = {'fixture': True}
        config['validation_manifest_hash'] = 'fixture-validation'
        return bundle[4], bundle[4]
    monkeypatch.setattr(run.training_data, 'build_dataloaders', dataloaders)
    builder = run.training_steps.build_optimizer_and_scheduler
    hashes = []
    def build(*args, **kwargs):
        optimizer, scheduler = builder(*args, **kwargs)
        common = optimizer.optimizer_for('O-common')
        original = common.step
        calls = 0
        def fail_second():
            nonlocal calls
            calls += 1
            if calls == 2:
                path = tmp_path / 'per-granularity-optimizer-smoke-001/checkpoints/latest.pt'
                hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
                original()
                raise RuntimeError('owner changed weights then failed')
            original()
        common.step = fail_second
        return optimizer, scheduler
    monkeypatch.setattr(run.training_steps, 'build_optimizer_and_scheduler', build)
    with pytest.raises(RuntimeError, match='owner changed weights'):
        run.run_training(config, model=model, tokenizer=object(), tokenized_dataset=[{}], device='cpu')
    ledger = run.ResourceAttemptLedger(config['run']['output_dir'], run_id=config['run']['run_id'])
    assert ledger.summary()['attempted_steps'] == 2
    assert ledger.summary()['measurement_complete']
    record = next(iter(ledger.attempts.values()))
    assert record['status'] == 'failed'
    assert record['failure']['stage'] == 'optimizer_step:O-common'
    assert record['failure']['last_durable_checkpoint_step'] == 1
    path = ledger.path.parent / 'checkpoints/latest.pt'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[0]
    assert torch.load(path, weights_only=False)['step'] == 1


@pytest.mark.parametrize('arm', ARMS)
def test_terminal_validation_counts_hash_and_idempotent_completion(tmp_path, arm, monkeypatch):
    from src.training.run import complete_ownership_terminal
    from src.utils.model_size import model_parameter_counts
    bundle = packed_fixture(tmp_path, arm)
    train(bundle)
    config, model, opt, clock, batches, state = bundle
    config['validation_manifest_hash'] = 'ordinary-fixture'
    config['run']['continuation']['enabled'] = True
    evaluation = [{'input_ids': torch.arange(1, 9).reshape(1, 8), 'labels': torch.tensor([[1, 2, -100, -100, 5, 6, 7, 8]])},
                  {'input_ids': torch.arange(2, 10).reshape(1, 8), 'labels': torch.arange(2, 10).reshape(1, 8)}]
    from src.evaluation.validation import evaluate_validation_loss
    labels = [arm[3:]] if arm.startswith('ST-') else WIDTHS
    local = None if arm.startswith('ST-') else labels[0]
    losses = [evaluate_validation_loss(model, [batch], 'cpu', granularity=local)['loss'] for batch in evaluation]
    expected_loss = (losses[0] * 5 + losses[1] * 7) / 12
    result = complete_ownership_terminal(config, model, opt, clock, state, evaluation, torch.device('cpu'))
    assert result['schema_version'] == 1 and result['evaluation_role'] == 'ordinary_validation'
    assert result['evaluation_target_tokens'] == 12 and result['evaluation_examples'] == 2
    assert [row['width'] for row in result['endpoints']] == list(labels)
    assert result['endpoints'][0]['loss'] == pytest.approx(expected_loss)
    assert result['endpoints'][0]['non_embedding_parameters'] == model_parameter_counts(model, granularity=local)['non_embedding_parameters']
    assert result['content_hash'] == stable_hash({k: v for k, v in result.items() if k != 'content_hash'})
    path = __import__('pathlib').Path(result['checkpoint_path'])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert result['checkpoint_sha256'] == digest
    before = [copy.deepcopy(x.state_dict()) for x in bundle[1:4]]
    def forbidden(*args, **kwargs): raise AssertionError('completion must reuse valid sidecar')
    import src.evaluation.validation as validation
    monkeypatch.setattr(validation, 'evaluate_validation_per_granularity', forbidden)
    assert complete_ownership_terminal(config, model, opt, clock, state, evaluation, torch.device('cpu')) == result
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    for item, expected in zip(bundle[1:4], before): assert_state_equal(item.state_dict(), expected)


@pytest.mark.parametrize('failure', ['checkpoint', 'evaluation', 'sidecar'])
def test_terminal_failure_and_recovery_preserve_checkpoint_and_take_no_steps(tmp_path, monkeypatch, failure):
    import src.training.run as run
    import src.evaluation.validation as validation
    bundle = packed_fixture(tmp_path)
    train(bundle)
    config, model, opt, clock, batches, state = bundle
    config['validation_manifest_hash'] = 'ordinary-fixture'
    config['run']['continuation']['enabled'] = True
    evaluation = [{'input_ids': torch.arange(1, 9).reshape(1, 8), 'labels': torch.arange(1, 9).reshape(1, 8)}]
    def fail(*args, **kwargs): raise RuntimeError('injected terminal failure')
    with monkeypatch.context() as patch:
        if failure == 'checkpoint': patch.setattr(cp, 'save_model_checkpoint', fail)
        elif failure == 'evaluation': patch.setattr(validation, 'evaluate_validation_per_granularity', fail)
        else: patch.setattr(run, 'write_json_artifact', fail)
        with pytest.raises((RuntimeError, OSError), match='injected terminal failure'):
            run.complete_ownership_terminal(config, model, opt, clock, state, evaluation, torch.device('cpu'))
    root = __import__('pathlib').Path(config['run']['output_dir'])
    assert not (root / 'terminal_validation_results.json').exists()
    checkpoint = root / 'checkpoints/latest.pt'
    before_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint.exists() else None
    for entry in opt.entries: monkeypatch.setattr(entry.optimizer, 'step', fail)
    result = run.complete_ownership_terminal(config, model, opt, clock, state, evaluation, torch.device('cpu'))
    if before_hash: assert result['checkpoint_sha256'] == before_hash
    # A valid content hash alone must not bless a stale or changed evaluation.
    sidecar = root / 'terminal_validation_results.json'
    bad = json.loads(sidecar.read_text()); bad['evaluation_role'] = 'final_holdout'
    bad['content_hash'] = stable_hash({k: v for k, v in bad.items() if k != 'content_hash'})
    sidecar.write_text(json.dumps(bad))
    with pytest.raises(ConfigError):
        run.complete_ownership_terminal(config, model, opt, clock, state, evaluation, torch.device('cpu'))


def test_runtime_terminal_sidecar_recovery_costs_and_zero_training_reentry(tmp_path, monkeypatch):
    import src.training.run as run
    import src.evaluation.optimizer_ownership as campaign
    from pathlib import Path
    monkeypatch.setattr(campaign, 'validate_materialized_config', lambda config: None)
    bundle = fixture(tmp_path)
    config = bundle[0]
    config['run']['continuation']['enabled'] = True
    config['outputs']['metrics_flush_interval_steps'] = 1
    evaluation = [{'input_ids': torch.arange(1, 9).reshape(1, 8), 'labels': torch.arange(1, 9).reshape(1, 8)}]
    def loaders(config, *args, **kwargs):
        config['_validation_manifest'] = {'fixture': True}
        config['validation_manifest_hash'] = 'fixture-validation'
        return fixture(tmp_path)[4], evaluation
    monkeypatch.setattr(run.training_data, 'build_dataloaders', loaders)
    original = run.write_json_artifact
    def fail_sidecar(path, *args, **kwargs):
        if Path(path).name == 'terminal_validation_results.json': raise RuntimeError('sidecar publication failed')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(run, 'write_json_artifact', fail_sidecar)
    with pytest.raises(RuntimeError, match='sidecar publication failed'):
        run.run_training(config, model=bundle[1], tokenizer=object(), tokenized_dataset=[{}], device='cpu')
    root = Path(config['run']['output_dir'])
    checkpoint = root / 'checkpoints/latest.pt'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert not (root / 'terminal_validation_results.json').exists()
    before = run.ResourceAttemptLedger(root, run_id=config['run']['run_id']).summary()
    monkeypatch.setattr(run, 'write_json_artifact', original)
    def forbidden(*args, **kwargs): raise AssertionError('completion-only must bypass training')
    monkeypatch.setattr(run.training_steps, 'train_for_steps', forbidden)
    for _ in range(2):
        run.run_training(config, model=fixture(tmp_path)[1], tokenizer=object(), tokenized_dataset=[{}], device='cpu')
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    after = run.ResourceAttemptLedger(root, run_id=config['run']['run_id']).summary()
    assert after['attempt_count'] == 3 and after['attempted_steps'] == before['attempted_steps'] == 8
    assert after['elapsed_seconds'] > before['elapsed_seconds'] and after['measurement_complete']
    assert json.loads((root / 'terminal_validation_results.json').read_text())['checkpoint_sha256'] == digest


def test_terminal_rejects_replaced_checkpoint_model_before_evaluating(tmp_path, monkeypatch):
    import src.training.run as run
    import src.evaluation.validation as validation
    bundle = packed_fixture(tmp_path)
    train(bundle)
    config, model, opt, clock, batches, state = bundle
    config['run']['continuation']['enabled'] = True
    config['validation_manifest_hash'] = 'ordinary-fixture'
    path = tmp_path / 'terminal.pt'
    save(bundle, path)
    state['latest_checkpoint_path'] = str(path)
    state['latest_checkpoint_step'] = state['last_completed_step']
    payload = torch.load(path, weights_only=False)
    next(iter(payload['model_state_dict'].values())).add_(1)
    torch.save(payload, path)
    def forbidden(*args, **kwargs): raise AssertionError('invalid checkpoint must not evaluate')
    monkeypatch.setattr(validation, 'evaluate_ownership_terminal', forbidden)
    with pytest.raises(ConfigError, match='evaluated model'):
        run.complete_ownership_terminal(config, model, opt, clock, state, [], torch.device('cpu'))
