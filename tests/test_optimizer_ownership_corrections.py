"""Corrected six-arm real update paths; reference never uses correction helpers."""
import copy
import hashlib
import json

import pytest
import torch

from src.training import steps, modeling
from src.training.optimizer_state import BlockOptimizerCollection, PerGranularityOptimizerCollection
from src.utils.config import ConfigError
from src.utils.reproducibility import stable_hash
from test_optimizer_ownership import runtime_fixture, assert_state_equal, WIDTHS, OWNERS
from test_optimizer_ownership_resume import train, save, load

SCOPES = ('shared', 'per_granularity', 'per_ffn_block')
FACTORS = (1., 4/3, 2., 4.)


def bundle(tmp_path, scope, mode, device='cpu'):
    result = runtime_fixture(tmp_path, scope=scope, correction_mode=mode, device=device)
    config = result[0]
    from src.evaluation.optimizer_ownership import membership_correction_contract
    # Small diagnostic contracts; production full contracts are tested separately.
    config['optimizer_ownership_contract'] = {
        'schema_version': 1, 'campaign_id': 'correction-test',
        'arm_id': dict(zip(SCOPES, ('C1', 'C2', 'C3')))[scope] + '-' + mode.upper(),
        'correction': membership_correction_contract(mode),
    }
    config['optimizer_ownership_contract_hash'] = stable_hash(config['optimizer_ownership_contract'])
    from src.utils.reproducibility import seed_training_randomness
    seed_training_randomness(config)
    return result


def parameter_factor(name):
    return FACTORS[int(name.rsplit('.', 1)[1])] if '_blocks.' in name else 1.


@pytest.mark.parametrize('scope', SCOPES)
@pytest.mark.parametrize('mode', ('gmc', 'lmc'))
@pytest.mark.parametrize('selected', WIDTHS)
@pytest.mark.parametrize('bias', (False, True))
def test_real_six_arm_updates_match_independent_adamw(tmp_path, monkeypatch, scope, mode, selected, bias):
    original_config = modeling.build_llama_config
    def with_bias(config):
        result = original_config(config)
        result.mlp_bias = bias
        return result
    monkeypatch.setattr(modeling, 'build_llama_config', with_bias)
    config, actual, _, _, batches, state = runtime_fixture(
        tmp_path, scope=scope, correction_mode=mode, max_steps=4)
    actual = actual.double()
    optimizer, clock = steps.build_optimizer_and_scheduler(actual, config['training'])
    # Independent reference: no hooks, explicit gradient multipliers, explicit
    # per-parameter learning rates. Includes common down bias at base LR.
    reference_config = copy.deepcopy(config)
    reference_config['model']['membership_correction'] = False
    reference = modeling.build_model(reference_config).double()
    reference.load_state_dict(actual.state_dict())
    ref_optimizers = {}
    for owner in (WIDTHS if scope == 'per_granularity' else ('shared',)):
        ref_optimizers[owner] = torch.optim.AdamW([
            {'params': [p], 'lr': 0.001, 'factor': parameter_factor(name) if mode == 'lmc' else 1.}
            for name, p in reference.named_parameters()
        ], betas=(.9, .95), eps=1e-8, weight_decay=.1)
    for layer in actual.model.layers:
        assert layer.mlp.gradient_membership_counts == [4, 3, 2, 1]
        assert tuple(layer.mlp.gradient_membership_correction_scales) == FACTORS
    original_clip = steps.clip_optimizer_gradients
    previous = {}
    selection = iter(('g1000', selected, selected, selected))
    original_select = steps._select_optimizer_window_action
    def select(*args, **kwargs):
        action = original_select(*args, **kwargs)
        width = next(selection)
        action['granularities'] = [width]
        kwargs['run_state']['global_sampling_state']['held_granularity'] = width
        return action
    monkeypatch.setattr(steps, '_select_optimizer_window_action', select)
    reference_step = [0]
    active_width = [None]
    def clip(model, training, width, **kw):
        active_width[0] = width
        reference.zero_grad(set_to_none=True)
        reference.configure_subnetwork(width)
        tokens = batches[0]['input_ids']
        reference(input_ids=tokens, labels=tokens).loss.backward()
        for name, p in reference.named_parameters():
            if p.grad is not None:
                p.grad.mul_(parameter_factor(name))
        for (name, p), (_, q) in zip(actual.named_parameters(), reference.named_parameters()):
            assert (p.grad is None) == (q.grad is None)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-7, atol=1e-10)
        # Independent clipping partition from names, without ownership helpers.
        if scope == 'per_ffn_block':
            for quarter in (*range(4), None):
                params = [p for n, p in reference.named_parameters()
                          if (int(n.rsplit('.', 1)[1]) if '_blocks.' in n else None) == quarter]
                torch.nn.utils.clip_grad_norm_(params, 1.)
        else:
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
        observed = original_clip(model, training, width, **kw)
        for p, q in zip(actual.parameters(), reference.parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-7, atol=1e-10)
        ref_opt = ref_optimizers[width if scope == 'per_granularity' else 'shared']
        base_lr = (optimizer.current_learning_rates[0] if hasattr(optimizer, 'current_learning_rates')
                   else optimizer.param_groups[0]['lr'])
        for group in ref_opt.param_groups:
            group['lr'] = base_lr * group['factor']
        previous.clear()
        previous.update({n: p.detach().clone() for n, p in actual.named_parameters()})
        ref_opt.step()
        reference_step[0] += 1
        return observed
    monkeypatch.setattr(steps, 'clip_optimizer_gradients', clip)
    def committed(step, **kw):
        width = active_width[0]
        assert clock.position == step if hasattr(clock, 'position') else clock.last_epoch == step
        for (name, p), (_, q) in zip(actual.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(p, q, rtol=2e-7, atol=2e-10)
            if p.grad is None:
                assert torch.equal(p, previous[name])
            if scope == 'per_ffn_block':
                owner = 'O-' + 'ABCD'[int(name.rsplit('.', 1)[1])] if '_blocks.' in name else 'O-common'
                opt = optimizer.optimizer_for(owner)
            elif scope == 'per_granularity':
                opt = optimizer.optimizer_for(width)
            else:
                opt = optimizer
            ref_opt = ref_optimizers[width if scope == 'per_granularity' else 'shared']
            for key, value in opt.state.get(p, {}).items():
                torch.testing.assert_close(value, ref_opt.state[q][key], rtol=2e-7, atol=2e-10)
        if scope == 'per_ffn_block':
            assert optimizer.total_successful_updates == step
            assert optimizer.successful_update_counts['O-common'] == step
            for i, owner in enumerate(OWNERS[:4]):
                assert optimizer.successful_update_counts[owner] == (step if i <= WIDTHS.index(width) else 1)
    steps.train_for_steps(config, actual, batches, [], optimizer, clock, torch.device('cpu'),
                          run_state=state,
                          successful_step_callback=committed)
    assert reference_step[0] == 4


@pytest.mark.parametrize('scope', SCOPES)
@pytest.mark.parametrize('mode', ('gmc', 'lmc'))
@pytest.mark.parametrize('device', ('cpu', 'cuda'))
def test_corrected_resume(tmp_path, scope, mode, device):
    source = bundle(tmp_path, scope, mode, device)
    train(source, stop=3)
    path = tmp_path / 'latest.pt'
    save(source, path)
    trace = []
    train(source, trace=trace)
    restored = bundle(tmp_path, scope, mode, device)
    load(restored, path)
    resumed_trace = []
    train(restored, trace=resumed_trace)
    assert trace == resumed_trace
    for left, right in zip(source[1:4], restored[1:4]):
        assert_state_equal(left.state_dict(), right.state_dict())
    assert source[-1]['step'] == restored[-1]['step'] == 8


@pytest.mark.parametrize('failure', (*OWNERS, 'correction', 'scheduler', 'accounting'))
def test_c3_corrected_failure_keeps_durable_checkpoint(tmp_path, monkeypatch, failure):
    from src.training import checkpointing as cp
    result = bundle(tmp_path, 'per_ffn_block', 'lmc')
    config, model, optimizer, clock, _, state = result
    train(result, stop=1)
    path = tmp_path / 'latest.pt'
    save(result, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    original_select = steps._select_optimizer_window_action
    def full(*args, **kwargs):
        action = original_select(*args, **kwargs)
        action['granularities'] = ['g1000']
        kwargs['run_state']['global_sampling_state']['held_granularity'] = 'g1000'
        return action
    monkeypatch.setattr(steps, '_select_optimizer_window_action', full)
    if failure == 'correction':
        target, method = steps, '_apply_concat_lmc_corrections'
    elif failure == 'scheduler':
        target, method = clock, 'step'
    elif failure == 'accounting':
        target, method = optimizer, 'record_successful_update'
    else:
        target, method = optimizer.optimizer_for(failure), 'step'
    original = getattr(target, method)
    def fail(*args, **kwargs):
        if failure == 'correction':
            original(args[0][:1])  # Failure after only the first block is corrected.
        else:
            original(*args, **kwargs)
        raise RuntimeError('injected corrected mutation')
    monkeypatch.setattr(target, method, fail)
    with pytest.raises(RuntimeError, match='injected corrected'):
        train(result)
    assert state['optimizer_poisoned'] and state['update_in_flight']
    for reason in ('periodic', 'failure', 'signal', 'finalization'):
        with pytest.raises(ConfigError, match='unsafe|poison'):
            cp.maybe_write_latest_checkpoint(config, model, optimizer, clock, None, state, reason=reason, step=2)
    with pytest.raises(ConfigError, match='unsafe|poison'):
        save(result, path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize('scope', SCOPES)
def test_correction_contract_mismatch_rejected_before_mutation(tmp_path, scope):
    source = bundle(tmp_path, scope, 'gmc')
    train(source, stop=2)
    path = tmp_path / 'latest.pt'
    save(source, path)
    target = bundle(tmp_path, scope, 'lmc')
    before = [copy.deepcopy(x.state_dict()) for x in target[1:4]]
    with pytest.raises(ConfigError):
        load(target, path)
    for value, expected in zip(target[1:4], before):
        assert_state_equal(value.state_dict(), expected)

# Reuse only external audit fixtures; real resolver/model and strict contracts run.
from test_optimizer_ownership_campaign import audited_inputs


def correction_recipe():
    from pathlib import Path
    import yaml
    return yaml.safe_load(Path('configs/controlled_exps/tinystories_instruct_optimizer_ownership_corrections.yaml').read_text())


def test_six_arm_materialization_and_membership(tmp_path, audited_inputs):
    from src.evaluation import optimizer_ownership as campaign
    runs = campaign.expand_campaign(correction_recipe(), prepared_corpus_dir=tmp_path/'corpus',
        tokenizer_dir=tmp_path/'tokenizer', run_output_root=tmp_path/'runs')
    assert [r['arm_id'] for r in runs] == list(campaign.CORRECTION_ARM_IDS)
    checks = campaign.inspect_campaign_models(runs)
    assert len(checks) == 6
    for run in runs:
        config = run['resolved_config']
        campaign.validate_materialized_config(config)
        assert run['assigned_updates'] == 348528
        assert run['assigned_tokens'] == 2855141376
        assert config['model']['membership_correction']
        assert config['optimizer_ownership_contract']['correction'] == campaign.membership_correction_contract(run['correction_mode'])
        broken = copy.deepcopy(config)
        broken['optimizer_ownership_contract']['correction']['factors'][-1] = 1.
        broken['optimizer_ownership_contract_hash'] = stable_hash(broken['optimizer_ownership_contract'])
        with pytest.raises(ConfigError, match='correction contract'):
            campaign.validate_materialized_config(broken)


@pytest.mark.parametrize('damage', ('schema', 'missing_arm', 'mode', 'lr', 'clip', 'budget'))
def test_corrected_campaign_rejects_changed_controls(tmp_path, audited_inputs, damage):
    from src.evaluation import optimizer_ownership as campaign
    recipe = correction_recipe()
    if damage == 'schema': recipe['schema_version'] = 1
    elif damage == 'missing_arm': recipe['arms'].pop('C3-LMC')
    elif damage == 'mode': recipe['arms']['C3-LMC']['model']['correction_mode'] = 'gmc'
    elif damage == 'lr': recipe['common']['training']['learning_rate'] = .001
    elif damage == 'clip': recipe['arms']['C3-LMC']['training']['gradient_clipping']['owner_max_norms']['O-D'] = 2.
    elif damage == 'budget': recipe['arms']['C1-LMC']['training']['token_budget'] = 8192
    with pytest.raises(ConfigError):
        campaign.expand_campaign(recipe, prepared_corpus_dir=tmp_path/'corpus',
            tokenizer_dir=tmp_path/'tokenizer', run_output_root=tmp_path/'runs')


def test_corrected_freeze_comparison_and_progress(tmp_path, audited_inputs, monkeypatch):
    from pathlib import Path
    import csv
    from src.evaluation import optimizer_ownership as campaign
    from test_optimizer_ownership_reporting import terminal_campaign
    original_dir, corrected_dir = tmp_path/'original', tmp_path/'corrected'
    original_dir.mkdir(); corrected_dir.mkdir()
    original_manifest, original_runs = terminal_campaign.__wrapped__(original_dir, audited_inputs, monkeypatch)
    corrected_manifest, corrected_runs = terminal_campaign.__wrapped__(
        corrected_dir, (correction_recipe(), *audited_inputs[1:]), monkeypatch)
    campaign.freeze_campaign(campaign_manifest=original_manifest, run_root=original_runs, output_dir=original_dir/'frozen')
    campaign.freeze_campaign(campaign_manifest=corrected_manifest, run_root=corrected_runs, output_dir=corrected_dir/'frozen')
    report = campaign.report_correction_comparison(manifest=corrected_dir/'frozen/frozen_manifest.json',
        reference_manifest=original_dir/'frozen/frozen_manifest.json', output_dir=tmp_path/'report')
    assert len(report['endpoints']) == 40
    assert report['paired_traces_verified'] and not report['holdout_evaluated']
    assert len(report['comparisons']) == 24
    assert len(report['figures']) == 6
    assert all(Path(p).stat().st_size > 100 for p in report['figures'])
    with (tmp_path/'report/optimizer_ownership_endpoints.csv').open() as handle:
        exported = list(csv.DictReader(handle))
    assert len(exported) == 40
    for row, actual in zip(report['endpoints'], exported):
        assert actual == {k: campaign.endpoint_csv_value(v) for k, v in row.items()}
    figure = campaign.endpoint_figure(report['endpoints'], metric='loss', partial=False)
    ax = figure.axes[0]
    assert len(ax.lines) == 13
    assert ax.get_legend_handles_labels()[1].count('Standalone') == 1
    assert all(line.get_marker() == '^' and line.get_color() == '#8B4513' for line in ax.lines[-4:])
    assert ax.get_xlabel() == 'Active non-embedding parameters' and ax.get_ylabel() == 'Loss'
    assert len(report['progress_coverage']) == 9
    assert all(v['last_update'] == 348528 for arm in report['progress_coverage'].values() for v in arm.values())
    # Frozen source changes and incomplete campaigns cannot be reported.
    (corrected_runs/'C3-LMC/metrics.csv').write_text('damaged')
    with pytest.raises(ConfigError):
        campaign.report_correction_comparison(manifest=corrected_dir/'frozen/frozen_manifest.json',
            reference_manifest=original_dir/'frozen/frozen_manifest.json', output_dir=tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
