"""Fixed IM policy through real campaign, optimizer and continuation paths."""
import copy
import random
from pathlib import Path

import pytest
import torch
import yaml

from src.evaluation import optimizer_ownership as campaign
from src.training import checkpointing as cp, steps
from src.utils.config import ConfigError
from src.utils.reproducibility import capture_rng_state, seed_for, stable_hash
from test_optimizer_ownership import assert_state_equal
from test_optimizer_ownership_campaign import audited_inputs, expand
from test_optimizer_ownership_resume import packed_fixture, train, save, load

ARMS = ('S1', 'S2', 'C1', 'C2', 'C3')
PROBABILITIES = [.12, .16, .24, .48]
RECIPE = Path('configs/controlled_exps/tinystories_instruct_inverse_membership.yaml')


def bundle(tmp_path, arm='C3'):
    result = packed_fixture(tmp_path, arm)
    config = result[0]
    config['model']['granularity_sampling_mode'] = 'fixed_global'
    config['model']['global_sampling_distribution'] = dict(zip(campaign.WIDTH_LABELS, PROBABILITIES))
    config['optimizer_ownership_contract']['arm_id'] = arm + '-IM'
    config['optimizer_ownership_contract']['sampling'] = campaign.inverse_membership_sampling_contract(config)
    config['optimizer_ownership_contract_hash'] = stable_hash(config['optimizer_ownership_contract'])
    result[-1]['global_sampling_state'] = cp.build_initial_global_sampling_state(config)
    return result


def test_five_arm_preflight_counts_and_traces(tmp_path, audited_inputs):
    runs = expand(tmp_path, yaml.safe_load(RECIPE.read_text()))
    assert [r['arm_id'] for r in runs] == [a+'-IM' for a in ARMS]
    checks = campaign.inspect_campaign_models(runs)
    traces = []
    for run, check in zip(runs, checks):
        assert check['counts'] == {w['label']: w['non_embedding_parameters'] for w in campaign.WIDTHS}
        assert run['assigned_updates'] == 348528
        assert run['assigned_tokens'] == 2855141376
        assert run['resolved_config']['model']['correction_mode'] == 'none'
        campaign.validate_materialized_config(run['resolved_config'])
        config = copy.deepcopy(run['resolved_config'])
        config['training']['max_steps'] = 200
        traces.append(campaign.expected_action_trace(config))
    assert all(t == traces[0] for t in traces)
    generator = random.Random(seed_for(runs[0]['resolved_config'], 'granularity_selection'))
    from collections import Counter
    expected = Counter(generator.choices(campaign.WIDTH_LABELS, weights=PROBABILITIES, k=1)[0] for _ in range(200))
    assert traces[0]['counts'] == dict(expected)


@pytest.mark.parametrize('damage', ['probability', 'width', 'budget', 'correction', 'clip', 'hold', 'schema'])
def test_reject_wrong_recipe(tmp_path, audited_inputs, damage):
    recipe = yaml.safe_load(RECIPE.read_text())
    arm = recipe['arms']['C3-IM']
    if damage == 'probability': arm['model']['global_sampling_distribution']['g250'] = .13
    elif damage == 'width': recipe['common']['model']['granularities'].reverse()
    elif damage == 'budget': arm['training']['token_budget'] -= 8192
    elif damage == 'correction': arm['model']['correction_mode'] = 'gmc'
    elif damage == 'clip': arm['training']['gradient_clipping']['owner_max_norms']['O-A'] = 2.
    elif damage == 'hold': arm['model']['global_sampling_interval_steps'] = 2
    elif damage == 'schema': recipe['schema_version'] = 1
    with pytest.raises(ConfigError): expand(tmp_path, recipe)


@pytest.mark.parametrize('uniform,expected', [(0.,0),(.119999,0),(.12,1),(.279999,1),(.28,2),(.519999,2),(.52,3),(.99999,3)])
def test_production_categorical_boundaries(monkeypatch, uniform, expected):
    generator = random.Random(42)
    monkeypatch.setattr(generator, 'random', lambda: uniform)
    monkeypatch.setattr(steps, 'dedicated_random', lambda *a: generator)
    config = {'model': {'granularity_sampling_mode':'fixed_global', 'granularities':list(campaign.WIDTH_LABELS),
                        'global_sampling_distribution':dict(zip(campaign.WIDTH_LABELS, PROBABILITIES))}}
    assert steps.select_random_granularity_index(config, 4, torch.device('cpu')) == expected


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('boundary', [1,2,3])
def test_exact_fixed_resume_across_packed_epoch(tmp_path, arm, boundary):
    source = bundle(tmp_path, arm)
    train(source, stop=boundary)
    path = tmp_path/'latest.pt'
    save(source, path)
    trace = []
    train(source, trace=trace)
    restored = bundle(tmp_path, arm)
    load(restored, path)
    resumed = []
    train(restored, trace=resumed)
    assert trace == resumed
    for a,b in zip(source[1:4], restored[1:4]): assert_state_equal(a.state_dict(), b.state_dict())
    assert_state_equal(source[-1]['global_sampling_state'], restored[-1]['global_sampling_state'])
    counts = source[-1]['global_sampling_state']['exposure_counts']
    assert sum(counts.values()) == source[-1]['step']


@pytest.mark.parametrize('damage', ['policy','probabilities','state_policy','counts','rng'])
def test_fixed_checkpoint_rejects_before_mutation(tmp_path, damage):
    source = bundle(tmp_path)
    train(source, stop=3)
    path = tmp_path/'latest.pt'
    save(source, path)
    payload = torch.load(path, weights_only=False)
    if damage == 'policy': payload['optimizer_ownership_contract']['sampling']['policy'] = 'uniform'
    elif damage == 'probabilities': payload['optimizer_ownership_contract']['sampling']['probabilities'] = [.25]*4
    elif damage == 'state_policy': payload['global_sampling_state']['sampling_policy'] = 'uniform'
    elif damage == 'counts': payload['global_sampling_state']['exposure_counts']['g250'] += 1
    elif damage == 'rng':
        for rng in [payload['reproducibility']['rng_state'], *payload['reproducibility']['rng_states_by_rank']]:
            rng['dedicated']['granularity_selection'] = random.Random(999).getstate()
    torch.save(payload, path)
    restored = bundle(tmp_path)
    before = [copy.deepcopy(v.state_dict()) for v in restored[1:4]]
    rng = capture_rng_state()
    with pytest.raises(ConfigError): load(restored, path)
    for value, expected in zip(restored[1:4], before): assert_state_equal(value.state_dict(), expected)
    assert_state_equal(capture_rng_state(), rng)


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('width', campaign.WIDTH_LABELS)
def test_real_fixed_paths_match_independent_adamw(tmp_path, monkeypatch, arm, width):
    actual = bundle(tmp_path, arm)
    config, model, optimizer, clock, batches, state = actual
    reference = copy.deepcopy(model)
    # Independent optimizers over the same reference weights, with grouping from
    # physical names rather than the runtime owner collection.
    ref_opts = {}
    if arm == 'C3':
        for group in ('O-A','O-B','O-C','O-D','O-common'):
            params = [p for n,p in reference.named_parameters() if
                      ('O-'+'ABCD'[int(n.rsplit('.',1)[1])] if '_blocks.' in n else 'O-common') == group]
            ref_opts[group] = torch.optim.AdamW(params, lr=.001, betas=(.9,.95), eps=1e-8, weight_decay=.1)
    else:
        for group in campaign.WIDTH_LABELS if arm in ('S2','C2') else ('shared',):
            ref_opts[group] = torch.optim.AdamW(reference.parameters(), lr=.001, betas=(.9,.95), eps=1e-8, weight_decay=.1)
    choices = iter(('g1000',width))
    original_select = steps._select_optimizer_window_action
    def select(*args, **kw):
        action = original_select(*args, **kw)
        selected = next(choices)
        action['granularities'] = [selected]
        action['sampled_probability'] = config['model']['global_sampling_distribution'][selected]
        state['global_sampling_state']['held_granularity'] = selected
        return action
    monkeypatch.setattr(steps, '_select_optimizer_window_action', select)
    tokens = []
    model.register_forward_pre_hook(lambda m,a,kw: tokens.append(kw['input_ids']), with_kwargs=True)
    original_clip = steps.clip_optimizer_gradients
    previous = {}
    def clip(model, training, selected, **kw):
        previous.clear(); previous.update({n:p.detach().clone() for n,p in model.named_parameters()})
        reference.zero_grad(set_to_none=True)
        reference.configure_subnetwork(selected)
        reference(input_ids=tokens[-1], labels=tokens[-1]).loss.backward()
        for p,q in zip(model.parameters(),reference.parameters()):
            assert (p.grad is None) == (q.grad is None)
            if p.grad is not None: torch.testing.assert_close(p.grad,q.grad)
        keys = (['O-'+q for q in 'ABCD'[:campaign.WIDTH_LABELS.index(selected)+1]]+['O-common'] if arm=='C3'
                else [selected if arm in ('S2','C2') else 'shared'])
        rate = optimizer.current_learning_rates[0] if hasattr(optimizer,'current_learning_rates') else optimizer.param_groups[0]['lr']
        for key in keys:
            ref_opt = ref_opts[key]
            for group in ref_opt.param_groups: group['lr'] = rate
            torch.nn.utils.clip_grad_norm_([p for g in ref_opt.param_groups for p in g['params']],1.)
            ref_opt.step()
        return original_clip(model,training,selected,**kw)
    monkeypatch.setattr(steps,'clip_optimizer_gradients',clip)
    def committed(step,**kw):
        assert (clock.position if hasattr(clock,'position') else clock.last_epoch) == step
        for (name,p),q in zip(model.named_parameters(),reference.parameters()):
            torch.testing.assert_close(p,q,rtol=2e-6,atol=1e-7)
            if p.grad is None: assert torch.equal(p,previous[name])
        if step == 2: raise StopIteration
    with pytest.raises(StopIteration):
        steps.train_for_steps(config,model,batches,[],optimizer,clock,torch.device('cpu'),run_state=state,successful_step_callback=committed)
    assert sum(state['optimizer_width_selection_counts'].values()) == 2
    if arm == 'C3': assert state['optimizer_update_counts']['O-common'] == 2


@pytest.mark.parametrize('failure', ['O-A','O-C','O-common','scheduler','accounting'])
def test_fixed_c3_partial_update_never_replaces_checkpoint(tmp_path, monkeypatch, failure):
    import hashlib
    current = bundle(tmp_path)
    config, model, optimizer, clock, batches, state = current
    train(current,stop=1)
    path=tmp_path/'durable.pt'; save(current,path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    original_select=steps._select_optimizer_window_action
    def select(*args,**kw):
        action=original_select(*args,**kw)
        action['granularities']=['g1000']; action['sampled_probability']=.48
        state['global_sampling_state']['held_granularity']='g1000'
        return action
    monkeypatch.setattr(steps,'_select_optimizer_window_action',select)
    target=clock if failure=='scheduler' else optimizer if failure=='accounting' else optimizer.optimizer_for(failure)
    method='record_successful_update' if failure=='accounting' else 'step'
    original=getattr(target,method)
    def fail(*args,**kw):
        original(*args,**kw)
        raise RuntimeError('injected failure')
    monkeypatch.setattr(target,method,fail)
    with pytest.raises(RuntimeError,match='injected'): train(current)
    assert state['optimizer_poisoned'] and state['update_in_flight']
    with pytest.raises(ConfigError): save(current,path)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
