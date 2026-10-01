"""Separate C4 corrections: real concat gradients, AdamW decay and restore."""
import copy
from pathlib import Path

import pytest
import torch

from src.training import steps, checkpointing as cp
from src.training.modeling import build_model
from src.utils.config import ConfigError, resolve_run_config

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def config_for(grid, kind):
    return resolve_run_config(ROOT / f'configs/controlled_exps/tinystories_instruct_c4_separate_{grid}-{kind}.yaml', create_output_dirs=False)


def backward(model, width, device):
    model.zero_grad(set_to_none=True)
    model.configure_subnetwork(width)
    tokens = torch.arange(1, 17, device=device).reshape(1, 16)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        loss = model(input_ids=tokens, labels=tokens).loss
    loss.backward()


def clip(model, opt, config, width):
    selected = opt.active_owner_ids(width)
    for owner in opt.owners:
        if owner.owner_id not in selected:
            for p in owner.parameters:
                p.grad = None
    return steps.clip_optimizer_gradients(model, config['training'], width, owners=opt.owners, selected_owner_ids=selected)


def commit(opt, clock, width):
    selected = opt.active_owner_ids(width)
    evidence = [opt.step_owner(owner) for owner in selected]
    clock.step()
    clock.synchronize(opt)
    opt.record_successful_update(width, returned_owners=selected)
    opt.validate_accounting(step=clock.position)
    return evidence


@pytest.mark.parametrize('device_name', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA gate only'))])
@pytest.mark.parametrize('grid', ['linear', 'geometric'])
@pytest.mark.parametrize('kind', ['GMC', 'LMC-only'])
def test_all_width_gradient_isolation_joint_clip_and_manual_adamw(grid, kind, device_name):
    device = torch.device(device_name)
    config = config_for(grid, kind)
    if device_name == 'cuda':
        from src.utils.reproducibility import configure_strict_determinism
        if not torch.cuda.is_initialized():
            configure_strict_determinism(config)
    raw_config = resolve_run_config(ROOT / f'configs/controlled_exps/tinystories_instruct_c4_selected_block_{grid}.yaml', create_output_dirs=False)
    torch.manual_seed(42)
    model = build_model(config).to(device)
    raw = build_model(raw_config).to(device)
    raw.load_state_dict(model.state_dict())
    opt, clock = steps.build_optimizer_and_scheduler(model, config['training'])
    # Nonzero nominal LR, including genuine nonzero moment history for each owner.
    clock.step(); clock.synchronize(opt)
    for width in config['model']['granularities'] * 2:
        raw.load_state_dict(model.state_dict())
        backward(raw, width, device)
        backward(model, width, device)
        selected = opt.active_owner_ids(width)
        owner = next(o for o in opt.owners if o.owner_id == selected[0])
        factor = config['training']['c4_correction']['factors'][owner.owner_id]
        raw_params = dict(raw.named_parameters())
        name_by_id = {id(p): n for n, p in model.named_parameters()}
        for o in (owner, opt.owners[-1]):
            for p in o.parameters:
                reference = raw_params[name_by_id[id(p)]].grad
                assert reference is not None
                expected = reference * (factor if kind == 'GMC' and o == owner else 1.)
                torch.testing.assert_close(p.grad, expected, rtol=3e-5, atol=1e-7)
        earlier = [o for o in opt.owners if o.owner_id not in selected]
        if width != config['model']['granularities'][0]:
            assert any(p.grad is not None for o in earlier for p in o.parameters)
        frozen = {o.owner_id: ([p.detach().clone() for p in o.parameters], copy.deepcopy(opt.optimizer_for(o.owner_id).state_dict())) for o in earlier}
        pre = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.float()) for o in (owner, opt.owners[-1]) for p in o.parameters]))
        obs = clip(model, opt, config, width)
        coef = min(1., 1. / (float(pre) + 1e-6))
        assert obs['combined_post_norm'] <= 1.00001
        assert obs['groups'][owner.owner_id]['coefficient'] == pytest.approx(coef, rel=1e-5)
        assert obs['groups'][owner.owner_id]['coefficient'] == obs['groups']['O-common']['coefficient']
        live = opt.optimizer_for(owner.owner_id)
        manual_params = [torch.nn.Parameter(p.detach().clone()) for p in owner.parameters]
        manual = torch.optim.AdamW(manual_params, **{k: v for k, v in live.defaults.items() if k not in ('params', 'decoupled_weight_decay')})
        manual.load_state_dict(copy.deepcopy(live.state_dict()))
        nominal = opt.current_learning_rates[0]
        multiplier = factor if kind == 'LMC-only' else 1.
        for group in manual.param_groups:
            group['lr'] = nominal * multiplier
        for p, q in zip(owner.parameters, manual_params):
            q.grad = p.grad.detach().clone()
        manual.step()
        # Clock has been advanced once for nonzero warmup LR; logical counts
        # still start at zero. Verify and commit the two owners explicitly.
        evidence = [opt.step_owner(oid) for oid in selected]
        previous_position = clock.position
        clock.step(); clock.synchronize(opt)
        opt.record_successful_update(width, returned_owners=selected)
        opt.validate_accounting(step=opt.total_successful_updates)
        assert clock.position == previous_position + 1
        assert evidence[0]['effective_learning_rates'] == [nominal * multiplier]
        assert evidence[1]['effective_learning_rates'] == [nominal]
        for p, q in zip(owner.parameters, manual_params):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
            for component in ('exp_avg', 'exp_avg_sq', 'step'):
                torch.testing.assert_close(live.state[p][component], manual.state[q][component], rtol=0, atol=0)
        for o in earlier:
            before, state = frozen[o.owner_id]
            assert all(torch.equal(p, b) for p, b in zip(o.parameters, before))
            assert_same(opt.optimizer_for(o.owner_id).state_dict()['state'], state['state'])
        opt.validate_synchronized_learning_rates()


def assert_same(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('grid', ['linear', 'geometric'])
@pytest.mark.parametrize('kind', ['GMC', 'LMC-only'])
@pytest.mark.parametrize('position', [0, 63, 64, 65, 87132, 174264, 348527])
def test_schedule_boundary_resume_and_incompatible_metadata(grid, kind, position):
    config = config_for(grid, kind)
    model = build_model(config)
    opt, clock = steps.build_optimizer_and_scheduler(model, config['training'])
    # State-seeded schedule boundary probe, not a claim of full-budget training.
    carrier = clock._scheduler
    rates = [base * fn(position) for base, fn in zip(carrier.base_lrs, carrier.lr_lambdas)]
    state = clock.state_dict()
    state['position'] = position
    state['scheduler_state_dict'].update(last_epoch=position, _step_count=position + 1, _last_lr=rates)
    state['last_committed_learning_rates'] = [base * fn(position - 1) for base, fn in zip(carrier.base_lrs, carrier.lr_lambdas)] if position else None
    # The clock carrier's param group also records the nominal scheduled LR.
    for group in state.get('carrier_optimizer_state_dict', {}).get('param_groups', []):
        group['lr'] = rates[0]
    clock.load_state_dict(state); clock.synchronize(opt)
    widths = config['model']['granularities']
    for width in [widths[-1], widths[1], widths[-1]]:
        backward(model, width, torch.device('cpu'))
        clip(model, opt, config, width)
        [opt.step_owner(o) for o in opt.active_owner_ids(width)]
        clock.step(); clock.synchronize(opt)
        opt.record_successful_update(width, returned_owners=opt.active_owner_ids(width))
    clone_model = build_model(config); clone_model.load_state_dict(model.state_dict())
    clone, clone_clock = steps.build_optimizer_and_scheduler(clone_model, config['training'])
    clone.load_state_dict(opt.state_dict()); clone_clock.load_state_dict(clock.state_dict()); clone_clock.synchronize(clone)
    for width in [widths[-1], widths[-1], widths[2]]:
        for m, o, c in [(model, opt, clock), (clone_model, clone, clone_clock)]:
            backward(m, width, torch.device('cpu')); clip(m, o, config, width)
            [o.step_owner(oid) for oid in o.active_owner_ids(width)]
            c.step(); c.synchronize(o); o.record_successful_update(width, returned_owners=o.active_owner_ids(width))
        assert_same(model.state_dict(), clone_model.state_dict())
        assert_same(opt.state_dict(), clone.state_dict())
        assert_same(clock.state_dict(), clone_clock.state_dict())
    damaged = copy.deepcopy(opt.state_dict()); damaged['c4_correction']['gradient_correction'] = not (kind == 'GMC')
    before = copy.deepcopy(clone.state_dict())
    with pytest.raises(ConfigError):
        clone.load_state_dict(damaged)
    assert_same(before, clone.state_dict())
    other = config_for(grid, 'LMC-only' if kind == 'GMC' else 'GMC')
    assert cp._optimizer_checkpoint_contract(config) != cp._optimizer_checkpoint_contract(other)


def test_owner_failure_restores_nominal_lr_without_committing(monkeypatch):
    config = config_for('linear', 'LMC-only')
    model = build_model(config); opt, clock = steps.build_optimizer_and_scheduler(model, config['training'])
    clock.step(); clock.synchronize(opt)
    nominal = opt.current_learning_rates
    def fail():
        assert opt.optimizer_for('O-D').param_groups[0]['lr'] == nominal[0] * 4
        raise RuntimeError('after mutation simulation')
    monkeypatch.setattr(opt.optimizer_for('O-D'), 'step', fail)
    with pytest.raises(RuntimeError):
        opt.step_owner('O-D')
    assert opt.validate_synchronized_learning_rates() == nominal
    assert opt.total_successful_updates == 0
    assert clock.position == 1


def runtime_bundle(tmp_path, kind, *, horizon=8):
    from test_optimizer_ownership import runtime_fixture
    config, model, _, _, batches, _ = runtime_fixture(tmp_path, max_steps=horizon, correction_mode='gmc' if kind == 'GMC' else 'none')
    t = config['training']
    t['block_update_policy'] = 'selected_block'
    t['gradient_clipping'] = {'mode': 'global', 'max_norm': 1., 'norm_type': 2, 'stabilization_epsilon': 1e-6}
    t['c4_correction'] = copy.deepcopy(config_for('linear', kind)['training']['c4_correction'])
    t['optimizer_state_contract'].update(block_update_policy='selected_block', c4_correction=t['c4_correction'])
    opt, clock = steps.build_optimizer_and_scheduler(model, t)
    return config, model, opt, clock, batches, cp.build_initial_continuation_state(config)


def train_bundle(bundle, stop=None):
    config, model, opt, clock, batches, state = bundle
    def boundary(**kw):
        if kw['step'] == stop:
            raise StopIteration('committed boundary')
    try:
        steps.train_for_steps(config, model, batches, [], opt, clock, torch.device('cpu'), run_state=state, successful_step_callback=boundary)
    except StopIteration:
        pass


def save_bundle(bundle, path):
    config, model, opt, clock, _, state = bundle
    cp.save_model_checkpoint(config, model, opt, clock, path, {'checkpoint_status': 'latest', 'checkpoint_metric': None, 'checkpoint_metric_value': None, 'checkpoint_selection_step': None}, state)


@pytest.mark.parametrize('kind', ['GMC', 'LMC-only'])
@pytest.mark.parametrize('boundary', [1, 2, 3, 5])
def test_full_checkpoint_action_rng_data_cursor_resume(tmp_path, kind, boundary):
    bundle = runtime_bundle(tmp_path, kind)
    train_bundle(bundle, boundary)
    path = tmp_path / 'latest.pt'; save_bundle(bundle, path)
    train_bundle(bundle)
    clone = runtime_bundle(tmp_path, kind)
    restored = cp.load_checkpoint_state(path, clone[1], clone[2], clone[3], config=clone[0], train_dataloader=clone[4])
    clone[5].clear(); clone[5].update(restored)
    train_bundle(clone)
    for a, b in zip(bundle[1:4], clone[1:4]):
        assert_same(a.state_dict(), b.state_dict())
    for key in ('step', 'tokens_seen', 'content_tokens_seen', 'epoch', 'batch_index', 'global_sampling_state'):
        assert_same(bundle[5][key], clone[5][key])
    incompatible = runtime_bundle(tmp_path, 'GMC' if kind == 'LMC-only' else 'LMC-only')
    snapshot = copy.deepcopy(incompatible[1].state_dict())
    with pytest.raises(ConfigError):
        cp.load_checkpoint_state(path, incompatible[1], incompatible[2], incompatible[3], config=incompatible[0])
    assert_same(snapshot, incompatible[1].state_dict())


@pytest.mark.parametrize('failure', ['selected', 'common', 'clock'])
def test_actual_trainer_failure_cannot_save_complete_update(tmp_path, monkeypatch, failure):
    bundle = runtime_bundle(tmp_path, 'LMC-only')
    train_bundle(bundle, 1)
    path = tmp_path / 'latest.pt'; save_bundle(bundle, path)
    checkpoint = path.read_bytes()
    opt, clock = bundle[2:4]
    def fail_after_original(original):
        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError('injected after mutation')
        return fail
    if failure == 'clock':
        monkeypatch.setattr(clock, 'step', fail_after_original(clock.step))
    elif failure == 'common':
        item = opt.optimizer_for('O-common')
        monkeypatch.setattr(item, 'step', fail_after_original(item.step))
    else:
        for owner in opt.owners[:-1]:
            item = opt.optimizer_for(owner.owner_id)
            monkeypatch.setattr(item, 'step', fail_after_original(item.step))
    with pytest.raises(RuntimeError, match='injected after mutation'):
        train_bundle(bundle)
    assert bundle[5].get('optimizer_poisoned') or bundle[5].get('update_in_flight')
    with pytest.raises((ConfigError, RuntimeError)):
        save_bundle(bundle, path)
    assert path.read_bytes() == checkpoint
    assert opt.total_successful_updates == 1
