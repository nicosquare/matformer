"""Full-horizon analytic verification; these are not applied-update records."""
import math
import pytest
from src.training.schedules import complexity_log_exponents, warmup_polynomial_learning_rate as rate

COUNTS = dict(zip(('g250','g500','g750','g1000'), (377408,426560,475712,524864)))

@pytest.mark.parametrize('policy', ['uniform', 'complexity_log'])
def test_all_widths_full_horizon(policy):
    exponents = complexity_log_exponents(COUNTS) if policy == 'complexity_log' else dict.fromkeys(COUNTS, 1.)
    assert complexity_log_exponents(COUNTS) == pytest.approx(dict(zip(COUNTS, (2.,1.4432006280490555,.9471931630317243,.5))))
    for width, gamma in exponents.items():
        def lr(p): return rate(p, peak=.008, warmup_steps=64, horizon=348528, exponent=gamma)
        for p, expected in [(0,0.),(63,.007875),(64,.008),(65,.008*(348463/348464)**gamma),(174296,.008*.5**gamma),(348527,.008*(1/348464)**gamma),(348528,0.)]:
            assert lr(p) == pytest.approx(expected, rel=1e-14, abs=1e-20)
        previous = .008
        for p in range(64,348529):
            value = lr(p)
            assert math.isfinite(value) and 0 <= value <= previous
            previous = value
        assert lr(348528) == 0.

@pytest.mark.parametrize('kwargs', [dict(position=-1),dict(position=348529),dict(position=True),dict(position=1.5),dict(warmup_steps=0),dict(horizon=64),dict(exponent=0),dict(exponent=float('nan')),dict(peak=-1)])
def test_invalid_rates(kwargs):
    args=dict(position=0,peak=.008,warmup_steps=64,horizon=348528,exponent=1.); args.update(kwargs)
    with pytest.raises(ValueError): rate(**args)

@pytest.mark.parametrize('counts', [{}, {'a':0,'b':1}, {'a':1,'b':1}, {'a':True,'b':2}])
def test_invalid_complexity(counts):
    with pytest.raises(ValueError): complexity_log_exponents(counts)

@pytest.mark.parametrize('arm', ['S1-linear-poly','S1-linear-CaLR','S2-linear-poly','S2-linear-CaLR'])
def test_actual_updates_match_all_group_manual_adamw(tmp_path,arm):
    import copy
    import torch
    from test_linear_calr_resume import fixture, train
    from test_optimizer_ownership import assert_state_equal
    from src.training import steps
    from src.training.schedules import polynomial_schedule_evidence
    from src.training.optimizer_state import PerGranularityOptimizerCollection
    bundle=fixture(tmp_path,arm)
    config,model,opt,clock,_,state=bundle
    owners=[e.optimizer for e in opt.entries] if hasattr(opt,'entries') else [opt]
    for owner in owners:
        first=owner.param_groups[0]
        parameters=first['params']; first['params']=parameters[:1]
        owner.add_param_group({**{k:v for k,v in first.items() if k!='params'},'params':parameters[1:]})
    clock.synchronize(opt)
    options=[[{k:copy.deepcopy(v) for k,v in group.items() if k not in ('lr','params')} for group in owner.param_groups] for owner in owners]
    for owner in owners:
        assert [id(p) for group in owner.param_groups for p in group['params']]==[id(p) for p in model.parameters()]
    original=steps._maybe_apply_concat_lmc_optimizer_step
    observed=[]
    changed_tails=[]
    def compare(config,model,owner):
        # Clone only in the test; a manual optimizer takes precisely the captured gradients.
        reference=copy.deepcopy(model)
        mapping=dict(reference.named_parameters())
        reference_groups=[]
        names={id(p):n for n,p in model.named_parameters()}
        for group in owner.param_groups:
            reference_groups.append({**{k:v for k,v in group.items() if k!='params'},
                                     'params':[mapping[names[id(p)]] for p in group['params']]})
        manual=torch.optim.AdamW(reference_groups)
        manual.load_state_dict(copy.deepcopy(owner.state_dict()))
        for p in model.parameters():mapping[names[id(p)]].grad=p.grad.detach().clone() if p.grad is not None else None
        width=state['optimizer_active_owner_granularity'] or model.current_layer_granularities[0]
        evidence=polynomial_schedule_evidence(config['training']['linear_calr_schedule_contract'],width,clock.position)
        assert all(g['lr']==evidence['analytic_effective_learning_rate'] for g in owner.param_groups)
        before=copy.deepcopy(model.state_dict()); other=[]
        if isinstance(opt,PerGranularityOptimizerCollection):
            other=[(e.optimizer,copy.deepcopy(e.optimizer.state_dict())) for e in opt.entries if e.optimizer is not owner]
        tails=[]
        for module in model.modules():
            from src.models.ffn import ModifiedLlamaMLP
            if isinstance(module,ModifiedLlamaMLP) and width!='g1000':
                prefix=next(m['prefix_width'] for m in module.ffn_prefix_metadata if m['name']==width)
                entry=next(e for e in module.physical_parameter_metadata() if e.component=='gate_weight')
                assert entry.parameter.grad.shape==entry.parameter.shape
                assert torch.count_nonzero(entry.parameter.grad[prefix:])==0
                tails.append((entry.parameter,prefix,entry.parameter[prefix:].detach().clone()))
        manual.step();original(config,model,owner)
        if clock.position>0:
            changed_tails.extend(not torch.equal(p[prefix:],saved) for p,prefix,saved in tails)
        assert_state_equal(model.state_dict(),reference.state_dict());assert_state_equal(owner.state_dict(),manual.state_dict())
        for obj,saved in other:assert_state_equal(obj.state_dict(),saved)
        if clock.position==0:
            assert_state_equal(model.state_dict(),before)
            assert all(s['step'].item()==1 for s in owner.state.values())
        observed.append(width)
    from unittest.mock import patch
    with patch.object(steps,'_maybe_apply_concat_lmc_optimizer_step',compare):train(bundle)
    assert set(observed)==set(COUNTS)
    assert changed_tails and all(changed_tails)
    assert clock.position==72 and clock.current_learning_rates==(0.,)
    assert state['last_applied_schedule_record']['pre_update_position']==71
    assert options==[[{k:v for k,v in group.items() if k not in ('lr','params')} for group in owner.param_groups] for owner in owners]
    with pytest.raises(steps.ConfigError,match='terminal'):
        steps.apply_selected_schedule_rates(config,opt,owners[0],clock,'g250',72,{},state)
