"""Phase-2 contract/formula verification; runtime integration comes later."""
import copy
import math

import pytest
import yaml

from src.training.schedules import (complexity_log_exponents,
    warmup_polynomial_learning_rate, polynomial_schedule_evidence)
from src.utils.config import (ConfigError, parse_linear_calr_schedule_contract,
    linear_calr_schedule_contract_dict, validate_linear_calr_schedule_eligibility)
from src.utils.reproducibility import (build_linear_calr_schedule_signature,
    build_full_run_signature, seed_for)
from test_optimizer_ownership_resume import fixture


def contract(policy='complexity_log', scope='shared'):
    labels = ['g250', 'g500', 'g750', 'g1000']
    counts = dict(zip(labels, [377408, 426560, 475712, 524864]))
    return dict(version=1, family='warmup_polynomial',
        position_convention='pre_update_zero_based', peak=.008,
        warmup_steps=64, horizon=348528, exponent_policy=policy,
        complexity_definition='active_trainable_scalars_including_embeddings_head_tied_once_v1',
        ordered_granularities=labels, ffn_sizes=dict(zip(labels, [64,128,192,256])),
        complexity_counts=counts,
        reporting_counts=dict(zip(labels, [115264,164416,213568,262720])),
        exponents=complexity_log_exponents(counts) if policy == 'complexity_log' else dict.fromkeys(labels, 1.),
        gamma_min=.5, gamma_max=2., nominal_exponent=1,
        effective_rate_policy='temporary_all_groups_restore_nominal_v1',
        optimizer_state_scope=scope, campaign_id='campaign', arm_id='arm',
        run_id='campaign-arm-s42', seed=42, seed_stream_version=1)


def test_freeze_roundtrip_and_hash():
    raw = contract()
    frozen = parse_linear_calr_schedule_contract(raw)
    with pytest.raises(TypeError):
        frozen['exponents']['g250'] = 3
    raw['exponents']['g250'] = 4
    assert frozen['exponents']['g250'] == 2
    detached = linear_calr_schedule_contract_dict(frozen)
    assert build_linear_calr_schedule_signature(frozen) == build_linear_calr_schedule_signature(yaml.safe_load(yaml.safe_dump(detached)))
    changed = copy.deepcopy(detached); changed['run_id'] = 'other'
    assert build_linear_calr_schedule_signature(changed)[0] != build_linear_calr_schedule_signature(frozen)[0]


@pytest.mark.parametrize('key,value', [('version',True), ('horizon',64),
    ('warmup_steps',0), ('peak',float('nan')), ('nominal_exponent',2),
    ('optimizer_state_scope','per_ffn_block'), ('seed',True),
    ('effective_rate_policy','multiply'), ('exponent_policy','unknown'),
    ('gamma_min',0), ('run_id',''), ('ordered_granularities',['g250','g250'])])
def test_invalid_contract(key,value):
    raw=contract(); raw[key]=value
    with pytest.raises(ConfigError): parse_linear_calr_schedule_contract(raw)


def test_invalid_counts_and_policy():
    for field in ['ffn_sizes','complexity_counts','reporting_counts','exponents']:
        raw=contract(); raw[field]['g500']=0
        with pytest.raises(ConfigError): parse_linear_calr_schedule_contract(raw)
    raw=contract('uniform'); raw['exponents']['g250']=2
    with pytest.raises(ConfigError): parse_linear_calr_schedule_contract(raw)
    raw=contract(); raw['extra']=1
    with pytest.raises(ConfigError): parse_linear_calr_schedule_contract(raw)


@pytest.mark.parametrize('policy', ['uniform','complexity_log'])
def test_analytic_anchors(policy):
    raw=contract(policy)
    for g in raw['ordered_granularities']:
        for p,expected in [(0,0),(63,.007875),(64,.008),(348528,0)]:
            row=polynomial_schedule_evidence(raw,g,p)
            assert row['evidence_kind']=='analytic'
            assert row['analytic_effective_learning_rate']==pytest.approx(expected, abs=1e-18)
        mid=64+(348528-64)//2
        assert polynomial_schedule_evidence(raw,g,mid)['analytic_effective_learning_rate']==pytest.approx(.008*.5**raw['exponents'][g])
        assert polynomial_schedule_evidence(raw,g,348527)['analytic_effective_learning_rate']>0
    assert raw['exponents']['g250'] >= raw['exponents']['g500'] >= raw['exponents']['g750'] >= raw['exponents']['g1000']
    with pytest.raises(ValueError): polynomial_schedule_evidence(raw,'unknown',0)


@pytest.mark.parametrize('p', [-1,348529,1.5,True])
def test_invalid_position(p):
    with pytest.raises(ValueError):
        warmup_polynomial_learning_rate(p,peak=.008,warmup_steps=64,horizon=348528)


def test_full_horizon_nonnegative():
    raw=contract()
    for gamma in raw['exponents'].values():
        for p in range(raw['horizon']+1):
            rate=warmup_polynomial_learning_rate(p,peak=.008,warmup_steps=64,horizon=348528,exponent=gamma)
            assert math.isfinite(rate) and 0 <= rate <= .008


@pytest.mark.parametrize('scope', ['shared','per_granularity'])
def test_eligibility_and_new_only_signatures(tmp_path, scope):
    config,*_=fixture(tmp_path,'S1' if scope=='shared' else 'S2')
    before=build_full_run_signature(config)
    raw=contract(scope=scope)
    model,run,training=config['model'],config['run'],config['training']
    model.update(variant='slicing',granularities=raw['ordered_granularities'],
        granularity_prefixes=dict(zip(raw['ordered_granularities'],[.25,.5,.75,1.])),
        intermediate_size=256,granularity_sampling_mode='global',correction_mode='none')
    run.update({key:raw[key] for key in ['campaign_id','arm_id','run_id','seed']})
    run.update(model_family='nested',sampling_mode='nested-random')
    training.update(linear_calr_schedule_contract=raw,resolved_learning_rate=.008,
        resolved_warmup_steps=64,max_steps=348528,optimizer_scheduler_clock='global_step',
        optimizer_state_scope=scope,optimizer_name='adamw',effective_world_size=1)
    validate_linear_calr_schedule_eligibility(config)
    original_streams=[seed_for(config,s) for s in ['model_initialization','granularity_selection','training_sampler']]
    signature=build_full_run_signature(config)
    assert 'linear_calr_schedule_contract_hash' in signature[1]
    run['run_id']=raw['run_id']='renamed'
    assert build_full_run_signature(config)[0]!=signature[0]
    assert original_streams==[seed_for(config,s) for s in ['model_initialization','granularity_selection','training_sampler']]
    model['variant']='concat'
    with pytest.raises(ConfigError): validate_linear_calr_schedule_eligibility(config)
    assert 'linear_calr_schedule_contract' not in before[1]
