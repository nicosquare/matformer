"""Compatibility boundary for the new campaign; no schema-6 behavior yet."""
import copy
import math

import pytest
import yaml

from src.evaluation import optimizer_ownership as campaign
from src.training import steps
from src.utils.reproducibility import build_optimizer_ownership_signature, stable_hash
from test_optimizer_ownership_campaign import audited_inputs, expand
from test_s1_warmup_campaign import (
    historical_campaign, WARMUP_RECIPE,
    test_historical_resolved_signatures_and_serialized_fields as check_legacy,
)
from test_optimizer_ownership_resume import (
    fixture, train, save, load, assert_state_equal,
)
from test_c4_separate_corrections import (
    test_schedule_boundary_resume_and_incompatible_metadata as check_c4_restore,
)


def test_schema1_through4_frozen_resolution(historical_campaign, tmp_path):
    # Reuse the existing immutable schema 1–3 signatures and schema-4 resolved
    # snapshot, including serialized field sets and JSON/YAML round trips.
    check_legacy(historical_campaign, tmp_path)


SCHEMA5 = {
    'S1-linear-w256': (
        '4aa6eb318119374a569d5716b06104f3de62ca863e798019ae3f8a3c4d6c2766',
        '8bf1b1ffb62094fb66649422f6c55f4d566fe3be6222d021b28ac63b959d293b'),
    'S1-geometric-w256': (
        '38523a3105de4b721560f1416e481d37e3b1ab495368e89ebd471e95e29ccba2',
        '26f594e9f1bd864f41c420b92771bef6e1b4c93381cca553d3ac790ce29897fb'),
}


def test_schema5_frozen_resolution(tmp_path, monkeypatch, audited_inputs):
    monkeypatch.setattr(campaign, '_provenance', lambda: {
        'code_revision': 'linear-calr-phase1-schema5-fixture',
        'working_tree_dirty': False,
    })
    runs = expand(tmp_path, yaml.safe_load(WARMUP_RECIPE.read_text()))
    assert {r['arm_id'] for r in runs} == set(SCHEMA5)
    for run in runs:
        config = copy.deepcopy(run['resolved_config'])
        for section, key in [('dataset', 'prepared_corpus_dir'),
                             ('model', 'tokenizer_dir'), ('run', 'output_dir'),
                             ('run', 'output_root')]:
            config[section][key] = config[section][key].replace(str(tmp_path), '<fixture>')
        expected_contract, expected_config = SCHEMA5[run['arm_id']]
        assert run['contract_hash'] == expected_contract
        assert stable_hash(config) == expected_config
        contract = run['optimizer_ownership_contract']
        for restored in [yaml.safe_load(yaml.safe_dump(contract)), copy.deepcopy(contract)]:
            assert build_optimizer_ownership_signature(restored) == (expected_contract, contract)
        campaign.validate_materialized_config(run['resolved_config'])
    assert not (tmp_path / 'runs').exists()


@pytest.mark.parametrize('arm', ['S1', 'S2', 'C3'])
@pytest.mark.parametrize('position', [0, 63, 64, 65, 87132, 174264, 348527, 348528])
def test_legacy_cosine_schedule_and_clock_restore(tmp_path, arm, position):
    config, model, _, _, _, _ = fixture(tmp_path, arm)
    training = copy.deepcopy(config['training'])
    training.update(scheduler_name='cosine', scheduler_kwargs={},
                    resolved_warmup_steps=64, max_steps=348528)
    optimizer, clock = steps.build_optimizer_and_scheduler(model, training)
    carrier = getattr(clock, '_scheduler', clock)
    expected = carrier.base_lrs[0] * (
        position / 64 if position < 64 else
        0.5 * (1 + math.cos(math.pi * (position - 64) / (348528 - 64))))
    assert carrier.base_lrs[0] * carrier.lr_lambdas[0](position) == pytest.approx(expected, abs=1e-18)
    # A state-seeded clock probe checks global, rather than owner-local time.
    state = clock.state_dict()
    scheduler_state = state.get('scheduler_state_dict', state)
    scheduler_state.update(last_epoch=position, _step_count=position + 1, _last_lr=[expected])
    if hasattr(clock, 'synchronize'):
        state['position'] = position
        state['last_committed_learning_rates'] = (
            [carrier.base_lrs[0] * carrier.lr_lambdas[0](position - 1)] if position else None)
        for group in state.get('carrier_optimizer_state_dict', {}).get('param_groups', []):
            group['lr'] = expected
    clock.load_state_dict(state)
    if hasattr(clock, 'synchronize'):
        clock.synchronize(optimizer)
    else:
        for group in optimizer.param_groups:
            group['lr'] = expected
    clone_optimizer, clone_clock = steps.build_optimizer_and_scheduler(model, training)
    clone_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    clone_clock.load_state_dict(copy.deepcopy(clock.state_dict()))
    if hasattr(clone_clock, 'synchronize'):
        clone_clock.synchronize(clone_optimizer)
    assert_state_equal(optimizer.state_dict(), clone_optimizer.state_dict())
    assert_state_equal(clock.state_dict(), clone_clock.state_dict())


@pytest.mark.parametrize('arm', ['S1', 'S2', 'C3'])
def test_legacy_committed_bundle_resume(tmp_path, arm):
    source = fixture(tmp_path, arm)
    train(source, stop=2)
    path = tmp_path / 'latest.pt'
    save(source, path)
    train(source)
    resumed = fixture(tmp_path, arm)
    load(resumed, path)
    train(resumed)
    for left, right in zip(source[1:4], resumed[1:4]):
        assert_state_equal(left.state_dict(), right.state_dict())


@pytest.mark.parametrize('grid', ['linear', 'geometric'])
@pytest.mark.parametrize('kind', ['GMC', 'LMC-only'])
def test_legacy_c4_schedule_and_restore(grid, kind):
    # Populated moments, mixed owners, exact continuation and rejection before
    # mutation are checked by the existing independent C4 regression helper.
    check_c4_restore(grid, kind, 64)

from pathlib import Path
from src.utils.config import ConfigError
CALR_RECIPE = Path('configs/controlled_exps/tinystories_instruct_linear_calr.yaml')

@pytest.fixture
def calr_runs(tmp_path, audited_inputs):
    return expand(tmp_path, yaml.safe_load(CALR_RECIPE.read_text()))


def test_four_arm_counts_budgets_and_closed_pairs(calr_runs):
    assert [r['arm_id'] for r in calr_runs] == ['S1-linear-poly','S1-linear-CaLR','S2-linear-poly','S2-linear-CaLR']
    assert sum(r['assigned_updates'] for r in calr_runs) == 1394112
    assert sum(r['assigned_tokens'] for r in calr_runs) == 11420565504
    checks = campaign.inspect_campaign_models(calr_runs)
    for r,c in zip(calr_runs,checks):
        campaign.validate_materialized_config(r['resolved_config'])
        assert c['complexity_counts'] == dict(zip(campaign.WIDTH_LABELS,[377408,426560,475712,524864]))
    for kind, pairs in [('schedule', [(0,1),(2,3)]), ('ownership',[(0,2),(1,3)])]:
        for a,b in pairs:
            assert campaign.audit_linear_calr_pair(calr_runs[a]['resolved_config'],calr_runs[b]['resolved_config'],kind=kind)['status']=='passed'

@pytest.mark.parametrize('section,key,value', [('run','seed',43),('model','correction_mode','gmc'),('model','granularities',['g125','g250','g500','g1000']),('training','gradient_clip_norm',2),('dataset','data_seed',43),('model','global_sampling_distribution',{'g250':.1})])
def test_schema6_changed_controls(tmp_path,audited_inputs,section,key,value):
    recipe=yaml.safe_load(CALR_RECIPE.read_text()); recipe['common'][section][key]=value
    with pytest.raises(ConfigError,match=key): expand(tmp_path,recipe)


def test_schema6_extra_arm_and_references(tmp_path,audited_inputs):
    recipe=yaml.safe_load(CALR_RECIPE.read_text()); recipe['arms']['S3']={}
    with pytest.raises(ConfigError,match='arms'): expand(tmp_path,recipe)
    recipe=yaml.safe_load(CALR_RECIPE.read_text()); recipe['references']['S1-cosine']['path']='changed'
    with pytest.raises(ConfigError,match='references'): expand(tmp_path,recipe)

@pytest.mark.parametrize('kind', ['schedule','ownership'])
def test_closed_audit_rejects_undeclared_difference(calr_runs,kind):
    left=calr_runs[0]['resolved_config']; right=copy.deepcopy(calr_runs[1 if kind=='schedule' else 2]['resolved_config'])
    right['monitoring']['notes']='undeclared'
    with pytest.raises(ConfigError,match='monitoring.notes'): campaign.audit_linear_calr_pair(left,right,kind=kind)


def test_fresh_tensors_and_complete_matching_streams(calr_runs,tmp_path,audited_inputs,monkeypatch):
    import torch
    import random
    import hashlib
    import numpy as np
    from src.training.modeling import build_model
    from src.training.packed_corpus import RepeatingNoPaddingDistributedBatchSampler
    from src.utils.reproducibility import seed_model_initialization,seed_for
    originals=[r for r in expand(tmp_path,audited_inputs[0]) if r['arm_id'] in ('S1','S2')]
    all_runs=calr_runs+originals
    # Observe the normal constructor, and retain just one reference state.
    state=None
    old=random.getstate(); old_np=np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            for run in all_runs:
                config=run['resolved_config']; seed_model_initialization(config)
                actual=build_model(config).state_dict()
                if state is None: state={k:v.clone() for k,v in actual.items()}
                else: assert all(torch.equal(v,state[k]) for k,v in actual.items())
    finally:
        random.setstate(old); np.random.set_state(old_np)
    actions=[campaign.expected_action_trace(r['resolved_config']) for r in all_runs]
    assert all(a==actions[0] for a in actions)
    for run in calr_runs:
        old=next(r for r in originals if r['arm_id']==run['reference_arm_id'])
        assert campaign.audit_linear_calr_pair(old['resolved_config'],run['resolved_config'],kind='counterpart')['status']=='passed'
    # Walk every runtime batch/cursor over the declared four-epoch horizon.
    # Samplers keep one epoch's order; neither batches nor the horizon accumulate.
    configs=[r['resolved_config'] for r in all_runs]
    samplers=[RepeatingNoPaddingDistributedBatchSampler(
        dataset_size=5576491,batch_size_per_rank=c['training']['batch_size_per_process'],rank=0,world_size=1,
        planned_sample_count=c['dataset']['optimizer_iteration']['planned_samples'],
        epoch_sample_count=c['dataset']['optimizer_iteration']['aligned_epoch_samples'],
        data_seed=c['dataset']['data_seed'],corpus_hash=c['dataset']['corpus_hash'],
        optimizer_training_manifest_hash=c['dataset']['optimizer_iteration']['optimizer_training_manifest_hash']) for c in configs]
    reference=None
    for sampler in samplers:
        digest=hashlib.sha256(); count=0
        for batch in sampler:
            assert len(batch)==64
            digest.update(np.asarray(batch,dtype='<u8').tobytes()); count+=1
        evidence=(count,digest.hexdigest(),sampler.fixed_epoch_set_hash)
        assert count==348528
        if reference is None: reference=evidence
        else: assert evidence==reference


def test_schema6_cli_missing_reference_is_nonzero(tmp_path,audited_inputs,capsys):
    from scripts.analyze_tinystories_optimizer_ownership import main
    with pytest.raises(SystemExit) as error:
        main(['preflight','--campaign',str(CALR_RECIPE),'--prepared-corpus-dir',str(tmp_path/'corpus'),
            '--tokenizer-dir',str(tmp_path/'tokenizer'),'--output-dir',str(tmp_path/'audit'),
            '--run-output-root',str(tmp_path/'runs')])
    assert error.value.code==1
    assert '--reference-root' in capsys.readouterr().err
    assert not (tmp_path/'audit').exists()


def test_preflight_saved_reference_selection_and_atomic_audits(tmp_path,audited_inputs,monkeypatch):
    import json
    from src.utils.reproducibility import build_optimizer_ownership_signature
    originals=expand(tmp_path,audited_inputs[0])
    checks=campaign.inspect_campaign_models(originals)
    for run,check in zip(originals,checks):
        topology=stable_hash(dict(parameters=check['parameter_descriptors'],owners=check['owners']))
        contract=run['optimizer_ownership_contract']; contract['parameter_topology_hash']=topology
        contract['clipping']={**run['clipping'],'schema_version':1,'topology_hash':topology}
        run['contract_hash']=build_optimizer_ownership_signature(contract)[0]
        for cfg in (run['resolved_config'],run['executable_config']):
            cfg['optimizer_ownership_contract']=contract; cfg['optimizer_ownership_contract_hash']=run['contract_hash']
    reference_root=tmp_path/'saved'
    for label,ref in campaign.LINEAR_CALR_REFERENCES.items():
        base=next(r for r in originals if r['arm_id']==(label[:2] if label.startswith('S1') or label.startswith('S2') else label))
        cfg=copy.deepcopy(base['resolved_config']); cfg['run']['arm_id']=ref['arm_id']; cfg['training']['resolved_learning_rate']=ref['peak']
        if label in ('S1-cosine','S2-cosine'):
            cfg['training']['effective_world_size_source']='single_process'
            cfg['training']['distributed'].update(enabled=False,fsdp={},local_rank=0,rank=0,world_size=1)
            for section,role in (('validation','ordinary_validation'),('adaptive_controller','controller'),('final_holdout','final_holdout')):
                cfg['evaluation'][section]['manifest_hash']=campaign.PINNED_DATA[role+'_manifest_hash']
        path=reference_root/ref['path']/'config.json'; path.parent.mkdir(parents=True); path.write_text(json.dumps(cfg))
    path=reference_root/'optimizer-ownership-v1/campaign/campaign_manifest.json'; path.parent.mkdir(parents=True)
    manifest=dict(schema_version=1,campaign_id='tinystories-optimizer-ownership-v1',runs=originals)
    manifest['manifest_hash']=stable_hash(manifest); path.write_text(json.dumps(manifest))
    before={str(p):p.read_bytes() for p in reference_root.rglob('*') if p.is_file()}
    selected=campaign.inspect_linear_calr_references(reference_root)
    assert len(selected['references'])==8 and selected['terminal_status']=='pending'
    def traces(runs,*args):
        return {r['arm_id']:dict(epochs=[dict(sha256='fixture-data')]*4,actions=campaign.expected_action_trace(r['resolved_config'])) for r in runs}
    monkeypatch.setattr(campaign,'build_expected_traces',traces)
    def schedule(config,path):
        path.write_text('evidence_kind\nanalytic\n')
        return dict(evidence_kind='analytic',source=campaign._source_record(path))
    monkeypatch.setattr(campaign,'export_linear_calr_schedule',schedule)
    kwargs=dict(campaign_path=CALR_RECIPE,prepared_corpus_dir=tmp_path/'corpus',tokenizer_dir=tmp_path/'tokenizer',
        output_dir=tmp_path/'published',run_output_root=tmp_path/'fresh',reference_root=reference_root)
    report=campaign.preflight_campaign(**kwargs)
    assert report['assigned_totals']==dict(updates=1394112,tokens=11420565504)
    assert report['control_audit_status']=='passed' and report['terminal_reference_status']=='pending'
    published=json.loads((tmp_path/'published/campaign_manifest.json').read_text())
    assert len(published['control_audits'])==8
    assert len(list((tmp_path/'published/configs').glob('*.yaml')))==4
    assert not (tmp_path/'fresh').exists()
    assert before=={str(p):p.read_bytes() for p in reference_root.rglob('*') if p.is_file()}
    path=reference_root/'optimizer-ownership-v1/runs/S1/config.json'
    cfg=json.loads(path.read_text()); cfg['model']['unexpected_sampling_control']=True; path.write_text(json.dumps(cfg))
    with pytest.raises(ConfigError,match='unexpected_sampling_control'): campaign.inspect_linear_calr_references(reference_root)
    del cfg['model']['unexpected_sampling_control']; cfg['evaluation']['validation']['manifest_hash']='changed'; path.write_text(json.dumps(cfg))
    with pytest.raises(ConfigError,match='manifest_hash'): campaign.inspect_linear_calr_references(reference_root)



def test_analytic_csv_is_labeled_and_complete(calr_runs,tmp_path):
    import csv
    config=copy.deepcopy(calr_runs[1]['resolved_config'])
    config['training']['linear_calr_schedule_contract']['horizon']=100
    path=tmp_path/'analytic.csv'
    result=campaign.export_linear_calr_schedule(config,path)
    rows=list(csv.DictReader(path.open()))
    assert result['rows']==404 and len(rows)==404
    assert {r['evidence_kind'] for r in rows}=={'analytic'}
    assert all(float(r['analytic_effective_learning_rate'])==0 for r in rows[-4:])


def test_counterpart_predating_disabled_diagnostics(calr_runs,tmp_path,audited_inputs):
    from src.utils.reproducibility import build_optimizer_ownership_signature
    old=next(r['resolved_config'] for r in expand(tmp_path,audited_inputs[0]) if r['arm_id']=='S1')
    old['evaluation'].pop('sign_dynamics')
    old['optimizer_ownership_contract']['evaluation'].pop('sign_dynamics')
    old['optimizer_ownership_contract_hash']=build_optimizer_ownership_signature(old['optimizer_ownership_contract'])[0]
    assert campaign.audit_linear_calr_pair(old,calr_runs[0]['resolved_config'],kind='counterpart')['status']=='passed'
    changed=copy.deepcopy(calr_runs[0]['resolved_config']); changed['evaluation']['sign_dynamics']['cadence_steps']=2
    changed['optimizer_ownership_contract']['evaluation']=copy.deepcopy(changed['evaluation'])
    changed['optimizer_ownership_contract_hash']=build_optimizer_ownership_signature(changed['optimizer_ownership_contract'])[0]
    with pytest.raises(ConfigError,match='cadence_steps'): campaign.audit_linear_calr_pair(old,changed,kind='counterpart')

@pytest.mark.parametrize('field', ['complexity_counts','reporting_counts','ffn_sizes','exponents'])
def test_schema6_rejects_changed_declared_counts(calr_runs,field):
    from src.utils.reproducibility import build_optimizer_ownership_signature
    changed=copy.deepcopy(calr_runs[0]['resolved_config'])
    changed['training']['linear_calr_schedule_contract'][field]['g250']+=1
    with pytest.raises(ConfigError): campaign.validate_materialized_config(changed)
