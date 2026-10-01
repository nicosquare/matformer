"""Synthetic CPU update/restore probes; never completed production training."""
import copy
import pytest
import torch
from src.evaluation import optimizer_ownership as campaign
from src.training import checkpointing as cp, steps
from src.training.modeling import build_model
from src.utils.reproducibility import stable_hash, seed_training_randomness, capture_rng_state
from src.utils.config import ConfigError
from test_optimizer_ownership import runtime_fixture, assert_state_equal
from test_optimizer_ownership_resume import train, save, load

ARMS = tuple(a['arm_id'] for a in campaign.LINEAR_CALR_ARMS)

def fixture(tmp_path, arm=ARMS[0], horizon=72, device=None, real_shape=False):
    import os
    device = device or os.environ.get('LINEAR_CALR_DIAGNOSTIC_DEVICE', 'cpu')
    definition = next(a for a in campaign.LINEAR_CALR_ARMS if a['arm_id'] == arm)
    config, _, _, _, batches, _ = runtime_fixture(tmp_path, scope=definition['state_scope'], max_steps=horizon, device=device)
    config['model']['variant'] = 'slicing'
    config['run']['run_id'] = 'synthetic-' + arm
    contract = campaign.linear_calr_schedule_contract(definition)
    contract.update(horizon=horizon, run_id=config['run']['run_id'])
    config['training'].update(linear_calr_schedule_contract=contract, resolved_learning_rate=.008,
                              resolved_warmup_steps=64)
    config['training']['optimizer_state_contract'].update(linear_calr_schedule_contract=contract,
        linear_calr_schedule_contract_hash=stable_hash(contract))
    config['optimizer_ownership_contract'] = dict(schema_version=1, arm_id=arm, campaign_id='synthetic-probe')
    config['optimizer_ownership_contract_hash'] = stable_hash(config['optimizer_ownership_contract'])
    if real_shape:
        config['model'].update(d_model=64,num_layers=4,num_attention_heads=4,vocab_size=2048,context_length=128,intermediate_size=256)
        config['training'].update(batch_size_per_process=64,expected_tokens_per_step=8192,token_budget=horizon*8192)
        tokens=torch.arange(128).reshape(1,128).repeat(64,1)
        batches=[dict(input_ids=tokens,labels=tokens.clone()) for _ in range(2)]
    seed_training_randomness(config)
    model = build_model(config).to(device)
    opt, clock = steps.build_optimizer_and_scheduler(model, config['training'])
    return config, model, opt, clock, batches, cp.build_initial_continuation_state(config)

@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('boundary', [1,63,64,65,69,72])
def test_own_arm_resume(tmp_path, arm, boundary):
    bundle = fixture(tmp_path, arm)
    train(bundle, stop=boundary)
    path=tmp_path/'latest.pt'; save(bundle,path)
    train(bundle); rng=capture_rng_state()
    restored=fixture(tmp_path,arm); load(restored,path); train(restored)
    for a,b in zip(bundle[1:4],restored[1:4]): assert_state_equal(a.state_dict(),b.state_dict())
    for key in ('step','tokens_seen','microstep','epoch','batch_index','global_sampling_state','last_applied_schedule_record','applied_schedule_watermark'):
        assert_state_equal(bundle[-1][key],restored[-1][key])
    assert_state_equal(rng,capture_rng_state())

@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('damage', ['arm','grid','scope','count','gamma','horizon','record','watermark','clock','rng'])
def test_reject_before_mutation(tmp_path, arm, damage):
    source=fixture(tmp_path,arm); train(source,stop=3)
    path=tmp_path/'latest.pt';save(source,path);payload=torch.load(path,weights_only=False)
    contract=payload['linear_calr_schedule_contract']
    if damage=='arm':contract['arm_id']='other'
    elif damage=='grid':contract['ordered_granularities'].reverse()
    elif damage=='scope':contract['optimizer_state_scope']='other'
    elif damage=='count':contract['complexity_counts']['g250']+=1
    elif damage=='gamma':contract['exponents']['g250']+=.1
    elif damage=='horizon':contract['horizon']+=1
    elif damage=='record':payload['last_applied_schedule_record']['applied_learning_rates']=[.4]
    elif damage=='watermark':payload['applied_schedule_watermark']['step']+=1
    elif damage=='clock':payload['scheduler_state_dict']['position']+=1
    elif damage=='rng':payload['reproducibility']['rng_state']['python']=()
    torch.save(payload,path); restored=fixture(tmp_path,arm)
    before=[copy.deepcopy(x.state_dict()) for x in restored[1:4]];rng=capture_rng_state()
    with pytest.raises(ConfigError):load(restored,path)
    for a,b in zip(restored[1:4],before):assert_state_equal(a.state_dict(),b)
    assert_state_equal(rng,capture_rng_state())

@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('stage', ['optimizer','restoration','scheduler','accounting'])
def test_post_mutation_failure_poisoned(tmp_path,monkeypatch,arm,stage):
    bundle=fixture(tmp_path,arm);train(bundle,stop=2)
    path=tmp_path/'latest.pt';save(bundle,path);before=path.read_bytes()
    opt,clock,state=bundle[2],bundle[3],bundle[-1]
    def fail(*a,**kw):raise RuntimeError('injected '+stage)
    if stage=='optimizer':
        owners=[e.optimizer for e in opt.entries] if hasattr(opt,'entries') else [opt]
        for owner in owners:
            original=owner.step
            def mutate(*a,_original=original,**kw):_original(*a,**kw);fail()
            monkeypatch.setattr(owner,'step',mutate)
    elif stage=='restoration':monkeypatch.setattr(steps,'restore_nominal_group_rates',fail)
    elif stage=='scheduler':monkeypatch.setattr(clock,'step',fail)
    else:monkeypatch.setattr(steps.training_data,'validate_campaign_sampler_boundary',fail)
    with pytest.raises(RuntimeError,match='injected'):train(bundle)
    assert state['optimizer_poisoned'] and state['update_in_flight']
    with pytest.raises(ConfigError,match='unsafe|poisoned'):save(bundle,path)
    assert before==path.read_bytes()


def seed_boundary(bundle, position):
    """Fabricate coherent synthetic histories at a global position, without training it."""
    from src.utils.reproducibility import dedicated_random
    config,model,opt,clock,_,state=bundle
    counts=dict.fromkeys(config['model']['granularities'],0)
    generator=dedicated_random(config,'granularity_selection')
    last=None
    for _ in range(position):
        last=config['model']['granularities'][generator.randrange(4)];counts[last]+=1
    sampling=state['global_sampling_state']
    sampling.update(held_granularity=last,window_index=position-1,successful_updates_in_window=1,
                    total_successful_updates=position,exposure_counts=counts)
    if hasattr(opt,'entries'):
        owners=[(e.optimizer,counts[e.granularity]) for e in opt.entries]
        opt.successful_update_counts=counts.copy();opt.total_successful_updates=position;opt.last_active_granularity=last
    else:owners=[(opt,position)]
    for owner,n in owners:
        for group in owner.param_groups:
            for p in group['params']:
                owner.state[p]=dict(step=torch.tensor(float(n)),exp_avg=torch.full_like(p,.003),exp_avg_sq=torch.full_like(p,.0002))
    raw=clock.state_dict();formula=clock._scheduler.lr_lambdas[0];peak=clock._scheduler.base_lrs[0]
    nominal=peak*formula(position);previous=peak*formula(position-1)
    raw['position']=position;raw['last_committed_learning_rates']=[previous]
    raw['scheduler_state_dict'].update(last_epoch=position,_step_count=position+1,_last_lr=[nominal])
    raw['carrier_optimizer_state_dict']['param_groups'][0]['lr']=nominal
    clock.load_state_dict(raw);clock.synchronize(opt)
    state.update(step=position,last_completed_step=position,microstep=position,tokens_seen=position*config['training']['expected_tokens_per_step'],
        content_tokens_seen=position*config['training']['expected_tokens_per_step'],epoch=(position-1)//2,batch_index=(position-1)%2+1,
        optimizer_width_selection_counts=counts.copy(),optimizer_update_counts=counts.copy() if hasattr(opt,'entries') else {'shared':position},
        optimizer_quarter_activation_counts={f'O-{q}':sum(counts[w] for w in config['model']['granularities'][i:]) for i,q in enumerate('ABCD')},
        optimizer_total_successful_updates=position,optimizer_last_active_granularity=last,global_scheduler_position=position)
    # A synthetic last record is explicit fixture state, not measured execution evidence.
    provenance=dict(epoch=state['epoch'],batch_index=max(state['batch_index']-1,0))
    state['last_optimizer_batch_provenance']=provenance
    from src.training.schedules import polynomial_schedule_evidence
    e=polynomial_schedule_evidence(config['training']['linear_calr_schedule_contract'],last,position-1)
    c=config['training']['linear_calr_schedule_contract']
    if hasattr(bundle[4], 'batch_sampler'):
        sampler=bundle[4].batch_sampler
        sampler.set_cursor(position*config['training']['batch_size_per_process'])
        state['sampler_state']=sampler.state_dict()
        state.update(epoch=state['sampler_state']['epoch_index'],batch_index=state['sampler_state']['within_epoch_cursor']//config['training']['batch_size_per_process'])
    record=dict(evidence_kind='applied',run_id=config['run']['run_id'],arm_id=c['arm_id'],schedule_contract_hash=stable_hash(c),
        step=position,pre_update_position=position-1,width=last,owner=last if hasattr(opt,'entries') else 'shared',
        complexity=c['complexity_counts'][last],exponent=c['exponents'][last],applied_learning_rates=[e['analytic_effective_learning_rate']],
        nominal_learning_rates=[e['analytic_nominal_learning_rate']],action_ordinal=position,batch_provenance=provenance,packed_tokens=config['training']['expected_tokens_per_step'])
    state.update(last_applied_schedule_record=record,
        applied_schedule_watermark=dict(step=position,last_record_hash=stable_hash(record),chain_hash=stable_hash(['synthetic-seeded-probe',position])))

@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('boundary', [63,64,65,174296,87132,174264,261396,348527,348528])
def test_full_horizon_seeded_resume(tmp_path,arm,boundary):
    bundle=packed_boundary_fixture(tmp_path,arm);seed_boundary(bundle,boundary)
    path=tmp_path/'seeded.pt';save(bundle,path)
    train(bundle,stop=boundary+2);rng=capture_rng_state()
    restored=packed_boundary_fixture(tmp_path,arm);load(restored,path);train(restored,stop=boundary+2)
    for a,b in zip(bundle[1:4],restored[1:4]):assert_state_equal(a.state_dict(),b.state_dict())
    for key in ('step','epoch','batch_index','tokens_seen','global_sampling_state','last_applied_schedule_record','applied_schedule_watermark'):
        assert_state_equal(bundle[-1][key],restored[-1][key])
    assert_state_equal(rng,capture_rng_state())

@pytest.mark.parametrize('arm', ARMS)
def test_terminal_output_recovery_zero_updates(tmp_path,monkeypatch,arm):
    from pathlib import Path
    from src.training import run
    from src.evaluation import validation
    bundle=fixture(tmp_path,arm);train(bundle)
    config,model,opt,clock,_,state=bundle
    root=Path(config['run']['output_dir']);root.mkdir(parents=True,exist_ok=True)
    config['validation_manifest_hash']='ordinary-fixture'
    path=root/'terminal.pt';save(bundle,path)
    state.update(latest_checkpoint_path=str(path),latest_checkpoint_step=72)
    config['validation_manifest_hash']='ordinary-fixture'
    monkeypatch.setattr(run,'_validate_terminal_endpoints',lambda *a:None)
    endpoints=[dict(evaluation_examples=1,evaluation_target_tokens=7)]
    monkeypatch.setattr(validation,'evaluate_ownership_terminal',lambda *a:endpoints)
    before=path.read_bytes();original=run.write_json_artifact
    monkeypatch.setattr(run,'write_json_artifact',lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('sidecar failed')))
    with pytest.raises(RuntimeError,match='sidecar failed'):run.complete_ownership_terminal(config,model,opt,clock,state,[],torch.device('cpu'))
    monkeypatch.setattr(run,'write_json_artifact',original)
    owners=[e.optimizer for e in opt.entries] if hasattr(opt,'entries') else [opt]
    for owner in owners:monkeypatch.setattr(owner,'step',lambda:pytest.fail('terminal recovery trained'))
    result=run.complete_ownership_terminal(config,model,opt,clock,state,[],torch.device('cpu'))
    assert result['actual_updates']==72 and result['actual_tokens']==72*8 and path.read_bytes()==before

@pytest.mark.skipif(__import__('os').environ.get('LINEAR_CALR_DIAGNOSTIC_DEVICE')!='cuda',reason='separately authorized CUDA diagnostic')
@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('boundary',[63,64,65,87132,174264,261396,348527,348528])
def test_gpu_boundary(tmp_path,arm,boundary):
    bundle=packed_boundary_fixture(tmp_path,arm,device='cuda',real_shape=True);seed_boundary(bundle,boundary)
    path=tmp_path/'seeded-cuda.pt';save(bundle,path);train(bundle,stop=boundary+1)
    restored=packed_boundary_fixture(tmp_path,arm,device='cuda',real_shape=True);load(restored,path);train(restored,stop=boundary+1)
    for a,b in zip(bundle[1:4],restored[1:4]):assert_state_equal(a.state_dict(),b.state_dict())


class SyntheticPackedDataset(torch.utils.data.Dataset):
    """Virtual designated sequences; no corpus or training completion claim."""
    def __init__(self, size, context, vocab):
        self.size,self.context,self.vocab=size,context,vocab
    def __len__(self):return self.size
    def __getitem__(self,index):
        tokens=(torch.arange(self.context)+index) % self.vocab
        return dict(input_ids=tokens,labels=tokens.clone())


def packed_boundary_fixture(tmp_path,arm,*,device='cpu',real_shape=False):
    from src.training.packed_corpus import RepeatingNoPaddingDistributedBatchSampler,REPEATED_EPOCH_ORDER_VERSION
    bundle=list(fixture(tmp_path,arm,horizon=348528,device=device,real_shape=real_shape))
    config=bundle[0];batch=config['training']['batch_size_per_process'];context=config['model']['context_length']
    epoch_samples=87132*batch
    config['dataset'].update(mode='packed_mmap',data_seed=42,optimizer_iteration=dict(
        mode='repeat_epochs',epoch_order='deterministic_per_epoch',ordering_policy_version=REPEATED_EPOCH_ORDER_VERSION,
        aligned_epoch_samples=epoch_samples,aligned_epoch_tokens=epoch_samples*context,excluded_tail_samples=43,excluded_tail_tokens=43*context))
    sampler=RepeatingNoPaddingDistributedBatchSampler(epoch_samples+43,batch,0,1,
        planned_sample_count=348528*batch,epoch_sample_count=epoch_samples,corpus_hash='synthetic-corpus',optimizer_training_manifest_hash='synthetic-role')
    bundle[4]=torch.utils.data.DataLoader(SyntheticPackedDataset(epoch_samples+43,context,config['model']['vocab_size']),batch_sampler=sampler)
    bundle[2]._ownership_dataloader=bundle[4]
    return bundle

@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('damage',[None,'record','hash','truncated'])
def test_streamed_trace_checkpoint_watermark(tmp_path,arm,damage):
    from src.utils.metrics import append_optimizer_ownership_observation,validate_linear_calr_trace_watermark
    import json
    from pathlib import Path
    bundle=fixture(tmp_path,arm);config,model,opt,clock,batches,state=bundle
    opt._ownership_observer=lambda:append_optimizer_ownership_observation(config,state,train_dataloader=batches)
    train(bundle,stop=3);path=tmp_path/'latest.pt';save(bundle,path)
    payload=torch.load(path,weights_only=False)
    trace=Path(config['run']['output_dir'])/'optimizer_ownership_trace.jsonl'
    rows=[json.loads(line) for line in trace.read_text().splitlines()]
    if damage=='record':rows[0]['applied_schedule_record']['applied_learning_rates']=[.8]
    elif damage=='hash':rows[0]['applied_schedule_watermark']['chain_hash']='bad'
    elif damage=='truncated':rows.pop()
    if damage:trace.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    if damage:
        with pytest.raises(ConfigError,match='trace|watermark'):validate_linear_calr_trace_watermark(config,payload)
    else:validate_linear_calr_trace_watermark(config,payload)


def test_real_campaign_diagnostic_retains_canonical_contract(calr_runs,tmp_path):
    import yaml
    from scripts.preflight_tinystories_linear_calr import diagnostic_identity
    from src.utils.config import resolve_run_config
    for definition in calr_runs:
        raw=copy.deepcopy(definition['executable_config']);raw['run']['output_dir']=str(tmp_path/'probe'/definition['run_id'])
        path=tmp_path/'probe.yaml';path.write_text(yaml.safe_dump(raw))
        with diagnostic_identity(definition,tmp_path/'probe',definition['run_id']):
            resolved=resolve_run_config(path,create_output_dirs=True)
            campaign.validate_materialized_config(resolved)
            model=build_model(resolved)
            opt,clock=steps.build_optimizer_and_scheduler(model,resolved['training'])
            assert clock.current_learning_rates==(0.,)

from test_linear_calr_campaign import calr_runs
from test_optimizer_ownership_campaign import audited_inputs


def test_costs_keep_validation_and_failed_attempts_separate(tmp_path,monkeypatch):
    from src.training.run import _resource_attempt_observer,ResourceAttemptLedger
    import src.training.run as runtime
    bundle=fixture(tmp_path);config,_,opt,_,_,state=bundle
    clock=iter(range(100))
    monkeypatch.setattr(runtime.time,'perf_counter',lambda:float(next(clock)))
    observer=_resource_attempt_observer(config,torch.device('cpu'),started_at=0.,source_checkpoint=None)
    opt._resource_observer=observer
    observer(run_state=state,boundary='attempt')
    observer(run_state=state,boundary='committed')
    steps.measured_ordinary_validation(config,opt,state,lambda:None)
    observer(run_state=state,boundary='attempt')
    observer(run_state=state,boundary='failed')
    result=ResourceAttemptLedger(config['run']['output_dir'],run_id=config['run']['run_id']).summary()
    assert result['ordinary_validation_seconds']==1. and result['training_update_seconds']==2.
    assert result['failed_attempt_seconds']==result['elapsed_seconds'] and result['attempted_steps']==2
