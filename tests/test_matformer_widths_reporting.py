"""Reporting prerequisites for the standalone barrier and restart-safe finalizer."""
import copy
import csv
import json
import shutil
from pathlib import Path

import pytest
import yaml

from src.evaluation import optimizer_ownership as campaign
from src.utils.config import ConfigError
from test_optimizer_ownership_campaign import audited_inputs
import test_optimizer_ownership_reporting as legacy


@pytest.fixture
def terminal_campaign(tmp_path, audited_inputs, monkeypatch):
    recipe = yaml.safe_load(Path('configs/controlled_exps/tinystories_instruct_matformer_widths.yaml').read_text())
    return legacy.terminal_campaign.__wrapped__(tmp_path, (recipe, *audited_inputs[1:]), monkeypatch)


def test_new_reports_and_selected_barrier_reader(tmp_path, terminal_campaign):
    manifest, root = terminal_campaign
    selected = campaign.inspect_selected_terminals(manifest, ['ST-g125', 'ST-g250', 'ST-g500', 'ST-g1000'])
    assert len(selected) == 4
    campaign.freeze_campaign(campaign_manifest=manifest, run_root=root, output_dir=tmp_path/'frozen')
    report = campaign.report_campaign(manifest=tmp_path/'frozen/frozen_manifest.json', output_dir=tmp_path/'report')
    rows = json.loads((tmp_path/'report/endpoints.json').read_text())['endpoints']
    with (tmp_path/'report/endpoints.csv').open() as f: exported = list(csv.DictReader(f))
    assert len(rows) == len(exported) == report['endpoint_count'] == 24
    assert len(report['individual_reports']) == 9
    assert [r['ffn_dimension'] for r in rows[:4]] == [32, 64, 128, 256]
    assert exported == [{k: campaign.endpoint_csv_value(v) for k,v in row.items()} for row in rows]
    assert report['output_sha256']


@pytest.mark.parametrize('field,value', [('global_step', 87132), ('evaluation_role', 'final_holdout'), ('evaluation_target_tokens', 7), ('actual_tokens', 1), ('contract_hash', 'old'), ('checkpoint_path', 'best.pt')])
def test_new_terminal_rejections(tmp_path, terminal_campaign, field, value):
    legacy.test_freeze_rejects_terminal_mismatch(tmp_path, terminal_campaign, field, value)


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'nonfinite', 'count', 'targets', 'exp', 'role'])
def test_new_endpoint_rejections(tmp_path, terminal_campaign, mutation):
    legacy.test_freeze_rejects_bad_endpoints_even_partial(tmp_path, terminal_campaign, mutation)


def test_combined_uses_only_historical_standalones(tmp_path, terminal_campaign, audited_inputs, monkeypatch):
    manifest, root = terminal_campaign
    campaign.freeze_campaign(campaign_manifest=manifest, run_root=root, output_dir=tmp_path/'new-frozen')
    old_root = tmp_path/'old'; old_root.mkdir()
    old_manifest, old_runs = legacy.terminal_campaign.__wrapped__(old_root, audited_inputs, monkeypatch)
    campaign.freeze_campaign(campaign_manifest=old_manifest, run_root=old_runs, output_dir=old_root/'frozen')
    for arm in ('S1','S2','C1','C2','C3'): shutil.rmtree(old_runs/arm)
    report = campaign.report_matformer_widths_comparison(manifest=tmp_path/'new-frozen/frozen_manifest.json',
        reference_manifest=old_root/'frozen/frozen_manifest.json', output_dir=tmp_path/'combined')
    rows = json.loads((tmp_path/'combined/combined_endpoints.json').read_text())['endpoints']
    assert report['endpoint_count'] == len(rows) == 28
    old_g750=next(r for r in rows if r['historical_reference'] and r['width']=='g750')
    assert (old_g750['width_fraction'],old_g750['ffn_dimension'],old_g750['non_embedding_parameters']) == (.75,192,213568)
    with (tmp_path/'combined/combined_endpoints.csv').open() as f:
        assert list(csv.DictReader(f)) == [{k: campaign.endpoint_csv_value(v) for k,v in r.items()} for r in rows]
    figure = campaign.matformer_endpoint_figure(rows, metric='loss')
    assert len(figure.axes[0].lines) == 13
    assert all(len(line.get_xdata()) == 4 for line in figure.axes[0].lines[:5])
    assert all(line.get_linestyle() == 'None' for line in figure.axes[0].lines[5:])
    assert len({tuple(r['endpoint_identity']) for r in rows}) == 28
    for dimension in (64,128,256):
        assert len([r for r in rows if r['arm_id'].startswith('ST-') and r['ffn_dimension']==dimension]) == 2
    assert all(Path(path).stat().st_size > 100 for path in report['figures'])
    before = (tmp_path/'combined/comparison_report.json').read_bytes()
    legacy.change_terminal(old_runs, 'ST-g750', lambda r: r.update(actual_tokens=1))
    with pytest.raises(ConfigError):
        campaign.report_matformer_widths_comparison(manifest=tmp_path/'new-frozen/frozen_manifest.json',
            reference_manifest=old_root/'frozen/frozen_manifest.json', output_dir=tmp_path/'combined')
    assert (tmp_path/'combined/comparison_report.json').read_bytes() == before


@pytest.mark.parametrize('arm',['C1','C3'])
def test_short_real_clipping_reaches_strict_terminal_reader(tmp_path,monkeypatch,arm):
    import torch
    from test_matformer_widths_campaign import mw_resume_fixture
    from src.training import steps
    from src.training.run import complete_ownership_terminal,build_ownership_run_summary
    from src.utils.metrics import MetricsJournal,append_optimizer_ownership_observation
    bundle=mw_resume_fixture(tmp_path,arm)
    config,model,opt,clock,batches,state=bundle
    # Explicit short fixture budgets and ordinary-evaluation identity. No full
    # corpus, production budget or production readiness is represented here.
    config['run']['continuation']['enabled']=True
    config['validation_manifest_hash']='diagnostic-ordinary'
    contract=json.loads(json.dumps(config['optimizer_ownership_contract']))
    config['optimizer_ownership_contract']=contract
    contract.update(evaluation={'validation':copy.deepcopy(config['evaluation']['validation'])},
        initialization=None,clipping=copy.deepcopy(config['training']['gradient_clipping']))
    config['optimizer_ownership_contract_hash']=campaign.stable_hash(contract)
    expected=dict(actions=campaign.expected_action_trace(config),epochs=campaign.expected_epoch_traces(copy.copy(batches.batch_sampler),epochs=4))
    journal=MetricsJournal(config['run']['output_dir'],artifact_state=state,artifact_io_config=config)
    opt._ownership_dataloader=batches
    opt._ownership_observer=lambda:append_optimizer_ownership_observation(config,state,train_dataloader=batches)
    steps.train_for_steps(config,model,batches,[],opt,clock,torch.device('cpu'),run_state=state,metrics_journal=journal)
    journal.flush()
    evaluation=[dict(input_ids=torch.arange(1,9).reshape(1,8),labels=torch.arange(1,9).reshape(1,8))]
    terminal=complete_ownership_terminal(config,model,opt,clock,state,evaluation,torch.device('cpu'))
    summary=dict(run_id=config['run']['run_id'],status='completed',**build_ownership_run_summary(config,model,opt,state))
    root=Path(config['run']['output_dir']);(root/'run_summary.json').write_text(json.dumps(summary))
    definition={k:terminal[k] for k in ('arm_id','campaign_id','run_id','contract_hash','representation','state_scope','clipping','initialization','assigned_updates','assigned_tokens','assigned_epochs')}
    definition.update(optimizer_ownership_contract=contract,resolved_config=config,endpoint_widths=['g125','g250','g500','g1000'],source_width=None)
    monkeypatch.setattr(campaign,'TOKENS_PER_UPDATE',8)
    monkeypatch.setattr(campaign,'VALIDATION_SEQUENCES',1)
    monkeypatch.setattr(campaign,'VALIDATION_TARGET_TOKENS',7)
    monkeypatch.setattr(campaign,'PINNED_DATA',{**campaign.PINNED_DATA,'ordinary_validation_manifest_hash':'diagnostic-ordinary'})
    saved=campaign._inspect_terminal_run(root,definition,expected,allow_partial=False)
    assert len(saved['endpoints'])==4 and saved['observations']['committed_updates']==8
    (root/'optimizer_ownership_clipping.jsonl').write_text('')
    with pytest.raises(ConfigError,match='clipping'):
        campaign._inspect_terminal_run(root,definition,expected,allow_partial=False)


@pytest.mark.parametrize('source',['latest.pt','run_summary.json','terminal_validation_results.json','optimizer_ownership_trace.jsonl','metrics.csv'])
def test_new_frozen_source_tampering(tmp_path,terminal_campaign,source):
    legacy.test_report_rejects_replaced_frozen_source(tmp_path,terminal_campaign,source)


@pytest.mark.parametrize('mutation',['summary_control','summary_tokens','checkpoint_step','checkpoint_purpose','actions','epochs'])
def test_new_budget_checkpoint_and_trace_rejections(tmp_path,terminal_campaign,monkeypatch,mutation):
    legacy.test_freeze_rejects_controls_checkpoint_and_trace(tmp_path,terminal_campaign,monkeypatch,mutation)


def test_new_duplicate_and_failed_freeze_publication(tmp_path,terminal_campaign,monkeypatch):
    legacy.test_freeze_duplicate_run_dirs_and_publication_failure(tmp_path,terminal_campaign,monkeypatch)


def test_new_failed_report_publication(tmp_path,terminal_campaign,monkeypatch):
    legacy.test_report_stages_exports_and_preserves_sources(tmp_path,terminal_campaign,monkeypatch)


def test_new_resource_attempt_invalidates_freeze(tmp_path,terminal_campaign):
    legacy.test_report_detects_new_resource_attempt_after_freeze(tmp_path,terminal_campaign)


@pytest.mark.parametrize('field,value',[('width_fraction',1.),('ffn_dimension',192)])
def test_dense_source_width_cannot_be_relabelled(tmp_path,terminal_campaign,field,value):
    manifest,root=terminal_campaign
    legacy.change_terminal(root,'ST-g125',lambda d:d['endpoints'][0].update({field:value}))
    with pytest.raises(ConfigError,match=field):
        campaign.inspect_selected_terminals(manifest,['ST-g125'])


def test_complete_resource_claim_requires_measured_attempts(tmp_path,terminal_campaign):
    manifest,root=terminal_campaign
    path=root/'ST-g125/run_summary.json';value=json.loads(path.read_text())
    value['optimizer_ownership']['resources']['measurement_complete']=True
    path.write_text(json.dumps(value))
    with pytest.raises(ConfigError,match='resource'):
        campaign.inspect_selected_terminals(manifest,['ST-g125'])
