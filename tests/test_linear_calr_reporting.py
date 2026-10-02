"""Provenance and arithmetic fixtures; synthetic budgets do not imply training."""
import copy
import csv
import json
import math
from pathlib import Path

import pytest
import torch

from src.evaluation import optimizer_ownership as campaign
from src.utils.config import ConfigError
from test_optimizer_ownership_campaign import audited_inputs
from test_optimizer_ownership_reporting import terminal_campaign, change_terminal


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def reference_fixture(tmp_path, terminal_campaign):
    manifest_path, runs = terminal_campaign
    base = tmp_path/'references'
    native = base/'optimizer-ownership-v1'
    native.mkdir(parents=True)
    (native/'campaign').symlink_to(manifest_path.parent, target_is_directory=True)
    (native/'runs').symlink_to(runs, target_is_directory=True)
    manifest = campaign._read_preflight_manifest(manifest_path)
    for run in manifest['runs']:
        write(runs/run['arm_id']/'config.json', run['resolved_config'])
    for scope in ('S1','S2'):
        label = scope+'-cosine-004'; ref = campaign.LINEAR_CALR_REFERENCES[label]
        root = base/ref['path']; root.mkdir(parents=True)
        original = next(r for r in manifest['runs'] if r['arm_id']==scope)
        config = copy.deepcopy(original['resolved_config'])
        config['run'].update(arm_id=ref['arm_id'],run_id=ref['arm_id'],output_dir=str(root))
        config['training'].update(resolved_learning_rate=.004,learning_rate=.004)
        config['training']['scheduler']['resolved_warmup_steps']=64
        config['evaluation']['validation']['manifest_hash']=campaign.PINNED_DATA['ordinary_validation_manifest_hash']
        write(root/'config.json',config)
        checkpoint=root/'checkpoints/latest.pt'; checkpoint.parent.mkdir()
        # Legacy S1/S2 did not maintain these newer ownership fields.
        payload=dict(checkpoint_kind='resumable_training',checkpoint_schema_version=1,
            run_id=ref['arm_id'],step=348528,tokens_seen=2855141376,
            global_scheduler_position=None,optimizer_total_successful_updates=0,
            optimizer_training_manifest_hash=campaign.PINNED_DATA['optimizer_training_manifest_hash'],
            ordinary_validation_manifest_hash=campaign.PINNED_DATA['ordinary_validation_manifest_hash'],
            tokenizer_manifest_hash=campaign.PINNED_DATA['tokenizer_manifest_hash'],
            model_state_dict={'fixture':torch.ones(1)},scheduler_state_dict={'last_epoch':348528},
            optimizer_state_dict={'fixture':{}} if scope=='S1' else None,
            optimizer_state_collection={'fixture':{}} if scope=='S2' else None,
            reproducibility={'source_files_sha256':{'fixture':'source'}})
        torch.save(payload,checkpoint)
        summary=dict(status='completed',run_id=ref['arm_id'],committed_optimizer_steps=348528,
            tokens_seen=2855141376,terminal_checkpoint_path=str(checkpoint),
            terminal_checkpoint_sha256=campaign._source_record(checkpoint)['sha256'],
            terminal_checkpoint_purpose='resumable_training',
            validation_manifest_hash=campaign.PINNED_DATA['ordinary_validation_manifest_hash'],
            optimizer_training_manifest_hash=campaign.PINNED_DATA['optimizer_training_manifest_hash'],
            tokenizer_manifest_hash=campaign.PINNED_DATA['tokenizer_manifest_hash'],
            validation_loss_aggregation=campaign.VALIDATION_AGGREGATION)
        write(root/'run_summary.json',summary)
        fields=['granularity','non_embedding_parameters','final_validation_loss','final_validation_perplexity',
                'evaluation_target_tokens','validation_manifest_hash']
        with (root/'scaling_results.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for width in campaign.WIDTHS:
                w.writerow(dict(granularity=width['label'],non_embedding_parameters=width['non_embedding_parameters'],
                    final_validation_loss=2.,final_validation_perplexity=math.exp(2),evaluation_target_tokens=36195,
                    validation_manifest_hash=summary['validation_manifest_hash']))
        with (root/'metrics.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=['run_id','step','split','granularity','loss','perplexity','learning_rate',
                'validation_manifest_hash','validation_loss_aggregation','evaluation_target_tokens']);w.writeheader()
            for width in campaign.WIDTHS:
                for step in (64,348528):
                    w.writerow(dict(run_id=ref['arm_id'],step=step,split='validation',granularity=width['label'],
                        loss=2.,perplexity=math.exp(2),validation_manifest_hash=summary['validation_manifest_hash'],
                        validation_loss_aggregation=campaign.VALIDATION_AGGREGATION,evaluation_target_tokens=36195))
    return base


def test_readonly_eight_references_twenty_endpoints(reference_fixture):
    before={str(p):campaign._source_record(p)['sha256'] for p in reference_fixture.rglob('*') if p.is_file()}
    refs=campaign.validate_linear_calr_references(reference_fixture)
    assert len(refs['runs'])==8
    assert sum(len(r['endpoints']) for r in refs['runs'])==20
    assert refs['status']=='complete'
    assert before=={str(p):campaign._source_record(p)['sha256'] for p in reference_fixture.rglob('*') if p.is_file()}
    assert {r['adapter'] for r in refs['runs']}=={'native','legacy_peak_lr_v1'}


@pytest.mark.parametrize('damage',['missing','intermediate','hash','role','budget','manifest','exp','count','identity'])
def test_native_reference_rejects_damage(reference_fixture,damage):
    root=reference_fixture/'optimizer-ownership-v1/runs/S1'
    if damage=='missing':(root/'terminal_validation_results.json').unlink()
    elif damage=='identity':
        p=root/'config.json';d=json.loads(p.read_text());d['run']['run_id']='other';write(p,d)
    else:
        def mutate(d):
            if damage=='exp':d['endpoints'][0]['perplexity']=1.
            elif damage=='count':d['endpoints'][0]['non_embedding_parameters']=1
            else:d.update({{'intermediate':'global_step','hash':'checkpoint_sha256','role':'evaluation_role',
                'budget':'actual_tokens','manifest':'validation_manifest_hash'}[damage]:1})
        change_terminal(root.parent,'S1',mutate)
    with pytest.raises((ConfigError,OSError)):campaign.validate_linear_calr_references(reference_fixture)


@pytest.mark.parametrize('damage',['checkpoint','step','tokens','kind','identity','manifest','scheduler','endpoint','metrics','control'])
def test_legacy_reference_rejects_damage(reference_fixture,damage):
    root=reference_fixture/campaign.LINEAR_CALR_REFERENCES['S1-cosine-004']['path']
    if damage=='checkpoint':(root/'checkpoints/latest.pt').unlink()
    elif damage in ('step','tokens','kind','identity','manifest','scheduler'):
        p=root/'checkpoints/latest.pt';d=torch.load(p,weights_only=False)
        d[{'step':'step','tokens':'tokens_seen','kind':'checkpoint_kind','identity':'run_id',
           'manifest':'ordinary_validation_manifest_hash','scheduler':'scheduler_state_dict'}[damage]]=0
        torch.save(d,p)
        s=json.loads((root/'run_summary.json').read_text());s['terminal_checkpoint_sha256']=campaign._source_record(p)['sha256'];write(root/'run_summary.json',s)
    elif damage=='control':
        p=root/'config.json';d=json.loads(p.read_text());d['training']['adam_beta2']=.5;write(p,d)
    elif damage=='endpoint':
        p=root/'scaling_results.csv';p.write_text(p.read_text().replace('7.38905609893065','1.0'))
    else:
        p=root/'metrics.csv';p.write_text(p.read_text().replace('348528','348527'))
    with pytest.raises((ConfigError,OSError)):campaign.validate_linear_calr_references(reference_fixture)


@pytest.fixture
def rows():
    result=[]
    for width in campaign.WIDTHS:
        for series,peak,offset in [('standalone-cosine',.008,.0),('S1-cosine',.008,.1),('S2-cosine',.008,.2),
            ('S1-cosine',.004,.11),('S2-cosine',.004,.21),('S1-polynomial',.008,.3),('S2-polynomial',.008,.4),
            ('S1-carl',.008,.5),('S2-carl',.008,.7)]:
            loss=2.+offset+width['source_fraction']/10
            result.append(dict(endpoint_id=f"{series}-{peak}-{width['label']}",series=series,peak_lr=peak,
                width=width['label'],loss=loss,perplexity=math.exp(loss),non_embedding_parameters=width['non_embedding_parameters'],
                actual_updates=87132 if series.startswith('standalone') else 348528))
    return result


def test_pairs_and_interactions(rows):
    pairs,interactions,supplemental=campaign.linear_calr_comparisons(rows)
    assert len(pairs)==48 and len(interactions)==4 and len(supplemental)==16
    assert {f:sum(p['family']==f for p in pairs) for f in {p['family'] for p in pairs}}=={
        'CaLR-minus-polynomial':8,'S2-minus-S1':8,'new-minus-cosine-008':16,'new-minus-standalone':16}
    by={r['endpoint_id']:r for r in rows}
    for p in pairs:
        a,b=by[p['left_endpoint_id']],by[p['right_endpoint_id']]
        assert p['delta_loss']==a['loss']-b['loss']
        assert p['perplexity_ratio']==a['perplexity']/b['perplexity']
        assert p['relative_gap']==p['perplexity_ratio']-1
    assert all(math.isclose(r['interaction_loss'],.1) for r in interactions)


@pytest.mark.parametrize('damage',['duplicate','missing','nonfinite'])
def test_comparisons_reject_invalid_rows(rows,damage):
    if damage=='duplicate':rows.append(copy.deepcopy(rows[0]))
    elif damage=='missing':rows.pop()
    else:rows[0]['loss']=float('nan')
    with pytest.raises(ConfigError):campaign.linear_calr_comparisons(rows)


@pytest.mark.parametrize('peak',[.008,.004])
@pytest.mark.parametrize('metric',['loss','perplexity'])
def test_seven_series_endpoint_figures(rows,peak,metric):
    import matplotlib.pyplot as plt
    fig=campaign.linear_calr_endpoint_figure(rows,metric=metric,cosine_peak=peak)
    ax=fig.axes[0]
    assert ax.get_legend_handles_labels()[1]==list(campaign.LINEAR_CALR_PLOT_LABELS)
    assert len(ax.lines)==6 and len(ax.collections)==1
    expected=[w['non_embedding_parameters'] for w in campaign.WIDTHS]
    assert all(list(line.get_xdata())==expected for line in ax.lines)
    assert all(str(peak) in fig.texts[0].get_text() for _ in [0])
    plt.close(fig)


def test_measured_progress_and_missing_lr(rows,tmp_path):
    root=tmp_path/'run';root.mkdir()
    (root/'metrics.csv').write_text('run_id,step,split,granularity,loss,learning_rate\nrun,64,validation,g250,2.1,\nrun,348528,validation,g250,2.0,\nrun,1,train,g250,2.3,0.0\n')
    run=dict(run_id='run',run_dir=str(root),series='S1-cosine',peak_lr=.008,endpoints=[dict(width='g250',loss=2.)])
    progress,lr,notes=campaign.linear_calr_measured_trajectories(run)
    assert len(progress)==2 and not lr and notes['missing_applied_lr']
    # A scalar learning_rate alone is not sufficient proof of an applied rate.
    observations=[dict(series=r['series'],peak_lr=r['peak_lr'],width=r['width'],step=64,loss=3.) for r in rows if not r['series'].startswith('standalone')]
    fig=campaign.linear_calr_progress_figure(rows,observations,cosine_peak=.008)
    for ax in fig.axes:
        assert len(ax.collections)==1
        assert ax.collections[0].get_offsets()[0,0]==87132
        assert all(len(line.get_xdata())==1 for line in ax.lines)
    __import__('matplotlib.pyplot',fromlist=['close']).close(fig)


def test_complete_atomic_publication_and_manifest(rows,tmp_path,monkeypatch):
    sources=[campaign._source_record(Path(__file__).resolve())]
    terminals=[dict(series=s,peak_lr=p,run_id=f'{s}-{p}',endpoints=[r for r in rows if r['series']==s and r['peak_lr']==p],sources=sources)
        for s,p in dict.fromkeys((r['series'],r['peak_lr']) for r in rows)]
    monkeypatch.setattr(campaign,'collect_linear_calr_report',lambda **k:dict(endpoints=rows,runs=terminals,sources=sources,
        statuses=dict(new_terminals='complete',references='complete'),reasons=[]))
    def measured(run):
        progress=[dict(series=run['series'],peak_lr=run['peak_lr'],width=r['width'],step=r['actual_updates'],loss=r['loss']) for r in run['endpoints']]
        lr=[dict(series=run['series'],peak_lr=run['peak_lr'],width=r['width'],pre_update_position=0,learning_rate=0.) for r in run['endpoints']]
        return progress,lr,dict(missing_validation=False,missing_applied_lr=False)
    monkeypatch.setattr(campaign,'linear_calr_measured_trajectories',measured)
    out=tmp_path/'report'
    report=campaign.report_linear_calr(campaign_manifest='fixture',run_root='fixture',reference_root='fixture',output_dir=out)
    assert report['status']=='complete' and len(report['figures'])==16
    for stem,count in [('endpoints',36),('paired_differences',48),('interactions',4)]:
        with (out/(stem+'.csv')).open() as f:assert len(list(csv.DictReader(f)))==count
    manifest=json.loads((out/'plot_sources.json').read_text())
    assert manifest['sources']==sources
    for name,digest in manifest['output_sha256'].items():assert campaign._source_record(out/name)['sha256']==digest
    with pytest.raises(ConfigError,match='occupied'):campaign.report_linear_calr(campaign_manifest='fixture',run_root='fixture',reference_root='fixture',output_dir=out)


def test_incomplete_preserves_valid_new_evidence(rows,tmp_path,monkeypatch):
    new=[r for r in rows if 'polynomial' in r['series'] or 'carl' in r['series']]
    monkeypatch.setattr(campaign,'collect_linear_calr_report',lambda **k:dict(endpoints=new,runs=[],sources=[],
        statuses=dict(new_terminals='complete',references='incomplete'),reasons=['missing reference']))
    report=campaign.report_linear_calr(campaign_manifest='fixture',run_root='fixture',reference_root='fixture',output_dir=tmp_path/'partial')
    assert report['status']=='incomplete' and report['new_terminals']=='complete'
    assert len(json.loads((tmp_path/'partial/endpoints.json').read_text())['endpoints'])==16
    assert not (tmp_path/'partial/paired_differences.csv').exists()


def test_cli_incomplete_returns_nonzero(tmp_path,monkeypatch,capsys):
    from scripts import analyze_tinystories_optimizer_ownership as cli
    monkeypatch.setattr(cli,'report_linear_calr',lambda **k:dict(status='incomplete',reasons=['missing reference']))
    with pytest.raises(SystemExit) as exc:
        cli.main(['report-linear-calr','--campaign-manifest','fixture','--run-root','fixture','--reference-root','fixture','--output-dir',str(tmp_path/'report')])
    assert exc.value.code==1


@pytest.mark.parametrize('damage',['record','chain','identity','position','rate'])
def test_applied_lr_rejects_corrupt_committed_trace(tmp_path,damage):
    root=tmp_path/'run';root.mkdir()
    (root/'metrics.csv').write_text('run_id,step,split,granularity,loss\nrun,348528,validation,g250,2.0\n')
    record=dict(evidence_kind='applied',run_id='run',step=1,pre_update_position=0,width='g250',applied_learning_rates=[0.])
    row=dict(step=1,width='g250',applied_schedule_record=record,
        applied_schedule_watermark=dict(step=1,last_record_hash=campaign.stable_hash(record),chain_hash=campaign.stable_hash([None,record])))
    if damage=='record':record['evidence_kind']='analytic'
    elif damage=='chain':row['applied_schedule_watermark']['chain_hash']='bad'
    elif damage=='identity':record['run_id']='other'
    elif damage=='position':record['pre_update_position']=1
    else:record['applied_learning_rates']=[float('nan')]
    (root/'optimizer_ownership_trace.jsonl').write_text(json.dumps(row)+'\n')
    run=dict(run_id='run',run_dir=str(root),series='S1-carl',peak_lr=.008,endpoints=[dict(width='g250',loss=2.)])
    with pytest.raises(ConfigError):campaign.linear_calr_measured_trajectories(run)


def test_changed_sources_abort_atomic_publication(rows,tmp_path,monkeypatch):
    source=tmp_path/'source';source.write_text('original');sources=[campaign._source_record(source)]
    monkeypatch.setattr(campaign,'collect_linear_calr_report',lambda **k:dict(endpoints=rows,runs=[],sources=sources,
        statuses=dict(new_terminals='complete',references='complete'),reasons=[]))
    original=campaign.linear_calr_endpoint_figure
    def changed(*a,**kw):
        fig=original(*a,**kw);source.write_text('changed');return fig
    monkeypatch.setattr(campaign,'linear_calr_endpoint_figure',changed)
    with pytest.raises(ConfigError,match='changed'):
        campaign.report_linear_calr(campaign_manifest='fixture',run_root='fixture',reference_root='fixture',output_dir=tmp_path/'report')
    assert not (tmp_path/'report').exists()


def test_findings_cover_all_widths_and_caveats(rows):
    pairs,interactions,_=campaign.linear_calr_comparisons(rows)
    text=campaign.linear_calr_findings(rows,pairs,interactions)
    for word in (*campaign.WIDTH_LABELS,'S1','S2','interaction','0.004','standalone','One seed','AdamW','87132','348528'):
        assert word in text


@pytest.mark.parametrize('module',['test_linear_calr_reporting','tests.test_linear_calr_reporting'])
@pytest.mark.parametrize('outcome',['passed','failure','skipped','absent'])
def test_cpu_reporting_acceptance_requires_executed_suite(tmp_path,monkeypatch,outcome,module):
    from scripts import preflight_tinystories_linear_calr as diagnostics
    source=tmp_path/'source';source.mkdir();output=tmp_path/'output';output.mkdir()
    def execute(cmd,**kwargs):
        tag='' if outcome=='passed' else '<'+outcome+'/>' if outcome!='absent' else ''
        case='' if outcome=='absent' else f'<testcase classname="{module}" name="acceptance">'+tag+'</testcase>'
        failures=int(outcome=='failure');skipped=int(outcome=='skipped')
        (output/'pytest.xml').write_text(f'<testsuites><testsuite tests="1" failures="{failures}" errors="0" skipped="{skipped}">{case}</testsuite></testsuites>')
        return type('Result',(),{'returncode':failures})()
    monkeypatch.setattr(diagnostics.subprocess,'run',execute)
    result=diagnostics.pytest_check(source,output)
    assert result['reporting_fixture_status']==('passed' if outcome=='passed' else 'pending')
    assert 'tests/test_linear_calr_reporting.py' in result['command']

from test_linear_calr_campaign import calr_runs


@pytest.mark.parametrize('arm',[a['arm_id'] for a in campaign.LINEAR_CALR_ARMS])
@pytest.mark.parametrize('damage',['identity','model'])
def test_new_native_terminal_validates_actual_bundle(calr_runs,tmp_path,monkeypatch,arm,damage):
    from src.training.modeling import build_model
    from src.training import data
    monkeypatch.setattr(data,'build_packed_mmap_dataloaders',lambda *a:(None,None,None,{}))
    definition=next(r for r in calr_runs if r['arm_id']==arm)
    root=tmp_path/arm;root.mkdir()
    write(root/'config.json',definition['resolved_config'])
    model=build_model(definition['resolved_config'])
    cp=root/'latest.pt'
    payload=dict(model_state_dict=model.state_dict(),run_id='wrong')
    if damage=='model':payload['model_state_dict']={'bad':torch.ones(1)}
    torch.save(payload,cp)
    write(root/'terminal_validation_results.json',dict(checkpoint_path=str(cp)))
    # Terminal/schema checks are covered separately; this test exercises the real
    # model-shape/whole-bundle validator after terminal admission, without install.
    monkeypatch.setattr(campaign,'_inspect_terminal_run',lambda *a,**k:dict(sources=[],endpoints=[]))
    with pytest.raises(ConfigError):campaign._linear_calr_native_terminal(root,definition,{})


def test_packed_terminal_stages_sampler_without_changing_saved_signature(calr_runs,tmp_path,monkeypatch):
    from src.training import checkpointing, data
    definition=calr_runs[0]
    config=copy.deepcopy(definition['resolved_config'])
    config['dataset']['mode']='packed_mmap'
    config['comparison_control_signature']='saved-runtime-signature'
    root=tmp_path/'packed';root.mkdir()
    write(root/'config.json',config)
    write(root/'terminal_validation_results.json',dict(checkpoint_path=str(root/'latest.pt')))
    monkeypatch.setattr(campaign,'_linear_calr_saved_config',lambda *a:config)
    monkeypatch.setattr(campaign,'_inspect_terminal_run',lambda *a,**k:dict(sources=[]))
    monkeypatch.setattr(campaign,'validate_linear_calr_controls',lambda *a:None)
    monkeypatch.setattr(torch,'load',lambda *a,**k:dict(model_state_dict={},applied_schedule_watermark={}))
    loader=object()
    def prepare(staged,device):
        assert staged is not config
        staged['comparison_control_signature']='recomputed-loader-signature'
        return loader,None,None,{}
    monkeypatch.setattr(data,'build_packed_mmap_dataloaders',prepare)
    monkeypatch.setattr(checkpointing,'_validate_model_state_before_load',lambda *a:None)
    def validate(payload,saved,*args,**kwargs):
        assert saved['comparison_control_signature']=='saved-runtime-signature'
        assert kwargs['train_dataloader'] is loader
        assert kwargs['inspect_only'] is True
        return None,{}
    monkeypatch.setattr(checkpointing,'_validate_ownership_payload',validate)
    monkeypatch.setattr(checkpointing,'_validate_ownership_action_rng',lambda *a:None)
    campaign._linear_calr_native_terminal(root,definition,{})
    assert config['comparison_control_signature']=='saved-runtime-signature'
