"""Read-only selected-reference contracts; terminal payloads are synthetic."""
import copy
import json
import shutil
from pathlib import Path

import pytest
import yaml

from src.evaluation import optimizer_ownership as campaign
from src.utils.config import ConfigError
from test_optimizer_ownership_campaign import audited_inputs
import test_optimizer_ownership_reporting as legacy


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def reference_roots(tmp_path, audited_inputs, monkeypatch):
    roots = {}
    references = copy.deepcopy(campaign.WARMUP_REFERENCES)
    for grid, suffix in [('linear', 'optimizer_ownership'), ('geometric', 'matformer_widths')]:
        root = tmp_path/grid; root.mkdir()
        recipe = yaml.safe_load(Path(f'configs/controlled_exps/tinystories_instruct_{suffix}.yaml').read_text())
        manifest_path, runs = legacy.terminal_campaign.__wrapped__(root, (recipe, *audited_inputs[1:]), monkeypatch)
        (root/'preflight').rename(root/'campaign')
        manifest = json.loads((root/'campaign/campaign_manifest.json').read_text())
        references[grid]['s1_config_sha256'] = campaign._source_record(root/'campaign/configs/S1.yaml')['sha256']
        jobs = []
        for run in manifest['runs']:
            arm = run['arm_id']
            if arm not in references[grid]['selected_arms']:
                shutil.rmtree(runs/arm)
                continue
            write_json(runs/arm/'config.json', run['resolved_config'])
            job = str(1000+len(jobs)); uuid = 'process-'+arm
            record = dict(arm_id=arm, run_id=run['run_id'], job_id=job, status='completed',
                          command=['sbatch', '--gres=gpu:1', '--config', str(root/'campaign/configs'/f'{arm}.yaml')])
            measured = dict(status='completed', attempted_steps=run['assigned_updates'], elapsed_seconds=10.,
                measurement_complete=True, peak_allocated_bytes=100, peak_reserved_bytes=200,
                source_checkpoint=None, sequence=1, started_at='2026-09-22T00:00:00+00:00', failure=None)
            if grid == 'geometric':
                record['bindings'] = dict(manifest_hash=manifest['manifest_hash'],
                    config_sha256={arm: campaign._source_record(root/'campaign/configs'/f'{arm}.yaml')['sha256']})
                record['attempt_id'] = 2 if arm == 'S1' else 1
                measured.update(slurm_job_id=job, launch_attempt_id=record['attempt_id'], process_uuid=uuid, run_id=run['run_id'])
                write_json(root/'launchers'/f"worker-{arm}-{record['attempt_id']}.json",
                           dict(job_id=job, arm_id=arm, attempt_id=record['attempt_id'], status='completed', returncode=0, process_uuid=uuid))
                if arm == 'S1':
                    write_json(root/'launchers/cuda-entry-S1-2.json', dict(job_id=job, requested_device='cuda:0', required_precision='bf16', config=str(root/'campaign/configs/S1.yaml')))
            write_json(runs/arm/'resource_attempts.json', dict(schema_version=1, run_id=run['run_id'], attempts={uuid: measured}))
            from src.training.run import ResourceAttemptLedger
            summary_path = runs/arm/'run_summary.json'; summary = json.loads(summary_path.read_text())
            summary['optimizer_ownership']['resources'] = ResourceAttemptLedger(runs/arm, run_id=run['run_id']).summary()
            write_json(summary_path, summary)
            jobs.append(record)
        write_json(root/'launchers'/('training-submissions.json' if grid=='linear' else 'submissions.json'), dict(jobs=jobs))
        roots[grid] = root
    # Fixtures pin their own config bytes, never weaken the real recipe pins.
    monkeypatch.setattr(campaign, 'WARMUP_REFERENCES', references)
    monkeypatch.setattr(campaign, '_historical_job_accounting', lambda job: dict(job_id=job, state='COMPLETED', exit_code='0:0'))
    return roots


def inspect(roots):
    return campaign.inspect_warmup_references(linear_reference_root=roots['linear'], geometric_reference_root=roots['geometric'])


def test_ten_selected_terminals_sixteen_endpoints_without_cancelled_arms(reference_roots):
    before = {str(p): campaign._source_record(p)['sha256'] for root in reference_roots.values() for p in root.rglob('*') if p.is_file()}
    result = inspect(reference_roots)
    assert result['status'] == 'passed'
    assert sum(len(g['terminals']) for g in result['grids'].values()) == 10
    assert sum(len(t['endpoints']) for g in result['grids'].values() for t in g['terminals']) == 16
    assert result['grids']['linear']['terminals'][0]['run_id'] != result['grids']['geometric']['terminals'][1]['run_id']
    assert before == {str(p): campaign._source_record(p)['sha256'] for root in reference_roots.values() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('mutation', ['cpu', 'worker', 'job', 'stale_config', 'budget', 'role', 'targets', 'hash', 'missing', 'cancelled'])
def test_reject_invalid_selected_references(reference_roots, mutation):
    root = reference_roots['geometric']; run = root/'runs/S1'
    if mutation == 'cpu':
        p=run/'resource_attempts.json'; d=json.loads(p.read_text()); d['attempts']['process-S1']['peak_allocated_bytes']=None; write_json(p,d)
    elif mutation in ('worker', 'job'):
        p=root/'launchers/worker-S1-2.json'; d=json.loads(p.read_text()); d['status' if mutation=='worker' else 'job_id']='invalid'; write_json(p,d)
    elif mutation == 'stale_config':
        with (root/'campaign/configs/S1.yaml').open('a') as stream: stream.write('\n# changed\n')
    elif mutation == 'missing': shutil.rmtree(run)
    elif mutation == 'cancelled':
        p=root/'launchers/submissions.json'; d=json.loads(p.read_text()); d['jobs'][-1]['status']='cancelled'; write_json(p,d)
    else:
        changes={'budget':('actual_tokens',1), 'role':('evaluation_role','final_holdout'), 'targets':('evaluation_target_tokens',1), 'hash':('checkpoint_sha256','bad')}
        k,v=changes[mutation]; legacy.change_terminal(root/'runs','S1',lambda d:d.update({k:v}))
    with pytest.raises((ConfigError, FileNotFoundError)): inspect(reference_roots)


@pytest.fixture
def preflight_inputs(tmp_path, audited_inputs, reference_roots, monkeypatch):
    recipe = yaml.safe_load(Path('configs/controlled_exps/tinystories_instruct_s1_warmup.yaml').read_text())
    recipe['references'] = copy.deepcopy(campaign.WARMUP_REFERENCES)
    source = tmp_path/'warmup.yaml'; source.write_text(yaml.safe_dump(recipe))
    # Reference fixture paths locate the same synthetic corpus/tokenizer identity.
    # Real saved controls share exact paths. Use those same controls here.
    for grid, root in reference_roots.items():
        manifest_path = root/'campaign/campaign_manifest.json'; m=json.loads(manifest_path.read_text())
        for run in m['runs']:
            for cfg in ('resolved_config','executable_config'):
                run[cfg]['model']['tokenizer_dir'] = str(tmp_path/'tokenizer')
                run[cfg]['dataset']['prepared_corpus_dir'] = str(tmp_path/'corpus')
            (root/'campaign/configs'/f"{run['arm_id']}.yaml").write_text(yaml.safe_dump(run['executable_config'],sort_keys=False))
            p=root/'runs'/run['arm_id']/'config.json'
            if p.exists(): write_json(p,run['resolved_config'])
        m['manifest_hash']=campaign.stable_hash({k:v for k,v in m.items() if k!='manifest_hash'}); write_json(manifest_path,m)
        if grid == 'geometric':
            submissions_path=root/'launchers/submissions.json';submissions=json.loads(submissions_path.read_text())
            for job in submissions['jobs']:
                job['bindings']=dict(manifest_hash=m['manifest_hash'],config_sha256={job['arm_id']:campaign._source_record(root/'campaign/configs'/f"{job['arm_id']}.yaml")['sha256']})
            write_json(submissions_path,submissions)
        campaign.WARMUP_REFERENCES[grid]['s1_config_sha256']=campaign._source_record(root/'campaign/configs/S1.yaml')['sha256']
    recipe['references']=copy.deepcopy(campaign.WARMUP_REFERENCES);source.write_text(yaml.safe_dump(recipe))
    # Preserve the synthetic full-budget expected trace contract of legacy fixture.
    def traces(runs, *args):
        return {r['arm_id']: dict(epochs=[dict(epoch_index=i,sha256=f'epoch-{i}') for i in range(4)], actions={'sha256':'elastic-actions'}) for r in runs}
    monkeypatch.setattr(campaign,'build_expected_traces',traces)
    return dict(campaign_path=source, prepared_corpus_dir=tmp_path/'corpus', tokenizer_dir=tmp_path/'tokenizer',
        output_dir=tmp_path/'new/campaign',run_output_root=tmp_path/'new/runs',
        linear_reference_root=reference_roots['linear'],geometric_reference_root=reference_roots['geometric'])


def test_preflight_atomic_publication_and_identical_verification(preflight_inputs, capsys):
    from scripts.analyze_tinystories_optimizer_ownership import main
    args=preflight_inputs
    argv=['preflight']
    for key,value in args.items():
        argv.extend(['--'+('campaign' if key=='campaign_path' else key.replace('_','-')),str(value)])
    main(argv)
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='passed' and result['run_count']==2
    output=args['output_dir']; before={str(p):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    manifest=campaign._read_preflight_manifest(output/'campaign_manifest.json')
    assert set(manifest['arm_grids'])=={'S1-linear-w256','S1-geometric-w256'}
    assert 'width_grid' not in manifest
    assert len(manifest['control_audits'])==2
    assert len(list((output/'schedules').glob('*.csv')))==2
    assert (output/'references/selection.json').exists()
    assert campaign.preflight_campaign(**args)==result
    assert before=={str(p):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    (args['run_output_root']/'S1-linear-w256').mkdir(parents=True)
    with pytest.raises(ConfigError,match='occupied'): campaign.preflight_campaign(**args)


@pytest.mark.parametrize('failure', ['reference','trace','publication','missing_flag','unrelated_root'])
def test_preflight_failure_publishes_nothing(preflight_inputs, monkeypatch, failure):
    args=preflight_inputs
    if failure=='reference':
        legacy.change_terminal(args['geometric_reference_root']/'runs','S1',lambda d:d.update(actual_tokens=1))
    elif failure=='trace':
        monkeypatch.setattr(campaign,'build_expected_traces',lambda runs,*a:{r['arm_id']:dict(epochs=[],actions={}) for r in runs})
    elif failure=='publication':
        original=Path.rename
        def fail(self,target):
            if Path(target)==args['output_dir']: raise OSError('injected publication failure')
            return original(self,target)
        monkeypatch.setattr(Path,'rename',fail)
    elif failure=='missing_flag': args['linear_reference_root']=None
    else:
        args['output_dir'].parent.mkdir();(args['output_dir'].parent/'unrelated').write_text('occupied')
    with pytest.raises((ConfigError,OSError)): campaign.preflight_campaign(**args)
    assert not args['output_dir'].exists()
    assert not campaign._reservation_path(args['run_output_root']).exists()


def test_reference_changed_during_read_is_rejected(reference_roots, monkeypatch):
    original=campaign.inspect_historical_device
    changed=False
    def inspect_and_change(root, run, manifest):
        nonlocal changed
        result=original(root,run,manifest)
        if not changed:
            path=Path(root)/'runs'/run['arm_id']/'run_summary.json'
            path.write_text(path.read_text()+'\n')
            changed=True
        return result
    monkeypatch.setattr(campaign,'inspect_historical_device',inspect_and_change)
    with pytest.raises(ConfigError,match='changed'): inspect(reference_roots)


def test_cli_requires_both_reference_flags(tmp_path, capsys):
    from scripts.analyze_tinystories_optimizer_ownership import main
    with pytest.raises(SystemExit) as error:
        main(['preflight','--campaign','configs/controlled_exps/tinystories_instruct_s1_warmup.yaml',
              '--prepared-corpus-dir',str(tmp_path/'corpus'),'--tokenizer-dir',str(tmp_path/'tokenizer'),
              '--output-dir',str(tmp_path/'campaign'),'--run-output-root',str(tmp_path/'runs')])
    assert error.value.code==1
    assert '--linear-reference-root and --geometric-reference-root' in capsys.readouterr().err
    assert not (tmp_path/'campaign').exists()


@pytest.fixture
def report_inputs(preflight_inputs, monkeypatch):
    """Synthetic terminal payloads; full trace IO and GPU execution are separate tests."""
    import torch
    args = preflight_inputs
    campaign.preflight_campaign(**args)
    manifest_path = args['output_dir']/'campaign_manifest.json'
    manifest = campaign._read_preflight_manifest(manifest_path)
    for run in manifest['runs']:
        grid = run['grid_id']; old_root = args[grid+'_reference_root']/'runs/S1'
        root = Path(run['output_path']); shutil.copytree(old_root, root)
        old = campaign._read_json(old_root/'terminal_validation_results.json')
        sidecar = copy.deepcopy(old)
        for key in ('campaign_id', 'arm_id', 'run_id', 'contract_hash', 'representation', 'state_scope', 'clipping', 'initialization'):
            sidecar[key] = run[key]
        sidecar['contract'] = run['optimizer_ownership_contract']
        checkpoint = root/'latest.pt'
        payload = torch.load(checkpoint, weights_only=False)
        payload.update(run_id=run['run_id'], optimizer_ownership_contract=sidecar['contract'], optimizer_ownership_contract_hash=run['contract_hash'])
        torch.save(payload, checkpoint)
        sidecar.update(checkpoint_path=str(checkpoint), checkpoint_sha256=campaign._source_record(checkpoint)['sha256'], checkpoint_bytes=checkpoint.stat().st_size)
        import math
        for i, row in enumerate(sidecar['endpoints']):
            row.update(loss=2 + (i-1)*.1, perplexity=math.exp(2+(i-1)*.1))
        sidecar['content_hash'] = campaign.stable_hash({k:v for k,v in sidecar.items() if k!='content_hash'})
        write_json(root/'terminal_validation_results.json', sidecar)
        summary = campaign._read_json(root/'run_summary.json'); summary['run_id']=run['run_id']
        audit = summary['optimizer_ownership']
        audit.update({k:run[k] for k in ('run_id','campaign_id','arm_id','contract_hash')}); audit['contract']=sidecar['contract']
        audit['checkpoint'].update({f'terminal_checkpoint_{k}':sidecar[f'checkpoint_{k}'] for k in ('path','sha256','bytes')})
        (root/'resource_attempts.json').unlink()
        audit['resources'].update(measurement_complete=False)
        write_json(root/'run_summary.json', summary)
        write_json(root/'config.json',run['resolved_config'])
    # Real execution identity is independently exercised by launcher tests.
    monkeypatch.setattr(campaign, 'inspect_warmup_execution', lambda root, run, manifest: dict(status='passed', sources=[]))
    for grid in ('linear','geometric'):
        for root, arm, warmup in [(args[grid+'_reference_root']/'runs/S1','S1',64),
                (args['run_output_root']/f'S1-{grid}-w256',f'S1-{grid}-w256',256)]:
            import csv
            schema = 1 if grid=='linear' else 4
            widths = campaign.campaign_widths(5 if warmup==256 else schema, arm)
            run_id = campaign._read_json(root/'run_summary.json')['run_id']
            with (root/'metrics.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=['step','split','granularity','loss','learning_rate','run_id','optimizer_step_committed','optimizer_action_id'])
                writer.writeheader()
                for step in range(1,1025):
                    writer.writerow(dict(step=step,split='train',granularity=widths[(step-1)%4]['label'],loss=2.5-step/2048,
                        learning_rate=.008*(step-1)/warmup if step<=warmup else .008,run_id=run_id,optimizer_step_committed=True,optimizer_action_id=f'action-{step}'))
                    if step%64==0:
                        for w in widths: writer.writerow(dict(step=step,split='validation',granularity=w['label'],loss=2.4-step/2048,run_id=run_id))
            with (root/'optimizer_ownership_trace.jsonl').open('w') as f:
                for step in range(1,1025): f.write(json.dumps(dict(step=step,attempt_id='process',action_id=f'action-{step}',run_id=run_id))+'\n')
    return dict(campaign_manifest=manifest_path,run_root=args['run_output_root'],output_dir=args['output_dir'].parent/'reports/new'), dict(
        manifest=args['output_dir'].parent/'reports/new/frozen_manifest.json',linear_reference_root=args['linear_reference_root'],
        geometric_reference_root=args['geometric_reference_root'],output_dir=args['output_dir'].parent/'reports/comparison')


def assert_table_parity(root, stem, key):
    import csv
    rows=json.loads((root/f'{stem}.json').read_text())[key]
    with (root/f'{stem}.csv').open() as f: exported=list(csv.DictReader(f))
    assert exported==[{k:campaign.endpoint_csv_value(v) for k,v in r.items()} for r in rows]
    return rows


def test_warmup_complete_report_cardinality_values_figures(report_inputs):
    freeze, args=report_inputs
    campaign.freeze_campaign(**freeze)
    assert len(assert_table_parity(freeze['output_dir'],'endpoints','endpoints'))==8
    report=campaign.report_s1_warmup(**args)
    assert report['status']==report['endpoint_status']==report['early_status']=='complete'
    rows=assert_table_parity(args['output_dir'],'endpoints','endpoints')
    deltas=assert_table_parity(args['output_dir'],'paired_deltas','paired_deltas')
    early=assert_table_parity(args['output_dir'],'early_metrics','observations')
    assert len(rows)==24 and len(deltas)==8
    assert len({tuple(r['endpoint_identity']) for r in rows})==24
    for r in rows:
        assert all(k in r for k in ('evaluation_path','evaluation_sha256','config_sha256','source_records','execution_evidence','expected_exposure','resources','global_step','matching_original_s1','matching_standalone'))
        assert r['evaluation_target_tokens']==36195
    by_id={tuple(r['endpoint_identity']):r for r in rows}
    for d in deltas:
        old,new,st=(by_id[tuple(d[k])] for k in ('original_endpoint','new_endpoint','standalone_endpoint'))
        for metric in ('loss','perplexity'):
            assert d['delta_'+metric]==new[metric]-old[metric]
            assert d['old_standalone_gap_'+metric]==old[metric]-st[metric]
            assert d['new_standalone_gap_'+metric]==new[metric]-st[metric]
    assert len(report['figures'])==16
    assert all(Path(p).stat().st_size>100 for p in report['figures'])
    assert not any(r['step']==0 and r['metric']=='loss' for r in early)
    assert all(r['source_kind']=='recorded' for r in early)
    for grid in ('linear','geometric'):
        subset=[r for r in rows if r['grid_id']==grid]
        figure=campaign.warmup_endpoint_figure(subset, metric='loss')
        lines=figure.axes[0].lines
        assert len(lines)==3 and lines[0].get_linestyle()=='None'
        expected=[w['non_embedding_parameters'] for w in campaign.campaign_widths(5,f'S1-{grid}-w256')]
        assert all(list(line.get_xdata())==expected for line in lines)
        figure=campaign.warmup_early_figure(early, grid=grid, metric='loss', end_step=1024)
        assert len(figure.axes)==2
        assert all(tuple(ax.get_xlim())==(0,1024) for ax in figure.axes)
    assert len(report['output_sha256'])>=23


@pytest.mark.parametrize('damage',['duplicate','nonfinite','nonterminal','budget','role','targets','exp','device'])
def test_new_freeze_rejects_invalid_evidence(report_inputs,monkeypatch,damage):
    freeze,_=report_inputs
    arm='S1-linear-w256'
    def mutate(d):
        if damage=='duplicate':d['endpoints'].append(d['endpoints'][0])
        elif damage=='nonfinite':d['endpoints'][0]['loss']=float('nan')
        elif damage=='exp':d['endpoints'][0]['perplexity']=1
        else:d.update({{'nonterminal':'global_step','budget':'actual_tokens','role':'evaluation_role','targets':'evaluation_target_tokens'}[damage]:1})
    if damage=='device':
        def reject(*a): raise ConfigError('invalid device')
        monkeypatch.setattr(campaign,'inspect_warmup_execution',reject)
    else:legacy.change_terminal(freeze['run_root'],arm,mutate)
    with pytest.raises(ConfigError):campaign.freeze_campaign(**freeze)
    assert not freeze['output_dir'].exists()


@pytest.mark.parametrize('damage',['missing_train','missing_validation','duplicate','nonfinite','missing_reference','stale','changed'])
def test_incomplete_comparison_preserves_new_endpoints(report_inputs,monkeypatch,damage):
    freeze,args=report_inputs
    root=args['linear_reference_root']/'runs/S1'; path=root/'metrics.csv'
    text=path.read_text()
    if damage=='missing_train':path.write_text('\n'.join(x for x in text.splitlines() if ',train,' not in x)+'\n')
    elif damage=='missing_validation':path.write_text('\n'.join(x for x in text.splitlines() if ',validation,' not in x)+'\n')
    elif damage=='duplicate':path.write_text(text+text.splitlines()[1]+'\n')
    elif damage=='nonfinite':path.write_text(text.replace('2.49951171875','nan',1))
    campaign.freeze_campaign(**freeze)
    before={p.name:p.read_bytes() for p in freeze['output_dir'].iterdir()}
    if damage=='missing_reference':shutil.rmtree(args['linear_reference_root'])
    elif damage=='stale':(freeze['run_root']/'S1-linear-w256/metrics.csv').write_text('changed')
    elif damage=='changed':
        original=campaign.warmup_endpoint_figure
        def changed(*a,**k):
            figure=original(*a,**k);path.write_text(path.read_text()+'\n');return figure
        monkeypatch.setattr(campaign,'warmup_endpoint_figure',changed)
    with pytest.raises((ConfigError,OSError)):campaign.report_s1_warmup(**args)
    assert before=={p.name:p.read_bytes() for p in freeze['output_dir'].iterdir()}
    diagnostic=json.loads((args['output_dir']/'comparison_report.json').read_text())
    assert diagnostic['status']=='incomplete' and diagnostic['reasons']
    assert not (args['output_dir']/'endpoints.json').exists()


def test_early_reconstruction_gaps_and_replay(report_inputs):
    freeze,args=report_inputs
    import csv
    root=freeze['run_root']/'S1-linear-w256'; path=root/'metrics.csv'
    with path.open() as f: reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
    rows=[r for r in rows if not (r['split']=='train' and r['step']=='2')]
    for r in rows:r['learning_rate']=''
    replay=dict(rows[0],loss='99',optimizer_step_committed='False')
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows([replay,*rows])
    manifest=campaign._read_preflight_manifest(freeze['campaign_manifest']);run=manifest['runs'][0]
    observations,notes=campaign.extract_warmup_early(run,root,end_step=1024)
    assert any(r['source_kind']=='reconstructed_schedule' for r in observations)
    assert not any(r['metric']=='loss' and r['value']==99 for r in observations)
    assert notes['training_missing_steps']==[2]
    assert not any(r['metric']=='loss' and r['step'] in (0,2) and r['split']=='train' for r in observations)
    assert next(r for r in observations if r['metric']=='learning_rate' and r['step']==257)['value']==.008
    with pytest.raises(ConfigError,match='1024'):campaign.extract_warmup_early(run,root,end_step=1000)


def test_early_duplicate_resolved_only_with_committed_process(report_inputs):
    import csv
    freeze,_=report_inputs
    root=freeze['run_root']/'S1-linear-w256';path=root/'metrics.csv'
    with path.open() as f:reader=csv.DictReader(f);fields=[*reader.fieldnames,'attempt_id'];rows=list(reader)
    for row in rows:row['attempt_id']='process'
    replay=dict(rows[0],loss='99',attempt_id='abandoned-process')
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows([replay,*rows])
    run=campaign._read_preflight_manifest(freeze['campaign_manifest'])['runs'][0]
    observations,_=campaign.extract_warmup_early(run,root)
    assert not any(r['value']==99 and r['metric']=='loss' for r in observations)
    assert {r['attempt_id'] for r in observations}=={'process'}
