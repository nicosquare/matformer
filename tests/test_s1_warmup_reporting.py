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
