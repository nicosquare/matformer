#!/usr/bin/env python3
"""Prepare and reconcile exactly four linear polynomial/CaLR runs from a tested snapshot."""
from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import run_tinystories_matformer_widths as shared
from src.evaluation import optimizer_ownership as campaign
from src.utils.config import ConfigError
from src.utils.reproducibility import stable_hash

# Reuse operational primitives, never the nine-arm admission/barrier policy.
read, save, digest = shared.read, shared.save, shared.digest
sealed, check_seal, lock, source_files = shared.sealed, shared.check_seal, shared.lock, shared.source_files
command, queue_state, scheduler_limits = shared.command, shared.queue_state, shared.scheduler_limits
PYTHON, QOS, EXCLUDED_NODES = shared.PYTHON, shared.QOS, shared.EXCLUDED_NODES
ACTIVE, TERMINAL = shared.ACTIVE, shared.TERMINAL
ARMS = tuple(a['arm_id'] for a in campaign.campaign_arms(6))
GPU_COVERAGE = ('all_widths', 'applied_lr', 'warmup_resume', 'epoch_resume', 'terminal_recovery', 'invalid_restore', 'partial_failure')
REAL_SHAPE = dict(d_model=64, num_layers=4, num_attention_heads=4, batch_size=64, context_length=128)


def bindings(root):
    """Hash current inputs, including audits and the actual executable snapshot."""
    import yaml
    root = Path(root).resolve()
    manifest_path = root/'campaign/campaign_manifest.json'
    manifest = campaign._read_preflight_manifest(manifest_path)
    if manifest['schema_version'] != 6 or manifest['campaign_id'] != campaign.LINEAR_CALR_CAMPAIGN_ID:
        raise ConfigError('Expected schema-6 linear CaLR campaign')
    if [r['arm_id'] for r in manifest['runs']] != list(ARMS) or Path(manifest['run_output_root']) != root/'runs':
        raise ConfigError('Campaign arms/run root differ from prepared identity')
    if any(r['assigned_updates'] != 348528 or r['assigned_tokens'] != 2855141376 for r in manifest['runs']):
        raise ConfigError('Linear CaLR assigned budgets differ from exact four-arm protocol')
    preflight = read(root/'campaign/preflight.json')
    if preflight.get('status') != 'passed' or preflight.get('manifest_hash') != manifest['manifest_hash']:
        raise ConfigError('Missing or stale preflight')
    campaign._check_sources([manifest['recipe_source'], *manifest['reference_selection']['sources']])
    if manifest['reference_selection'].get('status') != 'config_inspected':
        raise ConfigError('Reference selection did not pass')
    artifacts = manifest['artifact_sha256']
    for relative, expected in artifacts.items():
        if digest(root/'campaign'/relative) != expected:
            raise ConfigError(f'Prepared audit/config/trace changed: {relative}')
    configs, contracts = {}, {}
    for run in manifest['runs']:
        arm = run['arm_id']; path = root/'campaign/configs'/f'{arm}.yaml'
        if yaml.safe_load(path.read_text()) != run['executable_config']:
            raise ConfigError(f'Executable config differs from preflight: {arm}')
        if manifest['control_audits'][arm].get('status') != 'passed':
            raise ConfigError(f'Counterpart controls did not pass: {arm}')
        configs[arm] = digest(path); contracts[arm] = run['contract_hash']
    snapshot = read(root/'diagnostics/source-manifest.json'); check_seal(snapshot, 'source snapshot')
    actual = source_files(root/'source')
    if not actual or actual != snapshot['files'] or stable_hash(actual) != snapshot['source_sha256']:
        raise ConfigError('Tested source snapshot changed')
    for relative in ('scripts/run_tinystories_linear_calr.py', 'scripts/preflight_tinystories_linear_calr.py', 'scripts/train_cuda_required.py'):
        if relative not in actual: raise ConfigError(f'Snapshot missing executed entry point: {relative}')
    for run in manifest['runs']:
        for relative, expected in run['initialization']['source_files_sha256'].items():
            if actual.get(relative) != expected: raise ConfigError(f'Snapshot differs from preflight: {relative}')
    inputs=read(root/'diagnostics/inputs.json');check_seal(inputs,'snapshot inputs')
    return dict(snapshot_inputs_sha256=digest(root/'diagnostics/inputs.json'),campaign_id=manifest['campaign_id'], manifest_hash=manifest['manifest_hash'],
        campaign_manifest_sha256=digest(manifest_path), preflight_sha256=digest(root/'campaign/preflight.json'),
        recipe_sha256=manifest['recipe_source']['sha256'], source_sha256=stable_hash(actual),
        config_sha256=configs, config_set_sha256=stable_hash(configs), contracts=contracts,
        artifacts_sha256=stable_hash(artifacts), references_sha256=stable_hash(manifest['reference_selection']))


def verify_gate(root, mode, expected=None, path=None):
    expected = bindings(root) if expected is None else expected
    path = Path(path) if path else Path(root)/'diagnostics'/f'{mode}-gate.json'
    gate = read(path); check_seal(gate, f'{mode} gate')
    if gate.get('mode') != mode or gate.get('status') != 'passed' or gate.get('bindings') != expected:
        raise ConfigError(f'{mode} gate absent, failed or stale: {path}')
    checks = gate.get('checks')
    if not checks or any(c.get('returncode') != 0 or c.get('tests', 0) <= 0 or c.get('failures', 0) or c.get('errors', 0) for c in checks):
        raise ConfigError(f'{mode} gate lacks passed executed checks')
    for check in checks:
        if not check.get('command') or not check.get('artifacts'):
            raise ConfigError(f'{mode} gate requires command/log artifact evidence')
        campaign._check_sources(check.get('artifacts', []))
    if mode == 'gpu':
        cpu = verify_gate(root, 'cpu', expected)
        if (not gate.get('job_id') or not gate.get('hardware') or gate.get('precision') != 'bf16'
                or gate.get('cpu_gate_hash') != cpu['content_hash'] or gate.get('real_shape_arms') != list(ARMS)
                or gate.get('real_shape') != REAL_SHAPE or gate.get('coverage') != list(GPU_COVERAGE)
                or any(c.get('skipped', 0) for c in checks)):
            raise ConfigError('GPU gate requires both real-shape bf16 grids, every boundary check, no skips, matching CPU evidence')
        intents = read(Path(root)/'diagnostics/submissions.json')['jobs']
        matches = [j for j in intents if j.get('intent_uuid') == gate.get('intent_uuid')]
        if len(matches) != 1:
            raise ConfigError('GPU gate lacks a unique matching diagnostic intent')
        intent = matches[0]
        accounting = scheduler_history(root, intent)
        worker = read(Path(root)/'diagnostics'/f"worker-{intent['intent_uuid']}.json")
        if (not accounting or accounting.get('job_id') != gate['job_id']
                or accounting.get('state') != 'COMPLETED' or accounting.get('exit_code') != '0:0'
                or worker.get('job_id') != gate['job_id'] or worker.get('returncode') != 0
                or worker.get('gate_hash') != gate['content_hash']):
            raise ConfigError('GPU readiness awaits matching successful worker and Slurm accounting')
    return gate


def verify_plan(root, *, gpu=True):
    root = Path(root).resolve()
    plan = read(root/'launchers/plan.json'); check_seal(plan, 'launch plan')
    expected = bindings(root)
    if plan.get('bindings') != expected or plan.get('campaign_root') != str(root) or plan.get('arms') != list(ARMS):
        raise ConfigError('Launch plan source/config/campaign identity changed')
    cpu = verify_gate(root, 'cpu', expected)
    if cpu['content_hash'] != plan['cpu_gate_hash']: raise ConfigError('Prepared CPU evidence changed')
    if gpu:
        verify_authorization(root, 'production', expected)
        verify_gate(root, 'gpu', expected)
        if cpu.get('reporting_fixture_status') != 'passed':
            raise ConfigError('Production requires phase-5 reporting fixture acceptance')
    return plan


def prepare(root, cpu_evidence):
    root = Path(root).resolve()
    with lock(root/'launchers/prepare.lock'):
        expected = bindings(root); cpu = verify_gate(root, 'cpu', expected, cpu_evidence)
        manifest = read(root/'campaign/campaign_manifest.json')
        required = dict(campaign_id=manifest['campaign_id'], manifest_path=str(root/'campaign/campaign_manifest.json'),
            manifest_hash=manifest['manifest_hash'], run_ids=[r['run_id'] for r in manifest['runs']])
        if read(campaign._reservation_path(root/'runs')) != required:
            raise ConfigError('Preflight reservation does not match; cannot adopt')
        path = root/'launchers/plan.json'
        if path.exists(): return verify_plan(root, gpu=False)
        if (root/'runs').exists() and any((root/'runs').iterdir()): raise ConfigError('Preparation cannot adopt occupied runs')
        for name in ('diagnostics', 'launchers', 'logs', 'runs', 'reports'): (root/name).mkdir(exist_ok=True)
        canonical = root/'diagnostics/cpu-gate.json'
        if Path(cpu_evidence).resolve() != canonical:
            if canonical.exists() and read(canonical) != cpu: raise ConfigError('Existing CPU evidence differs')
            save(canonical, cpu)
        plan = sealed(dict(schema_version=1, campaign_id=manifest['campaign_id'], campaign_root=str(root),
            arms=list(ARMS), bindings=expected, cpu_gate_hash=cpu['content_hash'],
            references=manifest['references'],
            reference_root=read(root/'diagnostics/inputs.json')['reference_root'],
            assigned_updates={r['arm_id']: r['assigned_updates'] for r in manifest['runs']},
            assigned_tokens=dict.fromkeys(ARMS,2855141376),
            campaign_totals=dict(updates=1394112,tokens=11420565504)))
        save(path, plan)
        save(root/'launchers/status.json', dict(cpu_gate='passed', gpu_readiness='pending', production='pending',
            terminals=dict.fromkeys(ARMS,'pending'), new_report='pending', endpoint_comparison='pending',comparison='pending',
            readiness=dict(cpu='passed',gpu='pending'),submission='pending',execution='pending'))
        return plan


def continuation(root, arm):
    if arm not in ARMS: raise ConfigError(f'Unknown linear CaLR arm: {arm}')
    root = Path(root)
    runtime = root/'runs'/arm/'config.json'
    summary = root/'runs'/arm/'run_summary.json'
    if runtime.is_file() and summary.is_file() and read(summary).get('status') == 'completed':
        manifest = campaign._read_preflight_manifest(root/'campaign/campaign_manifest.json')
        definition = next(run for run in manifest['runs'] if run['arm_id'] == arm)
        terminal = campaign._linear_calr_native_terminal(root/'runs'/arm, definition, manifest['expected_traces'][arm])
        return dict(mode='completion_only', step=definition['assigned_updates'],
            checkpoint=dict(path=str(root/'runs'/arm/'checkpoints/latest.pt'),
                step=definition['assigned_updates'], sha256=terminal['checkpoint_sha256']))
    return shared.continuation(root, arm)


def scheduler_history(root, intent):
    """Only sacct establishes job outcome. A worker record cannot replace it."""
    args = ['sacct', '-X', '--noheader', '--parsable2', '--user='+getpass.getuser(),
        '--starttime='+intent['created_date'], '--name='+intent['name'],
        '--format=JobIDRaw,JobName%100,State,ElapsedRaw,ExitCode']
    if intent.get('job_id'):
        # Once submission is durable, query its exact ID. Name/time filtering
        # can omit an otherwise available completed accounting record.
        args = ['sacct', '-X', '--noheader', '--parsable2', '--jobs='+str(intent['job_id']),
            '--format=JobIDRaw,JobName%100,State,ElapsedRaw,ExitCode']
    try: raw = command(args)
    except (OSError, subprocess.SubprocessError): return None
    rows = [dict(zip(('job_id','name','state','allocation_seconds','exit_code'), line.split('|'))) for line in raw.splitlines()]
    rows = [r for r in rows if r.get('name') == intent['name'] and r.get('job_id','').isdigit()]
    if len({r['job_id'] for r in rows}) > 1: raise ConfigError('Ambiguous scheduler submission identity')
    if not rows: return None
    row = rows[0]
    if intent.get('job_id') and row['job_id'] != intent['job_id']: raise ConfigError('Scheduler job identity changed')
    return row


def capacity(active, intents, limits):
    """Reserve a running slot for every pending/uncertain job, across all QoS."""
    jobs = {str(r['job_id']) for r in active}
    jobs.update(str(j.get('job_id') or j['name']) for j in intents if j['status'] not in ('completed','retryable','failed'))
    return max(0, min(2, limits['max_running'], 4, limits['max_submitted']) - len(jobs))


def diagnostic_reservations(root):
    path = Path(root)/'diagnostics/submissions.json'
    if not path.exists(): return []
    record = read(path)
    for intent in record['jobs']:
        if intent['status'] in ('completed','failed'): continue
        info = scheduler_history(root,intent)
        if info is not None and info['state'].split()[0].rstrip('+') in TERMINAL:
            intent['scheduler_accounting'] = info
            intent['status'] = 'completed' if info['state']=='COMPLETED' and info.get('exit_code')=='0:0' else 'failed'
    save(path,record)
    return record['jobs']


def sbatch_command(root, name, entry, *, log_label):
    root = Path(root)
    # Slurm replaces SLURM_CONF in the allocation environment. Reassert the
    # submitting client's accounting endpoint before worker-side gate checks.
    client_config = os.environ.get('SLURM_CONF')
    if client_config:
        entry = ['env', 'SLURM_CONF='+client_config, *entry]
    return ['sbatch', '--parsable', '--job-name='+name, '--partition=cscc-gpu-p', '--qos='+QOS,
        '--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--gres=gpu:1', '--mem=16G', '--time=24:00:00',
        '--exclude='+EXCLUDED_NODES, '--no-requeue', '--chdir='+str(root/'source'),
        '--output='+str(root/'logs'/f'{log_label}-%j.out'), '--error='+str(root/'logs'/f'{log_label}-%j.err'),
        '--wrap='+shlex.join(entry)]


def execution_evidence(root, intent, plan):
    """Check actual trainer/resource evidence against this successful job/worker."""
    from src.training.run import ResourceAttemptLedger
    root = Path(root); arm = intent['arm_id']; attempt = intent['attempt_id']
    worker_path = root/'launchers'/f'worker-{arm}-{attempt}.json'
    entry_path = root/'launchers'/f'cuda-entry-{arm}-{attempt}.json'
    worker, entry = read(worker_path), read(entry_path)
    account = intent['scheduler_accounting']
    if account['state'].split()[0].rstrip('+') != 'COMPLETED' or account.get('exit_code') != '0:0':
        raise ConfigError('Terminal requires successful matching Slurm accounting')
    required = dict(arm_id=arm, attempt_id=attempt, job_id=intent['job_id'], name=intent['name'],
        intent_uuid=intent['intent_uuid'], bindings=plan['bindings'], cpu_gate_hash=plan['cpu_gate_hash'], gpu_gate_hash=intent['gpu_gate_hash'])
    if any(worker.get(k) != v for k,v in required.items()) or worker.get('returncode') != 0 or worker.get('status') != 'completed':
        raise ConfigError('Terminal requires successful matching worker/source/config evidence')
    if (entry.get('job_id') != intent['job_id'] or entry.get('requested_device') != 'cuda:0'
            or entry.get('required_precision') != 'bf16' or entry.get('status') != 'starting'
            or Path(entry.get('config','')) != root/'campaign/configs'/f'{arm}.yaml'
            or entry.get('nvidia_smi',{}).get('returncode') != 0):
        raise ConfigError('Missing matching CUDA-required entry evidence')
    output = root/'runs'/arm; config = read(output/'config.json')
    if config['training'].get('resolved_mixed_precision') != 'bf16': raise ConfigError('Actual execution was not bf16')
    ledger = ResourceAttemptLedger(output, run_id=config['run']['run_id'])
    record = ledger.attempts.get(worker.get('process_uuid'))
    if (not record or record.get('slurm_job_id') != intent['job_id'] or record.get('launch_attempt_id') != attempt
            or record.get('status') != 'completed' or not record.get('peak_allocated_bytes')
            or not record.get('peak_reserved_bytes')):
        raise ConfigError('Missing actual CUDA allocator/process/precision evidence')
    terminal = campaign.inspect_selected_terminals(root/'campaign/campaign_manifest.json', [arm], run_root=root/'runs')[0]
    own = continuation(root, arm)
    if own['mode'] != 'completion_only' or own['checkpoint']['sha256'] != terminal['checkpoint_sha256']:
        raise ConfigError('Terminal is not the validated complete own bundle')
    return dict(terminal=terminal, resources=ledger.summary(), sources=[campaign._source_record(p) for p in (worker_path,entry_path,output/'config.json',ledger.path)])


def reconcile(root, record, active, plan):
    for intent in record['jobs']:
        output = Path(root)/'runs'/intent['arm_id']
        outputs_present = all((output/name).is_file() for name in ('terminal_validation_results.json', 'run_summary.json'))
        if intent['status'] == 'completed' and not outputs_present:
            own = continuation(root,intent['arm_id'])
            if own['mode'] != 'completion_only':
                raise ConfigError('Lost terminal outputs lack a validated full-horizon checkpoint')
            intent.update(status='retryable',continuation=own)
        if intent['status'] == 'completed':
            intent['execution_evidence']=execution_evidence(root,intent,plan)
            continue
        if intent['status'] == 'retryable': continue
        matches = [r for r in active if r['name'] == intent['name']]
        if len(matches)>1: raise ConfigError('Duplicate scheduler submission name')
        info = matches[0] if matches else scheduler_history(root,intent)
        if info is None:
            intent['status']='uncertain'; continue
        if intent.get('job_id') and intent['job_id'] != info['job_id']: raise ConfigError('Submission job identity changed')
        intent['job_id']=info['job_id']; state=info['state'].split()[0].rstrip('+')
        if state in ACTIVE:
            intent['status']='submitted'; continue
        if state not in TERMINAL:
            intent['status']='uncertain'; continue
        intent['scheduler_accounting']=info
        worker = Path(root)/'launchers'/f"worker-{intent['arm_id']}-{intent['attempt_id']}.json"
        if state=='COMPLETED' and info.get('exit_code')=='0:0' and outputs_present:
            if not worker.exists():
                intent['status']='pending_worker'; continue
            intent['execution_evidence']=execution_evidence(root,intent,plan); intent['status']='completed'
        else:
            own=continuation(root,intent['arm_id'])
            if own['mode']=='fresh':
                entry = Path(root)/'launchers'/f"cuda-entry-{intent['arm_id']}-{intent['attempt_id']}.json"
                if state != 'FAILED' or worker.exists() or entry.exists() or output.exists():
                    raise ConfigError('Ended attempt without durable own checkpoint; reconcile before retry')
                # The worker writes its identity before creating run state or
                # invoking training. This failed admission never started a run.
                intent['failure_stage'] = 'before_worker_admission'
            intent.update(status='retryable',continuation=own)


def _queue(root, once=False):
    root=Path(root).resolve()
    with lock(root/'launchers/queue.lock'):
        while True:
            plan=verify_plan(root)
            path=root/'launchers/submissions.json'
            record=read(path) if path.exists() else dict(campaign_id=plan['campaign_id'],jobs=[])
            if record['campaign_id']!=plan['campaign_id'] or any(j['arm_id'] not in ARMS for j in record['jobs']):
                raise ConfigError('Submission campaign/arm identity changed')
            active=queue_state()
            reconcile(root,record,active,plan); save(path,record)
            completed={j['arm_id'] for j in record['jobs'] if j['status']=='completed'}
            # Ambiguous submission is a reconciliation state, never a retry grant.
            uncertain=any(j['status'] in ('submitting','uncertain','pending_worker') for j in record['jobs'])
            for arm in (() if uncertain else ARMS):
                own_intents=[j for j in record['jobs'] if j['arm_id']==arm]
                if arm in completed or any(j['status'] not in ('completed','retryable') for j in own_intents): continue
                # Shared admission lock also serializes diagnostic submission.
                with lock(root/'launchers/admission.lock'):
                    active=queue_state(); limits=scheduler_limits()
                    if not capacity(active,record['jobs']+diagnostic_reservations(root),limits): break
                    verify_plan(root); own=continuation(root,arm)
                    attempt=max((j['attempt_id'] for j in own_intents),default=0)+1
                    identity=uuid.uuid4().hex; name=f'calr-{arm}-a{attempt}-{identity[:8]}'
                    intent=dict(arm_id=arm,attempt_id=attempt,intent_uuid=identity,name=name,status='submitting',
                        created_date=time.strftime('%Y-%m-%d',time.gmtime()),intent_time=time.time(),continuation=own,
                        bindings=plan['bindings'],cpu_gate_hash=plan['cpu_gate_hash'],gpu_gate_hash=read(root/'diagnostics/gpu-gate.json')['content_hash'])
                    entry=[PYTHON,str(root/'source/scripts/run_tinystories_linear_calr.py'),'worker',
                        '--campaign-root',str(root),'--arm',arm,'--attempt-id',str(attempt)]
                    cmd=sbatch_command(root,name,entry,log_label=f'{arm}-a{attempt}');intent['command']=cmd
                    record['jobs'].append(intent);save(path,record)
                    result=command(cmd).split(';')[0]
                    if not result.isdigit(): raise ConfigError('Uncertain sbatch response; durable intent retained')
                    intent.update(job_id=result,status='submitted');save(path,record)
            allocation=[j.get('scheduler_accounting',{}).get('allocation_seconds') for j in record['jobs']]
            status=dict(cpu_gate='passed',gpu_readiness='passed',production='complete' if len(completed)==len(ARMS) else 'pending',
                terminals={a:'complete' if a in completed else 'pending' for a in ARMS},completed=sorted(completed),
                new_report='pending',endpoint_comparison='pending',jobs=record['jobs'],
                validated_updates=len(completed)*348528,validated_training_tokens=len(completed)*2855141376,
                comparison='pending',readiness=dict(cpu='passed',gpu='passed'),
                submission='pending' if uncertain else 'reconciled',
                execution='complete' if len(completed)==len(ARMS) else 'pending',
                resources=dict(scheduler_allocation_seconds=sum(float(v) for v in allocation if v is not None),
                    scheduler_measurement_complete=all(v is not None for v in allocation),
                    scope='Allocation seconds overlap process time; never added to process costs'),time=time.time())
            save(root/'launchers/status.json',status)
            if once or len(completed)==len(ARMS):return status
            time.sleep(30)


def queue(root,once=False):
    try:return _queue(root,once)
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
        save(Path(root)/'launchers/admission-error.json',dict(error=str(error),time=time.time(),admission='blocked'))
        raise


def worker(root,arm,attempt_id):
    root=Path(root).resolve();job=os.environ.get('SLURM_JOB_ID')
    if not job:raise ConfigError('GPU worker requires sbatch')
    if arm not in ARMS:raise ConfigError(f'Unknown linear CaLR arm: {arm}')
    with lock(root/'launchers'/f'{arm}.writer.lock'):
        plan=verify_plan(root)
        matches=[j for j in read(root/'launchers/submissions.json')['jobs'] if j['arm_id']==arm and j['attempt_id']==attempt_id]
        if len(matches)!=1:raise ConfigError('Worker requires unique durable intent')
        intent=matches[0]
        if (intent.get('job_id',job)!=job or os.environ.get('SLURM_JOB_NAME')!=intent['name']
                or intent['status'] not in ('submitting','submitted','uncertain') or intent['bindings']!=plan['bindings']
                or intent['cpu_gate_hash']!=plan['cpu_gate_hash'] or intent['gpu_gate_hash']!=read(root/'diagnostics/gpu-gate.json')['content_hash']):
            raise ConfigError('Worker intent/gate/job identity differs')
        own=continuation(root,arm)
        if own!=intent['continuation']:raise ConfigError('Own-run checkpoint changed since submission')
        path=root/'launchers'/f'worker-{arm}-{attempt_id}.json'
        if path.exists():raise ConfigError('Launch attempt already has a process; cannot reuse')
        process=uuid.uuid4().hex
        record=dict(arm_id=arm,attempt_id=attempt_id,job_id=job,name=intent['name'],process_uuid=process,
            intent_uuid=intent['intent_uuid'],bindings=plan['bindings'],cpu_gate_hash=plan['cpu_gate_hash'],gpu_gate_hash=intent['gpu_gate_hash'],
            status='running',started_at=time.time(),continuation=own,measurement_complete=False)
        save(path,record)
        from src.training.run import ResourceAttemptLedger
        manifest=read(root/'campaign/campaign_manifest.json');run_id=next(r['run_id'] for r in manifest['runs'] if r['arm_id']==arm)
        ledger=ResourceAttemptLedger(root/'runs'/arm,run_id=run_id)
        ledger.observe(process,sequence=1,run_id=run_id,launch_attempt_id=attempt_id,slurm_job_id=job,
            process_uuid=process,status='running',elapsed_seconds=None,attempted_steps=None,
            peak_allocated_bytes=None,peak_reserved_bytes=None,source_checkpoint=own['checkpoint'])
        env=dict(os.environ,OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',
            MATFORMER_PROCESS_UUID=process,MATFORMER_LAUNCH_ATTEMPT_ID=str(attempt_id))
        cmd=[PYTHON,str(root/'source/scripts/train_cuda_required.py'),'--config',str(root/'campaign/configs'/f'{arm}.yaml'),
            '--entry-evidence',str(root/'launchers'/f'cuda-entry-{arm}-{attempt_id}.json')]
        record['command']=cmd;save(path,record)
        try:
            result=subprocess.run(cmd,cwd=root/'source',env=env)
            resources=ResourceAttemptLedger(root/'runs'/arm,run_id=run_id).summary()
            record.update(returncode=result.returncode,status='completed' if result.returncode==0 else 'failed',
                finished_at=time.time(),measurement_complete=resources['measurement_complete'],resources=resources)
            save(path,record);return result.returncode
        except BaseException as error:
            record.update(status='failed',error=str(error),finished_at=time.time());save(path,record);raise


def report(root, report_source=None):
    """Dispatch the reporting operation without queueing or training."""
    root = Path(root).resolve()
    plan = verify_plan(root, gpu=False)
    source = Path(report_source).resolve() if report_source else root/'source'
    cmd = [PYTHON, str(source/'scripts/analyze_tinystories_optimizer_ownership.py'),
        'report-linear-calr', '--campaign-manifest', str(root/'campaign/campaign_manifest.json'),
        '--run-root', str(root/'runs'), '--reference-root', plan['reference_root'],
        '--output-dir', str(root/'reports/comparison')]
    result = subprocess.run(cmd,cwd=source)
    status_path=root/'launchers/status.json'
    status=read(status_path) if status_path.exists() else {}
    status.update(comparison='complete' if result.returncode==0 else 'incomplete',
        report_dispatch=dict(command=cmd,returncode=result.returncode,time=time.time()))
    save(status_path,status)
    return result.returncode


def verify_authorization(root, purpose, expected=None):
    """A subsequent user instruction, recorded durably, is distinct from readiness."""
    expected = bindings(root) if expected is None else expected
    path = Path(root)/'authorizations'/f'{purpose}.json'
    if not path.is_file():
        raise ConfigError(f'Missing durable explicit {purpose} execution authorization: {path}')
    record = read(path); check_seal(record, 'execution authorization')
    if (record.get('purpose') != purpose or record.get('bindings') != expected
            or record.get('authorized') is not True or not record.get('user_instruction')
            or not record.get('recorded_at')):
        raise ConfigError('Execution authorization absent or stale for tested source/config')
    return record


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','queue','worker','report'))
    parser.add_argument('--campaign-root',type=Path,required=True)
    parser.add_argument('--cpu-evidence',type=Path)
    parser.add_argument('--report-source',type=Path,help='Separate immutable reporting snapshot; never used for training')
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--attempt-id',type=int)
    parser.add_argument('--once',action='store_true');args=parser.parse_args(argv)
    root=args.campaign_root.resolve()
    if args.report_source is not None and args.mode != 'report':parser.error('--report-source is only valid for report')
    if REPO!=root/'source' and not (args.mode=='report' and args.report_source is not None and REPO==args.report_source.resolve()):
        if args.mode!='prepare':parser.error('Execute queue/worker/report from the tested source snapshot')
        # Convenience prepare command still runs the exact tested executable.
        return subprocess.call([PYTHON,str(root/'source/scripts/run_tinystories_linear_calr.py'),*(sys.argv[1:] if argv is None else argv)])
    if args.mode=='prepare':
        if args.cpu_evidence is None:parser.error('prepare requires --cpu-evidence')
        prepare(root,args.cpu_evidence)
    elif args.mode=='report':
        try:
            return report(root,args.report_source)
        except (ValueError, OSError, RuntimeError) as error:
            print(f'Campaign report incomplete: {error}', file=sys.stderr)
            return 1
    elif args.mode=='worker':
        if args.arm is None or args.attempt_id is None:parser.error('worker requires --arm and --attempt-id')
        return worker(root,args.arm,args.attempt_id)
    else:queue(root,args.once)
    return 0


if __name__=='__main__':sys.exit(main())
