#!/usr/bin/env python3
"""Prepare, stage and reconcile the nine-run MatFormer width campaign."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from src.evaluation import optimizer_ownership as campaign
from src.utils.config import ConfigError
from src.utils.metrics import write_json_artifact
from src.utils.reproducibility import stable_hash

ARMS = tuple(a['arm_id'] for a in campaign.campaign_arms(4))
STANDALONES, ELASTICS = ARMS[:4], ARMS[4:]
PYTHON = '/home/ivo.navarrete/.conda/envs/elasticnn/bin/python'
QOS = 'cscc-gpu-qos'
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'RESIZING', 'REQUEUED', 'REQUEUE_FED', 'REQUEUE_HOLD', 'SIGNALING', 'STAGE_OUT'}
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_json_artifact(Path(path), value)


def sealed(value):
    return {**value, 'content_hash': stable_hash(value)}


def check_seal(value, label):
    campaign._check_content_hash(value, 'content_hash', label)


@contextmanager
def lock(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ConfigError(f'Already locked: {path}') from error
        yield


def source_files(root):
    root = Path(root)
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.suffix != '.pyc'}


def bindings(root):
    """Read current immutable inputs; never relabel an old gate with new hashes."""
    root = Path(root).resolve()
    manifest_path = root/'campaign/campaign_manifest.json'
    manifest = campaign._read_preflight_manifest(manifest_path)
    if manifest['schema_version'] != 4 or manifest['campaign_id'] != campaign.MATFORMER_CAMPAIGN_ID:
        raise ConfigError('Expected schema-4 MatFormer campaign')
    if Path(manifest['run_output_root']) != root/'runs':
        raise ConfigError('Campaign run root differs from prepared identity')
    preflight = read(root/'campaign/preflight.json')
    if preflight.get('status') != 'passed' or preflight.get('manifest_hash') != manifest['manifest_hash']:
        raise ConfigError('Missing or stale preflight')
    import yaml
    configs = {}
    for run in manifest['runs']:
        path = root/'campaign/configs'/f"{run['arm_id']}.yaml"
        if yaml.safe_load(path.read_text()) != run['executable_config']:
            raise ConfigError(f'Config differs from preflight: {path}')
        configs[run['arm_id']] = digest(path)
    snapshot = read(root/'diagnostics/source-manifest.json')
    check_seal(snapshot, 'source snapshot')
    actual = source_files(root/'source')
    if actual != snapshot['files']:
        raise ConfigError('Tested source snapshot changed')
    for run in manifest['runs']:
        for relative, expected in run['initialization']['source_files_sha256'].items():
            if actual.get(relative) != expected:
                raise ConfigError(f'Snapshot differs from preflight source: {relative}')
    return dict(campaign_id=manifest['campaign_id'], manifest_hash=manifest['manifest_hash'],
        campaign_manifest_sha256=digest(manifest_path), preflight_sha256=digest(root/'campaign/preflight.json'),
        source_sha256=stable_hash(actual), config_sha256=configs, config_set_sha256=stable_hash(configs))


def verify_gate(root, mode, expected=None, path=None):
    expected = bindings(root) if expected is None else expected
    path = Path(path) if path else Path(root)/'diagnostics'/f'{mode}-gate.json'
    gate = read(path); check_seal(gate, f'{mode} gate')
    if gate.get('mode') != mode or gate.get('status') != 'passed' or gate.get('bindings') != expected:
        raise ConfigError(f'{mode} gate absent, failed or stale: {path}')
    results = gate.get('checks')
    if not results or any(r.get('returncode') != 0 or r.get('tests', 0) <= 0 or r.get('failures', 0) or r.get('errors', 0) for r in results):
        raise ConfigError(f'{mode} gate lacks passed executed checks')
    for result in results:
        for artifact in result.get('artifacts', []):
            if digest(artifact['path']) != artifact['sha256']:
                raise ConfigError(f'{mode} gate result changed: {artifact["path"]}')
    if mode == 'gpu':
        cpu = verify_gate(root, 'cpu', expected)
        if (not gate.get('job_id') or not gate.get('hardware') or any(r.get('skipped') for r in results)
                or gate.get('cpu_gate_hash') != cpu['content_hash'] or gate.get('real_shape_arms') != list(ARMS)):
            raise ConfigError('GPU gate requires sbatch, all nine bf16 arms, no skips and matching CPU evidence')
    return gate


def verify_plan(root, *, gpu=True):
    root = Path(root).resolve()
    plan = read(root/'launchers/plan.json'); check_seal(plan, 'launch plan')
    expected = bindings(root)
    if plan.get('bindings') != expected or plan.get('campaign_root') != str(root) or plan.get('arms') != list(ARMS):
        raise ConfigError('Launch plan source/config/campaign identity changed')
    cpu = verify_gate(root, 'cpu', expected)
    if cpu['content_hash'] != plan['cpu_gate_hash']:
        raise ConfigError('Prepared CPU evidence changed')
    if gpu:
        verify_gate(root, 'gpu', expected)
    return plan


def prepare(root, cpu_evidence, reference_manifest=None):
    root = Path(root).resolve()
    with lock(root/'launchers/prepare.lock'):
        expected = bindings(root)
        cpu = verify_gate(root, 'cpu', expected, cpu_evidence)
        manifest = read(root/'campaign/campaign_manifest.json')
        reservation = read(campaign._reservation_path(root/'runs'))
        required = dict(campaign_id=manifest['campaign_id'], manifest_path=str(root/'campaign/campaign_manifest.json'),
            manifest_hash=manifest['manifest_hash'], run_ids=[r['run_id'] for r in manifest['runs']])
        if reservation != required:
            raise ConfigError('Preflight reservation does not match; cannot adopt')
        path = root/'launchers/plan.json'
        if path.exists():
            return verify_plan(root, gpu=False)
        if (root/'runs').exists() and any((root/'runs').iterdir()):
            raise ConfigError('Preparation cannot adopt occupied runs')
        for name in ('diagnostics', 'launchers', 'logs', 'runs', 'reports'):
            (root/name).mkdir(exist_ok=True)
        canonical = root/'diagnostics/cpu-gate.json'
        if Path(cpu_evidence).resolve() != canonical:
            if canonical.exists() and read(canonical) != cpu:
                raise ConfigError('Existing CPU evidence differs')
            save(canonical, cpu)
        reference = str(Path(reference_manifest).resolve()) if reference_manifest else None
        plan = sealed(dict(schema_version=1, campaign_id=manifest['campaign_id'], campaign_root=str(root),
            arms=list(ARMS), bindings=expected, cpu_gate_hash=cpu['content_hash'],
            created_date=time.strftime('%Y-%m-%d', time.gmtime()), reference_manifest=reference,
            reference_status='available_unvalidated' if reference and Path(reference).is_file() else 'outstanding',
            assigned_updates={r['arm_id']: r['assigned_updates'] for r in manifest['runs']}))
        save(path, plan)
        return plan


def standalone_barrier(root, plan):
    root = Path(root)
    path = root/'launchers/standalone-barrier.json'
    terminals = campaign.inspect_selected_terminals(root/'campaign/campaign_manifest.json', STANDALONES)
    manifest = read(root/'campaign/campaign_manifest.json')
    value = sealed(dict(schema_version=1, status='complete', bindings=plan['bindings'],
        contracts={r['arm_id']: r['contract_hash'] for r in manifest['runs'] if r['arm_id'] in STANDALONES}, terminals=terminals))
    if path.exists():
        saved = read(path); check_seal(saved, 'standalone barrier')
        if saved != value:
            raise ConfigError('Previously validated standalone barrier changed')
    else:
        save(path, value)
    return value


def command(args):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=45).stdout.strip()


def queue_state():
    result = command(['squeue', '--noheader', '--user='+getpass.getuser(), '--format=%i|%j|%T|%q'])
    return [dict(zip(('job_id','name','state','qos'), line.split('|'))) for line in result.splitlines() if line]


def scheduler_limits():
    config = command(['scontrol','show','config'])
    enforce = re.search(r'(?m)^AccountingStorageEnforce\s*=\s*(.*)$', config)
    if not enforce or not {'limits','qos'} <= set(enforce[1].strip().split(',')):
        raise ConfigError('Scheduler limit enforcement unavailable')
    state = command(['scontrol','show','assoc_mgr','flags=qos'])
    block = next((b for b in re.split(r'(?m)(?=^QOS=)',state) if b.startswith('QOS='+QOS+'(')), '')
    user = re.search(r'(?m)^\s+'+re.escape(getpass.getuser())+r'\(\d+\)\n([^\n]+)', block)
    if user is None or shutil.which('sacctmgr') is None:
        return controller_limits(enforce[1])
    values = dict(re.findall(r'(MaxJobsPU|MaxSubmitJobsPU)=(\d+)\(', user[1])) if user else {}
    if set(values) != {'MaxJobsPU','MaxSubmitJobsPU'} or not 0 < int(values['MaxJobsPU']) <= 2:
        raise ConfigError('Cannot guarantee user-wide running ceiling')
    running, submitted = int(values['MaxJobsPU']), min(4,int(values['MaxSubmitJobsPU']))
    if submitted < 1:
        raise ConfigError('No verified submission capacity')
    associations = command(['sacctmgr','-nP','show','assoc','where','user='+getpass.getuser(), 'format=MaxJobs,MaxSubmitJobs'])
    for row in associations.splitlines():
        fields = row.split('|')
        if len(fields) < 2:
            raise ConfigError('Malformed association limits')
        for index, value in enumerate(fields[:2]):
            if value.strip().isdigit() and int(value) > 0:
                if index == 0: running = min(running, int(value))
                else: submitted = min(submitted, int(value))
    return dict(max_running=running, max_submitted=submitted, qos=QOS, association_limits=associations, enforcement=enforce[1])


def controller_limits(enforcement):
    # scontrol prints QoS ceilings only alongside usage rows. An idle user may
    # have no row. Query the same controller's configured values, never infer
    # this user's ceiling from another user's usage or a historical record.
    with tempfile.TemporaryDirectory(prefix='mw-slurm-limits-') as temporary:
        executable = str(Path(temporary)/'limits')
        command(['gcc', '-Wall', '-Werror', str(REPO/'scripts/matformer_slurm_limits.c'),
                 '-lslurm', '-o', executable])
        raw = command([executable, QOS, getpass.getuser()])
    rows = [line.split('|') for line in raw.splitlines()]
    qos = [row for row in rows if row[0] == 'QOS']
    if len(qos) != 1 or len(qos[0]) != 7:
        raise ConfigError('Missing configured controller QoS limits')
    values = [int(value) for value in qos[0][1:]]
    if not 0 < values[0] <= 2 or not 0 < values[1] < 0xfffffffe:
        raise ConfigError('Cannot guarantee user-wide running/submission ceiling')
    associations = {int(row[1]): row for row in rows if row[0] == 'ASSOC' and len(row) == 8}
    selected = [row for row in associations.values() if row[3] == getpass.getuser()]
    if not selected:
        raise ConfigError('Missing user association limits')
    applicable = {}
    for row in selected:
        seen = set()
        while True:
            identity, parent = int(row[1]), int(row[2])
            if identity in seen:
                raise ConfigError('Cyclic controller association ancestry')
            seen.add(identity); applicable[identity] = row
            if parent == 0:
                break
            if parent not in associations:
                raise ConfigError('Missing parent association limits')
            row = associations[parent]
    for row in applicable.values():
        values.extend(int(value) for value in row[4:])
    # Slurm NO_VAL/INFINITE are not finite ceilings. Apply every stricter
    # positive user/account/group limit; Slurm enforces shared group usage.
    running = min([2] + [v for v in values[::2] if 0 < v < 0xfffffffe])
    submitted = min([4] + [v for v in values[1::2] if 0 < v < 0xfffffffe])
    return dict(max_running=running, max_submitted=submitted, qos=QOS,
                association_limits=list(applicable.values()), qos_limits=qos[0],
                enforcement=enforcement, query='slurm_load_assoc_mgr_info')


def scheduler_history(root, intent):
    """Unknown visibility stays unknown; it never grants a retry."""
    args = ['sacct','-X','--noheader','--parsable2','--user='+getpass.getuser(),
        '--starttime='+intent['created_date'], '--name='+intent['name'], '--format=JobIDRaw,JobName%100,State,ElapsedRaw']
    try:
        rows = [dict(zip(('job_id','name','state','allocation_seconds'), line.split('|'))) for line in command(args).splitlines()]
    except (OSError, subprocess.SubprocessError):
        rows = []
    rows = [r for r in rows if r.get('name') == intent['name'] and r.get('job_id','').isdigit()]
    if intent.get('job_id'):
        rows = [r for r in rows if r['job_id'] == intent['job_id']]
    workers = [read(p) for p in (Path(root)/'launchers').glob(f"worker-{intent['arm_id']}-{intent['attempt_id']}.json")]
    ids = {r['job_id'] for r in rows + workers}
    if len(ids) > 1:
        raise ConfigError('Ambiguous scheduler/worker submission identity')
    if rows:
        return rows[0]
    if workers:
        worker = workers[0]
        if worker['status'] != 'running' and 'returncode' not in worker:
            return None
        return dict(job_id=worker['job_id'], name=intent['name'], state='RUNNING' if worker['status']=='running' else 'COMPLETED' if worker.get('returncode') == 0 else 'FAILED', allocation_seconds=None)
    return None


def continuation(root, arm):
    """Validate own complete bundle before admitting any occupied identity."""
    import torch
    from src.training import checkpointing as cp, data, modeling, steps, run as training_run
    from src.utils.config import resolve_run_config
    root = Path(root)
    output = root/'runs'/arm
    if not output.exists():
        return {'mode': 'fresh', 'step': 0, 'checkpoint': None}
    path = output/'checkpoints/latest.pt'
    if not path.is_file():
        raise ConfigError(f'Occupied run has no durable own checkpoint: {output}')
    config = resolve_run_config(root/'campaign/configs'/f'{arm}.yaml')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    # Restore validation needs real model descriptors and the real packed sampler,
    # but no GPU, forward pass, or mutation of the checkpoint.
    # Construction only consumes the CPU tensor RNG. Do not initialize a CUDA
    # context in the scheduler/worker parent; the trainer owns the sole GPU process.
    rng = torch.get_rng_state()
    try:
        model = modeling.build_model(config)
        optimizer, scheduler = steps.build_optimizer_and_scheduler(model, config['training'])
        loader, _, _, manifest = data.build_packed_mmap_dataloaders(config, torch.device('cpu'))
        training_run._attach_probabilistic_role_provenance(config, training_run._packed_role_partition(manifest))
        _, saved_rng = cp._validate_ownership_payload(payload, config, model, optimizer, scheduler, train_dataloader=loader, inspect_only=True)
        cp._validate_ownership_action_rng(payload, config, saved_rng)
    finally:
        torch.set_rng_state(rng)
    step = payload['step']
    return dict(mode='completion_only' if step == config['training']['max_steps'] else 'resume', step=step,
        checkpoint=dict(path=str(path), step=step, sha256=digest(path)))


def progress(root, arm, horizon):
    from scripts.run_tinystories_inverse_membership import read_tail
    rows = read_tail(Path(root)/'runs'/arm/'heartbeats.jsonl')
    measured = [r for r in rows if isinstance(r.get('step'),int) and r.get('elapsed_seconds') is not None]
    rate = None
    if len(measured)>1:
        first,last = measured[0],measured[-1]
        if last['step'] > first['step'] and last['elapsed_seconds'] >= first['elapsed_seconds']:
            rate = (last['elapsed_seconds']-first['elapsed_seconds'])/(last['step']-first['step'])
    step = measured[-1]['step'] if measured else 0
    return dict(step=step, assigned_updates=horizon, fraction=step/horizon,
        eta_seconds=max(0,horizon-step)*rate if rate is not None else None)


def _queue(root, once=False):
    root = Path(root).resolve()
    with lock(root/'launchers/queue.lock'):
        while True:
            plan = verify_plan(root)
            path = root/'launchers/submissions.json'
            record = read(path) if path.exists() else dict(campaign_id=plan['campaign_id'], jobs=[])
            if record['campaign_id'] != plan['campaign_id']:
                raise ConfigError('Submission campaign identity changed')
            active = queue_state()
            for intent in record['jobs']:
                if intent['status'] in ('completed','retryable'):
                    continue
                matches = [r for r in active if r['name'] == intent['name']]
                if len(matches)>1:
                    raise ConfigError('Duplicate scheduler submission name')
                info = matches[0] if matches else scheduler_history(root, intent)
                if info is None:
                    # Preserve known IDs too: accounting can lag after squeue.
                    intent['status'] = 'uncertain'; save(path,record)
                    raise ConfigError(f"Uncertain submission; reconciliation required: {intent['name']}")
                if intent.get('job_id') and intent['job_id'] != info['job_id']:
                    raise ConfigError('Submission job identity changed')
                intent['job_id'] = info['job_id']
                scheduler_state = info['state'].split()[0].rstrip('+')
                if scheduler_state in ACTIVE:
                    intent['status'] = 'submitted'
                elif scheduler_state in TERMINAL:
                    intent['scheduler_accounting'] = info
                    terminal = root/'runs'/intent['arm_id']/'terminal_validation_results.json'
                    if terminal.exists():
                        campaign.inspect_selected_terminals(root/'campaign/campaign_manifest.json', [intent['arm_id']])
                        intent['status'] = 'completed'
                    else:
                        own = continuation(root,intent['arm_id'])
                        if own['mode']=='fresh':
                            raise ConfigError('Ended attempt without a durable checkpoint; reconcile before retry')
                        intent['status'] = 'retryable'; intent['continuation'] = own
                else:
                    intent['status'] = 'uncertain'; save(path,record)
                    raise ConfigError(f'Unrecognized scheduler state: {scheduler_state}')
                save(path,record)
            completed = {j['arm_id'] for j in record['jobs'] if j['status']=='completed'}
            barrier_path = root/'launchers/standalone-barrier.json'
            if barrier_path.exists() or set(STANDALONES) <= completed:
                standalone_barrier(root,plan)
                eligible = ARMS
            else:
                eligible = STANDALONES
            for arm in eligible:
                own_intents = [j for j in record['jobs'] if j['arm_id']==arm]
                if arm in completed or any(j['status'] not in ('completed','retryable') for j in own_intents):
                    continue
                limits = scheduler_limits(); active = queue_state()
                outstanding = {j['job_id'] for j in record['jobs'] if j['status']=='submitted'}
                if (len({r['job_id'] for r in active} | outstanding) >= limits['max_submitted']
                        or any(r.get('qos') != QOS for r in active)
                        or sum(r.get('state')=='RUNNING' for r in active) > limits['max_running']):
                    break
                verify_plan(root)
                if arm in ELASTICS: standalone_barrier(root,plan)
                own = continuation(root,arm)
                attempt = max((j['attempt_id'] for j in own_intents), default=0)+1
                name = f'mw-v1-{arm}-a{attempt}'
                intent = dict(arm_id=arm, attempt_id=attempt, name=name, status='submitting',
                    created_date=plan['created_date'], intent_time=time.time(), continuation=own,
                    bindings=plan['bindings'], cpu_gate_hash=plan['cpu_gate_hash'],
                    gpu_gate_hash=read(root/'diagnostics/gpu-gate.json')['content_hash'])
                worker_cmd = [PYTHON,str(root/'source/scripts/run_tinystories_matformer_widths.py'),'worker',
                    '--campaign-root',str(root),'--arm',arm,'--attempt-id',str(attempt)]
                cmd = ['sbatch','--parsable','--job-name='+name,'--partition=cscc-gpu-p','--qos='+QOS,
                    '--nodes=1','--ntasks=1','--cpus-per-task=4','--gres=gpu:1','--mem=16G','--time=24:00:00',
                    '--exclude=gpu-[05,50,51]','--no-requeue','--chdir='+str(root/'source'),
                    '--output='+str(root/'logs'/f'{arm}-a{attempt}-%j.out'), '--error='+str(root/'logs'/f'{arm}-a{attempt}-%j.err'),
                    '--wrap='+shlex.join(worker_cmd)]
                intent['command'] = cmd
                record['jobs'].append(intent); save(path,record)
                result = command(cmd).split(';')[0]
                if not result.isdigit():
                    raise ConfigError('Uncertain sbatch response; durable intent retained')
                intent.update(job_id=result,status='submitted'); save(path,record)
            status = dict(production='complete' if len(completed)==9 else 'pending', completed=sorted(completed),
                progress={arm:progress(root,arm,plan['assigned_updates'][arm]) for arm in ARMS},
                jobs=record['jobs'], time=time.time())
            save(root/'launchers/status.json',status)
            if once or len(completed)==9: return status
            time.sleep(30)


def queue(root, once=False):
    try:
        return _queue(root, once)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        path = Path(root)/'launchers/admission-error.json'
        save(path, dict(error=str(error), time=time.time(), admission='blocked'))
        raise


def worker(root, arm, attempt_id):
    root = Path(root).resolve()
    job = os.environ.get('SLURM_JOB_ID')
    if not job: raise ConfigError('GPU worker requires sbatch')
    with lock(root/'launchers'/f'{arm}.writer.lock'):
        plan = verify_plan(root)
        matches = [j for j in read(root/'launchers/submissions.json')['jobs'] if j['arm_id']==arm and j['attempt_id']==attempt_id]
        if len(matches)!=1: raise ConfigError('Worker requires a unique durable submission intent')
        intent = matches[0]
        if (intent.get('job_id',job)!=job or os.environ.get('SLURM_JOB_NAME') != intent['name']
                or intent['status'] not in ('submitting','submitted','uncertain') or intent['bindings']!=plan['bindings']
                or intent.get('cpu_gate_hash', plan['cpu_gate_hash']) != plan['cpu_gate_hash']
                or intent['gpu_gate_hash'] != read(root/'diagnostics/gpu-gate.json')['content_hash']):
            raise ConfigError('Worker intent/gate/job identity differs')
        if arm in ELASTICS: standalone_barrier(root,plan)
        own = continuation(root,arm)
        if own != intent['continuation']:
            raise ConfigError('Own-run checkpoint changed since submission')
        path = root/'launchers'/f'worker-{arm}-{attempt_id}.json'
        if path.exists(): raise ConfigError('Launch attempt already has a process; cannot reuse')
        process = uuid.uuid4().hex
        record = dict(arm_id=arm, attempt_id=attempt_id, job_id=job, name=intent['name'], process_uuid=process,
            status='running', started_at=time.time(), continuation=own, measurement_complete=False)
        save(path,record)
        env = dict(os.environ, OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
            MATFORMER_PROCESS_UUID=process, MATFORMER_LAUNCH_ATTEMPT_ID=str(attempt_id))
        from src.training.run import ResourceAttemptLedger
        manifest = read(root/'campaign/campaign_manifest.json')
        run_id = next(r['run_id'] for r in manifest['runs'] if r['arm_id'] == arm)
        ledger = ResourceAttemptLedger(root/'runs'/arm, run_id=run_id)
        ledger.observe(process, sequence=1, run_id=run_id, launch_attempt_id=attempt_id, slurm_job_id=job,
            process_uuid=process, status='running', elapsed_seconds=None, attempted_steps=None,
            peak_allocated_bytes=None, peak_reserved_bytes=None, source_checkpoint=own['checkpoint'])
        cmd = [PYTHON,str(root/'source/train.py'),'--config',str(root/'campaign/configs'/f'{arm}.yaml')]
        try:
            result = subprocess.run(cmd,cwd=root/'source',env=env)
            record.update(returncode=result.returncode,status='completed' if result.returncode==0 else 'failed',
                finished_at=time.time(), measurement_complete=True)
            save(path,record)
            return result.returncode
        except BaseException as error:
            record.update(status='failed', error=str(error), finished_at=time.time())
            save(path,record)
            raise


def validate_report(path, *, endpoints):
    record = read(path); check_seal(record,'report')
    if record.get('status')!='complete' or record.get('endpoint_count')!=endpoints:
        raise ConfigError(f'Incomplete report: {path}')
    campaign._check_sources(record['input_sources'])
    for relative, expected in record['output_sha256'].items():
        if digest(Path(path).parent/relative)!=expected:
            raise ConfigError(f'Report output changed: {relative}')
    return record


def report(root, reference_manifest=None):
    root = Path(root).resolve()
    with lock(root/'launchers/report.lock'):
        plan = verify_plan(root, gpu=False)
        completion = dict(production='pending', new_report='pending', combined_report='outstanding')
        status_path = root/'launchers/completion.json'
        frozen = root/'reports/frozen/frozen_manifest.json'
        try:
            if not frozen.exists():
                campaign.freeze_campaign(campaign_manifest=root/'campaign/campaign_manifest.json',run_root=root/'runs',output_dir=frozen.parent)
            campaign._validated_comparison_sources(frozen,schema_version=4)
            completion['production']='complete'
            new = root/'reports/new/comparison_report.json'
            if new.exists():
                saved_new = validate_report(new,endpoints=24)
                if saved_new['input_sources'][0] != campaign._source_record(frozen):
                    raise ConfigError('New report uses a different frozen manifest')
            else: campaign.report_campaign(manifest=frozen,output_dir=new.parent)
            completion['new_report']='complete'; save(status_path,completion)
            reference = reference_manifest or plan.get('reference_manifest')
            if not reference or not Path(reference).is_file():
                raise ConfigError('Historical reference unavailable; combined report outstanding')
            combined = root/'reports/combined/comparison_report.json'
            if combined.exists():
                saved = validate_report(combined,endpoints=28)
                if str(Path(reference).resolve()) not in {s['path'] for s in saved['input_sources']}:
                    raise ConfigError('Combined report uses a different requested reference')
            else:
                campaign.report_matformer_widths_comparison(manifest=frozen,reference_manifest=reference,output_dir=combined.parent)
            completion['combined_report']='complete'; save(status_path,completion)
            return completion
        except (OSError,ValueError,RuntimeError) as error:
            completion['outstanding_reason']=str(error); save(status_path,completion)
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','queue','worker','report'))
    parser.add_argument('--campaign-root',required=True,type=Path)
    parser.add_argument('--cpu-evidence',type=Path)
    parser.add_argument('--reference-manifest',type=Path)
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--attempt-id',type=int)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args(argv)
    if REPO != args.campaign_root.resolve()/'source':
        parser.error('Run operational commands from the tested campaign source snapshot')
    if args.mode=='prepare':
        if args.cpu_evidence is None: parser.error('prepare requires --cpu-evidence')
        prepare(args.campaign_root,args.cpu_evidence,args.reference_manifest)
    elif args.mode=='worker':
        if args.arm is None or args.attempt_id is None: parser.error('worker requires --arm and --attempt-id')
        return worker(args.campaign_root,args.arm,args.attempt_id)
    elif args.mode=='report': report(args.campaign_root,args.reference_manifest)
    else: queue(args.campaign_root,args.once)
    return 0


if __name__=='__main__':
    sys.exit(main())
