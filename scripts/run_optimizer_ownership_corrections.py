#!/usr/bin/env python3
"""Execute and monitor only the authorized six-arm correction campaign."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

BASE = Path('/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/campaigns/concat-gmc-lmc-v1')
ARMS = ('C1-GMC', 'C1-LMC', 'C2-GMC', 'C2-LMC', 'C3-GMC', 'C3-LMC')
PYTHON = '/home/ivo.navarrete/.conda/envs/elasticnn/bin/python'
QOS = 'cscc-gpu-qos'


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def command(args):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=45).stdout.strip()


def queue_state():
    output = command(['squeue', '--noheader', '--user=ivo.navarrete', '--format=%i|%j|%T|%b|%M|%R|%q'])
    return [dict(zip(('job_id', 'name', 'state', 'gres', 'elapsed', 'node', 'qos'), line.split('|'))) for line in output.splitlines()]


def verify_scheduler_limits():
    """Require controller-enforced concurrency before admitting pending work."""
    config = command(['scontrol', 'show', 'config'])
    enforce = re.search(r'(?m)^AccountingStorageEnforce\s*=\s*(.*)$', config)
    if not enforce or not {'limits', 'qos'} <= set(enforce[1].strip().split(',')):
        raise RuntimeError('Slurm QoS limit enforcement is not verified')
    state = command(['scontrol', 'show', 'assoc_mgr', 'flags=qos'])
    blocks = re.split(r'(?m)(?=^QOS=)', state)
    block = next((b for b in blocks if b.startswith('QOS='+QOS+'(')), '')
    user = re.search(r'(?m)^      ivo\.navarrete\(\d+\)\n([^\n]+)', block)
    limits = dict(re.findall(r'(MaxJobsPU|MaxSubmitJobsPU)=(\d+)\(', user[1])) if user else {}
    if limits != {'MaxJobsPU': '2', 'MaxSubmitJobsPU': '4'}:
        raise RuntimeError(f'Expected two-running/four-submitted QoS limits: {limits}')
    return {'time': time.time(), 'qos': QOS, 'enforcement': enforce[1].strip(),
            'user_limits': user[0].strip(), 'max_running': 2, 'max_submitted': 4}


def verify_sources(plan):
    for relative, expected in plan['source_files_sha256'].items():
        path = Path(plan['source_dir']) / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Pinned source changed: {path}')
    for arm, expected in plan['config_sha256'].items():
        if hashlib.sha256((BASE/'campaign/configs'/f'{arm}.yaml').read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Pinned config changed: {arm}')


def read_tail(path, size=131072):
    if not path.exists():
        return []
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-size))
        data = stream.read().splitlines()
    rows = []
    for line in data:
        try: rows.append(json.loads(line))
        except (ValueError, UnicodeError): pass
    return rows


def progress(arm, gpu_index=None):
    root = BASE/'runs'/arm
    heartbeats = read_tail(root/'heartbeats.jsonl')
    traces = read_tail(root/'optimizer_ownership_trace.jsonl')
    checkpoint = root/'checkpoints/latest.pt'
    measured = [h for h in heartbeats if isinstance(h.get('step'), int) and h.get('elapsed_seconds') is not None]
    recent_seconds = None
    if len(measured) > 1:
        last = measured[-1]
        first = next((h for h in reversed(measured[:-1]) if h['step'] < last['step'] - 100), measured[0])
        if last['step'] > first['step']:
            recent_seconds = (last['elapsed_seconds']-first['elapsed_seconds'])/(last['step']-first['step'])
    gpu_logs = sorted((BASE/'logs').glob(f'{arm}-*-gpu.csv'), key=lambda p: p.stat().st_mtime)
    gpu_observation = None
    if gpu_logs:
        with gpu_logs[-1].open('rb') as handle:
            handle.seek(max(0, gpu_logs[-1].stat().st_size-4096))
            lines = handle.read().decode(errors='replace').splitlines()
            if gpu_index is not None:
                # Early v3 logs contain ordered NVML inventory without indices;
                # later logs record the physical index explicitly.
                with gpu_logs[-1].open() as inventory:
                    header = inventory.readline()
                    first = [inventory.readline() for _ in range(gpu_index+1)]
                if header.startswith('index,'):
                    matching = [line for line in lines if line.split(',')[0].strip() == str(gpu_index)]
                else:
                    first_fields = first[-1].split(',')
                    uuid = first_fields[1].strip() if len(first_fields) > 1 else None
                    matching = [line for line in lines if uuid and uuid in line]
                gpu_observation = matching[-1] if matching else None
    return {'last_heartbeat': heartbeats[-1] if heartbeats else None,
            'recent_seconds_per_update': recent_seconds,
            'eta_seconds': recent_seconds*(348528-measured[-1]['step']) if recent_seconds else None,
            'recent_timing_scope': 'includes validation/checkpoint intervals; startup excluded by step deltas',
            'gpu_observation': gpu_observation,
            'last_trace': traces[-1] if traces else None,
            'checkpoint_bytes': checkpoint.stat().st_size if checkpoint.exists() else None,
            'checkpoint_age_seconds': time.time()-checkpoint.stat().st_mtime if checkpoint.exists() else None}


def worker(arm, plan):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU worker requires sbatch')
    lock = (BASE/'launchers'/f'{arm}.writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_sources(plan)
    root = BASE/'runs'/arm
    if root.exists() and not (root/'checkpoints/latest.pt').exists():
        raise RuntimeError(f'Occupied run has no resumable checkpoint: {root}')
    temporary = BASE/'diagnostics'/f'tmp-{arm}'
    temporary.mkdir(exist_ok=True)
    env = dict(os.environ, TMPDIR=str(temporary), SLURM_TMPDIR=str(temporary), OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    job = os.environ['SLURM_JOB_ID']
    gpu_log = (BASE/'logs'/f'{arm}-{job}-gpu.csv').open('w')
    sampler = subprocess.Popen(['nvidia-smi', '--query-gpu=index,timestamp,uuid,utilization.gpu,memory.used,power.draw',
                                '--format=csv', '--loop=30'], stdout=gpu_log, stderr=subprocess.STDOUT)
    try:
        result = subprocess.run([PYTHON, 'train.py', '--config', str(BASE/'campaign/configs'/f'{arm}.yaml'),
                                 '--output-dir', str(root)], cwd=plan['source_dir'], env=env)
        try:
            allocation = command(['scontrol', 'show', 'job', job, '-o'])
        except (subprocess.SubprocessError, OSError):
            allocation = None
        save(BASE/'launchers'/f'worker-{job}.json', {'job_id':job, 'arm_id':arm,
             'worker_exit_code':result.returncode, 'name':os.environ.get('SLURM_JOB_NAME'), 'finished_at':time.time(),
             'allocation':allocation, 'source_dir':plan['source_dir']})
        return result.returncode
    finally:
        sampler.terminate(); sampler.wait(timeout=10); gpu_log.close()


def accounting(job_id):
    # This cluster's accounting database may be unavailable. The controller
    # retains completed job records long enough for the 30-second monitor.
    try:
        text = command(['scontrol', 'show', 'job', job_id, '-o'])
        fields = dict(piece.split('=', 1) for piece in text.split() if '=' in piece)
        duration = fields['RunTime']
        days, clock = duration.split('-') if '-' in duration else ('0', duration)
        h, m, s = map(int, clock.split(':'))
        return {'job_id':job_id, 'name':fields['JobName'], 'state':fields['JobState'],
                'exit_code':fields['ExitCode'], 'elapsed_seconds':int(days)*86400+h*3600+m*60+s,
                'start':fields['StartTime'], 'end':fields['EndTime'],
                'allocated_resources':fields.get('AllocTRES'), 'nodes':fields.get('NodeList'),
                'source':'scontrol', 'raw':text}
    except (subprocess.SubprocessError, OSError):
        pass
    try:
        text = command(['sacct', '-X', '-j', job_id, '--noheader', '--parsable2',
                        '--format=JobIDRaw,JobName%80,State,ExitCode,ElapsedRaw,Start,End,AllocTRES%150,NodeList'])
        rows = [dict(zip(('job_id','name','state','exit_code','elapsed_seconds','start','end','allocated_resources','nodes'), line.split('|'))) for line in text.splitlines()]
        return next((r for r in rows if r['job_id'] == job_id), None)
    except (subprocess.SubprocessError, OSError):
        path = BASE/'launchers'/f'worker-{job_id}.json'
        if not path.exists():
            raise RuntimeError(f'No scheduler or durable worker accounting for {job_id}')
        worker = json.loads(path.read_text())
        if worker['worker_exit_code'] != 0:
            return {'job_id':job_id, 'state':'FAILED', 'source':'durable_worker', 'worker':worker}
        return {'job_id':job_id, 'state':'COMPLETED', 'source':'durable_worker',
                'scheduler_history_unavailable':True, 'worker':worker}


def queue(plan, once=False):
    lock = (BASE/'launchers/queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_sources(plan)
    for gate in ('cpu', 'gpu'):
        evidence = json.loads((BASE/'diagnostics'/f'{gate}-gate.json').read_text())
        if evidence['status'] != 'passed' or evidence['runtime_source_sha256'] != plan['runtime_source_sha256']:
            raise RuntimeError(f'{gate} gate absent, failed or stale')
    path = BASE/'launchers/submissions.json'
    record = json.loads(path.read_text()) if path.exists() else {'campaign_id': plan['campaign_id'], 'jobs': []}
    while True:
        active = queue_state()
        active_ids = {r['job_id'] for r in active}
        for job in record['jobs']:
            if not job.get('job_id'):
                matches = [r for r in active if r['name'] == job['name']]
                if not matches:
                    # Reconcile an interrupted sbatch response with scheduler history.
                    try:
                        out = command(['sacct', '-X', '--user=ivo.navarrete', '--starttime='+plan['created_date'],
                                       '--name='+job['name'], '--noheader', '--parsable2', '--format=JobIDRaw,JobName%80'])
                        matches = [{'job_id': line.split('|')[0]} for line in out.splitlines() if line.split('|')[1] == job['name']]
                    except (subprocess.SubprocessError, OSError):
                        workers = [json.loads(p.read_text()) for p in (BASE/'launchers').glob('worker-*.json')]
                        matches = [r for r in workers if r.get('name') == job['name']]
                if len(matches) != 1:
                    raise RuntimeError(f'Ambiguous submission intent; preserve and reconcile before retry: {job}')
                job['job_id'] = matches[0]['job_id']; save(path, record)
            if job['job_id'] in active_ids and job.get('gpu_index') is None:
                allocation = command(['scontrol', 'show', 'job', job['job_id'], '-dd', '-o'])
                match = re.search(r'IDX:(\d+)\)', allocation)
                if match:
                    job['gpu_index'] = int(match.group(1))
                    job['allocation_record'] = allocation
                    save(path, record)
            if job['job_id'] not in active_ids and job.get('status') != 'completed':
                info = accounting(job['job_id'])
                if info is None or info['state'] in ('PENDING', 'RUNNING', 'COMPLETING'):
                    continue
                job['slurm_accounting'] = info
                terminal = BASE/'runs'/job['arm_id']/'terminal_validation_results.json'
                if info['state'] == 'COMPLETED' and terminal.exists():
                    job['status'] = 'completed'
                    job['terminal_sha256'] = hashlib.sha256(terminal.read_bytes()).hexdigest()
                    print(f"Completed {job['arm_id']} job {job['job_id']}", flush=True)
                else:
                    job['status'] = 'needs_reconciliation'
                    save(path, record)
                    raise RuntimeError(f"Attempt ended without valid terminal: {job['arm_id']} {info}. Reconcile checkpoint/resource ledger before continuation.")
                save(path, record)
        completed = {j['arm_id'] for j in record['jobs'] if j.get('status') == 'completed'}
        submitted = {j['arm_id'] for j in record['jobs']}
        pending = [a for a in ARMS if a not in submitted]
        # Slurm enforces two running across this user's QoS jobs, including
        # pending-job start races. Count all user jobs against four submitted;
        # defer admission if another QoS is active outside that shared cap.
        admitted = set()
        for arm in pending:
            active = queue_state()
            if len({r['job_id'] for r in active} | admitted) >= 4 or any(r.get('qos') != QOS for r in active):
                break
            save(BASE/'launchers/scheduler-limits.json', verify_scheduler_limits())
            verify_sources(plan)
            if (BASE/'runs'/arm).exists():
                raise RuntimeError(f'Fresh output already occupied: {arm}')
            name = 'oo-corr-v1-' + arm + '-a1'
            job = {'arm_id': arm, 'name': name, 'attempt': 1, 'status': 'submitting', 'intent_time': time.time()}
            record['jobs'].append(job); save(path, record)
            cmd = ['sbatch', '--parsable', '--job-name='+name, '--partition=cscc-gpu-p', '--qos='+QOS,
                   '--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--gres=gpu:1', '--mem=16G', '--time=24:00:00',
                   '--exclude=gpu-[05,50,51]', '--no-requeue', '--chdir='+plan['source_dir'],
                   '--output='+str(BASE/'logs'/f'{arm}-%j.out'), '--error='+str(BASE/'logs'/f'{arm}-%j.err'),
                   str(BASE/'launchers/worker.sbatch'), arm]
            job['command'] = cmd; save(path, record)
            job['job_id'] = command(cmd).split(';')[0]
            if not job['job_id'].isdigit(): raise RuntimeError('Invalid sbatch response')
            admitted.add(job['job_id'])
            job['status'] = 'submitted'; save(path, record)
            print(f"Submitted fresh {arm}: {job['job_id']}", flush=True)
        active = queue_state()
        submitted = {j['arm_id'] for j in record['jobs']}
        state = {'time': time.time(), 'pid': os.getpid(), 'completed': sorted(completed),
                 'remaining': [a for a in ARMS if a not in completed], 'slurm': active,
                 'progress': {a: progress(a, next(j.get('gpu_index') for j in record['jobs'] if j['arm_id'] == a)) for a in ARMS if a in submitted},
                 'max_running': 2, 'max_submitted': 4, 'admission_limit': 4,
                 'running_limit_enforcement': 'Slurm QoS MaxJobsPU=2'}
        save(BASE/'launchers/status.json', state)
        if len(completed) == 6:
            print('Six scheduler attempts completed; validating terminals and generating comparison.', flush=True)
            from src.evaluation.optimizer_ownership import freeze_campaign, report_correction_comparison
            frozen_dir = BASE/'reports/frozen'
            if not frozen_dir.exists():
                freeze_campaign(campaign_manifest=BASE/'campaign/campaign_manifest.json',
                                run_root=BASE/'runs', output_dir=frozen_dir)
            report_dir = BASE/'reports/comparison'
            if not report_dir.exists():
                report_correction_comparison(manifest=frozen_dir/'frozen_manifest.json',
                    reference_manifest=plan['reference_manifest'], output_dir=report_dir)
            save(BASE/'launchers/completion.json', {'status': 'completed', 'report': str(report_dir), 'time': time.time()})
            return
        if once: return
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('queue', 'worker'))
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    plan = json.loads((BASE/'launchers/plan.json').read_text())
    if tuple(plan['arms']) != ARMS or Path(plan['campaign_root']) != BASE:
        raise RuntimeError('Wrong campaign plan')
    sys.path.insert(0, plan['source_dir'])
    if args.mode == 'worker':
        if args.arm is None: parser.error('worker requires --arm')
        return worker(args.arm, plan)
    queue(plan, args.once)
    return 0


if __name__ == '__main__':
    sys.exit(main())
