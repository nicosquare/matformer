#!/usr/bin/env python3
"""Monitor the accepted launcher revision and record actual per-attempt GPU use."""
import argparse
import os
from pathlib import Path
import signal
import socket
import sys
import time
import traceback


def observe_gpu(root, ops, intent):
    arm, attempt = intent['arm_id'], intent['attempt_id']
    entry_path = root/'launchers'/f'cuda-entry-{arm}-{attempt}.json'
    worker_path = root/'launchers'/f'worker-{arm}-{attempt}.json'
    config_path = root/'runs'/arm/'config.json'
    ledger_path = root/'runs'/arm/'resource_attempts.json'
    if not all(p.exists() for p in (entry_path, worker_path, config_path, ledger_path)):
        return None
    entry, worker, config, ledger = (ops.read(p) for p in (entry_path, worker_path, config_path, ledger_path))
    if entry['job_id'] != intent['job_id'] or worker['job_id'] != intent['job_id']:
        raise RuntimeError('GPU observation job identity differs')
    if config['training'].get('resolved_mixed_precision') != 'bf16':
        ops.command(['scancel', intent['job_id']])
        raise RuntimeError(f'{arm}: cancelled unexpected CPU/non-BF16 production')
    measured = ledger['attempts'].get(worker['process_uuid'])
    if not measured or not measured.get('peak_allocated_bytes') or not measured.get('attempted_steps'):
        return None
    if str(measured.get('slurm_job_id')) != intent['job_id']:
        raise RuntimeError('GPU ledger job identity differs')
    return dict(arm_id=arm, attempt_id=attempt, job_id=intent['job_id'],
                host=entry['host'], process_id=entry['process_id'], process_uuid=worker['process_uuid'],
                resolved_mixed_precision='bf16', peak_allocated_bytes=measured['peak_allocated_bytes'],
                peak_reserved_bytes=measured['peak_reserved_bytes'], attempted_steps=measured['attempted_steps'],
                elapsed_seconds=measured['elapsed_seconds'], observed_at=time.time(),
                entry_path=str(entry_path), ledger_path=str(ledger_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', required=True, type=Path)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    sys.path.insert(0, str(root/'launchers/cuda-required-v1'))
    sys.path.insert(1, str(root/'launchers'))
    from scripts import run_tinystories_matformer_widths as ops
    from reconcile_matformer_terminal_resources import reconcile
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with ops.lock(root/'launchers/background-checker.lock'):
        path = root/'launchers/background-checker.json'
        gpu_path = root/'launchers/gpu-verification.json'
        seen = ops.read(gpu_path).get('attempts', {}) if gpu_path.exists() else {}
        record = dict(status='running', pid=os.getpid(), host=socket.gethostname(), started_at=time.time(),
                      campaign_root=str(root), checker_sha256=ops.digest(__file__),
                      reconciliation_helper_sha256=ops.digest(root/'launchers/reconcile_matformer_terminal_resources.py'),
                      execution_source=str(root/'launchers/cuda-required-v1'),
                      excluded_nodes=ops.EXCLUDED_NODES, check_interval_seconds=30)
        ops.save(path, record)
        try:
            while True:
                if (ops.digest(__file__) != record['checker_sha256']
                        or ops.digest(root/'launchers/reconcile_matformer_terminal_resources.py') != record['reconciliation_helper_sha256']):
                    raise RuntimeError('Monitor/reconciliation source changed')
                ops.verify_execution_revision(root)
                reconcile(root, ops)
                result = ops.queue(root, once=True)
                active = {r['job_id'] for r in ops.queue_state()}
                for intent in result['jobs']:
                    if intent.get('job_id') in active:
                        observation = observe_gpu(root, ops, intent)
                        if observation:
                            seen[intent['job_id']] = observation
                ops.save(gpu_path, dict(attempts=seen, updated_at=time.time()))
                record.update(last_cycle_at=time.time(), completed=result['completed'], gpu_verified_jobs=sorted(seen))
                ops.save(path, record)
                if args.once or result['production'] == 'complete':
                    record['status'] = 'cycle_complete' if args.once else 'production_validated'
                    break
                time.sleep(30)
        except SystemExit as error:
            record.update(status='stopped', exit_code=error.code)
            raise
        except BaseException as error:
            record.update(status='blocked', error=str(error))
            traceback.print_exc()
            raise
        finally:
            record['finished_at'] = time.time()
            ops.save(path, record)


if __name__ == '__main__':
    main()
