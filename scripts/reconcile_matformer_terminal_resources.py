"""Reconcile the known post-summary elapsed-time observation from saved evidence.

Operational helper outside the immutable campaign snapshot. Training, terminal
evaluation, checkpoints and strict validators are unchanged. Only a successful,
ended attempt with otherwise identical resource observations is eligible.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import tempfile
import time


def reconciled_summary(summary, measured):
    old = summary['resource_summary']
    audit = summary['optimizer_ownership']
    resources = audit['resources']
    if summary['status'] != 'completed' or not measured['measurement_complete']:
        raise ValueError('Only completed, measured terminal resources can reconcile')
    if set(old) != set(measured):
        raise ValueError('Resource summary fields differ')
    for field, value in old.items():
        if resources.get(field) != value:
            raise ValueError(f'Nested resource summary differs: {field}')
        if field != 'elapsed_seconds' and value != measured[field]:
            raise ValueError(f'Non-timing resource mismatch: {field}')
    before, after = old['elapsed_seconds'], measured['elapsed_seconds']
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
           for v in (before, after)) or after < before:
        raise ValueError('Invalid or decreasing measured elapsed time')
    if summary['training_wall_time_seconds'] != before:
        raise ValueError('Top-level elapsed time differs')
    tokens = audit['tokens_seen']
    attempted_tokens = old['attempted_steps'] * audit['packed_tokens_per_update']
    for field, numerator in [('useful_committed_tokens_per_second', tokens),
                             ('attempted_tokens_per_second', attempted_tokens)]:
        if resources[field] != numerator / before:
            raise ValueError(f'Existing derived throughput differs: {field}')
    result = copy.deepcopy(summary)
    result['resource_summary']['elapsed_seconds'] = after
    result['training_wall_time_seconds'] = after
    result['optimizer_ownership']['resources'].update(
        elapsed_seconds=after, useful_committed_tokens_per_second=tokens / after,
        attempted_tokens_per_second=attempted_tokens / after)
    return result


def reconcile(root, ops):
    """Use the caller's verified snapshot modules; never patch their validators."""
    from src.training.run import ResourceAttemptLedger

    root = Path(root)
    changed = []
    with ops.lock(root/'launchers/queue.lock'):
        manifest = ops.campaign._read_preflight_manifest(root/'campaign/campaign_manifest.json')
        intents = ops.read(root/'launchers/submissions.json')['jobs']
        active = ops.queue_state()
        for definition in manifest['runs']:
            arm = definition['arm_id']
            output = root/'runs'/arm
            path = output/'run_summary.json'
            if not path.exists() or not (output/'terminal_validation_results.json').exists():
                continue
            # Do not touch any attempt while Slurm still reports it active.
            own = [j for j in intents if j['arm_id'] == arm]
            if not own or any(a['job_id'] == j.get('job_id') for a in active for j in own):
                continue
            with ops.lock(root/'launchers'/f'{arm}.writer.lock'):
                summary = ops.read(path)
                measured = ResourceAttemptLedger(output, run_id=definition['run_id']).summary()
                if summary['optimizer_ownership']['resources']['elapsed_seconds'] == measured['elapsed_seconds']:
                    continue  # The strict queue still checks every other field.
                ops.verify_plan(root)
                if (root/'reports/frozen/frozen_manifest.json').exists() or (
                        arm in ops.STANDALONES and (root/'launchers/standalone-barrier.json').exists()):
                    raise ValueError('Refusing to change resources already bound by a barrier/freeze')
                latest = max(own, key=lambda j: j['attempt_id'])
                worker_path = root/'launchers'/f"worker-{arm}-{latest['attempt_id']}.json"
                worker = ops.read(worker_path)
                if (worker.get('status') != 'completed' or worker.get('returncode') != 0
                        or worker['job_id'] != latest['job_id']):
                    raise ValueError('Successful matching worker required before resource reconciliation')
                accounting = ops.command(['sacct', '-X', '--noheader', '--parsable2',
                    '--jobs='+latest['job_id'], '--format=JobIDRaw,State,ExitCode,ElapsedRaw,NodeList'])
                rows = [line.split('|') for line in accounting.splitlines()]
                if len(rows) != 1 or rows[0][:3] != [latest['job_id'], 'COMPLETED', '0:0']:
                    raise ValueError('Actual successful scheduler terminal required')
                original = path.read_bytes()
                before_hash = ops.digest(path)
                ledger_path = output/'resource_attempts.json'
                ledger_hash = ops.digest(ledger_path)
                attempts = ops.read(ledger_path)['attempts']
                final_attempt = attempts.get(worker['process_uuid'], {})
                if (final_attempt.get('status') != 'completed'
                        or final_attempt.get('slurm_job_id') != latest['job_id']
                        or final_attempt.get('launch_attempt_id') != latest['attempt_id']):
                    raise ValueError('Final ledger observation does not identify the successful worker')
                candidate = reconciled_summary(summary, measured)
                archive = root/'launchers/resource-reconciliations'/f'{arm}-{before_hash}'
                archive.mkdir(parents=True, exist_ok=True)
                backup = archive/'original-run_summary.json'
                if backup.exists():
                    if backup.read_bytes() != original:
                        raise ValueError('Existing original summary backup differs')
                else:
                    with backup.open('xb') as stream:
                        stream.write(original)
                # Validate every terminal/checkpoint/evaluation/trace/clipping
                # field on a staged view before publishing even these five cells.
                with tempfile.TemporaryDirectory(prefix='candidate-', dir=archive) as scratch:
                    stage = Path(scratch)
                    for source in output.iterdir():
                        if source.name != 'run_summary.json':
                            (stage/source.name).symlink_to(source, target_is_directory=source.is_dir())
                    ops.save(stage/'run_summary.json', candidate)
                    inspected = ops.campaign._inspect_terminal_run(stage, definition,
                        manifest['expected_traces'][arm], allow_partial=False)
                    ops.campaign._check_sources(inspected['sources'])
                    after_hash = ops.digest(stage/'run_summary.json')
                if ops.digest(path) != before_hash or ops.digest(ledger_path) != ledger_hash:
                    raise ValueError('Terminal resource evidence changed during reconciliation')
                receipt = dict(status='publishing', reason='Second completed resource observation in trainer finally after summary publication',
                    arm_id=arm, run_id=definition['run_id'], job_id=latest['job_id'],
                    original_sha256=before_hash, corrected_sha256=after_hash,
                    ledger_sha256=ledger_hash, worker_sha256=ops.digest(worker_path),
                    checkpoint_sha256=inspected['checkpoint_sha256'],
                    terminal_content_hash=inspected['terminal_content_hash'],
                    old_elapsed_seconds=summary['resource_summary']['elapsed_seconds'],
                    new_elapsed_seconds=measured['elapsed_seconds'],
                    accounting=accounting, helper_sha256=ops.digest(__file__), time=time.time())
                ops.save(archive/'receipt.json', ops.sealed(receipt))
                ops.save(path, candidate)
                try:
                    ops.campaign.inspect_selected_terminals(root/'campaign/campaign_manifest.json', [arm])
                except BaseException:
                    ops.save(path, summary)
                    receipt['status'] = 'rolled_back'
                    ops.save(archive/'receipt.json', ops.sealed(receipt))
                    raise
                receipt['status'] = 'complete'
                ops.save(archive/'receipt.json', ops.sealed(receipt))
                changed.append(receipt)
                print(f"Reconciled {arm} elapsed seconds {receipt['old_elapsed_seconds']} -> {receipt['new_elapsed_seconds']}; strict terminal passed", flush=True)
    return changed


if __name__ == '__main__':
    import sys
    campaign_root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(campaign_root/'source'))
    from scripts import run_tinystories_matformer_widths as snapshot_ops
    reconcile(campaign_root, snapshot_ops)
