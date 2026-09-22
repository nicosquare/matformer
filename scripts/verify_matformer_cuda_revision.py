#!/usr/bin/env python3
"""Acceptance evidence for a launcher revision over an immutable scientific snapshot."""
import argparse
from pathlib import Path
import os
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_tinystories_matformer_widths as ops
from scripts import preflight_tinystories_matformer_widths as probes
from scripts.train_cuda_required import required_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('cpu', 'gpu'), required=True)
    parser.add_argument('--campaign-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    assert REPO == root/'launchers/cuda-required-v1'
    ops.verify_plan(root)
    bindings = ops.bindings(root)
    files = ops.source_files(REPO)
    directory = root/'diagnostics/cuda-required-v1'
    output = directory/f'{args.mode}-{time.time_ns()}'
    output.mkdir(parents=True)
    gate = dict(mode=args.mode, status='failed', bindings=bindings, execution_files=files,
                checks=[], holdout_evaluated=False)
    try:
        if args.mode == 'gpu':
            import torch
            from src.training import run
            if not os.environ.get('SLURM_JOB_ID'):
                raise ops.ConfigError('GPU acceptance requires sbatch')
            cpu = ops.read(directory/'cpu-gate.json')
            ops.check_seal(cpu, 'execution CPU gate')
            if cpu['status'] != 'passed' or cpu['execution_files'] != files or cpu['bindings'] != bindings:
                raise ops.ConfigError('Execution CPU gate is stale')
            limits, active = ops.scheduler_limits(), ops.queue_state()
            if (len(active) > limits['max_submitted']
                    or sum(r['state'] == 'RUNNING' for r in active) > limits['max_running']
                    or any(r.get('qos') != ops.QOS for r in active)):
                raise ops.ConfigError('Live diagnostic limits exceeded')
            allocation = ops.command(['scontrol', 'show', 'job', os.environ['SLURM_JOB_ID'], '-o'])
            original = run.run_training
            def guarded(config):
                return required_training(config, training_function=original)
            run.run_training = guarded
            try:
                measurements = probes.gpu_real_shapes(root, output)
            finally:
                run.run_training = original
            gate.update(job_id=os.environ['SLURM_JOB_ID'], hardware=torch.cuda.get_device_name(),
                        allocation=allocation, cpu_gate_hash=cpu['content_hash'],
                        real_shape_arms=[r['arm_id'] for r in measurements], probes=measurements)
            gate['checks'].append(dict(returncode=0, tests=9, failures=0, errors=0, skipped=0,
                command=[sys.executable, __file__, '--mode', 'gpu', '--campaign-root', str(root)],
                artifacts=[dict(path=str(output/'real-shapes.json'), sha256=ops.digest(output/'real-shapes.json')),
                           *[a for probe in measurements for a in probe['artifacts']]]))
        else:
            probes.CPU_SUITES += ('test_cuda_required_training.py',)
        check = probes.pytest_check(REPO, output, gpu=args.mode == 'gpu')
        gate['checks'].append(check)
        if (check['returncode'] or not check['tests'] or check['failures'] or check['errors']
                or (args.mode == 'gpu' and check['skipped'])):
            raise ops.ConfigError(f'Acceptance failed: {output}/pytest.log')
        if ops.bindings(root) != bindings or ops.source_files(REPO) != files:
            raise ops.ConfigError('Acceptance inputs changed')
        gate['status'] = 'passed'
    except BaseException as error:
        gate['error'] = str(error)
        raise
    finally:
        ops.save(directory/f'{args.mode}-gate.json', ops.sealed(gate))
    print(f'{args.mode} acceptance passed: {directory}', flush=True)


if __name__ == '__main__':
    main()
