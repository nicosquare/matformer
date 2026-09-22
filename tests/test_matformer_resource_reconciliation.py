import copy

import pytest

from scripts.reconcile_matformer_terminal_resources import reconciled_summary


def example():
    measured = dict(elapsed_seconds=10., attempted_steps=4, measurement_complete=True,
                    peak_allocated_bytes=20, peak_reserved_bytes=30, attempt_count=1)
    summary = dict(status='completed', resource_summary=copy.deepcopy(measured),
        training_wall_time_seconds=10., scientific={'checkpoint':'unchanged'},
        optimizer_ownership=dict(tokens_seen=32, packed_tokens_per_update=8,
            resources={**measured, 'useful_committed_tokens_per_second':3.2,
                       'attempted_tokens_per_second':3.2}))
    measured['elapsed_seconds'] = 10.01
    return summary, measured


def test_only_elapsed_and_derived_rates_change_without_mutating_inputs():
    summary, measured = example()
    original = copy.deepcopy(summary)
    result = reconciled_summary(summary, measured)
    assert summary == original
    assert result['resource_summary'] == measured
    assert result['optimizer_ownership']['resources']['attempted_tokens_per_second'] == 32/10.01
    result['resource_summary']['elapsed_seconds'] = 10.
    result['training_wall_time_seconds'] = 10.
    result['optimizer_ownership']['resources'].update(elapsed_seconds=10.,
        useful_committed_tokens_per_second=3.2, attempted_tokens_per_second=3.2)
    assert result == original
    corrected = reconciled_summary(summary, measured)
    assert reconciled_summary(corrected, measured) == corrected


@pytest.mark.parametrize('field,value', [('attempted_steps',5), ('peak_allocated_bytes',21),
    ('peak_reserved_bytes',31), ('attempt_count',2), ('measurement_complete',False),
    ('elapsed_seconds',9.), ('elapsed_seconds',float('nan')), ('elapsed_seconds',True)])
def test_rejects_non_timing_or_invalid_measurement(field, value):
    summary, measured = example()
    measured[field] = value
    with pytest.raises(ValueError): reconciled_summary(summary, measured)


@pytest.mark.parametrize('damage', ['nested', 'wall', 'throughput', 'failed', 'schema'])
def test_rejects_inconsistent_original_evidence(damage):
    summary, measured = example()
    if damage == 'nested': summary['optimizer_ownership']['resources']['attempted_steps'] = 3
    elif damage == 'wall': summary['training_wall_time_seconds'] = 9.
    elif damage == 'throughput': summary['optimizer_ownership']['resources']['attempted_tokens_per_second'] = 99.
    elif damage == 'failed': summary['status'] = 'failed'
    else: measured['unknown'] = 0
    with pytest.raises(ValueError): reconciled_summary(summary, measured)
