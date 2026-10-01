"""Four-only queue admission and 32-row terminal reporting contract."""
import json
from pathlib import Path

import pytest

from scripts import run_tinystories_c4_separate_corrections as queue
from scripts import report_tinystories_c4_separate_corrections as report
from src.utils.config import ConfigError


def test_exactly_four_fresh_arms_idempotent_submission(tmp_path, monkeypatch):
    (tmp_path / 'diagnostics').mkdir()
    queue.save(tmp_path / 'diagnostics/cpu-gate.json', dict(status='passed', source_manifest_sha256='source'))
    monkeypatch.setattr(queue, 'check_gate', lambda root: dict(job_id='7'))
    monkeypatch.setattr(queue, 'digest', lambda path: 'source')
    monkeypatch.setattr(queue, 'admission', lambda root: (dict(max_running=2, max_submitted=4), []))
    monkeypatch.setattr(queue, 'resolve_run_config', lambda path, **kw: {'run':{'output_dir':str(tmp_path / 'runs' / Path(path).stem)}})
    calls=[]
    def sbatch(root, name, entry, label, **kwargs):
        calls.append((name, entry, label)); return str(100 + len(calls))
    monkeypatch.setattr(queue, 'sbatch', sbatch)
    queue.submit(tmp_path); queue.submit(tmp_path)
    assert len(calls) == 4
    intents=queue.read(tmp_path / 'diagnostics/production-intents.json')
    assert [r['arm'] for r in intents['jobs']] == list(queue.ARMS)
    assert all('c4-separate-' in name and 'worker' in entry for name, entry, _ in calls)


def test_failed_gate_prevents_submission(tmp_path, monkeypatch):
    def failed(root): raise ConfigError('failed CUDA gate')
    monkeypatch.setattr(queue, 'check_gate', failed)
    with pytest.raises(ConfigError): queue.submit(tmp_path)
    assert not (tmp_path / 'diagnostics/production-intents.json').exists()


def test_report_exact_rows_differences_labels_and_missing_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'sha', lambda path: 'sha')
    calls=[]
    def collect(root, grid, reference, widths, **kwargs):
        offset = .1 if kwargs.get('kind') == 'GMC' else .2
        def points(add):
            return [dict(width=w, non_embedding_parameters=100*(i+1), loss=2.+add, perplexity=8.+add, update=348528, evaluation_role='ordinary_validation', validation_manifest_hash=report.MANIFEST) for i,w in enumerate(widths)]
        return points(offset), points(0), [], [], {'sha':'source'}
    monkeypatch.setattr(report, 'collect', collect)
    monkeypatch.setattr(report, 'plot', lambda output, *args: calls.append(args))
    monkeypatch.setattr('sys.argv', ['report', '--root', str(tmp_path)])
    report.main()
    endpoints=json.loads((tmp_path/'report/endpoints.json').read_text())
    differences=json.loads((tmp_path/'report/differences.json').read_text())
    assert len(endpoints)==32 and len(differences)==32 and len(calls)==2
    assert len({(r['grid'],r['arm'],r['width']) for r in endpoints})==32
    assert sum(r['reference']=='standalone' for r in differences)==16
    assert sum(r['reference']=='uncorrected-C4' for r in differences)==16
    # A missing terminal fails collection before publishing any report outputs.
    missing=tmp_path/'missing'; missing.mkdir()
    def absent(*args, **kwargs): raise FileNotFoundError('terminal missing')
    monkeypatch.setattr(report, 'collect', absent)
    monkeypatch.setattr('sys.argv', ['report', '--root', str(missing)])
    with pytest.raises(FileNotFoundError): report.main()
    assert not (missing/'report').exists()
