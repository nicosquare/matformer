"""Policy comparison fixtures retain strict terminal readers and never read holdout."""
import csv
import json
from pathlib import Path
import pytest
import yaml
from src.evaluation import optimizer_ownership as campaign
from src.utils.config import ConfigError
from test_optimizer_ownership_campaign import audited_inputs
from test_optimizer_ownership_reporting import terminal_campaign


def test_fixed_freeze_and_full_comparison(tmp_path, audited_inputs, monkeypatch):
    original_dir,current_dir=tmp_path/'original',tmp_path/'current'
    original_dir.mkdir(); current_dir.mkdir()
    old_manifest,old_runs=terminal_campaign.__wrapped__(original_dir,audited_inputs,monkeypatch)
    recipe=yaml.safe_load(Path('configs/controlled_exps/tinystories_instruct_inverse_membership.yaml').read_text())
    manifest,runs=terminal_campaign.__wrapped__(current_dir,(recipe,*audited_inputs[1:]),monkeypatch)
    campaign.freeze_campaign(campaign_manifest=old_manifest,run_root=old_runs,output_dir=original_dir/'frozen')
    campaign.freeze_campaign(campaign_manifest=manifest,run_root=runs,output_dir=current_dir/'frozen')
    frozen=current_dir/'frozen/frozen_manifest.json'
    new=campaign.report_campaign(manifest=frozen,output_dir=current_dir/'report')
    assert len(new['endpoints'])==20
    assert all(r['sampling_policy']=='fixed_inverse_membership' for r in new['endpoints'])
    assert len(campaign.endpoint_figure(new['endpoints'],metric='loss',partial=False).axes[0].lines)==5
    report=campaign.report_inverse_membership_comparison(manifest=frozen,reference_manifest=original_dir/'frozen/frozen_manifest.json',output_dir=tmp_path/'comparison')
    assert len(report['endpoints'])==44 and len(report['comparisons'])==20
    assert report['paired_epoch_traces_verified'] and not report['holdout_evaluated']
    assert len(report['figures'])==6 and all(Path(p).stat().st_size>100 for p in report['figures'])
    assert len(report['progress_coverage'])==10
    assert all(v['last_update']==348528 for arm in report['progress_coverage'].values() for v in arm.values())
    with (tmp_path/'comparison/optimizer_ownership_endpoints.csv').open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==44
    fig=campaign.endpoint_figure(report['endpoints'],metric='loss',partial=False)
    ax=fig.axes[0]
    assert len(ax.lines)==14
    assert ax.get_legend_handles_labels()[1].count('Standalone')==1
    assert {line.get_linestyle() for line in ax.lines[:10]}=={'-','--'}
    assert all(line.get_marker()=='^' and line.get_color()=='#8B4513' for line in ax.lines[-4:])
    assert len(report['exposure_comparisons'])==5
    (runs/'C3-IM/metrics.csv').write_text('damaged')
    with pytest.raises(ConfigError):
        campaign.report_inverse_membership_comparison(manifest=frozen,reference_manifest=original_dir/'frozen/frozen_manifest.json',output_dir=tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
