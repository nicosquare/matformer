#!/usr/bin/env python3
"""Read-only terminal g1000 loss check on a fixed optimizer-training panel."""

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.validation import evaluate_validation_loss
from src.training.data import collate_language_model_batch
from src.training.modeling import build_model
from src.training.packed_corpus import PackedMMapDataset, load_corpus_manifest, sha256_file
from src.utils.reproducibility import stable_hash


BASE = Path('/nfs-stor/ivo.navarrete/results/elasticnn')
RUNS = {
    'standalone': BASE / 'optimizer-ownership-v1/runs/ST-g1000',
    'linear': BASE / 'optimizer-ownership-coverage-balanced-v1/runs/tinystories-coverage-balanced-v1-S1-s42',
    'geometric': BASE / 'optimizer-ownership-coverage-balanced-geometric-v1/runs/tinystories-coverage-balanced-geometric-v1-S1-s42',
}
PANEL_SIZE = 256


def read(path):
    return json.loads(path.read_text())


def terminal_validation(root, config, checkpoint_hash):
    sidecar = root / 'terminal_validation_results.json'
    if sidecar.exists():
        record = read(sidecar)
        assert record['content_hash'] == stable_hash({k: v for k, v in record.items() if k != 'content_hash'})
        assert record['checkpoint_sha256'] == checkpoint_hash
        assert record['global_step'] == config['training']['max_steps']
        assert record['validation_manifest_hash'] == config['validation_manifest_hash']
        assert record['evaluation_role'] == 'ordinary_validation'
        row = next(r for r in record['endpoints'] if r['width'] == 'g1000')
        return row['loss'], row['evaluation_examples'], row['evaluation_target_tokens'], str(sidecar)
    last = None
    with (root / 'metrics.csv').open(newline='') as source:
        for row in csv.DictReader(source):
            if row['split'] == 'validation' and row['granularity'] == 'g1000':
                last = row
    assert last is not None
    assert int(last['step']) == config['training']['max_steps']
    assert last['validation_manifest_hash'] == config['validation_manifest_hash']
    assert last['validation_loss_aggregation'] == 'target_token_weighted_causal_shift_float64'
    return float(last['loss']), int(last['evaluation_examples']), int(last['evaluation_target_tokens']), str(root / 'metrics.csv')


def main(output_dir):
    assert torch.cuda.is_available(), 'A CUDA GPU is required for the saved bf16 evaluation protocol'
    device = torch.device('cuda:0')
    configs = {name: read(root / 'config.json') for name, root in RUNS.items()}
    first = configs['standalone']
    shared_keys = ('corpus_hash', 'corpus_permutation_hash', 'optimizer_training_manifest_hash', 'validation_manifest_hash', 'validation_loss_aggregation')
    for key in shared_keys:
        assert len({c[key] for c in configs.values()}) == 1, key
    assert first['validation_loss_aggregation'] == 'target_token_weighted_causal_shift_float64'
    prepared = Path(first['dataset']['prepared_corpus_dir'])
    assert all(Path(c['dataset']['prepared_corpus_dir']) == prepared for c in configs.values())
    corpus = load_corpus_manifest(prepared, verify_shards=False)
    assert corpus['corpus_hash'] == first['corpus_hash']
    assert corpus['roles']['optimizer_training']['manifest_hash'] == first['optimizer_training_manifest_hash']
    assert corpus['roles']['ordinary_validation']['manifest_hash'] == first['validation_manifest_hash']
    permutation_path = prepared / corpus['training_order']['path']
    assert sha256_file(permutation_path) == first['corpus_permutation_hash']
    permutation = np.memmap(permutation_path, mode='r', dtype='<u8')
    epoch_size = first['dataset']['optimizer_iteration']['aligned_epoch_samples']
    assert all(c['dataset']['optimizer_iteration']['aligned_epoch_samples'] == epoch_size for c in configs.values())
    assert all(c['dataset']['optimizer_iteration']['fixed_epoch_set_hash'] == first['dataset']['optimizer_iteration']['fixed_epoch_set_hash'] for c in configs.values())
    assert all(c['model']['granularities'] == ['g1000'] or 'g1000' in c['model']['granularities'] for c in configs.values())
    shape_keys = ('d_model', 'num_layers', 'num_attention_heads', 'intermediate_size', 'context_length', 'vocab_size')
    for key in shape_keys:
        assert len({c['model'][key] for c in configs.values()}) == 1, key
    # Every run trains on this fixed prefix; its first IDs are independent of later batch ordering.
    ids = [int(x) for x in permutation[:PANEL_SIZE]]
    assert len(set(ids)) == PANEL_SIZE and all(0 <= x < len(permutation) for x in ids)
    selection = {'rule': f'first {PANEL_SIZE} packed sequence IDs from the stored optimizer-training permutation prefix, in stored order',
                 'packed_sequence_ids': ids, 'permutation_sha256': first['corpus_permutation_hash'],
                 'fixed_epoch_set_hash': first['dataset']['optimizer_iteration']['fixed_epoch_set_hash']}
    selection['sha256'] = hashlib.sha256(json.dumps(selection, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    dataset = PackedMMapDataset(prepared, 'optimizer_training')
    loader = DataLoader(Subset(dataset, ids), batch_size=64, shuffle=False, collate_fn=collate_language_model_batch)
    rows = []
    for name, root in RUNS.items():
        config = configs[name]
        checkpoint_path = root / 'checkpoints/latest.pt'
        checkpoint_hash = sha256_file(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        assert checkpoint['checkpoint_kind'] == 'resumable_training'
        assert checkpoint['run_id'] == config['run']['run_id']
        assert checkpoint['step'] == config['training']['max_steps']
        assert checkpoint['tokens_seen'] == config['training']['token_budget']
        assert checkpoint['corpus_hash'] == config['corpus_hash']
        assert checkpoint['granularities'] == config['model']['granularities']
        assert checkpoint['optimizer_iteration']['fixed_epoch_set_hash'] == selection['fixed_epoch_set_hash']
        summary = read(root / 'run_summary.json')
        assert summary['status'] == 'completed' and summary['committed_optimizer_steps'] == checkpoint['step']
        assert read(root / 'training_manifest.json')['manifest_hash'] == config['optimizer_training_manifest_hash']
        assert read(root / 'validation_manifest.json')['manifest_hash'] == config['validation_manifest_hash']
        validation_loss, validation_examples, validation_targets, validation_source = terminal_validation(root, config, checkpoint_hash)
        model = build_model(config)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        del checkpoint
        model.to(device)
        measured = evaluate_validation_loss(model, loader, device, granularity='g1000', config=config)
        assert measured['evaluation_examples'] == PANEL_SIZE
        assert measured['evaluation_target_tokens'] == PANEL_SIZE * (config['model']['context_length'] - 1)
        assert sha256_file(checkpoint_path) == checkpoint_hash
        rows.append({'model': name, 'run_id': config['run']['run_id'], 'checkpoint_sha256': checkpoint_hash,
                     'checkpoint_path': str(checkpoint_path), 'step': summary['committed_optimizer_steps'],
                     'train_subset_loss': measured['loss'], 'train_subset_examples': measured['evaluation_examples'],
                     'train_subset_target_tokens': measured['evaluation_target_tokens'],
                     'validation_loss': validation_loss, 'validation_examples': validation_examples,
                     'validation_target_tokens': validation_targets, 'validation_source': validation_source})
        del model
        torch.cuda.empty_cache()
    baseline = rows[0]
    for row in rows:
        row['train_minus_standalone'] = row['train_subset_loss'] - baseline['train_subset_loss']
        row['validation_minus_standalone'] = row['validation_loss'] - baseline['validation_loss']
        row['validation_minus_train'] = row['validation_loss'] - row['train_subset_loss']
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {'selection': selection, 'rows': rows, 'shared_identities': {k: first[k] for k in shared_keys},
              'model_shape': {k: first['model'][k] for k in shape_keys}, 'device': torch.cuda.get_device_name(device),
              'precision': 'CUDA bf16 autocast, matching saved evaluation protocol', 'holdout_evaluated': False,
              'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
    (output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    with (output_dir / 'losses.csv').open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    labels = [r['model'] for r in rows]
    colors = ['#39475b', '#25857a', '#ba6b40']
    for axis, field, title in [(axes[0], 'train_subset_loss', 'Optimizer-training subset'), (axes[1], 'validation_loss', 'Ordinary validation')]:
        bars = axis.bar(labels, [r[field] for r in rows], color=colors)
        axis.set_title(title); axis.set_ylabel('Causal LM loss / target token')
        axis.set_ylim(0, max(r[field] for r in rows) * 1.15)
        for bar, row in zip(bars, rows):
            axis.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{row[field]:.4f}', ha='center', fontsize=9)
    fig.suptitle('Terminal g1000 loss: same training examples, saved validation endpoints')
    fig.savefig(output_dir / 'loss_comparison.png', dpi=180)
    plt.close(fig)
    print(json.dumps({'output_dir': str(output_dir), 'rows': rows, 'selection_sha256': selection['sha256']}, indent=2), flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: check_tinystories_g1000_train_validation_gap.py OUTPUT_DIR')
    main(Path(sys.argv[1]))
