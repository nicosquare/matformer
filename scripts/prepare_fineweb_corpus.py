#!/usr/bin/env python3
"""Prepare the reusable full-source FineWeb sample-100BT packed corpus."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from datasets import load_dataset
from transformers import AutoTokenizer

from src.training.packed_corpus import (
    iter_streaming_documents_with_ordered_prefetch,
    load_existing_corpus_if_matching,
    preparation_work_dir,
    prepare_packed_corpus,
)
from src.training.fineweb_tokenizer import load_tokenizer_manifest


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--prepared-tokenizer-dir",
        required=True,
        help="Verified immutable tokenizer directory containing tokenizer_manifest.json",
    )
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset-config", default="sample-100BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument(
        "--shard-token-capacity",
        type=int,
        default=256 * 1024 * 1024,
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=100_000,
        help="FineWeb streaming shuffle buffer; the resulting order is seed-pinned",
    )
    parser.add_argument(
        "--tokenization-workers",
        type=positive_int,
        default=1,
        help=(
            "Ordered tokenizer worker threads; increase for faster future corpus "
            "builds without changing packed-token order"
        ),
    )
    parser.add_argument(
        "--source-read-workers",
        type=positive_int,
        default=None,
        help=(
            "Ordered streaming source-reader threads. Defaults to "
            "--tokenization-workers and preserves the exact shuffled order."
        ),
    )
    parser.add_argument(
        "--progress-interval-seconds",
        type=positive_float,
        default=60.0,
        help="Emit live preparation progress to stderr at this cadence",
    )
    return parser.parse_args(argv)


def _summary(
    args: argparse.Namespace,
    manifest: dict,
    tokenizer_manifest: dict,
    *,
    status: str,
    resumed_document_count: int,
    elapsed_seconds: float,
) -> dict:
    source_document_count = int(manifest["source"]["shuffled_document_count"])
    training = manifest["roles"]["optimizer_training"]
    return {
        "status": status,
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "corpus_hash": manifest["corpus_hash"],
        "training_token_count": training["token_count"],
        "training_sequence_count": training["sequence_count"],
        "source_document_count": source_document_count,
        "training_document_count": training["source_document_count"],
        "shard_count": sum(
            len(role["shards"]) for role in manifest["roles"].values()
        ),
        "resumed_document_count": int(resumed_document_count),
        "elapsed_seconds": elapsed_seconds,
        "documents_per_second": (
            (source_document_count - resumed_document_count) / elapsed_seconds
            if elapsed_seconds > 0 and status != "already_prepared"
            else None
        ),
        "role_manifest_hashes": manifest["role_manifest_hashes"],
        "tokenizer_manifest_hash": tokenizer_manifest["manifest_hash"],
        "tokenization_workers": args.tokenization_workers,
        "source_read_workers": args.source_read_workers,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.source_read_workers is None:
        args.source_read_workers = args.tokenization_workers
    started_at = time.monotonic()
    tokenizer_manifest = load_tokenizer_manifest(
        args.prepared_tokenizer_dir, verify_files=True
    )
    existing = load_existing_corpus_if_matching(
        args.output_dir,
        tokenizer_manifest=tokenizer_manifest,
        tokenizer_name=tokenizer_manifest["tokenizer_name"],
        tokenizer_revision=tokenizer_manifest["manifest_hash"],
        source_dataset=args.dataset,
        source_config=args.dataset_config,
        source_split=args.split,
        text_column=args.text_column,
        data_seed=args.data_seed,
        context_length=args.context_length,
        shard_token_capacity=args.shard_token_capacity,
        shuffle_buffer_size=args.shuffle_buffer_size,
    )
    if existing is not None:
        print(
            json.dumps(
                _summary(
                    args,
                    existing,
                    tokenizer_manifest,
                    status="already_prepared",
                    resumed_document_count=0,
                    elapsed_seconds=time.monotonic() - started_at,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    work_dir = preparation_work_dir(args.output_dir)
    resumed_document_count = 0
    resumed_token_count = 0
    was_resumable = work_dir.exists()
    if was_resumable:
        try:
            checkpoint = json.loads(
                (work_dir / "preparation_progress.json").read_text(encoding="utf-8")
            )
            resumed_document_count = int(
                checkpoint.get("committed_shuffled_document_count", 0)
            )
            training_state = checkpoint.get("role_states", {}).get(
                "optimizer_training", {}
            )
            training_manifest = checkpoint.get("role_manifests", {}).get(
                "optimizer_training", {}
            )
            resumed_token_count = int(
                training_state.get(
                    "token_count", training_manifest.get("token_count", 0)
                )
            )
        except (OSError, ValueError, json.JSONDecodeError):
            # The preparation implementation provides the authoritative corrupt
            # checkpoint error without modifying the partial artifact.
            pass
    tokenizer = AutoTokenizer.from_pretrained(
        args.prepared_tokenizer_dir,
        local_files_only=True,
    )
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        streaming=True,
    )
    fingerprint = getattr(dataset, "_fingerprint", None)
    if not fingerprint:
        fingerprint = (
            f"{args.dataset}:{args.dataset_config}:{args.split}:"
            f"streaming-shuffle-{args.data_seed}-{args.shuffle_buffer_size}"
        )
    dataset = dataset.shuffle(
        seed=args.data_seed,
        buffer_size=args.shuffle_buffer_size,
    )
    dataset = iter_streaming_documents_with_ordered_prefetch(
        dataset,
        workers=args.source_read_workers,
    )

    def report_progress(progress: dict) -> None:
        elapsed = max(time.monotonic() - started_at, 1e-9)
        if progress["event"] in {"resume_replay", "resume_replay_completed"}:
            replayed = int(progress["replayed_document_count"])
            replay_target = int(progress["replay_target_document_count"])
            replay_rate = float(progress["replay_documents_per_second"])
            replay_remaining = max(replay_target - replayed, 0)
            replay_eta = replay_remaining / replay_rate if replay_rate > 0 else 0.0
            replay_percent = (
                100.0 * replayed / replay_target if replay_target else 100.0
            )
            print(
                "[corpus-progress] "
                f"event={progress['event']} phase=resume_replay "
                f"replayed_documents={replayed:,}/{replay_target:,} "
                f"replay_percent={replay_percent:.2f} "
                f"elapsed_seconds={progress['replay_elapsed_seconds']:.1f} "
                f"documents_per_second={replay_rate:.1f} "
                f"eta_seconds={replay_eta:.1f} "
                f"source_read_workers={progress['source_read_workers']}",
                file=sys.stderr,
                flush=True,
            )
            return
        documents_done = max(
            int(progress["committed_document_count"]) - resumed_document_count,
            0,
        )
        tokens_done = max(
            int(progress["optimizer_token_count"]) - resumed_token_count,
            0,
        )
        print(
            "[corpus-progress] "
            f"event={progress['event']} phase={progress['phase']} "
            f"documents={progress['committed_document_count']:,} "
            f"optimizer_tokens={progress['optimizer_token_count']:,} "
            f"completed_shards={progress['completed_optimizer_shard_count']} "
            f"current_shard={progress['current_optimizer_shard_index']:05d} "
            f"shard_offset_tokens={progress['current_optimizer_shard_offset']:,} "
            f"elapsed_seconds={elapsed:.1f} "
            f"documents_per_second={documents_done / elapsed:.1f} "
            f"tokens_per_second={tokens_done / elapsed:.1f} "
            f"source_read_workers={progress['source_read_workers']} "
            f"tokenization_workers={progress['tokenization_workers']}",
            file=sys.stderr,
            flush=True,
        )

    print(
        "[corpus-progress] "
        f"event={'resuming' if was_resumable else 'starting'} "
        f"documents={resumed_document_count:,} "
        f"optimizer_tokens={resumed_token_count:,} "
        f"interval_seconds={args.progress_interval_seconds:g} "
        f"source_read_workers={args.source_read_workers} "
        f"tokenization_workers={args.tokenization_workers}",
        file=sys.stderr,
        flush=True,
    )
    manifest = prepare_packed_corpus(
        dataset,
        tokenizer,
        Path(args.output_dir),
        tokenizer_name=tokenizer_manifest["tokenizer_name"],
        tokenizer_revision=tokenizer_manifest["manifest_hash"],
        source_dataset=args.dataset,
        source_config=args.dataset_config,
        source_split=args.split,
        source_fingerprint=str(fingerprint),
        text_column=args.text_column,
        data_seed=args.data_seed,
        context_length=args.context_length,
        shard_token_capacity=args.shard_token_capacity,
        shuffle_buffer_size=args.shuffle_buffer_size,
        tokenization_workers=args.tokenization_workers,
        source_read_workers=args.source_read_workers,
        tokenizer_manifest=tokenizer_manifest,
        progress_callback=report_progress,
        progress_interval_seconds=args.progress_interval_seconds,
    )
    print(
        json.dumps(
            _summary(
                args,
                manifest,
                tokenizer_manifest,
                status="resumed_and_prepared" if was_resumable else "prepared",
                resumed_document_count=resumed_document_count,
                elapsed_seconds=time.monotonic() - started_at,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
