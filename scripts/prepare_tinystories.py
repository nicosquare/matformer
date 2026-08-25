#!/usr/bin/env python3
"""Prepare the controlled TinyStories tokenizer and packed-mmap corpus."""

# ruff: noqa: E402  # Add the repository root before importing src.

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from datasets import load_dataset
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.packed_corpus import (
    load_existing_corpus_if_matching,
    preparation_work_dir,
    prepare_packed_corpus,
)
from src.training.tinystories import (
    TINYSTORIES_CONTEXT_LENGTH,
    TINYSTORIES_DATASET_CONFIG,
    TINYSTORIES_DATASET_NAME,
    TINYSTORIES_DATASET_REVISION,
    TINYSTORIES_DATASET_SPLIT,
    TINYSTORIES_OPTIMIZER_TOKEN_COUNT,
    TINYSTORIES_ROLE_COUNTS,
    TINYSTORIES_TOKENIZER_STORY_COUNT,
    corpus_source_contract,
    dataset_split_fingerprint,
    load_existing_tinystories_tokenizer_if_matching,
    role_ordered_stories,
    train_tinystories_tokenizer,
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--dataset", default=TINYSTORIES_DATASET_NAME)
    parser.add_argument(
        "--dataset-revision",
        default=TINYSTORIES_DATASET_REVISION,
        help="Pinned Hugging Face dataset revision",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        default=None,
        help="Optional datasets cache; otherwise datasets honors HF_HOME",
    )
    parser.add_argument(
        "--tokenizer-story-count",
        type=positive_int,
        default=TINYSTORIES_TOKENIZER_STORY_COUNT,
    )
    parser.add_argument("--max-chunk-bytes", type=positive_int, default=4_096)
    parser.add_argument(
        "--optimizer-token-count",
        type=positive_int,
        default=TINYSTORIES_OPTIMIZER_TOKEN_COUNT,
    )
    parser.add_argument(
        "--shard-token-capacity",
        type=positive_int,
        default=8 * 1024 * 1024,
    )
    parser.add_argument("--tokenization-workers", type=positive_int, default=1)
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=60.0,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.optimizer_token_count % TINYSTORIES_CONTEXT_LENGTH:
        raise SystemExit(
            "--optimizer-token-count must be divisible by the 128-token context"
        )
    if args.progress_interval_seconds <= 0:
        raise SystemExit("--progress-interval-seconds must be positive")

    started_at = time.monotonic()
    print(
        "[tinystories-progress] "
        f"event=dataset_loading dataset={args.dataset} "
        f"revision={args.dataset_revision} "
        f"cache_dir={args.dataset_cache_dir or 'datasets_default'}",
        file=sys.stderr,
        flush=True,
    )
    dataset = load_dataset(
        args.dataset,
        revision=args.dataset_revision,
        cache_dir=args.dataset_cache_dir,
    )
    missing_splits = {"train", "validation"} - set(dataset)
    if missing_splits:
        raise SystemExit(
            "TinyStories dataset is missing required splits: "
            + ", ".join(sorted(missing_splits))
        )
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]
    print(
        "[tinystories-progress] "
        f"event=dataset_loaded train_stories={len(train_dataset):,} "
        f"validation_stories={len(validation_dataset):,} "
        f"elapsed_seconds={time.monotonic() - started_at:.1f}",
        file=sys.stderr,
        flush=True,
    )
    train_fingerprint = dataset_split_fingerprint(
        train_dataset,
        split="train",
        dataset_name=args.dataset,
        dataset_revision=args.dataset_revision,
    )
    validation_fingerprint = dataset_split_fingerprint(
        validation_dataset,
        split="validation",
        dataset_name=args.dataset,
        dataset_revision=args.dataset_revision,
    )

    tokenizer_manifest = load_existing_tinystories_tokenizer_if_matching(
        args.tokenizer_dir,
        train_split_fingerprint=train_fingerprint,
        dataset_name=args.dataset,
        dataset_revision=args.dataset_revision,
        story_count=args.tokenizer_story_count,
        max_chunk_bytes=args.max_chunk_bytes,
    )
    tokenizer_status = "already_prepared"
    if tokenizer_manifest is None:
        tokenizer_started_at = time.monotonic()
        print(
            "[tinystories-progress] "
            f"event=tokenizer_training_started "
            f"stories={args.tokenizer_story_count:,} "
            f"tokenizer_dir={Path(args.tokenizer_dir).expanduser().resolve()}",
            file=sys.stderr,
            flush=True,
        )
        tokenizer_manifest = train_tinystories_tokenizer(
            train_dataset,
            args.tokenizer_dir,
            train_split_fingerprint=train_fingerprint,
            dataset_name=args.dataset,
            dataset_revision=args.dataset_revision,
            story_count=args.tokenizer_story_count,
            max_chunk_bytes=args.max_chunk_bytes,
        )
        tokenizer_status = "prepared"
        print(
            "[tinystories-progress] "
            f"event=tokenizer_training_completed "
            f"stories={args.tokenizer_story_count:,} "
            f"elapsed_seconds={time.monotonic() - tokenizer_started_at:.1f}",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            "[tinystories-progress] "
            f"event=tokenizer_reused "
            f"tokenizer_dir={Path(args.tokenizer_dir).expanduser().resolve()}",
            file=sys.stderr,
            flush=True,
        )

    source_fingerprint, role_sources = corpus_source_contract(
        train_split_fingerprint=train_fingerprint,
        validation_split_fingerprint=validation_fingerprint,
        train_row_count=len(train_dataset),
        validation_row_count=len(validation_dataset),
        dataset_name=args.dataset,
        dataset_revision=args.dataset_revision,
    )
    corpus_manifest = load_existing_corpus_if_matching(
        args.corpus_dir,
        tokenizer_manifest=tokenizer_manifest,
        tokenizer_name=tokenizer_manifest["tokenizer_name"],
        tokenizer_revision=tokenizer_manifest["manifest_hash"],
        source_dataset=args.dataset,
        source_config=TINYSTORIES_DATASET_CONFIG,
        source_split=TINYSTORIES_DATASET_SPLIT,
        text_column="text",
        data_seed=42,
        context_length=TINYSTORIES_CONTEXT_LENGTH,
        shard_token_capacity=args.shard_token_capacity,
        shuffle_buffer_size=0,
        reserved_role_counts=TINYSTORIES_ROLE_COUNTS,
        optimizer_token_limit=args.optimizer_token_count,
        minimum_optimizer_document_count=args.tokenizer_story_count,
        role_source_provenance=role_sources,
    )
    corpus_status = "already_prepared"
    if corpus_manifest is None:
        work_dir = preparation_work_dir(args.corpus_dir)
        resumed_document_count = 0
        resumed_token_count = 0
        was_resumable = work_dir.exists()
        if was_resumable:
            try:
                checkpoint = json.loads(
                    (work_dir / "preparation_progress.json").read_text(
                        encoding="utf-8"
                    )
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
                # The corpus builder provides the authoritative checkpoint error.
                pass

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_dir,
            local_files_only=True,
        )
        corpus_started_at = time.monotonic()

        def report_progress(progress: dict) -> None:
            elapsed = max(time.monotonic() - corpus_started_at, 1e-9)
            if progress["event"] in {
                "resume_replay",
                "resume_replay_completed",
            }:
                replayed = int(progress["replayed_document_count"])
                replay_target = int(progress["replay_target_document_count"])
                replay_rate = float(progress["replay_documents_per_second"])
                replay_remaining = max(replay_target - replayed, 0)
                replay_eta = (
                    replay_remaining / replay_rate if replay_rate > 0 else 0.0
                )
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
                int(progress["committed_document_count"])
                - resumed_document_count,
                0,
            )
            tokens_done = max(
                int(progress["optimizer_token_count"]) - resumed_token_count,
                0,
            )
            print(
                "[corpus-progress] "
                f"event={progress['event']} phase={progress['phase']} "
                f"role={progress.get('role')} "
                f"documents={progress['committed_document_count']:,} "
                f"optimizer_tokens={progress['optimizer_token_count']:,} "
                f"completed_shards={progress['completed_optimizer_shard_count']} "
                f"current_shard={progress['current_optimizer_shard_index']:05d} "
                f"shard_offset_tokens="
                f"{progress['current_optimizer_shard_offset']:,} "
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
            f"target_optimizer_tokens={args.optimizer_token_count:,} "
            f"interval_seconds={args.progress_interval_seconds:g} "
            f"source_read_workers=1 "
            f"tokenization_workers={args.tokenization_workers}",
            file=sys.stderr,
            flush=True,
        )
        corpus_manifest = prepare_packed_corpus(
            role_ordered_stories(train_dataset, validation_dataset),
            tokenizer,
            Path(args.corpus_dir),
            tokenizer_name=tokenizer_manifest["tokenizer_name"],
            tokenizer_revision=tokenizer_manifest["manifest_hash"],
            source_dataset=args.dataset,
            source_config=TINYSTORIES_DATASET_CONFIG,
            source_split=TINYSTORIES_DATASET_SPLIT,
            source_fingerprint=source_fingerprint,
            text_column="text",
            data_seed=42,
            context_length=TINYSTORIES_CONTEXT_LENGTH,
            shard_token_capacity=args.shard_token_capacity,
            shuffle_buffer_size=0,
            tokenization_workers=args.tokenization_workers,
            source_read_workers=1,
            tokenizer_manifest=tokenizer_manifest,
            reserved_role_counts=TINYSTORIES_ROLE_COUNTS,
            optimizer_token_limit=args.optimizer_token_count,
            minimum_optimizer_document_count=args.tokenizer_story_count,
            role_source_provenance=role_sources,
            progress_callback=report_progress,
            progress_interval_seconds=args.progress_interval_seconds,
        )
        corpus_status = "resumed_and_prepared" if was_resumable else "prepared"
        print(
            "[corpus-progress] "
            f"event=completed status={corpus_status} "
            f"documents={corpus_manifest['source']['shuffled_document_count']:,} "
            f"optimizer_tokens="
            f"{corpus_manifest['available_optimizer_token_count']:,} "
            f"elapsed_seconds={time.monotonic() - corpus_started_at:.1f}",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            "[corpus-progress] "
            f"event=already_prepared "
            f"corpus_dir={Path(args.corpus_dir).expanduser().resolve()} "
            f"optimizer_tokens="
            f"{corpus_manifest['available_optimizer_token_count']:,}",
            file=sys.stderr,
            flush=True,
        )

    print(
        json.dumps(
            {
                "tokenizer_status": tokenizer_status,
                "dataset": args.dataset,
                "dataset_revision": args.dataset_revision,
                "train_split_fingerprint": train_fingerprint,
                "validation_split_fingerprint": validation_fingerprint,
                "tokenizer_dir": str(Path(args.tokenizer_dir).resolve()),
                "tokenizer_manifest_hash": tokenizer_manifest["manifest_hash"],
                "corpus_status": corpus_status,
                "corpus_dir": str(Path(args.corpus_dir).resolve()),
                "corpus_hash": corpus_manifest["corpus_hash"],
                "optimizer_token_count": corpus_manifest[
                    "available_optimizer_token_count"
                ],
                "optimizer_sequence_count": corpus_manifest[
                    "available_optimizer_sequence_count"
                ],
                "reserved_role_counts": corpus_manifest["reserved_role_counts"],
                "elapsed_seconds": time.monotonic() - started_at,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
