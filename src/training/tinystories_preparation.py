"""Shared orchestration for TinyStories-family preparation commands."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class TinyStoriesPreparationSpec:
    """Dataset-specific operations used by the shared preparation lifecycle."""

    display_name: str
    progress_prefix: str
    dataset_name: str
    dataset_revision: str
    dataset_config: str
    dataset_split: str
    context_length: int
    tokenizer_document_count: int
    tokenizer_document_noun: str
    role_counts: Mapping[str, int]
    corpus_name: str | None
    train_count_label: str
    validation_count_label: str
    split_fingerprint: Callable[..., str]
    load_existing_tokenizer: Callable[..., dict[str, Any] | None]
    train_tokenizer: Callable[..., dict[str, Any]]
    corpus_source_contract: Callable[..., tuple[str, dict[str, dict[str, Any]]]]
    role_ordered_documents: Callable[..., Any]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def optimizer_token_count(value: str) -> int | None:
    if value.strip().lower() == "all":
        return None
    return positive_int(value)


def parse_preparation_args(
    argv: Sequence[str] | None,
    *,
    description: str,
    dataset_name: str,
    dataset_revision: str,
    tokenizer_document_count: int,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--dataset", default=dataset_name)
    parser.add_argument(
        "--dataset-revision",
        default=dataset_revision,
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
        default=tokenizer_document_count,
        help="Number of optimizer-eligible source documents used by the tokenizer",
    )
    parser.add_argument("--max-chunk-bytes", type=positive_int, default=4_096)
    parser.add_argument(
        "--optimizer-token-count",
        type=optimizer_token_count,
        default=None,
        metavar="TOKENS|all",
        help=(
            "Context-aligned optimizer-token cap, or 'all' to consume every "
            "optimizer-eligible source document (default: all)"
        ),
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


def run_tinystories_preparation(
    args: argparse.Namespace,
    *,
    spec: TinyStoriesPreparationSpec,
    load_dataset_fn: Callable[..., Mapping[str, Any]],
    auto_tokenizer_class: Any,
    load_existing_corpus_fn: Callable[..., dict[str, Any] | None],
    prepare_packed_corpus_fn: Callable[..., dict[str, Any]],
    preparation_work_dir_fn: Callable[[str | Path], Path],
) -> None:
    """Prepare or reuse a tokenizer and corpus under one deterministic flow."""

    if (
        args.optimizer_token_count is not None
        and args.optimizer_token_count % spec.context_length
    ):
        raise SystemExit(
            "--optimizer-token-count must be divisible by the "
            f"{spec.context_length}-token context"
        )
    if args.progress_interval_seconds <= 0:
        raise SystemExit("--progress-interval-seconds must be positive")

    started_at = time.monotonic()
    print(
        f"[{spec.progress_prefix}] "
        f"event=dataset_loading dataset={args.dataset} "
        f"revision={args.dataset_revision} "
        f"cache_dir={args.dataset_cache_dir or 'datasets_default'}",
        file=sys.stderr,
        flush=True,
    )
    dataset = load_dataset_fn(
        args.dataset,
        revision=args.dataset_revision,
        cache_dir=args.dataset_cache_dir,
    )
    missing_splits = {"train", "validation"} - set(dataset)
    if missing_splits:
        raise SystemExit(
            f"{spec.display_name} dataset is missing required splits: "
            + ", ".join(sorted(missing_splits))
        )
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]
    print(
        f"[{spec.progress_prefix}] "
        f"event=dataset_loaded {spec.train_count_label}={len(train_dataset):,} "
        f"{spec.validation_count_label}={len(validation_dataset):,} "
        f"elapsed_seconds={time.monotonic() - started_at:.1f}",
        file=sys.stderr,
        flush=True,
    )
    train_fingerprint = spec.split_fingerprint(
        train_dataset,
        split="train",
        dataset_name=args.dataset,
        dataset_revision=args.dataset_revision,
    )
    validation_fingerprint = spec.split_fingerprint(
        validation_dataset,
        split="validation",
        dataset_name=args.dataset,
        dataset_revision=args.dataset_revision,
    )

    tokenizer_manifest = spec.load_existing_tokenizer(
        args.tokenizer_dir,
        train_split_fingerprint=train_fingerprint,
        dataset_name=args.dataset,
        dataset_revision=args.dataset_revision,
        document_count=args.tokenizer_story_count,
        max_chunk_bytes=args.max_chunk_bytes,
    )
    tokenizer_status = "already_prepared"
    if tokenizer_manifest is None:
        tokenizer_started_at = time.monotonic()
        print(
            f"[{spec.progress_prefix}] "
            "event=tokenizer_training_started "
            f"{spec.tokenizer_document_noun}={args.tokenizer_story_count:,} "
            f"tokenizer_dir={Path(args.tokenizer_dir).expanduser().resolve()}",
            file=sys.stderr,
            flush=True,
        )
        tokenizer_manifest = spec.train_tokenizer(
            train_dataset,
            args.tokenizer_dir,
            train_split_fingerprint=train_fingerprint,
            dataset_name=args.dataset,
            dataset_revision=args.dataset_revision,
            document_count=args.tokenizer_story_count,
            max_chunk_bytes=args.max_chunk_bytes,
        )
        tokenizer_status = "prepared"
        print(
            f"[{spec.progress_prefix}] "
            "event=tokenizer_training_completed "
            f"{spec.tokenizer_document_noun}={args.tokenizer_story_count:,} "
            f"elapsed_seconds={time.monotonic() - tokenizer_started_at:.1f}",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"[{spec.progress_prefix}] "
            f"event=tokenizer_reused "
            f"tokenizer_dir={Path(args.tokenizer_dir).expanduser().resolve()}",
            file=sys.stderr,
            flush=True,
        )

    source_fingerprint, role_sources = spec.corpus_source_contract(
        train_split_fingerprint=train_fingerprint,
        validation_split_fingerprint=validation_fingerprint,
        train_row_count=len(train_dataset),
        validation_row_count=len(validation_dataset),
        dataset_name=args.dataset,
        dataset_revision=args.dataset_revision,
    )
    corpus_manifest = load_existing_corpus_fn(
        args.corpus_dir,
        corpus_name=spec.corpus_name,
        tokenizer_manifest=tokenizer_manifest,
        tokenizer_name=tokenizer_manifest["tokenizer_name"],
        tokenizer_revision=tokenizer_manifest["manifest_hash"],
        source_dataset=args.dataset,
        source_config=spec.dataset_config,
        source_split=spec.dataset_split,
        text_column="text",
        data_seed=42,
        context_length=spec.context_length,
        shard_token_capacity=args.shard_token_capacity,
        shuffle_buffer_size=0,
        reserved_role_counts=spec.role_counts,
        optimizer_token_limit=args.optimizer_token_count,
        minimum_optimizer_document_count=args.tokenizer_story_count,
        role_source_provenance=role_sources,
    )
    corpus_status = "already_prepared"
    if corpus_manifest is None:
        work_dir = preparation_work_dir_fn(args.corpus_dir)
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

        tokenizer = auto_tokenizer_class.from_pretrained(
            args.tokenizer_dir,
            local_files_only=True,
        )
        corpus_started_at = time.monotonic()

        def report_progress(progress: Mapping[str, Any]) -> None:
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
                "completed_shards="
                f"{progress['completed_optimizer_shard_count']} "
                f"current_shard={progress['current_optimizer_shard_index']:05d} "
                "shard_offset_tokens="
                f"{progress['current_optimizer_shard_offset']:,} "
                f"elapsed_seconds={elapsed:.1f} "
                f"documents_per_second={documents_done / elapsed:.1f} "
                f"tokens_per_second={tokens_done / elapsed:.1f} "
                f"source_read_workers={progress['source_read_workers']} "
                f"tokenization_workers={progress['tokenization_workers']}",
                file=sys.stderr,
                flush=True,
            )

        optimizer_token_target = (
            "all_available"
            if args.optimizer_token_count is None
            else f"{args.optimizer_token_count:,}"
        )
        print(
            "[corpus-progress] "
            f"event={'resuming' if was_resumable else 'starting'} "
            f"documents={resumed_document_count:,} "
            f"optimizer_tokens={resumed_token_count:,} "
            f"target_optimizer_tokens={optimizer_token_target} "
            f"interval_seconds={args.progress_interval_seconds:g} "
            "source_read_workers=1 "
            f"tokenization_workers={args.tokenization_workers}",
            file=sys.stderr,
            flush=True,
        )
        corpus_manifest = prepare_packed_corpus_fn(
            spec.role_ordered_documents(train_dataset, validation_dataset),
            tokenizer,
            Path(args.corpus_dir),
            corpus_name=spec.corpus_name,
            tokenizer_name=tokenizer_manifest["tokenizer_name"],
            tokenizer_revision=tokenizer_manifest["manifest_hash"],
            source_dataset=args.dataset,
            source_config=spec.dataset_config,
            source_split=spec.dataset_split,
            source_fingerprint=source_fingerprint,
            text_column="text",
            data_seed=42,
            context_length=spec.context_length,
            shard_token_capacity=args.shard_token_capacity,
            shuffle_buffer_size=0,
            tokenization_workers=args.tokenization_workers,
            source_read_workers=1,
            tokenizer_manifest=tokenizer_manifest,
            reserved_role_counts=spec.role_counts,
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
            "optimizer_tokens="
            f"{corpus_manifest['available_optimizer_token_count']:,} "
            f"elapsed_seconds={time.monotonic() - corpus_started_at:.1f}",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            "[corpus-progress] "
            "event=already_prepared "
            f"corpus_dir={Path(args.corpus_dir).expanduser().resolve()} "
            "optimizer_tokens="
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
                "optimizer_token_request": (
                    "all"
                    if args.optimizer_token_count is None
                    else args.optimizer_token_count
                ),
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


__all__ = [
    "TinyStoriesPreparationSpec",
    "optimizer_token_count",
    "parse_preparation_args",
    "positive_int",
    "run_tinystories_preparation",
]
