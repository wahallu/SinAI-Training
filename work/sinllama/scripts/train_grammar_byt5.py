#!/usr/bin/env python3
"""Fine-tune ByT5 for Sinhala grammar correction.

The legacy v10 JSONL has no source/document IDs. This trainer therefore:
  * normalizes Unicode and removes exact duplicate input/output rows;
  * creates a deterministic development set from rare correction pairs;
  * removes bridge rows that would expose a development correction pair to train;
  * records hashes, split counts, package versions, and arguments.

Stage 6 must never be passed to this program.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import platform
import random
import unicodedata
from pathlib import Path


PREFIX = "grammar: "
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Training JSONL with input/output fields")
    parser.add_argument("--model", default="google/byt5-small", help="HF model ID or local model path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-source-length", type=int, default=1024)
    parser.add_argument("--max-target-length", type=int, default=1024)
    parser.add_argument("--dev-ratio", type=float, default=0.05)
    parser.add_argument(
        "--max-dev-pair-frequency",
        type=int,
        default=2,
        help="Only correction pairs occurring at most this often can seed development",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Deterministic smoke-test subset")
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit CPU training (intended only for tiny diagnostic runs)",
    )
    return parser.parse_args()


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def stable_key(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()


def edit_pairs(source: str, target: str) -> frozenset[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Return exact whitespace-token edits, including insertion/deletion/reorder edits."""
    old, new = source.split(), target.split()
    edits = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, old, new, autojunk=False
    ).get_opcodes():
        if tag != "equal":
            edits.append((tuple(old[i1:i2]), tuple(new[j1:j2])))
    return frozenset(edits)


def load_rows(path: Path, max_samples: int | None) -> tuple[list[dict], dict]:
    raw_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    rows, seen = [], set()
    malformed = duplicates = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            source = normalize(str(row.get("input", "")))
            target = normalize(str(row.get("output", "")))
            if not source or not target:
                malformed += 1
                continue
            key = (source, target)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            rows.append({"input": source, "output": target, "source_line": line_number})

    # Hash order makes a smoke subset deterministic without taking the first N rows.
    if max_samples is not None:
        if max_samples < 20:
            raise ValueError("--max-samples must be at least 20")
        rows = sorted(rows, key=lambda r: stable_key(r["input"], r["output"]))[:max_samples]

    if len(rows) < 20:
        raise ValueError(f"Only {len(rows)} usable rows were found")
    return rows, {
        "source_path": str(path.resolve()),
        "source_sha256": raw_sha,
        "usable_unique_rows": len(rows),
        "exact_duplicates_removed": duplicates,
        "malformed_rows_removed": malformed,
    }


def make_pair_disjoint_split(
    rows: list[dict], dev_ratio: float, max_pair_frequency: int, seed: int
) -> tuple[list[dict], list[dict], list[dict], dict]:
    if not 0.01 <= dev_ratio <= 0.25:
        raise ValueError("--dev-ratio must be between 0.01 and 0.25")

    row_edits = [edit_pairs(r["input"], r["output"]) for r in rows]
    pair_frequency = collections.Counter(pair for edits in row_edits for pair in edits)
    wanted = max(2, round(len(rows) * dev_ratio))
    changed_wanted = round(wanted * 0.65)
    clean_wanted = wanted - changed_wanted

    def order(index: int) -> str:
        return stable_key(str(seed), rows[index]["input"], rows[index]["output"])

    changed_candidates = [
        i
        for i, edits in enumerate(row_edits)
        if edits and all(pair_frequency[pair] <= max_pair_frequency for pair in edits)
    ]
    clean_candidates = [i for i, edits in enumerate(row_edits) if not edits]
    changed_candidates.sort(key=order)
    clean_candidates.sort(key=order)

    if len(changed_candidates) < changed_wanted:
        raise ValueError(
            f"Only {len(changed_candidates)} rare-pair changed rows are available, "
            f"but {changed_wanted} are needed. Increase --max-dev-pair-frequency."
        )
    if len(clean_candidates) < clean_wanted:
        raise ValueError(
            f"Only {len(clean_candidates)} clean rows are available, but {clean_wanted} are needed"
        )

    dev_indices = set(changed_candidates[:changed_wanted] + clean_candidates[:clean_wanted])
    reserved_pairs = set()
    for i in dev_indices:
        reserved_pairs.update(row_edits[i])

    train, dev, dropped = [], [], []
    for i, row in enumerate(rows):
        if i in dev_indices:
            dev.append(row)
        elif row_edits[i] & reserved_pairs:
            dropped.append(row)
        else:
            train.append(row)

    train_pairs = {pair for row in train for pair in edit_pairs(row["input"], row["output"])}
    dev_pairs = {pair for row in dev for pair in edit_pairs(row["input"], row["output"])}
    overlap = train_pairs & dev_pairs
    if overlap:
        raise AssertionError(f"Pair-disjoint split failed: {len(overlap)} shared edits")

    stats = {
        "strategy": "deduplicated deterministic rare-pair-disjoint split",
        "seed": seed,
        "requested_dev_ratio": dev_ratio,
        "max_dev_pair_frequency": max_pair_frequency,
        "train_rows": len(train),
        "dev_rows": len(dev),
        "bridge_rows_dropped": len(dropped),
        "train_changed": sum(r["input"] != r["output"] for r in train),
        "dev_changed": sum(r["input"] != r["output"] for r in dev),
        "train_distinct_edits": len(train_pairs),
        "dev_distinct_edits": len(dev_pairs),
        "shared_train_dev_edits": len(overlap),
    }
    return train, dev, dropped, stats


def main() -> None:
    args = parse_args()

    # Heavy imports occur after argparse so --help works even before GPU packages are installed.
    import datasets
    import torch
    import transformers
    from datasets import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA GPU not found. Fix the GPU environment or pass --allow-cpu for diagnostics.")

    random.seed(args.seed)
    set_seed(args.seed)
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, data_stats = load_rows(data_path, args.max_samples)
    train_rows, dev_rows, dropped_rows, split_stats = make_pair_disjoint_split(
        rows, args.dev_ratio, args.max_dev_pair_frequency, args.seed
    )

    print("\nDataset")
    print(f"  source SHA-256 : {data_stats['source_sha256']}")
    print(f"  unique rows    : {len(rows):,}")
    print(f"  train/dev/drop : {len(train_rows):,} / {len(dev_rows):,} / {len(dropped_rows):,}")
    print(f"  shared edits   : {split_stats['shared_train_dev_edits']}")

    membership_path = output_dir / "split_membership.jsonl"
    with membership_path.open("w", encoding="utf-8") as handle:
        for split_name, split_rows in (
            ("train", train_rows),
            ("development", dev_rows),
            ("bridge-dropped", dropped_rows),
        ):
            for row in split_rows:
                handle.write(
                    json.dumps(
                        {
                            "split": split_name,
                            "source_line": row["source_line"],
                            "row_sha256": stable_key(row["input"], row["output"]),
                        }
                    )
                    + "\n"
                )
    split_stats["membership_path"] = str(membership_path.resolve())
    split_stats["membership_sha256"] = hashlib.sha256(
        membership_path.read_bytes()
    ).hexdigest()

    print(f"\nLoading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, torch_dtype=dtype)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    def token_length(row: dict, field: str, prefix: str = "") -> int:
        return len(tokenizer(prefix + row[field], add_special_tokens=True)["input_ids"])

    source_lengths = [token_length(row, "input", PREFIX) for row in rows]
    target_lengths = [token_length(row, "output") for row in rows]
    source_truncated = sum(length > args.max_source_length for length in source_lengths)
    target_truncated = sum(length > args.max_target_length for length in target_lengths)
    if source_truncated or target_truncated:
        raise SystemExit(
            "Refusing silent truncation: "
            f"{source_truncated} source and {target_truncated} target rows exceed limits. "
            "Raise --max-source-length/--max-target-length."
        )

    def tokenize_batch(batch: dict) -> dict:
        return tokenizer(
            [PREFIX + text for text in batch["input"]],
            text_target=batch["output"],
            max_length=args.max_source_length,
            truncation=True,
        )

    train_dataset = Dataset.from_list(train_rows).map(
        tokenize_batch,
        batched=True,
        remove_columns=list(train_rows[0]),
        desc="Tokenizing train",
    )
    dev_dataset = Dataset.from_list(dev_rows).map(
        tokenize_batch,
        batched=True,
        remove_columns=list(dev_rows[0]),
        desc="Tokenizing development",
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100
    )

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16
    eval_batch_size = args.eval_batch_size or args.batch_size
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=args.save_total_limit,
        predict_with_generate=False,
        report_to="none",
        dataloader_num_workers=args.num_workers,
        seed=args.seed,
        data_seed=args.seed,
        save_safetensors=True,
    )

    manifest = {
        "task": "Sinhala grammar correction with ByT5",
        "prefix": PREFIX,
        "unicode_normalization": "NFC",
        "arguments": vars(args),
        "data": data_stats,
        "split": split_stats,
        "lengths": {
            "max_source_tokens_observed": max(source_lengths),
            "max_target_tokens_observed": max(target_lengths),
            "max_source_length": args.max_source_length,
            "max_target_length": args.max_target_length,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)] if args.epochs >= 3 else []
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=callbacks,
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    trainer.save_state()

    metrics = dict(result.metrics)
    metrics.update(trainer.evaluate())
    metrics["peak_gpu_memory_bytes"] = (
        torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    )
    (output_dir / "final_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(f"\nTraining complete: {output_dir}")
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Final eval loss: {metrics.get('eval_loss')}")


if __name__ == "__main__":
    main()
