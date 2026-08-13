#!/usr/bin/env python3
"""Fine-tune mT5 for full-sentence Sinhala grammar correction.

The trainer reuses the frozen deterministic rare-pair-disjoint split strategy
used for ByT5 v01. Checkpoints are selected using safely applied, generated
development edit F0.5 rather than teacher-forced loss. Stages 2-6 must never be
passed as --data.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import platform
import random
from pathlib import Path

from grammar_edit_script import (
    correction_metrics,
    sanitize_generated_token_ids,
    validate_full_candidate,
)
from train_grammar_byt5 import load_rows, make_pair_disjoint_split, stable_key


PREFIX = "grammar: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Training JSONL only; never a test stage")
    parser.add_argument("--model", default="google/mt5-small")
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
    parser.add_argument("--generation-max-length", type=int, default=1024)
    parser.add_argument("--dev-ratio", type=float, default=0.05)
    parser.add_argument("--max-dev-pair-frequency", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import datasets
    import numpy as np
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
        raise SystemExit("CUDA GPU not found. Pass --allow-cpu only for a tiny diagnostic.")
    random.seed(args.seed)
    set_seed(args.seed)
    data_path, output_dir = Path(args.data), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, data_stats = load_rows(data_path, args.max_samples)
    train_rows, dev_rows, dropped_rows, split_stats = make_pair_disjoint_split(
        rows, args.dev_ratio, args.max_dev_pair_frequency, args.seed
    )

    membership_path = output_dir / "split_membership.jsonl"
    with membership_path.open("w", encoding="utf-8") as handle:
        for split_name, split_rows in (
            ("train", train_rows),
            ("development", dev_rows),
            ("bridge-dropped", dropped_rows),
        ):
            for row in split_rows:
                handle.write(json.dumps({
                    "split": split_name,
                    "source_line": row["source_line"],
                    "row_sha256": stable_key(row["input"], row["output"]),
                }) + "\n")
    split_stats["membership_path"] = str(membership_path.resolve())
    split_stats["membership_sha256"] = hashlib.sha256(membership_path.read_bytes()).hexdigest()

    development_gold_path = output_dir / "development_gold.jsonl"
    with development_gold_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(dev_rows, 1):
            handle.write(json.dumps({
                "id": f"dev-{index:06d}",
                "input": row["input"],
                "output": row["output"],
                "source_line": row["source_line"],
            }, ensure_ascii=False) + "\n")

    print("Dataset")
    print(f"  source SHA-256 : {data_stats['source_sha256']}")
    print(f"  unique rows    : {len(rows):,}")
    print(f"  train/dev/drop : {len(train_rows):,} / {len(dev_rows):,} / {len(dropped_rows):,}")
    print(f"  shared edits   : {split_stats['shared_train_dev_edits']}")
    print(f"Loading {args.model} ...")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, dtype=dtype)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    source_lengths = [
        len(tokenizer(PREFIX + row["input"], add_special_tokens=True)["input_ids"])
        for row in rows
    ]
    target_lengths = [
        len(tokenizer(row["output"], add_special_tokens=True)["input_ids"])
        for row in rows
    ]
    source_over = sum(length > args.max_source_length for length in source_lengths)
    target_over = sum(length > args.max_target_length for length in target_lengths)
    if source_over or target_over:
        raise SystemExit(
            f"Refusing truncation: {source_over} sources and {target_over} targets exceed limits. "
            "Raise --max-source-length/--max-target-length."
        )

    def tokenize_batch(batch: dict) -> dict:
        encoded = tokenizer(
            [PREFIX + text for text in batch["input"]],
            max_length=args.max_source_length,
            truncation=True,
        )
        encoded["labels"] = tokenizer(
            text_target=batch["output"],
            max_length=args.max_target_length,
            truncation=True,
        )["input_ids"]
        return encoded

    train_dataset = Dataset.from_list(train_rows).map(
        tokenize_batch, batched=True, remove_columns=list(train_rows[0]), desc="Tokenizing train"
    )
    dev_dataset = Dataset.from_list(dev_rows).map(
        tokenize_batch, batched=True, remove_columns=list(dev_rows[0]), desc="Tokenizing development"
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100
    )
    dev_sources = [row["input"] for row in dev_rows]
    dev_targets = [row["output"] for row in dev_rows]

    def generation_finished(token_ids: list[int]) -> bool:
        eos_id = tokenizer.eos_token_id
        return eos_id is None or eos_id in set(token_ids)

    def safely_decode(prediction_ids) -> tuple[list[str], list[list[int]]]:
        clean_ids = sanitize_generated_token_ids(
            prediction_ids, tokenizer.pad_token_id, len(tokenizer)
        )
        return tokenizer.batch_decode(clean_ids, skip_special_tokens=True), clean_ids

    def evaluate_generations(prediction_ids) -> tuple[dict[str, float], list[dict]]:
        raw_candidates, clean_ids = safely_decode(prediction_ids)
        safe_predictions = []
        status_counts: collections.Counter[str] = collections.Counter()
        reason_counts: collections.Counter[str] = collections.Counter()
        details = []
        for source, target, candidate, token_ids in zip(
            dev_sources, dev_targets, raw_candidates, clean_ids
        ):
            finished = generation_finished(token_ids)
            result = validate_full_candidate(
                source, candidate, generation_finished=finished
            )
            safe_predictions.append(result.text)
            status_counts[result.status] += 1
            reason_counts.update(result.reasons)
            details.append({
                "input": source,
                "output": target,
                "raw_prediction": candidate,
                "prediction": result.text,
                "candidate_status": result.status,
                "safety_reasons": list(result.reasons),
                "generation_finished": finished,
            })
        metrics = correction_metrics(dev_sources, dev_targets, safe_predictions)
        raw_metrics = correction_metrics(dev_sources, dev_targets, raw_candidates)
        metrics.update({f"raw_{key}": value for key, value in raw_metrics.items()})
        count = max(1, len(details))
        metrics.update({
            "candidate_invalid_rate": status_counts["INVALID"] / count,
            "candidate_rejected_rate": status_counts["REJECTED"] / count,
            "candidate_applied_rate": status_counts["APPLIED"] / count,
            "candidate_keep_rate": status_counts["KEEP"] / count,
        })
        return metrics, details

    def compute_metrics(eval_prediction) -> dict[str, float]:
        prediction_ids = eval_prediction.predictions
        if isinstance(prediction_ids, tuple):
            prediction_ids = prediction_ids[0]
        metrics, _ = evaluate_generations(prediction_ids)
        return metrics

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size or args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="edit_f0_5",
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=1,
        report_to="none",
        dataloader_num_workers=args.num_workers,
        seed=args.seed,
        data_seed=args.seed,
        save_safetensors=True,
    )

    manifest = {
        "task": "Full-sentence Sinhala grammar correction with mT5",
        "prefix": PREFIX,
        "unicode_normalization": "NFC",
        "checkpoint_selection": "generated development edit F0.5 after safety validation",
        "arguments": vars(args),
        "data": data_stats,
        "split": split_stats,
        "development_gold_sha256": hashlib.sha256(development_gold_path.read_bytes()).hexdigest(),
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
            "numpy": np.__version__,
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
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    trainer.save_state()

    development_output = trainer.predict(dev_dataset, metric_key_prefix="eval")
    prediction_ids = development_output.predictions
    if isinstance(prediction_ids, tuple):
        prediction_ids = prediction_ids[0]
    generated_metrics, details = evaluate_generations(prediction_ids)
    development_predictions_path = output_dir / "development_predictions.jsonl"
    with development_predictions_path.open("w", encoding="utf-8") as handle:
        for index, detail in enumerate(details, 1):
            handle.write(json.dumps({"id": f"dev-{index:06d}", **detail}, ensure_ascii=False) + "\n")

    metrics = dict(result.metrics)
    metrics.update(development_output.metrics)
    metrics.update({f"eval_{key}": value for key, value in generated_metrics.items()})
    metrics["development_predictions_sha256"] = hashlib.sha256(
        development_predictions_path.read_bytes()
    ).hexdigest()
    metrics["peak_gpu_memory_bytes"] = (
        torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    )
    (output_dir / "final_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(f"Training complete: {output_dir}")
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Generated development edit F0.5: {metrics.get('eval_edit_f0_5')}")
    print(f"Development candidate rejection: {metrics.get('eval_candidate_rejected_rate')}")


if __name__ == "__main__":
    main()
