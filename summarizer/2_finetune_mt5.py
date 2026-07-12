"""
SinhalaJournal-LLM | Step 2: Fine-tune mT5-base for Abstractive Summarization
-------------------------------------------------------------------------------
Takes the cleaned JSONL from Step 1 and fine-tunes mT5-base.
  - Input  (source) : full news article content
  - Output (target) : news title (used as pseudo-summary)

Usage:
    python 02_finetune_mt5.py --data data/cleaned_dataset.jsonl --output models/sinhala-mt5-summarizer

Requirements:
    pip install transformers datasets sentencepiece rouge-score accelerate
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODEL_NAME       = "google/mt5-base"   # change to mt5-small if GPU < 16GB
MAX_INPUT_LEN    = 512                 # tokens for article content
MAX_TARGET_LEN   = 64                  # tokens for title/summary
PREFIX           = "summarize: "       # T5-style task prefix
TRAIN_SPLIT      = 0.90
VAL_SPLIT        = 0.05
# TEST = remaining 0.05 — saved separately for 03_evaluate.py
SEED             = 42


# ──────────────────────────────────────────────
# LOAD & SPLIT DATASET
# ──────────────────────────────────────────────
def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def split_data(records: list[dict], seed: int = SEED) -> DatasetDict:
    random.seed(seed)
    random.shuffle(records)
    n = len(records)
    n_train = int(n * TRAIN_SPLIT)
    n_val   = int(n * VAL_SPLIT)

    train = records[:n_train]
    val   = records[n_train : n_train + n_val]
    test  = records[n_train + n_val :]

    print(f"   Train : {len(train):,}")
    print(f"   Val   : {len(val):,}")
    print(f"   Test  : {len(test):,}  ← saved to data/test_set.jsonl for evaluation")

    # Save test split for Step 3
    Path("data").mkdir(exist_ok=True)
    with open("data/test_set.jsonl", "w", encoding="utf-8") as f:
        for item in test:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def to_hf(items):
        return Dataset.from_dict({
            "content": [i["content"] for i in items],
            "title":   [i["title"]   for i in items],
        })

    return DatasetDict({"train": to_hf(train), "validation": to_hf(val)})


# ──────────────────────────────────────────────
# TOKENIZE
# ──────────────────────────────────────────────
def make_tokenize_fn(tokenizer):
    def tokenize(batch):
        inputs = [PREFIX + text for text in batch["content"]]
        model_inputs = tokenizer(
            inputs,
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=batch["title"],
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return tokenize


# ──────────────────────────────────────────────
# ROUGE METRIC
# ──────────────────────────────────────────────
def make_compute_metrics(tokenizer):
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # Clip preds to valid vocab range (fixes mT5 OverflowError)
        preds  = np.clip(preds,  0, tokenizer.vocab_size - 1)

        # Replace -100 (padding) in labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds  = tokenizer.batch_decode(preds,   skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels,  skip_special_tokens=True)

        # Strip whitespace
        decoded_preds  = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=False,   # Sinhala — no stemmer
        )
        return {k: round(v * 100, 2) for k, v in result.items()}

    return compute_metrics


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main(data_path: str, output_dir: str):
    print(f"\n📂 Loading cleaned dataset: {data_path}")
    records = load_jsonl(data_path)
    print(f"   Total articles: {len(records):,}")

    print("\n📊 Splitting dataset...")
    dataset = split_data(records)

    print(f"\n🤖 Loading tokenizer & model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    print("\n🔤 Tokenizing...")
    tokenize_fn    = make_tokenize_fn(tokenizer)
    tokenized      = dataset.map(tokenize_fn, batched=True, remove_columns=["content", "title"])

    data_collator  = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    compute_metrics = make_compute_metrics(tokenizer)

    # ── Training Arguments ──────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir                  = output_dir,
        num_train_epochs            = 5,
        per_device_train_batch_size = 8,       # increase to 16 if GPU RAM allows
        per_device_eval_batch_size  = 8,
        warmup_steps                = 500,
        weight_decay                = 0.01,
        learning_rate               = 5e-5,
        fp16                        = False,    # set False if GPU doesn't support fp16
        predict_with_generate       = True,
        generation_max_length       = MAX_TARGET_LEN,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "rougeL",
        greater_is_better           = True,
        logging_steps               = 100,
        save_total_limit            = 2,       # keep only best 2 checkpoints
        report_to                   = "none",  # change to "wandb" if you use W&B
        seed                        = SEED,
    )

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = tokenized["train"],
        eval_dataset    = tokenized["validation"],
        processing_class   = tokenizer,
        data_collator   = data_collator,
        compute_metrics = compute_metrics,
    )

    print(f"\n🚀 Starting training — model will be saved to: {output_dir}")
    print(f"   Epochs       : {training_args.num_train_epochs}")
    print(f"   Batch size   : {training_args.per_device_train_batch_size}")
    print(f"   Learning rate: {training_args.learning_rate}\n")

    trainer.train()

    print("\n💾 Saving final model and tokenizer...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model saved to: {output_dir}")
    print(f"\n   Next step → run: python 03_evaluate.py --model {output_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune mT5 for Sinhala summarization")
    parser.add_argument("--data",   required=True, help="Path to cleaned JSONL from Step 1")
    parser.add_argument("--output", default="models/sinhala-mt5-summarizer",
                        help="Directory to save the fine-tuned model")
    args = parser.parse_args()
    main(args.data, args.output)
