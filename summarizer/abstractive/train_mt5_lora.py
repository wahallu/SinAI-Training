"""
SinhalaJournal-LLM | Fine-tune mT5-base with PEFT LoRA for Summarization
-------------------------------------------------------------------------
Trains mT5-base using the exact same Qwen silver summary dataset and filtering
rules as SinLLaMA-v05 (5_train_summarizer_qwen.py) for a direct, scientific comparison.

Usage:
    cd /home/jovyan
    python summarizer/abstractive/train_mt5_lora.py

Input:
    /home/jovyan/summarizer/data/5_qwen_summaries.jsonl

Output LoRA adapter:
    /home/jovyan/work/sinllama/models/adapters/summarization_mt5_v01
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate


# ─────────────────────────────────────────────
# PATHS & CONFIG (Matching 5_train_summarizer_qwen.py)
# ─────────────────────────────────────────────
HERE = Path(__file__).resolve().parent.parent           # summarizer/
DATA_DIR = HERE / "data"

DEFAULT_MODEL_PATH = "/home/jovyan/summarizer/models/mt5-base"
FALLBACK_MODEL_PATH = "google/mt5-base"

DEFAULT_TRAIN_DATA = "/home/jovyan/summarizer/data/5_qwen_summaries.jsonl"
DEFAULT_OUTPUT_ADAPTER = "/home/jovyan/work/sinllama/models/adapters/summarization_mt5_v01"

PREFIX = "summarize: "
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 150

NUM_EPOCHS = 5
BATCH_SIZE = 4
GRAD_ACCUM = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
TRAIN_SPLIT = 0.85
SEED = 42

# Length filtering rules (Identical to 5_train_summarizer_qwen.py)
MIN_ARTICLE_TOKENS = 50
MIN_SUMMARY_TOKENS = 10
MAX_SUMMARY_TOKENS = 150
MIN_COMP_RATIO     = 0.05
MAX_COMP_RATIO     = 0.50


def get_base_model_path() -> str:
    candidates = [
        DEFAULT_MODEL_PATH,
        "/home/jovyan/work/summarizer/models/mt5-base",
        str(HERE / "models" / "mt5-base"),
        FALLBACK_MODEL_PATH,
    ]
    for path in candidates:
        if os.path.exists(path) and (
            os.path.isfile(os.path.join(path, "config.json"))
            or os.path.isfile(os.path.join(path, "model.safetensors"))
            or os.path.isfile(os.path.join(path, "pytorch_model.bin"))
        ):
            print(f"[INFO] Using local raw mT5 base model at: {path}")
            return path
    print(f"[INFO] Local mT5 model not found in candidates. Falling back to HF: {FALLBACK_MODEL_PATH}")
    return FALLBACK_MODEL_PATH


def load_jsonl(path: str) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ─────────────────────────────────────────────
# DATASET WITH QWEN FILTERING RULES
# ─────────────────────────────────────────────
class QwenSummarizationDataset(TorchDataset):
    def __init__(self, records: list, tokenizer, max_input_len: int, max_target_len: int):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.samples = []
        self.kept = 0
        self.filtered = 0

        for rec in records:
            article = rec.get("content", rec.get("article", "")).strip()
            summary = rec.get("qwen_summary", rec.get("summary", rec.get("title", ""))).strip()

            if not article or not summary:
                self.filtered += 1
                continue

            article_tokens = len(tokenizer.encode(article, add_special_tokens=False))
            summary_tokens = len(tokenizer.encode(summary, add_special_tokens=False))

            if article_tokens < MIN_ARTICLE_TOKENS:
                self.filtered += 1
                continue

            if summary_tokens < MIN_SUMMARY_TOKENS or summary_tokens > MAX_SUMMARY_TOKENS:
                self.filtered += 1
                continue

            compression_ratio = summary_tokens / article_tokens if article_tokens else 0.0

            if compression_ratio < MIN_COMP_RATIO or compression_ratio > MAX_COMP_RATIO:
                self.filtered += 1
                continue

            self.samples.append({"content": PREFIX + article, "target": summary})
            self.kept += 1

        print(f"   Kept samples     : {self.kept:,}")
        print(f"   Filtered samples : {self.filtered:,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        model_inputs = self.tokenizer(
            sample["content"],
            max_length=self.max_input_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        labels = self.tokenizer(
            text_target=sample["target"],
            max_length=self.max_target_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        label_ids = labels["input_ids"]
        label_ids = [(tok if tok != self.tokenizer.pad_token_id else -100) for tok in label_ids]
        model_inputs["labels"] = label_ids
        return model_inputs


def make_compute_metrics(tokenizer):
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = np.clip(preds, 0, tokenizer.vocab_size - 1)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(
            predictions=[p.strip() for p in decoded_preds],
            references=[l.strip() for l in decoded_labels],
            use_stemmer=False,
        )
        return {k: round(v * 100, 2) for k, v in result.items()}

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune mT5-base+LoRA using Qwen Summarizer Dataset")
    parser.add_argument("--data", default=DEFAULT_TRAIN_DATA, help="Path to 5_qwen_summaries.jsonl")
    parser.add_argument("--output_adapter", default=DEFAULT_OUTPUT_ADAPTER, help="Path to save trained LoRA adapter")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    random.seed(SEED)

    train_data_path = args.data
    if not os.path.exists(train_data_path):
        alt_paths = [
            str(DATA_DIR / "5_qwen_summaries.jsonl"),
            str(DATA_DIR / "train.jsonl"),
            str(DATA_DIR / "cleaned_dataset.jsonl"),
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                train_data_path = alt
                break

    print(f"\n📂 Loading dataset: {train_data_path}")
    records = load_jsonl(train_data_path)
    if not records:
        raise FileNotFoundError(f"No records found at {train_data_path}")

    print(f"   Total raw articles: {len(records):,}")
    random.shuffle(records)

    base_model_path = get_base_model_path()
    print(f"\n🤖 Loading Tokenizer & Base Model from {base_model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
    )

    print("\n📊 Filtering dataset according to SinLLaMA-v05 rules...")
    n_train = int(len(records) * TRAIN_SPLIT)
    train_dataset = QwenSummarizationDataset(records[:n_train], tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)
    val_dataset = QwenSummarizationDataset(records[n_train:], tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)

    print(f"   Train samples: {len(train_dataset):,}")
    print(f"   Val samples  : {len(val_dataset):,}")

    # ─────────────────────────────────────────────
    # PEFT LoRA CONFIG (Targeting Seq2Seq modules)
    # ─────────────────────────────────────────────
    print("\n⚡ Configuring PEFT LoRA for mT5...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=32,                  # Matching LORA_RANK = 32
        lora_alpha=64,           # Matching LORA_ALPHA = 64
        lora_dropout=0.05,
        target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )
    compute_metrics = make_compute_metrics(tokenizer)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    output_dir = args.output_adapter
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        fp16=use_fp16,
        bf16=use_bf16,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        seed=SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"\n🚀 Starting mT5-base+LoRA training...")
    print(f"   Epochs       : {args.epochs}")
    print(f"   Batch size   : {args.batch_size} × grad_accum {GRAD_ACCUM} = effective {args.batch_size * GRAD_ACCUM}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Output       : {output_dir}\n")

    trainer.train()

    print(f"\n💾 Saving fine-tuned mT5 LoRA adapter to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Training complete! Saved adapter → {output_dir}")


if __name__ == "__main__":
    main()
