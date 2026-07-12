"""
Fine-tune mT5-base for Sinhala abstractive summarization.

mT5 is pre-trained on mC4 which includes Sinhala — correct choice for this language.
Training target: article title (short, single-sentence Sinhala news headline).

Usage:
    cd /home/jovyan
    python summarizer/abstractive/train_mt5.py

Prerequisites:
    python summarizer/data/split_dataset.py   ← run once first

Output:
    summarizer/models/sinhala-mt5/
"""

import json
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
import evaluate

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
HERE        = Path(__file__).parent.parent          # summarizer/
DATA_DIR    = HERE / "data"
MODEL_OUT   = HERE / "models" / "sinhala-mt5"

BASE_MODEL      = "google/mt5-base"
PREFIX          = "summarize: "     # T5-style task prefix
MAX_INPUT_LEN   = 512
MAX_TARGET_LEN  = 64

NUM_EPOCHS      = 5
BATCH_SIZE      = 2                # shared GPU — keeps peak memory ~4-5 GB
GRAD_ACCUM      = 16               # effective batch = 32 (same training dynamics)
LEARNING_RATE   = 5e-5
WEIGHT_DECAY    = 0.01
WARMUP_STEPS    = 500
EVAL_STRATEGY   = "epoch"
SEED            = 42

# Cap val size so evaluation is fast during training
MAX_VAL_SAMPLES = 5_000


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def load_jsonl(path: Path, max_rows: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def to_hf_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_dict({
        "content": [r["content"] for r in rows],
        "title":   [r["title"]   for r in rows],
    })


# ─────────────────────────────────────────────
# TOKENISATION
# ─────────────────────────────────────────────

def make_tokenize_fn(tokenizer):
    def tokenize(batch):
        inputs = [PREFIX + t for t in batch["content"]]
        model_inputs = tokenizer(
            inputs,
            max_length = MAX_INPUT_LEN,
            truncation = True,
            padding    = False,
        )
        labels = tokenizer(
            text_target = batch["title"],
            max_length  = MAX_TARGET_LEN,
            truncation  = True,
            padding     = False,
        )
        # Replace pad token id with -100 so the loss ignores padding positions.
        # DataCollatorForSeq2Seq does this during batching, but doing it here
        # too makes the intent explicit and avoids version-dependent behaviour.
        label_ids = labels["input_ids"]
        label_ids = [
            [(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
            for seq in label_ids
        ]
        model_inputs["labels"] = label_ids
        return model_inputs
    return tokenize


# ─────────────────────────────────────────────
# METRIC
# Using HuggingFace evaluate during training (fast approximate ROUGE).
# Final evaluation uses the grapheme-cluster ROUGE in evaluate_mt5.py.
# ─────────────────────────────────────────────

def make_compute_metrics(tokenizer):
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds  = np.clip(preds, 0, tokenizer.vocab_size - 1)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(
            predictions = [p.strip() for p in decoded_preds],
            references  = [l.strip() for l in decoded_labels],
            use_stemmer = False,
        )
        return {k: round(v * 100, 2) for k, v in result.items()}

    return compute_metrics


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    train_path = DATA_DIR / "train.jsonl"
    val_path   = DATA_DIR / "val.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found.\n"
            "Run: python summarizer/data/split_dataset.py"
        )

    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    print(f"Loading train : {train_path}")
    train_rows = load_jsonl(train_path)
    print(f"  {len(train_rows):,} rows")

    print(f"Loading val   : {val_path}")
    val_rows = load_jsonl(val_path, MAX_VAL_SAMPLES)
    print(f"  {len(val_rows):,} rows  (capped at {MAX_VAL_SAMPLES:,})")

    dataset = DatasetDict({
        "train"     : to_hf_dataset(train_rows),
        "validation": to_hf_dataset(val_rows),
    })

    print(f"\nLoading {BASE_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    model.config.tie_word_embeddings = False    # silence the checkpoint warning
    model.gradient_checkpointing_enable()       # trades ~30% speed for ~40% less activation memory

    print("Tokenizing ...")
    tokenize_fn = make_tokenize_fn(tokenizer)
    tokenized   = dataset.map(
        tokenize_fn,
        batched        = True,
        remove_columns = ["content", "title"],
        desc           = "tokenising",
    )

    data_collator   = DataCollatorForSeq2Seq(
        tokenizer,
        model              = model,
        padding            = True,
        label_pad_token_id = -100,      # explicitly mask padding in labels
    )
    compute_metrics = make_compute_metrics(tokenizer)

    import torch
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = str(MODEL_OUT),
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LEARNING_RATE,
        weight_decay                = WEIGHT_DECAY,
        warmup_steps                = WARMUP_STEPS,
        fp16                        = use_fp16,
        bf16                        = use_bf16,
        predict_with_generate       = True,
        generation_max_length       = MAX_TARGET_LEN,
        eval_strategy               = EVAL_STRATEGY,
        save_strategy               = EVAL_STRATEGY,
        load_best_model_at_end      = True,
        metric_for_best_model       = "rougeL",
        greater_is_better           = True,
        logging_steps               = 200,
        save_total_limit            = 2,
        report_to                   = "none",
        seed                        = SEED,
    )

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = tokenized["train"],
        eval_dataset    = tokenized["validation"],
        processing_class = tokenizer,
        data_collator   = data_collator,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    steps_per_epoch = len(train_rows) // (BATCH_SIZE * GRAD_ACCUM)
    print(f"\nTraining config:")
    print(f"  Base model       : {BASE_MODEL}")
    print(f"  Train examples   : {len(train_rows):,}")
    print(f"  Batch / device   : {BATCH_SIZE}  ×  grad_accum {GRAD_ACCUM}  =  effective {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Steps / epoch    : {steps_per_epoch:,}")
    print(f"  Max epochs       : {NUM_EPOCHS}")
    print(f"  Precision        : {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}")
    print(f"  Output           : {MODEL_OUT}")

    print("\nStarting training ...")
    trainer.train()

    print("\nSaving final model ...")
    trainer.save_model(str(MODEL_OUT))
    tokenizer.save_pretrained(str(MODEL_OUT))
    print(f"Model saved → {MODEL_OUT}")
    print("\nNext: python summarizer/abstractive/evaluate_mt5.py")


if __name__ == "__main__":
    main()
