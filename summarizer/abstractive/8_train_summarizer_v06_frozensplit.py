"""
SinhalaJournal-LLM | Step 8c: v06 recipe retrained on the frozen split
-------------------------------------------------------------------
Retrains the v06 recipe (raw 6_multilength_summaries.jsonl corpus, v06's
simple quality filter — no word-glue/numeric-unit check) on the frozen,
article-level train/val partitions from 8_freeze_dataset_split.py, instead
of 6_train_summarizer.py's internal sample-level 85/15 split.

This is byte-identical to 6_train_summarizer.py in every hyperparameter
(LoRA rank/alpha/dropout, epochs, batch, lr, seed, target modules, prompt
template) — the ONLY substantive change is where the train/val samples come
from: data/summarization_frozen_{train,val}.jsonl (fixed article-level
partitions, seed 42, built from the raw corpus — the same universe v06's
original training used) instead of an internal shuffle+split of the whole
corpus. See 8_freeze_dataset_split.py's docstring for why the frozen split
is built from the raw file, not the cleaned one: v06 and v07 are different
datasets, and that difference is the actual thing under comparison — using
the raw-file split as the shared boundary keeps both recipes comparable
while leaving each one's own data source intact.

Output goes to a STAGING directory outside ADAPTERS_DIR, not the live
adapters directory — serve_sinai.py's find_latest_adapters() auto-serves
whichever adapter folder sorts to the highest version number, so a new
folder appearing under ADAPTERS_DIR could go live via a version-sort tie
before anyone has looked at its evaluation. Promote manually once a
decision is made (see SUMMARIZATION_NEXT_STEPS.md).

Usage:
    python abstractive/8_train_summarizer_v06_frozensplit.py

Output:
    /home/jovyan/work/sinllama/models/adapters_staging/summarization_sinllama_v06_frozensplit/
"""

from unsloth import FastLanguageModel  # MUST be first import

import json
import torch
import unicodedata
from pathlib import Path
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
SINLLAMA_BASE   = "/home/jovyan/work/sinllama/models/SinLLaMA-merged-base"
OUTPUT_ADAPTER  = "/home/jovyan/work/sinllama/models/adapters_staging/summarization_sinllama_v06_frozensplit"

FROZEN_TRAIN_PATH = "/home/jovyan/summarizer/data/summarization_frozen_train.jsonl"
FROZEN_VAL_PATH = "/home/jovyan/summarizer/data/summarization_frozen_val.jsonl"

# Optional supplementary single-length data (long bucket only), unchanged
# from 6_train_summarizer.py. No url overlap risk with the frozen split's
# held-out val/test partitions since it's single-length data outside the
# corpus being split, added to train only.
QWEN_SUPPLEMENT_PATH = "/home/jovyan/summarizer/data/5_qwen_summaries.jsonl"

# ──────────────────────────────────────────────
# CONFIG — identical to 6_train_summarizer.py (TRAIN_SPLIT removed: the
# split is now decided by the frozen files, not by this script)
# ──────────────────────────────────────────────
MAX_SEQ_LENGTH    = 2048
LORA_RANK         = 32
LORA_ALPHA        = 64
LORA_DROPOUT      = 0.05
NUM_EPOCHS        = 3
BATCH_SIZE        = 2
GRAD_ACCUMULATION = 8
LEARNING_RATE     = 2e-4
SEED              = 42

MIN_ARTICLE_TOKENS = 50

BUCKET_FILTERS = {
    "short":  {"min_ratio": 0.04, "max_ratio": 0.18, "max_summary_tokens": 70},
    "medium": {"min_ratio": 0.12, "max_ratio": 0.32, "max_summary_tokens": 120},
    "long":   {"min_ratio": 0.22, "max_ratio": 0.55, "max_summary_tokens": 190},
}
MIN_SUMMARY_TOKENS = 10

# ──────────────────────────────────────────────
# PROMPT FORMAT — identical to 6_train_summarizer.py / 7_train_summarizer.py
# ──────────────────────────────────────────────
LENGTH_LINES = {
    "short":  "සාරාංශය ඉතා කෙටි විය යුතුය — මුල් ලිපියේ දිගෙන් 10%ක් පමණ.",
    "medium": "සාරාංශය මධ්‍යම දිගකින් විය යුතුය — මුල් ලිපියේ දිගෙන් 20%ක් පමණ.",
    "long":   "සාරාංශය සවිස්තරාත්මක විය යුතුය — මුල් ලිපියේ දිගෙන් 35%ක් පමණ.",
}


def format_prompt(article: str, summary: str, bucket: str) -> str:
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

පහත සිංහල පුවත් ලිපිය සාරාංශ කරන්න.

ලිපියේ ඇති තොරතුරු පමණක් භාවිතා කරන්න.
{LENGTH_LINES[bucket]}
අමතර අදහස්, විශ්ලේෂණ හෝ නව තොරතුරු එකතු නොකරන්න.

Article:
{article}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{summary}<|eot_id|>
"""


# ──────────────────────────────────────────────
# QUALITY FILTERS — identical to 6_train_summarizer.py (v06's simple filter,
# no word-glue/numeric-unit check)
# ──────────────────────────────────────────────
def summary_is_clean(summary: str) -> bool:
    if not summary:
        return False
    if "�" in summary:
        return False
    if summary.startswith("#") or "\n#" in summary:
        return False
    if unicodedata.combining(summary[0]):
        return False
    return True


# ──────────────────────────────────────────────
# DATASET — identical to 6_train_summarizer.py
# ──────────────────────────────────────────────
class SummarizationDataset(TorchDataset):
    def __init__(self, samples: list, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        full_text = self.samples[idx]

        split_token = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        if split_token in full_text:
            prompt_text = full_text.split(split_token)[0] + split_token
        else:
            prompt_text = full_text

        encoded = self.tokenizer(
            full_text, max_length=self.max_length, truncation=True,
            padding=False, return_tensors=None, add_special_tokens=False,
        )
        prompt_encoded = self.tokenizer(
            prompt_text, max_length=self.max_length, truncation=True,
            padding=False, return_tensors=None, add_special_tokens=False,
        )

        prompt_len = len(prompt_encoded["input_ids"])
        input_ids = encoded["input_ids"]
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        if len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]
        elif len(labels) < len(input_ids):
            labels = labels + [-100] * (len(input_ids) - len(labels))
        encoded["labels"] = labels
        return encoded


def load_jsonl(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def build_samples_from_records(records: list, tokenizer, include_qwen: bool) -> tuple[list, dict]:
    """Same filtering logic as 6_train_summarizer.py's build_samples(), but
    reading from an already-fixed list of records (the frozen partition)
    instead of re-deriving the partition itself."""
    stats = {b: {"kept": 0, "filtered": 0} for b in BUCKET_FILTERS}
    samples = []
    seen_pairs = set()

    def try_add(article: str, summary: str, bucket: str, url: str):
        key = (url, bucket)
        if url and key in seen_pairs:
            return
        cfg = BUCKET_FILTERS[bucket]

        if not article or not summary_is_clean(summary):
            stats[bucket]["filtered"] += 1
            return

        article_tokens = len(tokenizer.encode(article, add_special_tokens=False))
        summary_tokens = len(tokenizer.encode(summary, add_special_tokens=False))

        if article_tokens < MIN_ARTICLE_TOKENS:
            stats[bucket]["filtered"] += 1
            return
        if summary_tokens < MIN_SUMMARY_TOKENS or summary_tokens > cfg["max_summary_tokens"]:
            stats[bucket]["filtered"] += 1
            return
        ratio = summary_tokens / article_tokens
        if ratio < cfg["min_ratio"] or ratio > cfg["max_ratio"]:
            stats[bucket]["filtered"] += 1
            return

        samples.append(format_prompt(article, summary, bucket))
        seen_pairs.add(key)
        stats[bucket]["kept"] += 1

    for rec in records:
        article = rec.get("content", "").strip()
        url = rec.get("url", "")
        for bucket in BUCKET_FILTERS:
            summary = rec.get(f"summary_{bucket}", "").strip()
            if summary:
                try_add(article, summary, bucket, url)

    if include_qwen and Path(QWEN_SUPPLEMENT_PATH).exists():
        for rec in load_jsonl(QWEN_SUPPLEMENT_PATH):
            article = rec.get("content", "").strip()
            summary = rec.get("qwen_summary", "").strip()
            if article and summary:
                try_add(article, summary, "long", rec.get("url", ""))

    return samples, stats


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  SinhalaJournal-LLM | v06 recipe, frozen split")
    print("=" * 60)

    for p in (FROZEN_TRAIN_PATH, FROZEN_VAL_PATH):
        if not Path(p).exists():
            raise SystemExit(f"Missing {p} — run abstractive/8_freeze_dataset_split.py first.")

    print("\n🔹 Loading tokenizer from pre-merged base...")
    tokenizer = AutoTokenizer.from_pretrained(SINLLAMA_BASE, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"   Vocab size: {len(tokenizer):,} tokens")

    print("\n🔹 Loading pre-merged SinLLaMA base (4bit)...")
    model, _ = FastLanguageModel.from_pretrained(
        model_name=SINLLAMA_BASE,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        local_files_only=True,
        attn_implementation="eager",
    )

    print(f"\n🔹 Adding summarization LoRA (rank={LORA_RANK}, dropout={LORA_DROPOUT})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=True,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    print(f"\n📂 Building train samples from frozen partition ({FROZEN_TRAIN_PATH})...")
    train_records = load_jsonl(FROZEN_TRAIN_PATH)
    train_samples, train_stats = build_samples_from_records(train_records, tokenizer, include_qwen=True)
    print("   Per-bucket (train):")
    for bucket, s in train_stats.items():
        print(f"     {bucket:<7}: kept {s['kept']:,}  filtered {s['filtered']:,}")

    print(f"\n📂 Building val samples from frozen partition ({FROZEN_VAL_PATH})...")
    val_records = load_jsonl(FROZEN_VAL_PATH)
    val_samples, val_stats = build_samples_from_records(val_records, tokenizer, include_qwen=False)
    print("   Per-bucket (val):")
    for bucket, s in val_stats.items():
        print(f"     {bucket:<7}: kept {s['kept']:,}  filtered {s['filtered']:,}")

    train_dataset = SummarizationDataset(train_samples, tokenizer, MAX_SEQ_LENGTH)
    val_dataset = SummarizationDataset(val_samples, tokenizer, MAX_SEQ_LENGTH)
    print(f"\n   Train : {len(train_dataset):,}")
    print(f"   Val   : {len(val_dataset):,}")

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8)
    Path(OUTPUT_ADAPTER).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_ADAPTER,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_steps=16,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        seed=SEED,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    total_steps = (len(train_dataset) // (BATCH_SIZE * GRAD_ACCUMULATION)) * NUM_EPOCHS
    print("\n🚀 Starting training...")
    print(f"   Epochs          : {NUM_EPOCHS}")
    print(f"   Effective batch : {BATCH_SIZE * GRAD_ACCUMULATION}")
    print(f"   Total steps     : ~{total_steps:,}")
    print(f"   Output          : {OUTPUT_ADAPTER}\n")

    trainer.train()

    print("\n💾 Saving summarization adapter...")
    model.save_pretrained(OUTPUT_ADAPTER)
    tokenizer.save_pretrained(OUTPUT_ADAPTER)

    print("\n✅ Training complete!")
    print(f"   Saved  : {OUTPUT_ADAPTER}")
    print("\n   ➡  Next: python abstractive/8_train_summarizer_v07_frozensplit.py"
          " (if not already run), then evaluate both against"
          " data/summarization_frozen_eval_subset.jsonl")


if __name__ == "__main__":
    main()
