"""
SinhalaJournal-LLM | Step 7: Length-conditioned summarization LoRA
-------------------------------------------------------------------
Trains summarization_sinllama_v07 on multi-length silver data. Each article
contributes up to three training samples (short / medium / long), each with
a length instruction in the prompt — so the adapter learns *native* length
control, which the web app's summary-length drawer requires.

v07 vs v06 (6_train_summarizer.py): trains on
6_multilength_summaries_clean.jsonl (produced by running
clean_multilength_dataset.py over 6_multilength_summary_generator.py's raw
output) instead of the uncleaned data, and summary_is_clean() now also
rejects word-spacing-glue and numeric-unit (ලක්ෂ/මිලියන/කෝටි/බිලියන)
mismatches via data_quality_checks.py — defense in depth against the same
defect classes 7_multilength_summary_generator.py filters at generation
time for any future (re)generation run.

Also ingests (optionally) the single-length 5_qwen_summaries.jsonl leftovers
whose compression ratio fits the long bucket — the 73.6% of that run that
the old global 0.05-0.50 filter would have discarded is mostly valid "long"
data under per-bucket bands.

Usage:
    python abstractive/7_train_summarizer.py

Output:
    /home/jovyan/work/sinllama/models/adapters/summarization_sinllama_v07/
"""

from unsloth import FastLanguageModel  # MUST be first import

import json
import torch
import random
import unicodedata
from pathlib import Path
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

from data_quality_checks import detect_word_glue, check_numeric_unit_consistency

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
SINLLAMA_BASE   = "/home/jovyan/work/sinllama/models/SinLLaMA-merged-base"
OUTPUT_ADAPTER  = "/home/jovyan/work/sinllama/models/adapters/summarization_sinllama_v07"
TRAIN_DATA_PATH = "/home/jovyan/summarizer/data/6_multilength_summaries_clean.jsonl"

# Optional supplementary single-length data (long bucket only). Set to None
# to train purely on the multi-length set.
QWEN_SUPPLEMENT_PATH = "/home/jovyan/summarizer/data/5_qwen_summaries.jsonl"

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MAX_SEQ_LENGTH    = 2048
LORA_RANK         = 32
LORA_ALPHA        = 64
# 0.05, NOT 0.0 — dropout 0.0 activates Unsloth's fast CUDA LoRA kernel,
# which breaks on 4-bit quantized weights (see CLAUDE.md critical
# constraints; this was the actual bug fixed between v03 and v04).
LORA_DROPOUT      = 0.05
# 3 epochs, down from the 5 used for v02-v05: the multi-length set is ~3x
# larger per article, and 5 epochs on the bigger set risks memorizing
# teacher phrasing rather than the summarization behavior.
NUM_EPOCHS        = 3
BATCH_SIZE        = 2
GRAD_ACCUMULATION = 8
LEARNING_RATE     = 2e-4
TRAIN_SPLIT       = 0.85
SEED              = 42

MIN_ARTICLE_TOKENS = 50

# Per-bucket bands (token-space mirror of the generator's word-space bands).
# Must stay consistent with 7_multilength_summary_generator.LENGTH_BUCKETS.
BUCKET_FILTERS = {
    "short":  {"min_ratio": 0.04, "max_ratio": 0.18, "max_summary_tokens": 70},
    "medium": {"min_ratio": 0.12, "max_ratio": 0.32, "max_summary_tokens": 120},
    "long":   {"min_ratio": 0.22, "max_ratio": 0.55, "max_summary_tokens": 190},
}
MIN_SUMMARY_TOKENS = 10

# ──────────────────────────────────────────────
# PROMPT FORMAT
# ──────────────────────────────────────────────
# Identical Llama-3 chat structure to v02-v05 training, with ONE added
# instruction line carrying the length condition. Serving
# (work/tasks/summarizer.py) must send the byte-identical template with the
# matching length line once v06 deploys.
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
# QUALITY FILTERS
# ──────────────────────────────────────────────
def summary_is_clean(summary: str, article: str) -> bool:
    """Defensive re-check of the generator's validation (supplement data
    from 5_qwen never went through it)."""
    if not summary:
        return False
    if "�" in summary:
        return False
    if summary.startswith("#") or "\n#" in summary:
        return False
    if unicodedata.combining(summary[0]):
        return False
    if detect_word_glue(summary):
        return False
    if check_numeric_unit_consistency(summary, article):
        return False
    return True


# ──────────────────────────────────────────────
# DATASET
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


def build_samples(tokenizer) -> tuple[list, dict]:
    """Returns (formatted training samples, per-bucket kept/filtered stats)."""
    stats = {b: {"kept": 0, "filtered": 0} for b in BUCKET_FILTERS}
    samples = []
    seen_pairs = set()  # (url, bucket) dedupe across primary + supplement

    def try_add(article: str, summary: str, bucket: str, url: str):
        key = (url, bucket)
        if url and key in seen_pairs:
            return
        cfg = BUCKET_FILTERS[bucket]

        if not article or not summary_is_clean(summary, article):
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

    # Primary: multi-length records
    for rec in load_jsonl(TRAIN_DATA_PATH):
        article = rec.get("content", "").strip()
        url = rec.get("url", "")
        for bucket in BUCKET_FILTERS:
            summary = rec.get(f"summary_{bucket}", "").strip()
            if summary:
                try_add(article, summary, bucket, url)

    # Supplement: 5_qwen single-length output re-bucketed as "long" where the
    # ratio fits. This reclaims most of what the old global filter discarded.
    if QWEN_SUPPLEMENT_PATH and Path(QWEN_SUPPLEMENT_PATH).exists():
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
    print("  SinhalaJournal-LLM | v07 Length-Conditioned Summarizer")
    print("=" * 60)

    random.seed(SEED)

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

    print("\n📂 Building length-conditioned samples...")
    samples, stats = build_samples(tokenizer)
    print("   Per-bucket:")
    for bucket, s in stats.items():
        print(f"     {bucket:<7}: kept {s['kept']:,}  filtered {s['filtered']:,}")

    random.shuffle(samples)
    n_train = int(len(samples) * TRAIN_SPLIT)
    train_dataset = SummarizationDataset(samples[:n_train], tokenizer, MAX_SEQ_LENGTH)
    val_dataset = SummarizationDataset(samples[n_train:], tokenizer, MAX_SEQ_LENGTH)
    print(f"   Train : {len(train_dataset):,}")
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
    print("\n   ➡  Next: python abstractive/7_test_summarizer.py")


if __name__ == "__main__":
    main()
