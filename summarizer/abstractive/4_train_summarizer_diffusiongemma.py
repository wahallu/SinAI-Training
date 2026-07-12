"""
SinhalaJournal-LLM | Step 5: Fine-tune SinLLaMA + LoRA for Summarization
-------------------------------------------------------------------------
Uses the pre-merged SinLLaMA base (prepare_sinllama_base.py).

Usage:
    python train_summarizer.py

Output:
    /home/jovyan/work/sinllama/models/adapters/summarization_sinllama_v02/
"""

from unsloth import FastLanguageModel  # MUST be first import

import json
import torch
import random
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
OUTPUT_ADAPTER  = "/home/jovyan/work/sinllama/models/adapters/summarization_sinllama_v04"
TRAIN_DATA_PATH = "/home/jovyan/summarizer/data/4_diffusiongemma_summaries.jsonl"

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MAX_SEQ_LENGTH    = 2048
LORA_RANK         = 32
LORA_ALPHA        = 64
LORA_DROPOUT      = 0.0
NUM_EPOCHS        = 5
BATCH_SIZE        = 2
GRAD_ACCUMULATION = 8
LEARNING_RATE     = 2e-4
TRAIN_SPLIT       = 0.85
SEED              = 42

# Length filtering
MIN_ARTICLE_TOKENS = 50
MIN_SUMMARY_TOKENS = 10
MAX_SUMMARY_TOKENS = 150
MIN_COMP_RATIO     = 0.05
MAX_COMP_RATIO     = 0.50


# ──────────────────────────────────────────────
# PROMPT FORMAT
# ──────────────────────────────────────────────
def format_prompt(article: str, summary: str) -> str:
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

පහත සිංහල පුවත් ලිපිය සාරාංශ කරන්න.

ලිපියේ ඇති තොරතුරු පමණක් භාවිතා කරන්න.
සාරාංශය මුල් ලිපියේ දිගෙන් 10% ත් 30% ත් අතර විය යුතුය.
අමතර අදහස්, විශ්ලේෂණ හෝ නව තොරතුරු එකතු නොකරන්න.

Article:
{article}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{summary}<|eot_id|>
"""


# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────
class SummarizationDataset(TorchDataset):
    def __init__(self, records: list, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.kept = 0
        self.filtered = 0

        for rec in records:
            article = rec.get("content", "").strip()
            summary = rec.get("diffusiongemma_summary", "").strip()

            if not article or not summary:
                self.filtered += 1
                continue

            article_tokens = len(tokenizer.encode(article, add_special_tokens=False))
            summary_tokens = len(tokenizer.encode(summary, add_special_tokens=False))

            if article_tokens < MIN_ARTICLE_TOKENS:
                self.filtered += 1
                continue

            if summary_tokens < MIN_SUMMARY_TOKENS:
                self.filtered += 1
                continue

            if summary_tokens > MAX_SUMMARY_TOKENS:
                self.filtered += 1
                continue

            compression_ratio = summary_tokens / article_tokens if article_tokens else 0.0

            if compression_ratio < MIN_COMP_RATIO or compression_ratio > MAX_COMP_RATIO:
                self.filtered += 1
                continue

            self.samples.append(format_prompt(article, summary))
            self.kept += 1

        print(f"   Kept samples     : {self.kept}")
        print(f"   Filtered samples : {self.filtered}")
        print(f"   Tokenizing {len(self.samples)} samples...")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        full_text = self.samples[idx]

        # Split into prompt and response parts
        split_token = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        if split_token in full_text:
            prompt_text = full_text.split(split_token)[0] + split_token
        else:
            prompt_text = full_text

        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )

        prompt_encoded = self.tokenizer(
            prompt_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
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
                records.append(json.loads(line))
    return records


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("\n" + "=" * 55)
    print("  SinhalaJournal-LLM | Summarization LoRA Training")
    print("=" * 55)

    random.seed(SEED)

    print("\n🔹 Loading tokenizer from pre-merged base...")
    tokenizer = AutoTokenizer.from_pretrained(
        SINLLAMA_BASE,
        local_files_only=True,
    )
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
    print("   SinLLaMA base loaded ✅")

    print(f"\n🔹 Adding summarization LoRA (rank={LORA_RANK})...")
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

    print("\n📂 Loading training data...")
    records = load_jsonl(TRAIN_DATA_PATH)
    random.shuffle(records)

    n_train = int(len(records) * TRAIN_SPLIT)
    train_dataset = SummarizationDataset(records[:n_train], tokenizer, MAX_SEQ_LENGTH)
    val_dataset = SummarizationDataset(records[n_train:], tokenizer, MAX_SEQ_LENGTH)

    print(f"   Train : {len(train_dataset)}")
    print(f"   Val   : {len(val_dataset)}")

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

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
        save_steps=50,
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
    print(f"   Total steps     : ~{total_steps}")
    print(f"   Output          : {OUTPUT_ADAPTER}\n")

    trainer.train()

    print("\n💾 Saving summarization adapter...")
    model.save_pretrained(OUTPUT_ADAPTER)
    tokenizer.save_pretrained(OUTPUT_ADAPTER)

    print("\n✅ Training complete!")
    print(f"   Saved  : {OUTPUT_ADAPTER}")
    print("\n   ➡  Next: python test_summarizer.py")


if __name__ == "__main__":
    main()
