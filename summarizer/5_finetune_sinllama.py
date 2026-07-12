"""
SinhalaJournal-LLM | Step 5: Fine-tune SinLLaMA + LoRA for Summarization
-------------------------------------------------------------------------
Fix for zero grad_norm: merge SinLlama_v01 into base weights first,
then add ONE clean summarization LoRA on top.

  4-bit base → merge SinLlama → summarization LoRA  ✅ gradients flow

Usage:
    python 5_finetune_sinllama.py

Output:
    /home/jovyan/work/sinllama/models/adapters/summarization_sinllama_v01/
"""

# Unsloth MUST be first import
from unsloth import FastLanguageModel

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
from peft import PeftModel


# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_MODEL_PATH  = "/home/jovyan/work/sinllama/models/llama-3-8b"
SINLLAMA_ADAPTER = "/home/jovyan/work/sinllama/models/SinLlama_v01"
TOKENIZER_PATH   = "/home/jovyan/work/sinllama/models/Extended-Sinhala-LLaMA"
OUTPUT_ADAPTER   = "/home/jovyan/work/sinllama/models/adapters/summarization_sinllama_v01"
TRAIN_DATA_PATH  = "/home/jovyan/summarizer/summarization_dataset.jsonl"

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MAX_SEQ_LENGTH   = 1024
LORA_RANK        = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.05
NUM_EPOCHS       = 5
BATCH_SIZE       = 2
GRAD_ACCUMULATION= 8          # effective batch = 16
LEARNING_RATE    = 2e-4
TRAIN_SPLIT      = 0.85
SEED             = 42


# ──────────────────────────────────────────────
# PROMPT FORMAT
# ──────────────────────────────────────────────
def format_prompt(article: str, summary: str) -> str:
    return (
        "### Instruction:\n"
        "ඔබ සිංහල පුවත් ලිපි සාරාංශ කිරීමේ විශේෂඥයෙකි.\n"
        "පහත සිංහල පුවත් ලිපිය කියවා, ලිපියේ ප්‍රධාන කරුණු ඇතුළත් සාරාංශයක් ලියන්න.\n"
        "සාරාංශය ලිපියේ දිග මෙන් 10% ක් පමණ විය යුතුය.\n\n"
        f"Article:\n{article}\n\n"
        f"### Response:\n{summary}"
    )


# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────
class SummarizationDataset(TorchDataset):
    def __init__(self, records: list, tokenizer, max_length: int):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = []

        for rec in records:
            article = rec.get("content", "").strip()
            summary = rec.get("summary", "").strip()
            if article and summary:
                self.samples.append(format_prompt(article, summary))

        print(f"   Tokenizing {len(self.samples)} samples...")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.samples[idx],
            max_length     = self.max_length,
            truncation     = True,
            padding        = False,
            return_tensors = None,
        )
        encoded["labels"] = encoded["input_ids"].copy()
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
    print("\n" + "="*55)
    print("  SinhalaJournal-LLM | Summarization LoRA Training")
    print("="*55)

    random.seed(SEED)

    # ── Step 1: Load tokenizer ───────────────────────────────────
    print(f"\n🔹 Loading Extended-Sinhala-LLaMA tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    print(f"   Vocab size: {len(tokenizer):,} tokens")

    # ── Step 2: Load base model ──────────────────────────────────
    print(f"\n🔹 Loading LLaMA-3-8B base model (4bit)...")
    model, _ = FastLanguageModel.from_pretrained(
        model_name       = BASE_MODEL_PATH,
        max_seq_length   = MAX_SEQ_LENGTH,
        dtype            = torch.bfloat16,
        load_in_4bit     = True,
        local_files_only = True,
        attn_implementation = "eager",
    )

    # ── Step 3: Resize embeddings ────────────────────────────────
    print(f"\n🔹 Resizing embeddings...")
    model = model.to("cpu")
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model = model.to("cuda")
    print(f"   Resized to {len(tokenizer):,} ✅")

    # ── Step 4: Load SinLlama_v01 and MERGE into base ────────────
    # This bakes SinLlama knowledge into base weights permanently.
    # Result is a plain model with no active LoRA — safe to add new one.
    print(f"\n🔹 Loading and merging SinLlama_v01 into base weights...")
    model = PeftModel.from_pretrained(
        model,
        SINLLAMA_ADAPTER,
        local_files_only    = True,
        ensure_weight_tying = True,
    )
    model = model.merge_and_unload()   # ← bakes adapter into weights, removes LoRA layers
    print(f"   SinLlama_v01 merged ✅  (no active LoRA remaining)")

    # ── Step 5: Add ONE clean summarization LoRA ─────────────────
    # Now model is plain — FastLanguageModel.get_peft_model() works correctly
    print(f"\n🔹 Adding summarization LoRA (rank={LORA_RANK})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r                          = LORA_RANK,
        lora_alpha                 = LORA_ALPHA,
        lora_dropout               = LORA_DROPOUT,
        bias                       = "none",
        use_gradient_checkpointing = True,
        target_modules             = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Step 6: Build datasets ───────────────────────────────────
    print(f"\n📂 Loading training data...")
    records = load_jsonl(TRAIN_DATA_PATH)
    random.shuffle(records)

    n_train       = int(len(records) * TRAIN_SPLIT)
    train_dataset = SummarizationDataset(records[:n_train], tokenizer, MAX_SEQ_LENGTH)
    val_dataset   = SummarizationDataset(records[n_train:], tokenizer, MAX_SEQ_LENGTH)

    print(f"   Train : {len(train_dataset)}")
    print(f"   Val   : {len(val_dataset)}")

    # ── Step 7: Collator & training args ─────────────────────────
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model             = model,
        padding           = True,
        pad_to_multiple_of= 8,
    )

    Path(OUTPUT_ADAPTER).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir                  = OUTPUT_ADAPTER,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUMULATION,
        learning_rate               = LEARNING_RATE,
        warmup_steps                = 16,
        bf16                        = True,
        logging_steps               = 10,
        save_steps                  = 50,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        save_total_limit            = 2,
        report_to                   = "none",
        seed                        = SEED,
        dataloader_num_workers      = 0,
        remove_unused_columns       = False,
    )

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        eval_dataset  = val_dataset,
        data_collator = collator,
    )

    # ── Step 8: Train ────────────────────────────────────────────
    total_steps = (len(train_dataset) // (BATCH_SIZE * GRAD_ACCUMULATION)) * NUM_EPOCHS
    print(f"\n🚀 Starting training...")
    print(f"   Epochs          : {NUM_EPOCHS}")
    print(f"   Effective batch : {BATCH_SIZE * GRAD_ACCUMULATION}")
    print(f"   Total steps     : ~{total_steps}")
    print(f"   Output          : {OUTPUT_ADAPTER}\n")
    print(f"   ⚠️  Watch for loss DROPPING and grad_norm > 0\n")

    trainer.train()

    # ── Step 9: Save ─────────────────────────────────────────────
    print(f"\n💾 Saving summarization adapter...")
    model.save_pretrained(OUTPUT_ADAPTER)
    tokenizer.save_pretrained(OUTPUT_ADAPTER)

    print(f"\n✅ Training complete!")
    print(f"   Saved  : {OUTPUT_ADAPTER}")
    print(f"\n   ➡  Next: python 6_test_summarizer.py\n")


if __name__ == "__main__":
    main()