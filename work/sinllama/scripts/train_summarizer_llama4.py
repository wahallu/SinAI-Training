"""
SinhalaJournal-LLM | Step 5: Fine-tune SinLLaMA + LoRA for Summarization
-------------------------------------------------------------------------
✅ UPDATED: Uses the pre-merged SinLLaMA base (prepare_sinllama_base.py).
   The old 4-step chain (load base → resize → load adapter → merge) is
   replaced by a single FastLanguageModel.from_pretrained() call.

  pre-merged SinLLaMA base → summarization LoRA  ✅ gradients flow

Usage:
    python train_summarizer.py

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


# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
# ✅ NEW: single path replaces BASE_MODEL_PATH + SINLLAMA_ADAPTER + TOKENIZER_PATH
SINLLAMA_BASE   = "/home/jovyan/work/sinllama/models/SinLLaMA-merged-base"
OUTPUT_ADAPTER  = "/home/jovyan/work/sinllama/models/adapters/summarization_sinllama_v02"
TRAIN_DATA_PATH = "/home/jovyan/summarizer/data/llama4_summaries.jsonl"


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MAX_SEQ_LENGTH   = 2048
LORA_RANK        = 32
LORA_ALPHA       = 64
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
def format_prompt(article, summary):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

        පහත සිංහල පුවත් ලිපිය සාරාංශ කරන්න.
        
        ලිපියේ තොරතුරු පමණක් භාවිතා කරන්න.
        වචන 15-35 අතර සාරාංශයක් ලියන්න.
        
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
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = []

        for rec in records:
            article = rec.get("content", "").strip()
            summary = rec.get("llama4_summary", "").strip()
        
            if not article or not summary:
                continue
        
            article_tokens = len(
                tokenizer.encode(
                    article,
                    add_special_tokens=False
                )
            )
        
            summary_tokens = len(
                tokenizer.encode(
                    summary,
                    add_special_tokens=False
                )
            )
        
            # Remove very short articles
            if article_tokens < 50:
                continue
        
            # Remove bad summaries
            if summary_tokens < 10:
                continue
        
            # Remove excessively long summaries
            if summary_tokens > 150:
                continue
        
            compression_ratio = summary_tokens / article_tokens
        
            # Keep only summaries between 10% and 30%
            if compression_ratio < 0.10:
                continue
        
            if compression_ratio > 0.30:
                continue
        
            self.samples.append(
                format_prompt(article, summary)
            )

        print(f"   Tokenizing {len(self.samples)} samples...")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        full_text = self.samples[idx]
        
        # Split into prompt and response parts
        # prompt_text includes everything up to and including "### Response:\n"
        parts = full_text.split("### Response:\n")
        if len(parts) > 1:
            prompt_text = parts[0] + "### Response:\n"
        else:
            prompt_text = full_text

        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        prompt_encoded = self.tokenizer(
            prompt_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        prompt_len = len(prompt_encoded["input_ids"])
        input_ids = encoded["input_ids"]
        
        # Build labels: -100 for prompt, actual tokens for response
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        
        # Safety check: ensure labels and input_ids match in length
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
    print("\n" + "="*55)
    print("  SinhalaJournal-LLM | Summarization LoRA Training")
    print("="*55)

    random.seed(SEED)

    # ── Step 1: Load tokenizer ───────────────────────────────────
    # ✅ NEW: tokenizer is saved inside SinLLaMA-merged-base —
    #         no separate TOKENIZER_PATH needed
    print(f"\n🔹 Loading tokenizer from pre-merged base...")
    tokenizer = AutoTokenizer.from_pretrained(
        SINLLAMA_BASE,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    print(f"   Vocab size: {len(tokenizer):,} tokens")

    # ── Step 2: Load pre-merged SinLLaMA base ───────────────────
    # ✅ NEW: replaces the old 4-step chain:
    #   1. FastLanguageModel.from_pretrained(llama-3-8b)       ← gone
    #   2. model.resize_token_embeddings(...)                   ← gone
    #   3. PeftModel.from_pretrained(model, SinLlama_v01)      ← gone
    #   4. model.merge_and_unload()                             ← gone
    #
    # Embedding size is already correct — no resize needed.
    print(f"\n🔹 Loading pre-merged SinLLaMA base (4bit)...")
    model, _ = FastLanguageModel.from_pretrained(
        model_name          = SINLLAMA_BASE,
        max_seq_length      = MAX_SEQ_LENGTH,
        dtype               = torch.bfloat16,
        load_in_4bit        = True,
        local_files_only    = True,
        attn_implementation = "eager",
    )
    print(f"   SinLLaMA base loaded ✅  (no active LoRA — clean base for new adapter)")

    # ── Step 3: Add ONE clean summarization LoRA ─────────────────
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

    # ── Step 4: Build datasets ───────────────────────────────────
    print(f"\n📂 Loading training data...")
    records = load_jsonl(TRAIN_DATA_PATH)
    random.shuffle(records)

    n_train       = int(len(records) * TRAIN_SPLIT)
    train_dataset = SummarizationDataset(records[:n_train], tokenizer, MAX_SEQ_LENGTH)
    val_dataset   = SummarizationDataset(records[n_train:], tokenizer, MAX_SEQ_LENGTH)

    print(f"   Train : {len(train_dataset)}")
    print(f"   Val   : {len(val_dataset)}")

    # ── Step 5: Collator & training args ─────────────────────────
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

    # ── Step 6: Train ────────────────────────────────────────────
    total_steps = (len(train_dataset) // (BATCH_SIZE * GRAD_ACCUMULATION)) * NUM_EPOCHS
    print(f"\n🚀 Starting training...")
    print(f"   Epochs          : {NUM_EPOCHS}")
    print(f"   Effective batch : {BATCH_SIZE * GRAD_ACCUMULATION}")
    print(f"   Total steps     : ~{total_steps}")
    print(f"   Output          : {OUTPUT_ADAPTER}\n")
    print(f"   ⚠️  Watch for loss DROPPING and grad_norm > 0\n")

    trainer.train()

    # ── Step 7: Save ─────────────────────────────────────────────
    print(f"\n💾 Saving summarization adapter...")
    model.save_pretrained(OUTPUT_ADAPTER)
    tokenizer.save_pretrained(OUTPUT_ADAPTER)

    print(f"\n✅ Training complete!")
    print(f"   Saved  : {OUTPUT_ADAPTER}")
    print(f"\n   ➡  Next: python 6_test_summarizer.py\n")


if __name__ == "__main__":
    main()