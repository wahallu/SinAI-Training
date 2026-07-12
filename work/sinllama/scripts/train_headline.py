from unsloth import FastLanguageModel
import json, random, torch, unicodedata
from pathlib import Path
from dataclasses import dataclass
from typing import Any
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer

SINLLAMA_BASE   = "/home/jovyan/work/sinllama/models/SinLLaMA-merged-base"
OUTPUT_ADAPTER  = "/home/jovyan/work/sinllama/models/adapters/headline_sinllama_v17"
TRAIN_DATA_PATH = "/home/jovyan/work/sinllama/data/headline_dataset_48k_balanced_train.jsonl"
VAL_DATA_PATH   = "/home/jovyan/work/sinllama/data/headline_dataset_48k_balanced_val.jsonl"

MAX_SEQ_LENGTH    = 768
MAX_ARTICLE_CHARS = 2000
IGNORE_INDEX      = -100

LORA_RANK         = 64
LORA_ALPHA        = 128
LORA_DROPOUT      = 0.08

NUM_EPOCHS        = 8
BATCH_SIZE        = 2
GRAD_ACCUMULATION = 4
LEARNING_RATE     = 5e-5

EVAL_STEPS        = 300
WEIGHT_DECAY      = 0.05
WARMUP_RATIO      = 0.08
LABEL_SMOOTHING   = 0.0
GRAD_CLIP         = 1.0
SEED              = 42
RESPONSE_MARKER   = "### Response:\n"

MIN_HEADLINE_CHARS = 5
MAX_HEADLINE_WORDS = 12
MIN_ARTICLE_CHARS  = 50

KNOWN_CATEGORIES = {
    "Business", "International", "Politics", "General",
    "Sports", "Entertainment", "Law and Order",
    "Editorial", "Science", "Health", "Human Rights", "Technology"
}


def normalize_sinhala(text):
    return unicodedata.normalize("NFC", text.strip()) if text else ""


def is_valid_headline(headline):
    if len(headline) < MIN_HEADLINE_CHARS: return False
    if len(headline.split()) > MAX_HEADLINE_WORDS: return False
    if not any("\u0d80" <= ch <= "\u0dff" for ch in headline): return False
    return True


def is_valid_article(article):
    return len(article.strip()) >= MIN_ARTICLE_CHARS


def build_prompt(article):
    article = normalize_sinhala(article)
    if "Article:" in article:
        cat, body = article.split("Article:", 1)
        article = cat.strip() + "\nArticle: " + body.strip()[:MAX_ARTICLE_CHARS]
    else:
        article = article[:MAX_ARTICLE_CHARS]

    return (
        "### Instruction:\n"
        "Generate a concise Sinhala news headline for the article below.\n\n"
        "Rules:\n"
        "- Use formal Sinhala journalism style matching the article category\n"
        "- Between 4 and 7 words -- never fewer than 4\n"
        "- Capture the key person, event, number, or outcome\n"
        "- Output ONLY the headline, nothing else\n\n"
        f"### Input:\n{article}\n\n"
        f"{RESPONSE_MARKER}"
    )


class HeadlineDataset(TorchDataset):
    def __init__(self, records, tokenizer, max_length):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = []
        skipped_empty = skipped_quality = skipped_truncated = 0
        cat_counts = {}

        for rec in records:
            article  = rec.get("input",  "").strip()
            headline = rec.get("output", "").strip()

            if not article or not headline:
                skipped_empty += 1; continue

            article_body = article.split("Article:", 1)[1].strip() if "Article:" in article else article
            headline     = normalize_sinhala(headline)

            if not is_valid_headline(headline):
                skipped_quality += 1; continue
            if not is_valid_article(article_body):
                skipped_quality += 1; continue

            cat = "General"
            if "Category:" in article:
                cat = article.split("\n")[0].replace("Category:", "").strip()
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

            prompt     = build_prompt(article)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids   = tokenizer(
                prompt + headline + tokenizer.eos_token,
                max_length=max_length, truncation=True,
                padding=False, add_special_tokens=False,
            )["input_ids"]

            prompt_len = len(prompt_ids)
            if prompt_len >= len(full_ids):
                skipped_truncated += 1; continue

            labels = full_ids.copy()
            labels[:prompt_len] = [IGNORE_INDEX] * prompt_len
            self.samples.append({"input_ids": full_ids, "labels": labels})

        hl_tok    = [sum(1 for l in s["labels"] if l != IGNORE_INDEX) for s in self.samples]
        avg_hl    = sum(hl_tok)/len(hl_tok) if hl_tok else 0
        avg_total = sum(len(s["input_ids"]) for s in self.samples)/len(self.samples) if self.samples else 0

        print(f"   Loaded   : {len(self.samples)}")
        print(f"   Skipped  : empty={skipped_empty}  quality={skipped_quality}  truncated={skipped_truncated}")
        print(f"   Avg headline tokens : {avg_hl:.1f}")
        print(f"   Avg total tokens    : {avg_total:.1f}")
        print(f"   Category distribution:")
        max_count = max(cat_counts.values()) if cat_counts else 1
        for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
            bar = chr(9608) * min(30, int(cnt / max_count * 30))
            print(f"     {cat:20s} : {cnt:5d}  {bar}")

        if cat_counts:
            min_cat, min_count = min(cat_counts.items(), key=lambda x: x[1])
            if min_count < max_count * 0.5:
                print(f"\n   WARNING: '{min_cat}' has only {min_count} samples vs "
                      f"{max_count} in the largest category.")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


@dataclass
class MaskedCollator:
    tokenizer: Any
    max_length: int
    def __call__(self, features):
        ids_list  = [f["input_ids"] for f in features]
        labs_list = [f["labels"]    for f in features]
        max_len = min(max(len(x) for x in ids_list), self.max_length)
        p_ids, p_labs, p_attn = [], [], []
        for ids, labs in zip(ids_list, labs_list):
            pad = max_len - len(ids)
            p_ids.append( ids  + [self.tokenizer.pad_token_id] * pad)
            p_labs.append(labs + [IGNORE_INDEX]                * pad)
            p_attn.append([1]  * len(ids) + [0]                * pad)
        return {
            "input_ids":      torch.tensor(p_ids,  dtype=torch.long),
            "labels":         torch.tensor(p_labs, dtype=torch.long),
            "attention_mask": torch.tensor(p_attn, dtype=torch.long),
        }


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: records.append(json.loads(line))
    return records


def main():
    print("\n" + "="*70)
    print("  Headline LoRA Training v16")
    print("  Dataset  : 48K train / balanced across 12 categories")
    print("  Same config as v15: r=64, LR=5e-5, cosine_with_restarts")
    print("  Changes  : EVAL_STEPS=300, patience=15 (scaled for larger data)")
    print("="*70)
    random.seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(SINLLAMA_BASE, local_files_only=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"   Vocab size: {len(tokenizer):,}")

    model, _ = FastLanguageModel.from_pretrained(
        model_name=SINLLAMA_BASE, max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16, load_in_4bit=True, local_files_only=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", use_gradient_checkpointing=True,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    print("\nLoading datasets...")
    train_records = load_jsonl(TRAIN_DATA_PATH)
    val_records   = load_jsonl(VAL_DATA_PATH)
    random.shuffle(train_records)

    print("\nTrain set:")
    train_ds = HeadlineDataset(train_records, tokenizer, MAX_SEQ_LENGTH)
    print("\nVal set:")
    val_ds   = HeadlineDataset(val_records,   tokenizer, MAX_SEQ_LENGTH)
    collator  = MaskedCollator(tokenizer=tokenizer, max_length=MAX_SEQ_LENGTH)

    Path(OUTPUT_ADAPTER).mkdir(parents=True, exist_ok=True)

    steps_per_epoch = len(train_ds) // (BATCH_SIZE * GRAD_ACCUMULATION)
    total_steps     = steps_per_epoch * NUM_EPOCHS
    warmup_steps    = int(total_steps * WARMUP_RATIO)

    print(f"\n  Steps per epoch : ~{steps_per_epoch}")
    print(f"  Total steps     : ~{total_steps}")
    print(f"  Warmup steps    : {warmup_steps}")
    print(f"  Eval every      : {EVAL_STEPS} steps (~{EVAL_STEPS/steps_per_epoch:.2f} epochs)")

    print(f"\nExpected healthy loss:")
    print(f"  Step   10 : ~3.0 - 5.0")
    print(f"  Step  300 : ~1.5 - 2.3")
    print(f"  Step 1000 : ~1.1 - 1.8")
    print(f"  Step 2500 : ~0.8 - 1.4")

    args = TrainingArguments(
        output_dir=OUTPUT_ADAPTER,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine_with_restarts",
        weight_decay=WEIGHT_DECAY,
        bf16=True, fp16=False,
        logging_steps=25,
        max_grad_norm=GRAD_CLIP,
        label_smoothing_factor=LABEL_SMOOTHING,
        eval_strategy="steps", eval_steps=EVAL_STEPS,
        save_strategy="steps", save_steps=EVAL_STEPS,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none", seed=SEED,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, dataset_text_field=None,
        max_seq_length=MAX_SEQ_LENGTH, packing=False, args=args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=15)],
    )

    print(f"\n  LoRA rank       : {LORA_RANK}")
    print(f"  LoRA alpha      : {LORA_ALPHA}")
    print(f"  LoRA dropout    : {LORA_DROPOUT}")
    print(f"  Learning rate   : {LEARNING_RATE}")
    print(f"  Weight decay    : {WEIGHT_DECAY}")
    print(f"  LR scheduler    : cosine_with_restarts")
    print(f"  Label smoothing : {LABEL_SMOOTHING}  (must stay 0.0)")
    print(f"  Early stopping  : patience=15\n")

    trainer.train()

    model.save_pretrained(OUTPUT_ADAPTER)
    tokenizer.save_pretrained(OUTPUT_ADAPTER)
    print(f"\nDone! Saved to: {OUTPUT_ADAPTER}")


if __name__ == "__main__":
    main()