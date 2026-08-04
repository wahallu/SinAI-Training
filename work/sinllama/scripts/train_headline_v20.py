"""Headline LoRA training v20 — length-conditioned + input-and-output
artifact cleaning.

v19 (train_headline_v18.py's length conditioning + clean_headline_dataset.py's
reference-headline cleaning) cut the measured artifact rate from 11.2% to
1.1% overall (see CLAUDE.md). But live-server testing on fresh articles after
v19 deployed still showed tags leaking through occasionally — "(වීඩියෝ)",
"[video]", "- photo" — for two reasons v19 didn't address:

  1. Its word list / separator class missed square-bracket tags and bare
     dash-prefixed words.
  2. It only cleaned the reference *headline*. Scraped articles often carry
     an inline tag right next to the sentence a headline gets built from, and
     the model can copy that tag into a generated headline regardless of how
     clean the training label was — the tag is in the *input*, not the label.

v20 changes exactly one thing from v19: it trains on
headline_dataset_48k_balanced_{train,val}_clean_v20.jsonl, produced by
scripts/clean_headline_dataset_v20.py, which widens the artifact word list
and — the real fix — also strips inline tags from the article `input` field,
not just the reference `output`. Run clean_headline_dataset_v20.py before
this script if the _clean_v20 files don't exist yet.

Note this is the retrain-side half of the fix. The immediate,
retrain-independent half already shipped in backend-api:
app/core/text_cleaning.py strips article-side tags before prompting and
headline-output tags before returning a response, so live behavior improved
before this adapter even exists. v20 is what reduces how often that safety
net has to do anything, and how often the model reaches for the pattern on
articles the safety net's word list doesn't happen to cover.

Bands are non-overlapping (short 3-5, medium 6-7, long 8-10) and must stay
byte-identical to HEADLINE_LENGTHS in the serving path:
  - SinhalaJournalLLM/apps/backend-api/app/core/prompts.py  (builds the prompt)
  - SinAI-Training/work/tasks/headline.py                   (token budgets)
Change one, change all three, or the prompt the model trained on stops
matching the prompt it's served.

Everything else — LoRA config, LR schedule, collator, loss masking, early
stopping, prompt format — is carried over unchanged from train_headline_v19.py
so that a v19-vs-v20 comparison isolates the one variable (input/output
cleaning breadth).
"""

from unsloth import FastLanguageModel
import json, random, torch, unicodedata
from pathlib import Path
from dataclasses import dataclass
from typing import Any
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer

SINLLAMA_BASE   = "/home/jovyan/work/sinllama/models/SinLLaMA-merged-base"
OUTPUT_ADAPTER  = "/home/jovyan/work/sinllama/models/adapters/headline_sinllama_v20"
TRAIN_DATA_PATH = "/home/jovyan/work/sinllama/data/headline_dataset_48k_balanced_train_clean_v20.jsonl"
VAL_DATA_PATH   = "/home/jovyan/work/sinllama/data/headline_dataset_48k_balanced_val_clean_v20.jsonl"

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
MIN_ARTICLE_CHARS  = 50

# ── Length bands ──
# Ordered shortest-first; a headline's word count picks exactly one band.
# Anything outside 3-10 words has no band to belong to and is dropped rather
# than forced into the nearest one — a 12-word headline labelled "long" would
# teach the model that "long" means "up to 12", which is not what the serving
# contract promises.
HEADLINE_LENGTHS = {
    "short":  {"min_words": 3, "max_words": 5},
    "medium": {"min_words": 6, "max_words": 7},
    "long":   {"min_words": 8, "max_words": 10},
}

# The dataset is not uniform across bands — real Sinhala news headlines cluster
# in the middle — so left alone the model would see far more "medium" than
# "long" and learn to ignore the band line for exactly the case v17 already
# fails. Cap every band at this multiple of the smallest one (downsampling the
# majority, never duplicating the minority: repeated examples at 8 epochs
# overfit the few long headlines that exist). Set to None to train on the raw
# distribution.
BUCKET_BALANCE_RATIO = 2.0


def normalize_sinhala(text):
    return unicodedata.normalize("NFC", text.strip()) if text else ""


def band_for(headline):
    """The band this headline's word count belongs to, or None when it falls
    outside every band."""
    words = len(headline.split())
    for name, band in HEADLINE_LENGTHS.items():
        if band["min_words"] <= words <= band["max_words"]:
            return name
    return None


def is_valid_headline(headline):
    if len(headline) < MIN_HEADLINE_CHARS: return False
    if not any("\u0d80" <= ch <= "\u0dff" for ch in headline): return False
    # Replaces v17's MAX_HEADLINE_WORDS check: band membership is now the
    # length filter, and it bounds both ends.
    return band_for(headline) is not None


def is_valid_article(article):
    return len(article.strip()) >= MIN_ARTICLE_CHARS


def build_prompt(article, length):
    """Byte-identical to v17's build_prompt except for the numbers on the
    rules line, which now come from the example's own band. Must match
    prompt_headline() in backend-api's app/core/prompts.py."""
    article = normalize_sinhala(article)
    if "Article:" in article:
        cat, body = article.split("Article:", 1)
        article = cat.strip() + "\nArticle: " + body.strip()[:MAX_ARTICLE_CHARS]
    else:
        article = article[:MAX_ARTICLE_CHARS]

    band = HEADLINE_LENGTHS[length]
    return (
        "### Instruction:\n"
        "Generate a concise Sinhala news headline for the article below.\n\n"
        "Rules:\n"
        "- Use formal Sinhala journalism style matching the article category\n"
        f"- Between {band['min_words']} and {band['max_words']} words"
        f" -- never fewer than {band['min_words']}\n"
        "- Capture the key person, event, number, or outcome\n"
        "- Output ONLY the headline, nothing else\n\n"
        f"### Input:\n{article}\n\n"
        f"{RESPONSE_MARKER}"
    )


class HeadlineDataset(TorchDataset):
    def __init__(self, records, tokenizer, max_length, balance_ratio=None):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = []
        skipped_empty = skipped_quality = skipped_truncated = 0
        skipped_no_band = 0
        cat_counts = {}
        band_counts = {}

        # Bucket first, balance second, tokenize last — balancing before
        # tokenizing means the discarded majority examples are never paid for.
        by_band = {name: [] for name in HEADLINE_LENGTHS}

        for rec in records:
            article  = rec.get("input",  "").strip()
            headline = rec.get("output", "").strip()

            if not article or not headline:
                skipped_empty += 1; continue

            article_body = article.split("Article:", 1)[1].strip() if "Article:" in article else article
            headline     = normalize_sinhala(headline)

            band = band_for(headline)
            if band is None:
                skipped_no_band += 1; continue
            if not is_valid_headline(headline):
                skipped_quality += 1; continue
            if not is_valid_article(article_body):
                skipped_quality += 1; continue

            cat = "General"
            if "Category:" in article:
                cat = article.split("\n")[0].replace("Category:", "").strip()

            by_band[band].append((article, headline, cat))

        raw_band_counts = {name: len(items) for name, items in by_band.items()}

        if balance_ratio:
            smallest = min(len(items) for items in by_band.values()) or 1
            cap = int(smallest * balance_ratio)
            for name, items in by_band.items():
                if len(items) > cap:
                    random.shuffle(items)
                    by_band[name] = items[:cap]

        for band_name, items in by_band.items():
            for article, headline, cat in items:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
                band_counts[band_name] = band_counts.get(band_name, 0) + 1

                prompt     = build_prompt(article, band_name)
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

        random.shuffle(self.samples)

        hl_tok    = [sum(1 for l in s["labels"] if l != IGNORE_INDEX) for s in self.samples]
        avg_hl    = sum(hl_tok)/len(hl_tok) if hl_tok else 0
        avg_total = sum(len(s["input_ids"]) for s in self.samples)/len(self.samples) if self.samples else 0

        print(f"   Loaded   : {len(self.samples)}")
        print(f"   Skipped  : empty={skipped_empty}  quality={skipped_quality}  "
              f"no_band={skipped_no_band}  truncated={skipped_truncated}")
        print(f"   Avg headline tokens : {avg_hl:.1f}")
        print(f"   Avg total tokens    : {avg_total:.1f}")

        print(f"   Length band distribution (raw -> after balancing):")
        for name in HEADLINE_LENGTHS:
            band = HEADLINE_LENGTHS[name]
            print(f"     {name:8s} ({band['min_words']}-{band['max_words']}w) : "
                  f"{raw_band_counts.get(name, 0):6d} -> {band_counts.get(name, 0):6d}")

        if band_counts:
            min_band, min_count = min(band_counts.items(), key=lambda x: x[1])
            max_count = max(band_counts.values())
            if min_count < max_count * 0.5:
                print(f"\n   WARNING: band '{min_band}' has only {min_count} samples vs "
                      f"{max_count} in the largest. The model will condition on "
                      f"length weakly for that band — consider mining more "
                      f"headlines in that word range before trusting it.")

        print(f"   Category distribution:")
        max_cat = max(cat_counts.values()) if cat_counts else 1
        for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
            bar = chr(9608) * min(30, int(cnt / max_cat * 30))
            print(f"     {cat:20s} : {cnt:5d}  {bar}")

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
    print("  Headline LoRA Training v20 -- length-conditioned, input+output cleaned")
    print("  Bands    : short 3-5 / medium 6-7 / long 8-10 words")
    print("  Dataset  : 48K balanced across 12 categories, re-bucketed by length,")
    print("             scraper tags stripped by clean_headline_dataset.py")
    print(f"  Balancing: cap each band at {BUCKET_BALANCE_RATIO}x the smallest"
          if BUCKET_BALANCE_RATIO else "  Balancing: off (raw distribution)")
    print("  Config   : unchanged from v19 so cleaning breadth is the only variable")
    print("="*70)
    random.seed(SEED)

    if not Path(TRAIN_DATA_PATH).exists() or not Path(VAL_DATA_PATH).exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found. Run scripts/clean_headline_dataset_v20.py "
            f"first to produce:\n  {TRAIN_DATA_PATH}\n  {VAL_DATA_PATH}"
        )

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
    train_ds = HeadlineDataset(train_records, tokenizer, MAX_SEQ_LENGTH,
                               balance_ratio=BUCKET_BALANCE_RATIO)
    print("\nVal set:")
    # Val is left unbalanced on purpose: eval should reflect the real
    # distribution, not the sampling used to train.
    val_ds   = HeadlineDataset(val_records, tokenizer, MAX_SEQ_LENGTH,
                               balance_ratio=None)
    collator  = MaskedCollator(tokenizer=tokenizer, max_length=MAX_SEQ_LENGTH)

    Path(OUTPUT_ADAPTER).mkdir(parents=True, exist_ok=True)

    steps_per_epoch = len(train_ds) // (BATCH_SIZE * GRAD_ACCUMULATION)
    total_steps     = steps_per_epoch * NUM_EPOCHS
    warmup_steps    = int(total_steps * WARMUP_RATIO)

    print(f"\n  Steps per epoch : ~{steps_per_epoch}")
    print(f"  Total steps     : ~{total_steps}")
    print(f"  Warmup steps    : {warmup_steps}")
    print(f"  Eval every      : {EVAL_STEPS} steps (~{EVAL_STEPS/max(steps_per_epoch,1):.2f} epochs)")

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
    print("\nNext: evaluate per band with scripts/test_headline_v20.py, then")
    print("serve_sinai.py picks v20 up automatically (highest version wins).")


if __name__ == "__main__":
    main()
