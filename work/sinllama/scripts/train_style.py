# Unsloth MUST be first import
from unsloth import FastLanguageModel

import json
import torch
import random
from pathlib import Path
from collections import Counter
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
# ✅ UPDATED: v08 → v09, trained on the new, much larger dataset below
OUTPUT_ADAPTER  = "/home/jovyan/work/sinllama/models/adapters/style_sinllama_v10"

# ✅ UPDATED: points at the new dataset (~22,237 articles - up from ~2,173
# for v08). NOTE: confirm this exact filename/path matches what you
# generated - update if it differs.
TRAIN_DATA_PATH = "/home/jovyan/style_rewriter/data/style_dataset2_dub.jsonl"


# ──────────────────────────────────────────────
# CONFIG - UPDATED
# ──────────────────────────────────────────────
# ✅ INCREASED: 2048 → 4096 (Sinhala needs more tokens)
MAX_SEQ_LENGTH    = 4096
# ✅ RAISED: 16 → 24. Rank was cut 32→16 to fix v08's overfitting, and
# that worked (train/eval gap went 2.28x → 1.26x, eval_loss 1.087 →
# 0.849). But character/morphology corruption persisted through that
# fix, suggesting 16 may have OVERcorrected and now limits fluency
# capacity. 24 is a middle ground: more capacity for clean Sinhala
# morphology, while weight_decay + 3 epochs + the much larger dataset
# still guard against the overfitting that rank 32 had.
LORA_RANK         = 24
LORA_ALPHA        = 48          # kept at 2x rank, same ratio as before
LORA_DROPOUT       = 0.05
# ✅ REDUCED: 5 → 3. Same overfitting evidence as above - with ~10x more
# articles than the v08 run, fewer passes are needed, and the v08
# train/eval loss gap suggests 5 epochs was already too many for the
# effective amount of genuinely new content per epoch (a lot of every
# example's tokens are repeated style boilerplate, which the model can
# memorize quickly).
NUM_EPOCHS        = 3
BATCH_SIZE        = 2
GRAD_ACCUMULATION = 8
LEARNING_RATE     = 2e-4
# ✅ NEW: was not set at all before (defaults to 0.0 in TrainingArguments).
# Standard, low-cost overfitting countermeasure - penalizes large weights,
# directly discouraging the kind of memorization behind the train/eval
# loss gap seen in v08.
WEIGHT_DECAY      = 0.01
TRAIN_SPLIT       = 0.85
SEED              = 42
WARMUP_STEPS      = 50          # Increased from 16

# ✅ NEW: generation parameters for inference
GEN_TEMPERATURE   = 0.1         # Lower = more deterministic
GEN_TOP_P         = 0.85
GEN_MAX_TOKENS    = 1024        # For inference testing

# ✅ NEW: drop rows flagged by generate_style_dataset.py's check_quality()
# unless explicitly allowed here. Keeping this strict by default since
# training on stutter/truncation artifacts would teach the model to
# reproduce them.
DROP_QC_ISSUES = {
    "missing_required_closing",
    "possible_stutter_duplication",
    "suspiciously_short",
    # ✅ NEW: added after manually reviewing v09 outputs found a
    # misgendered real person, an English word inserted into a direct
    # quote, and a fabricated duration - see generate_style_dataset.py's
    # check_quality() for the detection logic.
    "honorific_gender_mismatch",
    "foreign_symbol_added",
    "wrong_style_marker_present",
    "possible_numeric_hallucination",
}

# Style IDs - MUST stay byte-for-byte identical to the keys used in
# generate_style_dataset.py's STYLE_INSTRUCTIONS, since that's what
# produced the "style" field in the dataset.
STYLE_IDS = {
    "style_1_formal_news",
    "style_2_editorial",
    "style_3_sports",
    "style_4_youth",
    "style_5_feature",
}

# ✅ NEW: style-specific rules for better guidance.
# NOTE: this is the INFERENCE-TIME instruction shown to the model during
# training and later during generation from the trained adapter. It does
# NOT need to match the Sinhala prompt used in generate_style_dataset.py
# (that one was for the teacher model / NVIDIA API that produced the
# training data) - but it DOES need to stay byte-for-byte identical
# between this training script and whatever inference/serving script
# you use afterward, or the adapter will see a different prompt than
# it was trained on.
STYLE_RULES = {
    "style_4_youth": """
- Start with casual greeting (දන්නවද? / ඇහුවද? / මේක අහන්න!)
- Use casual Sinhala: ගොඩක්, ටිකක්, හිතෙනවා
- Short, punchy sentences
- End with: ඒ නිසා යාලුවනේ, මේ ගැන අනිවාර්යයෙන්ම දැනගන්න!
""",
    "style_3_sports": """
- Lead with most dramatic fact
- Action verbs: පහර දුන්නා, ජයග්‍රහණය, ප්‍රහාරය
""",
    "style_2_editorial": """
- Start with: විශ්ලේෂණය කරන විට
- Use first person plural: අපි, අපට
- Add analytical language: සැලකිය යුතුය, සමස්තයක් ලෙස
- Express viewpoint on facts
""",
    "style_1_formal_news": """
- Objective, passive voice
- Inverted pyramid - most important fact first
- No opinion language
- Keep ALL facts in same order
""",
    "style_5_feature": """
- Narrative opening: set the scene
- Descriptive language
- Present tense where possible
- Human angle emphasized
"""
}


# ──────────────────────────────────────────────
# PROMPT FORMAT - IMPROVED
# ──────────────────────────────────────────────
def format_prompt(instruction: str, article: str, rewritten: str) -> str:
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### IMPORTANT RULES:\n"
        "1. Keep ALL the same facts in the EXACT same order\n"
        "2. Maintain approximately the SAME length as the original\n"
        "3. Do NOT add unrelated content (no greetings, no cricket references unless in original)\n"
        "4. Do NOT change the meaning of any fact\n"
        "5. Do NOT add new facts or events\n"
        "6. Apply the style while preserving ALL factual content\n"
        # ✅ NEW: eval showed low BLEU/ROUGE-2 despite good style
        # markers - a likely cause is the model paraphrasing names,
        # numbers, and dates while restyling the surrounding sentence,
        # which breaks exact n-gram overlap even when the fact itself
        # is preserved. This makes that constraint explicit.
        "7. Keep all names, numbers, and dates in EXACTLY their original "
        "wording - only change sentence structure and tone around them\n"
        # ✅ NEW: manual review of v09 outputs found the model flipping
        # a real female minister's honorific (මහත්මිය -> මහතා) in 3 of 5
        # styles, altering text inside direct quotes, and inventing a
        # duration that appeared nowhere in the source. Decoding params
        # can't cause those - they're learned from training data, so the
        # constraints have to be in the training prompt itself.
        "8. Preserve gender-marked honorifics EXACTLY (මහත්මිය stays "
        "මහත්මිය, මහතා stays මහතා) - never change a person's honorific "
        "or implied gender\n"
        "9. Text inside quotation marks must be copied VERBATIM from the "
        "original - never reword, translate, or insert words into a quote\n"
        "10. Never introduce symbols, numbers, durations, or dates that "
        "do not appear in the source article\n\n"
        "### Input:\n"
        f"{article}\n\n"
        "### Response:\n"
        f"{rewritten}"
    )


# ──────────────────────────────────────────────
# ✅ NEW: CONVERSION FROM generate_style_dataset.py SCHEMA
# ──────────────────────────────────────────────
def convert_generated_record(rec: dict) -> dict | None:
    """
    Converts one row from generate_style_dataset.py's output format
    (content / style / rewritten_text / qc_issues / status) into the
    instruction / input / output / metadata format this trainer expects.

    Returns None if the row should be skipped entirely (failed
    generation, unknown style, or flagged by QC).
    """
    # Skip rows that failed generation outright (status: "failed" or
    # the "empty content" error path from generate_style_dataset.py).
    if rec.get("status") == "failed" or rec.get("error"):
        return None

    style_id = rec.get("style")
    if style_id not in STYLE_IDS:
        return None

    article   = (rec.get("content") or "").strip()
    rewritten = (rec.get("rewritten_text") or "").strip()
    if not article or not rewritten:
        return None

    # Skip rows flagged by generate_style_dataset.py's check_quality()
    # (missing required closing line, stutter/duplication artifacts,
    # suspiciously short output) unless that issue type was excluded
    # from DROP_QC_ISSUES above.
    qc_issues = set(rec.get("qc_issues", []))
    if qc_issues & DROP_QC_ISSUES:
        return None

    return {
        "instruction": STYLE_RULES.get(style_id, "").strip(),
        "input":       article,
        "output":      rewritten,
        "metadata": {
            "style_id": style_id,
            "url":      rec.get("url"),
            "category": rec.get("category"),
        },
    }


# ──────────────────────────────────────────────
# DATASET - WITH BETTER VALIDATION
# ──────────────────────────────────────────────
class StyleRewriterDataset(TorchDataset):
    def __init__(self, records: list, tokenizer, max_length: int):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = []
        self.truncated  = 0
        self.zero_length = 0

        for rec in records:
            instruction = rec.get("instruction", "").strip()
            article     = rec.get("input", "").strip()
            rewritten   = rec.get("output", "").strip()
            metadata    = rec.get("metadata", {})
            style_id    = metadata.get("style_id", "unknown")

            # ✅ NEW: filter out very short or empty outputs
            if len(rewritten) < 50:
                self.zero_length += 1
                continue

            if instruction and article and rewritten:
                self.samples.append({
                    "instruction": instruction,
                    "input":       article,
                    "output":      rewritten,
                    "full_text":   format_prompt(instruction, article, rewritten),
                    "style_id":    style_id,
                    "metadata":    metadata,
                })

        self._log_stats()
        print(f"   Tokenizing {len(self.samples)} samples...")

    def _log_stats(self):
        style_counts = Counter(s["style_id"] for s in self.samples)
        print(f"\n   Style distribution ({len(self.samples)} total):")
        for style, count in sorted(style_counts.items()):
            bar = "█" * (count // max(1, len(self.samples) // 40))
            print(f"     {style:<28} {count:>4}  {bar}")

        if self.zero_length:
            print(f"\n   ⚠️  Skipped {self.zero_length} records with too short output")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        encoded = self.tokenizer(
            sample["full_text"],
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        prompt_only = format_prompt(
            sample["instruction"], sample["input"], ""
        )
        prompt_len = len(self.tokenizer(
            prompt_only,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"])

        labels = [-100] * prompt_len + encoded["input_ids"][prompt_len:]

        if len(encoded["input_ids"]) == self.max_length:
            self.truncated += 1

        encoded["labels"] = labels
        return encoded


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
def load_jsonl(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate_records(records: list) -> list:
    """
    ✅ UPDATED: now converts raw generate_style_dataset.py rows
    (content / style / rewritten_text) into the instruction / input /
    output / metadata shape this trainer needs, dropping failed
    generations, unknown styles, QC-flagged rows, and anything with a
    too-short output - all in one pass.
    """
    valid, skipped, qc_dropped, failed_dropped = [], 0, 0, 0

    for rec in records:
        if rec.get("status") == "failed" or rec.get("error"):
            failed_dropped += 1
            continue

        qc_issues = set(rec.get("qc_issues", []))
        if qc_issues & DROP_QC_ISSUES:
            qc_dropped += 1
            continue

        converted = convert_generated_record(rec)
        if converted is None:
            skipped += 1
            continue

        if len(converted["output"].strip()) > 50:
            valid.append(converted)
        else:
            skipped += 1

    if failed_dropped:
        print(f"   ⚠️  Dropped {failed_dropped} records that failed generation")
    if qc_dropped:
        print(f"   ⚠️  Dropped {qc_dropped} records flagged by generation-time QC checks")
    if skipped:
        print(f"   ⚠️  Skipped {skipped} other records (missing fields or too short)")

    return valid


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("\n" + "="*58)
    print("  SinhalaJournal-LLM | Style Rewriter LoRA Training v03")
    print("  Fixed: Longer sequences, better prompts, style rules")
    print("  Updated: reads generate_style_dataset.py output directly")
    print("="*58)

    random.seed(SEED)

    # ── Step 1: Load tokenizer ───────────────────────────────────
    print(f"\n🔹 Loading tokenizer from pre-merged base...")
    tokenizer = AutoTokenizer.from_pretrained(
        SINLLAMA_BASE,
        local_files_only=True,
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"   Vocab size: {len(tokenizer):,} tokens")
    print(f"   Max token limit: {tokenizer.model_max_length}")

    # ── Step 2: Load pre-merged SinLLaMA base ───────────────────
    print(f"\n🔹 Loading pre-merged SinLLaMA base (4bit, max_seq={MAX_SEQ_LENGTH})...")
    model, _ = FastLanguageModel.from_pretrained(
        model_name          = SINLLAMA_BASE,
        max_seq_length      = MAX_SEQ_LENGTH,
        dtype               = torch.bfloat16,
        load_in_4bit        = True,
        local_files_only    = True,
        attn_implementation = "eager",
    )
    print(f"   SinLLaMA base loaded ✅")

    # ── Step 3: Add style rewriter LoRA ─────────────────────────
    print(f"\n🔹 Adding style rewriter LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})...")
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

    # ── Step 4: Load & validate data ────────────────────────────
    print(f"\n📂 Loading data from: {TRAIN_DATA_PATH}")
    records = load_jsonl(TRAIN_DATA_PATH)
    print(f"   Raw records   : {len(records)}")
    records = validate_records(records)
    print(f"   Valid records : {len(records)}")

    # ✅ NEW: Check if we have enough data
    if len(records) < 20:
        print(f"   ❌ ERROR: Only {len(records)} valid records. Need at least 100 for training.")
        print(f"   Please add more data to {TRAIN_DATA_PATH}")
        return

    # ✅ NEW: article-level stratified split so the same source article
    # never appears in both train and val under a different style -
    # otherwise val "leaks" seeing the same facts it trained on.
    by_url = {}
    for rec in records:
        url = rec["metadata"].get("url") or id(rec)
        by_url.setdefault(url, []).append(rec)

    urls = list(by_url.keys())
    random.shuffle(urls)
    n_train_urls = int(len(urls) * TRAIN_SPLIT)
    train_urls = set(urls[:n_train_urls])

    train_records = [r for u in urls if u in train_urls for r in by_url[u]]
    val_records   = [r for u in urls if u not in train_urls for r in by_url[u]]

    print(f"\n📊 Train split ({len(train_records)} records, {n_train_urls} articles):")
    train_dataset = StyleRewriterDataset(train_records, tokenizer, MAX_SEQ_LENGTH)

    print(f"\n📊 Validation split ({len(val_records)} records, {len(urls) - n_train_urls} articles):")
    val_dataset   = StyleRewriterDataset(val_records, tokenizer, MAX_SEQ_LENGTH)

    print(f"\n   Train samples : {len(train_dataset)}")
    print(f"   Val   samples : {len(val_dataset)}")

    # ── Step 5: Collator & training args ────────────────────────
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model              = model,
        padding            = True,
        pad_to_multiple_of = 8,
    )

    Path(OUTPUT_ADAPTER).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir                  = OUTPUT_ADAPTER,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUMULATION,
        learning_rate               = LEARNING_RATE,
        weight_decay                = WEIGHT_DECAY,
        warmup_steps                = WARMUP_STEPS,
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
        max_grad_norm               = 1.0,
        lr_scheduler_type           = "cosine",
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
    print(f"   Max seq length  : {MAX_SEQ_LENGTH}")
    print(f"   Effective batch : {BATCH_SIZE * GRAD_ACCUMULATION}")
    print(f"   ~Total steps    : {total_steps}")
    print(f"   Output dir      : {OUTPUT_ADAPTER}")
    print(f"\n   ✅ Watch for: loss decreasing, grad_norm > 0, eval_loss dropping\n")

    trainer.train()

    # ── Step 7: Save ─────────────────────────────────────────────
    print(f"\n💾 Saving style rewriter adapter to {OUTPUT_ADAPTER}...")
    model.save_pretrained(OUTPUT_ADAPTER)
    tokenizer.save_pretrained(OUTPUT_ADAPTER)

    # ✅ NEW: Save training config for reference
    config = {
        "max_seq_length": MAX_SEQ_LENGTH,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "grad_accumulation": GRAD_ACCUMULATION,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "truncated_train": train_dataset.truncated,
        "truncated_val": val_dataset.truncated,
        "train_data_path": TRAIN_DATA_PATH,
    }
    with open(f"{OUTPUT_ADAPTER}/training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"   Adapter saved : {OUTPUT_ADAPTER}")
    print(f"\n   📊 Summary:")
    print(f"   - Total train samples: {len(train_dataset)}")
    print(f"   - Truncated samples: {train_dataset.truncated}")
    print(f"   - Config saved to: {OUTPUT_ADAPTER}/training_config.json")
    print(f"\n   ➡  Next: python test_style_rewriter.py\n")


if __name__ == "__main__":
    main()