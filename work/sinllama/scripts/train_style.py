# ============================================================
# SinhalaJournal-LLM
# Sinhala Style Rewriter - LoRA Training v12
#
# Dataset: 7,555 cleaned rows
#
# IMPORTANT:
# - Unsloth MUST be imported first
# - Supports content/style/rewritten_text dataset format
# - Article-level train/validation split
# - No physical style duplication
# - Weighted style sampling
# - EOS-safe labels
# - Strong source/fact preservation
# ============================================================

from unsloth import FastLanguageModel

import json
import torch
import random
from pathlib import Path
from collections import Counter

from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import WeightedRandomSampler

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)


# ============================================================
# PATHS
# ============================================================

SINLLAMA_BASE = (
    "/home/jovyan/work/sinllama/models/SinLLaMA-merged-base"
)

# NEW ADAPTER
OUTPUT_ADAPTER = (
    "/home/jovyan/work/sinllama/models/adapters/style_sinllama_v12"
)

# 7,555-row cleaned dataset
TRAIN_DATA_PATH = (
    "/home/jovyan/style_rewriter/data/"
    "style_dataset2_fixed.jsonl"
)


# ============================================================
# CONFIG
# ============================================================

MAX_SEQ_LENGTH = 4096

# Keep rank 24 for this experiment.
# Do not increase rank and LR simultaneously.
LORA_RANK = 24
LORA_ALPHA = 48
LORA_DROPOUT = 0.05

# IMPORTANT:
# 7,555 clean rows do not require 5 epochs.
NUM_EPOCHS = 3

BATCH_SIZE = 2
GRAD_ACCUMULATION = 8

# Gentle LR for the cleaner dataset
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

TRAIN_SPLIT = 0.85
SEED = 42

WARMUP_RATIO = 0.05

# ============================================================
# GENERATION CONFIG
# ============================================================

GEN_TEMPERATURE = 0.1
GEN_TOP_P = 0.85
GEN_MAX_TOKENS = 1024


# ============================================================
# DATASET QC
# ============================================================

DROP_QC_ISSUES = {
    "missing_required_closing",
    "possible_stutter_duplication",
    "suspiciously_short",
    "honorific_gender_mismatch",
    "foreign_symbol_added",
    "wrong_style_marker_present",
    "possible_numeric_hallucination",
}


# ============================================================
# STYLE IDS
# ============================================================

STYLE_IDS = {
    "style_1_formal_news",
    "style_2_editorial",
    "style_3_sports",
    "style_4_youth",
    "style_5_feature",
}


# ============================================================
# STYLE INSTRUCTIONS
#
# IMPORTANT:
# Keep these identical in your inference script.
# ============================================================

STYLE_RULES = {

    "style_1_formal_news": """
Write the article as a professional Sinhala news report.

- Objective and factual
- Clear journalistic Sinhala
- Most important information first
- Formal vocabulary
- No personal opinion
- No unnecessary commentary
- Preserve all important facts
- Preserve names, numbers, dates and places
- Do not invent information
""",

    "style_2_editorial": """
Rewrite the article in a Sinhala editorial / analytical style.

- Analytical and reflective tone
- Discuss the significance of the reported facts
- Use natural editorial Sinhala
- May use phrases such as "සැලකිය යුතුය" or "සමස්තයක් ලෙස"
  only when they naturally fit
- Do not invent facts
- Do not exaggerate
- Preserve names, numbers, dates and places
- Preserve the factual basis of the original article
""",

    "style_3_sports": """
Rewrite the article in a Sinhala sports-news style.

- Energetic but professional
- Emphasize important actions, events and outcomes
- Use natural sports journalism vocabulary where appropriate
- Do not force sports terminology into non-sports stories
- Do not invent sporting events or actions
- Preserve all facts, names, numbers and dates
""",

    "style_4_youth": """
Rewrite the article in a modern, accessible Sinhala style
suitable for younger readers.

- Conversational but grammatically correct Sinhala
- Simple and clear sentences
- Engaging tone
- Natural modern vocabulary
- Do not force greetings into every article
- Do not use "යාලුවනේ" unless it naturally fits
- Do not add information
- Preserve names, numbers, dates and factual details
""",

    "style_5_feature": """
Rewrite the article in a Sinhala feature-writing style.

- Engaging narrative presentation
- More descriptive language where appropriate
- Smooth transitions
- Human-interest perspective where supported by the source
- Do not invent scenes, emotions or events
- Do not fabricate background information
- Preserve all factual information
"""
}


# ============================================================
# PROMPT FORMAT
# ============================================================

def format_prompt(
    instruction: str,
    article: str,
    rewritten: str = ""
) -> str:

    return (
        "### Instruction:\n"
        f"{instruction.strip()}\n\n"

        "### FACT PRESERVATION RULES:\n"
        "1. Preserve the meaning of every factual statement.\n"
        "2. Do not add facts that are absent from the source.\n"
        "3. Do not remove important factual information.\n"
        "4. Preserve all person names exactly.\n"
        "5. Preserve all organization names exactly.\n"
        "6. Preserve all locations exactly.\n"
        "7. Preserve numbers and numerical values exactly.\n"
        "8. Preserve dates exactly.\n"
        "9. Preserve quoted text exactly.\n"
        "10. Preserve gender-marked honorifics such as "
        "මහතා / මහත්මිය.\n"
        "11. Do not invent durations, measurements or statistics.\n"
        "12. Do not translate proper names unnecessarily.\n"
        "13. Do not introduce unrelated information.\n"
        "14. Change the writing STYLE, not the underlying facts.\n"
        "15. Use natural Sinhala grammar and morphology.\n"
        "16. Do not deliberately replace correct Sinhala words with "
        "different words merely to make the sentence different.\n\n"

        "### Input:\n"
        f"{article.strip()}\n\n"

        "### Response:\n"
        f"{rewritten}"
    )


# ============================================================
# DATA CONVERSION
# ============================================================

def convert_record(rec: dict):
    """
    Converts the cleaned dataset into the trainer format.

    Expected source fields:

        content
        style
        rewritten_text
        qc_issues
        status
        url
        category

    """

    # ----------------------------------------
    # Failed generation
    # ----------------------------------------

    if rec.get("status") == "failed":
        return None

    if rec.get("error"):
        return None

    # ----------------------------------------
    # Style
    # ----------------------------------------

    style_id = rec.get("style")

    if style_id not in STYLE_IDS:
        return None

    # ----------------------------------------
    # Content
    # ----------------------------------------

    article = str(
        rec.get("content") or ""
    ).strip()

    rewritten = str(
        rec.get("rewritten_text") or ""
    ).strip()

    if not article:
        return None

    if not rewritten:
        return None

    # ----------------------------------------
    # Minimum output length
    # ----------------------------------------

    if len(rewritten) < 50:
        return None

    # ----------------------------------------
    # QC
    # ----------------------------------------

    qc_issues = rec.get("qc_issues") or []

    if isinstance(qc_issues, str):
        qc_issues = [qc_issues]

    qc_issues = set(qc_issues)

    if qc_issues.intersection(DROP_QC_ISSUES):
        return None

    # ----------------------------------------
    # Metadata
    # ----------------------------------------

    return {
        "instruction": STYLE_RULES[style_id].strip(),
        "input": article,
        "output": rewritten,
        "metadata": {
            "style_id": style_id,
            "url": rec.get("url"),
            "category": rec.get("category"),
        }
    }


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path: str):

    records = []

    with open(path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError as e:

                print(
                    f"⚠️ Invalid JSON at line {line_number}: {e}"
                )

    return records


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_records(records):

    valid = []

    stats = Counter()

    for rec in records:

        converted = convert_record(rec)

        if converted is None:

            stats["skipped"] += 1
            continue

        valid.append(converted)

        stats[
            converted["metadata"]["style_id"]
        ] += 1

    print("\n" + "=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    print(
        f"Raw records     : {len(records):,}"
    )

    print(
        f"Valid records   : {len(valid):,}"
    )

    print(
        f"Skipped records : {stats['skipped']:,}"
    )

    print("\nStyle distribution:")

    for style_id in sorted(STYLE_IDS):

        print(
            f"  {style_id:<28} "
            f"{stats[style_id]:>5,}"
        )

    return valid


# ============================================================
# REMOVE DUPLICATE ARTICLE+STYLE PAIRS
# ============================================================

def remove_duplicates(records):

    seen = set()
    unique = []

    duplicates = 0

    for rec in records:

        article = rec["input"]
        style = rec["metadata"]["style_id"]

        key = (
            style,
            article.strip()
        )

        if key in seen:

            duplicates += 1
            continue

        seen.add(key)
        unique.append(rec)

    print(
        f"\nDuplicate article/style rows removed: "
        f"{duplicates:,}"
    )

    return unique


# ============================================================
# ARTICLE-LEVEL SPLIT
# ============================================================

def article_level_split(records):

    by_article = {}

    for rec in records:

        url = rec["metadata"].get("url")

        # URL is preferred because the same article exists
        # under multiple styles.
        if url:

            key = url

        else:

            # Fallback: article content itself.
            key = rec["input"]

        by_article.setdefault(key, []).append(rec)

    article_keys = list(by_article.keys())

    random.shuffle(article_keys)

    split_index = int(
        len(article_keys) * TRAIN_SPLIT
    )

    train_keys = set(
        article_keys[:split_index]
    )

    val_keys = set(
        article_keys[split_index:]
    )

    train_records = [
        rec
        for key in train_keys
        for rec in by_article[key]
    ]

    val_records = [
        rec
        for key in val_keys
        for rec in by_article[key]
    ]

    print("\n" + "=" * 60)
    print("ARTICLE-LEVEL SPLIT")
    print("=" * 60)

    print(
        f"Unique articles : {len(article_keys):,}"
    )

    print(
        f"Train articles  : {len(train_keys):,}"
    )

    print(
        f"Val articles    : {len(val_keys):,}"
    )

    print(
        f"Train rows      : {len(train_records):,}"
    )

    print(
        f"Val rows        : {len(val_records):,}"
    )

    return train_records, val_records


# ============================================================
# DATASET
# ============================================================

class StyleRewriterDataset(TorchDataset):

    def __init__(
        self,
        records,
        tokenizer,
        max_length
    ):

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples = []

        self.truncated = 0
        self.short_outputs = 0

        for rec in records:

            instruction = (
                rec.get("instruction") or ""
            ).strip()

            article = (
                rec.get("input") or ""
            ).strip()

            rewritten = (
                rec.get("output") or ""
            ).strip()

            metadata = rec.get(
                "metadata",
                {}
            )

            style_id = metadata.get(
                "style_id",
                "unknown"
            )

            if not instruction:
                continue

            if not article:
                continue

            if not rewritten:
                continue

            if len(rewritten) < 50:

                self.short_outputs += 1
                continue

            self.samples.append({

                "instruction": instruction,

                "input": article,

                "output": rewritten,

                "full_text": format_prompt(
                    instruction,
                    article,
                    rewritten
                ),

                "style_id": style_id,

                "metadata": metadata,
            })

        self._print_stats()

        print(
            f"   Tokenizing {len(self.samples):,} samples..."
        )


    # --------------------------------------------------------
    # Stats
    # --------------------------------------------------------

    def _print_stats(self):

        counts = Counter(
            s["style_id"]
            for s in self.samples
        )

        print("\nStyle distribution:")

        for style_id in sorted(STYLE_IDS):

            print(
                f"  {style_id:<28}"
                f"{counts[style_id]:>5,}"
            )

        if self.short_outputs:

            print(
                f"\n⚠️ Short outputs removed: "
                f"{self.short_outputs}"
            )


    # --------------------------------------------------------
    # Length
    # --------------------------------------------------------

    def __len__(self):

        return len(self.samples)


    # --------------------------------------------------------
    # Item
    # --------------------------------------------------------

    def __getitem__(self, idx):

        sample = self.samples[idx]

        # ----------------------------------------------------
        # Full training sequence
        # ----------------------------------------------------

        full_text = sample["full_text"]

        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        input_ids = encoded["input_ids"]

        # ----------------------------------------------------
        # Prompt-only sequence
        # ----------------------------------------------------

        prompt_text = format_prompt(
            sample["instruction"],
            sample["input"],
            ""
        )

        prompt_encoded = self.tokenizer(
            prompt_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        prompt_len = len(
            prompt_encoded["input_ids"]
        )

        # ----------------------------------------------------
        # EOS
        # ----------------------------------------------------

        eos_id = self.tokenizer.eos_token_id

        if (
            eos_id is not None
            and len(input_ids) < self.max_length
            and (
                not input_ids
                or input_ids[-1] != eos_id
            )
        ):

            input_ids = (
                input_ids + [eos_id]
            )

        # ----------------------------------------------------
        # Labels
        #
        # IMPORTANT:
        # Prompt tokens = -100
        # Output tokens = trainable
        # ----------------------------------------------------

        prompt_len = min(
            prompt_len,
            len(input_ids)
        )

        labels = (
            [-100] * prompt_len
            +
            input_ids[prompt_len:]
        )

        # ----------------------------------------------------
        # Safety
        # ----------------------------------------------------

        if len(labels) != len(input_ids):

            labels = labels[:len(input_ids)]

            if len(labels) < len(input_ids):

                labels += (
                    [-100]
                    *
                    (
                        len(input_ids)
                        -
                        len(labels)
                    )
                )

        if len(input_ids) >= self.max_length:

            self.truncated += 1

        return {

            "input_ids": input_ids,

            "attention_mask": [
                1
            ] * len(input_ids),

            "labels": labels,
        }


# ============================================================
# STYLE-WEIGHTED TRAINER
#
# Instead of physically duplicating minority rows,
# sample them with higher probability.
# ============================================================

class StyleBalancedTrainer(Trainer):

    def get_train_dataloader(self):

        dataset = self.train_dataset

        style_counts = Counter(
            sample["style_id"]
            for sample in dataset.samples
        )

        # Inverse-frequency weighting.
        #
        # Example:
        # 2,400 samples -> lower weight
        #   500 samples -> higher weight
        #
        # This prevents physically duplicating rows.
        weights = []

        for sample in dataset.samples:

            style = sample["style_id"]

            count = style_counts[style]

            weight = 1.0 / max(
                count,
                1
            )

            weights.append(weight)

        weights = torch.tensor(
            weights,
            dtype=torch.double
        )

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
        )

        return torch.utils.data.DataLoader(

            dataset,

            batch_size=(
                self.args.per_device_train_batch_size
            ),

            sampler=sampler,

            collate_fn=self.data_collator,

            num_workers=0,

            pin_memory=True,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        " SinhalaJournal-LLM | "
        "Style Rewriter LoRA Training v12"
    )

    print(
        " Dataset: 7,555-row cleaned dataset"
    )

    print(
        " Improvements:"
    )

    print(
        "   ✓ Article-level split"
    )

    print(
        "   ✓ No physical style duplication"
    )

    print(
        "   ✓ Weighted style sampling"
    )

    print(
        "   ✓ EOS-safe training"
    )

    print(
        "   ✓ Strong fact preservation"
    )

    print(
        "   ✓ Cleaner youth/editorial instructions"
    )

    print(
        "   ✓ 3 epochs for 7,555 clean rows"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # SEED
    # ========================================================

    random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(SEED)


    # ========================================================
    # TOKENIZER
    # ========================================================

    print(
        "\n🔹 Loading tokenizer..."
    )

    tokenizer = AutoTokenizer.from_pretrained(

        SINLLAMA_BASE,

        local_files_only=True,
    )

    tokenizer.pad_token = (
        tokenizer.eos_token
    )

    tokenizer.padding_side = "right"

    print(
        f"   Vocab size: "
        f"{len(tokenizer):,}"
    )

    print(
        f"   Model max length: "
        f"{tokenizer.model_max_length}"
    )


    # ========================================================
    # MODEL
    # ========================================================

    print(
        f"\n🔹 Loading SinLLaMA "
        f"(max_seq={MAX_SEQ_LENGTH})..."
    )

    model, _ = FastLanguageModel.from_pretrained(

        model_name=SINLLAMA_BASE,

        max_seq_length=MAX_SEQ_LENGTH,

        dtype=torch.bfloat16,

        load_in_4bit=True,

        local_files_only=True,

        attn_implementation="eager",
    )

    print(
        "   SinLLaMA base loaded ✅"
    )


    # ========================================================
    # LORA
    # ========================================================

    print(
        f"\n🔹 Adding LoRA "
        f"(rank={LORA_RANK}, "
        f"alpha={LORA_ALPHA})..."
    )

    model = FastLanguageModel.get_peft_model(

        model,

        r=LORA_RANK,

        lora_alpha=LORA_ALPHA,

        lora_dropout=LORA_DROPOUT,

        bias="none",

        use_gradient_checkpointing=True,

        target_modules=[

            "q_proj",

            "k_proj",

            "v_proj",

            "o_proj",

            "gate_proj",

            "up_proj",

            "down_proj",
        ],
    )


    trainable = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad
    )

    total = sum(

        p.numel()

        for p in model.parameters()
    )

    print(
        f"   Trainable: "
        f"{trainable:,} / "
        f"{total:,}"
    )

    print(
        f"   Trainable %: "
        f"{100 * trainable / total:.2f}%"
    )


    # ========================================================
    # LOAD DATA
    # ========================================================

    print(
        f"\n📂 Loading dataset:"
    )

    print(
        f"   {TRAIN_DATA_PATH}"
    )

    records = load_jsonl(
        TRAIN_DATA_PATH
    )


    # ========================================================
    # VALIDATE
    # ========================================================

    records = validate_records(
        records
    )


    # ========================================================
    # DUPLICATES
    # ========================================================

    records = remove_duplicates(
        records
    )


    # ========================================================
    # EXPECTED SIZE
    # ========================================================

    print(
        f"\n📌 Final usable rows: "
        f"{len(records):,}"
    )

    if len(records) < 7000:

        print(
            "\n⚠️ WARNING:"
        )

        print(
            "Expected approximately 7,555 "
            "clean rows."
        )

        print(
            f"Only {len(records):,} rows "
            "are currently usable."
        )


    if len(records) < 100:

        print(
            "\n❌ Not enough data for training."
        )

        return


    # ========================================================
    # ARTICLE SPLIT
    # ========================================================

    train_records, val_records = (
        article_level_split(records)
    )


    # ========================================================
    # TRAIN DATASET
    # ========================================================

    print(
        "\n📊 Building training dataset..."
    )

    train_dataset = StyleRewriterDataset(

        train_records,

        tokenizer,

        MAX_SEQ_LENGTH
    )


    # ========================================================
    # VALIDATION DATASET
    # ========================================================

    print(
        "\n📊 Building validation dataset..."
    )

    val_dataset = StyleRewriterDataset(

        val_records,

        tokenizer,

        MAX_SEQ_LENGTH
    )


    print("\n" + "=" * 60)

    print("FINAL TRAINING DATA")

    print("=" * 60)

    print(
        f"Train samples : "
        f"{len(train_dataset):,}"
    )

    print(
        f"Val samples   : "
        f"{len(val_dataset):,}"
    )


    # ========================================================
    # COLLATOR
    # ========================================================

    collator = DataCollatorForSeq2Seq(

        tokenizer,

        model=model,

        padding=True,

        pad_to_multiple_of=8,
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    Path(
        OUTPUT_ADAPTER
    ).mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # TRAINING ARGS
    # ========================================================

    training_args = TrainingArguments(

        output_dir=OUTPUT_ADAPTER,

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,

        per_device_eval_batch_size=BATCH_SIZE,

        gradient_accumulation_steps=(
            GRAD_ACCUMULATION
        ),

        learning_rate=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

        warmup_ratio=WARMUP_RATIO,

        bf16=True,

        logging_steps=10,

        eval_strategy="steps",

        eval_steps=100,

        save_strategy="steps",

        save_steps=100,

        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",

        greater_is_better=False,

        save_total_limit=3,

        report_to="none",

        seed=SEED,

        data_seed=SEED,

        dataloader_num_workers=0,

        remove_unused_columns=False,

        max_grad_norm=1.0,

        lr_scheduler_type="cosine",

        optim="adamw_torch",
    )


    # ========================================================
    # TRAINER
    # ========================================================

    trainer = StyleBalancedTrainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=val_dataset,

        data_collator=collator,
    )


    # ========================================================
    # ESTIMATE STEPS
    # ========================================================

    effective_batch = (
        BATCH_SIZE
        *
        GRAD_ACCUMULATION
    )

    steps_per_epoch = max(

        1,

        len(train_dataset)
        //
        effective_batch
    )

    total_steps = (
        steps_per_epoch
        *
        NUM_EPOCHS
    )


    print("\n" + "=" * 60)

    print("TRAINING CONFIGURATION")

    print("=" * 60)

    print(
        f"Epochs           : "
        f"{NUM_EPOCHS}"
    )

    print(
        f"Max sequence     : "
        f"{MAX_SEQ_LENGTH}"
    )

    print(
        f"Batch size       : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Gradient accum.  : "
        f"{GRAD_ACCUMULATION}"
    )

    print(
        f"Effective batch   : "
        f"{effective_batch}"
    )

    print(
        f"Learning rate    : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Warmup ratio     : "
        f"{WARMUP_RATIO}"
    )

    print(
        f"LoRA rank        : "
        f"{LORA_RANK}"
    )

    print(
        f"Approx steps     : "
        f"{total_steps:,}"
    )

    print(
        f"Output           : "
        f"{OUTPUT_ADAPTER}"
    )

    print("=" * 60)


    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "\n🚀 Starting training..."
    )

    trainer.train()


    # ========================================================
    # SAVE
    # ========================================================

    print(
        "\n💾 Saving adapter..."
    )

    model.save_pretrained(
        OUTPUT_ADAPTER
    )

    tokenizer.save_pretrained(
        OUTPUT_ADAPTER
    )


    # ========================================================
    # TRAINING CONFIG
    # ========================================================

    config = {

        "version": "v12",

        "dataset_rows": len(records),

        "train_samples": len(train_dataset),

        "val_samples": len(val_dataset),

        "max_seq_length":
            MAX_SEQ_LENGTH,

        "lora_rank":
            LORA_RANK,

        "lora_alpha":
            LORA_ALPHA,

        "lora_dropout":
            LORA_DROPOUT,

        "num_epochs":
            NUM_EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "batch_size":
            BATCH_SIZE,

        "gradient_accumulation":
            GRAD_ACCUMULATION,

        "effective_batch":
            effective_batch,

        "warmup_ratio":
            WARMUP_RATIO,

        "train_split":
            TRAIN_SPLIT,

        "seed":
            SEED,

        "train_data_path":
            TRAIN_DATA_PATH,

        "truncated_train":
            train_dataset.truncated,

        "truncated_val":
            val_dataset.truncated,

        "style_balancing":
            "weighted_random_sampler",

        "style_rules":
            STYLE_RULES,
    }


    with open(

        f"{OUTPUT_ADAPTER}/"
        "training_config.json",

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            config,

            f,

            ensure_ascii=False,

            indent=2
        )


    # ========================================================
    # DONE
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "✅ TRAINING COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Adapter: "
        f"{OUTPUT_ADAPTER}"
    )

    print(
        f"Train samples: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset):,}"
    )

    print(
        f"Truncated train: "
        f"{train_dataset.truncated:,}"
    )

    print(
        f"Truncated validation: "
        f"{val_dataset.truncated:,}"
    )

    print(
        "\n➡️ Next:"
    )

    print(
        "Run your inference/evaluation script "
        "against the new v12 adapter."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()