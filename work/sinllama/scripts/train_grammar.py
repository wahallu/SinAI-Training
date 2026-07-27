import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import json

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# ✅ NEW: point directly to pre-merged SinLLaMA base
#    Same as test script — no more 4-step chain
SINLLAMA_BASE = "./models/SinLLaMA-merged-base"
# ⚠️ UPDATE THIS to whatever you name cleaned_v5.jsonl on this box
#    (previous rounds: cleaned_v3 -> ...stage6 -> v16, cleaned_v4 -> v17)
DATA_PATH     = "data/grammar_manual_dataset_stage9.jsonl"
OUTPUT_DIR    = "./models/adapters/grammar_sinllama_v19"

# ✅ v18 changes BOTH data and LoRA capacity at once (user-directed).
#    NOTE: this deliberately breaks the one-variable-at-a-time rule the
#    earlier rounds followed — if v18 improves, we won't know how much
#    came from the MLP/r=32 capacity bump vs. the cleaned_v5 data. That
#    tradeoff was accepted to save a ~2.5h train cycle. If v18 REGRESSES,
#    re-run once with the LoRA block reverted to r=16 / attention-only
#    (kept in the comment below) to find out which half caused it.
#
#    Previous LoRA config, for easy rollback:
#      r = 16, lora_alpha = 16,
#      target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ✅ INCREASED: was 256 — paragraphs need more space
#    Single sentence: ~50-80 tokens
#    Paragraph (3 sentences): ~150-200 tokens
#    Prompt overhead: ~60 tokens
#    Total needed: ~260 tokens → use 512 safely
MAX_SEQ_LENGTH = 512

# ─────────────────────────────────────────────
# DATASET BALANCE CHECK
# ─────────────────────────────────────────────
print("🔹 Checking dataset balance...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f if line.strip()]

changed   = sum(1 for s in samples if s["input"].strip() != s["output"].strip())
unchanged = len(samples) - changed
print(f"   Total    : {len(samples)}")
print(f"   Changed  : {changed}  ({changed/len(samples)*100:.1f}%)")
print(f"   Unchanged: {unchanged} ({unchanged/len(samples)*100:.1f}%)")
if changed / len(samples) > 0.65:
    print("   ⚠️  WARNING: >65% changed — consider adding more no-change examples.")

# ─────────────────────────────────────────────
# LOAD TOKENIZER
# ✅ NEW: tokenizer already saved in SinLLaMA-merged-base
#    No separate TOKENIZER_PATH needed
# ─────────────────────────────────────────────
print("🔹 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    SINLLAMA_BASE,
    local_files_only=True,
)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

# ─────────────────────────────────────────────
# LOAD PRE-MERGED SINLLAMA BASE
# ✅ NEW: replaces the old 4-step chain:
#   1. FastLanguageModel.from_pretrained(llama-3-8b)   ← REMOVED
#   2. model.resize_token_embeddings(...)               ← REMOVED
#   3. PeftModel.from_pretrained(model, SinLlama_v01)  ← REMOVED
#   4. model.merge_and_unload()                         ← REMOVED
#
# The merged base already has the right embedding size baked in.
# ─────────────────────────────────────────────
print("🔹 Loading pre-merged SinLLaMA base...")
model, _ = FastLanguageModel.from_pretrained(
    model_name    = SINLLAMA_BASE,
    max_seq_length= MAX_SEQ_LENGTH,
    dtype         = None,
    load_in_4bit  = True,
)

# ─────────────────────────────────────────────
# ADD GRAMMAR LoRA
#
# IMPORTANT — why lora_dropout=0.05 and NOT 0.0:
#   Setting dropout=0.0 activates Unsloth's fast custom CUDA LoRA
#   kernel. After merge_and_unload() on a 4bit model, the bnb
#   quantized weights cause a dtype mismatch inside that kernel
#   (BFloat16 != float in fast_lora.py backward).
#   Using dropout=0.05 makes Unsloth fall back to standard PyTorch
#   autograd, which handles mixed dtypes correctly.
#
# NOTE: Now that we load SinLLaMA-merged-base directly (not merging
#   at runtime), this issue still applies — the base model weights
#   are still 4bit quantized when loaded via load_in_4bit=True.
#   Keep lora_dropout=0.05.
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# v18 CAPACITY BUMP — MLP layers + r 16->32
#
# WHY: several specific bugs survived 2+ rounds of targeted data
#   additions (register/word substitution ඇමතිවරයා->අමාත්‍යවරයා and
#   කිව්වා->පැවසුවේ; verb-stem selection පවත්වූහ vs පැවැත්වූහ; plural
#   literary endings on the same test sentences). These are lexical /
#   word-choice decisions, which in transformers live substantially in
#   the MLP (feed-forward) weights — and LoRA was only ever attached to
#   the attention projections, so no amount of data could reach them.
#   This was flagged as "Finding #2" back in the v14 investigation and
#   deferred until data alone showed diminishing returns. It has.
#
# COST: trainable params go from ~16M to ~84M (~5x). Expect training
#   time to rise from ~100 min to roughly 2.5h on the A40, and higher
#   VRAM use (should still fit in 44GB at batch 2 / seq 512 / 4bit).
#
# lora_alpha is kept equal to r (32/32), preserving the same
#   alpha/r = 1.0 scaling ratio the previous runs used — so this is a
#   capacity change, not an effective-learning-rate change.
# ─────────────────────────────────────────────
print("🔹 Adding grammar LoRA adapter...")
model = FastLanguageModel.get_peft_model(
    model,
    r              = 32,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",     # attention (as before)
        "gate_proj", "up_proj", "down_proj",        # v18: MLP — lexical/word-choice
    ],
    lora_alpha     = 32,
    lora_dropout   = 0.05,  # must stay 0.05 — see note above
    bias           = "none",
)
model.print_trainable_parameters()

# ─────────────────────────────────────────────
# PROMPT FORMAT
#
# v15 CHANGE: emit separate "prompt"/"completion" columns instead of
#   one concatenated "text" field.
#
# WHY: v14 (and everything before it) built a single "text" field
#   (Instruction + Input + Response, all concatenated) and trained
#   with train_on_inputs=True — but that isn't even a real SFTConfig/
#   SFTTrainer parameter in this TRL version (0.24.0; confirmed via
#   `from trl import DataCollatorForCompletionOnlyLM` also failing —
#   that class was removed entirely, see huggingface/trl discussion
#   #3826 and issue #5324). So it was silently ignored, and per TRL's
#   current docs, a plain "text"-field dataset always computes loss
#   over the FULL sequence — instruction, input, AND response tokens.
#   A large, fixed share of every gradient step was spent reproducing
#   the identical instruction text and the given input, which diluted
#   the signal available to learn the actual correction. This is the
#   leading suspect for why v14 (2,851 rows, +535 new/expanded rows
#   across 10 new grammar categories vs v13's 2,316) scored bit-for-
#   bit identical to v13 on the 57-example stage2 test set — see
#   train_roadmap.md, Finding #1.
#
# FIX: TRL's current (0.24.0) supported way to do completion-only loss
#   is the "prompt-completion" dataset format: separate "prompt" and
#   "completion" text columns, with SFTConfig(completion_only_loss=True)
#   (this is also the DEFAULT for prompt-completion datasets — passed
#   explicitly below only for clarity). No custom collator needed.
# ─────────────────────────────────────────────
INSTRUCTION_TEXT = (
    "Correct the grammar of the Sinhala sentence. "
    "ONLY fix errors. "
    "If the sentence is already correct, return it EXACTLY unchanged — "
    "do not rephrase, reorder, or change tense."
)

def format_prompt(example):
    prompt = (
        f"### Instruction:\n{INSTRUCTION_TEXT}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n"
    )
    # NOTE: do NOT manually append tokenizer.eos_token here — SFTConfig's
    # `eos_token` (defaults to processing_class.eos_token) handles turn/
    # sequence termination for prompt-completion datasets. Appending it
    # ourselves too risks a doubled EOS token.
    return {"prompt": prompt, "completion": example["output"]}

# ─────────────────────────────────────────────
# LOAD + FORMAT DATASET
# ─────────────────────────────────────────────
print("🔹 Loading dataset...")
dataset = load_dataset("json", data_files=DATA_PATH)
dataset = dataset.map(format_prompt, remove_columns=dataset["train"].column_names)
print(f"   Dataset size: {len(dataset['train'])} examples")

# ─────────────────────────────────────────────
# v15 ADDITION — small held-out validation split
#
# WHY: v14 trained 5 fixed epochs with zero eval signal in between —
#   the only feedback was the expensive full test_grammar.py run at
#   the end (train_roadmap.md, Finding #4). Holding out 5% purely for
#   periodic eval_loss monitoring doesn't change what the model learns
#   from the training rows themselves, it just adds visibility into
#   whether 5 epochs is under/over-fit. cleaned_v2.jsonl is already
#   shuffled (see gen_data.py), so a contiguous slice is a fine random
#   sample — but train_test_split reshuffles with a fixed seed anyway
#   so this is robust either way.
# ─────────────────────────────────────────────
split       = dataset["train"].train_test_split(test_size=0.05, seed=42)
train_data  = split["train"]
eval_data   = split["test"]
print(f"   Train split  : {len(train_data)} examples")
print(f"   Eval split   : {len(eval_data)} examples (held out, monitoring only)")

# ─────────────────────────────────────────────
# COMPUTE HYPERPARAMETERS
# Formula (consistent with all previous versions):
#   effective_batch = per_device(2) × grad_accum(4) = 8
#   steps_per_epoch = ceil(samples / 8)
#   max_steps       = steps_per_epoch × 5 epochs
#   warmup_steps    = round(max_steps × 0.1)
#   save_steps      = round(max_steps / 2)
# NOTE: n_samples is now the TRAIN split (95% of the file) now that a
#   held-out eval slice exists, so max_steps is computed on the same
#   basis actually used for gradient updates.
# ─────────────────────────────────────────────
import math
n_samples    = len(train_data)
steps_epoch  = math.ceil(n_samples / 8)
max_steps    = steps_epoch * 5
warmup_steps = round(max_steps * 0.1)
save_steps   = round(max_steps / 2)
# v18: eval once per epoch instead of twice per RUN. With ~5x the
#   trainable params, overfitting is a real risk and the old 2-evals-
#   total cadence was far too coarse to see it. This is monitoring only
#   — it does not change what the model learns.
#   WATCH FOR: if eval_loss stops falling (or rises) after epoch 3 while
#   train loss keeps dropping, that's overfitting — rerun with 3-4
#   epochs instead of 5.
eval_steps   = steps_epoch

print(f"\n📊 Hyperparameters:")
print(f"   samples      = {n_samples}")
print(f"   steps/epoch  = {steps_epoch}")
print(f"   max_steps    = {max_steps}")
print(f"   warmup_steps = {warmup_steps}")
print(f"   save_steps   = {save_steps}")
print(f"   eval_steps   = {eval_steps}")

# ─────────────────────────────────────────────
# TRAINER
#
# v15 CHANGE: args is now SFTConfig, not TrainingArguments.
#   TRL 0.24.0's SFTTrainer constructor doesn't accept
#   dataset_text_field / max_seq_length / packing / tokenizer= as
#   direct kwargs (confirmed against current trl docs) — those all
#   moved into SFTConfig (max_seq_length -> max_length), and the
#   tokenizer kwarg is now processing_class=. The v14 script passed
#   all of these the old way; they were most likely silently dropped
#   rather than erroring (Unsloth is known to shim old-style kwargs
#   for backward compatibility), which is further evidence
#   train_on_inputs=True never did anything either.
# ─────────────────────────────────────────────
trainer = SFTTrainer(
    model            = model,
    processing_class = tokenizer,
    train_dataset    = train_data,
    eval_dataset     = eval_data,

    args = SFTConfig(
        max_length                  = MAX_SEQ_LENGTH,
        packing                     = False,
        completion_only_loss        = True,  # default for prompt-completion data; explicit for clarity

        per_device_train_batch_size = 2,
        per_device_eval_batch_size  = 2,
        gradient_accumulation_steps = 4,
        max_steps                   = max_steps,
        learning_rate               = 5e-5,
        logging_steps               = 10,
        max_grad_norm               = 1.0,
        lr_scheduler_type           = "cosine",
        warmup_steps                = warmup_steps,
        output_dir                  = OUTPUT_DIR,
        save_steps                  = save_steps,
        save_total_limit            = 2,
        eval_strategy               = "steps",
        eval_steps                  = eval_steps,
        report_to                   = "none",
    ),
)

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
print("\n🚀 Training started...")
trainer.train()

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
print("💾 Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Training complete! Model saved to {OUTPUT_DIR}")
