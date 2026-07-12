import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, TrainingArguments
from peft import PeftModel
from datasets import load_dataset
from trl import SFTTrainer
import json

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# ✅ NEW: point directly to pre-merged SinLLaMA base
#    Same as test script — no more 4-step chain
SINLLAMA_BASE = "./models/SinLLaMA-merged-base"
DATA_PATH     = "data/grammar_manual_dataset_stage5.jsonl"
OUTPUT_DIR    = "./models/adapters/grammar_sinllama_v13"

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
print("🔹 Adding grammar LoRA adapter...")
model = FastLanguageModel.get_peft_model(
    model,
    r              = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha     = 16,
    lora_dropout   = 0.05,  # must stay 0.05 — see note above
    bias           = "none",
)
model.print_trainable_parameters()

# ─────────────────────────────────────────────
# PROMPT FORMAT
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
    full_text = prompt + example["output"] + tokenizer.eos_token
    return {"text": full_text}

# ─────────────────────────────────────────────
# LOAD + FORMAT DATASET
# ─────────────────────────────────────────────
print("🔹 Loading dataset...")
dataset = load_dataset("json", data_files=DATA_PATH)
dataset = dataset.map(format_prompt)
print(f"   Dataset size: {len(dataset['train'])} examples")

# ─────────────────────────────────────────────
# COMPUTE HYPERPARAMETERS
# Formula (consistent with all previous versions):
#   effective_batch = per_device(2) × grad_accum(4) = 8
#   steps_per_epoch = ceil(samples / 8)
#   max_steps       = steps_per_epoch × 5 epochs
#   warmup_steps    = round(max_steps × 0.1)
#   save_steps      = round(max_steps / 2)
# ─────────────────────────────────────────────
import math
n_samples    = len(dataset["train"])
steps_epoch  = math.ceil(n_samples / 8)
max_steps    = steps_epoch * 5
warmup_steps = round(max_steps * 0.1)
save_steps   = round(max_steps / 2)

print(f"\n📊 Hyperparameters:")
print(f"   samples      = {n_samples}")
print(f"   steps/epoch  = {steps_epoch}")
print(f"   max_steps    = {max_steps}")
print(f"   warmup_steps = {warmup_steps}")
print(f"   save_steps   = {save_steps}")

# ─────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────
trainer = SFTTrainer(
    model             = model,
    tokenizer         = tokenizer,
    train_dataset     = dataset["train"],
    dataset_text_field= "text",
    max_seq_length    = MAX_SEQ_LENGTH,
    packing           = False,
    train_on_inputs   = True,

    dataset_kwargs = {
        "add_special_tokens" : True,
        "append_concat_token": False,
    },

    args = TrainingArguments(
        per_device_train_batch_size = 2,
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