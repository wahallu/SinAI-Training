import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, TrainingArguments
from peft import PeftModel
from datasets import load_dataset
from trl import SFTTrainer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_MODEL = "./models/llama-3-8b"
SINLLAMA_ADAPTER = "./models/SinLlama_v01"
TOKENIZER_PATH = "./models/Extended-Sinhala-LLaMA"
DATA_PATH = "data/grammar_dataset_stage4.jsonl"
OUTPUT_DIR = "./models/adapters/grammar_sinllama_v4"
MAX_SEQ_LENGTH = 256

print("🔹 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ─────────────────────────────────────────────
# LOAD BASE MODEL
# ─────────────────────────────────────────────
print("🔹 Loading base model...")
model, _ = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

# 🔥 CRITICAL: match Sinhala vocab
print("🔹 Resizing embeddings...")
model = model.to("cpu")
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
model = model.to("cuda")

# ─────────────────────────────────────────────
# LOAD + MERGE SINLLAMA
# ─────────────────────────────────────────────
print("🔹 Loading SinLlama adapter...")
model = PeftModel.from_pretrained(model, SINLLAMA_ADAPTER)

print("🔹 Merging SinLlama...")
model = model.merge_and_unload()

# ─────────────────────────────────────────────
# ADD GRAMMAR LoRA
# ─────────────────────────────────────────────
print("🔹 Adding grammar adapter...")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)

# ─────────────────────────────────────────────
# LOAD DATASET (INSTRUCTION FORMAT)
# Each sample MUST be:
# {
#   "instruction": "...",
#   "input": "...",
#   "output": "..."
# }
# ─────────────────────────────────────────────
print("🔹 Loading dataset...")
dataset = load_dataset("json", data_files=DATA_PATH)

# 🔥 CRITICAL: proper instruction formatting
def format_prompt(example):
    prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
"""

    return {
        "text": prompt + example["output"],
        "prompt": prompt,  # 🔥 needed for masking
    }

dataset = dataset.map(format_prompt)

# ─────────────────────────────────────────────
# TRAINER (FIXED)
# ─────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,

    # 🔥 MOST IMPORTANT FIX
    train_on_inputs=False,

    dataset_kwargs={
        "add_special_tokens": True,
        "append_concat_token": False,
    },

    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=800,
        learning_rate=2e-4,
        logging_steps=5,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        output_dir=OUTPUT_DIR,
        report_to="none",
    ),
)

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
print("🚀 Training started...")
trainer.train()

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
print("💾 Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Training complete!")