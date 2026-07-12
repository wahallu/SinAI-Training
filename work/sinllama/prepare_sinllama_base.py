# prepare_sinllama_base.py
# Run ONCE on the GPU server — saves the merged model for everyone to use

import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from peft import PeftModel

BASE_MODEL       = "/home/jovyan/work/sinllama/models/llama-3-8b"
SINLLAMA_ADAPTER = "/home/jovyan/work/sinllama/models/SinLlama_v01"
TOKENIZER_PATH   = "/home/jovyan/work/sinllama/models/Extended-Sinhala-LLaMA"
MERGED_OUTPUT    = "/home/jovyan/work/sinllama/models/SinLLaMA-merged-base"   # ← shared base for all tasks

print("🔹 Loading base model...")
model, _ = FastLanguageModel.from_pretrained(
    model_name    = BASE_MODEL,
    max_seq_length= 1024,
    dtype         = None,
    load_in_4bit  = True,
    local_files_only = True,
    attn_implementation = "eager",
)

print("🔹 Loading tokenizer & resizing embeddings...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"
model = model.to("cpu")
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
model = model.to("cuda")

print("🔹 Loading + merging SinLLaMA adapter...")
model = PeftModel.from_pretrained(model, SINLLAMA_ADAPTER)
model = model.merge_and_unload()

print("💾 Saving merged base model...")
model.save_pretrained(MERGED_OUTPUT)
tokenizer.save_pretrained(MERGED_OUTPUT)
print(f"✅ Done! All scripts can now use: {MERGED_OUTPUT}")