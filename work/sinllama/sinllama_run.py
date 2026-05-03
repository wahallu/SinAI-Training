#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# hf auth l


# In[1]:


from unsloth import FastLanguageModel
from transformers import TextStreamer, AutoTokenizer
from peft import PeftModel
import torch


# In[2]:


# ── Config ─────────────────────────────────────────────────────────────────────
base_model_name = "/home/jovyan/work/sinllama/models/llama-3-8b"
adapter_name    = "/home/jovyan/work/sinllama/models/SinLlama_v01"
tokenizer_name  = "/home/jovyan/work/sinllama/models/Extended-Sinhala-LLaMA"

max_seq_length  = 2048
dtype           = torch.bfloat16
load_in_4bit    = True   # 🔥 IMPORTANT (prevents OOM)


# In[3]:


# ── Load tokenizer ─────────────────────────────────────────────────────────────
print("🔹 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name,
    local_files_only=True,
)


# In[ ]:


# ── Load base model ────────────────────────────────────────────────────────────
print("🔹 Loading base model...")
model, _ = FastLanguageModel.from_pretrained(
    model_name       = base_model_name,
    max_seq_length   = max_seq_length,
    dtype            = dtype,
    load_in_4bit     = load_in_4bit,
    local_files_only = True,
)


# In[ ]:


# ── 🔥 SAFE EMBEDDING RESIZE (CPU to avoid OOM) ────────────────────────────────
print("🔹 Resizing embeddings safely (CPU)...")
model = model.to("cpu")
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
model = model.to("cuda")


# In[ ]:


# ── Load LoRA adapter ──────────────────────────────────────────────────────────
print("🔹 Loading SinLlama adapter...")
model = PeftModel.from_pretrained(
    model,
    adapter_name,
    local_files_only=True,
    ensure_weight_tying=True,
)


# In[ ]:


# ── Enable fast inference ──────────────────────────────────────────────────────
FastLanguageModel.for_inference(model)
model.eval()


# In[ ]:


# ── Prompt formatter (STRONG CONTROL) ──────────────────────────────────────────
def format_prompt(user_input: str):
    return f"""### Instruction:
You are a helpful Sinhala AI assistant.
Answer ONLY the question in 1–2 short sentences.
Do NOT generate stories or extra text.

Question: {user_input}

### Response:
"""

# ── Generation config ──────────────────────────────────────────────────────────
def get_generation_config():
    return {
        "max_new_tokens": 40,
        "temperature": 0.2,
        "top_p": 0.7,
        "top_k": 40,
        "repetition_penalty": 1.2,
        "do_sample": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }


# In[ ]:


# ── Inference loop ─────────────────────────────────────────────────────────────
def run():
    print("\n✅ SinLlama ready!")
    print("   Sinhala input gives best results.")
    print("   Type 'quit' to exit.\n")

    gen_config = get_generation_config()
    streamer   = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    while True:
        prompt = input("\n📝 Prompt: ").strip()

        if not prompt:
            continue
        if prompt.lower() == "quit":
            print("👋 Bye!")
            break

        formatted_prompt = format_prompt(prompt)

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

        print("\n🤖 Output:\n")

        with torch.no_grad():
            model.generate(
                **inputs,
                streamer=streamer,
                **gen_config,
            )

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()

