import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from peft import PeftModel

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_MODEL = "./models/llama-3-8b"
SINLLAMA_ADAPTER = "./models/SinLlama_v01"
GRAMMAR_ADAPTER = "./models/adapters/grammar_sinllama_v4"
TOKENIZER_PATH = "./models/Extended-Sinhala-LLaMA"

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
    max_seq_length=256,
    dtype=None,
    load_in_4bit=True,
)

# 🔥 REQUIRED for SinLlama
print("🔹 Resizing embeddings...")
model = model.to("cpu")
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
model = model.to("cuda")

# ─────────────────────────────────────────────
# LOAD SINLLAMA
# ─────────────────────────────────────────────
print("🔹 Loading SinLlama...")
model = PeftModel.from_pretrained(model, SINLLAMA_ADAPTER)

# merge Sinhala knowledge
model = model.merge_and_unload()

# ─────────────────────────────────────────────
# LOAD GRAMMAR ADAPTER
# ─────────────────────────────────────────────
print("🔹 Loading grammar adapter...")
model.load_adapter(GRAMMAR_ADAPTER)

FastLanguageModel.for_inference(model)
model.eval()

# ─────────────────────────────────────────────
# FUNCTION
# ─────────────────────────────────────────────
def correct_sentence(sentence):
    prompt = f"""### Instruction:
Correct the grammar of the Sinhala sentence

### Input:
{sentence}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.1,
            top_p=0.9,
            do_sample=False,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Response:" in result:
        result = result.split("### Response:")[-1]

    return result.strip()

# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────
test_sentences = [
    "මම ගියාලා ගෙදරට",
    "ඔහුන් පාසලට ගියා",
    "මම කෑම කනවා දැන්",
    "මම ගියාලා පාසලට"
]

for s in test_sentences:
    print("\n====================")
    print("INPUT :", s)
    print("OUTPUT:", correct_sentence(s))