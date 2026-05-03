from huggingface_hub import snapshot_download
import os
import json

SAVE_DIR = "./models"

# ── Download base model (required by SinLlama adapter) ─────────────────────────
print("Downloading base model: unsloth/llama-3-8b...")
snapshot_download(
    repo_id   = "unsloth/llama-3-8b",
    local_dir = os.path.join(SAVE_DIR, "llama-3-8b"),
)

# ── Download SinLlama adapter ───────────────────────────────────────────────────
print("Downloading SinLlama adapter...")
snapshot_download(
    repo_id   = "polyglots/SinLlama_v01",
    local_dir = os.path.join(SAVE_DIR, "SinLlama_v01"),
)

# ── Download extended tokenizer ────────────────────────────────────────────────
print("Downloading Extended Sinhala Tokenizer...")
snapshot_download(
    repo_id   = "polyglots/Extended-Sinhala-LLaMA",
    local_dir = os.path.join(SAVE_DIR, "Extended-Sinhala-LLaMA"),
)

# ── Patch adapter_config.json to point to local base model ─────────────────────
adapter_config_path = os.path.join(SAVE_DIR, "SinLlama_v01", "adapter_config.json")

with open(adapter_config_path, "r") as f:
    adapter_config = json.load(f)

# Overwrite the base model path to local
adapter_config["base_model_name_or_path"] = os.path.abspath(
    os.path.join(SAVE_DIR, "llama-3-8b")
)

with open(adapter_config_path, "w") as f:
    json.dump(adapter_config, f, indent=2)

print(f"\n✅ Patched adapter_config.json → base model now points to local path")
print(f"✅ All models saved to: {os.path.abspath(SAVE_DIR)}")