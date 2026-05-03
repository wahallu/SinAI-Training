import random
import argparse

# ─────────────────────────────────────────────
# ARGUMENTS
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Shuffle JSONL dataset")
parser.add_argument("--input", required=True, help="Input JSONL file")
parser.add_argument("--output", required=True, help="Output shuffled JSONL file")
parser.add_argument("--seed", type=int, default=42, help="Random seed (for reproducibility)")
args = parser.parse_args()

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print(f"🔹 Loading dataset: {args.input}")

with open(args.input, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"📊 Total samples: {len(lines)}")

# ─────────────────────────────────────────────
# SHUFFLE
# ─────────────────────────────────────────────
random.seed(args.seed)
random.shuffle(lines)

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
with open(args.output, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"✅ Shuffled dataset saved to: {args.output}")

# ─────────────────────────────────────────────
# QUICK PREVIEW
# ─────────────────────────────────────────────
print("\n🔍 Preview (first 3 samples):\n")
for line in lines[:3]:
    print(line.strip())