import json
from collections import defaultdict

TRAIN_PATH = "/home/jovyan/work/sinllama/data/headline_dataset_48k_balanced_train.jsonl"
VAL_PATH   = "/home/jovyan/work/sinllama/data/headline_dataset_48k_balanced_val.jsonl"

weak_cats = ["Editorial", "Science", "Health", "Human Rights", "Technology"]

train_by_cat = defaultdict(set)
val_by_cat   = defaultdict(set)

with open(TRAIN_PATH) as f:
    for line in f:
        item = json.loads(line.strip())
        for cat in weak_cats:
            if f"Category: {cat}" in item["input"]:
                train_by_cat[cat].add(item["output"].strip())

with open(VAL_PATH) as f:
    for line in f:
        item = json.loads(line.strip())
        for cat in weak_cats:
            if f"Category: {cat}" in item["input"]:
                val_by_cat[cat].add(item["output"].strip())

print(f"{'Category':15s}  {'Train':>7s}  {'Val':>6s}  {'Overlap':>8s}  {'Leak %':>7s}")
print(f"{'-'*15}  {'-'*7}  {'-'*6}  {'-'*8}  {'-'*7}")
for cat in weak_cats:
    tr = train_by_cat[cat]
    vl = val_by_cat[cat]
    overlap = tr & vl
    pct = len(overlap) / len(vl) * 100 if vl else 0
    print(f"{cat:15s}  {len(tr):7d}  {len(vl):6d}  {len(overlap):8d}  {pct:6.1f}%")