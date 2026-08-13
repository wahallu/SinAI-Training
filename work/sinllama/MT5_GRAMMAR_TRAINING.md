# mT5 Sinhala grammar: simple GPU instructions

This experiment fine-tunes `google/mt5-small` to generate the complete corrected
Sinhala text. It is the comparable SentencePiece baseline for the completed
`google/byt5-small` v01 run. It does **not** use the failed ByT5 edit-script
format.

The training input is Stage 13 only. Never pass Stages 2, 3, 4, 5, or 6 to the
trainer. They are held-out evaluation data.

## 1. Open the GPU terminal

```bash
cd /home/jovyan/work/sinllama
python -m pip install -r requirements_mt5.txt
```

The existing working ByT5 environment can be reused. mT5 additionally needs
`sentencepiece`. Do not install Unsloth, `xformers`, or `torchao` for this run.

Verify the environment:

```bash
python - <<'PY'
import torch
import transformers
import sentencepiece

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("SentencePiece:", sentencepiece.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("mT5 imports: OK")
PY
```

If `PreTrainedModel` fails because an old `torchao` package is being imported,
remove that optional package and rerun the check:

```bash
python -m pip uninstall -y torchao
```

## 2. Verify the data and scripts

The training file must be:

```text
data/grammar_manual_dataset_stage13.jsonl
```

Its expected SHA-256 is:

```text
6e36a6efaf0f2ef5d1f3c1d664cb7d572b2a012788eb85939b4004f7aeeedb47
```

Run the dependency-free safety tests:

```bash
python scripts/test_grammar_edit_script.py
```

The current suite runs 15 tests and must end with `OK`.

## 3. Run an integration smoke test

This checks downloads, tokenization, generation, safety validation, checkpoint
selection, and artifact writing. It is not a model-quality result.

```bash
python scripts/train_grammar_mt5.py \
  --data data/grammar_manual_dataset_stage13.jsonl \
  --model google/mt5-small \
  --output-dir models/grammar_mt5_small_smoke \
  --max-samples 500 \
  --epochs 1 \
  --batch-size 2 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4
```

The preparation should report 470 training, 25 development, five
bridge-dropped rows, and `shared edits: 0`. Let the model download finish on the
first run. Do not test this smoke checkpoint on any held-out stage.

Inspect the result:

```bash
python - <<'PY'
import collections
import json
from pathlib import Path

folder = Path("models/grammar_mt5_small_smoke")
print(json.loads((folder / "final_metrics.json").read_text(encoding="utf-8")))
rows = [json.loads(line) for line in
        (folder / "development_predictions.jsonl").open(encoding="utf-8")]
print("Statuses:", collections.Counter(row["candidate_status"] for row in rows))
print("Reasons:", collections.Counter(
    reason for row in rows for reason in row.get("safety_reasons", [])
))
for row in rows[:10]:
    print(row["id"], row["candidate_status"], repr(row["raw_prediction"]))
PY
```

The smoke test passes operationally if training finishes, the artifact files
exist, and `shared edits` is zero. A zero edit score after only 500 examples is
not by itself a reason to abandon mT5. Stop only for a pipeline error, truncation
warning, non-zero shared edits, or consistently empty/corrupted artifacts.

## 4. Train the full mT5-small v01 baseline

Use a new output directory; do not overwrite the smoke run.

```bash
python scripts/train_grammar_mt5.py \
  --data data/grammar_manual_dataset_stage13.jsonl \
  --model google/mt5-small \
  --output-dir models/grammar_mt5_small_v01 \
  --epochs 3 \
  --batch-size 2 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4
```

The verified default split is 28,519 training, 1,518 development, and 330
bridge-dropped rows, with zero shared exact correction pairs. The effective
training batch size is 16. The script uses BF16 when the GPU supports it,
gradient checkpointing, greedy generation, and selects the best checkpoint by
safely applied generated-development edit F0.5—not by loss alone.

If CUDA runs out of memory, restart with:

```text
--batch-size 1 --eval-batch-size 2 --gradient-accumulation-steps 16
```

Do not shorten the token limits: the trainer deliberately stops instead of
silently truncating examples.

Resume an interrupted run by adding:

```text
--resume-from-checkpoint models/grammar_mt5_small_v01/checkpoint-XXXX
```

The output directory records the run configuration, package/GPU versions,
dataset and split hashes, split membership, development gold, raw generations,
safety decisions, and final metrics.

## 5. Accept and freeze the checkpoint

Review these two files first:

```text
models/grammar_mt5_small_v01/final_metrics.json
models/grammar_mt5_small_v01/development_predictions.jsonl
```

Record at least:

- `eval_edit_precision`, `eval_edit_recall`, and `eval_edit_f0_5`
- the corresponding `eval_raw_*` metrics before safety fallback
- `eval_change_exact` and `eval_clean_preservation`
- candidate applied, kept, rejected, and invalid rates
- the best checkpoint, runtime, dataset hash, and split hash

Reject the run if outputs are broadly empty, truncated, or repetitive, or if
safety fallback alone explains the exact score. If the generated development
outputs are usable, freeze the model directory and decoding settings before
opening a held-out test stage.

## 6. Evaluate frozen Stage 5

After checkpoint acceptance, generate deterministic, safety-validated
predictions:

```bash
python scripts/predict_grammar_mt5.py \
  --model models/grammar_mt5_small_v01 \
  --input-data data/grammar_test_stage5.jsonl \
  --output Tested_results/mt5_small_v01_stage5_predictions.jsonl \
  --batch-size 4
```

Then use the same scorer used for ByT5 and v27 comparisons:

```bash
python scripts/score_grammar_predictions.py \
  --predictions Tested_results/mt5_small_v01_stage5_predictions.jsonl \
  --gold data/grammar_test_stage5.jsonl \
  --train-data data/grammar_manual_dataset_stage13.jsonl \
  --output-prefix Tested_results/mt5_small_v01_stage5
```

Keep the prediction JSONL, its manifest, scorer JSON, and Markdown report. The
manifest contains raw-candidate safety counts; the prediction file retains both
the raw model generation and the final safe prediction.

Stage 6 must remain sealed until its manual review is complete and all selected
models have frozen checkpoints and decoding configurations.
