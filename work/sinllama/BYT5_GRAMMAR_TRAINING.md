# ByT5 Sinhala grammar: simple GPU instructions

These commands train `google/byt5-small` as a full encoder-decoder model. Stage 6
is never supplied to the trainer.

## 1. Open the GPU terminal

Use an NVIDIA A40 or another CUDA GPU with approximately 24 GB or more VRAM.

```bash
cd /home/jovyan/work/sinllama
python -m venv .venv-byt5
source .venv-byt5/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_byt5.txt
nvidia-smi
```

## 2. Put the training data in place

Copy `cleaned_v10_full.jsonl` to:

```text
/home/jovyan/work/sinllama/data/grammar_manual_dataset_stage13.jsonl
```

The JSONL rows must contain `input` and `output`. The script removes exact
duplicates and, because the old data has no document IDs, creates a deterministic
development split whose exact correction pairs do not overlap training. This
development split is only for checkpoint selection; Stage 6 remains the final
test.

With the current local v10 file and default split settings, the verified result is
30,367 unique rows: 28,519 train, 1,518 development, and 330 bridge rows excluded
to keep exact correction-pair overlap at zero. The trainer recalculates and prints
these values; do not continue if `shared edits` is not zero. It saves hashed split
membership and the complete run configuration inside the output model directory.

## 3. Smoke test first

```bash
python scripts/train_grammar_byt5.py \
  --data data/grammar_manual_dataset_stage13.jsonl \
  --output-dir models/grammar_byt5_small_smoke \
  --max-samples 500 \
  --epochs 1
```

The run must finish, save a model, and report zero shared train/development exact
correction pairs. Delete or ignore the smoke checkpoint; it is not an evaluation
candidate.

## 4. Train ByT5-small

```bash
python scripts/train_grammar_byt5.py \
  --data data/grammar_manual_dataset_stage13.jsonl \
  --model google/byt5-small \
  --output-dir models/grammar_byt5_small_v01 \
  --epochs 3 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4
```

Defaults already enable BF16 when supported, gradient checkpointing, a 1024-byte
source/target budget, greedy evaluation settings, and best-checkpoint loading by
development loss. On a memory error, rerun with `--batch-size 1
--gradient-accumulation-steps 16`; do not reduce sequence length before checking
whether that would truncate data.

Resume an interrupted run with:

```bash
python scripts/train_grammar_byt5.py \
  --data data/grammar_manual_dataset_stage13.jsonl \
  --output-dir models/grammar_byt5_small_v01 \
  --resume-from-checkpoint models/grammar_byt5_small_v01/checkpoint-XXXX
```

## 5. Evaluate Stage 5 before opening Stage 6

```bash
python scripts/predict_grammar_byt5.py \
  --model models/grammar_byt5_small_v01 \
  --input-data data/grammar_test_stage5.jsonl \
  --output Tested_results/byt5_small_v01_stage5_predictions.jsonl
```

Score it:

```bash
python scripts/score_grammar_predictions.py \
  --predictions Tested_results/byt5_small_v01_stage5_predictions.jsonl \
  --gold data/grammar_test_stage5.jsonl \
  --train-data data/grammar_manual_dataset_stage13.jsonl \
  --output-prefix Tested_results/byt5_small_v01_stage5
```

Inspect the Markdown report. Freeze the checkpoint and decoding configuration
before generating Stage 6 predictions.

## 6. Train ByT5-base later

After the small-model pipeline is verified, change only the model and output:

```bash
python scripts/train_grammar_byt5.py \
  --data data/grammar_manual_dataset_stage13.jsonl \
  --model google/byt5-base \
  --output-dir models/grammar_byt5_base_v01 \
  --epochs 3 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-4
```

Do not train ByT5-base until the small-model smoke test, prediction export, and
offline scorer all work end to end.
