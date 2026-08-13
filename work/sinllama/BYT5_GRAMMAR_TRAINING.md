# ByT5 Sinhala grammar: simple GPU instructions

These commands train `google/byt5-small` as a full encoder-decoder model. Stage 6
is never supplied to the trainer.

Recorded run status: the full `grammar_byt5_small_v01` three-epoch training run
completed on 2026-08-13. Its evidence and post-training evaluation commands are in
`Tested_results/grammar_byt5_small_v01_training_20260813.md`. Stage 5 was later
scored and showed that v01 is not competitive with v27; see
`../../../manual dataset/Tested_results/byt5_small_v01_stage5_analysis_20260813.md`.
The consolidated Stage 2–5 audit is in
`../../../manual dataset/Tested_results/byt5_small_v01_stage2-stage5_analysis_20260813.md`.
Do not start ByT5-base from the full-sentence recipe. Stage 6 remains unsealed.

The active next experiment is **ByT5-small v02**, which generates validated edit
scripts instead of copying the complete corrected sentence.

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

## 3. Test the v02 safety code

```bash
python scripts/test_grammar_edit_script.py
```

All tests must pass before training.

If an older copy fails after the training steps with
`ValueError: chr() arg not in range(0x110000)`, update
`grammar_edit_script.py`, `train_grammar_byt5_edits.py`, and
`test_grammar_edit_script.py`. Transformers may pad generated evaluation arrays
with `-100`; ByT5 cannot decode that value directly. The current scripts replace
evaluation-only padding with the tokenizer pad ID and include regression tests.

## 4. Smoke-test ByT5-small v02

```bash
python scripts/train_grammar_byt5_edits.py \
  --data data/grammar_manual_dataset_stage13.jsonl \
  --model google/byt5-small \
  --output-dir models/grammar_byt5_small_v02_smoke \
  --max-samples 500 \
  --epochs 1
```

The trainer will print the safety exclusions and the pair-disjoint split. It must
report `shared edits: 0` and finish with a generated development edit F0.5 value.
Do not evaluate the smoke model on Stages 2–6.

## 5. Train ByT5-small v02

```bash
python scripts/train_grammar_byt5_edits.py \
  --data data/grammar_manual_dataset_stage13.jsonl \
  --model google/byt5-small \
  --output-dir models/grammar_byt5_small_v02 \
  --epochs 3 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4
```

Default v02 preparation produces 23,867 safety-compatible unique rows: 22,434
train, 1,193 development, and 240 pair-bridge rows excluded. Another 6,500 of the
30,367 deduplicated v01 rows are excluded because their gold output violates the
runtime safety contract, chiefly by changing invisible Unicode format controls.
The precise counts and reasons are written to `run_manifest.json`.

Unlike v01, v02 selects checkpoints using generated development edit F0.5 after
strict script validation. Invalid or unsafe scripts become `KEEP`. The output
directory also contains `development_gold.jsonl` for an auditable development
replay. Do not choose a checkpoint using Stages 2–5.

## 6. Freeze and evaluate v02

Only after accepting the generated development metrics, run each frozen
regression set. Example for Stage 5:

```bash
python scripts/predict_grammar_byt5_edits.py \
  --model models/grammar_byt5_small_v02 \
  --input-data data/grammar_test_stage5.jsonl \
  --output Tested_results/byt5_small_v02_stage5_predictions.jsonl

python scripts/score_grammar_predictions.py \
  --predictions Tested_results/byt5_small_v02_stage5_predictions.jsonl \
  --gold data/grammar_test_stage5.jsonl \
  --train-data data/grammar_manual_dataset_stage13.jsonl \
  --output-prefix Tested_results/byt5_small_v02_stage5
```

The prediction rows retain the raw edit script, validation status, rejection
reasons, and safely applied final `prediction`. Inspect the manifest's safety
counts together with the scorer report.

## Archived v01 commands

The following commands document the completed v01 experiment. Do not rerun or
overwrite its artifacts.

### v01 smoke test

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

### v01 full training

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

### v01 Stage 5 evaluation

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

## 7. ByT5-base decision gate

Do not run the following command yet. Train ByT5-base only if v02 has no
catastrophic generation or protected factual mutation, preserves at least 95% of
clean development inputs, and materially improves generated development edit
F0.5. Then use the **edit-script trainer**, not the v01 full-sentence trainer:

After the small-model pipeline is verified, change only the model and output:

```bash
python scripts/train_grammar_byt5_edits.py \
  --data data/grammar_manual_dataset_stage13.jsonl \
  --model google/byt5-base \
  --output-dir models/grammar_byt5_base_v02 \
  --epochs 3 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-4
```

Do not train ByT5-base until the small-model v02 training, safe prediction export,
and offline scorer work end to end.
