# ByT5-small grammar v01 — training record

Date recorded: 2026-08-13  
Status: **training completed; Stage 5 evaluated; Stage 6 pending**

This record was transcribed from the completed GPU terminal output supplied after
the run. It records training evidence only. Development loss is not correction
accuracy, unseen-transfer recall, or a Stage 5/Stage 6 result.

## Run identity

| Field | Value |
|---|---|
| Base model | `google/byt5-small` |
| Output model | `models/grammar_byt5_small_v01` |
| Best checkpoint | `models/grammar_byt5_small_v01/checkpoint-5349` |
| Training data | `data/grammar_manual_dataset_stage13.jsonl` |
| Training-data SHA-256 | `6e36a6efaf0f2ef5d1f3c1d664cb7d572b2a012788eb85939b4004f7aeeedb47` |
| Unique rows after exact deduplication | 30,367 |
| Training rows | 28,519 |
| Development rows | 1,518 |
| Pair-bridge rows excluded | 330 |
| Shared exact train/development edits | **0** |
| Epochs | 3 |
| Per-device batch size | 2 |
| Gradient accumulation | 8 |
| Effective batch size | 16 |
| Learning rate | `1e-4` |
| Optimizer steps | 5,349 |
| Training runtime | 6,176.3396 seconds (about 1 h 42 min 56 s) |
| Train samples/second | 13.852 |
| Train steps/second | 0.866 |
| Final reported training loss | 0.1677242023 |

## Environment observed during setup

| Package/runtime | Version |
|---|---|
| Python | 3.11 |
| PyTorch | `2.6.0+cu124` |
| Transformers | `4.57.6` |
| Datasets | `4.3.0` |
| Accelerate | `1.13.0` |
| CUDA available | Yes |

The exact GPU model and peak allocated GPU memory were not present in the supplied
terminal transcript. Read them from `models/grammar_byt5_small_v01/run_manifest.json`
and `final_metrics.json` on the GPU machine before publishing hardware claims.

## Development-loss curve

| Epoch | Development loss |
|---:|---:|
| 1 | 0.0381541178 |
| 2 | 0.0360701680 |
| 3 | **0.0358547531** |

Development loss continued to improve through epoch 3, and the trainer selected
`checkpoint-5349`. The improvement from epoch 2 to epoch 3 was small
(`0.0002154149`, approximately 0.60%), but Stage 5 must decide whether it produced
a meaningful correction improvement. Do not extend training based on Stage 5 or
Stage 6 failures.

The checkpoint reload emitted:

```text
There were missing keys in the checkpoint model loaded:
['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight']
```

Training, reload, final development evaluation, and saving all completed. These
are tied/shared ByT5 embedding keys, so this terminal warning is recorded but is
not treated as a failed run. The next prediction reload is the practical integrity
check.

## What this result establishes

- The complete ByT5-small full-fine-tuning pipeline runs on the GPU.
- The legacy data was exactly deduplicated before splitting.
- The development split has zero exact correction-edit overlap with retained
  training rows.
- All three epochs completed and a best checkpoint was saved.

## What this result does not establish

- It does not show Stage 5 accuracy.
- It does not show unseen-pair or unseen-lemma transfer.
- It does not show correction precision, recall, or F0.5.
- It does not show clean-input restraint or protected-entity safety.
- It does not establish superiority over SinLLaMA v25 or v27.
- It is not a Stage 6 result; Stage 6 is still a draft pending native review and
  sealing.

## Required next steps

### 1. Export all Stage 5 predictions on the GPU

```bash
cd /home/jovyan/work/sinllama

python scripts/predict_grammar_byt5.py \
  --model models/grammar_byt5_small_v01 \
  --input-data data/grammar_test_stage5.jsonl \
  --output Tested_results/byt5_small_v01_stage5_predictions.jsonl
```

Do not use `--limit` for the recorded evaluation. Confirm that both files exist:

```bash
ls -lh Tested_results/byt5_small_v01_stage5_predictions.jsonl*
```

### 2. Score Stage 5

If the training and test JSONL files are present on the GPU machine, score there:

```bash
python scripts/score_grammar_predictions.py \
  --predictions Tested_results/byt5_small_v01_stage5_predictions.jsonl \
  --gold data/grammar_test_stage5.jsonl \
  --train-data data/grammar_manual_dataset_stage13.jsonl \
  --output-prefix Tested_results/byt5_small_v01_stage5
```

Preserve these four artifacts:

```text
Tested_results/byt5_small_v01_stage5_predictions.jsonl
Tested_results/byt5_small_v01_stage5_predictions.jsonl.manifest.json
Tested_results/byt5_small_v01_stage5.json
Tested_results/byt5_small_v01_stage5.md
```

Compare the report with v27 Stage 5: 43.1% overall exact, 23.7% change-needed
exact, 100% clean preservation, and 0% over-correction. The primary comparison is
edit precision/recall/F0.5 plus unseen-pair recall—not development loss.

## Stage 5 evaluation result

The frozen `grammar_byt5_small_v01` model generated and scored all 51 Stage 5
cases with greedy decoding.

Prediction SHA-256:
`65616b1dfe6aec9f9ed36059c26ba7e099d64c4ef3c69a1ad36effb6c3be52ae`

Gold SHA-256:
`ae4cba405bc58ac832cbe7164a6f5dfbb9e098abf9bfa5fc3258a289ecfd32d9`

| Metric | ByT5-small v01 | Count reconstructed from report |
|---|---:|---:|
| Overall exact match | 29.41% | 15/51 |
| Change-needed exact | 5.26% | 2/38 |
| Clean preservation | **100.00%** | 13/13 |
| Over-correction | **0.00%** | 0/13 |
| Under-correction | 57.89% | 22/38 |
| Wrong correction | 36.84% | 14/38 |
| Edit precision | 27.78% | 5/18 predicted edits correct |
| Edit recall | 9.26% | 5/54 gold edits recovered |
| Edit F0.5 | 19.84% | 0.1984 |
| Detection precision | 33.33% | 6/18 predicted edit locations correct |
| Detection recall | 11.11% | 6/54 gold edit locations found |
| Detection F1 | 16.67% | — |
| Unseen-pair recall | 9.30% | 4/43 unseen edits recovered |
| Number mutation | 5.88% | 3/51 cases |

The detection/correction difference shows that one additional edit location was
identified but filled with the wrong replacement: six gold locations were
detected, while only five exact gold edits were produced.

### Comparison with v27 on the same Stage 5 gold

| Metric | ByT5-small v01 | SinLLaMA v27 | Difference |
|---|---:|---:|---:|
| Overall exact | 29.41% (15/51) | 43.1% (22/51) | ByT5 -13.7 pp / -7 cases |
| Change-needed exact | 5.26% (2/38) | 23.7% (9/38) | ByT5 -18.4 pp / -7 cases |
| Clean preservation | 100% (13/13) | 100% (13/13) | equal |
| Over-correction | 0% (0/13) | 0% (0/13) | equal |

This first ByT5-small recipe is therefore **not competitive with v27 on Stage
5**. It learned strong copying/restraint but very weak correction behavior. Its
9.30% unseen-pair recall is also far below the historical v25-v27 unseen-transfer
range of approximately 48-50%, although that historical percentage was computed
across the broader staged evaluation and is not a perfectly identical Stage 5-only
denominator. Do not present the two unseen percentages as a strict paired
comparison.

### Evaluation limitations

- Stage 5 JSONL has no `category`, `context_required`, `edits`,
  `seen_lemma_family`, `protected_spans`, or `source_document_id` fields.
- Consequently, `unlabelled` is the only category, contextual N and unseen-lemma N
  are zero because metadata is absent, and protected-span mutation cannot be
  evaluated. These are not zero-error claims.
- Although the report labels the bootstrap as document-clustered, missing document
  IDs caused each row to be treated as a separate cluster. Stage 5 derives from
  four articles, so these intervals are example-bootstrap intervals and should not
  be reported as document-clustered confidence intervals.
- Number mutation is measurable directly from text and found 3/51 affected cases,
  but the supplied aggregate report does not reveal whether each change was
  gold-supported. Case-level prediction inspection is required.
- The exact 14 wrong corrections, 22 unchanged misses, four recovered unseen
  edits, and three number mutations cannot be linguistically classified from the
  Markdown aggregate alone.

### Case-level evidence still required

Preserve and copy these artifacts into the repository before completing the
failure analysis:

```text
Tested_results/byt5_small_v01_stage5_predictions.jsonl
Tested_results/byt5_small_v01_stage5_predictions.jsonl.manifest.json
Tested_results/byt5_small_v01_stage5.json
```

The predictions JSONL is the most important because it contains all 51 model
outputs. The scoring JSON contains the failure records and exact machine-readable
metrics. The prediction manifest supplies frozen decoding, latency, throughput,
input hash, and peak GPU memory.

These artifacts were subsequently uploaded and verified. The full case-level
analysis, v27 edit-level comparison, safety replay, and v02 recommendation are in:

`../../../../manual dataset/Tested_results/byt5_small_v01_stage5_analysis_20260813.md`

### 3. Freeze ByT5-small before Stage 6

After Stage 5 sanity checking, record:

- selected checkpoint path and checkpoint file hashes;
- `run_manifest.json` and `final_metrics.json`;
- prediction manifest, prefix, NFC normalization, maximum lengths, greedy
  decoding, Transformers/PyTorch versions, GPU, and peak memory;
- the Stage 5 report without changing the checkpoint from its failures.

### 4. Finish and seal Stage 6

The current `stage6-local-v1` has 600 draft cases but is explicitly
`draft-pending-human-review`. Complete two independent native-Sinhala reviews,
third-reviewer adjudication, lemma-family labels, near-duplicate family review,
and the final seal audit. Only then create the canonical input and private-gold
files.

### 5. Run the frozen model on input-only Stage 6

Once Stage 6 is sealed, copy only `grammar_stage6_inputs.jsonl` to the GPU and run:

```bash
python scripts/predict_grammar_byt5.py \
  --model models/grammar_byt5_small_v01 \
  --input-data data/grammar_stage6_inputs.jsonl \
  --output Tested_results/byt5_small_v01_stage6_predictions.jsonl
```

Score against private gold only in the offline evaluation environment, after all
models have submitted frozen predictions.

### 6. Continue the controlled comparison

Do not scale the v01 full-sentence recipe to ByT5-base. First validate the v02
edit-script formulation on its frozen development split. If that succeeds, run
ByT5-base, mT5-base, and mBART-50 with the validated formulation and Stage 6
protocol. Submit SinLLaMA v25 and v27 predictions to the same scorer. Do not use
Stage 6 to tune any system.

## Post-training evaluation update — 2026-08-13

Stages 2–5 are now complete. ByT5-small v01 is frozen and is not approved for
scaling to ByT5-base. Its correction-needed exact results were 9/42 on Stage 2,
0/10 on Stage 3, 0/26 on Stage 4, and 2/38 on Stage 5, all substantially below
v27. Stage 2 also exposed deletion of two Latin proper names on clean inputs;
Stage 5 exposed long-input deletion and repetition failures.

The consolidated comparison, artifact hashes, dataset-safety audit, and v02
decision are recorded at:

`../../../manual dataset/Tested_results/byt5_small_v01_stage2-stage5_analysis_20260813.md`

The next GPU experiment is `scripts/train_grammar_byt5_edits.py`, not a rerun of
the full-sentence trainer documented earlier in this record.

The first v02 smoke attempt completed its 24 optimization steps but failed during
generated-development decoding because Transformers padded prediction arrays with
`-100`, which ByT5's byte tokenizer passed to `chr()`. This was a script
compatibility failure, not a model result. The evaluator now converts invalid
padding IDs to the tokenizer pad ID before decoding, with two regression tests.

### ByT5-small v02 smoke result — 2026-08-13

The corrected v02 smoke pipeline completed end to end:

| Item | Result |
|---|---:|
| Original deterministic sample | 500 rows |
| Safety-compatible | 397 |
| Safety-excluded | 103 |
| Train / development / bridge-dropped | 373 / 20 / 4 |
| Shared train/development edits | 0 |
| Training steps / epochs | 24 / 1 |
| Runtime | 42.9363 seconds |
| Training loss | 4.32139 |
| Development loss | 3.22360 |
| Development exact | 35.00% |
| Development clean preservation | 100.00% |
| Development change exact | 0.00% |
| Development edit F0.5 | 0.00% |
| Invalid generated scripts | 100.00% |
| Applied generated scripts | 0.00% |
| Checkpoint | `models/grammar_byt5_small_v02_smoke/checkpoint-24` |

Interpretation: the implementation is operational, but this one-epoch 373-row
smoke model did not learn the output grammar. Every generated script failed
parsing and was safely converted to `KEEP`; therefore the 35% exact score is only
the clean fraction of the 20-row development sample, not learned correction.
Before a full run, inspect `development_predictions.jsonl`. If outputs show a
partially learned format, run a medium pilot; if they show systematic offset/JSON
failure, simplify the target representation before spending on full training.

The raw inspection showed 20/20 `invalid_json:Expecting value`. All generations
were Sinhala continuation text rather than JSON; many repeated phrases or clauses,
and none began to learn the required structure. v02a is therefore stopped after
the smoke test.

The replacement-format v02b removes numeric offsets and emits `KEEP` or lines of
`REPLACE ||| old ||| new`. Full-corpus preflight retains 22,344 rows: 20,986 train,
1,117 development, and 241 bridge-dropped, with zero shared edits. It additionally
excludes 802 rows with a repeated/ambiguous source span and 721 standalone
insertion/deletion rows. Run a new v02b smoke test before any full training.

### ByT5-small v02b replacement smoke result — 2026-08-13

The three-epoch replacement-format smoke run completed but failed its formulation
gate:

| Item | Result |
|---|---:|
| Original deterministic sample | 500 rows |
| Safety-compatible | 397 |
| Replacement-compatible | 369 |
| Train / development / bridge-dropped | 349 / 18 / 2 |
| Shared train/development edits | 0 |
| Epoch 1 / 2 / 3 development loss | 1.28156 / 1.17541 / 1.15435 |
| Development exact | 33.33% |
| Development clean preservation | 100.00% |
| Development change exact | 0.00% |
| Development edit F0.5 | 0.00% |
| Invalid generated scripts | 100.00% |
| Applied generated scripts | 0.00% |
| Runtime | 205.0487 seconds |
| Best checkpoint under edit-F0.5 selection | `checkpoint-22` |

All checkpoints tied at zero generated edit F0.5, so selecting the first checkpoint
does not indicate that epoch 1 was substantively best. Do not start the full v02b
run. Inspect expected versus raw development scripts to distinguish a strict-parser
mismatch from continued free-text generation.

Raw inspection ruled out a parser mismatch. None of 18 outputs contained `KEEP`
or `REPLACE |||`; 16 failed `replacement_line_schema`, two were empty/invalid
operation sequences, and most outputs repeated Sinhala phrases until near the
generation limit. The model was still performing free-text continuation.

Because 349 training rows provide only 66 optimizer updates, one final bounded
learnability pilot is permitted before abandoning generative scripts: a
deterministic 5,000-row sample for one epoch. Preflight yields 3,405 train, 182
development, and 44 bridge-dropped rows after safety/representation filtering,
with zero shared edits. Continue that pilot only if invalid scripts fall below
50%, at least one edit is safely applied, and generated edit F0.5 becomes nonzero.
Otherwise stop v02 and move to an explicit detector plus span-corrector design.
