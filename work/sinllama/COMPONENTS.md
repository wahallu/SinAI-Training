# SinLlama Components: Datasets & Version History

Detailed reference for the four SinLlama tasks — grammar correction, headline
generation, summarization, and style rewriting. For each: the training
dataset's actual schema, the test/eval dataset's schema and metrics, and how
the adapter changed version over version and why.

This file complements the root `CLAUDE.md`, which already documents headline
(v17-v20) and summarizer (v06+) version history in narrative form — those
sections aren't duplicated here in full, just the dataset-structure detail
CLAUDE.md doesn't cover, plus a condensed table. Grammar and style have no
existing version-history documentation anywhere else, so those two sections
are the primary new content here.

Everything below was verified directly against the files and scripts in this
repo as of 2026-08-14 (row counts via `wc -l`, schemas via reading actual
records, hyperparameters via reading the training scripts, metrics via
reading the scoring code) — not inferred from filenames or comments alone.
Where a script's own comment turned out to be stale or wrong, that's called
out explicitly, since it matters for anyone trying to reproduce a result.

---

## 1. Grammar Correction

### 1.1 Training dataset

Two unrelated file families live under `work/sinllama/data/`, per the
`*_stageN.jsonl` naming convention in CLAUDE.md (highest stage = current).

**Legacy / auto-generated — not used by any current script.**
`grammar_dataset_stage1-4.jsonl`, `grammar_dataset_shuffled.jsonl`,
`grammar_test.jsonl`. Small, synthetic SOV-reorder / tense-marker
perturbations (61 → 468 rows across stage1-4). Schema
`{"instruction", "input", "output"}` (stage3 drops `instruction`). None of
these paths appear in any training script's `DATA_PATH` — kept for history
only.

**Manual (hand-curated) — current.** `grammar_manual_dataset_stage1-13.jsonl`,
schema `{"instruction", "input", "output"}`:

```json
{"instruction": "Correct the grammar of the Sinhala sentence.ONLY fix errors...",
 "input": "හමාස් සටන්කාමීන් මීට පෙර ප්‍රකාශ කලේ...",
 "output": "හමාස් සටන්කාමීන් මීට පෙර ප්‍රකාශ කළේ..."}
```

The per-row `instruction` field is **not actually read** by
`train_grammar.py` / `train_grammar_v27.py` — both build one fixed
`INSTRUCTION_TEXT` constant applied to every row regardless of what's stored
in the JSONL. It's presentational/historical, not functional.

| Stage | Rows | Notes |
|---|---:|---|
| 1 | 901 | |
| 5 | 2,316 | trained v19 |
| 6 | 2,851 | trained v14 |
| 9 | 17,532 | trained v20/v21 |
| 11 | 27,064 | trained v22 |
| 12 | 36,006 | trained v23/v24 |
| **13 (current)** | **36,006** | trained **v25/v26/v27** |

Stage 12→13 is not a new-rows bump: stage13 is stage12's exact same 36,006
rows/order with only the `output` field corrected (221 human-ruled spellings
+ 6,002 word fixes across 4,999 rows) — inputs untouched. Changed-row share
rose 69.5%→73.5% as a result.

Every training run prints a dataset **fingerprint** (SHA-256 of the exact
input/output pairs) and four "canary" counts (e.g. how many rows teach
තරග→තරඟ), so a stale upload or wrong file version fails loudly instead of
silently reproducing an old run (`train_grammar.py:96-174`).

### 1.2 Test / eval dataset

`grammar_test_stage2.jsonl` … `stage5.jsonl` — same
`{"instruction","input","output"}` shape (the `instruction` field is again
ignored; `test_grammar.py` builds its own fixed prompt):

| Stage | Rows | What differentiates it |
|---|---:|---|
| 2 | 57 | Single-sentence news examples. |
| 3 | 10 | 2-3 sentence paragraphs. Zero already-correct rows — no-change accuracy is undefined here. |
| 4 | 36 | Paragraphs from 4 real published articles, incl. 10 already-correct controls. Never used to drive training-data decisions → clean generalization check. |
| 5 | 51 | Cases from 4 *different* real articles (built 2026-07-28, verified absent from the training corpus). 7 are errors real journalists actually published; the rest are wrong→right pairs the training data never teaches — measures **rule transfer, not memorization**. |

`grammar_stage6_inputs.jsonl` (286 rows) is a private-gold offline eval set:
schema is deliberately just `{"id", "input"}` — no output. Predictions are
generated on the GPU box and scored offline against gold that's never
uploaded here (`score_grammar_predictions.py`'s docstring: "this program
performs no model inference... keep private Stage 6 gold on the offline
scoring machine").

### 1.3 Metrics

Two different scorers exist, used for different purposes:

**`test_grammar.py`** (stage2-5, the harness every adapter v13→v27 was
compared with): exact-match overall / change-needed / no-change accuracy,
over-correction rate, plus ROUGE-1/2/L, sentence GLEU, char-F1, token P/R/F1
— all computed on **Sinhala grapheme clusters** (base char + combining
diacritics grouped as one token), because the standard `rouge_score` library
breaks on Sinhala Unicode (see CLAUDE.md's Critical Constraints).

**`score_grammar_predictions.py`** (private stage6 + ByT5/mT5 baselines): a
stricter, edit-based scorer.
- **change** vs **clean**: gold row needs a correction vs. is already correct.
- **Edit precision/recall/F0.5**: predicted vs. gold edit-tuples (from
  `difflib.SequenceMatcher` opcodes on whitespace tokens); F0.5 weights
  precision higher, since a false correction is costlier than a missed one.
- **Detection precision/recall/F1**: did the model notice *something* needed
  fixing at that span, regardless of whether the replacement was right.
- **Unseen-pair recall**: of gold edits whose exact (old, new) pair never
  appears in the training data's edit-pair set, what fraction did the model
  still get right — the generalization number.
- **Unseen-lemma recall**: stricter version at the lemma-family level
  (unseen conjugations of an otherwise-seen verb).
- **Contextual exact**: exact-match restricted to rows flagged
  `context_required` / `contextual-real-word`.
- **Restraint/exactness**: `clean_preservation` (left already-correct rows
  alone), `overcorrection_rate`, `undercorrection_rate` (changed row where
  prediction == source, i.e. did nothing), `wrong_correction_rate`
  (prediction is neither source nor target), `protected_mutation_rate`
  (altered a protected span, e.g. a quoted name), `number_mutation_rate`
  (changed a number — should never happen).

All metrics ship with 95% document-clustered bootstrap confidence intervals.

### 1.4 Version history

Shared recipe since v18: pre-merged `SinLLaMA-merged-base` (4-bit), LoRA on
`q/k/v/o_proj` **+** `gate/up/down_proj` (MLP block added at v18 specifically
to reach lexical/word-choice weights attention-only LoRA couldn't touch),
`lora_dropout=0.05`, separate prompt/completion columns with
`completion_only_loss=True` (fixed at v15 — before that, loss was computed
over the full instruction+input+response sequence, diluting the correction
signal and likely why v14 scored identically to v13 despite +535 rows),
`lr=5e-5` cosine, effective batch 8, `MAX_SEQ_LENGTH=512`.

| Version | LoRA r | Data (stage) | stage2 | stage3 | stage4 | stage5 | Notes |
|---|---:|---|---:|---:|---:|---:|---|
| v18 | 32 | stage5 (2,316) | 84.2% | 70.0% | — | — | MLP LoRA targets added |
| v19 | 32 | stage5 | 93.0% | 90.0% | — | — | |
| v20 | 32 | stage9 (17,532) | 80.7% | 70.0% | 66.7% | 37.3% | |
| v21 | 32 | stage9 | 80.7% | 70.0% | 66.7% | 37.3% | byte-identical to v20 — tiny data delta; prompted the fingerprint safeguard |
| v22 (en prompt) | 32 | stage11 (27,064) | 87.7% | 80.0% | 75.0% | 33.3% | |
| v22 (si prompt) | 32 | stage11 | 82.5% | 90.0% | 75.0% | 33.3% | off-training-distribution prompt, kept to measure the cost of prompt/training mismatch |
| v23 | 32 | stage12 (36,006, v9 answers) | 82.5% | 70.0% | 72.2% | 27.5% | regressed vs v22 — two confounded causes: hand-built share fell 40.2%→30.1% and step count rose 33% |
| v25 | 32 | stage13 (v10 answers) | 80.7% | 100.0% | 39.2%†| 35.3% | trained past its optimum — eval_loss bottomed at step 12828 (epoch 3.00) then rose |
| v26 | 32 | stage13 | 78.9% | 100.0% | 75.0% | 37.3% | step count capped at 12828 to fix v25's over-training |
| v27 | **4** | stage13 | 80.7% | 100.0% | 69.4% | 43.1% | rank-capacity experiment, see below |

† A standalone rerun of just stage4 on v25, 33 minutes later, same
adapter/config, scored 72.2% instead — not 39.2%. Both numbers are
file-verified; the discrepancy is unresolved (possibly generation
nondeterminism, or a since-edited copy of `grammar_test_stage4.jsonl`). Code
comments in `train_grammar_v27.py` cite the 72.2%-consistent set as "the
corrected benchmark" for v25 (stage2 80.7%, stage3 100.0%, stage4 72.2%,
stage5 39.2%), alongside a taught/untaught pair-generalization split of
88%→91% (taught) vs 42%→50% (untaught) vs v24. No v24 eval transcript
survives in `Tested_results/` to independently re-check those v24 numbers —
they're sourced from a code comment, not a saved run.

Continuous metrics (ROUGE-L, char-F1, GLEU) stay saturated at 0.98-1.00
across all these versions and aren't diagnostic — **exact-match accuracy is
the metric that actually moves.**

**v26 vs v27 — the deliberate experiment.** v25 (r=32) hit 91% on
training-taught word-fix pairs but only 50% on untaught pairs — a ceiling
every round since v22 had hit. At r=32 the adapter has ~6,900 trainable
parameters per distinct correction in the 36,006-row corpus — cheap enough
for gradient descent to memorize a lookup table (v25's train_loss of 0.003 is
the signature). v27 shrinks to r=4 (~860 params/correction, 10.5M trainable
params vs v26's 83.9M) to test whether that forces the model to compress
corrections into general orthographic rules instead of memorizing, which
should show up as rising untaught-pair recall even if raw accuracy dips —
consistent with v27's stronger stage5 (43.1% vs v26's 37.3%) despite a
weaker stage4.

**Baselines (ByT5-small, mT5-small).** Trained via `train_grammar_byt5.py` /
`train_grammar_mt5.py`, scored with `score_grammar_predictions.py`:
- **ByT5-small v01** (`google/byt5-small`, full seq2seq direct correction):
  3 epochs on stage13 (28,519 train / 1,518 dev, dedup'd to guarantee zero
  train/dev edit overlap). Edit F0.5 20-38%, overall exact match 27-35%
  across stages — well below SinLLaMA's 70-100%.
- **mT5-small v01**: scored 0% edit precision/recall/F0.5 and 100%
  under-correction rate on stage5 — it essentially learned to copy the input
  unchanged (25.5% overall exact match = only the 13 already-clean rows).
- **ByT5-small v02 (edit-script variant)**: generates a compact edit script
  (`[{"s":12,"e":18,"o":"...","n":"..."}]`, character-offset replacements or
  the literal token `KEEP`) instead of regenerating the full sentence — a
  small seq2seq model is much less likely to hallucinate insertions or
  mangle names/numbers when it only has to describe the diff. Malformed or
  unsafe scripts (deleting a Latin name, changing a number, mutating a
  zero-width joiner) are rejected back to `KEEP` rather than risking
  corrupted output — this safety net is unit-tested directly in
  `test_grammar_edit_script.py`.

---

## 2. Headline Generation

Full version narrative (v17→v20, root-cause analysis, the two-track
artifact-tag fix) is already documented in `CLAUDE.md`'s Headline section and
is accurate — cross-checked against the scripts' own docstrings. This section
adds the dataset structure CLAUDE.md doesn't cover.

### 2.1 Training dataset

Schema (all variants) — flat two-field JSONL:

```json
{"input": "Category: International\nArticle: ජනාධිපති මෛත්‍රීපාල සිරිසේන මහතා අද(29) ප්‍රංශය බලා පිටත්ව...",
 "output": "ජනපති අද ප්‍රංශයට යයි"}
```

No stored length/band field — the band is derived at load time from
`output`'s word count via:

```python
HEADLINE_LENGTHS = {
    "short":  {"min_words": 3, "max_words": 5},
    "medium": {"min_words": 6, "max_words": 7},
    "long":   {"min_words": 8, "max_words": 10},
}
```

A headline outside 3-10 words gets no band (dropped, not clamped).

| File | Rows | Notes |
|---|---:|---|
| `headline_dataset_48k_balanced_train.jsonl` | 43,740 | v17/v18 raw training data |
| `headline_dataset_48k_balanced_train_clean.jsonl` | 43,737 | v19: trailing scraper-tag stripped from `output` |
| `headline_dataset_48k_balanced_train_clean_v20.jsonl` | 43,737 | v20: `input` article body also cleaned, wider tag list |
| `headline_dataset_48k_balanced_val*.jsonl` (3 variants) | 4,794-4,795 | matching val splits |
| `headline_dataset_12k_train/val.jsonl` | 10,800 / 1,200 | orphaned — no script references these |
| `headline_dataset_24k_train/val.jsonl` | 21,605 / 2,395 | orphaned — no script references these |

The 12k/24k files predate the 48k set by file mtime (May 7 → Jun 23 → Jul 10)
but nothing in code or git history ties them together as an explicit growth
path — they're superseded/abandoned experiments, not a documented lineage.
`train_headline.py` (the original v17 script) already points at the 48k file.

Cleaning is visible on the same record: train row 25's `output` goes from
`...සූදානම්!` (v17/v18) → `...සූදානම්` (v19 strips the trailing `!`). v19→v20
only touches 1 line out of 43,737 in this checkout — a known gap where the
v19 separator regex doesn't include `/`, so `[Photo/Video]` only partially
strips.

### 2.2 Test dataset

`test_headline_v18/v19/v20.py` each load the **val split of the
same-version training dataset** (not a separate held-out test set), generate
each article at all 3 bands, and compute per band: `in_band` (generated word
count falls inside the *requested* band), `artifact` (matches a leftover
scraper-tag regex), `empty`, and own-band `rouge1`/`rougeL`/`bleu` (only
scored when the requested band matches the reference headline's natural
band).

---

## 3. Summarization

CLAUDE.md's Summarizer section documents v06's length-conditioning
mechanism accurately, but is stale on one point: it describes
`summarizer/abstractive/` as covering "adapters v02-v06" — a full v07
pipeline (`7_train_summarizer.py`, `7_test_summarizer.py`, data-quality
validation) is also live on disk (adapter `summarization_sinllama_v07/`,
mtime 2026-08-11, commit `86536dd`) and isn't mentioned there yet.

### 3.1 Training dataset

`7_train_summarizer.py` (current) loads
`summarizer/data/6_multilength_summaries_clean.jsonl` (35,547 rows);
`6_train_summarizer.py` loads the uncleaned `6_multilength_summaries.jsonl`
(35,569 rows) — the 22-row difference is exactly what v07's data-quality
pass drops (§3.3). Both also optionally ingest
`summarizer/data/5_qwen_summaries.jsonl` (170,923 rows, single-summary
schema) as supplementary "long"-bucket examples where the compression ratio
fits.

Schema — one row per **article**, with all three length variants as parallel
columns (not one row per band):

```json
{"source_file": "derana_merged.json",
 "title": "කොටස් වෙළෙඳපොළ අද පසු බසී",
 "content": "කොළඹ කොටස් වෙළෙඳපොළ මිල දර්ශකවල පසුබැස්මක් අද (24) දිනයේ දී වාර්තා විය...",
 "url": "...", "original_category": "Business", "mapped_category": "Business",
 "date_published": "",
 "summary_short": "කොළඹ කොටස් වෙළෙඳපොළ මිල දර්ශක පහළ ගොස්...",
 "summary_medium": "කොළඹ කොටස් වෙළෙඳපොළේ සියලු කොටස් හා S&P SL20 දර්ශක පහළ ගිය අතර...",
 "summary_long": "..."}
```

Length conditioning comes from these three parallel summaries plus a
compression-ratio filter (identical constants used at train and test time):

```python
BUCKET_FILTERS = {
    "short":  {"min_ratio": 0.04, "max_ratio": 0.18, "max_summary_tokens": 70},
    "medium": {"min_ratio": 0.12, "max_ratio": 0.32, "max_summary_tokens": 120},
    "long":   {"min_ratio": 0.22, "max_ratio": 0.55, "max_summary_tokens": 190},
}
```

`summarizer/summarization_dataset.jsonl` (500 rows, `title/content/summary`
schema) and `test_summarization_dataset.jsonl` (20 rows) are legacy —
unreferenced by any current script.

### 3.2 Test dataset

`7_test_summarizer.py` loads the same clean file, keeps only records with
all 3 `summary_{bucket}` fields present, samples N articles (default 15),
and generates at all 3 buckets per article.

### 3.3 v07: data-quality validation (new, not yet in CLAUDE.md)

`clean_multilength_dataset.py` + `data_quality_checks.py` drop any record
whose `summary_short/medium/long` trips either check:
- **`detect_word_glue()`**: flags a contiguous Sinhala/digit run ≥24 chars
  with no internal whitespace — a decoding glitch where words got
  concatenated.
- **`check_numeric_unit_consistency()`**: flags a number appearing in both
  article and summary but attached to a *different* scale unit
  (ලක්ෂ/මිලියන/කෝටි/බිලියන) — e.g. article says "මිලියන 476", summary says
  "ලක්ෂ 476", a ~10x factual error that a digit-only hallucination check
  would miss. Run against the full 35,569-row corpus, this dropped 22 rows.

The same two detectors are reused live at **eval** time in
`7_test_summarizer.py` as new metric fields:
- **`glue_pct`**: % of *generated* summaries where `detect_word_glue()`
  fires — measures the defect in live model output, not just frozen
  reference data.
- **`unit_mismatch_pct`**: % of generated summaries where
  `check_numeric_unit_consistency()` fires against the source article.

Alongside the existing `rougeL`/`rouge1`/`rouge2` (native grapheme-cluster
ROUGE), `mean_ratio` (generated/article token ratio), `in_band_pct`
(ratio falls inside the bucket's band), and `clean_end_pct` (output ends in
`.`/`?`/`!`, i.e. wasn't cut off mid-sentence). v06-era scripts had no
glue/unit-mismatch equivalent — this pair is v07's actual contribution.

---

## 4. Style Rewriting

Rewrites an article into one of 5 styles. Order used consistently across all
scripts: **formal → editorial → sports → youth → feature**
(`style_1_formal_news`, `style_2_editorial`, `style_3_sports`,
`style_4_youth`, `style_5_feature`; `DEFAULT_STYLE = "formal"`).

| Style | Emphasis |
|---|---|
| formal | Objective/passive voice, inverted pyramid (most important fact first), no opinion, formal verbs (ඇත/තිබේ/විය) |
| editorial | Analytical/reflective tone, logical cause→effect flow, sparse first-person, bans the stock closer "මෙම සියලු කරුණු සමස්තයක් ලෙස සලකන කල..." |
| sports | Energetic/active verbs, leads with the decisive result, sport-specific vocabulary only when the source *is* sports, ≤1 "!" |
| youth | Simple conversational Sinhala, short sentences, no emojis, bans the forced closer "ඒ නිසා යාලුවනේ..." |
| feature | Narrative flow, human-interest angle only if supported by source, bans invented scenes/weather/emotions/dialogue |

### 4.1 Training dataset

Two unrelated schema eras exist — do not conflate them.

**Era A (legacy, pre-v07) — not used by any current script.**
`style_dataset_stage1.jsonl` (100 rows), `stage2.jsonl` (600 rows), `stage3.jsonl`
(0 rows, empty), `style_dataset_improved_deduped.jsonl` (1,000 rows). Schema
`{"instruction", "input", "output", "metadata": {"style_id", "category", "headline", "source"}}`.

**Era B (current, v07 through v12).** Schema
`{"content", "style", "rewritten_text", "category", "url", "date_published"}`:

```json
{"content": "...", "category": "Local", "url": "https://sinhala.adaderana.lk/news/49740",
 "date_published": "", "style": "style_4_youth", "rewritten_text": "..."}
```

- `style_dataset2_final_cleaned.jsonl` (7,554 rows) — the **v11** training
  file.
- `/home/jovyan/style_rewriter/data/style_dataset2_fixed.jsonl` (7,555 rows)
  — the **v12/current** training file (`style_rewriter/` is gitignored, not
  checked into this repo, but present on disk).

Actual v12 file style distribution: `style_1_formal_news`=2,471,
`style_3_sports`=2,122, `style_4_youth`=1,425, `style_2_editorial`=998,
`style_5_feature`=539 — a ~4.6x imbalance (formal vs. feature) that's the
direct motivation for v12's weighted sampling (below).

**Data lineage:** a 521,980-article raw corpus (`train1.jsonl`) → ~4,447-article
subset → 5-style generation via NVIDIA API teacher → 22,236 generated rows →
offline QC audit → 7,554 clean rows (34.0% pass rate) → grammar-correction
pass (`Correct_style_dataset.py`) → word-corruption fixer → 7,555-row v12
file.

**`convert_record()`** (`train_style.py`) drops rows with `status=="failed"`,
an `error` field set, `style` not in `STYLE_IDS`, empty
`content`/`rewritten_text`, `rewritten_text` under 50 chars, or a
`qc_issues` entry in `DROP_QC_ISSUES`. **This last filter is dead code on
the real training files** — neither `style_dataset2_final_cleaned.jsonl` nor
`style_dataset2_fixed.jsonl` contains a `qc_issues`, `status`, or `error`
field; cleaning already happened offline before these files were written.

`article_level_split()` groups all 5 style-variants of one article together
(keyed by URL) before the 85/15 train/val split, so no article leaks across
the split with a different style label.

### 4.2 QC / quality-issue tags

Traced to `style_rewriter/data/audit_dataset_quality.py` (gitignored, not in
this repo — despite `train_style.py`'s comment attributing them to
`generate_style_dataset.py`'s `check_quality()`, which has never existed in
any git-tracked version of that file):

| Tag | Trigger |
|---|---|
| `missing_required_closing` | Style requires a fixed closing phrase and it's absent |
| `possible_stutter_duplication` | A 2-4 char Sinhala grapheme sequence immediately repeated |
| `suspiciously_short` | Rewrite under 20 characters |
| `honorific_gender_mismatch` | Source uses one honorific gender group, rewrite uses the other |
| `foreign_symbol_added` | Rewrite contains `&%#@` that wasn't in the source |
| `wrong_style_marker_present` | Rewrite contains another style's required marker phrase |
| `possible_numeric_hallucination` | Present in `train_style.py`'s drop-set but never actually produced anywhere — dead/aspirational |

Real numbers from the audit: of 22,236 generated rows, only 34.0% passed.
Dominant failure was `possible_stutter_duplication` (51.9% of all rows) and
`missing_required_closing` (25.4%). Per-style clean rate ranged from 56.4%
(formal) down to 12.3% (feature) — which is why feature ended up the
smallest, most imbalanced bucket.

### 4.3 Test dataset

**No dedicated style test/val file exists on disk** — unlike grammar's
`grammar_test*.jsonl` or headline's `*_val.jsonl`. `test_style.py` reuses the
training JSONL and reproduces `train_style.py`'s exact split (same seed, same
URL-keyed grouping) to carve out a held-out set at runtime, then further
filters to articles with all 5 styles present and ≥40 words, taking the 20
longest — 20 articles × 5 styles = 100 test cases.

**Important gap:** `test_style.py`'s `STYLE_ADAPTER` and `TEST_DATA_PATH`
still point at **v11** (`style_sinllama_v11`, `style_dataset2_final_cleaned.jsonl`)
and haven't been updated since commit `bbf3090` (2026-08-02) — before the
v12 rework (`8c08c0b`, 2026-08-14). As of this checkout there is no committed
eval run against v12; the last real numbers on file are v11's:

| Style | overall_quality | ROUGE-1 F1 | BLEU | length_preservation |
|---|---:|---:|---:|---:|
| formal | 0.8505 | 0.8683 | 0.7144 | 1.0000 |
| sports | 0.7828 | 0.8010 | 0.6004 | 0.9994 |
| editorial | 0.7244 | 0.7774 | 0.5504 | 0.7961 |
| youth | 0.7038 | 0.7241 | 0.4634 | 0.9840 |
| feature | 0.6765 | 0.7172 | 0.4733 | 0.8240 |
| **overall (n=100)** | **0.7476** | | | |

Overall score weights: ROUGE-1 F1 ×0.25, ROUGE-2 F1 ×0.20, BLEU ×0.20,
TF-IDF cosine ×0.20, length-preservation ×0.15.

`test_style_long.py` is **not a runnable script** despite the `.py`
extension — it's a saved plain-text console transcript from an earlier
prototype iteration, useful only as a historical example output.

### 4.4 Version history

v1-v6 predate this repo's git history entirely (root commit); the earliest
recoverable state already targets v07.

| Ver | LoRA r/α | Epochs | LR | Train data | What changed & why | Adapter on disk? |
|---|---|---:|---|---|---|---|
| v07 | 32/64 | 8 | 2e-4 | `style_dataset.jsonl` (~784 articles/style) | First git-tracked version; switched off the legacy instruction/input/output schema | No |
| v08 | 32/64 | 5 | 2e-4 | `style_dataset2_dub.jsonl` (~22,237 rows) | Bigger raw dataset; epochs cut from 8 — v07's 8 epochs over 784 articles/style had overfit | Yes |
| v09 | **16**/32 | 3 | 2e-4 | same | Rank halved + weight_decay added — overfitting fix. eval_loss 1.087→0.849, train/eval gap 2.28x→1.26x | No |
| v10 | **24**/48 | 3 | 2e-4 | same | Rank bumped back up — 16 undershot capacity | No |
| v11 | 24/48 | **5** | **1e-4** | `style_dataset2_final_cleaned.jsonl` (7,554 clean rows) | Smaller-but-clean dataset → more epochs + gentler LR; added article-level split, physical oversampling via `random.choices` to balance styles | Yes |
| **v12** | 24/48 | **3** | 1e-4 | `style_dataset2_fixed.jsonl` (7,555 rows, word-corruption-fixed) | "7,555 clean rows do not require 5 epochs"; **replaced physical oversampling with `WeightedRandomSampler`** (no duplicate rows); EOS-safe label handling; teacher model + API-key handling reworked in `generate_style_dataset.py` (env var instead of hardcoded) | Yes |

Only `style_sinllama_v08, v09, v11, v12` currently exist under
`models/adapters/` — v07 and v10 were presumably superseded and removed.
`serve_sinai.py` auto-selects the highest-numbered adapter present, so
production serves **v12** today; CLAUDE.md's architecture example (`style_sinllama_v07/`)
is stale and should be updated.

### 4.5 Known issues worth tracking

- **Training/inference prompt mismatch.** `train_style.py` explicitly
  comments "keep these [STYLE_RULES] identical in your inference script" —
  they currently aren't. Training uses short **English** bullet rules (9-11
  lines/style) plus a fixed 16-item fact-preservation block and an
  `### Input:` label. Production inference (`work/tasks/style.py`,
  `prompt_style()`) uses long **Sinhala** rules (30-80 lines/style, copied
  from the teacher-model prompts in `generate_style_dataset.py`, not from
  `train_style.py`) with no fact-preservation block and a `Text:` label. The
  deployed v12 adapter is being prompted at inference with a materially
  different instruction language, length, and template than what it was
  fine-tuned on — `test_style.py`'s own inline comments describe this class
  of mismatch having caused real score regressions in earlier evals.
- **`Correct_style_dataset.py`** (the grammar-correction pass in the data
  lineage) still has a hardcoded NVIDIA API key in source, unlike
  `generate_style_dataset.py` which was fixed at v12 to read from an
  environment variable. Should be rotated and switched to match.
- **No v12 eval exists yet** — `test_style.py` still targets v11 (see §4.3).
