# SinhalaJournal-LLM Summarization Component — Complete History (v01–v07)

This document is the full, honest account of the summarization component's
development, from the first mT5 baseline through seven SinLLaMA adapter
versions, ending with the leakage-controlled re-evaluation completed on
2026-08-26. It exists because the research paper (`summarizer/research-paper.tex`)
only covers v06/v07 in depth, and the day-to-day project docs
(`summarizer/SUMMARIZATION_NEXT_STEPS.md`, `CLAUDE.md`) are forward-looking
trackers, not a retrospective. Nothing here is inferred without a source —
where a claim can't be backed by a file, script, or persisted eval result in
the repo, that's stated explicitly rather than guessed.

**Where things stand today (2026-08-26):** v07 is live in production
(auto-selected by version-sort, not by merit). A leakage-controlled,
article-level evaluation completed today found **v06 and v07 are
statistically tied** on every automatic metric measured — the older
comparison that appeared to favor v06 was an artifact of a data-leak
affecting ~81% of the "held-out" test set, not a real quality difference.

**Update (2026-08-31):** a new set of evaluation numbers now exists,
carried in the PP2 Extended presentation deck (`PP2_Extended.pdf`,
section "B7 / Summarization"). These add two things the record did not
previously contain: a **measured cross-paradigm comparison**
(SinLLaMA-v07 vs. mT5-base vs. TextRank) and a **measured cross-version
comparison** (v04 → v05 → v06 → v07), both scored with an expanded metric
set that now includes BERTScore and GLEU alongside ROUGE. The single most
important new fact: **mT5-base is not merely weaker on Sinhala, it is
measurably broken** (ROUGE-L 0.2%, degenerate non-Sinhala output) — which
retires the earlier estimate in Section 17.2 that put it around 0.35–0.42.
These numbers are recorded in full in **Section 18**, along with an
explicit note that their sample size and eval provenance are not yet
established, and therefore that they do **not** yet overturn the
Section 13 frozen-split finding.

---

## 1. Overview: what "summarization" means in this project

The task: given a Sinhala news article, produce an abstractive (not
extracted-sentence) summary. Later versions (v06+) add length control —
the caller requests short (~10% of source length), medium (~20%), or long
(~35%), and the model is trained to hit that band.

All SinLLaMA-based versions (v02 onward) share one architecture: the shared
`SinLLaMA-merged-base` (Llama-3-8B + Sinhala vocabulary extension, see
`CLAUDE.md`) loaded in 4-bit NF4 with BF16 compute, with a task-specific LoRA
adapter trained on top. What changed release to release was almost entirely
**the training data** — which teacher model generated the silver summaries,
how much of it there was, and how it was filtered — not the model
architecture or the fine-tuning method. This turns out to be the throughline
of the whole history: every meaningful jump (and every scare) in this
component's story traces back to data, not architecture.

v01 is the exception: it's an mT5-base model, not a SinLLaMA adapter,
trained specifically as a controlled comparison point (Section 3).

---

## 2. v01 — mT5-base baseline

**What it is:** `google/mt5-base` fine-tuned with LoRA
(`abstractive/train_mt5_lora.py`), **not** a SinLLaMA adapter. Adapter
folder: `work/sinllama/models/adapters/summarization_mt5_v01/`.

**Why it exists:** trained deliberately on the *exact same* Qwen silver
summary dataset and filtering rules as SinLLaMA v05
(`data/5_qwen_summaries.jsonl`), per the script's own docstring: "for a
direct, scientific comparison." The intent was to isolate the effect of
base-model choice (mT5 vs. SinLLaMA/Llama-3) by holding the training data
constant.

**Status:** the adapter exists and is checkpointed
(`checkpoint-1504`, `checkpoint-1880`). It's still used today —
`work/serve_sinai.py` serves it as a comparison "teacher" model on the
live Model Comparison page (per `CLAUDE.md`). However, **no persisted
quantitative evaluation for v01 survives in this repo.** An
`evaluate_mt5.py` script existed at some point (its compiled
`.pyc` is still in `abstractive/__pycache__/`), but the source file itself
is gone — most likely removed in the `c406b23` "Clean up repo for GitHub"
commit, which explicitly excluded some models/datasets/caches. No numeric
mT5-vs-SinLLaMA comparison result is recoverable from the repo as it
stands.

**Superseded 2026-08-31:** a numeric mT5-vs-SinLLaMA comparison *does*
now exist — it just wasn't produced by the lost `evaluate_mt5.py`. The
PP2 Extended deck reports `mt5-base` scoring ROUGE-L 0.2% / ROUGE-1 0.2%
/ ROUGE-2 0.0% / BERTScore 74.1% / GLEU 0.0% against SinLLaMA-v07's
50.9 / 50.9 / 47.9 / 91.1 / 30.8, with sample output that is not Sinhala
at all. See Section 18. Note that the deck's `mt5-base` row is labelled
as the encoder-decoder *baseline*, and it is not yet confirmed whether it
refers to the LoRA-fine-tuned `summarization_mt5_v01` adapter described
here or to a stock, un-fine-tuned `google/mt5-base` — the distinction
matters a great deal for what the number actually proves, and is one of
the open provenance questions listed in Section 18.4.

---

## 3. v02 — first SinLLaMA adapter, Llama-4 Maverick silver data

**Training data:** `data/llama4_summaries_2.jsonl`, 28,986 rows. Summaries
generated by `abstractive/2_llama4_summary_generator.py` calling NVIDIA's
hosted `meta/llama-4-maverick-17b-128e-instruct` API, single summary per
article (no length conditioning yet — that's a v06 feature).

**Training config** (`abstractive/2_train_summarizer_llama4.py`):
rank=32, alpha=64, **dropout=0.0**, max_seq_len=2048, 5 epochs, effective
batch 16, lr=2e-4. Output: `summarization_sinllama_v02`.

**Result:** `summarizer/eval_results/eval_results.json`, N=10, dated
2026-05-11:

| Metric | Value |
|---|---:|
| ROUGE-1 | 0.0317 |
| ROUGE-2 | 0.0 |
| ROUGE-L | 0.0317 |
| Exact match | 0/10 |
| Strong match | 0/10 |
| Compression | 29.3% (predicted mean 40.4 words vs. reference mean 23.1) |

**This is a near-total failure** — ROUGE-1 of 0.03 means essentially no
lexical overlap with references at all, worse than what a reasonable
extractive baseline would produce by accident.

**Why, most likely:** `CLAUDE.md`'s Critical Constraints section states
plainly: *"`lora_dropout=0.05` minimum (not 0.0) — setting dropout to 0.0
activates Unsloth's fast CUDA LoRA kernel, which breaks on 4-bit quantized
weights due to dtype mismatches in backward passes."* v02's config sets
`LORA_DROPOUT = 0.0` — the exact condition this constraint warns against.
No training log survives for v02 to confirm this directly (see Section 12
for what logs do and don't exist), so this is stated as a **strong
correlation, not a confirmed root cause** — but it is consistent with
every other data point: v02 is the *only* summarizer training script in
this repo's history with `dropout=0.0`, and it's also the only version
with a nonsense-level result. Every later script (`3_`, `4_`, `5_`, `6_`,
`7_train_summarizer*.py`) uses `dropout=0.05`, and none of them reproduce
anything close to v02's collapse.

**Adapter folder status:** `summarization_sinllama_v02/` does **not**
exist in `work/sinllama/models/adapters/` today (only v03 through v07 are
present). It was presumably deleted or overwritten at some point — its
absence is itself informative, since it means v02 was abandoned rather
than kept as a labeled-bad reference point.

---

## 4. v03 — diffusiongemma silver data, round 1

**Training data:** `data/diffusiongemma_summaries.jsonl`, 29,000 rows.
Generated by `abstractive/3_diffusiongemma_summary_generator.py` via
NVIDIA's hosted `google/diffusiongemma-26b-a4b-it` model, with an explicit
Sinhala-language rule-based prompt (use only article info, don't add new
info, don't change entities/numbers, target 10–30% length, no
meta-commentary).

**Training config** (`abstractive/3_train_summarizer_diffusiongemma.py`):
rank=32, alpha=64, dropout=0.05 (fixed from v02), max_seq_len=2048, 5
epochs, effective batch 16, lr=2e-4. Output: `summarization_sinllama_v03`.

**Result:** **no persisted evaluation JSON survives for v03.** Unlike v02
and v04, no matching file exists in `eval_results/` or `6_eval_results/`.
`3_test_summarizer.py` exists and presumably was run manually at the time,
but its output wasn't saved to disk in a form that survived to today. This
is a real gap in the record, not a "zero result" — v03's quality is simply
unknown from this repo alone.

**Adapter status:** present at
`work/sinllama/models/adapters/summarization_sinllama_v03/`.

---

## 5. v04 — diffusiongemma silver data, round 2 (scaled up)

**Training data:** `data/4_diffusiongemma_summaries.jsonl`, **521,980
rows** — an 18x scale-up from v03's 29,000, using essentially the same
generation approach (`abstractive/4_diffusiongemma_summary_generator.py`,
same model and prompt rules) but run across nearly the entire 665,887-
article filtered corpus rather than a small pilot slice.

**Training config** (`abstractive/4_train_summarizer_diffusiongemma.py`):
identical to v03 — rank=32, alpha=64, dropout=0.05, max_seq_len=2048, 5
epochs, effective batch 16, lr=2e-4. Output: `summarization_sinllama_v04`.

**Result:** `summarizer/eval_results/eval_results-2.json`, N=15, dated
2026-07-12:

| Metric | Value |
|---|---:|
| ROUGE-1 | 0.6680 |
| ROUGE-2 | 0.3724 |
| ROUGE-L | 0.4215 |
| Exact match | 0/15 |
| Strong match | 1/15 |
| Partial match | 14/15 |
| Compression | 17.5% (predicted mean 30.3 words vs. reference mean 22.3) |
| Mean latency | 2.52s |

**This is a dramatic recovery from v02** (ROUGE-1 0.668 vs. 0.032) — most
plausibly explained by two compounding fixes: the dropout correction
(0.05 instead of 0.0, avoiding the broken CUDA kernel path) and the ~18x
larger, more diverse training corpus. Both changed at once between v02 and
v04 (v03 sits in between with the dropout fix but the smaller corpus, but
its result is unrecorded), so this repo's evidence can't cleanly separate
"dropout fix mattered" from "scale mattered" — both plausibly did.

**Adapter status:** present at
`work/sinllama/models/adapters/summarization_sinllama_v04/`.

---

## 6. v05 — Qwen3-80B silver data

**Training data:** `data/5_qwen_summaries.jsonl`, 170,923 rows. Generated
by `abstractive/5_qwen_summary_generator.py` via NVIDIA NIM's free
`qwen3-next-80b-a3b-instruct` endpoint.

A separate generator, `abstractive/5_gemini_summary_generator.py` (Gemini
2.0 Flash, multi-API-key rotation, resumable), was also built around this
time — its infrastructure (rate-limiting, resumable output) was later
reused directly for the Phase 5 LLM-judge factuality plan (Section 15).
Its own output, `data/gemini_summaries.jsonl`, is only 403 rows — a small
pilot run, not a dataset any tracked adapter version trained on.

**Training config** (`abstractive/5_train_summarizer_qwen.py`): identical
to v03/v04 — rank=32, alpha=64, dropout=0.05, max_seq_len=2048, 5 epochs,
effective batch 16, lr=2e-4. Output: `summarization_sinllama_v05`.

**Result:** **no persisted evaluation JSON survives for v05 either**,
same gap as v03. v05's quality is not independently knowable from this
repo. This dataset is also the one v01 (mT5) was deliberately trained on
for a base-model comparison (Section 3) — but since neither v05's own
eval nor v01's eval survived, that comparison's numeric result is lost
too, even though the controlled setup for it clearly existed at the time.

**Adapter status:** present at
`work/sinllama/models/adapters/summarization_sinllama_v05/`.

**Note on data reuse:** the "long"-bucket portion of this 170,923-row
Qwen file was later reused as a supplementary data source in both v06 and
v07's training scripts (`6_train_summarizer.py`, `7_train_summarizer.py`)
— specifically, the subset whose compression ratio fits the "long" bucket
under v06/v07's per-bucket bands, since a global 0.05–0.50 ratio filter
used at the time would have discarded 73.6% of it unnecessarily. So v05's
data didn't disappear after v05 — part of it fed forward into every
subsequent version.

---

## 7. v06 — length-conditioned summarization (short/medium/long)

This is the first version with **native length control** — the feature
the web app's summary-length selector actually needs, per
`6_train_summarizer.py`'s docstring.

**Training data:** `data/6_multilength_summaries.jsonl`, 35,569 articles,
each paired with three reference summaries (`summary_short`,
`summary_medium`, `summary_long`) targeting ~10%/20%/35% of source length.
Generated by `abstractive/6_multilength_summary_generator.py` — a
significant infrastructure change from v02-v05: instead of one hosted API,
this calls a **local, 9-router OpenAI-compatible gateway** running
`ollama/gpt-oss:120b`, with automatic quality validation (word caps,
length-ratio checks, hallucination/meta-commentary detection) and resume
support.

Training explodes each article into up to 3 samples (one per length
bucket), each carrying its own length instruction in the prompt, so the
model learns length as a genuine conditioned behavior rather than a fixed
habit. Optionally ingests the Qwen "long"-bucket leftovers described in
Section 6.

**Training config** (`abstractive/6_train_summarizer.py`, matches
`research-paper.tex` Table II): rank=32, alpha=64, dropout=0.05,
max_seq_len=2048, effective batch 16, lr=2e-4, **3 epochs** (down from 5 —
this is also where the "eval loss rises after epoch 3" lesson from the
grammar component, documented in the paper, gets applied project-wide).
Output: `summarization_sinllama_v06`.

**Tracked result:** `6_eval_results/v06_eval_20260726_025630.json`, N=15
articles × 3 lengths = 45 outputs, dated 2026-07-26 (this is the number
reported in `research-paper.tex` Table III):

| Band | Target compression | Observed compression | In-band | Clean ending | ROUGE-L |
|---|---:|---:|---:|---:|---:|
| Short | 10% | 11.77% | 93.3% (14/15) | 100% | 0.616 |
| Medium | 20% | 23.13% | 100% | 100% | 0.579 |
| Long | 35% | 36.29% | 100% | 100% | 0.549 |

44 of 45 outputs land in their requested band; all 45 end cleanly (no
mid-sentence truncation).

**But length control ≠ factual accuracy.** The paper documents a specific
failure found in a stored long prediction: a reference to "Portugal and
Uruguay" was changed to "Pakistan" — the model preserved fluency and
length but altered the actual entities in the story. A separate audit of
the *teacher data itself* (not the model's output) found 21 genuine
word-joining or number/unit errors plus one false positive in the silver
corpus — some changing the reported scale by 100x to 1000x (e.g. "455
billion" rendered as "455 million"). This is the origin of the
`data_quality_checks.py` glue/unit checks used from v07 onward.

**Decoding note:** an earlier `no_repeat_ngram_size=3` setting was found
to corrupt the opening grapheme cluster of Sinhala output (scans the full
`input_ids` including the prompt/article, blocking the summary from
opening with the article's own first phrase and truncating mid-grapheme,
e.g. "හිටපු" → "ිටපු"). Removed; current serving and eval scripts use
`repetition_penalty=1.15` with no n-gram blocking.

**Note on tracked-vs-untracked evals:** a second, much smaller file,
`6_eval_results/v06_eval_20260809_163811.json` (N=9, i.e. 3 articles × 3
lengths), also exists, dated 2026-08-09. This is a dev smoke-test, not the
tracked result — same pattern as the 3-article smoke test run before
today's full frozen-split evaluation (Section 11). It's mentioned here
only so its existence doesn't get mistaken for a second independent
result; the 45-output file is the one that matters and the one the paper
cites.

**Adapter status:** present at
`work/sinllama/models/adapters/summarization_sinllama_v06/`.

---

## 8. v07 — cleaned data, same recipe otherwise

**What changed from v06:** exactly one thing — the training data was
cleaned. `abstractive/clean_multilength_dataset.py` re-scans v06's
35,569-row corpus and drops any record whose summary contains a
word-spacing-glue or numeric-unit-mismatch defect (the two defect classes
identified from v06's teacher-data audit, Section 7), producing
`data/6_multilength_summaries_clean.jsonl` — **35,547 rows, 22 dropped.**
Verified (this session, Section 10) that this cleaning is pure row
removal, not content editing: all 35,547 shared rows are byte-identical
between the raw and cleaned files.

`abstractive/7_multilength_summary_generator.py` and
`abstractive/7_train_summarizer.py` also add the same glue/unit checks
at generation time and train time respectively, so future
regeneration/retraining rejects these defects at the source rather than
needing a retroactive cleaning pass — defense in depth, not just a
one-off fix.

**Training config:** identical to v06 in every hyperparameter (rank=32,
alpha=64, dropout=0.05, max_seq_len=2048, effective batch 16, lr=2e-4, 3
epochs) — data cleanliness was meant to be the only variable. Output:
`summarization_sinllama_v07`.

**Tracked result:** `6_eval_results/v07_eval_20260811_094434.json`, N=15
articles × 3 lengths = 45 outputs, dated 2026-08-11, same 45-output
protocol as v06's tracked eval:

| Band | Observed compression | In-band | Clean ending | ROUGE-L | Glue defects | Unit-mismatch |
|---|---:|---:|---:|---:|---:|---:|
| Short | 12.07% | 100% | 100% | 0.488 | 0% | 0% |
| Medium | 21.02% | 86.7% (13/15) | 100% | 0.518 | 0% | 0% |
| Long | 30.98% | 93.3% (14/15) | 100% | 0.488 | 0% | 0% |

**On this protocol, v07 looked worse than v06** — lower ROUGE-L on every
band (e.g. 0.549 vs. 0.488 for long), and a *worse* in-band rate on
medium/long despite a *better* one on short. This is the number
`research-paper.tex` originally reported as the summarizer's headline
v06-vs-v07 comparison. **It turned out to be measuring the wrong thing —
see Section 10.**

**Adapter status:** present at
`work/sinllama/models/adapters/summarization_sinllama_v07/`, checkpointed
2026-08-11 (`checkpoint-13980`, `checkpoint-20970`).

**Crucial deployment fact, unrelated to quality:** `work/serve_sinai.py`'s
`find_latest_adapters()` scans the adapters directory and auto-selects
whichever version-numbered folder parses highest — v07 > v06, so **v07 has
been serving live production traffic since it was trained**, purely
because of the version-sort tiebreak, not because anyone evaluated it and
chose it. This remained true even while the only existing comparison
(the 45-output result above) suggested v06 was better.

---

## 9. The documentation gap this session started from (2026-08-22)

On 2026-08-22, this project's `research-paper.tex` draft and a
ChatGPT-authored proposal document (`improve_summarization_component.md`)
both stated, incorrectly: *"the project records do not contain an
authoritative saved v07 training run/evaluation."*

This was checked directly against the filesystem and git history and found
to be **wrong**: v07 was fully trained, checkpointed, and had a real,
git-committed evaluation (`74ab381`, "feat(summarizer): add v06 and v07
evaluation results," 2026-08-14). `work/sinllama/COMPONENTS.md` (also
updated 2026-08-14) already correctly documented v07's existence — the
gap was that this fact hadn't propagated into the paper draft or into
`CLAUDE.md` (which at the time still said "adapters v02–v06"). Both docs,
plus `CLAUDE.md`, were corrected same-day (Phase 0 of the plan in
`SUMMARIZATION_NEXT_STEPS.md`).

More importantly, this is also when the fact that **v07 was already live
in production** (Section 8, version-sort accident) surfaced as something
that needed explicit acknowledgment — not changed, per explicit
instruction to make no production changes without approval, but written
down accurately.

---

## 10. Root cause: why the v06-vs-v07 comparison couldn't be trusted

Two independent problems were found in `6_train_summarizer.py` and
`7_train_summarizer.py` (both build data the same way):

**A. Train/eval split leaks across length variants of the same article.**
Each article is exploded into up to 3 samples (short/medium/long) *before*
the train/val split. The split then shuffles this flat sample list and
divides 85/15 (`7_train_summarizer.py:259` `random.seed(SEED)`, `:300`
`random.shuffle(samples)`, `:75` `TRAIN_SPLIT = 0.85`). So the short
summary of article X can land in train while the medium summary of the
*same* article X lands in validation — the model can see part of an
article's content during training and be "evaluated" on a different
length of the same article.

**B. No frozen, persisted, blind test set existed.** `7_test_summarizer.py`
(the script used for both v06's and v07's tracked 45-output evals) drew
N=15 random articles straight from the training corpus file itself, with
its own seed, every time it ran — not a held-out file set aside once. Both
the paper and the proposal doc already flagged this as a known limitation;
it turned out to be worse than a caveat.

**C. v06 and v07 don't even train on the same source file.** v06 trains on
the raw `6_multilength_summaries.jsonl` (35,569 rows); v07 trains on the
cleaned `6_multilength_summaries_clean.jsonl` (35,547 rows). Verified this
session that the cleaning is pure row removal — all 35,547 shared rows are
byte-identical between the two files, confirmed via direct Python
comparison. This mattered for correctly building a frozen split usable by
*both* recipes (Section 11) — a split built from the cleaned file would
have silently excluded the 22 raw-only rows from v06's retrain,
collapsing the exact variable under test.

One thing that turned out to be **simpler** than the ChatGPT proposal doc
assumed: it recommended grouping the corpus by article before splitting.
That step was unnecessary — the raw corpus already has exactly one row per
article (`wc -l` count equals unique-URL count), with `summary_short`/
`summary_medium`/`summary_long` as three columns on one record. The leak
isn't in the data's shape; it's entirely inside the training scripts'
explode-then-shuffle order. An article-level split just means splitting
*rows*, not grouping first.

---

## 11. Phase 1 — building a frozen, article-level split and measuring the damage

**`abstractive/8_freeze_dataset_split.py`** (run once, 2026-08-22): reads
the **raw** `6_multilength_summaries.jsonl` (not the cleaned file — see
Section 10C for why), asserts URL uniqueness, then does a single
deterministic 80/10/10 split by row (seed 42), persisting the result so it
can never be silently regenerated with different boundaries:

| Split | Rows |
|---|---:|
| Train | 28,455 |
| Val | 3,556 |
| Test | 3,558 |
| Eval subset (fixed slice of test, for repeated eval runs) | 300 |

Source file sha256: `2d1f21ee3ca13ab8f80b0fdca820f73e0acf62eb5922267672859db6f5ef0e61`.
A manifest (`data/summarization_frozen_split_manifest.json`) records the
seed, ratios, per-split URL-list hashes, and source hash for
reproducibility. The script refuses to overwrite existing frozen files —
enforced with a hard `SystemExit`, since the entire point of a frozen
split is that its boundaries never move once anything has been
trained or evaluated against them.

(This split file was accidentally built from the *cleaned* corpus on the
first attempt, which would have silently made "retraining v06" actually
mean "retrain v06's recipe on v07's dataset" — caught before use, when the
question "do v06 and v07 even train on the same data?" was asked directly;
rebuilt from the raw file per Section 10C.)

**`abstractive/8_audit_split_contamination.py`** (CPU-only, no model load
needed): replays each existing adapter's *actual* training script logic —
exact sample-building/filtering rules, `random.seed(42)`, and the same
`random.shuffle()` + 85/15 split — using only `(url, bucket)` identity
tuples (this works because `random.shuffle()`'s output depends only on
list length/order, not content, so the *identical* permutation can be
reproduced without needing the full original prompt text). This
reconstructs exactly which articles each adapter's original training
actually included, then checks how much that overlaps the new frozen test
set.

**Result** (`6_eval_results/split_contamination_audit_20260822_122918.json`):

| Adapter | Train articles | Own-split straddling | Frozen-test contamination | Frozen-eval-subset contamination |
|---|---:|---:|---:|---:|
| v06 | 74,103 | 12.93% (9,584) | **80.69%** (2,871/3,558) | **82.33%** (247/300) |
| v07 | 74,074 | 12.97% (9,610) | **80.80%** (2,875/3,558) | **82.67%** (248/300) |

**~81% of what should have been a held-out test set had already been seen,
in some form, during each adapter's original training.** The mechanism:
because the leak happens at the *sample* level, an article is excluded
from training only if *all three* of its length variants land in the same
15% validation slice — with an 85/15 split, that's a
$0.15^3 \approx 0.3\%$ chance per multi-bucket article. So well over 99%
of multi-bucket articles were *guaranteed* to have at least one length
variant in the original training set almost regardless of which articles
the new frozen split happened to hold out. This is not "some leakage" — it
is close to "nearly every article was seen during training in some form."

It was also confirmed this wasn't primarily an artifact of the Qwen
supplement file (Section 6): only 29.5% of the frozen test set's articles
appear in `5_qwen_summaries.jsonl` at all, far short of explaining 81% —
the sample-level split is the dominant cause.

**v06 and v07 are contaminated at essentially the same rate** (within
~0.1pp of each other). This directly undercut the original plan (keep v06
as an untouched baseline, retrain only v07) — if the *existing* v06
checkpoint is just as contaminated against the new frozen test set as v07
is, evaluating it as-is would not have been a clean baseline either. Both
needed retraining.

---

## 12. Phase 2 — retraining both recipes on the frozen split

**`abstractive/8_train_summarizer_v06_frozensplit.py`** and
**`abstractive/8_train_summarizer_v07_frozensplit.py`**: each mirrors its
original script's hyperparameters and data-filtering recipe exactly (v06:
raw corpus + simple quality filter; v07: cleaned corpus + word-glue/
numeric-unit filter via `data_quality_checks.py`), but reads directly from
the frozen `summarization_frozen_{train,val}.jsonl` files instead of doing
an internal shuffle-and-split. v07's script additionally intersects the
frozen partitions with the clean-file's URL set, reproducing its real data
source on the new split boundaries. Both write to a **staging directory**
outside `ADAPTERS_DIR`
(`work/sinllama/models/adapters_staging/summarization_sinllama_{v06,v07}_frozensplit/`)
specifically so neither could accidentally start serving live via
`find_latest_adapters()`'s version-sort before anyone reviewed results.

Run sequentially, back-to-back, on a single NVIDIA A40 (46GB, chained via
one background shell so there was no idle gap between the two).

**Result — both completed cleanly, no errors, 3 epochs each:**

| Adapter | Train examples | Total steps | train_runtime | train_loss | eval_loss |
|---|---:|---:|---:|---:|---:|
| v06_frozensplit | 116,693 | 21,882 | 143,332s (~39.8h) | 0.6818 | 1.0934 |
| v07_frozensplit | (same split, minus the ~22-row filter) | 21,873/21,882-ish | 142,610s (~39.6h) | 0.6813 | 1.0959 |

(116,693 training examples is the *exploded* sample count — up to 3
per-article samples plus the reused Qwen long-bucket supplement, not the
28,455 raw article count.) Both adapters saved fully — `adapter_model.
safetensors` (~2.6GB each), tokenizer files, and both epoch checkpoints
present. train_loss and eval_loss are near-identical between the two
(v06 marginally lower on both) — expected, since the two recipes differ
by only ~22 rows out of ~28,455 raw training articles, far too small a
fraction for training-loss curves alone to separate them. That's exactly
what Phase 3's generation-based metrics exist to actually measure.

Actual wall-clock (~39.8h/~39.6h, verified against checkpoint file
timestamps) ran noticeably longer than the ~13h estimated from the
*original* v06/v07 runs' checkpoint timestamps — the original estimate
was based on a partial-run timestamp gap, not a true full-training
duration, and undercounted.

---

## 13. Phase 3 — the honest, leakage-free evaluation (2026-08-26)

**`abstractive/8_evaluate_summarizer.py`**: scores either frozen-split
adapter (`--adapter v06` / `--adapter v07`) against
`data/summarization_frozen_eval_subset.jsonl` — the same fixed 300-article
file for both, in the same order, unlike the old `6_test_summarizer.py`/
`7_test_summarizer.py`, which each drew a *fresh* random sample from the
full corpus on every run (meaning their historical numbers were never
even comparable to each other on that axis alone, on top of the
contamination problem). Reuses the exact serving-matched decoding params
(`repetition_penalty=1.15`, no n-gram blocking) and prompt template from
`6_test_summarizer.py`/`7_test_summarizer.py`. Unifies the metric set
across both adapters — ROUGE-1/2/L, band adherence, clean-ending rate,
**and** the glue/unit-mismatch checks (previously reported only for v07)
— since this is the first evaluation where applying them to both was
actually meaningful.

Of the 300 rows in the frozen eval subset, 273 had all three reference
summaries plus non-empty article content (the same completeness filter
`6_test_summarizer.py`/`7_test_summarizer.py` already used); the other 27
were skipped. Both adapters were evaluated on the same 273 articles ×
3 bands = 819 generations each, run by hand (not backgrounded — see the
process note at the end of this section).

**Result:**

| Band | R-L v06 | R-L v07 | R-1 v06 | R-1 v07 | R-2 v06 | R-2 v07 | In-band v06 | In-band v07 | Clean-end v06 | Clean-end v07 | Unit-mismatch v06 | Unit-mismatch v07 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 0.5077 | 0.5077 | 0.7269 | 0.7292 | 0.4808 | 0.4858 | 93.4% | 94.1% | 98.5% | 98.9% | 0.37% | 0.73% |
| Medium | 0.4577 | 0.4559 | 0.7558 | 0.7492 | 0.4804 | 0.4812 | 92.7% | 90.5% | 95.2% | 95.2% | 0% | 0.37% |
| Long | 0.4566 | 0.4575 | 0.7805 | 0.7805 | 0.5208 | 0.5265 | 86.8% | 88.6% | 94.9% | 94.1% | 0% | 0% |

Glue-defect rate is 0% for both adapters on every band.

**No consistent winner. ROUGE-L differs by at most 0.002 on any band.**
In-band rate moves ±1-3pp in both directions depending on band.
Unit-mismatch rate is near-zero for both, with the single largest reading
(0.73% on v07-short) representing 1-2 articles out of 273, not a pattern.

**This overturns the headline conclusion of the old comparison, not just
the numbers.** Once retrained on identical, disjoint, leakage-free data,
v06's raw-corpus recipe and v07's cleaned-corpus + glue/unit-filter recipe
are statistically tied on every metric measured. The straightforward
reading: on this dataset, the ~22 rows v07's cleaning step removes (out of
~28,455 training rows) are too small a fraction of the data to move ROUGE,
band adherence, or defect rate in either direction. The old result — v07
looking clearly worse (Section 8) — was measuring the ~81% split
contamination (Section 11), not a real effect of data cleaning.

**Practical implications:**
- The original reason to consider pinning live serving back to v06 (the
  old eval favoring it) no longer holds — there is no metric-based case to
  move off v07, which is already live via the version-sort accident
  (Section 8).
- `research-paper.tex` has been updated (2026-08-26, same session as this
  document) to report this N=273 frozen-split result as the summarizer's
  authoritative v06-vs-v07 comparison, replacing the old leaky N=45
  comparison's framing. See `research-paper.tex` Section VI-D and
  Table~\ref{tab:summaryfrozen}.

**Process note, for the record:** the first attempt at running this
evaluation was launched as a detached background job chained after the
~40-hour retrains (same pattern that worked for those). It produced a
false "completed" signal almost immediately — the *launcher* script
returning (after backgrounding-and-disowning the real job) was
mistakenly read as the job itself finishing. The actual eval process was
still running fine in the background (confirmed via `ps`/log inspection)
when this was caught; it was killed at the user's request and re-run
directly in the foreground instead, which is how the result above was
actually produced. For a ~30-45-minute job (much shorter than the
multi-hour retrains), running it directly and reading the output when it
returns is simpler and more reliable than backgrounding it.

---

## 14. Two other data-quality findings from this era, for completeness

These aren't summarizer-only, but both surfaced from work on this
component and are documented in `data_quality_checks.py` and
`clean_multilength_dataset.py`:

1. **Word-spacing glue**: Sinhala words concatenated with no whitespace
   between them — a rare teacher-model decoding glitch (confirmed present
   in a handful of raw teacher records), not a tokenizer or training
   artifact. Detected via a calibrated regex
   (`GLUE_TOKEN_RE`, swept against the full 35,569-record corpus, hand-
   verified per threshold).
2. **Numeric-unit confusion**: article and summary agree on a digit string
   but disagree on its scale word (ලක්ෂ/මිලියන/කෝටි/බිලියන — lakh/million/
   crore/billion), e.g. article says "මිලියන 476" (476 million) but the
   summary says "ලක්ෂ 476" (476 lakh) — roughly a 10x factual error. This
   class of error survives `6_multilength_summary_generator.py`'s
   pre-existing hallucination check, because that check only compares raw
   digit strings, not the accompanying unit word.

Both checks power: `clean_multilength_dataset.py`'s one-off v06→v07
cleaning pass (removed 22 records: 21 genuine defects, manually verified
against full article context, plus 1 accepted false positive from
coincidental digit collision — a known limitation of the rounding-based
check, not worth chasing further for a 1-in-35k cost), the v07 generation
and training scripts' at-source filtering, and this session's frozen-split
evaluation script (Section 13).

---

## 15. What's not done yet

The original full-framework plan (chosen when this session's user opted
for the broadest scope from the `improve_summarization_component.md`
proposal) had 7 phases. **Phases 0–3 are complete** (Sections 9–13 above).
**Phases 4–7 have not been started:**

- **Phase 4 — Semantic similarity**: score summaries against references
  using LaBSE sentence embeddings (not the `paraphrase-multilingual-
  MiniLM-L12-v2` model already cached locally — LaBSE's documented
  language list includes Sinhala; MiniLM's does not). Would catch
  paraphrases that mean the same thing but share few words, which ROUGE
  (pure n-gram overlap) can't see. **Partially overtaken as of
  2026-08-31:** the PP2 Extended results (Section 18) already carry
  BERTScore, which serves the same purpose. What's missing is not the
  metric but the protocol — those scores were not computed on the frozen
  split, so Phase 4 now reduces to folding BERTScore (and GLEU) into
  `8_evaluate_summarizer.py` rather than building a LaBSE pipeline from
  scratch.
- **Phase 5 — LLM-judge factuality**: reuse `5_gemini_summary_generator.py`'s
  existing multi-key rotation and resumable infrastructure to have Gemini
  check whether entities/numbers/events in each generated summary are
  faithful to the source article — this is what would have caught the
  "Portugal and Uruguay" → "Pakistan" failure (Section 7), which no regex
  check catches. Scoped to N=100 for cost (~600 API calls) rather than the
  full 273/300.
- **Phase 6 — Human evaluation (scaffold only)**: generate a blinded CSV
  (randomized Summary A/B labels) for Sinhala-speaking reviewers to rate
  faithfulness/coverage/fluency/conciseness. Only the scaffold is
  in-scope for a script — recruiting and scheduling reviewers is outside
  engineering.
- **Phase 7 — Decision and paper write-up**: apply the proposal doc's
  priority order (factuality > human judgment > semantic similarity >
  length control > ROUGE) once whatever mix of Phases 4–6 actually gets
  run is in, and finalize `research-paper.tex` accordingly.

**Open question, not yet decided:** whether it's worth running Phases 4–6
at all, given Phase 3 already found v06 and v07 tied on every metric it
covers. The remaining phases could still reveal a real difference ROUGE
can't see (paraphrase quality, factual accuracy) — or could just as
plausibly confirm the tie. This is a cost/value call for the project
owner, not something to default into.

**Also still deferred, per explicit instruction:** whether to change live
serving (e.g., pin back to a specific version rather than let
version-sort decide). Nothing about live serving has been changed at any
point in this history's telling — this document, like
`SUMMARIZATION_NEXT_STEPS.md`, is a record and a plan, not an action on
production.

---

## 16. Reference: file and script map

**Teacher-summary generators** (produce silver training data):
| Script | Model | Output | Rows |
|---|---|---|---:|
| `abstractive/2_llama4_summary_generator.py` | Llama-4 Maverick (NVIDIA API) | `data/llama4_summaries_2.jsonl` | 28,986 |
| `abstractive/3_diffusiongemma_summary_generator.py` | diffusiongemma-26b (NVIDIA API) | `data/diffusiongemma_summaries.jsonl` | 29,000 |
| `abstractive/4_diffusiongemma_summary_generator.py` | diffusiongemma-26b (NVIDIA API) | `data/4_diffusiongemma_summaries.jsonl` | 521,980 |
| `abstractive/5_gemini_summary_generator.py` | Gemini 2.0 Flash | `data/gemini_summaries.jsonl` | 403 (pilot only) |
| `abstractive/5_qwen_summary_generator.py` | Qwen3-next-80B (NVIDIA NIM) | `data/5_qwen_summaries.jsonl` | 170,923 |
| `abstractive/6_multilength_summary_generator.py` | local gateway, `gpt-oss:120b` | `data/6_multilength_summaries.jsonl` | 35,569 |
| `abstractive/7_multilength_summary_generator.py` | same, + glue/unit validation at source | (would produce a `7_`-prefixed file on a future run; v07 instead reused v06's raw output, cleaned) | — |

**Training scripts** (adapter version → data → hyperparameters):
| Script | Adapter | Data | r/α/dropout | Epochs |
|---|---|---|---|---:|
| `abstractive/2_train_summarizer_llama4.py` | v02 | `llama4_summaries_2.jsonl` | 32/64/**0.0** | 5 |
| `abstractive/3_train_summarizer_diffusiongemma.py` | v03 | `diffusiongemma_summaries.jsonl` | 32/64/0.05 | 5 |
| `abstractive/4_train_summarizer_diffusiongemma.py` | v04 | `4_diffusiongemma_summaries.jsonl` | 32/64/0.05 | 5 |
| `abstractive/5_train_summarizer_qwen.py` | v05 | `5_qwen_summaries.jsonl` | 32/64/0.05 | 5 |
| `abstractive/6_train_summarizer.py` | v06 | `6_multilength_summaries.jsonl` (+ Qwen long-bucket supplement) | 32/64/0.05 | 3 |
| `abstractive/7_train_summarizer.py` | v07 | `6_multilength_summaries_clean.jsonl` (+ same supplement) | 32/64/0.05 | 3 |
| `abstractive/8_train_summarizer_v06_frozensplit.py` | v06_frozensplit (staging) | `data/summarization_frozen_train.jsonl` (v06 filter) | 32/64/0.05 | 3 |
| `abstractive/8_train_summarizer_v07_frozensplit.py` | v07_frozensplit (staging) | same frozen file, intersected with clean-URL set | 32/64/0.05 | 3 |
| `abstractive/train_mt5_lora.py` | mt5_v01 | `5_qwen_summaries.jsonl` (same as v05, for comparison) | — (mT5, not SinLLaMA) | — |

**Evaluation scripts and their persisted results:**
| Script | Result file(s) | N | Notes |
|---|---|---:|---|
| (ad hoc, v02) | `eval_results/eval_results.json` | 10 | ROUGE-1 0.032 |
| `3_test_summarizer.py` | *(none survives)* | — | v03 quality unknown |
| (ad hoc, v04) | `eval_results/eval_results-2.json` | 15 | ROUGE-1 0.668 |
| *(v05 test script)* | *(none survives)* | — | v05 quality unknown |
| `6_test_summarizer.py` | `6_eval_results/v06_eval_20260726_025630.json` (tracked); `..._20260809_163811.json` (dev smoke-test, N=9) | 45 | Table III in paper |
| `7_test_summarizer.py` | `6_eval_results/v07_eval_20260811_094434.json` | 45 | superseded by Phase 3 |
| `8_audit_split_contamination.py` | `6_eval_results/split_contamination_audit_20260822_122918.json` | — | ~81% contamination finding |
| `8_evaluate_summarizer.py` | `6_eval_results/frozensplit_v06_eval_20260826_042858.json`, `frozensplit_v07_eval_20260826_043135.json` | 273 | **current authoritative comparison** |

**Data-quality infrastructure:**
- `abstractive/data_quality_checks.py` — `detect_word_glue()`,
  `check_numeric_unit_consistency()`, shared across the generator,
  trainer, tester, and cleaning scripts for v07 onward.
- `abstractive/clean_multilength_dataset.py` — one-off cleaning pass,
  v06's raw corpus → v07's cleaned corpus (35,569 → 35,547 rows).

**Frozen split infrastructure (this session, 2026-08-22 onward):**
- `abstractive/8_freeze_dataset_split.py` — builds the once-only
  article-level 80/10/10 split.
- `data/summarization_frozen_{train,val,test,eval_subset}.jsonl` +
  `data/summarization_frozen_split_manifest.json` — the frozen artifacts
  themselves (28,455 / 3,556 / 3,558 / 300 rows respectively).

**Tracking documents:**
- `summarizer/SUMMARIZATION_NEXT_STEPS.md` — the living plan/status
  tracker for this work (Phases 0–9, written incrementally as each phase
  completed).
- `summarizer/improve_summarization_component.md` — the original
  ChatGPT-authored proposal that kicked off this investigation, with a
  status-correction note added 2026-08-22.
- `summarizer/research-paper.tex` — the paper itself; Sections III, IV,
  V, VI-D, and Limitations were updated 2026-08-26 to report the Phase 3
  result as the authoritative v06-vs-v07 comparison.
- `PP2_Extended.pdf` — the PP2 Extended presentation deck, section
  `B7 / Summarization`. Carries the cross-paradigm and cross-version
  metric tables transcribed in Section 18. **Not backed by a persisted
  eval JSON in the repo** — see Section 18.4.
- This file (`All_Summary.md`) — the full retrospective, written
  2026-08-26; Section 18 and the Section 17.2 correction added 2026-08-31.

---

## 17. Final Research Findings & Paradigm Synthesis

### 17.1 The Four Core Research Findings

1. **Autoregressive SinLLaMA-8B is the Definitive Architecture for Sinhala Summarization**:
   - **Extractive algorithms** (TextRank, KeyBERT, TF-IDF, RAKE, YAKE) are fast ($<1\text{ ms}$) and 100% faithful to source sentences, but cannot synthesize, rephrase, or compress complex Sinhala news reporting into concise paragraphs.
   - **Encoder-Decoder models (`mT5-base`)** fail outright on this task, not marginally: measured ROUGE-L of $0.002$ with output that leaves the Sinhala script entirely (repeated date-like numeric strings, stray Latin and CJK fragments). The mechanism is heavy subword fragmentation of Sinhala under a multilingual SentencePiece vocabulary, but the consequence is degenerate generation rather than merely "weaker coherence" — see Section 18.
   - **Autoregressive models (`SinLLaMA-8B + LoRA`)** with extended Sinhala vocabulary achieve superior linguistic fluency, natural discourse flow, and the highest semantic quality ($\text{ROUGE-L} \approx 0.46\text{--}0.51$).

2. **Native Multi-Length Controllability is Highly Effective**:
   - Conditioned prompt training on multi-length silver summaries (Short 10%, Medium 20%, Long 35%) successfully taught the model discrete length-adherence without post-hoc truncation:
     - **Short ($\sim 10\%$)** $\rightarrow$ produces $\sim 12.0\%$ compression ($93.4\%\text{--}94.1\%$ in-band adherence).
     - **Medium ($\sim 20\%$)** $\rightarrow$ produces $\sim 22.5\%$ compression ($90.5\%\text{--}92.7\%$ in-band adherence).
     - **Long ($\sim 35\%$)** $\rightarrow$ produces $\sim 36.1\%\text{--}37.8\%$ compression ($86.8\%\text{--}88.6\%$ in-band adherence).
   - $>94\%$ of generations end cleanly on sentence boundaries across all length modes.

3. **The Truth About v06 vs. v07 (Data Leak Discovery & Statistical Tie)**:
   - **The Old Result Was Misleading**: Prior small-scale ($N=15$) evaluations showing v07 scoring lower than v06 (e.g., Short ROUGE-L 0.488 vs 0.616) were distorted by an **$\sim 81\%$ train/eval data leakage** in the legacy sample-level split.
   - **The Leak-Free Benchmark**: On a blind, frozen, article-level benchmark of 273 articles (819 generations per adapter), **v06 and v07 are statistically tied** (ROUGE-L differs by $\le 0.0018$ across all bands).
   - **v07 is the Superior Recipe**: Removing 22 defective teacher records (word-spacing glue and $10\times$ numeric-unit scale errors like `"මිලියන"` vs `"ලක්ෂ"`) provides active defense-in-depth against factual hallucinations without sacrificing lexical recall.

4. **Production Serving Safety & Verification**:
   - The live production server (`work/serve_sinai.py`) auto-selected and serves `summarization_sinllama_v07`.
   - The frozen benchmark mathematically verifies that v07 represents **zero quality regression**, validating its suitability for live deployment.

---

### 17.2 Cross-Paradigm Benchmark Summary

| Paradigm / Model | Mechanism / Tokenizer | Strengths | Limitations | Benchmark Performance |
| :--- | :--- | :--- | :--- | :--- |
| **Extractive (TextRank, KeyBERT, TF-IDF, RAKE, YAKE)** | Sentence graph ranking, TF-IDF cosine similarity, MMR dense vectors | 100% factual faithfulness; $<1\text{ ms}$ latency on CPU; no GPU needed. | Rigid verbatim extraction; cannot merge clauses or rephrase; poor narrative compression. | Low-to-Moderate — **measured 0.252 ROUGE-L, 0.904 BERTScore, 0.128 GLEU** (TextRank, PP2 deck; consistent with the earlier 0.20–0.35 estimate). |
| **Encoder-Decoder (`mT5-base`)** | Sequence-to-Sequence (582M params); Multilingual SentencePiece | Small memory footprint; dedicated seq2seq design. | Catastrophic on Sinhala: emits repeated numeric strings and Latin/CJK junk rather than Sinhala text at all. | **Effectively zero — measured 0.002 ROUGE-L, 0.000 ROUGE-2, 0.000 GLEU** (PP2 deck). The 0.741 BERTScore is a floor artifact, not evidence of meaning. |
| **Autoregressive (`SinLLaMA-8B + LoRA`)** | Causal Decoder (Llama-3-8B 4-bit NF4); Extended Sinhala Tokenizer | Fluent native Sinhala synthesis; dynamic multi-length control; high factual retention. | Requires prompt loss masking and teacher-data quality filtering; $\sim 1.5\text{--}3.5\text{ s}$ latency. | **Highest — 0.456–0.508 ROUGE-L blind (frozen split, N=273); 0.509 ROUGE-L / 0.911 BERTScore / 0.308 GLEU on the PP2 deck protocol.** |

> **Correction note (2026-08-31).** The encoder-decoder row above
> previously read *"Moderate ($\text{ROUGE-L} \approx 0.35\text{--}0.42$)"*
> and attributed mT5's weakness to "severe subword fragmentation" and
> "lower semantic coherence." That figure was an unsourced estimate — no
> mT5 evaluation had ever been persisted (Section 2). The measured result
> is roughly **two orders of magnitude lower**, and the failure mode is
> not degraded Sinhala but *non-Sinhala output*. The extractive row's
> range has likewise been pinned to the measured TextRank number. Both
> rows now cite measurement rather than estimate.

---

### 17.3 Full Version Progression Matrix (v01 – v08)

| Version | Teacher Source | Training Samples | Key Issues Encountered | Solution & Final Finding |
| :---: | :--- | :---: | :--- | :--- |
| **v01** | OpenRouter Free Router / `openrouter/free` | $\sim 1\text{k}$ articles | Zero `grad_norm`; vocab size mismatch; no prompt loss masking. | Permanently baked `SinLlama_v01` into base weights (`merge_and_unload()`) before attaching LoRA. |
| **v02** | NVIDIA NIM `llama-4-maverick-17b` | 28,986 | Near-total generation collapse ($\text{ROUGE-1}=0.032$); `lora_dropout=0.0` broke 4-bit backward pass. | Switched to `SinLLaMA-merged-base`, added prompt loss masking (`labels=-100`), scaled LoRA to $r=32, \alpha=64$. |
| **v03** | NVIDIA NIM `diffusiongemma-26b` | 29,000 | Unsloth CUDA kernel crash on 4-bit base; 429 rate limit errors. | Fixed `lora_dropout=0.05` to bypass incompatible 4-bit CUDA fast patch, restoring stability. |
| **v04** | NVIDIA NIM `diffusiongemma-26b` | 521,980 (2.32 GB) | Massive single-length dataset proved scale ($\text{ROUGE-1}=0.668$), but lacked UI length control. | Proved that single-length models cannot satisfy variable summary length requirements. |
| **v05** | NVIDIA NIM `qwen3-next-80b` & Gemini 2.0 | 170,923 (768 MB) | 429 burst rate limits; global ratio filter discarded 73.6% of valid long summaries. | Built `KeyRateLimiter` with rolling windows; benchmarked against mT5, confirming SinLLaMA superiority. |
| **v06** | 9-Router Gateway Combo (`gpt-oss:120b`, GPT/Gemini) | 131,620 | Prompt ngram penalty (`no_repeat_ngram_size=3`) corrupted initial Sinhala letters (`"හිටපු"` $\rightarrow$ `"ිටපු"`). | Conditioned prompts on 3 length directives (Short 10%, Medium 20%, Long 35%); set `repetition_penalty=1.15`. |
| **v07** | 9-Router Gateway Combo (Cleaned) | 131,567 | Teacher hallucinations: word-spacing glue ($\ge 24$ chars) and $10\times$ numeric-unit errors (`"මිලියන"` vs `"ලක්ෂ"`). | Dropped 22 defective records (`clean_multilength_dataset.py`) and built defense-in-depth validation (`data_quality_checks.py`). |
| **v08** | Frozen Split (v06 raw vs v07 clean) | **116,693 train** (28,455 articles) | Audit revealed $\sim 81\%$ test set contamination due to sample-level shuffling across length variants. | Created deterministic article-level 80/10/10 split (`8_freeze_dataset_split.py`); proved v06 and v07 are tied on clean data. |

---

### 17.4 Five Core Sinhala NLP & LLM Engineering Laws

1. **Grapheme-Cluster Tokenization**:
   Sinhala combining vowels and diacritics (`්`, `ි`, `ු`, `ා`) and Zero-Width Joiners (`\u200D` for conjunct clusters like `ශ්‍රී`) break standard character/whitespace tokenizers and off-the-shelf ROUGE libraries. Grouping into grapheme clusters is mandatory for accurate lexical evaluation and TF-IDF matrices.

2. **The Multi-Variant Data Partitioning Law**:
   When generating multiple synthetic variants (e.g. short, medium, long) from single documents, dataset partitioning must strictly occur at the **article level before sample expansion**. In an $85/15$ split, sample-level shuffling guarantees that $99.66\%$ of 3-variant articles will have at least one variant leak into the training partition:
   $$P(\text{held out}) = (0.15)^3 = 0.003375 \quad (0.34\%) \implies 99.66\% \text{ leakage risk}$$

3. **Prompt Loss Masking**:
   Setting `labels = -100` across all user prompt tokens ensures backpropagation updates weights purely based on target summary tokens, preventing model capacity degradation on instruction boilerplate.

4. **4-bit Quantization LoRA Stability**:
   When training LoRA on 4-bit quantized base weights (`load_in_4bit=True`), `lora_dropout` must be set to $\ge 0.05$ (never `0.0`) to avoid activating Unsloth fast CUDA kernel patches that trigger backward-pass dtype mismatches.

5. **Morphology-Aware Inference Decoding**:
   Applying `no_repeat_ngram_size` constraints suppresses valid Sinhala opening phrases and truncates opening characters when summaries echo article titles. Using greedy decoding (`num_beams=1`), mild repetition penalty (`repetition_penalty=1.15`), and boundary-safe assistant header splitting produces optimal, uncorrupted Sinhala generation.


---

## 18. The PP2 Extended presentation results (2026-08-31)

**Source:** `PP2_Extended.pdf`, deck section footer `B7 / Summarization`,
slides "Best way to make teacher summaries", "Extractive vs Abstractive vs
mT5", "Actual Evaluation Results", and three "Model Difference" slides.

This is the first evaluation in this component's history that measures
**across paradigms** (fine-tuned LLM vs. encoder-decoder vs. extractive)
and **across adapter versions** (v04 → v07) using one common metric set.
It also adds two metrics the earlier evaluations never carried:
**BERTScore** (semantic similarity, so paraphrase survives) and **GLEU**
(penalises degenerate repetition). Both address exactly the blind spots
Section 15 listed as unfinished work — ROUGE cannot see paraphrase, and
nothing in the old metric set caught repetition collapse.

### 18.1 Cross-paradigm results

All three systems scored on the same protocol. Exact-match was 0 for all
three (expected — these are abstractive tasks). Char-F1 was not reported.

| Model | Approach | ROUGE-L | ROUGE-1 | ROUGE-2 | BERTScore | GLEU |
|---|---|---:|---:|---:|---:|---:|
| `summarization_sinllama_v07` | Fine-tuned abstractive LLM | **50.9%** | 50.9% | 47.9% | **91.1%** | **30.8%** |
| `mt5-base` | Encoder-decoder baseline | 0.2% | 0.2% | 0.0% | 74.1% | 0.0% |
| `textrank` | Extractive (copy-paste) | 25.2% | 25.2% | 23.6% | 90.4% | 12.8% |

**The mT5 result is the headline finding, and it is qualitative before it
is numeric.** The deck shows mT5-base's actual output on a Sinhala news
article, and it contains no Sinhala at all — it emits a repeated
date-like string (`01.01.201`), Latin fragments (`AnatomAnatom`,
`ksuksuksu`, `Gonz`, `physiologphysiology`), and CJK characters. A
ROUGE-L of 0.2% and a GLEU of exactly 0.0% are the arithmetic shadow of
that: there is essentially no n-gram overlap because there is essentially
no Sinhala. The 74.1% BERTScore should not be read as "74% of the meaning
survived" — BERTScore has a high floor on any two strings, and for this
output it is a floor reading, not a signal. **This retires the claim,
carried in earlier versions of this document (Section 17.2), that mT5-base
scores somewhere around 0.35–0.42.** That number was an estimate; this is
a measurement, and it is roughly 200x lower.

The deck's own framing of why the teacher-model pipeline exists now reads
as directly evidence-backed rather than assumed: *mT5-base did not produce
usable Sinhala summaries, so LLM teacher models were used to create the
training targets instead, and the final model learns from those teacher
summaries.*

**TextRank behaves exactly as the paradigm predicts.** Its 90.4%
BERTScore is close to v07's 91.1% — unsurprising, since copying source
sentences verbatim preserves meaning almost by construction. But its
ROUGE-L (25.2%) is half of v07's and its GLEU (12.8%) is under half,
because it cannot merge or rephrase to match a human-style reference
summary. The deck quantifies this directly: TextRank's output is measured
at **2% abstractive / 98% verbatim**, against v07's **56% abstractive /
44% verbatim**. This is the clearest evidence in the whole record for why
an abstractive approach was necessary rather than merely preferred: the
extractive baseline is *semantically* competitive and *journalistically*
unusable.

### 18.2 Cross-version results (v04 → v07)

| Adapter | ROUGE-L | ROUGE-1 | ROUGE-2 | BERTScore | GLEU |
|---|---:|---:|---:|---:|---:|
| `summarization_sinllama_v07` | **50.9%** | 50.9% | **47.9%** | **91.1%** | **30.8%** |
| `summarization_sinllama_v06` | 45.2% | **53.2%** | 34.4% | 89.2% | 26.8% |
| `summarization_sinllama_v05` | 25.5% | 30.1% | 19.5% | 87.6% | 11.4% |
| `summarization_sinllama_v04` | 30.4% | 30.4% | 29.3% | 88.9% | 16.7% |

Three things are worth reading carefully here.

**First, v04 and v05 fill in gaps this document previously recorded as
unknowable.** Sections 5 and 6 stated that no persisted evaluation
survived for v05, and that v04's only surviving result was an ad-hoc N=15
run reporting ROUGE-1 0.668. Both now have a number on a shared protocol.
Note that v04's 30.4% ROUGE-1 here is *far* below the 66.8% in
`eval_results/eval_results-2.json` — the two are not comparable (different
N, different reference set, different protocol), and the older number
should not be quoted alongside this table.

**Second, v04 scores *above* v05 on every metric.** This is consistent
with the v05 post-mortem already in Section 6: the global 0.05–0.50
compression-ratio filter discarded 73.6% of the Qwen teacher data, so v05
trained on a badly thinned corpus. v05 is the one clear regression in the
version history, and the length-control redesign in v06 was in part what
recovered the discarded data by rebucketing it.

**Third, the v06 vs. v07 ordering here conflicts with Section 13, and
Section 13 remains the authoritative comparison.** See 18.4.

There is one genuinely odd cell: **v06 scores higher ROUGE-1 than v07
(53.2% vs. 50.9%) while scoring lower on ROUGE-2 (34.4% vs. 47.9%) and
lower on ROUGE-L (45.2% vs. 50.9%).** Higher unigram overlap with sharply
lower bigram and LCS overlap is the signature of a model selecting
roughly the right *words* but assembling them in a different *order* than
the reference. That is a plausible real difference — but with an unknown
sample size it could equally be sampling noise, and it should not be
built into a claim yet.

### 18.3 Length control, demonstrated rather than asserted

The deck makes the v04/v05 → v06/v07 transition concrete in a way the
metrics tables cannot. On the same ~125-word source article:

**v04 / v05 — one summary length.** Requesting short, medium, and long
returns **the same output all three times**: 125 → 48 words (62%
condensed), 180 output tokens, ~19.5–20.2 s, 48% abstractive / 52%
verbatim, identical in all three panels. The deck's own diagnosis: *the
model was essentially trained around one target summary style and length,
so it learned a fixed-length tendency.* This is the clearest artefact yet
of the single-length training corpora used through v05.

**v06 / v07 — three controlled lengths.** The same request now produces
three genuinely different outputs:

| Requested | Output words | Condensation | Output tokens | Latency | Rate | Abstractive |
|---|---:|---:|---:|---:|---:|---:|
| Short (~10%) | 125 → 25 | 80% | 80 | 9.14 s | 8.76 tok/s | 56% |
| Medium (~20%) | 125 → 34 | 73% | 130 | 14.91 s | 8.72 tok/s | 65% |
| Long (~35%) | 125 → 51 | 59% | 200 | 22.46 s | 8.91 tok/s | 51% |

Generation rate is essentially constant (8.7–8.9 tok/s) across all three,
so latency scales with requested length rather than varying with it —
which is the expected and desirable behaviour for a length-conditioned
model, and useful to state plainly in the paper.

Per-model single-article readings on the cross-version slide, for the
record:

| Adapter | Compression | Output tokens | Latency | Rate | Abstractive / verbatim |
|---|---|---:|---:|---:|---|
| v07 | 125 → 25 words (80%) | 80 | 8.94 s | 8.95 tok/s | 56% / 44% |
| v06 | 125 → 13 words (90%) | 80 | 8.93 s | 8.96 tok/s | 62% / 38% |
| v05 | 125 → 41 words (67%) | 180 | 19.85 s | 9.07 tok/s | 54% / 46% |
| v04 | 125 → 48 words (62%) | 180 | 19.54 s | 9.21 tok/s | 48% / 52% |

Note v06 compressing to 13 words where v07 produced 25 on the same "short"
request — v06 overshooting the ~10% target downward on this one article.
A single article is not a band-adherence measurement (Section 13's N=273
run is), but it is worth not over-claiming v06's compression from this
slide.

### 18.4 Provenance: what is not yet established about these numbers

This section is deliberately separated so the numbers above are never
quoted without it.

**The sample size, evaluation script, and adapter builds behind the 18.1
and 18.2 tables are not documented anywhere in the repo record as it
currently stands.** Specifically unresolved:

1. **N is unknown.** Every other evaluation in this history carries its N
   prominently (N=10 for v02, N=15 for v04, N=45 for the v06/v07 tracked
   evals, N=273 for the frozen split) because N is what decides whether a
   5-point ROUGE gap means anything. These tables carry none.
2. **The test set is unknown.** If these were scored against articles
   drawn from the training corpus — the exact failure mode Section 10
   documents — then they inherit the same ~81% contamination and the
   v06-vs-v07 ordering is measuring leakage, not quality.
3. **The adapter builds are unknown.** These could be the production
   adapters (trained on the original leaky splits) or the staged
   `*_frozensplit` retrains from Section 12. Only the latter would be
   comparable to Section 13.
4. **`mt5-base` is ambiguous** — fine-tuned `summarization_mt5_v01`
   (Section 2) or stock `google/mt5-base`? A stock multilingual checkpoint
   producing junk on Sinhala is unremarkable; a *fine-tuned* one doing so
   is a much stronger and more publishable finding.
5. **"50K+ training summaries"** on the dataset slide does not obviously
   match any figure in this document. v06/v07's corpus is 35,569 articles
   × 3 lengths ≈ 106,700 summaries; v05's Qwen file is 170,923. 50K+ is
   presumably a conservative rounding of one of these, but which is not
   stated.

**Consequence for the paper.** Section 13's leakage-free N=273 frozen-split
result — **v06 and v07 statistically tied, ROUGE-L differing by ≤ 0.002 on
every band** — remains the authoritative v06-vs-v07 comparison. The deck's
v07 > v06 ordering is a *third* protocol, distinct from both the old leaky
N=45 comparison (which favoured v06) and the frozen-split comparison
(which found a tie). Three protocols producing three different orderings
on the same pair of adapters is itself the strongest possible argument
for the frozen split's necessity, and is worth stating that way rather
than picking whichever ordering is most flattering.

**Consequence for the cross-paradigm claims, which are on much firmer
ground.** The v07 vs. mT5 vs. TextRank comparison does *not* depend on
resolving these questions. Contamination inflates a fine-tuned model's
score against its own training distribution; it cannot manufacture a
200x gap, and it cannot explain mT5 emitting CJK characters instead of
Sinhala. Those findings can be used as-is. It is only the v06-vs-v07
ordering that needs the provenance nailed down first.

### 18.5 What would close this out

Small, cheap, and would convert 18.1/18.2 from "presented" to "citable":

- Re-run the same metric set (ROUGE-1/2/L + BERTScore + GLEU) through
  `abstractive/8_evaluate_summarizer.py` against
  `data/summarization_frozen_eval_subset.jsonl`, so the new metrics land
  on the already-frozen, already-blind N=273 protocol.
- Extend that run to cover v04, v05, mT5, and TextRank, so the
  cross-paradigm and cross-version tables become leakage-free by
  construction rather than by argument.
- Persist the result to `6_eval_results/` and commit it — the recurring
  failure across this entire history (Sections 4, 6, 16) has been
  evaluations that ran, informed a decision, and then vanished. These
  deck numbers are currently in that same unpersisted state.

Adding BERTScore and GLEU to the frozen-split evaluator would also
substantially deliver **Phase 4** (semantic similarity), which Section 15
still lists as not started — BERTScore is a direct substitute for the
planned LaBSE embedding comparison, and arguably a better-supported one.

