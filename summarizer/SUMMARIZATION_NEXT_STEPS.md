# Summarizer: What Happened, and What to Do Next

Status: **Phase 0 (correct the record) and Phase 1 (frozen split +
contamination audit) are done. Phase 2's training scripts are written and
the retrain-both decision is made — training itself has not been run yet.**
Phases 3–7 (new eval script, semantic similarity, LLM-judge factuality,
human eval, paper rewrite) are still planned, not started. See §7 for the
contamination finding and §8 for the Phase 2 decision and exact commands.

---

## 1. What happened

You asked me to read `research-paper.tex` and the ChatGPT-authored proposal
`improve_summarization_component.md` and say what we should do next. Both
documents were written today and share the same starting assumption:

> "v07 is the cleaned follow-up dataset/model recipe... the project records
> do not contain an authoritative saved v07 training run/evaluation."

I checked this against the actual filesystem and git history. **It's wrong.**

- A v07 adapter is fully trained:
  `work/sinllama/models/adapters/summarization_sinllama_v07/`, both
  checkpoints present (`checkpoint-13980`, `checkpoint-20970`), written
  Aug 11.
- It has a real, git-committed evaluation:
  `summarizer/6_eval_results/v07_eval_20260811_094434.json`, committed in
  `74ab381` ("feat(summarizer): add v06 and v07 evaluation results", Aug 14).
- There's a directly comparable v06 evaluation from the same protocol:
  `summarizer/6_eval_results/v06_eval_20260726_025630.json`.
- `work/sinllama/COMPONENTS.md` (updated Aug 14) already correctly documents
  that v07 exists — so this wasn't hidden, just not propagated into the paper
  or into `CLAUDE.md` (which still only mentions "adapters v02–v06").

**More importantly: the live server is already serving v07 to real users,
today**, not because anyone chose it for quality but because
`work/serve_sinai.py`'s `find_latest_adapters()` scans `ADAPTERS_DIR` and
auto-picks whichever folder parses to the highest version number
(`serve_sinai.py:44-102`). v07 > v06, so v07 wins the sort, unconditionally.

And the one apples-to-apples comparison that already exists doesn't favor
v07. Same protocol, both N=45 (15 articles × 3 length bands):

| band   | ROUGE-L v06 | ROUGE-L v07 | in-band v06 | in-band v07 |
|--------|------------:|------------:|-------------:|-------------:|
| short  | 0.616       | 0.488       | 93.3%        | 100%         |
| medium | 0.579       | 0.518       | 100%         | 86.7%        |
| long   | 0.549       | 0.488       | 100%         | 93.3%        |

On this data, **v07 is currently worse on ROUGE-L across every band**, and
mixed (worse on medium/long, better on short) on in-band rate. So the thing
that's live in production right now is the version that looks weaker on the
only comparable measurement that exists.

**I have not changed anything about live serving.** That's a separate,
explicitly deferred decision — see §4.

---

## 2. The real underlying problem

Neither the v06 nor the v07 number above should be trusted as a final
verdict, and this is the part the proposal doc (`improve_summarization_component.md`)
gets right even though its premise about v07 not existing was wrong. Two
concrete problems, both verified directly against the code:

**A. The train/eval split leaks across length variants of the same article.**
`abstractive/7_train_summarizer.py` (and `6_train_summarizer.py` the same
way) builds a flat list of up to 3 samples per article (short/medium/long),
shuffles the *flat list*, then splits 85/15:

```
abstractive/7_train_summarizer.py:259   random.seed(SEED)
abstractive/7_train_summarizer.py:300   random.shuffle(samples)
abstractive/7_train_summarizer.py:75    TRAIN_SPLIT = 0.85
```

Because the split happens after the explosion into samples, the short
summary of article X can end up in train while the medium summary of the
same article X ends up in eval. That's leakage — the model can see part of
an article's content during training and get evaluated on a different
length of the same article.

**B. There's no frozen, persisted, blind test set.** `7_test_summarizer.py`
picks N=15 random articles straight out of the same corpus file
(`6_multilength_summaries_clean.jsonl`) every time it runs, with its own
random seed. It's not a held-out file that was set aside once and never
touched again — it's a fresh resample from the same pool the model may have
trained on. Both the paper and the proposal doc already flag this as a
limitation, and it's confirmed accurate.

**One thing that makes fixing this easier than the proposal doc assumes:**
I checked whether the corpus needs to be grouped by article before splitting
(as the proposal doc instructs), and it doesn't — it's already there:

```
$ wc -l data/6_multilength_summaries_clean.jsonl
35547
# unique url-ish keys in the file: 35547
```

Row count equals unique-URL count. Each row already has `summary_short`,
`summary_medium`, and `summary_long` as three columns on *one* record. The
leak isn't in the data shape — it's entirely inside the training script's
explode-then-shuffle logic. An article-level split just means splitting
*rows* (not grouping first), which is simpler than the proposal doc's
described procedure.

**C. v06 and v07 don't even train on the same dataset**, which the frozen
split had to account for (this came up when you asked whether they share a
source file — they don't). v06 trains on `6_multilength_summaries.jsonl`
(raw, 35,569 rows); v07 trains on `6_multilength_summaries_clean.jsonl`
(35,547 rows — 22 rows dropped for word-glue/number-unit/unrelated-content
defects). Verified the cleaning is pure row removal, not content edits: all
35,547 shared rows are byte-identical between the two files. The frozen
split (§Phase 1) is built from the **raw** 35,569-row file — the full
article universe both recipes ultimately draw from — precisely so that
retraining both recipes on it doesn't quietly collapse the variable being
tested (see Phase 2/§8).

---

## 3. Recommended plan (full framework, as you chose)

When asked how far to take this, you chose the full scope from the proposal
doc: fix the split and get an honest rerun, **plus** semantic similarity,
**plus** LLM-judge factuality and a human-eval scaffold. Below is the
sequenced version of that, grounded in what's actually in this repo — not
just a restatement of the proposal doc's abstract steps.

### Phase 0 — Correct the record (cheap, no compute) — DONE
- Fix `research-paper.tex`: the two "no tracked v07 model result exists"
  claims (~lines 248–249, ~676–678) are wrong — replace with an accurate
  statement that v07 exists and has a prior eval, but that eval used the
  same leaky, small-N protocol as v06's, so it isn't yet a valid basis for
  choosing between them. Also reconcile Fig. 1 (the architecture diagram
  already labels the deployed node "Summarizer v07") with the prose, which
  currently says v06 is "the latest model with a tracked evaluation
  artifact" — the figure happens to be the accurate one.
- Add a short status note to `improve_summarization_component.md` correcting
  its premise (its recommendations remain valid; only the "v07 doesn't
  exist yet" framing needs fixing).
- Update `CLAUDE.md`'s Summarizer section: "v02–v06" → "v02–v07", note v07's
  status honestly (cleaned-data recipe, existing eval currently looks worse,
  currently live-serving via version-sort not quality, frozen-split
  re-evaluation planned).

### Phase 1 — Article-level frozen split — DONE
- `abstractive/8_freeze_dataset_split.py`: reads the **raw**
  `6_multilength_summaries.jsonl` (35,569 rows, already 1-per-article; not
  the cleaned file — see §2C above for why), asserts URL uniqueness,
  deterministic split 80/10/10 by row (seed 42), writes
  `data/summarization_frozen_{train,val,test}.jsonl` plus a fixed
  `summarization_frozen_eval_subset.jsonl` (N=300, drawn once) so repeated
  evaluation runs don't keep resampling. Writes a manifest (input file
  hash, seed, ratios, per-split counts/URL-hash) for reproducibility.
  **Run** — output: 28,455 train / 3,556 val / 3,558 test / 300
  eval-subset rows, `data/summarization_frozen_split_manifest.json`. (This
  was rebuilt once already: the first version was accidentally built from
  the *cleaned* 35,547-row file, which would have silently excluded the 22
  raw-only articles from v06's retrain — see §2C.)
- `abstractive/8_audit_split_contamination.py`: replays the *existing*
  v06/v07 training scripts' exact sample-building + shuffle + split logic
  (CPU-only, no model load needed) to measure precisely how much of each
  existing adapter's original training data overlaps the new frozen test
  set. **Run** — see §4 below for the result, which is more severe than
  expected and changes the Phase 2 recommendation.

### Phase 2 — Retraining — scripts written, decision made, not yet run
See §5 for the full reasoning and exact commands. Short version: both v06
and v07 need retraining on the frozen split (not just v07) because the
contamination audit found the *existing* v06 checkpoint is just as
contaminated against the new frozen test set as v07 is (§7) — so evaluating
the existing v06 checkpoint as-is would not have been a clean baseline
either.
- `abstractive/8_train_summarizer_v06_frozensplit.py` — v06's recipe (raw
  corpus, simple quality filter), reading the frozen train/val files
  directly instead of re-splitting internally. Same hyperparameters as
  `6_train_summarizer.py`.
- `abstractive/8_train_summarizer_v07_frozensplit.py` — v07's recipe
  (cleaned corpus + word-glue/number-unit filter), same approach. Since the
  frozen split is built from the raw file, this script intersects the
  frozen train/val partitions with the clean-file's url set before running
  `build_samples()`, reproducing v07's real data source on the new split
  boundaries. Same hyperparameters as `7_train_summarizer.py`.
- Both write to a **staging directory outside `ADAPTERS_DIR`**
  (`work/sinllama/models/adapters_staging/summarization_sinllama_{v06,v07}_frozensplit/`)
  so neither can accidentally start serving live via a version-sort tie
  before anyone has looked at results.
- Keep the existing v06/v07 checkpoints' original eval numbers too, labeled
  reference-only with their ~81% contamination caveat — for continuity with
  the already-committed evals, not as decision inputs.

### Phase 3 — Evaluation script consolidation
- New `abstractive/8_evaluate_summarizer.py` (not an in-place edit of
  `7_test_summarizer.py` — different shape of task once it spans two models,
  a frozen dataset, and four metric families). Reuses the existing prompt
  template, token budgets, decoding params (`repetition_penalty=1.15`, no
  `no_repeat_ngram_size` — deliberately absent, it corrupts the opening
  Sinhala grapheme cluster, do not reintroduce it), and the existing
  `data_quality_checks.py` glue/unit checks, unchanged.
- New `abstractive/summarizer_metrics.py`: pure extraction of the
  grapheme-cluster-safe `sinhala_tokenize()`/`rouge_scores()` logic that's
  currently duplicated across `4_test_summarizer.py`, `6_test_summarizer.py`,
  and `7_test_summarizer.py` (and again, separately, in
  `work/sinllama/scripts/test_grammar.py` — not touched, out of scope). No
  behavior change, just de-duplication for the new script to import from.
  The older numbered eval scripts are left alone as historical artifacts.

### Phase 4 — Semantic similarity
- Use **LaBSE**, not the `paraphrase-multilingual-MiniLM-L12-v2` model
  that's already cached locally — LaBSE's documented language list includes
  Sinhala (`si`); the cached MiniLM model's does not. Trusting an
  undocumented-for-Sinhala model would violate the proposal doc's own
  warning (§8) not to assume multilingual BERTScore/embedding models work
  for Sinhala without checking.
- Before trusting it on real data: build a small (~15–20 pair) hand-authored
  Sinhala sanity set — true paraphrases, near-duplicates, entity-swapped
  adversarial pairs (e.g. swap a country name — this is deliberately
  designed to test whether the metric would catch what ROUGE and it itself
  can't), and unrelated pairs. Confirm the similarity ordering makes sense
  before reporting any number from it.
- Add `sentence-transformers` (already installed in the environment, 5.4.1,
  but currently undeclared) to `requirements.txt`.

### Phase 5 — LLM-judge factuality
- Reuse `5_gemini_summary_generator.py`'s existing multi-key rotation,
  rate-limiting, and resumable-output infrastructure directly — it's a
  working pattern already in this repo, not something to rebuild.
- New `abstractive/8_llm_judge_factuality.py`: one Gemini 2.0 Flash call per
  (article, generated summary), `temperature=0`, structured JSON output
  covering entities-in-reference / entities-preserved / entities-altered
  (this is what would have caught the known "Portugal and Uruguay" →
  "Pakistan" failure — the existing regex checks in `data_quality_checks.py`
  only catch numeric/unit mismatches, not entity substitution), plus
  numbers-preserved/altered and an event-polarity-flip flag. Compute
  `entity_preservation_rate`, `number_preservation_rate`, and
  `factual_error_rate` locally from the judge's raw lists — don't let the
  judge invent its own composite score.
- **Cost control**: run this on a fixed N=100 stratified sub-sample of the
  N=300 eval subset, not the full 300 or the full ~3,555-row test partition.
  Full-scale (300×3×2-3 models ≈ 1,800–2,700 calls) would take multiple
  hours to over a day at the existing script's free-tier rate limit
  (`RPM_LIMIT=14`). N=100×3×2 ≈ 600 calls is tractable in under an hour.
  ROUGE and semantic similarity still run on the full N=300.
- Keep `data_quality_checks.py`'s regex checks running as-is in parallel —
  cheap, deterministic, catches different defect classes. Report both sets
  of numbers separately, don't merge into one score.

### Phase 6 — Human evaluation (scaffold only)
- New `abstractive/8_generate_human_eval_sheet.py`: draws ~40–50 articles
  from the Phase 5 sample, emits both models' summaries under randomized,
  blinded "Summary A / Summary B" labels with empty 1–5 rating columns
  (faithfulness, coverage, fluency, conciseness, overall) as a CSV. A
  separate, not-distributed key file maps A/B back to real model names.
- New `abstractive/8_score_human_eval.py`: scores a filled-in sheet once you
  have it — means, stddev, and inter-rater agreement if ≥2 reviewers rate
  the same items.
- **This phase only produces the scaffold.** Recruiting and scheduling
  actual Sinhala-speaking (ideally journalism-experienced) reviewers is
  outside anything a script can do — that's a logistics step for you.

### Phase 7 — Decision and paper write-up
- Apply the proposal doc's decision hierarchy (factuality > human
  faithfulness > semantic quality > length control > ROUGE, §18–19)
  mechanically against whatever the Phase 3–6 numbers actually show — this
  is a judgment call made once real data exists, not something to pre-decide.
- Update `research-paper.tex`: replace Table II (currently v06-only,
  15-article) with the new dual-model multi-metric table; rewrite the "News
  Summarization" results prose using whichever of the proposal doc's two
  pre-drafted outcome paragraphs (§27) actually matches; update Limitations
  (remove "lacks article-level blind split," add whatever new limitations
  the LLM-judge sample size / single-judge-model / human-eval reviewer count
  introduce); reconcile Fig. 1 permanently based on the real decision.

---

## 4. Contamination audit result — changes the Phase 2 plan

Running `8_audit_split_contamination.py` against the frozen test set
produced this (numbers below are from the final, raw-file-built split —
re-run after the §2C correction; the first pass, against the
cleaned-file-built split, gave 81.10%/81.02% — same conclusion, ~0.3pp
noise from the 22-article universe difference):

| adapter | train articles | own-split straddling | frozen-TEST contamination | frozen-EVAL-SUBSET contamination |
|---|---:|---:|---:|---:|
| v06 | 74,103 | 12.93% | **80.69%** (2,871/3,558) | **82.33%** (247/300) |
| v07 | 74,074 | 12.97% | **80.80%** (2,875/3,558) | **82.67%** (248/300) |

Two things worth understanding about this number:

1. **It's not (mainly) about the qwen supplement file.** I checked whether
   the 170,923-row `5_qwen_summaries.jsonl` supplement (re-bucketed into
   "long" samples during training) was driving the overlap — only 29.5% of
   the frozen test set's articles appear in that file, nowhere near enough
   to explain 81%. The real cause is simpler and worse: because the split
   happens at the *sample* level (each article contributes up to 3
   short/medium/long samples), an article needs *all three* of its samples
   to land in the 15% validation slice to be absent from the original
   training set. With an 85% train / 15% val split, the chance of that for
   a 3-sample article is `0.15³ ≈ 0.3%` — so well over 99% of multi-bucket
   articles are *guaranteed* to have at least one variant in the original
   training data almost regardless of which 3,556 articles the new frozen
   split happens to hold out. This is a much stronger effect than "some
   leakage" — it's close to "nearly every article was seen during training
   in some form."
2. **v06 and v07 are contaminated at essentially the same rate** (81.10%
   vs. 81.02%, i.e. within noise). This is different from what Phase 2's
   original recommendation assumed. That recommendation was: keep v06
   untouched as a historical baseline and evaluate it as-is on the new
   frozen test set, retrain only v07. But if evaluating the *existing* v06
   checkpoint on the frozen test set is *also* ~81% contaminated, that
   evaluation isn't a clean baseline either — it has the same problem it
   was supposed to be free of.

This is why Phase 2 needs to retrain **both** v06 and v07, not just v07 —
see §5.

---

## 5. Phase 2 decision — retrain both, commands ready, not yet run

Decided: retrain both v06 and v07 on the frozen split's train partition,
each with its own existing hyperparameters/data-cleaning recipe unchanged
(v06: raw corpus + simple quality filter; v07: cleaned corpus + glue/unit
filters). This directly answers "does cleaning help," using two
freshly-trained models neither of which has seen the frozen test set. The
two training scripts are written (`abstractive/8_train_summarizer_v06_frozensplit.py`,
`abstractive/8_train_summarizer_v07_frozensplit.py` — see Phase 2 above for
what each does differently). **Neither has been run yet.**

Cost: two full LoRA fine-tunes, same class as the original runs. The
original v07 training run took ~13 hours (checkpoint timestamps: Aug 10
19:54 → Aug 11 08:56 final). Budget roughly a day for both, run
sequentially on the single A40 (46GB, ~36GB free at last check).

**Commands** (run in your own terminal/tmux/screen — not something to hold
open in a chat session for 13-26h):

```bash
cd /home/jovyan/summarizer/abstractive

# v06 recipe on the frozen split
nohup python3 8_train_summarizer_v06_frozensplit.py > v06_frozensplit_train.log 2>&1

# then, once that finishes:
nohup python3 8_train_summarizer_v07_frozensplit.py > v07_frozensplit_train.log 2>&1
```

Or chained as one background job:

```bash
cd /home/jovyan/summarizer/abstractive
nohup bash -c '
  python3 8_train_summarizer_v06_frozensplit.py > v06_frozensplit_train.log 2>&1 &&
  python3 8_train_summarizer_v07_frozensplit.py > v07_frozensplit_train.log 2>&1
' > frozensplit_retrain_both.log 2>&1 &
disown
```

Both write to `work/sinllama/models/adapters_staging/`, not `ADAPTERS_DIR`,
so neither goes live via `find_latest_adapters()`'s version-sort before
results are reviewed. **Phase 3 (the evaluation script that actually scores
both resulting checkpoints against `summarization_frozen_eval_subset.jsonl`)
is not built yet** — needed once training finishes so the checkpoints don't
just sit there unscored.

---

## 6. Open decision — deferred, not resolved

I asked whether to pin the live server back to v06 now, given it's currently
serving v07 by version-sort accident and the one existing comparable eval
favors v06. **You said: don't change anything yet.** So `serve_sinai.py` is
untouched, and v07 continues to be what real users get today, purely by
version-number sort, until you decide otherwise. This should be revisited
independently of the rest of this plan — it doesn't require waiting for
Phase 0–7 to finish if you want to act on it sooner.

---

## 7. Cost / risk flags

- **LLM-judge is the single most expensive part of the full framework.**
  Full-scale (all 300 eval-subset articles × 3 bands × 2–3 models) would be
  1,800–2,700 Gemini calls — multiple hours to over a day on free-tier rate
  limits. The N=100 recommendation above cuts this to ~600 calls, under an
  hour, but is itself a scope reduction worth confirming.
- **Human-eval logistics are entirely outside engineering scope** — the
  scaffold prepares the artifact; running it needs actual reviewers.
- **GPU cost for automatic metrics is non-trivial but bounded**: ~300×3×2–3
  ≈ 1,800–2,700 generations across the models being compared.
- **Retraining v07 on the frozen split is a full LoRA fine-tune**, the same
  cost class as the original v07 training run — not incremental, and it
  blocks the primary comparison numbers until it finishes.

---

## 8. Suggested next action

Phase 0 and Phase 1 are done (§Phase 0, §Phase 1 above). Phase 2's two
training scripts are ready — see §8 for the exact commands. Once both
finish, Phase 3 (`abstractive/8_evaluate_summarizer.py` — not yet built)
is the next thing needed to actually score them against
`summarization_frozen_eval_subset.jsonl`. Phases 4–7 (semantic similarity,
LLM-judge factuality, human eval, paper rewrite) remain scoped but not
started, per the cost flags in §5.
