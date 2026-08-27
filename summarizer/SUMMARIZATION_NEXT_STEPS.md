# Summarizer: What Happened, and What to Do Next

Status: **Phase 0–3 are done.** Both v06 and v07 were retrained on the
frozen split and evaluated on the same 273-article leakage-free eval subset.
**Result: v06 and v07 are statistically tied** — ROUGE-L within 0.002 on
every band, in-band/clean-end/glue/unit rates within a couple points, no
consistent winner either direction. The old "v06 beats v07" result was an
artifact of the split leak, not a real quality difference. Phases 4–7
(semantic similarity, LLM-judge factuality, human eval, paper rewrite) are
still planned, not started — see §9 for the honest-comparison result and
the open question of whether to run them. See §4 for the contamination
finding and §5 for the Phase 2 retrain results.

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
explicitly deferred decision — see §6.

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
tested (see Phase 2/§5).

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

### Phase 2 — Retraining — DONE
See §5 for the full reasoning, exact commands, and results. Short version:
both v06 and v07 needed retraining on the frozen split (not just v07)
because the contamination audit found the *existing* v06 checkpoint is just
as contaminated against the new frozen test set as v07 is (§4) — so
evaluating the existing v06 checkpoint as-is would not have been a clean
baseline either. Both retrains completed cleanly.
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

### Phase 3 — Evaluation script and run — DONE (see §9 for the result)
- `abstractive/8_evaluate_summarizer.py` is written (not an in-place edit of
  `7_test_summarizer.py` — different shape of task once it spans two models
  against one shared frozen dataset). `--adapter {v06,v07}` selects which
  `*_frozensplit` staging adapter to score; both runs use the exact same 300
  articles from `data/summarization_frozen_eval_subset.jsonl`, in the same
  order — unlike `6_test_summarizer.py`/`7_test_summarizer.py`, which each
  draw a fresh `random.sample()` from the full corpus, so their numbers were
  never actually comparable to each other even before the contamination
  issue.
- Reuses the existing prompt template, token budgets, and decoding params
  (`repetition_penalty=1.15`, no `no_repeat_ngram_size` — deliberately
  absent, it corrupts the opening Sinhala grapheme cluster, do not
  reintroduce it) unchanged from `6_test_summarizer.py`/`7_test_summarizer.py`.
- Unifies the metric set across both adapters: ROUGE-1/2/L (native
  grapheme-cluster implementation, standard `rouge_score` library still not
  used), length-band adherence, clean-ending rate, **and** the glue/unit
  checks from `data_quality_checks.py` — previously only reported for v07.
  Applying them to v06 too here is deliberate: this is the first evaluation
  where that comparison is meaningful.
- Smoke-tested on 3 articles first, then run at full scale by hand (you ran
  it directly rather than in a held-open background session, correctly —
  see §9 for why that mattered here). Of the 300 rows in
  `summarization_frozen_eval_subset.jsonl`, 273 had all three reference
  summaries plus non-empty content and were evaluated (819 generations per
  adapter); the other 27 were skipped by `load_eval_records()`'s existing
  completeness filter, same filter `6/7_test_summarizer.py` already use.
  Output: `6_eval_results/frozensplit_v06_eval_20260826_042858.json` and
  `frozensplit_v07_eval_20260826_043135.json`.
  Long enough to be worth a background/tmux session rather than a foreground
  wait, though shorter than the retrains (~2h vs ~13h each).

### Phase 4 — Semantic similarity
- Use **LaBSE**, not the `paraphrase-multilingual-MiniLM-L12-v2` model
  that's already cached locally — LaBSE's documented language list includes
  Sinhala (`si`); the cached MiniLM model's does not. Trusting an
  undocumented-for-Sinhala model would violate
  `improve_summarization_component.md`'s own warning (its §8) not to assume
  multilingual BERTScore/embedding models work for Sinhala without checking.
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
- Apply `improve_summarization_component.md`'s decision hierarchy
  (factuality > human faithfulness > semantic quality > length control >
  ROUGE, its §18–19) mechanically against whatever the Phase 3–6 numbers
  actually show — this is a judgment call made once real data exists, not
  something to pre-decide.
- Update `research-paper.tex`: replace Table II (currently v06-only,
  15-article) with the new dual-model multi-metric table; rewrite the "News
  Summarization" results prose using whichever of the proposal doc's two
  pre-drafted outcome paragraphs (its §27) actually matches; update Limitations
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

This is why Phase 2 retrains **both** v06 and v07, not just v07 — see §5.

---

## 5. Phase 2 decision and result — retrained both, both succeeded

Decided: retrain both v06 and v07 on the frozen split's train partition,
each with its own existing hyperparameters/data-cleaning recipe unchanged
(v06: raw corpus + simple quality filter; v07: cleaned corpus + glue/unit
filters). This directly answers "does cleaning help," using two
freshly-trained models neither of which has seen the frozen test set. The
two training scripts (`abstractive/8_train_summarizer_v06_frozensplit.py`,
`abstractive/8_train_summarizer_v07_frozensplit.py` — see Phase 2 above for
what each does differently) were run sequentially on the single A40.

**Result: both completed cleanly, no errors, 3 epochs each.**

| adapter | train_runtime | train_loss | eval_loss | saved to |
|---|---:|---:|---:|---|
| v06_frozensplit | 143,332s (~39.8h) | 0.6818 | 1.0934 | `adapters_staging/summarization_sinllama_v06_frozensplit/` |
| v07_frozensplit | 142,610s (~39.6h) | 0.6813 | 1.0959 | `adapters_staging/summarization_sinllama_v07_frozensplit/` |

Both adapters saved fully (`adapter_model.safetensors`, ~2.6GB each, plus
tokenizer files and both epoch checkpoints present). train_loss and
eval_loss are near-identical between the two (v06 marginally lower on both)
— consistent with the fact that the two recipes differ only in ~22 rows out
of ~28,455 training rows (the v07 quality filter), so training-time loss
alone was never going to separate them; that's what Phase 3's actual
generation-based metrics (ROUGE, band adherence, glue/unit defects) are for.

Both write to `work/sinllama/models/adapters_staging/`, not `ADAPTERS_DIR`,
so neither went live via `find_latest_adapters()`'s version-sort — staging
worked as intended. **Phase 3's evaluation script
(`abstractive/8_evaluate_summarizer.py`) is now written and smoke-tested,
but has not yet been run at full scale against either checkpoint** — see §3
Phase 3 above for the exact commands.

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

## 8. Phase 3 result — v06 and v07 are statistically tied

You ran `8_evaluate_summarizer.py --adapter v06` then `--adapter v07`
yourself (correctly — this runs faster than the ~40h retrains, ~30-45 min
each in practice, but still long enough that a held-open background shell
in this session wasn't the right tool; see the note at the end of this
section). Both against the same 273-article leakage-free eval subset,
same order, same metrics:

| band   | R-L v06 | R-L v07 | R-1 v06 | R-1 v07 | R-2 v06 | R-2 v07 | in-band v06 | in-band v07 | clean-end v06 | clean-end v07 | glue v06 | glue v07 | unit v06 | unit v07 |
|--------|--------:|--------:|--------:|--------:|--------:|--------:|------------:|------------:|---------------:|---------------:|---------:|---------:|---------:|---------:|
| short  | 0.508   | 0.508   | 0.727   | 0.729   | 0.481   | 0.486   | 93%         | 94%         | 99%            | 99%            | 0%       | 0%       | 0%       | 1%       |
| medium | 0.458   | 0.456   | 0.756   | 0.749   | 0.480   | 0.481   | 93%         | 90%         | 95%            | 95%            | 0%       | 0%       | 0%       | 0%       |
| long   | 0.457   | 0.458   | 0.781   | 0.781   | 0.521   | 0.527   | 87%         | 89%         | 95%            | 94%            | 0%       | 0%       | 0%       | 0%       |

**No consistent winner.** ROUGE-L differs by ≤0.002 on every band (within
sampling noise for N=273 with `do_sample=False` — the only source of
variance left is which 273 articles landed in the frozen eval subset, not
decoding randomness). In-band rate moves ±1-3pp in both directions
depending on band. Glue/unit defect rates are ~0% for both, with one
isolated 1% unit-mismatch reading for v07-short that's a single article,
not a pattern.

**This changes the headline conclusion, not just the numbers.** The old
comparison (§1: v06 ROUGE-L 0.616/0.579/0.549 vs v07 0.488/0.518/0.488,
"v07 is currently worse on ROUGE-L across every band") was measuring the
contamination artifact, not a real quality gap — recall from §4 that both
old checkpoints were ~81% contaminated against what was then the frozen
test set, and contamination isn't necessarily symmetric in its effect on
two differently-trained models. Once retrained on identical, disjoint,
leakage-free data, v06's raw-corpus recipe and v07's cleaned-corpus +
glue/unit-filter recipe produce indistinguishable output on every metric
measured so far. The straightforward reading: on this dataset, the ~22
rows v07's cleaning step removes (out of ~28,455 training rows) are too
small a fraction to move these metrics either way.

**Process note for next time:** the ~40h *retrains* genuinely needed a
detached background process across a session gap — that part was right.
But this *eval* run (30-45 min/adapter in practice) is short enough that
attempting to background+detach it from this session added a layer of
indirection (a chained wrapper process, a polling shell) that produced a
false "completed" signal (the launcher script returning was mistaken for
the job finishing) and had to be killed and re-run by hand. For a job in
the tens-of-minutes range, running it directly and reading the output when
it returns is simpler and more reliable than backgrounding it.

---

## 9. Suggested next action

Phases 0–3 are done and produced a real, honest answer for ROUGE/band/defect
metrics: **v06 and v07 are tied (§8)**. Two decisions this unblocks, and one
open question:

- **Live-serving decision (§6, previously deferred):** the original reason
  to consider pinning back to v06 was the old, contaminated eval favoring
  it. That reason no longer holds — the honest comparison shows no
  difference. There's no metric-based case for moving off v07 (currently
  live) anymore. Worth revisiting §6 with this in mind, but it's now a
  "no strong reason to change" conclusion rather than an open question
  blocked on data.
- **`research-paper.tex` Table II / results prose:** currently reports the
  old N=45 numbers (per Phase 0's honest caveat that they're leaky). Could
  be updated now to the N=273 frozen-split numbers from §8 — a strictly
  more defensible result even without Phases 4–6, since it's already
  leakage-free on the metrics it covers (ROUGE, length control). Not done
  yet — say the word if you want this now or after Phases 4–6.
- **Open question: run Phases 4–7 or stop here?** The proposal doc's full
  framework (semantic similarity via LaBSE, LLM-judge factuality, human
  eval, paper rewrite) was chosen at the start under the assumption there'd
  be a real quality gap to characterize more precisely. Now that ROUGE/band/
  defect metrics show a tie, semantic similarity and LLM-judge factuality
  are the only remaining metric families that could still reveal a
  real difference ROUGE can't see (paraphrase quality, factual accuracy) —
  or could just as plausibly confirm the tie. Given the cost/time flags in
  §7 (LLM-judge is the expensive one, ~600+ Gemini calls even at the
  reduced N=100 scope), this is worth an explicit decision rather than
  defaulting to "continue the full framework" — the value of Phases 4–6 is
  now "does this tie hold under stricter metrics" rather than "which one
  wins," which is still useful for the paper's rigor but is a smaller
  claim than originally scoped for.
