## Status update (2026-08-22)

This document's opening premise — "the project records do not contain an
authoritative saved v07 training run/evaluation" — is **factually stale** as
of today. A v07 adapter exists, is fully trained
(`work/sinllama/models/adapters/summarization_sinllama_v07/`, Aug 11), and
has a git-committed evaluation (`summarizer/6_eval_results/v07_eval_20260811_094434.json`,
committed Aug 14). It is also currently the **live-deployed** adapter in
`serve_sinai.py`, purely because `find_latest_adapters()` auto-selects the
highest version-numbered folder — not a quality decision.

On the one existing apples-to-apples comparison (same 45-output protocol as
v06's tracked eval), **v07 currently scores worse than v06** on ROUGE-L in
every length band and mixed on in-band rate.

None of this changes this document's core recommendation. The reason
neither number is trustworthy is exactly what this document already
diagnoses: `6_train_summarizer.py`/`7_train_summarizer.py` split their flat
sample list (short/medium/long exploded per article) at the *sample* level,
not the article level, so a train/eval split-contamination audit
(`abstractive/8_audit_split_contamination.py`) found that **~81% of a newly
built article-level frozen test set was already present in each adapter's
original training data** — v06 and v07 both, at nearly identical rates. So
"v07 lost" is not evidence v07 is worse; it's an artifact of an evaluation
protocol that can't currently distinguish the two. See
`SUMMARIZATION_NEXT_STEPS.md` for the corrected status and phased plan (the
article-level frozen split described in Section 3 below has now been built:
`data/summarization_frozen_{train,val,test}.jsonl` +
`summarization_frozen_eval_subset.jsonl`, seed 42, manifest at
`data/summarization_frozen_split_manifest.json`).

One correction to this document's own assumption, found while building the
split: Section 3 below describes needing to "identify the unique source
articles" before splitting, as if grouping were required. It isn't —
`6_multilength_summaries_clean.jsonl` already has exactly one row per
article (verified: row count equals unique-URL count), with
`summary_short`/`summary_medium`/`summary_long` as columns on that one row.
An article-level split is just a row-level split; no grouping step needed.

---

## Purpose

This document defines a clean, reproducible plan to improve the Sinhala news summarization component and determine the **best summarization adapter**.

The main recommendation is:

> **Do not immediately start experimenting with many new adapters. First clean the evaluation protocol, retrain the cleaned v07 model once, and compare v06 and v07 on the same frozen article-level test set using semantic, factual, controllability, and human-oriented measures—not ROUGE alone.**

---

## 1. Current Situation

The summarization component currently has two important versions:

* **v06** — trained on the original 35,569-row silver corpus and has a tracked evaluation.
* **v07** — cleaned version of the dataset with 35,547 rows, but the project records do not contain an authoritative saved v07 training run/evaluation. 

The v07 cleaning removed 22 problematic records involving:

* word joining,
* number/unit inconsistencies,
* unrelated content,
* other teacher-data quality issues. 

Therefore, the current situation should **not** be described as:

> “v07 is proven to be better than v06.”

It should currently be described as:

> “v07 is the cleaned follow-up dataset/model recipe, while v06 is the latest version with a tracked evaluation artifact.”

---

# 2. Problems We Need to Fix

## 2.1 The evaluation set is too small

The tracked v06 evaluation contains:

* 15 articles
* 3 requested summary lengths
* 45 generated summaries total.

The results show good length control:

| Length | Target compression | Observed | In-band | Clean ending |
| ------ | -----------------: | -------: | ------: | -----------: |
| Short  |                10% |   11.77% |  93.33% |         100% |
| Medium |                20% |   23.13% |    100% |         100% |
| Long   |                35% |   36.29% |    100% |         100% |

However, this is not large enough to support a strong general summarization-quality claim. 

---

## 2.2 The current split is not sufficiently clean

The dataset contains three summaries for each article:

* short,
* medium,
* long.

The previous evaluation/training design expanded these variants before splitting. Consequently, different length variants of the same article could potentially cross the train/evaluation boundary. The paper itself identifies this as a limitation. 

This must be fixed.

### Required rule

> **Split by article, not by summary row.**

If Article A has:

```text
Article A → Short
Article A → Medium
Article A → Long
```

all three must belong to exactly one partition:

```text
TRAIN
or
VALIDATION
or
TEST
```

Never split them independently.

---

# 3. Create a New Frozen Dataset Split

Use the cleaned **35,547-row v07 corpus**.

First identify the unique source articles.

Then create an article-level split such as:

```text
80% TRAIN
10% VALIDATION
10% TEST
```

The exact percentage can be adjusted according to the actual number of unique articles.

The critical requirement is:

```text
same article ID
       ↓
same partition
```

### Save permanently

```text
summarization_v07_train.jsonl
summarization_v07_validation.jsonl
summarization_v07_test.jsonl
```

The test file must then be treated as **frozen**.

Do not repeatedly modify or sample the test set during model development.

---

# 4. Should We Retrain?

## Yes — but only v07 first.

Do **not** immediately train:

```text
v08
v09
v10
v11
...
```

That will make the experiment difficult to interpret.

Instead:

```text
Existing v06
      │
      │
Cleaned v07 dataset
      │
      ▼
Train v07
      │
      ▼
Frozen test evaluation
```

The purpose of retraining v07 is not simply to obtain a higher score.

The purpose is to answer:

> **Does cleaning the training data improve summarization quality?**

That is a much stronger research question.

---

# 5. Keep the v06 Model

Do not delete or replace v06.

v06 is an important experimental baseline because it was trained before the 22-row data cleaning.

The comparison should become:

```text
v06
Original silver dataset
        │
        ├──────────────┐
        │              │
        ▼              ▼
     Frozen Test Set
        │
        │
v07
Cleaned dataset
```

Both models must be evaluated using:

* the same articles,
* the same requested lengths,
* the same prompts,
* the same decoding configuration,
* the same metrics.

This makes the v06 → v07 comparison meaningful.

---

# 6. Do Not Select the Adapter Using ROUGE Alone

This is one of the most important changes.

The existing evaluation uses grapheme-based ROUGE. The project record explicitly notes that this is not a sufficient independent estimate of Sinhala summarization quality. 

ROUGE should therefore remain as a **supporting lexical-overlap metric**, not the primary definition of summary quality.

---

# 7. Recommended Evaluation Framework

The new evaluation should have four major dimensions.

## 7.1 ROUGE — Supporting metric

Keep:

* ROUGE-1
* ROUGE-2
* ROUGE-L

But report clearly that these measure **surface overlap**, not complete semantic quality.

For Sinhala, document exactly how tokenization is performed.

Do not describe a high ROUGE score as automatically meaning a better summary.

---

# 8. Semantic Evaluation

Add a semantic similarity metric such as **BERTScore or an appropriate multilingual/Sinhala-capable semantic evaluator**.

The purpose is to answer:

> Does the generated summary express approximately the same meaning as the reference even when different words are used?

For example:

```text
Reference:
රජය නව ආනයන ප්‍රතිපත්තියක් හඳුන්වා දුන්නේය.

Generated:
නව ආනයන නීති හඳුන්වාදීමට රජය පියවර ගෙන ඇත.
```

The wording is different, but the meaning may be substantially similar.

ROUGE can penalize this difference.

A semantic metric can provide complementary evidence.

### Important

Do not automatically assume that any multilingual BERTScore model is reliable for Sinhala.

Before adopting it:

1. identify the underlying encoder,
2. verify Sinhala coverage,
3. run it on representative Sinhala examples,
4. inspect whether the rankings make sense.

---

# 9. Factuality Should Be a Major Metric

For a **news summarizer**, factual preservation is arguably more important than lexical overlap.

The current project already contains an important failure example where a generated summary changed **Portugal and Uruguay to Pakistan**. 

The data audit also identified number/unit errors capable of changing reported magnitudes by factors of 100–1000. 

Therefore, the new evaluation must explicitly measure factual preservation.

---

# 10. Build a News-Factuality Evaluation

For each test article/summary pair, evaluate whether the summary preserves important factual information.

### Entity preservation

Check:

* people,
* organizations,
* countries,
* cities/locations,
* institutions.

### Numeric preservation

Check:

* numbers,
* percentages,
* dates,
* currencies,
* units,
* quantities.

### Event preservation

Check:

* main event,
* actor,
* action,
* outcome.

### Polarity

Check whether:

```text
approved
```

becomes:

```text
rejected
```

or whether:

```text
did not happen
```

becomes:

```text
happened
```

---

# 11. Suggested Factuality Score

Create a simple interpretable evaluation.

For example:

```text
Entity Preservation Rate
= correctly preserved factual entities
  /
  factual entities present in reference
```

and separately:

```text
Number Preservation Rate
= correctly preserved numbers
  /
  numbers requiring preservation
```

and:

```text
Factual Error Rate
= summaries containing at least one verified factual error
  /
  total summaries
```

These are much easier to explain to a reviewer than an opaque composite score.

Do not invent a final weighted score unless the weighting is justified.

---

# 12. Human Evaluation

This should be the strongest quality check.

Automatic metrics cannot completely determine whether a Sinhala news summary is journalistically acceptable.

Use human evaluators with strong Sinhala proficiency and preferably journalism/editorial experience.

Each summary can be rated from **1–5** on:

### Faithfulness

Does the summary accurately represent the source?

### Coverage

Does it retain the important information?

### Fluency

Is the Sinhala natural and readable?

### Conciseness

Does it remove unnecessary information?

### Overall quality

Would this be acceptable as a news summary?

---

# 13. Human Evaluation Design

Do not evaluate only v07.

Use:

```text
v06
v07
```

on the same frozen articles.

Randomize the order of outputs so evaluators do not know which model generated which summary.

Ideally, use multiple evaluators per item.

Then calculate:

* mean score,
* standard deviation,
* inter-rater agreement where appropriate.

This allows us to determine whether v07 is actually preferred by humans even if ROUGE decreases.

---

# 14. Length Control Must Remain

The existing v06 results show that the model performs well at controlling requested compression ratios. 

Therefore, length control should remain an explicit evaluation dimension.

Measure:

```text
Target ratio
Observed ratio
Absolute ratio error
In-band rate
```

For example:

```text
Target = 0.20
Observed = 0.23

Absolute error = |0.23 - 0.20|
               = 0.03
```

This is more informative than simply saying the model produced a “medium” summary.

---

# 15. Check Summary Ending Quality

Keep the existing:

```text
Clean ending rate
```

This is useful because a summary can be semantically good but still be cut off.

The existing v06 model achieved 100% clean endings across the 45-output diagnostic. 

Keep this metric in the new evaluation.

---

# 16. Check Hallucination Explicitly

Add a simple human/automatic annotation:

```text
No unsupported fact
Unsupported entity
Unsupported number
Unsupported event
Other hallucination
```

This is particularly important because fluent Sinhala generation can hide factual errors.

---

# 17. Recommended Final Evaluation Table

The final experiment should produce something like:

| Dimension    | Metric              | v06 | v07 |
| ------------ | ------------------- | --: | --: |
| Lexical      | ROUGE-1             |     |     |
| Lexical      | ROUGE-2             |     |     |
| Lexical      | ROUGE-L             |     |     |
| Semantic     | Semantic similarity |     |     |
| Factual      | Entity preservation |     |     |
| Factual      | Number preservation |     |     |
| Factual      | Factual error rate  |     |     |
| Length       | Mean ratio error    |     |     |
| Length       | In-band rate        |     |     |
| Fluency      | Human score         |     |     |
| Coverage     | Human score         |     |     |
| Faithfulness | Human score         |     |     |
| Overall      | Human score         |     |     |

This table will make the adapter decision much clearer.

---

# 18. Adapter Selection Rule

Do **not** say:

> “The adapter with the highest ROUGE is the best.”

Instead use a decision hierarchy.

### Priority 1 — Factuality

A model that changes:

```text
Sri Lanka → India
10 million → 1 billion
approved → rejected
```

should not win merely because its ROUGE is higher.

### Priority 2 — Human faithfulness

Prefer summaries that human evaluators consider factually faithful.

### Priority 3 — Semantic quality

Use semantic similarity as complementary evidence.

### Priority 4 — Length control

The summary should obey the requested compression target.

### Priority 5 — Lexical overlap

ROUGE is useful as supporting evidence.

---

# 19. Recommended Decision Logic

A practical selection rule is:

```text
If v07 has:
    better/equal factuality
    AND better/equal human faithfulness
    AND acceptable semantic quality
    AND acceptable length control

then select v07
```

Even if:

```text
ROUGE(v07) < ROUGE(v06)
```

that does **not automatically mean v07 is worse**.

Conversely:

```text
ROUGE(v07) > ROUGE(v06)
```

does not automatically justify selecting v07 if factuality becomes worse.

---

# 20. Investigate the ROUGE Drop

If v07 gets lower ROUGE than v06, do not immediately reject it.

The current records already suggest a plausible reason: some v06 teacher targets contained unrelated source-page content. 

A model trained on noisy targets may reproduce that material and receive higher lexical overlap against similarly noisy references.

Therefore:

```text
Higher ROUGE
       ≠
Higher news-summary quality
```

This should be treated as a **hypothesis to test**, not stated as a proven explanation unless the new evaluation confirms it.

---

# 21. Inspect the Actual Outputs

Metrics alone are not enough.

For every model, manually inspect examples from:

### Category A — Good summaries

Correct and concise.

### Category B — High ROUGE but factually wrong

These are especially important.

### Category C — Low ROUGE but semantically correct

These demonstrate the limitation of lexical metrics.

### Category D — Hallucination

Unsupported entity/number/event.

### Category E — Length failure

Too short or too long.

### Category F — Repetition

Repeated phrases or sentences.

This qualitative analysis should accompany the numerical results.

---

# 22. Decoding Configuration

Keep summarization decoding deterministic for the main comparison.

The project already records deterministic decoding and length-specific token limits for summarization. 

Do not introduce random sampling between v06 and v07.

Both adapters must use exactly the same:

```text
temperature
top-p
max tokens
prompt
length instruction
stopping rules
```

where applicable.

---

# 23. Do Not Reintroduce `no_repeat_ngram_size=3`

The project already found that a standard no-repeat n-gram constraint corrupted opening Sinhala grapheme sequences and was removed. 

Therefore:

> Keep the Sinhala-safe decoding configuration.

Do not add the constraint again merely because it is common in English summarization systems.

---

# 24. Training Configuration for v07

The current documented v06 configuration is:

```text
Base model:       SinLlama
Quantization:     4-bit NF4
Compute:          BF16
LoRA rank:        32
LoRA alpha:       64
Dropout:          0.05
Max sequence:     2048
Effective batch:  16
Learning rate:    2e-4
Epochs:           3
Seed:             42
```

The current paper records these values for the tracked summarization configuration. 

For the first v07 retraining:

> **Keep the same configuration.**

This is important.

We are trying to isolate:

```text
data cleaning
```

rather than simultaneously changing:

```text
data + LoRA rank + learning rate + epochs + decoding
```

---

# 25. Do Not Perform Hyperparameter Search Yet

Do not start with:

```text
r=8
r=16
r=32
r=64

LR=1e-4
LR=2e-4
LR=5e-4

3 epochs
5 epochs
8 epochs
```

That would create a large experimental matrix before we know whether the cleaned dataset and evaluation protocol work.

First establish:

> **v06 vs cleaned-v07 under a controlled experiment.**

Only after that should we consider hyperparameter optimization.

---

# 26. Required Experiment Sequence

## Step 1 — Dataset audit

Verify:

* duplicate articles,
* empty articles,
* duplicate summaries,
* malformed Sinhala,
* numbers,
* units,
* unrelated text,
* source/summary alignment.

---

## Step 2 — Article-level split

Create:

```text
train
validation
test
```

with all three summary lengths of an article kept together.

---

## Step 3 — Freeze test set

Save the test set and do not use it for model development.

---

## Step 4 — Verify v06 adapter

Confirm that the original v06 adapter can still be loaded.

If available, run it on the new frozen test set.

---

## Step 5 — Retrain v07

Use:

```text
35,547 cleaned rows
```

with the existing summarization training configuration.

---

## Step 6 — Save the complete v07 experiment

Save:

```text
adapter
training configuration
training log
validation loss
checkpoint
dataset hash
train/validation/test split hashes
```

This is important for reproducibility.

---

## Step 7 — Evaluate v06

Run v06 using the exact same:

```text
test articles
prompts
lengths
decoding
metrics
```

---

## Step 8 — Evaluate v07

Run v07 using exactly the same protocol.

---

## Step 9 — Automatic evaluation

Calculate:

```text
ROUGE
Semantic similarity
Entity preservation
Number preservation
Factual error rate
Length error
In-band rate
Clean ending rate
```

---

## Step 10 — Human evaluation

Evaluate a representative subset of the frozen test set.

Compare v06 and v07 blindly.

---

## Step 11 — Qualitative error analysis

Collect examples of:

```text
hallucination
entity substitution
number corruption
wrong event
missing important fact
repetition
length failure
excellent abstraction
```

---

## Step 12 — Select final adapter

Select the model using the evaluation hierarchy rather than ROUGE alone.

---

# 27. What the Paper Should Eventually Claim

If v07 wins the new evaluation:

> “The cleaned v07 training recipe was selected as the final summarization adapter based on improved factual preservation and human-rated faithfulness while maintaining controlled summary length. ROUGE was retained as a lexical-overlap diagnostic rather than the sole quality criterion.”

If v06 wins:

> “Although the cleaned v07 recipe improved training-data quality, it did not consistently improve summarization quality under the frozen evaluation. The tracked v06 adapter therefore remained preferable under the evaluated criteria.”

Both outcomes are scientifically valid.

---

# 28. What We Should NOT Claim

Do not claim:

> “v07 is better because its dataset is cleaner.”

Until it is evaluated.

Do not claim:

> “Higher ROUGE means better summary.”

Do not claim:

> “BERTScore proves factual correctness.”

Semantic similarity and factuality are different dimensions.

Do not claim:

> “The model has high summarization accuracy.”

There is no single established “summarization accuracy” metric in the current experiment.

---

# 29. Final Recommended Architecture

The cleaned summarization experiment should ultimately look like:

```text
                 CLEANED v07 DATA
                    35,547
                       │
                       ▼
              ARTICLE-LEVEL SPLIT
                       │
          ┌────────────┼────────────┐
          │            │            │
        TRAIN          DEV         TEST
          │            │            │
          ▼            ▼            │
       SinLlama      Checkpoint     │
          +                         │
       LoRA v07                     │
          │                         │
          └─────────────┐           │
                        ▼           ▼
                    FINAL FROZEN TEST
                           │
              ┌────────────┼────────────┐
              │            │            │
           Semantic     Factuality    Control
              │            │            │
          BERTScore     Entities       Length
                         Numbers        Ending
              │            │            │
              └────────────┼────────────┘
                           │
                           ▼
                    HUMAN EVALUATION
                           │
                           ▼
                    v06 vs v07
                           │
                           ▼
                 FINAL ADAPTER
```

---

# 30. Final Recommendation

### Do this

**1. Clean and freeze the test set.**

**2. Split by article, not by summary row.**

**3. Keep v06 as the baseline.**

**4. Retrain v07 once using the existing training configuration.**

**5. Evaluate v06 and v07 on exactly the same test set.**

**6. Keep ROUGE, but treat it as lexical-overlap evidence.**

**7. Add semantic evaluation.**

**8. Add explicit factuality evaluation for entities, numbers, dates, units and events.**

**9. Add human evaluation for faithfulness, coverage, fluency, conciseness and overall quality.**

**10. Select the final adapter based primarily on factuality + human faithfulness, with semantic quality and length control as supporting criteria.**

### The key research principle

> **For a Sinhala news summarizer, the best adapter is not necessarily the adapter with the highest ROUGE. The best adapter is the one that produces concise, fluent summaries while preserving the meaning and factual content of the source.**

This approach also directly addresses the current weakness identified in the project records: the existing 45-output v06 evaluation is useful for demonstrating **length controllability**, but it is not strong enough to establish overall summarization quality. 

**Note:** some earlier uploaded files have expired from the current file workspace. If we need to implement the actual v07 training/evaluation scripts or verify the existing adapter/checkpoint rather than just use this plan, those files will need to be uploaded again.
