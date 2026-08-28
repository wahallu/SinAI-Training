# Grammar v24/v26 Dual-Gold Tested Results

Test date: **2026-08-26**  
Status: **TESTED — deterministic CPU-only rescoring of saved predictions**  
Inference status: **not rerun**; the predictions were extracted from the original v24 and v26 evaluation transcripts.

## Result

| Model | Old gold exact match | Repaired gold exact match | Repaired minus old |
|---|---:|---:|---:|
| v24 | 92/154 (**59.7%**) | 89/154 (**57.8%**) | **-1.9 pp** |
| v26 | 95/154 (**61.7%**) | 101/154 (**65.6%**) | **+3.9 pp** |
| v26 minus v24 | **+1.9 pp** | **+7.8 pp** | — |

The repaired-gold values reproduce the comparison in question: **57.8% for v24** and **65.6% for v26**. Under old gold, the difference is only **1.9 percentage points**. Therefore, the full repaired-gold difference cannot be attributed solely to better general correction.

The training logs confirm nearly matched, but not perfectly identical, runs:

| Model | Training data | Rows | Train/eval split | Trainable parameters | Steps |
|---|---|---:|---:|---:|---:|
| v24 | Stage 12, fingerprint `466bfa728af0818d` | 36,006 | 34,205 / 1,801 | 83,886,080 | 12,856 (3.01 epochs) |
| v26 | Stage 13, fingerprint `87f0a58f26b26ae2` | 36,006 | 34,205 / 1,801 | 83,886,080 | 12,828 (3.00 epochs) |

The 28-step and warmup-schedule difference is small, but it is another reason not to state that target repair alone causally produced the full change.

## Gold-change groups

The 154-example Stage 2–5 benchmark contains:

- **145 cases** where old gold = repaired gold;
- **9 cases** where old gold != repaired gold;
- among the 9 repaired-gold cases, **6 kept the same input** and **3 repaired the input as well as the gold**.

| Gold group | Model | Accuracy vs old gold | Accuracy vs repaired gold |
|---|---|---:|---:|
| Gold unchanged (n=145) | v24 | 88/145 (**60.7%**) | 88/145 (**60.7%**) |
| Gold unchanged (n=145) | v26 | 95/145 (**65.5%**) | 95/145 (**65.5%**) |
| Gold repaired (n=9) | v24 | 4/9 (**44.4%**) | 1/9 (**11.1%**) |
| Gold repaired (n=9) | v26 | 0/9 (**0.0%**) | 6/9 (**66.7%**) |

On the **145 unchanged-gold cases**, v26 exceeds v24 by **7 exact matches**, or **4.8 percentage points**. This is evidence that v26's improvement is not limited to matching the nine repaired benchmark targets. The larger **7.8-point** overall improvement under repaired gold is nevertheless partly associated with target-convention alignment.

The saved v24 transcript predates two other Stage 5 input repairs already present in the pre-v10 backup. Consequently, 2 of these 145 rows did not give v24 and v26 byte-identical inputs. Both models missed those two rows, so excluding them preserves the 7-match difference:

| Strict comparison | v24 | v26 | Difference |
|---|---:|---:|---:|
| Gold unchanged and transcript inputs identical (n=143) | 88/143 (**61.5%**) | 95/143 (**66.4%**) | **+4.9 pp** |

## Strict same-input subset of repaired-gold cases

| Subset | Model | Accuracy vs old gold | Accuracy vs repaired gold |
|---|---|---:|---:|
| Gold repaired, input unchanged (n=6) | v24 | 1/6 (**16.7%**) | 1/6 (**16.7%**) |
| Gold repaired, input unchanged (n=6) | v26 | 0/6 (**0.0%**) | 3/6 (**50.0%**) |
| Gold and input repaired (n=3) | v24 | 3/3 (**100.0%**) | 0/3 (**0.0%**) |
| Gold and input repaired (n=3) | v26 | 0/3 (**0.0%**) | 3/3 (**100.0%**) |

The final three-row subset is completely confounded: the v24 transcript was generated from the old input and the v26 transcript from the repaired input. It must not be used as causal evidence about model quality. The strict 143-row unchanged-gold, identical-input subset is the cleanest available comparison in these saved benchmark predictions.

## Training-corpus check

An executed row-by-row audit confirmed that `cleaned_v9_full.jsonl` (Stage 12) and `cleaned_v10_full.jsonl` (Stage 13):

- each contain **36,006 rows**;
- have **identical inputs in identical order** (0 input differences);
- differ in the target/output on **4,999 rows**.

Those 4,999 rows are **training-corpus rows**, not the 154-example Stage 2–5 benchmark. There are no saved v24/v26 predictions over all 36,006 corpus inputs in this checkout, and the v24/v26 adapters are not present locally. Therefore, a tested 36,006-row dual-gold table cannot be produced from the available artifacts. The tested result above uses all 154 saved benchmark predictions for each model.

## Method

The scorer:

1. parses every `INPUT`, `PREDICT`, and `EXPECTED` block from the saved transcripts;
2. aligns rows by stage and zero-based row index;
3. applies the original evaluator's rule: `prediction.strip() == gold.strip()`;
4. scores each unchanged prediction against both the pre-v10 backup gold and current repaired gold;
5. reports gold-unchanged, gold-repaired, same-input, and repaired-input groups separately.

Reproduction command, run from the `SinAI-Training` repository root:

```bash
python3 work/sinllama/scripts/score_grammar_dual_gold.py \
  --old-gold-dir '../manual dataset/test data' \
  --repaired-gold-dir '../manual dataset/test data' \
  --model 'v24=../manual dataset/Tested_results/results v24.md' \
  --model 'v26=../manual dataset/Tested_results/v26 adap reults.md'
```

The scoring script is [`score_grammar_dual_gold.py`](../../scripts/score_grammar_dual_gold.py).

## Source hashes (SHA-256)

Prediction transcripts:

- v24: `81db26b1a00ddde0c280a8cffa08f8509580e2e18eadc35108208c5c9d2190cf`
- v26: `0219e43ac1597fe68f6d385468d1b2a358d524404a298fb777e2b54f84994b3e`

Old benchmark gold:

- Stage 2: `27ad3127476ea38513b7e8a4def6796498bf3494dc52be044ebff13cd0789c2b`
- Stage 3: `f2e92c7f8c74ef2d1456751c1f2835d6043b0dbef6b1b7f830387ed55aaf236b`
- Stage 4: `e62a2542f5b74656791e6c95826861e775ad1e9a564cee589e9005133e03bbc7`
- Stage 5: `374adf84920fa8ccef7c1ff0699af380b0f9548480d126abf3e0ceb7503c4204`

Repaired benchmark gold:

- Stage 2: `9d1d52b3a502724a973d35844253cde0d6690f6d99277564b8ebbbaa3260beef`
- Stage 3: `09c2968819244cd31f55fe5636cb197a61302b07f6ab1535baf6ef80f8445e43`
- Stage 4: `9d58b75fa9e6d24db2eff9f56753f174c098a6a311164b1ad8e31c6a2799f022`
- Stage 5: `ae4cba405bc58ac832cbe7164a6f5dfbb9e098abf9bfa5fc3258a289ecfd32d9`

Training corpora:

- Stage 12 / `cleaned_v9_full.jsonl`: `eaa455ddff349d3f4c7ab384d503da0ce603ac630d0f5515a01451cd34eb07a8`
- Stage 13 / `cleaned_v10_full.jsonl`: `6e36a6efaf0f2ef5d1f3c1d664cb7d572b2a012788eb85939b4004f7aeeedb47`

## Scientifically safe claim

> Following target repair, exact match on the 154-example Stage 2–5 benchmark increased from 57.8% for v24 to 65.6% for v26 when both saved prediction sets were scored against repaired gold. On the 145 cases whose gold was unchanged, exact match increased from 60.7% to 65.5%; on the stricter 143-case subset that also gave both models identical inputs, it increased from 61.5% to 66.4%. When scored against old gold, the overall difference was 59.7% to 61.7%. These results are consistent with improved general correction as well as increased alignment to the repaired target convention, but they do not isolate the full gain causally.
