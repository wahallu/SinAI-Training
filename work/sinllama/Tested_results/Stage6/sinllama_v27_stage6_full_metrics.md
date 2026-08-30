# SinLLaMA v27 grammar adapter — stage6 full metrics

## Evaluation evidence

- Predictions: `/Users/nisalfonseka/Documents/GitHub/Research/manual dataset/stage6/stage 6 final results/sinllama_v27_stage6_predictions.jsonl`
- Predictions SHA-256: `238aff4b5034c48f8b1436b60f6497dbbc4c9bbcecdbfa6f2a84239aeeb32f31`
- Gold: `/Users/nisalfonseka/Documents/GitHub/Research/manual dataset/stage6/private/grammar_stage6_gold.private.jsonl`
- Gold SHA-256: `e80f5dfe82076adaf94de147f8c4b79ed8b534347291ec87f5746509bb9f3727`
- Prediction field: `prediction`
- Evaluated samples: **286**

## EXACT-MATCH RESULTS — stage6

```text
Overall accuracy      : 163/286  →  57.0%
Change-needed accuracy: 63/143  →  44.1%
No-change accuracy    : 100/143  →  69.9%
Over-correction rate  : 43/143  →  30.1%  (changed correct sentences)
```

## CONTINUOUS METRICS — stage6

Macro-average over all samples:

```text
ROUGE-1   (grapheme): 0.9925
ROUGE-2   (grapheme): 0.9845
ROUGE-L   (grapheme): 0.9924
Sentence GLEU       : 0.9808
Char-level F1       : 0.9932
Token Precision     : 0.9921
Token Recall        : 0.9929
Token F1            : 0.9925
```

## STAGE SUMMARY

| Metric | stage6 |
|---|---:|
| Overall accuracy | 57.0% |
| Change accuracy | 44.1% |
| No-change accuracy | 69.9% |
| Over-correction rate | 30.1% |
| ROUGE-L | 0.9924 |
| Char-F1 | 0.9932 |
| GLEU | 0.9808 |
| Token F1 | 0.9925 |

## CATEGORY BREAKDOWN

| Category | N | Exact | Accuracy | ROUGE-L | Char-F1 | GLEU | Token F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 143 | 100/143 | 69.9% | 0.9952 | 0.9956 | 0.9877 | 0.9952 |
| natural-error | 55 | 31/55 | 56.4% | 0.9942 | 0.9945 | 0.9848 | 0.9942 |
| orthography-unseen | 80 | 29/80 | 36.2% | 0.9853 | 0.9880 | 0.9636 | 0.9856 |
| spacing-punctuation | 8 | 3/8 | 37.5% | 1.0000 | 0.9957 | 1.0000 | 1.0000 |

## METRIC GUIDE FOR SINHALA GRAMMAR CORRECTION

| Metric | Poor | OK | Good |
|---|---:|---:|---:|
| ROUGE-L | < 0.80 | 0.80–0.93 | > 0.93 |
| Char-F1 | < 0.85 | 0.85–0.95 | > 0.95 |
| GLEU | < 0.50 | 0.50–0.80 | > 0.80 |
| Token F1 | < 0.80 | 0.80–0.93 | > 0.93 |
| Over-correction | > 30% | 10–30% | < 10% |

## WARNINGS

- ⚠️ Change accuracy is low. Review failed corrections, dataset quality, and decoding settings.
- ⚠️ Over-correction is high (30.1%).

## INTERPRETATION NOTES

- Exact-match accuracy requires the full prediction to equal the supplied gold output.
- Continuous metrics use the historical `test_grammar.py` definitions: ROUGE and GLEU use its Unicode combining-mark tokens; Char-F1 uses code-point multiset overlap; the reported token metrics use grapheme-token multiset overlap.
- ROUGE, GLEU, Char-F1, and Token-F1 can remain high when only a small part of a long sentence is incorrect. They must not replace change-needed accuracy or over-correction reporting.
- Scores measure agreement with the supplied Stage 6 automatic gold, not independent human adjudication of every valid Sinhala correction.

## REPRODUCTION COMMAND

```bash
python3 work/sinllama/scripts/score_grammar_stage6_full_metrics.py --predictions "../manual dataset/stage6/stage 6 final results/sinllama_v27_stage6_predictions.jsonl" --gold "../manual dataset/stage6/private/grammar_stage6_gold.private.jsonl" --output "work/sinllama/Tested_results/Stage6/sinllama_v27_stage6_full_metrics.md" --stage-name "stage6" --system-name "SinLLaMA v27 grammar adapter"
```
