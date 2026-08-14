# Sinhala grammar prediction evaluation

- predictions: `/home/jovyan/work/sinllama/Tested_results/byt5_small_v01_stage4_predictions.jsonl`
- predictions SHA-256: `0fe70c3b46cb05e635321b7ab566cacb7d8c800417640a4c951c117dbda62d77`
- gold: `/home/jovyan/work/sinllama/data/grammar_test_stage4.jsonl`
- gold SHA-256: `9d58b75fa9e6d24db2eff9f56753f174c098a6a311164b1ad8e31c6a2799f022`
- examples: **36** (26 change / 10 clean)
- bootstrap: 1000 document-clustered samples

## Primary metrics

| Metric | Result |
|---|---:|
| Edit precision | 57.14% (33.33%–81.82%) |
| Edit recall | 12.70% (5.17%–20.69%) |
| Edit F0.5 | 33.61% (16.80%–47.10%) |
| Detection precision | 71.43% |
| Detection recall | 15.87% |
| Detection F1 | 25.97% |
| Unseen-pair recall (N=33) | 15.15% (3.45%–28.12%) |
| Unseen-lemma recall (N=0) | 0.00% |
| Contextual exact (N=0) | 0.00% (0.00%–0.00%) |

## Restraint and exactness

| Metric | Result |
|---|---:|
| Overall exact match | 27.78% (13.89%–41.74%) |
| Change-needed exact | 0.00% (0.00%–0.00%) |
| Clean preservation | 100.00% (100.00%–100.00%) |
| Over-correction | 0.00% (0.00%–0.00%) |
| Under-correction | 61.54% |
| Wrong correction | 38.46% |
| Protected-span mutation | 0.00% (0.00%–0.00%) |
| Number mutation | 0.00% (0.00%–0.00%) |

## Per category

| Category | N | Exact | Edit P | Edit R | F0.5 |
|---|---:|---:|---:|---:|---:|
| unlabelled | 36 | 27.78% | 57.14% | 12.70% | 33.61% |

Confidence intervals are document-clustered. If the gold has no `source_document_id`, each example is treated as its own cluster.
