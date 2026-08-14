# Sinhala grammar prediction evaluation

- predictions: `/home/jovyan/work/sinllama/Tested_results/byt5_small_v01_stage3_predictions.jsonl`
- predictions SHA-256: `ae204e6b2e13178451337c19bbd45461a6779c1ef077b5bb050e1d7b22f3fb73`
- gold: `/home/jovyan/work/sinllama/data/grammar_test_stage3.jsonl`
- gold SHA-256: `09c2968819244cd31f55fe5636cb197a61302b07f6ab1535baf6ef80f8445e43`
- examples: **10** (10 change / 0 clean)
- bootstrap: 1000 document-clustered samples

## Primary metrics

| Metric | Result |
|---|---:|
| Edit precision | 50.00% (0.00%–100.00%) |
| Edit recall | 8.82% (0.00%–15.62%) |
| Edit F0.5 | 25.86% (0.00%–39.69%) |
| Detection precision | 66.67% |
| Detection recall | 11.76% |
| Detection F1 | 20.00% |
| Unseen-pair recall (N=4) | 0.00% (0.00%–0.00%) |
| Unseen-lemma recall (N=0) | 0.00% |
| Contextual exact (N=0) | 0.00% (0.00%–0.00%) |

## Restraint and exactness

| Metric | Result |
|---|---:|
| Overall exact match | 0.00% (0.00%–0.00%) |
| Change-needed exact | 0.00% (0.00%–0.00%) |
| Clean preservation | 0.00% (0.00%–0.00%) |
| Over-correction | 0.00% (0.00%–0.00%) |
| Under-correction | 60.00% |
| Wrong correction | 40.00% |
| Protected-span mutation | 0.00% (0.00%–0.00%) |
| Number mutation | 0.00% (0.00%–0.00%) |

## Per category

| Category | N | Exact | Edit P | Edit R | F0.5 |
|---|---:|---:|---:|---:|---:|
| unlabelled | 10 | 0.00% | 50.00% | 8.82% | 25.86% |

Confidence intervals are document-clustered. If the gold has no `source_document_id`, each example is treated as its own cluster.
