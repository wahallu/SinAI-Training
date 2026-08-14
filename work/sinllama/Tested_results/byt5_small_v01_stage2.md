# Sinhala grammar prediction evaluation

- predictions: `/home/jovyan/work/sinllama/Tested_results/byt5_small_v01_stage2_predictions.jsonl`
- predictions SHA-256: `9edfae0c5d6f7a66f63ff22fa901e5c8cbfd77318e0ef1b38ea5b1660b0cff99`
- gold: `/home/jovyan/work/sinllama/data/grammar_test_stage2.jsonl`
- gold SHA-256: `9d1d52b3a502724a973d35844253cde0d6690f6d99277564b8ebbbaa3260beef`
- examples: **57** (42 change / 15 clean)
- bootstrap: 1000 document-clustered samples

## Primary metrics

| Metric | Result |
|---|---:|
| Edit precision | 52.94% (26.67%–78.57%) |
| Edit recall | 18.00% (7.41%–30.62%) |
| Edit F0.5 | 38.14% (17.70%–56.29%) |
| Detection precision | 64.71% |
| Detection recall | 22.00% |
| Detection F1 | 32.84% |
| Unseen-pair recall (N=17) | 0.00% (0.00%–0.00%) |
| Unseen-lemma recall (N=0) | 0.00% |
| Contextual exact (N=0) | 0.00% (0.00%–0.00%) |

## Restraint and exactness

| Metric | Result |
|---|---:|
| Overall exact match | 35.09% (22.81%–49.12%) |
| Change-needed exact | 21.43% (9.09%–35.42%) |
| Clean preservation | 73.33% (50.00%–94.44%) |
| Over-correction | 26.67% (5.56%–50.00%) |
| Under-correction | 69.05% |
| Wrong correction | 9.52% |
| Protected-span mutation | 0.00% (0.00%–0.00%) |
| Number mutation | 0.00% (0.00%–0.00%) |

## Per category

| Category | N | Exact | Edit P | Edit R | F0.5 |
|---|---:|---:|---:|---:|---:|
| unlabelled | 57 | 35.09% | 52.94% | 18.00% | 38.14% |

Confidence intervals are document-clustered. If the gold has no `source_document_id`, each example is treated as its own cluster.
