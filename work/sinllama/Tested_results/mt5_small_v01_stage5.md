# Sinhala grammar prediction evaluation

- predictions: `/home/jovyan/work/sinllama/Tested_results/mt5_small_v01_stage5_predictions.jsonl`
- predictions SHA-256: `bfa1beb7d233789e482a814571aea27da3ba8069ee84d163293c46b7b23383c2`
- gold: `/home/jovyan/work/sinllama/data/grammar_test_stage5.jsonl`
- gold SHA-256: `ae4cba405bc58ac832cbe7164a6f5dfbb9e098abf9bfa5fc3258a289ecfd32d9`
- examples: **51** (38 change / 13 clean)
- bootstrap: 1000 document-clustered samples

## Primary metrics

| Metric | Result |
|---|---:|
| Edit precision | 0.00% (0.00%–0.00%) |
| Edit recall | 0.00% (0.00%–0.00%) |
| Edit F0.5 | 0.00% (0.00%–0.00%) |
| Detection precision | 0.00% |
| Detection recall | 0.00% |
| Detection F1 | 0.00% |
| Unseen-pair recall (N=43) | 0.00% (0.00%–0.00%) |
| Unseen-lemma recall (N=0) | 0.00% |
| Contextual exact (N=0) | 0.00% (0.00%–0.00%) |

## Restraint and exactness

| Metric | Result |
|---|---:|
| Overall exact match | 25.49% (13.73%–37.25%) |
| Change-needed exact | 0.00% (0.00%–0.00%) |
| Clean preservation | 100.00% (100.00%–100.00%) |
| Over-correction | 0.00% (0.00%–0.00%) |
| Under-correction | 100.00% |
| Wrong correction | 0.00% |
| Protected-span mutation | 0.00% (0.00%–0.00%) |
| Number mutation | 0.00% (0.00%–0.00%) |

## Per category

| Category | N | Exact | Edit P | Edit R | F0.5 |
|---|---:|---:|---:|---:|---:|
| unlabelled | 51 | 25.49% | 0.00% | 0.00% | 0.00% |

Confidence intervals are document-clustered. If the gold has no `source_document_id`, each example is treated as its own cluster.
