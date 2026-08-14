# Sinhala grammar prediction evaluation

- predictions: `/home/jovyan/work/sinllama/Tested_results/byt5_small_v01_stage5_predictions.jsonl`
- predictions SHA-256: `65616b1dfe6aec9f9ed36059c26ba7e099d64c4ef3c69a1ad36effb6c3be52ae`
- gold: `/home/jovyan/work/sinllama/data/grammar_test_stage5.jsonl`
- gold SHA-256: `ae4cba405bc58ac832cbe7164a6f5dfbb9e098abf9bfa5fc3258a289ecfd32d9`
- examples: **51** (38 change / 13 clean)
- bootstrap: 1000 document-clustered samples

## Primary metrics

| Metric | Result |
|---|---:|
| Edit precision | 27.78% (9.09%–50.00%) |
| Edit recall | 9.26% (2.38%–16.98%) |
| Edit F0.5 | 19.84% (5.68%–34.62%) |
| Detection precision | 33.33% |
| Detection recall | 11.11% |
| Detection F1 | 16.67% |
| Unseen-pair recall (N=43) | 9.30% (2.22%–19.15%) |
| Unseen-lemma recall (N=0) | 0.00% |
| Contextual exact (N=0) | 0.00% (0.00%–0.00%) |

## Restraint and exactness

| Metric | Result |
|---|---:|
| Overall exact match | 29.41% (17.65%–43.14%) |
| Change-needed exact | 5.26% (0.00%–13.64%) |
| Clean preservation | 100.00% (100.00%–100.00%) |
| Over-correction | 0.00% (0.00%–0.00%) |
| Under-correction | 57.89% |
| Wrong correction | 36.84% |
| Protected-span mutation | 0.00% (0.00%–0.00%) |
| Number mutation | 5.88% (0.00%–13.73%) |

## Per category

| Category | N | Exact | Edit P | Edit R | F0.5 |
|---|---:|---:|---:|---:|---:|
| unlabelled | 51 | 29.41% | 27.78% | 9.26% | 19.84% |

Confidence intervals are document-clustered. If the gold has no `source_document_id`, each example is treated as its own cluster.
