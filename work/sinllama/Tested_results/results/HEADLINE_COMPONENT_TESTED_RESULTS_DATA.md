# Headline Component Tested Results — Recorded Data

Generated from the JSON files in `SinAI-Training/work/sinllama/Tested_results/results/` on 2026-08-18.

This document contains values recorded in the source files and percentages calculated directly from recorded counts. It does not add qualitative findings, recommendations, or model-quality claims. Metric values are displayed to six decimal places. `NR` means the field is not recorded in that source file.

## Source inventory

| Source file | Format | Per-example rows | SHA-256 |
|---|---:|---:|---|
| `headline_eval_results_v08.json` | Summary + results | 600 | `7836781b6f4941e50278e1f9ed32a2a7425a3047bfe6f765014ca11df58be358` |
| `headline_eval_results_v09.json` | Summary + results | 600 | `8fdf7332e53899555be479378158c0b410dec1d48b013db106e84c69f83b45d4` |
| `headline_eval_results_v11.json` | Summary + category + results | 600 | `75d05b5420e0d87ae0171afbd4cf28772d2699f2ced2b5d91dff4689bf3b3bf4` |
| `headline_eval_results_v12.json` | Summary + category + results | 1,200 | `15c906ce8d9bb2bf79b9695582458c3a9943fea31fecf79028e12242ce62884f` |
| `headline_eval_results_v13.json` | Summary + category + results | 1,200 | `3276905856d26aa992be6bf1908075bfe40d82e8e91d73b7bc77184edf2f4dc7` |
| `headline_eval_results_v14.json` | Summary + category + results | 2,395 | `0343587e5f03f275fd5c36a6ad73f923db63d1fb13b27898c9ab1290eeaeb7d2` |
| `headline_eval_results_v16.json` | Summary + category + results | 6,732 | `d5c868d03b9cc64b32ec5874a399c82db9890436fd7b500c807e33328343e9ee` |
| `headline_eval_results_v17.json` | Summary + category + results | 4,795 | `b5116b1738a14d5bae9fa50fd9d3c2aa4be225801963ae477179991e5ed1946b` |
| `headline_eval_results_v18.json` | Band aggregates | 0 | `6537a78cc2d99e6d40492c318348979b8221a2bf2534c50c7692efba053dd360` |
| `headline_eval_results_v19.json` | Band aggregates | 0 | `0015e546b74279c9a3294d594ea4f622c3d031afaf3e70ca2d8024ae34934be6` |
| `headline_eval_results_v20.json` | Band aggregates | 0 | `cf32bc8a7a7f76a841f58e0f1381c43b6ae35b9727ca442fcb0aed073473799b` |
| `hirunews_general_eval_results_v17.json` | Summary + category + results | 100 | `7b5cd989c87e4d3c513634208c6a9f95817eb2f6ef98b9b81c74c339bef53b82` |

## Recorded configurations: summary evaluations

| Source | Adapter recorded in file | Dataset recorded in file | LoRA rank | Sampling | Temperature | Retry temperature | Top-p | Length penalty | Repetition penalty | Min tokens | Max tokens | No-repeat n-gram | Retry on short |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v08 | `headline_sinllama_v08` | NR | NR | true | 0.3 | NR | 0.9 | NR | NR | NR | 60 | NR | NR |
| v09 | `v09` | NR | NR | true | 0.3 | NR | 0.9 | 0.6 | NR | NR | 40 | NR | NR |
| v11 | `headline_sinllama_v10` | NR | 128 | true | 0.2 | NR | 0.85 | 0.7 | 1.15 | NR | 50 | NR | NR |
| v12 | `headline_sinllama_v12` | `12K train / 1.2K val` | 64 | true | 0.2 | NR | 0.85 | 0.7 | 1.15 | NR | 50 | NR | NR |
| v13 | `headline_sinllama_v13` | `12K train / 1.2K val` | 32 | true | 0.3 | NR | 0.9 | 1.0 | 1.1 | NR | 60 | NR | NR |
| v14 | `headline_sinllama_v14` | `24K train / ~2.4K val / 12 categories` | 64 | true | 0.3 | 0.6 | 0.9 | 1.0 | 1.1 | 5 | 60 | 2 | true |
| v16 | `headline_sinllama_v16` | `48K train / balanced 12 categories` | 64 | true | 0.3 | 0.6 | 0.9 | 1.0 | 1.1 | 5 | 60 | 2 | true |
| v17 | `headline_sinllama_v16` | `48K train / balanced 12 categories` | 64 | true | 0.3 | 0.6 | 0.9 | 1.0 | 1.1 | 5 | 60 | 2 | true |
| Hiru News v17 | `headline_sinllama_v17` | `Hirunews General Test Dataset` | NR | true | 0.3 | 0.6 | 0.9 | 1.0 | 1.1 | 5 | 60 | 2 | true |

All configurations record `nfc_normalization: true`. v09–v17 record `loss_masking: true` where the field is present. The v08 configuration records this note: `beam search disabled — Unsloth tuple KV cache is incompatible with HF beam search (.reorder_cache)`.

## Overall metric records

| Source | Samples | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | Exact matches | Empty outputs | Average generated words | Average reference words | Average length ratio | Retried |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v08 | 600 | 0.180309 | 0.053235 | 0.173983 | NR | 0 | NR | 12.991667 | 5.318333 | 2.616575 | NR |
| v09 | 600 | 0.247075 | 0.079886 | 0.241861 | NR | 4 | NR | 4.775000 | 5.318333 | 0.959777 | NR |
| v11 | 600 | 0.114931 | 0.020564 | 0.112767 | 0.002477 | 0 | 0 | 5.103333 | 5.318333 | 1.028968 | NR |
| v12 | 1,200 | 0.145147 | 0.032676 | 0.143268 | 0.002116 | 4 | 0 | 4.660000 | 5.386667 | 0.923211 | NR |
| v13 | 1,200 | 0.181371 | 0.049244 | 0.178693 | 0.002887 | 7 | 0 | 5.111667 | 5.386667 | 1.015000 | NR |
| v14 | 2,395 | 0.138903 | 0.024655 | 0.135347 | 0.001952 | 1 | 1 | 6.427975 | 6.606681 | 1.059545 | NR |
| v16 | 6,732 | 0.293308 | 0.162635 | 0.290962 | 0.092495 | 485 | 1 | 6.236631 | 6.447712 | 1.034727 | 530 |
| v17 | 4,795 | 0.138213 | 0.025535 | 0.135803 | 0.001337 | 0 | 1 | 5.851303 | 6.567258 | 0.970368 | 522 |
| Hiru News v17 | 100 | 0.156657 | 0.028050 | 0.152142 | 0.000000 | 0 | 0 | 6.250000 | 8.850000 | 0.814500 | 10 |

### Character-overlap metric records

| Source | Character ROUGE-3 | Character ROUGE-4 | Character ROUGE-L |
|---|---:|---:|---:|
| v08 | 0.256817 | 0.212943 | 0.416777 |
| v09 | 0.320703 | 0.271011 | 0.506571 |
| v11 | 0.207183 | 0.160459 | 0.426983 |
| v12 | 0.241690 | 0.194017 | 0.457141 |
| v13 | NR | NR | NR |
| v14 | NR | NR | NR |
| v16 | NR | NR | NR |
| v17 | NR | NR | NR |
| Hiru News v17 | NR | NR | NR |

## Recorded length distributions

Percentages in parentheses are derived as `bucket count / total_samples × 100`. The bucket names are preserved from each JSON file. `Unbucketed` is derived as `total_samples - sum(recorded bucket counts)`.

| Source | Under 4 | 4–5 | 4–7 | 6–10 | 8–10 | Over 10 | Unbucketed |
|---|---:|---:|---:|---:|---:|---:|---:|
| v08 | 11 (1.83%) | 105 (17.50%) | NR | 300 (50.00%) | NR | 184 (30.67%) | 0 |
| v09 | 128 (21.33%) | NR | 440 (73.33%) | NR | 31 (5.17%) | 1 (0.17%) | 0 |
| v11 | 106 (17.67%) | NR | 435 (72.50%) | NR | 59 (9.83%) | 0 (0.00%) | 0 |
| v12 | 289 (24.08%) | NR | 844 (70.33%) | NR | 67 (5.58%) | 0 (0.00%) | 0 |
| v13 | 184 (15.33%) | NR | 905 (75.42%) | NR | 111 (9.25%) | 0 (0.00%) | 0 |
| v14 | 2 (0.08%) | NR | 1,782 (74.41%) | NR | 610 (25.47%) | 0 (0.00%) | 1 |
| v16 | 114 (1.69%) | NR | 5,016 (74.51%) | NR | 1,601 (23.78%) | 0 (0.00%) | 1 |
| v17 | 46 (0.96%) | NR | 3,985 (83.11%) | NR | 763 (15.91%) | 0 (0.00%) | 1 |
| Hiru News v17 | 0 (0.00%) | NR | 80 (80.00%) | NR | 20 (20.00%) | 0 (0.00%) | 0 |

## Per-category metric records

`Character ROUGE-L` is recorded only in the v11 and v12 per-category objects.

| Source | Category | Count | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | Character ROUGE-L | Average generated words |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v11 | General | 300 | 0.115870 | 0.024858 | 0.115092 | 0.004953 | 0.432659 | 4.846667 |
| v11 | Politics | 150 | 0.113631 | 0.019537 | 0.110753 | 0.000000 | 0.413588 | 5.493333 |
| v11 | Business | 150 | 0.114356 | 0.013003 | 0.110131 | 0.000000 | 0.429026 | 5.226667 |
| v12 | Politics | 300 | 0.133382 | 0.027201 | 0.132173 | 0.002596 | 0.448461 | 5.006667 |
| v12 | General | 600 | 0.143513 | 0.030993 | 0.142208 | 0.001266 | 0.465310 | 4.338333 |
| v12 | Business | 300 | 0.160178 | 0.041514 | 0.156482 | 0.003333 | 0.449482 | 4.956667 |
| v13 | Politics | 300 | 0.158944 | 0.041520 | 0.158239 | 0.003815 | NR | 5.513333 |
| v13 | General | 600 | 0.187963 | 0.052702 | 0.185250 | 0.003304 | NR | 4.680000 |
| v13 | Business | 300 | 0.190616 | 0.050053 | 0.186031 | 0.001125 | NR | 5.573333 |
| v14 | General | 322 | 0.136816 | 0.028184 | 0.134031 | 0.003106 | NR | 6.456522 |
| v14 | Business | 322 | 0.141914 | 0.020510 | 0.139754 | 0.000000 | NR | 6.593168 |
| v14 | Law and Order | 230 | 0.133731 | 0.021414 | 0.129811 | 0.004004 | NR | 7.034783 |
| v14 | Entertainment | 322 | 0.132196 | 0.028180 | 0.129894 | 0.000000 | NR | 6.015528 |
| v14 | International | 322 | 0.156403 | 0.021620 | 0.152822 | 0.003449 | NR | 6.295031 |
| v14 | Health | 48 | 0.176736 | 0.029535 | 0.176736 | 0.000000 | NR | 5.979167 |
| v14 | Politics | 322 | 0.147664 | 0.033226 | 0.141975 | 0.002477 | NR | 6.475155 |
| v14 | Science | 75 | 0.115662 | 0.010769 | 0.108896 | 0.000000 | NR | 7.320000 |
| v14 | Sports | 322 | 0.153848 | 0.028114 | 0.148297 | 0.002627 | NR | 6.304348 |
| v14 | Human Rights | 14 | 0.079418 | 0.000000 | 0.079418 | 0.000000 | NR | 6.714286 |
| v14 | Editorial | 86 | 0.023949 | 0.002907 | 0.023949 | 0.000000 | NR | 6.000000 |
| v14 | Technology | 10 | 0.181709 | 0.018182 | 0.181709 | 0.000000 | NR | 5.000000 |
| v16 | Technology | 400 | 0.727549 | 0.577219 | 0.726761 | 0.373168 | NR | 5.940000 |
| v16 | Entertainment | 722 | 0.125224 | 0.026687 | 0.122367 | 0.002847 | NR | 5.849030 |
| v16 | General | 722 | 0.141108 | 0.024872 | 0.139088 | 0.002368 | NR | 6.355956 |
| v16 | Law and Order | 400 | 0.242214 | 0.107643 | 0.238748 | 0.049638 | NR | 6.740000 |
| v16 | International | 722 | 0.155703 | 0.029230 | 0.153390 | 0.001569 | NR | 6.271468 |
| v16 | Politics | 722 | 0.150231 | 0.032061 | 0.146722 | 0.004118 | NR | 6.383657 |
| v16 | Sports | 722 | 0.171282 | 0.035612 | 0.167352 | 0.002158 | NR | 6.293629 |
| v16 | Editorial | 400 | 0.633853 | 0.535809 | 0.632592 | 0.421900 | NR | 6.382500 |
| v16 | Health | 400 | 0.604418 | 0.386889 | 0.604170 | 0.172970 | NR | 4.850000 |
| v16 | Science | 400 | 0.484310 | 0.308711 | 0.483859 | 0.167130 | NR | 6.890000 |
| v16 | Human Rights | 400 | 0.629532 | 0.504917 | 0.627555 | 0.347437 | NR | 6.200000 |
| v16 | Business | 722 | 0.150912 | 0.026581 | 0.148208 | 0.000479 | NR | 6.497230 |
| v17 | Sports | 722 | 0.164555 | 0.034446 | 0.160730 | 0.003450 | NR | 5.826870 |
| v17 | Business | 722 | 0.151530 | 0.029996 | 0.147500 | 0.000640 | NR | 6.126039 |
| v17 | Law and Order | 230 | 0.115732 | 0.015642 | 0.112081 | 0.001788 | NR | 5.839130 |
| v17 | General | 722 | 0.135868 | 0.025212 | 0.134020 | 0.001340 | NR | 6.092798 |
| v17 | Politics | 722 | 0.129178 | 0.021986 | 0.127645 | 0.000569 | NR | 6.006925 |
| v17 | Entertainment | 722 | 0.124275 | 0.023921 | 0.123223 | 0.000758 | NR | 5.407202 |
| v17 | International | 722 | 0.148405 | 0.025030 | 0.146178 | 0.001552 | NR | 5.800554 |
| v17 | Editorial | 86 | 0.033780 | 0.001938 | 0.033455 | 0.000000 | NR | 5.023256 |
| v17 | Science | 75 | 0.101379 | 0.008721 | 0.099475 | 0.000000 | NR | 5.786667 |
| v17 | Health | 48 | 0.146743 | 0.036245 | 0.146743 | 0.000000 | NR | 5.333333 |
| v17 | Human Rights | 14 | 0.056227 | 0.000000 | 0.053463 | 0.000000 | NR | 5.428571 |
| v17 | Technology | 10 | 0.131960 | 0.033333 | 0.129762 | 0.000000 | NR | 5.800000 |
| Hiru News v17 | General | 100 | 0.156657 | 0.028050 | 0.152142 | 0.000000 | NR | 6.250000 |

## Length-band evaluation records: v18–v20

Each file records `sample_size: 300` and the same word-count bands:

| Band | Minimum words | Maximum words |
|---|---:|---:|
| Short | 3 | 5 |
| Medium | 6 | 7 |
| Long | 8 | 10 |

### Band counts

The percentages are derived from the recorded `n` value in each row.

| Source | Adapter | Band | n | In band | In-band rate | Artifact | Artifact rate | Empty |
|---|---|---|---:|---:|---:|---:|---:|---:|
| v18 | `headline_sinllama_v18` | Short | 300 | 266 | 88.67% | 1 | 0.33% | 0 |
| v18 | `headline_sinllama_v18` | Medium | 300 | 228 | 76.00% | 33 | 11.00% | 0 |
| v18 | `headline_sinllama_v18` | Long | 300 | 234 | 78.00% | 67 | 22.33% | 0 |
| v19 | `headline_sinllama_v19` | Short | 300 | 269 | 89.67% | 0 | 0.00% | 0 |
| v19 | `headline_sinllama_v19` | Medium | 300 | 223 | 74.33% | 1 | 0.33% | 0 |
| v19 | `headline_sinllama_v19` | Long | 300 | 225 | 75.00% | 9 | 3.00% | 0 |
| v20 | `headline_sinllama_v20` | Short | 300 | 254 | 84.67% | 1 | 0.33% | 0 |
| v20 | `headline_sinllama_v20` | Medium | 300 | 226 | 75.33% | 3 | 1.00% | 0 |
| v20 | `headline_sinllama_v20` | Long | 300 | 239 | 79.67% | 7 | 2.33% | 0 |

### Own-band metric records

| Source | n | ROUGE-1 | ROUGE-L | BLEU |
|---|---:|---:|---:|---:|
| v18 | 300 | 0.123828 | 0.120580 | 0.000000 |
| v19 | 300 | 0.133909 | 0.129651 | 0.000876 |
| v20 | 300 | 0.139238 | 0.135730 | 0.000876 |

## Per-example data fields

The full per-example records remain in the source JSON files listed above. Their recorded fields are:

| Sources | Fields |
|---|---|
| v08–v09 | `index`, `category`, `expected`, `generated`, `rouge1_word`, `rouge2_word`, `rougeL_word`, `rouge_char3`, `rouge_char4`, `rouge_charL`, `exact_match`, `word_count_gen`, `word_count_ref`, `length_ratio` |
| v11–v12 | All v08–v09 fields plus `bleu`, `has_sinhala`, and `is_empty` |
| v13 | `index`, `category`, `expected`, `generated`, `rouge1`, `rouge2`, `rougeL`, `bleu`, `exact_match`, `has_sinhala`, `is_empty`, `word_count_gen`, `word_count_ref`, `length_ratio` |
| v14, v16, v17, and Hiru News v17 | All v13 fields plus `was_retried` |
| v18–v20 | No per-example `results` array is recorded |
