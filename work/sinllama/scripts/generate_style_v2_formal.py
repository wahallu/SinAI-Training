#!/usr/bin/env python3
"""Generate 2000 fresh style_1_formal_news rewrites via NVIDIA-hosted
openai/gpt-oss-120b (base_url https://integrate.api.nvidia.com/v1).

Parallel-safe: run alongside the other 4 generate_style_v2_*.py scripts
in separate terminals - each pulls from a disjoint shard of train1.jsonl
(shard 0 of 5 here), so no two scripts rewrite the same source article.

    export NVIDIA_API_KEY='...'
    python generate_style_v2_formal.py

Resumable: re-running skips URLs already written to the output file
and stops as soon as 2000 valid rows exist. See style_gen_common.py
for the shared generation/validation engine and --flags (--target,
--concurrency, --output, --overwrite, ...).

Output: style_rewriter/data/style_dataset_v2_style_1_formal_news.jsonl
"""

from style_gen_common import run

STYLE_ID = "style_1_formal_news"
SHARD_INDEX = 0

if __name__ == "__main__":
    run(STYLE_ID, SHARD_INDEX)
