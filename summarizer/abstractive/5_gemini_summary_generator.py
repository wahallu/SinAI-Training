"""
SinhalaJournal-LLM | Step 5: Generate summaries using Gemini 2.0 Flash
-----------------------------------------------------------------------
Generates silver-label Sinhala summaries using Google Gemini 2.0 Flash.
Supports multiple API keys in parallel to maximise throughput.
Resumes automatically from a partial output file.

Usage:
    # Single key via env var:
    export GEMINI_API_KEY=your_key
    python abstractive/5_gemini_summary_generator.py

    # Multiple keys via file (one key per line):
    python abstractive/5_gemini_summary_generator.py --keys api_keys.txt

    # Custom paths:
    python abstractive/5_gemini_summary_generator.py \
        --input data/train.jsonl \
        --output data/5_gemini_summaries.jsonl
"""

import os
import json
import time
import argparse
import threading
from pathlib import Path
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
MODEL_NAME = "gemini-2.0-flash"

# Paid tier: ~2000 RPM. Free tier: 15 RPM.
# Set PAID_TIER=True if billing is enabled on your Google Cloud project.
PAID_TIER     = False
RPM_LIMIT     = 1900 if PAID_TIER else 14
DELAY_SECONDS = 60.0 / RPM_LIMIT
MAX_RETRIES   = 7

GENERATION_CONFIG = types.GenerateContentConfig(
    temperature=0.0,
    max_output_tokens=512,
)

PROMPT_TEMPLATE = """ඔබ සිංහල පුවත් සාරාංශකරණ විශේෂඥයෙකි.

පහත නීති අනිවාර්යයෙන්ම අනුගමනය කරන්න:
1. ලිපියේ ඇති තොරතුරු පමණක් භාවිතා කරන්න.
2. නව තොරතුරු එකතු නොකරන්න.
3. පුද්ගලයන්, ස්ථාන, සංඛ්‍යා වෙනස් නොකරන්න.
4. අදහස්, විශ්ලේෂණ හෝ අනුමාන එකතු නොකරන්න.
5. සාරාංශය මුල් ලිපියේ දිගෙන් 10% ත් 30% ත් අතර විය යුතුය.
6. ප්‍රධාන කරුණු පමණක් ඇතුළත් කරන්න.
7. සාරාංශය පමණක් ලබා දෙන්න.

Article:
{content}
"""


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def load_api_keys(keys_file: str) -> list[str]:
    keys = []
    if Path(keys_file).exists():
        with open(keys_file, "r") as f:
            keys = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    if not keys:
        single = os.getenv("GEMINI_API_KEY")
        if single:
            keys = [single]
    return keys


def load_processed_urls(output_path: Path) -> set:
    processed = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "url" in data:
                        processed.add(data["url"])
                except Exception:
                    continue
    return processed


# ──────────────────────────────────────────────────────────────
# WORKER
# ──────────────────────────────────────────────────────────────
def worker(api_key: str, input_queue: Queue, output_file: Path,
           lock: threading.Lock, pbar):
    client = genai.Client(api_key=api_key)

    while True:
        try:
            record = input_queue.get(timeout=5)
        except Empty:
            break

        try:
            content = record.get("content", "").strip()
            if not content:
                with lock:
                    with open(output_file, "a", encoding="utf-8") as f:
                        err = record.copy()
                        err["error"] = "empty content"
                        f.write(json.dumps(err, ensure_ascii=False) + "\n")
                        f.flush()
                pbar.update(1)
                continue

            prompt = PROMPT_TEMPLATE.format(content=content)
            success = False
            retries = 0

            while not success and retries < MAX_RETRIES:
                try:
                    t0 = time.time()
                    response = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=prompt,
                        config=GENERATION_CONFIG,
                    )
                    summary = (response.text or "").strip()

                    result = record.copy()
                    result["gemini_summary"] = summary

                    with lock:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            f.flush()
                            os.fsync(f.fileno())

                    success = True

                    # Respect RPM limit
                    elapsed = time.time() - t0
                    remaining = DELAY_SECONDS - elapsed
                    if remaining > 0:
                        time.sleep(remaining)

                except Exception as e:
                    err_msg = str(e)
                    retries += 1
                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                        # Exponential backoff capped at 5 min
                        wait = min(300, 15 * (2 ** (retries - 1)))
                        print(f"\n[RateLimit] key={api_key[:8]}… waiting {wait}s (retry {retries}/{MAX_RETRIES})")
                        time.sleep(wait)
                    elif "404" in err_msg or "INVALID_ARGUMENT" in err_msg:
                        print(f"\n[Fatal] key={api_key[:8]}… {e}")
                        break
                    else:
                        print(f"\n[Error {retries}/{MAX_RETRIES}] {repr(e)}")
                        time.sleep(min(60, 5 * retries))

            if not success:
                with lock:
                    with open(output_file, "a", encoding="utf-8") as f:
                        err = record.copy()
                        err["status"] = "failed"
                        err["error"] = f"failed after {retries} retries"
                        f.write(json.dumps(err, ensure_ascii=False) + "\n")
                        f.flush()
                        os.fsync(f.fileno())

        finally:
            input_queue.task_done()
            pbar.update(1)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="/home/jovyan/summarizer/data/train.jsonl")
    parser.add_argument("--output", default="/home/jovyan/summarizer/data/5_gemini_summaries.jsonl")
    parser.add_argument("--keys",   default="api_keys.txt",
                        help="File with one Gemini API key per line")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    api_keys = load_api_keys(args.keys)
    if not api_keys:
        print("No API keys found. Set GEMINI_API_KEY or provide --keys file.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.touch(exist_ok=True)

    processed_urls = load_processed_urls(output_path)

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    to_process = [r for r in records if r.get("url") not in processed_urls]

    print(f"Model          : {MODEL_NAME}")
    print(f"API keys       : {len(api_keys)}")
    print(f"Total articles : {len(records)}")
    print(f"Already done   : {len(processed_urls)}")
    print(f"Remaining      : {len(to_process)}")
    print(f"Output         : {output_path.resolve()}")

    if not to_process:
        print("Everything already processed.")
        return

    input_queue = Queue()
    for r in to_process:
        input_queue.put(r)

    lock = threading.Lock()

    with tqdm(total=len(to_process), desc="Summarizing") as pbar:
        with ThreadPoolExecutor(max_workers=len(api_keys)) as executor:
            for key in api_keys:
                executor.submit(worker, key, input_queue, output_path, lock, pbar)
            input_queue.join()

    print(f"\nComplete. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
