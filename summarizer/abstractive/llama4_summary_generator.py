"""
Optimized Llama-4 Maverick Summary Generator for Sinhala Articles

This script uses the NVIDIA Llama-4 Maverick API to generate summaries.
It is optimized for speed by using multiple concurrent requests even with a single API key.

Usage:
    1. Set NVIDIA_API_KEY in a .env file or as an environment variable.
    2. Run: python abstractive/llama4_summary_generator.py --input data/test.jsonl --output data/llama4_summaries.jsonl --concurrency 10
"""

import os
import json
import time
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env
load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
# Using Maverick as requested (17B activated / 400B total MoE)
DEFAULT_MODEL = "meta/llama-4-maverick-17b-128e-instruct"

# Recommended System Persona from Llama-4 Model Card
SYSTEM_PROMPT = """You are an expert conversationalist who responds to the best of your ability. 
You are companionable and confident. Organize information thoughtfully. Always avoid templated language. 
You are Llama 4. Your knowledge cutoff date is August 2024. 
Respond in the language the user speaks to you in."""

# Task specific instruction
USER_INSTRUCTION = """
ඔබ සිංහල පුවත් සාරාංශකරණ විශේෂඥයෙකි.

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

def load_api_key():
    """Loads a single NVIDIA API key."""
    key = "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"
    
    return key

def load_processed_urls(output_path: Path) -> set:
    """Returns a set of URLs already processed to allow resuming."""
    processed = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "url" in data:
                        processed.add(data["url"])
                except:
                    continue
    return processed

def worker(api_key, input_queue, output_file, lock, pbar, model_name):
    """Worker thread that processes articles."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }
    
    while True:
        try:
            record = input_queue.get(timeout=5)
        except Empty:
            break

        content = record.get("content", "").strip()
        if not content:
            input_queue.task_done()
            pbar.update(1)
            continue

        prompt = USER_INSTRUCTION.format(content=content)
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1024,
            "temperature": 0,
            "frequency_penalty": 0.2,
            "top_p": 1.0,
            "stream": False
        }
        
        success = False
        retries = 0
        while not success and retries < 5:
            try:
                response = requests.post(INVOKE_URL, headers=headers, json=payload, timeout=60)
                
                if response.status_code == 200:
                    res_json = response.json()
                    summary = res_json['choices'][0]['message']['content'].strip()
                    
                    # Store result
                    result = record.copy()
                    result["llama4_summary"] = summary
                    
                    with lock:
                        with open(output_file, "a", encoding="utf-8") as out_f:
                            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            out_f.flush()
                    
                    success = True
                elif response.status_code == 429:
                    # Rate limit - exponential backoff
                    retries += 1
                    wait_time = 2 ** retries + 5
                    time.sleep(wait_time)
                else:
                    # Log error but don't stop the whole process
                    print(f"\n[Error {response.status_code}]: {response.text[:200]}")
                    retries += 1
                    time.sleep(5)
                    
            except Exception as e:
                retries += 1
                time.sleep(5)

        input_queue.task_done()
        pbar.update(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/test.jsonl")
    parser.add_argument("--output", default="data/llama4_summaries.jsonl")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of parallel requests")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Input file {input_path} not found.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = load_api_key()
    if not api_key:
        print("Error: NVIDIA_API_KEY not found in environment or api_keys.txt.")
        return

    processed_urls = load_processed_urls(output_path)
    
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    to_process = [r for r in records if r.get("url") not in processed_urls]
    
    print(f"Model: {args.model}")
    print(f"Total articles: {len(records)}")
    print(f"Already processed: {len(processed_urls)}")
    print(f"Remaining: {len(to_process)}")
    print(f"Concurrency: {args.concurrency}")
    
    if not to_process:
        print("Everything already processed.")
        return

    input_queue = Queue()
    for r in to_process:
        input_queue.put(r)
    
    lock = threading.Lock()
    
    with tqdm(total=len(to_process), desc="Summarizing") as pbar:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            for _ in range(args.concurrency):
                executor.submit(worker, api_key, input_queue, output_path, lock, pbar, args.model)
            
            input_queue.join()

    print(f"\nComplete. Results: {output_path}")

if __name__ == "__main__":
    main()
