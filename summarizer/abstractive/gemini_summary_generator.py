"""
Multi-Key Gemini Summary Generator for Sinhala Articles

This script uses multiple Google Gemini API keys in parallel to generate
abstractive summaries for the dataset at high speed.

Usage:
    1. Create a file named 'api_keys.txt' with one API key per line.
    2. Run: python abstractive/gemini_summary_generator.py --input data/test.jsonl --output data/gemini_summaries.jsonl
"""

import os
import json
import time
import argparse
from pathlib import Path
from google import genai
from tqdm import tqdm
from dotenv import load_dotenv
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env if present
load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME = "gemini-flash-latest"
RPM_LIMIT = 14  # Per key
DELAY_SECONDS = 60 / RPM_LIMIT

# Sinhala Prompt enforcing 10-30% length
PROMPT_TEMPLATE = """ඔබ සිංහල පුවත් ලිපි සාරාංශ කිරීමේ විශේෂඥයෙකි.
පහත සිංහල පුවත් ලිපිය කියවා, ලිපියේ ප්‍රධාන කරුණු ඇතුළත් සාරාංශයක් ලියන්න.
සාරාංශය ලිපියේ දිග මෙන් 10% ත් 30% ත් අතර විය යුතුය.
සාරාංශය පමණක් ලබා දෙන්න.

Article:
{content}
"""

def load_api_keys(file_path="api_keys.txt"):
    """Loads API keys from a file or environment variable."""
    keys = []
    if Path(file_path).exists():
        with open(file_path, "r") as f:
            keys = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    if not keys:
        single_key = os.getenv("GEMINI_API_KEY")
        if single_key:
            keys = [single_key]
    
    return keys

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

def worker(api_key, input_queue, output_file, lock, pbar):
    """Worker thread that processes articles using a specific API key."""
    client = genai.Client(api_key=api_key)
    
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

        prompt = PROMPT_TEMPLATE.format(content=content)
        
        success = False
        retries = 0
        while not success and retries < 3:
            try:
                # Add delay to respect RPM before request
                start_time = time.time()
                
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt
                )
                summary = response.text.strip()
                
                # Store result
                result = record.copy()
                result["gemini_summary"] = summary
                
                with lock:
                    with open(output_file, "a", encoding="utf-8") as out_f:
                        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out_f.flush()
                
                success = True
                
                # Wait remaining time to respect RPM
                elapsed = time.time() - start_time
                if elapsed < DELAY_SECONDS:
                    time.sleep(DELAY_SECONDS - elapsed)
                    
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    retries += 1
                    wait_time = 60 * retries
                    time.sleep(wait_time)
                elif "404" in err_msg:
                    # Model not found or something fatal
                    print(f"\nFatal error with key {api_key[:8]}...: {e}")
                    input_queue.put(record) # Put back
                    input_queue.task_done()
                    return 
                else:
                    retries += 1
                    time.sleep(5)

        input_queue.task_done()
        pbar.update(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/test.jsonl")
    parser.add_argument("--output", default="data/gemini_summaries.jsonl")
    parser.add_argument("--keys", default="api_keys.txt", help="File containing API keys (one per line)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Input file {input_path} not found.")
        return

    api_keys = load_api_keys(args.keys)
    if not api_keys:
        print("No API keys found. Please provide api_keys.txt or set GEMINI_API_KEY.")
        return

    print(f"Loaded {len(api_keys)} API keys.")
    
    processed_urls = load_processed_urls(output_path)
    
    # Load input records
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    
    # Filter out already processed
    to_process = [r for r in records if r.get("url") not in processed_urls]
    
    print(f"Total articles: {len(records)}")
    print(f"Already processed: {len(processed_urls)}")
    print(f"Remaining: {len(to_process)}")
    
    if not to_process:
        print("Everything already processed.")
        return

    # Initialize queue and lock
    input_queue = Queue()
    for r in to_process:
        input_queue.put(r)
    
    lock = threading.Lock()
    
    with tqdm(total=len(to_process), desc="Summarizing") as pbar:
        with ThreadPoolExecutor(max_workers=len(api_keys)) as executor:
            for key in api_keys:
                executor.submit(worker, key, input_queue, output_path, lock, pbar)
            
            input_queue.join()

    print(f"\nGeneration complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
