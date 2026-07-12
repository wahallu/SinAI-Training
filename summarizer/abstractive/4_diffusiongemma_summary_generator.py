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

load_dotenv()

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "google/diffusiongemma-26b-a4b-it"

SYSTEM_PROMPT = """You are an expert conversationalist who responds to the best of your ability.
You are companionable and confident. Organize information thoughtfully.
Respond in the language the user speaks to you in."""

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
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not found in environment.")
    return api_key

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

def worker(api_key, input_queue, output_file, lock, pbar, model_name):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    while True:
        try:
            record = input_queue.get(timeout=5)
        except Empty:
            break

        try:
            content = record.get("content", "").strip()
            if not content:
                with lock:
                    with open(output_file, "a", encoding="utf-8") as out_f:
                        error_record = record.copy()
                        error_record["error"] = "empty content"
                        out_f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                        out_f.flush()
                pbar.update(1)
                continue

            prompt = USER_INSTRUCTION.format(content=content)

            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 512,
                "temperature": 0.0,
                "top_p": 1.0,
                "stream": False,
            }

            success = False
            retries = 0

            while not success and retries < 5:
                try:
                    response = requests.post(
                        INVOKE_URL,
                        headers=headers,
                        json=payload,
                        timeout=120,
                    )

                    if response.status_code == 200:
                        res_json = response.json()

                        # Try a few possible response shapes
                        summary = ""
                        try:
                            message = res_json["choices"][0]["message"]

                            summary = message.get("content")
                            
                            if summary is None:
                                summary = ""
                            
                            summary = summary.strip()
                        except Exception:
                            summary = str(res_json)

                        result = record.copy()
                        result["diffusiongemma_summary"] = summary

                        with lock:
                            with open(output_file, "a", encoding="utf-8") as out_f:
                                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                                out_f.flush()
                                os.fsync(out_f.fileno())

                        success = True

                    else:
                        print(f"\n[Error {response.status_code}] {response.text[:500]}")
                        retries += 1
                        time.sleep((2 ** retries) + 2)

                except Exception as e:
                    print(f"\n[Exception] {repr(e)}")
                    retries += 1
                    time.sleep(3)

            if not success:
                with lock:
                    with open(output_file, "a", encoding="utf-8") as out_f:
                        error_record = record.copy()
                        error_record["status"] = "failed"
                        error_record["error"] = f"failed after {retries} retries"
                        out_f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                        out_f.flush()
                        os.fsync(out_f.fileno())

        finally:
            input_queue.task_done()
            pbar.update(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/home/jovyan/summarizer/data/train.jsonl")
    parser.add_argument("--output", default="/home/jovyan/summarizer/data/4_diffusiongemma_summaries.jsonl")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.touch(exist_ok=True)
    print(f"Saving to: {output_path.resolve()}")

    if not input_path.exists():
        print(f"Input file {input_path} not found.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = load_api_key()
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