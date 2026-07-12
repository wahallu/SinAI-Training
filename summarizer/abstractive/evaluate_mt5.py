"""
Inference script for fine-tuned mT5 Sinhala Summarizer.

Usage:
    python abstractive/evaluate_mt5.py --model models/sinhala-mt5/checkpoint-81560 --input data/test.jsonl --output data/mt5_preds.jsonl --limit 10
"""

import torch
import json
import argparse
from pathlib import Path
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/sinhala-mt5/checkpoint-81560", help="Path to fine-tuned model")
    parser.add_argument("--input", default="data/test.jsonl", help="Input JSONL file")
    parser.add_argument("--output", default="data/mt5_preds.jsonl", help="Output JSONL file")
    parser.add_argument("--limit", type=int, default=10, help="Number of articles to process")
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model and tokenizer from {args.model}...")
    tokenizer = T5Tokenizer.from_pretrained(args.model)
    model = MT5ForConditionalGeneration.from_pretrained(args.model).to(device)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file {input_path} not found.")
        return

    # Load records
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    
    # Take sample
    to_process = records[:args.limit]
    print(f"Processing {len(to_process)} articles...")

    results = []
    for record in tqdm(to_process):
        content = record.get("content", "").strip()
        if not content:
            continue

        # Prefix for T5/mT5 task
        input_text = "summarize: " + content
        
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=args.max_source_length
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=args.max_target_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        res = record.copy()
        res["mt5_summary"] = summary
        results.append(res)

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"\nInference complete. Results saved to {args.output}")
    
    # Print sample comparison
    print("\n" + "="*50)
    print("SAMPLE COMPARISON")
    print("="*50)
    for i, res in enumerate(results[:3]):
        print(f"\n[{i+1}] Title: {res['title']}")
        print(f"mT5 Summary: {res['mt5_summary']}")
        print("-" * 30)

if __name__ == "__main__":
    main()
