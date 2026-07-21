import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"]  = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import StoppingCriteria, StoppingCriteriaList
from typing import Optional
import contextlib
import threading
import torch
import time
import unicodedata
from collections import Counter

import re

from task_registry import TASKS, VALID_STYLES, DEFAULT_STYLE

app = FastAPI()

model_path = "/home/jovyan/work/sinllama/models/SinLLaMA-merged-base"
ADAPTERS_DIR = "/home/jovyan/work/sinllama/models/adapters"


def find_latest_adapters() -> dict[str, str]:
    """
    Scans ADAPTERS_DIR and returns a dictionary of the latest adapter paths for each task.
    Automatically matches versions by checking folder names and sorting by version numbers.
    """
    fallback_paths = {
        "grammar"   : "/home/jovyan/work/sinllama/models/adapters/grammar_sinllama_v13",
        "headline"  : "/home/jovyan/work/sinllama/models/adapters/headline_sinllama_v17",
        "style"     : "/home/jovyan/work/sinllama/models/adapters/style_sinllama_v07",
        "summarizer": "/home/jovyan/work/sinllama/models/adapters/summarization_sinllama_v04",
    }

    target_dir = ADAPTERS_DIR
    if not os.path.exists(target_dir):
        # Fallback to local workspace relative path if server path is missing
        workspace_fallback = os.path.abspath(os.path.join(os.path.dirname(__file__), "models/adapters"))
        if os.path.exists(workspace_fallback):
            target_dir = workspace_fallback
        else:
            return fallback_paths

    adapters_by_task = {
        "grammar": [],
        "headline": [],
        "style": [],
        "summarizer": []
    }

    try:
        for name in os.listdir(target_dir):
            path = os.path.join(target_dir, name)
            if not os.path.isdir(path) or not os.path.isfile(os.path.join(path, "adapter_config.json")):
                continue

            name_lower = name.lower()
            if name_lower.startswith("grammer_") or name_lower.startswith("grammar_"):
                adapters_by_task["grammar"].append((name, path))
            elif name_lower.startswith("headline_"):
                adapters_by_task["headline"].append((name, path))
            elif name_lower.startswith("style_"):
                adapters_by_task["style"].append((name, path))
            elif name_lower.startswith("summarization_") or name_lower.startswith("summarizer_"):
                adapters_by_task["summarizer"].append((name, path))
    except Exception as e:
        print(f"Error listing adapters directory: {e}")
        return fallback_paths

    latest_paths = {}

    def get_version(name: str) -> float:
        # Match _v12, _v13, _v04, etc.
        match = re.search(r'_v(\d+)(?:\.(\d+))?', name.lower())
        if match:
            major = int(match.group(1))
            minor = int(match.group(2)) if match.group(2) else 0
            return major + minor * 0.01
        return 0.0

    for task, items in adapters_by_task.items():
        if items:
            items.sort(key=lambda x: get_version(x[0]), reverse=True)
            latest_paths[task] = items[0][1]
            print(f"[INFO] Automatically selected latest {task} adapter: {items[0][0]}")
        else:
            latest_paths[task] = fallback_paths[task]
            print(f"[INFO] No {task} adapters found in directory. Using fallback: {fallback_paths[task]}")

    return latest_paths


# ─────────────────────────────────────────────
# TASK ADAPTER PATHS (automatically resolved to latest checkpoints)
# ─────────────────────────────────────────────
ADAPTER_PATHS: dict[str, str] = find_latest_adapters()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

quant_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_compute_dtype    = torch.bfloat16,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_use_double_quant = True,
)

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config  = quant_config,
    dtype          = torch.bfloat16,
    local_files_only     = True,
    device_map           = "auto",
    attn_implementation  = "eager",
)


def _validate_adapter_paths(paths: dict[str, str]) -> None:
    missing = [
        f"  - {task}: {path}"
        for task, path in paths.items()
        if not os.path.isfile(os.path.join(path, "adapter_config.json"))
    ]
    if missing:
        raise RuntimeError(
            "SinLlama startup aborted — missing adapter path(s):\n"
            + "\n".join(missing)
            + "\nFix ADAPTER_PATHS at the top of serve_sinai.py."
        )


_validate_adapter_paths(ADAPTER_PATHS)

from peft import PeftModel

_adapter_names = list(ADAPTER_PATHS.keys())
model = PeftModel.from_pretrained(
    model, ADAPTER_PATHS[_adapter_names[0]], adapter_name=_adapter_names[0]
)
for _name in _adapter_names[1:]:
    model.load_adapter(ADAPTER_PATHS[_name], adapter_name=_name)

model.set_adapter(_adapter_names[0])   # arbitrary initial default; every request sets it explicitly
LOADED_ADAPTERS = set(_adapter_names)  # used by /health, /tasks

# NOTE: This server loads the base model via plain transformers (not
# Unsloth's FastLanguageModel), specifically to avoid Unsloth's fused
# fast-decode kernels. Those kernels index each LoRA-wrapped module's
# lora_A[active_adapter] / lora_B[active_adapter] directly with no fallback,
# and are activated unconditionally on any cached model.generate() call once
# a model is loaded through FastLanguageModel — calling (or skipping)
# FastLanguageModel.for_inference() has no effect on this, contrary to what
# an earlier version of this file assumed. Our adapters don't all target the
# same modules — grammar_sinllama_v13 only targets attention projections
# (q/k/v/o_proj), while headline/style/summarizer also target the MLP
# projections (gate/up/down_proj). Those MLP Linear layers get wrapped once
# and shared across every adapter that touches them, so switching the active
# adapter to "grammar" hit a KeyError inside Unsloth's fast MLP kernel
# (fast_swiglu_inference -> get_lora_parameters_bias) because "grammar" was
# never added to that layer's lora_A/lora_B dict. This is a known upstream
# limitation (unslothai/unsloth#2322), not something fixable from adapter
# config alone. Standard PEFT eager forward (what plain transformers uses)
# tolerates this correctly — it skips LoRA entirely on a module an adapter
# doesn't target. This trades Unsloth's fused-kernel decode speed for
# correctness across all four structurally-different adapters.
model.eval()


# ─────────────────────────────────────────────
# PROMPT DISPATCH
# ─────────────────────────────────────────────
def build_prompt(task: str, text: str, style: Optional[str] = None) -> str:
    """
    Pass-through if the client already sent a fully-formed prompt.
    Otherwise wrap raw text in the task's own template (owned by that
    task's module under tasks/ — see task_registry.py).
    """
    if "### Instruction:" in text:
        return text
    spec = TASKS.get(task, TASKS["grammar"])
    return spec.prompt_builder(text, style=style or DEFAULT_STYLE)


# ─────────────────────────────────────────────
# STOPPING CRITERIA
# ─────────────────────────────────────────────
class SequenceStop(StoppingCriteria):
    def __init__(self, stop_token_ids: list[list[int]], prompt_len: int):
        self.prompt_len     = prompt_len
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        generated = input_ids[0][self.prompt_len:].tolist()
        if not generated:
            return False
        for stop in self.stop_token_ids:
            if len(generated) >= len(stop) and generated[-len(stop):] == stop:
                return True
        return False


STOP_SEQUENCES = ["###", "<|eot_id|>", "<|end_of_text|>"]


def default_decode(tokenizer, outputs, prompt_len: int) -> tuple[str, int]:
    """Used by any task that doesn't register its own TaskSpec.decode."""
    new_tokens = outputs[0][prompt_len:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    for seq in STOP_SEQUENCES:
        if result.endswith(seq):
            result = result[:-len(seq)].strip()
    return result, len(new_tokens)


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class PromptRequest(BaseModel):
    prompt : str
    task   : str            = "grammar"
    # Only used when task == "style".
    # Options: formal | sports | youth | editorial | feature
    # Defaults to "formal" if omitted.
    style  : Optional[str] = None


# ─────────────────────────────────────────────
# CORE GENERATE
# ─────────────────────────────────────────────
# Guards `set_adapter()` + `model.generate()` as one atomic unit. Today this
# runs synchronously inside async route handlers, which already serializes
# all requests on the event loop — this lock makes that guarantee explicit
# and keeps it correct if that ever moves onto a threadpool.
# threading.Lock (not asyncio.Lock) since it must hold across real OS
# threads either way.
_generation_lock = threading.Lock()


def run_generation(task_name: str, raw_text: str, style: Optional[str], active_adapter: str) -> dict:
    """Shared generation core used by both /generate and /compare.

    `task_name` selects which TaskSpec governs prompt format, token budget,
    repetition_penalty, and decode post-processing — one of the four owned
    task modules, or "base" for the raw-instruction playground format.
    `active_adapter` is the literal PEFT adapter to activate via
    model.set_adapter() — "base" disables all adapters. These are decoupled
    because /compare can run any task's prompt against the "base" (no
    adapter) model to measure the fine-tune's lift, or against any
    discovered adapter version for that task category.
    """
    spec = TASKS.get(task_name, TASKS["grammar"])

    full_prompt = build_prompt(task_name, raw_text, style)
    inputs      = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    prompt_len  = inputs["input_ids"].shape[1]

    stop_token_ids = [
        tokenizer.encode(seq, add_special_tokens=False)
        for seq in STOP_SEQUENCES
    ]

    max_new_tokens    = spec.max_new_tokens(raw_text, prompt_len)
    stopping_criteria = StoppingCriteriaList([SequenceStop(stop_token_ids, prompt_len)])

    adapter_ctx = model.disable_adapter() if active_adapter == "base" else contextlib.nullcontext()

    with _generation_lock:
        if active_adapter != "base":
            model.set_adapter(active_adapter)
        with adapter_ctx, torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens     = max_new_tokens,
                do_sample          = False,
                temperature        = 1.0,
                repetition_penalty = spec.repetition_penalty,
                eos_token_id       = tokenizer.eos_token_id,
                pad_token_id       = tokenizer.eos_token_id,
                stopping_criteria  = stopping_criteria,
                use_cache          = True,
            )
            if spec.no_repeat_ngram_size:
                gen_kwargs["no_repeat_ngram_size"] = spec.no_repeat_ngram_size
            outputs = model.generate(**inputs, **gen_kwargs)

    decode_fn = spec.decode or default_decode
    result, output_token_count = decode_fn(tokenizer, outputs, prompt_len)

    return {
        "text"            : result,
        "prompt_len"      : prompt_len,
        "max_new_tokens"  : max_new_tokens,
        "output_tokens"   : output_token_count,
    }


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
@app.post("/generate")
async def generate(req: PromptRequest):
    if req.task not in TASKS:
        req.task = "grammar"

    if req.task == "style":
        if req.style and req.style not in VALID_STYLES:
            raise HTTPException(
                status_code = 422,
                detail      = {
                    "error"  : f"Unknown style '{req.style}'.",
                    "valid"  : sorted(VALID_STYLES),
                    "default": DEFAULT_STYLE,
                },
            )
        if not req.style:
            req.style = DEFAULT_STYLE

    gen = run_generation(req.task, req.prompt, req.style, active_adapter=req.task)

    return {
        "response"      : gen["text"] if gen["text"] and len(gen["text"]) >= 2 else req.prompt,
        "task"          : req.task,
        "style"         : req.style if req.task == "style" else None,
        "input_tokens"  : gen["prompt_len"],
        "max_cap_used"  : gen["max_new_tokens"],
        "output_tokens" : gen["output_tokens"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "adapters_loaded": sorted(list(LOADED_ADAPTERS))}


@app.get("/tasks")
def list_tasks():
    """Returns supported tasks, available styles, and rendered prompt examples."""
    sample = "කොළඹ කොටස් වෙළෙඳපොළ මිල දර්ශකවල පසුබැස්මක් අද දිනයේ දී වාර්තා විය."
    return {
        "tasks": list(TASKS.keys()),
        "adapters": {task: os.path.basename(path) for task, path in ADAPTER_PATHS.items()},
        "styles": {
            "available"  : sorted(VALID_STYLES),
            "default"    : DEFAULT_STYLE,
            "applies_to" : "style task only",
        },
        "prompt_examples": {
            "grammar"   : TASKS["grammar"].prompt_builder(sample),
            "headline"  : TASKS["headline"].prompt_builder(sample),
            "summarizer": TASKS["summarizer"].prompt_builder(sample),
            "base"      : TASKS["base"].prompt_builder(sample),
            **{
                f"style/{s}": TASKS["style"].prompt_builder(sample, style=s)
                for s in sorted(VALID_STYLES)
            },
        },
    }


# ─────────────────────────────────────────────
# MODEL COMPARISON INTEGRATION (Merged for shared VRAM and single ngrok tunnel)
# ─────────────────────────────────────────────
from typing import List

# NLTK for Sentence GLEU
try:
    from nltk.translate.gleu_score import sentence_gleu
    HAS_GLEU = True
except ImportError:
    HAS_GLEU = False
    print("[WARNING] nltk not installed. Running without NLTK GLEU.")


class CompareRequest(BaseModel):
    input_text: str
    adapters: List[str]
    task: str = "grammar"
    style: Optional[str] = None
    reference_text: Optional[str] = None


def discover_adapters() -> dict[str, str]:
    """Scans ADAPTERS_DIR for folders containing adapter_config.json."""
    adapters = {}
    if not os.path.exists(ADAPTERS_DIR):
        workspace_fallback = os.path.abspath(os.path.join(os.path.dirname(__file__), "models/adapters"))
        if os.path.exists(workspace_fallback):
            target_dir = workspace_fallback
        else:
            return adapters
    else:
        target_dir = ADAPTERS_DIR

    try:
        for name in os.listdir(target_dir):
            path = os.path.join(target_dir, name)
            if os.path.isdir(path) and os.path.isfile(os.path.join(path, "adapter_config.json")):
                adapters[name] = path
    except Exception as e:
        print(f"Error scanning adapters directory: {e}")

    return adapters


def get_adapter_category(name: str) -> str:
    """Classifies an adapter by its prefix."""
    name_lower = name.lower()
    if name_lower.startswith("grammer_") or name_lower.startswith("grammar_"):
        return "grammar"
    elif name_lower.startswith("headline_"):
        return "headline"
    elif name_lower.startswith("style_"):
        return "style"
    elif name_lower.startswith("summarization_") or name_lower.startswith("summarizer_"):
        return "summarizer"
    else:
        return "custom"


# ─────────────────────────────────────────────
# SINHALA NLP METRICS
# ─────────────────────────────────────────────
def sinhala_tokenize(text: str) -> List[str]:
    """Character-level tokenizer for Sinhala (Unicode grapheme clusters)."""
    tokens = []
    chars  = list(text)
    i = 0
    while i < len(chars):
        cluster = chars[i]
        i += 1
        while i < len(chars) and unicodedata.combining(chars[i]):
            cluster += chars[i]
            i += 1
        if cluster.strip():
            tokens.append(cluster)
    return tokens


def token_prf(pred: str, ref: str) -> tuple:
    pred_toks = sinhala_tokenize(pred)
    ref_toks  = sinhala_tokenize(ref)
    pred_cnt  = Counter(pred_toks)
    ref_cnt   = Counter(ref_toks)
    common    = sum((pred_cnt & ref_cnt).values())
    p = common / len(pred_toks) if pred_toks else 0.0
    r = common / len(ref_toks)  if ref_toks  else 0.0
    f = 2 * p * r / (p + r)    if (p + r)   else 0.0
    return p, r, f


def char_f1(pred: str, ref: str) -> float:
    pc = Counter(pred)
    rc = Counter(ref)
    common = sum((pc & rc).values())
    p = common / len(pred) if pred else 0.0
    r = common / len(ref)  if ref  else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def gleu_score(pred: str, ref: str) -> float:
    if not HAS_GLEU:
        return 0.0
    hyp  = sinhala_tokenize(pred)
    refs = [sinhala_tokenize(ref)]
    return sentence_gleu(refs, hyp)


def rouge_scores(pred: str, ref: str) -> dict:
    def ngrams(tokens, n):
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

    def lcs_length(a, b):
        m, n = len(a), len(b)
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(prev[j], curr[j-1])
            prev, curr = curr, [0] * (n + 1)
        return prev[n]

    pred_toks = sinhala_tokenize(pred)
    ref_toks  = sinhala_tokenize(ref)
    if not pred_toks or not ref_toks:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    p1 = ngrams(pred_toks, 1)
    r1 = ngrams(ref_toks,  1)
    c1 = sum((p1 & r1).values())
    prec1 = c1 / len(pred_toks)
    rec1  = c1 / len(ref_toks)
    f1_1  = 2*prec1*rec1/(prec1+rec1) if (prec1+rec1) else 0.0

    p2 = ngrams(pred_toks, 2)
    r2 = ngrams(ref_toks,  2)
    c2 = sum((p2 & r2).values())
    prec2 = c2 / max(len(pred_toks)-1, 1)
    rec2  = c2 / max(len(ref_toks)-1,  1)
    f1_2  = 2*prec2*rec2/(prec2+rec2) if (prec2+rec2) else 0.0

    lcs = lcs_length(pred_toks, ref_toks)
    precL = lcs / len(pred_toks)
    recL  = lcs / len(ref_toks)
    f1_L  = 2*precL*recL/(precL+recL) if (precL+recL) else 0.0

    return {"rouge1": f1_1, "rouge2": f1_2, "rougeL": f1_L}


def over_correction_rate(pred: str, inp: str, ref: str) -> bool:
    return (inp.strip() == ref.strip()) and (pred.strip() != ref.strip())


# ─────────────────────────────────────────────
# COMPARISON ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/adapters")
def get_adapters():
    """Returns all dynamically discovered adapters grouped by task type."""
    discovered = discover_adapters()

    grammar_adapters = []
    headline_adapters = []
    style_adapters = []
    summarizer_adapters = []
    custom_adapters = []

    for name in discovered.keys():
        cat = get_adapter_category(name)
        if cat == "grammar":
            grammar_adapters.append(name)
        elif cat == "headline":
            headline_adapters.append(name)
        elif cat == "style":
            style_adapters.append(name)
        elif cat == "summarizer":
            summarizer_adapters.append(name)
        else:
            custom_adapters.append(name)

    grammar_adapters.sort()
    headline_adapters.sort()
    style_adapters.sort()
    summarizer_adapters.sort()
    custom_adapters.sort()

    return {
        "adapters": {
            "grammar": grammar_adapters,
            "headline": headline_adapters,
            "style": style_adapters,
            "summarizer": summarizer_adapters,
            "custom": custom_adapters
        },
        "loaded_in_gpu": sorted(list(LOADED_ADAPTERS)),
        "mode": "gpu"
    }


@app.post("/compare")
async def compare_models(req: CompareRequest):
    """Runs evaluation inference on all requested adapters and base model."""
    discovered = discover_adapters()
    results = []

    for adapter_name in req.adapters:
        # Resolve path
        if adapter_name != "base" and adapter_name not in discovered:
            discovered = discover_adapters()
            if adapter_name not in discovered:
                results.append({
                    "adapter_name": adapter_name,
                    "category": get_adapter_category(adapter_name),
                    "output_text": f"Error: Adapter '{adapter_name}' not found on server.",
                    "latency_ms": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "throughput_tokens_per_sec": 0,
                    "metrics": {}
                })
                continue

        # Load dynamically if not loaded
        if adapter_name != "base" and adapter_name not in LOADED_ADAPTERS:
            adapter_path = discovered[adapter_name]
            print(f"[INFO] Loading adapter on-the-fly: {adapter_name}")
            try:
                with _generation_lock:
                    model.load_adapter(adapter_path, adapter_name=adapter_name)
                    LOADED_ADAPTERS.add(adapter_name)
            except Exception as e:
                results.append({
                    "adapter_name": adapter_name,
                    "category": get_adapter_category(adapter_name),
                    "output_text": f"Error loading adapter: {str(e)}",
                    "latency_ms": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "throughput_tokens_per_sec": 0,
                    "metrics": {}
                })
                continue

        start_time = time.perf_counter()

        try:
            gen = run_generation(req.task, req.input_text, req.style, active_adapter=adapter_name)
            output_text  = gen["text"] if gen["text"] and len(gen["text"]) >= 2 else req.input_text
            input_tokens = gen["prompt_len"]
            output_tokens = gen["output_tokens"]

        except Exception as e:
            output_text = f"Inference Error: {str(e)}"
            input_tokens = 0
            output_tokens = 0

        latency_sec = time.perf_counter() - start_time
        latency_ms = latency_sec * 1000
        throughput = output_tokens / latency_sec if latency_sec > 0 else 0

        metrics = {}
        if req.reference_text and req.reference_text.strip():
            ref = req.reference_text
            p, r, f1 = token_prf(output_text, ref)
            cf1 = char_f1(output_text, ref)
            gleu = gleu_score(output_text, ref)
            rouge = rouge_scores(output_text, ref)
            over_corr = over_correction_rate(output_text, req.input_text, ref)

            metrics = {
                "token_precision": round(p, 4),
                "token_recall": round(r, 4),
                "token_f1": round(f1, 4),
                "char_f1": round(cf1, 4),
                "gleu": round(gleu, 4),
                "rouge1": round(rouge["rouge1"], 4),
                "rouge2": round(rouge["rouge2"], 4),
                "rougeL": round(rouge["rougeL"], 4),
                "over_correction": over_corr,
                "exact_match": output_text.strip() == ref.strip()
            }

        results.append({
            "adapter_name": adapter_name,
            "category": get_adapter_category(adapter_name) if adapter_name != "base" else "base",
            "output_text": output_text,
            "latency_ms": round(latency_ms, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "throughput_tokens_per_sec": round(throughput, 2),
            "metrics": metrics
        })

    return results
