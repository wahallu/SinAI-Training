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


def get_adapter_version(name: str) -> float:
    """Extracts a comparable version number from an adapter folder name
    (e.g. "summarization_sinllama_v06" -> 6.0, "grammar_sinllama_v13" ->
    13.0). Shared by find_latest_adapters() (picks the highest version per
    task at startup) and the summarizer length-conditioning default in
    run_generation() below (picks a prompt format based on the specific
    version actually in use for a given request)."""
    match = re.search(r'_v(\d+)(?:\.(\d+))?', name.lower())
    if match:
        major = int(match.group(1))
        minor = int(match.group(2)) if match.group(2) else 0
        return major + minor * 0.01
    return 0.0


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

    for task, items in adapters_by_task.items():
        if items:
            items.sort(key=lambda x: get_adapter_version(x[0]), reverse=True)
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

try:
    import peft
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda *a, **k: False

    if hasattr(peft.import_utils, "check_min_version"):
        _orig_check = peft.import_utils.check_min_version
        def _safe_check(pkg, ver, *a, **k):
            if "torchao" in str(pkg).lower():
                return False
            try:
                return _orig_check(pkg, ver, *a, **k)
            except Exception:
                return False
        peft.import_utils.check_min_version = _safe_check

    import peft.tuners.lora.torchao
    peft.tuners.lora.torchao.is_torchao_available = lambda *a, **k: False
except Exception:
    pass

try:
    import peft.tuners.lora.model
    peft.tuners.lora.model.is_torchao_available = lambda *a, **k: False
except Exception:
    pass

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
def build_prompt(task: str, text: str, style: Optional[str] = None, length: Optional[str] = None) -> str:
    """
    Pass-through if the client already sent a fully-formed prompt.
    Otherwise wrap raw text in the task's own template (owned by that
    task's module under tasks/ — see task_registry.py).

    `length` (short|medium|long) is only meaningful for the summarizer
    task's v06+ prompt (see work/tasks/summarizer.py); every other task's
    prompt builder ignores unknown kwargs via **_, same as `style` already
    does for non-style tasks.
    """
    if "### Instruction:" in text:
        return text
    spec = TASKS.get(task, TASKS["grammar"])
    return spec.prompt_builder(text, style=style or DEFAULT_STYLE, length=length)


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
    # Only used when task == "summarizer". Options: short | medium | long.
    # If omitted, run_generation() auto-defaults to "medium" when the
    # resolved adapter is v06+ (which requires a length line — see
    # SUMMARIZER_LENGTH_CONDITIONED_FROM_VERSION below), or falls back to
    # the legacy v02-v05 fixed-instruction prompt otherwise.
    length : Optional[str] = None


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

EXTRACTIVE_METHODS = {"tfidf", "textrank", "rake", "yake", "keybert"}

# v06 was the first summarizer adapter trained with length-conditioning
# baked into every training example (see
# summarizer/abstractive/6_train_summarizer.py) — there is no "bare"
# prompt mode in its training data. Update this threshold if a future
# summarizer version changes the prompt format again.
SUMMARIZER_LENGTH_CONDITIONED_FROM_VERSION = 6.0


def _summarizer_adapter_folder_name(active_adapter: str) -> str:
    """Resolves the actual adapter folder name in use for a summarizer
    request, so its version can be checked against
    SUMMARIZER_LENGTH_CONDITIONED_FROM_VERSION.

    /generate always passes the generic PEFT adapter name "summarizer"
    (registered at startup as an alias for whatever find_latest_adapters()
    resolved) — look the real folder name up via ADAPTER_PATHS. /compare
    passes the specific folder name directly (e.g.
    "summarization_sinllama_v06"), since it can load any adapter on the fly
    by name — use it as-is."""
    if active_adapter == "summarizer":
        return os.path.basename(ADAPTER_PATHS.get("summarizer", ""))
    return active_adapter


# ─────────────────────────────────────────────
# MT5 TEACHER MODEL SUPPORT
# ─────────────────────────────────────────────
_MT5_MODEL = None
_MT5_TOKENIZER = None
_MT5_LOCK = threading.Lock()


def get_mt5_path() -> str:
    candidates = [
        "/home/jovyan/summarizer/models/mt5-base",
        "/home/jovyan/work/summarizer/models/mt5-base",
        "/home/jovyan/work/sinllama/models/mt5-base",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../summarizer/models/mt5-base")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "models/mt5-base")),
        "google/mt5-base",
    ]
    for path in candidates:
        if os.path.exists(path) and (
            os.path.isfile(os.path.join(path, "config.json"))
            or os.path.isfile(os.path.join(path, "model.safetensors"))
            or os.path.isfile(os.path.join(path, "pytorch_model.bin"))
        ):
            return path
    return candidates[0]


_MT5_BASE_MODEL = None
_MT5_TOKENIZER = None
_MT5_LOADED_ADAPTERS = {}
_MT5_LOCK = threading.Lock()


def run_mt5_generation(raw_text: str, active_adapter: str = "mt5-base") -> dict:
    global _MT5_BASE_MODEL, _MT5_TOKENIZER, _MT5_LOADED_ADAPTERS
    mt5_path = get_mt5_path()

    with _MT5_LOCK:
        if _MT5_BASE_MODEL is None or _MT5_TOKENIZER is None:
            print(f"[INFO] Initializing raw mT5-base model from: {mt5_path}")
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            _MT5_TOKENIZER = AutoTokenizer.from_pretrained(mt5_path)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _MT5_BASE_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
                mt5_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            ).to(device)
            _MT5_BASE_MODEL.eval()

        target_model = _MT5_BASE_MODEL
        discovered = discover_adapters()
        is_base_request = active_adapter.lower() in ("mt5", "mt5-base", "mt5_base")

        if not is_base_request and active_adapter in discovered:
            adapter_path = discovered[active_adapter]
            if active_adapter not in _MT5_LOADED_ADAPTERS:
                print(f"[INFO] Loading PEFT LoRA adapter for mT5: {active_adapter} from {adapter_path}")
                try:
                    from peft import PeftModel
                    peft_mt5 = PeftModel.from_pretrained(_MT5_BASE_MODEL, adapter_path, adapter_name=active_adapter)
                    peft_mt5.eval()
                    _MT5_LOADED_ADAPTERS[active_adapter] = peft_mt5
                except Exception as exc:
                    print(f"[ERROR] Failed loading PEFT LoRA for mT5 ({active_adapter}): {exc}")

            if active_adapter in _MT5_LOADED_ADAPTERS:
                target_model = _MT5_LOADED_ADAPTERS[active_adapter]

        if is_base_request and _MT5_LOADED_ADAPTERS:
            target_model = list(_MT5_LOADED_ADAPTERS.values())[0]

        adapter_ctx = (
            target_model.disable_adapter()
            if (is_base_request and hasattr(target_model, "disable_adapter"))
            else contextlib.nullcontext()
        )

    input_text = "summarize: " + raw_text
    device = next(target_model.parameters()).device
    inputs = _MT5_TOKENIZER(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad(), adapter_ctx:
        outputs = target_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_length=180,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
        )

    summary = _MT5_TOKENIZER.decode(outputs[0], skip_special_tokens=True).strip()
    from tasks.summarizer import heal_sinhala_text
    summary = heal_sinhala_text(summary)

    input_tokens = len(sinhala_tokenize(raw_text))
    output_tokens = len(sinhala_tokenize(summary))

    return {
        "text": summary if summary and len(summary) >= 2 else raw_text,
        "prompt_len": input_tokens,
        "max_new_tokens": 180,
        "output_tokens": output_tokens,
    }


def run_generation(task_name: str, raw_text: str, style: Optional[str], active_adapter: str, length: Optional[str] = None) -> dict:
    """Shared generation core used by both /generate and /compare.

    `length` is only consumed by the summarizer task (see build_prompt and
    the max_new_tokens call below) — it's threaded through unconditionally
    but every other task ignores it.
    """
    adapter_clean = active_adapter.lower().replace("extractive_", "")
    task_clean = task_name.lower().replace("extractive_", "")

    if adapter_clean.startswith("mt5") or adapter_clean.startswith("summarization_mt5") or task_clean.startswith("mt5"):
        return run_mt5_generation(raw_text, active_adapter=active_adapter)

    if adapter_clean in EXTRACTIVE_METHODS or task_clean in EXTRACTIVE_METHODS or task_name == "extractive":
        method = adapter_clean if adapter_clean in EXTRACTIVE_METHODS else task_clean
        if method not in EXTRACTIVE_METHODS:
            method = "textrank"
        from tasks.extractive import run_extractive
        result_text = run_extractive(method, raw_text, n=3)
        input_tokens = len(sinhala_tokenize(raw_text))
        output_tokens = len(sinhala_tokenize(result_text))
        return {
            "text": result_text,
            "prompt_len": input_tokens,
            "max_new_tokens": 180,
            "output_tokens": output_tokens,
        }

    spec = TASKS.get(task_name, TASKS["grammar"])

    # For the summarizer task, if the caller didn't specify a length, infer
    # one from the actual adapter being used: v06+ has no "bare" prompt mode
    # in its training data (see SUMMARIZER_LENGTH_CONDITIONED_FROM_VERSION
    # above), so leaving length unset would send it the legacy v02-v05
    # instruction it never saw. v02-v05 have no such requirement and keep
    # the old behavior unchanged (length=None -> legacy fixed-instruction
    # line, via tasks/summarizer.py's default).
    effective_length = length
    if task_name == "summarizer" and effective_length is None:
        folder_name = _summarizer_adapter_folder_name(active_adapter)
        if get_adapter_version(folder_name) >= SUMMARIZER_LENGTH_CONDITIONED_FROM_VERSION:
            effective_length = "medium"

    full_prompt = build_prompt(task_name, raw_text, style, effective_length)
    inputs      = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    prompt_len  = inputs["input_ids"].shape[1]

    stop_token_ids = [
        tokenizer.encode(seq, add_special_tokens=False)
        for seq in STOP_SEQUENCES
    ]

    # length is only accepted by tasks/summarizer.py's max_new_tokens — every
    # other task's max_new_tokens has a fixed (raw_text, prompt_token_len)
    # signature with no catch-all kwargs, so it can't be passed unconditionally.
    if task_name == "summarizer":
        max_new_tokens = spec.max_new_tokens(raw_text, prompt_len, length=effective_length)
    else:
        max_new_tokens = spec.max_new_tokens(raw_text, prompt_len)
    stopping_criteria = StoppingCriteriaList([SequenceStop(stop_token_ids, prompt_len)])

    adapter_ctx = model.disable_adapter() if active_adapter == "base" else contextlib.nullcontext()

    with _generation_lock:
        if active_adapter != "base":
            model.set_adapter(active_adapter)
        with adapter_ctx, torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens     = max_new_tokens,
                do_sample          = spec.do_sample,
                temperature        = spec.temperature,
                repetition_penalty = spec.repetition_penalty,
                eos_token_id       = tokenizer.eos_token_id,
                pad_token_id       = tokenizer.eos_token_id,
                stopping_criteria  = stopping_criteria,
                use_cache          = True,
            )
            if spec.no_repeat_ngram_size:
                gen_kwargs["no_repeat_ngram_size"] = spec.no_repeat_ngram_size
            if spec.do_sample:
                gen_kwargs["top_p"] = spec.top_p
            if spec.min_new_tokens:
                gen_kwargs["min_new_tokens"] = spec.min_new_tokens
            outputs = model.generate(**inputs, **gen_kwargs)

    decode_fn = spec.decode or default_decode
    result, output_token_count = decode_fn(tokenizer, outputs, prompt_len)

    return {
        "text"            : result,
        "prompt_len"      : prompt_len,
        "max_new_tokens"  : max_new_tokens,
        "output_tokens"   : output_token_count,
        "length_used"     : effective_length,
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

    gen = run_generation(req.task, req.prompt, req.style, active_adapter=req.task, length=req.length)

    return {
        "response"      : gen["text"] if gen["text"] and len(gen["text"]) >= 2 else req.prompt,
        "task"          : req.task,
        "style"         : req.style if req.task == "style" else None,
        # Reports the length actually used (post server-side default), not
        # just the raw request field, so a caller who omitted it can see
        # what was applied.
        "length"        : gen.get("length_used") if req.task == "summarizer" else None,
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
    # Only used when task == "summarizer". Same auto-default behavior as
    # PromptRequest.length — see run_generation(). Note each adapter in
    # `adapters` gets its own version check, so mixing v02-v05 and v06+
    # adapters in one comparison request correctly gives each the prompt
    # format it was actually trained on.
    length: Optional[str] = None
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
    if "mt5" in name_lower:
        return "mt5"
    elif name_lower in EXTRACTIVE_METHODS or name_lower.startswith("extractive_"):
        return "extractive"
    elif name_lower.startswith("grammer_") or name_lower.startswith("grammar_"):
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
    mt5_adapters = []
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
        elif cat == "mt5":
            mt5_adapters.append(name)
        else:
            custom_adapters.append(name)

    grammar_adapters.sort()
    headline_adapters.sort()
    style_adapters.sort()
    summarizer_adapters.sort()
    mt5_adapters = sorted(list(set(mt5_adapters + ["mt5-base"])))
    custom_adapters.sort()

    extractive_adapters = ["tfidf", "textrank", "rake", "yake", "keybert"]

    return {
        "adapters": {
            "grammar": grammar_adapters,
            "headline": headline_adapters,
            "style": style_adapters,
            "summarizer": summarizer_adapters,
            "extractive": extractive_adapters,
            "mt5": mt5_adapters,
            "custom": custom_adapters
        },
        "loaded_in_gpu": sorted(list(LOADED_ADAPTERS)),
        "mode": "gpu"
    }


@app.post("/compare")
async def compare_models(req: CompareRequest):
    """Runs evaluation inference on all requested adapters, base model, extractive methods, and mT5."""
    discovered = discover_adapters()
    results = []

    for adapter_name in req.adapters:
        is_extractive = (
            adapter_name.lower() in EXTRACTIVE_METHODS
            or adapter_name.lower().replace("extractive_", "") in EXTRACTIVE_METHODS
            or adapter_name.lower().startswith("extractive_")
        )
        is_mt5 = "mt5" in adapter_name.lower()

        if not is_extractive and not is_mt5 and adapter_name != "base" and adapter_name not in discovered:
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

        # Load dynamically if PEFT adapter and not loaded
        if not is_extractive and not is_mt5 and adapter_name != "base" and adapter_name not in LOADED_ADAPTERS:
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
            gen = run_generation(req.task, req.input_text, req.style, active_adapter=adapter_name, length=req.length)
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
