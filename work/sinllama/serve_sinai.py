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

app = FastAPI()

model_path = "/home/jovyan/work/sinllama/models/SinLLaMA-merged-base"

# ─────────────────────────────────────────────
# TASK ADAPTER PATHS  (edit versions here only)
# Keys must exactly match the `task` strings used throughout this file
# (PROMPT_BUILDERS, compute_max_tokens) — that identity is what lets
# model.set_adapter(task) work with zero new request/response fields.
# ─────────────────────────────────────────────
ADAPTER_PATHS: dict[str, str] = {
    "grammar"   : "/home/jovyan/work/sinllama/models/adapters/grammar_sinllama_v13",
    "headline"  : "/home/jovyan/work/sinllama/models/adapters/headline_sinllama_v17",
    "style"     : "/home/jovyan/work/sinllama/models/adapters/style_sinllama_v07",
    "summarizer": "/home/jovyan/work/sinllama/models/adapters/summarization_sinllama_v04",
}

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
# STYLE DEFINITIONS
# Each entry is a Sinhala instruction describing
# exactly how the model should rewrite the article.
# To add a new style, just add a key here — nothing
# else in the file needs to change.
# ─────────────────────────────────────────────
STYLE_INSTRUCTIONS: dict[str, str] = {
    "formal": (
        "පහත සිංහල පාඨය නිල හා වෘත්තීය පුවත් ශෛලියට (formal news style) නැවත ලියන්න.\n"
        "සරල, නිවැරදි සිංහල භාෂාව භාවිත කරන්න. "
        "ආත්මීය හෝ අනවශ්‍ය සංවාදාත්මක වචන ඉවත් කරන්න."
    ),
    "sports": (
        "පහත සිංහල පාඨය ජීවමාන හා ශක්තිමත් ක්‍රීඩා පුවත් ශෛලියට (sports journalism style) නැවත ලියන්න.\n"
        "ක්‍රියාශීලී ක්‍රියා පද, ශක්තිමත් ගොනු ශීර්ෂ, හා ක්‍රීඩා ශබ්ද කෝෂය භාවිත කරන්න."
    ),
    "youth": (
        "පහත සිංහල පාඨය තරුණ පාඨකයන් ඉලක්ක කරගත් සරල, ගතිකාරී ශෛලියකට (youth/casual style) නැවත ලියන්න.\n"
        "සරල වාක්‍ය, කෙළින්ම කතා කරන ලෙස, හා නවීන සිංහල ප්‍රකාශන භාවිත කරන්න. "
        "ඉතා කාර්යාල ලෙසට ලියූ ශෛලිය ඉවත් කරන්න."
    ),
    "editorial": (
        "පහත සිංහල පාඨය ගැඹුරු විශ්ලේෂණාත්මක සංස්කාරකීය ශෛලියකට (editorial/opinion style) නැවත ලියන්න.\n"
        "කරුණු ඉදිරිපත් කරමින් විශ්ලේෂණය, ආකල්ප, හා ගැඹුරු සිතුවිලි ඇතුළත් කරන්න. "
        "ශක්තිමත් හා ඒත්තු ගැන්වෙන ශෛලිය භාවිත කරන්න."
    ),
    "feature": (
        "පහත සිංහල පාඨය කතා කරන ආකාරයේ feature ලිපි ශෛලියකට (feature writing style) නැවත ලියන්න.\n"
        "දෘශ්‍යමාන භාෂාව, ජීවිත කතා ශෛලිය, හා කාව්‍යාත්මක පාඨ භාවිත කරන්න. "
        "කරුණු ඉදිරිපත් කිරීම සිත් ඇදගන්නා සුළු වේ."
    ),
}

DEFAULT_STYLE = "formal"
VALID_STYLES  = set(STYLE_INSTRUCTIONS.keys())


# ─────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────
def prompt_grammar(text: str, **_) -> str:
    return (
        "### Instruction:\n"
        "ඔබ සිංහල භාෂා විශේෂඥයෙකි.\n"
        "පහත සිංහල පාඨයේ ඇති වාකරණ දෝෂ, අක්ෂර වින්‍යාස දෝෂ සහ විරාම ලකුණු දෝෂ නිවැරදි කරන්න.\n"
        "නිවැරදි කළ පාඨය පමණක් ලියන්න. වෙනත් කිසිදු පැහැදිලි කිරීමක් එකතු නොකරන්න.\n\n"
        f"Text:\n{text}\n\n"
        "### Response:\n"
    )


def prompt_headline(text: str, **_) -> str:
    return (
        "### Instruction:\n"
        "ඔබ සිංහල පුවත් සංස්කාරකයෙකි.\n"
        "පහත සිංහල පුවත් ලිපිය කියවා, ලිපිය සඳහා සංක්ෂිප්ත හා ආකර්ශනීය ශීර්ෂ පාඨයක් (headline) ලියන්න.\n"
        "ශීර්ෂ පාඨය වචන 10කට නොඉක්මවිය යුතුය.\n\n"
        f"Article:\n{text}\n\n"
        "### Response:\n"
    )


def prompt_summarizer(text: str, **_) -> str:
    word_count = len(text.split())
    target     = max(20, int(word_count * 0.10))
    return (
        "### Instruction:\n"
        "ඔබ සිංහල පුවත් ලිපි සාරාංශ කිරීමේ විශේෂඥයෙකි.\n"
        "පහත සිංහල පුවත් ලිපිය කියවා, ලිපියේ ප්‍රධාන කරුණු ඇතුළත් සාරාංශයක් ලියන්න.\n"
        f"සාරාංශය වචන {target}කට සීමා කරන්න.\n\n"
        f"Article:\n{text}\n\n"
        "### Response:\n"
    )


def prompt_style(text: str, style: str = DEFAULT_STYLE, **_) -> str:
    instruction = STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS[DEFAULT_STYLE])
    return (
        "### Instruction:\n"
        "ඔබ සිංහල ලේඛන විශේෂඥයෙකි.\n"
        f"{instruction}\n"
        "අර්ථය වෙනස් නොකරන්න. ස්වාභාවික සිංහල භාවිත කරන්න.\n\n"
        f"Text:\n{text}\n\n"
        "### Response:\n"
    )


def prompt_base(text: str, **_) -> str:
    """No task adapter — general instruction-following on the merged base model."""
    return (
        "### Instruction:\n"
        f"{text}\n\n"
        "### Response:\n"
    )


PROMPT_BUILDERS: dict[str, callable] = {
    "grammar"   : prompt_grammar,
    "headline"  : prompt_headline,
    "summarizer": prompt_summarizer,
    "style"     : prompt_style,
    "base"      : prompt_base,
}


def build_prompt(task: str, text: str, style: Optional[str] = None) -> str:
    """
    Pass-through if the client already sent a fully-formed prompt.
    Otherwise wrap raw text in the task template.
    """
    if "### Instruction:" in text:
        return text
    builder = PROMPT_BUILDERS.get(task, prompt_grammar)
    return builder(text, style=style or DEFAULT_STYLE)


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


# ─────────────────────────────────────────────
# DYNAMIC TOKEN CAP
# ─────────────────────────────────────────────
def compute_max_tokens(task: str, input_token_len: int) -> int:
    rules = {
        "grammar"   : lambda n: max(64, int(n * 1.5)),
        "headline"  : lambda n: 60,
        "summarizer": lambda n: max(64, int(n * 0.3)),
        "style"     : lambda n: max(64, int(n * 1.5)),
        "base"      : lambda n: 512,
    }
    ceiling = {
        "grammar"   : 600,
        "headline"  : 60,
        "summarizer": 300,
        "style"     : 600,
        "base"      : 512,
    }
    fn  = rules.get(task, lambda n: 200)
    cap = fn(input_token_len)
    return min(cap, ceiling.get(task, 400))


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
# Guards `set_adapter()` + `model.generate()` as one atomic unit. Today
# generate_response() runs synchronously inside the async route handler,
# which already serializes all requests on the event loop — this lock makes
# that guarantee explicit and keeps it correct if that ever moves onto a
# threadpool. threading.Lock (not asyncio.Lock) since it must hold across
# real OS threads either way.
_generation_lock = threading.Lock()


def generate_response(raw_text: str, task: str, style: Optional[str] = None) -> dict:
    full_prompt = build_prompt(task, raw_text, style)

    inputs     = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    prompt_len = inputs["input_ids"].shape[1]

    stop_token_ids = [
        tokenizer.encode(seq, add_special_tokens=False)
        for seq in STOP_SEQUENCES
    ]

    max_new_tokens    = compute_max_tokens(task, prompt_len)
    stopping_criteria = StoppingCriteriaList([SequenceStop(stop_token_ids, prompt_len)])

    # "base" runs the merged model with every LoRA adapter disabled — the
    # playground's raw, task-agnostic mode. Every other task activates its
    # named adapter via set_adapter() as before.
    adapter_ctx = model.disable_adapter() if task == "base" else contextlib.nullcontext()

    with _generation_lock:
        if task != "base":
            model.set_adapter(task)
        with adapter_ctx, torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens     = max_new_tokens,
                do_sample          = False,
                temperature        = 1.0,
                repetition_penalty = 1.3,
                eos_token_id       = tokenizer.eos_token_id,
                pad_token_id       = tokenizer.eos_token_id,
                stopping_criteria  = stopping_criteria,
                use_cache          = True,
            )

    new_tokens = outputs[0][prompt_len:]
    result     = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    for seq in STOP_SEQUENCES:
        if result.endswith(seq):
            result = result[:-len(seq)].strip()

    return {
        "response"      : result if result and len(result) >= 2 else raw_text,
        "task"          : task,
        "style"         : style if task == "style" else None,
        "input_tokens"  : prompt_len,
        "max_cap_used"  : max_new_tokens,
        "output_tokens" : len(new_tokens),
    }


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
@app.post("/generate")
async def generate(req: PromptRequest):
    if req.task not in PROMPT_BUILDERS:
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

    return generate_response(req.prompt, req.task, req.style)


@app.get("/health")
def health():
    return {"status": "ok", "adapters_loaded": sorted(LOADED_ADAPTERS)}


@app.get("/tasks")
def list_tasks():
    """Returns supported tasks, available styles, and rendered prompt examples."""
    sample = "කොළඹ කොටස් වෙළෙඳපොළ මිල දර්ශකවල පසුබැස්මක් අද දිනයේ දී වාර්තා විය."
    return {
        "tasks": list(PROMPT_BUILDERS.keys()),
        "adapters": {task: os.path.basename(path) for task, path in ADAPTER_PATHS.items()},
        "styles": {
            "available"  : sorted(VALID_STYLES),
            "default"    : DEFAULT_STYLE,
            "applies_to" : "style task only",
        },
        "prompt_examples": {
            "grammar"   : prompt_grammar(sample),
            "headline"  : prompt_headline(sample),
            "summarizer": prompt_summarizer(sample),
            "base"      : prompt_base(sample),
            **{
                f"style/{s}": prompt_style(sample, style=s)
                for s in sorted(VALID_STYLES)
            },
        },
    }