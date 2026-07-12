import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"]  = "1"

from unsloth import FastLanguageModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import StoppingCriteria, StoppingCriteriaList
from typing import Optional
import torch

app = FastAPI()

model_path = "/home/jovyan/work/sinllama/models/SinLLaMA-merged-base"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name          = model_path,
    dtype               = torch.bfloat16,
    load_in_4bit        = True,
    local_files_only    = True,
    device_map          = "auto",
    attn_implementation = "eager",
)
FastLanguageModel.for_inference(model)
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


PROMPT_BUILDERS: dict[str, callable] = {
    "grammar"   : prompt_grammar,
    "headline"  : prompt_headline,
    "summarizer": prompt_summarizer,
    "style"     : prompt_style,
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
    }
    ceiling = {
        "grammar"   : 600,
        "headline"  : 60,
        "summarizer": 300,
        "style"     : 600,
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

    with torch.no_grad():
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
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    """Returns supported tasks, available styles, and rendered prompt examples."""
    sample = "කොළඹ කොටස් වෙළෙඳපොළ මිල දර්ශකවල පසුබැස්මක් අද දිනයේ දී වාර්තා විය."
    return {
        "tasks": list(PROMPT_BUILDERS.keys()),
        "styles": {
            "available"  : sorted(VALID_STYLES),
            "default"    : DEFAULT_STYLE,
            "applies_to" : "style task only",
        },
        "prompt_examples": {
            "grammar"   : prompt_grammar(sample),
            "headline"  : prompt_headline(sample),
            "summarizer": prompt_summarizer(sample),
            **{
                f"style/{s}": prompt_style(sample, style=s)
                for s in sorted(VALID_STYLES)
            },
        },
    }