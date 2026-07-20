import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"]  = "1"

from unsloth import FastLanguageModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import StoppingCriteria, StoppingCriteriaList
import torch

app = FastAPI()

BASE_PATH    = "/home/jovyan/work/sinllama/models/SinLLaMA-merged-base"
ADAPTER_PATH = "/home/jovyan/work/sinllama/models/adapters/headline_sinllama_v17"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name          = BASE_PATH,
    dtype               = torch.bfloat16,
    load_in_4bit        = True,
    local_files_only    = True,
    device_map          = "auto",
    attn_implementation = "eager",
)
model.load_adapter(ADAPTER_PATH)
FastLanguageModel.for_inference(model)
model.eval()


# ─────────────────────────────────────────────
# PROMPT TEMPLATE
# ─────────────────────────────────────────────
def prompt_headline(text: str) -> str:
    return (
        "### Instruction:\n"
        "ඔබ සිංහල පුවත් සංස්කාරකයෙකි.\n"
        "පහත සිංහල පුවත් ලිපිය කියවා, ලිපිය සඳහා සංක්ෂිප්ත හා ආකර්ශනීය ශීර්ෂ පාඨයක් (headline) ලියන්න.\n"
        "ශීර්ෂ පාඨය වචන 10කට නොඉක්මවිය යුතුය.\n\n"
        f"Article:\n{text}\n\n"
        "### Response:\n"
    )


def build_prompt(text: str) -> str:
    """Pass-through if the client already sent a fully-formed prompt."""
    if "### Instruction:" in text:
        return text
    return prompt_headline(text)


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
MAX_NEW_TOKENS = 60


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class PromptRequest(BaseModel):
    prompt: str


# ─────────────────────────────────────────────
# CORE GENERATE
# ─────────────────────────────────────────────
def generate_response(raw_text: str) -> dict:
    full_prompt = build_prompt(raw_text)

    inputs     = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    prompt_len = inputs["input_ids"].shape[1]

    stop_token_ids = [
        tokenizer.encode(seq, add_special_tokens=False)
        for seq in STOP_SEQUENCES
    ]
    stopping_criteria = StoppingCriteriaList([SequenceStop(stop_token_ids, prompt_len)])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens     = MAX_NEW_TOKENS,
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
        "response"     : result if result and len(result) >= 2 else raw_text,
        "task"         : "headline",
        "input_tokens" : prompt_len,
        "max_cap_used" : MAX_NEW_TOKENS,
        "output_tokens": len(new_tokens),
    }


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
@app.post("/generate")
async def generate(req: PromptRequest):
    return generate_response(req.prompt)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    """Returns the served task and a rendered prompt example."""
    sample = "කොළඹ කොටස් වෙළෙඳපොළ මිල දර්ශකවල පසුබැස්මක් අද දිනයේ දී වාර්තා විය."
    return {
        "task"           : "headline",
        "adapter"        : ADAPTER_PATH,
        "prompt_example" : prompt_headline(sample),
    }
