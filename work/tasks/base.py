"""Raw playground mode — no LoRA adapter, just the merged base model
following a plain instruction. Not owned by any single task's team; shared
neutral default used by /generate's task="base"."""


def prompt_base(text: str, **_) -> str:
    return (
        "### Instruction:\n"
        f"{text}\n\n"
        "### Response:\n"
    )


def max_new_tokens(raw_text: str, prompt_token_len: int) -> int:
    return 512


REPETITION_PENALTY = 1.3
