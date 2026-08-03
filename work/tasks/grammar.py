"""Grammar-checker task — owned independently of headline/style/summarizer.
Editing this file only changes the grammar adapter's prompt, token budget,
and generation behavior; it cannot affect the other three tasks."""

# Byte-for-byte identical to INSTRUCTION_TEXT in
# work/sinllama/scripts/train_grammar.py and test_grammar.py. This is not a
# style choice — completion_only_loss=True training conditioned the adapter
# on this exact fixed string; a different instruction (even a
# same-meaning one, even in a different language) moves inference off the
# distribution the LoRA was tuned under. If you touch either training or
# this copy, update both together.
INSTRUCTION_TEXT = (
    "Correct the grammar of the Sinhala sentence. "
    "ONLY fix errors. "
    "If the sentence is already correct, return it EXACTLY unchanged — "
    "do not rephrase, reorder, or change tense."
)


def prompt_grammar(text: str, **_) -> str:
    return (
        f"### Instruction:\n{INSTRUCTION_TEXT}\n\n"
        f"### Input:\n{text}\n\n"
        "### Response:\n"
    )


def max_new_tokens(raw_text: str, prompt_token_len: int) -> int:
    return max(64, min(600, int(prompt_token_len * 1.5)))


# NOTE: before this file was split out, /generate used repetition_penalty=1.3
# for every non-summarizer task (no grammar-specific case), while /compare
# used 1.0 specifically for grammar — a real divergence between the two
# endpoints that had gone unnoticed. Consolidating onto one shared
# generation path (see serve_sinai.py's run_generation()) forces picking one
# value; set to 1.0 here since /compare's explicit "if task == grammar"
# condition looked deliberate rather than accidental.
# Grammar owner: confirm this is the value you want, or change it — this is
# the only place it needs to change.
REPETITION_PENALTY = 1.0

# Sampling params used only when a caller explicitly requests an ensemble
# (num_candidates > 1 on /generate). The normal single-candidate path stays
# greedy (TaskSpec.do_sample=False for grammar, untouched by this) — these
# values apply only to the extra sampled candidates a caller opted into.
# Conservative rather than creative, matching the same reasoning that keeps
# do_sample=False the default for this faithfulness-sensitive task. Starting
# point is headline's already-tuned sampling values; re-tune independently
# once an ensemble eval run (test_grammar.py, once it accepts a
# --num-candidates flag) shows whether these actually help recall on the
# compound-error cases or not.
ENSEMBLE_TEMPERATURE = 0.3
ENSEMBLE_TOP_P = 0.9
