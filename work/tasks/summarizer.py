"""Summarizer task — owned independently of grammar/headline/style. Editing
this file only affects the summarizer adapters' prompt, token budget,
generation params, and decode post-processing; it cannot affect the other
three tasks.

The deployed summarizer adapters (v02-v05) were trained on a Llama-3 chat
template (see summarizer/abstractive/{2,3,4,5}_train_summarizer_*.py), NOT
the Alpaca "### Instruction:" format the other three tasks use — do not
change prompt_summarizer() to match the other tasks' format without
retraining the adapters."""

ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def prompt_summarizer(text: str, **_) -> str:
    # Must stay byte-for-byte identical to build_prompt() in
    # summarizer/abstractive/4_test_summarizer.py / the format_prompt() in
    # {2,3,4,5}_train_summarizer_*.py.
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "පහත සිංහල පුවත් ලිපිය සාරාංශ කරන්න.\n\n"
        "ලිපියේ ඇති තොරතුරු පමණක් භාවිතා කරන්න.\n"
        "සාරාංශය මුල් ලිපියේ දිගෙන් 10% ත් 30% ත් අතර විය යුතුය.\n"
        "අමතර අදහස්, විශ්ලේෂණ හෝ නව තොරතුරු එකතු නොකරන්න.\n\n"
        f"Article:\n{text}\n\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def max_new_tokens(raw_text: str, prompt_token_len: int) -> int:
    """Calibrated directly from an audit of 5_qwen_summaries.jsonl (v05's
    actual training data) rather than guessed: of 151,438 raw Qwen-generated
    silver summaries, 73.6% get rejected by 5_train_summarizer_qwen.py's own
    MIN/MAX_COMP_RATIO + MAX_SUMMARY_TOKENS filters (Qwen's natural output
    averages ~55% compression despite being instructed to produce 10-30%).
    Of the 26.4% that survive and the model actually trains on, compression
    ratio (summary_words / article_words) clusters at median=0.408,
    P75=0.460, P90=0.486 — roughly 50% higher than the 20-30% this function
    previously assumed, which is why summaries kept truncating no matter how
    much the budget was raised: the model was trained to want ~45-49% of the
    article's length, not 20-30% of it.

    Uses P90 (~0.49) as the target ratio so most articles get enough budget
    to finish. Word->token multiplier and buffer are kept generous (not
    precisely measured) because over-provisioning is ~free: generation
    already stops early via <|eot_id|>/stopping criteria once the model
    naturally finishes, so a bigger ceiling only matters when the model
    actually wants to keep going. Hard-capped near training's own
    MAX_SUMMARY_TOKENS=150 ceiling (+ buffer) — no training example ever had
    a longer summary than that, so there's no basis for budgeting past it."""
    word_count = len(raw_text.split())
    target_words = word_count * 0.49
    return max(40, min(int(target_words * 3.0) + 30, 180))


REPETITION_PENALTY = 1.15

# Deliberately no NO_REPEAT_NGRAM_SIZE constant: HF's
# NoRepeatNGramLogitsProcessor scans the whole input_ids history, prompt
# included. Since the prompt IS the source article, setting this (as
# 4_test_summarizer.py does, to 3) blocks the model from opening the summary
# with the article's own opening phrase — observed corrupting the leading
# consonant of the summary's first word whenever it echoed the article's
# first few words (e.g. "හිටපු" -> "ිටපු", "ශ්‍රී ලංකා" -> "්‍රී ලංකා").


def decode(tokenizer, outputs, prompt_len: int) -> tuple[str, int]:
    """Decodes the FULL sequence and splits on the literal assistant-header
    string instead of slicing outputs[0][prompt_len:] — matches
    generate_summary() in summarizer/abstractive/4_test_summarizer.py,
    which does this specifically to avoid corrupting the first Sinhala
    grapheme cluster when the prompt/response boundary falls mid-token."""
    new_tokens = outputs[0][prompt_len:]
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if ASSISTANT_HEADER not in full_text:
        result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return result, len(new_tokens)

    result = full_text.split(ASSISTANT_HEADER, 1)[1]
    result = result.split("<|eot_id|>")[0]
    # U+FFFD (the decode replacement character) has shown up, across
    # repeated test runs, exactly where some adapters stop summarizing and
    # drift into unrelated continuation (markdown headers, meta-commentary
    # echoing the prompt, off-topic tangents). Treat its first occurrence as
    # an implicit stop marker rather than passing the derailed tail through.
    result = result.split("�")[0].strip()
    return result, len(new_tokens)
