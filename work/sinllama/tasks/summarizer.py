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
    """Started as the 20%/120-cap formula from generate_summary() in
    summarizer/abstractive/4_test_summarizer.py, but that was tuned against
    that script's own (shorter) test set — real multi-paragraph articles hit
    the cap and got truncated mid-sentence. Widened to 30%/150, anchored on
    the adapters' own training-time MAX_SUMMARY_TOKENS=150 ceiling (see
    summarizer/abstractive/2_train_summarizer_llama4.py) rather than
    guessing further."""
    word_count = len(raw_text.split())
    return max(40, min(150, int(word_count * 0.30)))


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
