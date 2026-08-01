"""Headline-generator task — owned independently of grammar/style/summarizer.
Editing this file only changes the headline adapter's prompt, token budget,
and generation behavior; it cannot affect the other three tasks.

`length` (short|medium|long) selects a target word band. Passing length=None
(the default) reproduces the pre-length-control prompt and token budget
byte-for-byte, which is what headline_sinllama_v17 and every earlier adapter
expect — v17 was trained on 48K examples that all carried the same fixed
"Between 4 and 7 words" rule line (see
work/sinllama/scripts/train_headline.py), so there is no length conditioning
in its training distribution. Until a length-conditioned adapter exists (v18,
see scripts/train_headline_v18.py) the band line is a nudge the model obeys
only partly; the hard guarantee comes from word-count enforcement downstream
in backend-api's headline_service.py. Nothing here defaults length on the
caller's behalf — an omitted length keeps today's exact behavior.
"""

# Non-overlapping word bands: every word count maps to exactly one band, so
# "did this headline land in the requested band" is answerable without a
# tiebreak rule. Must stay in sync with HEADLINE_LENGTHS in backend-api's
# app/core/prompts.py — that's the copy the live web app actually prompts
# with, since the backend sends fully-formed prompts that build_prompt()
# passes through untouched. This copy drives the token budgets below (which
# apply to every request, formed prompt or not) and the raw-text path used
# by /compare and /tasks.
HEADLINE_LENGTHS = {
    "short":  {"min_words": 3, "max_words": 5},
    "medium": {"min_words": 6, "max_words": 7},
    "long":   {"min_words": 8, "max_words": 10},
}

DEFAULT_LENGTH = "medium"

# The pre-length-control instruction line, kept as the length=None default so
# existing callers (/compare, /tasks, any raw-text client) are unaffected.
LEGACY_LINE = "ශීර්ෂ පාඨය වචන 10කට නොඉක්මවිය යුතුය."


def resolve_length(length: str = None) -> str:
    """Normalizes a client-supplied length onto a known band, defaulting to
    medium. Callers that want the legacy prompt must pass length=None rather
    than route through this."""
    if not length:
        return DEFAULT_LENGTH
    length = length.strip().lower()
    return length if length in HEADLINE_LENGTHS else DEFAULT_LENGTH


def prompt_headline(text: str, length: str = None, **_) -> str:
    if length:
        band = HEADLINE_LENGTHS[resolve_length(length)]
        line = (
            f"ශීර්ෂ පාඨය වචන {band['min_words']}ත් {band['max_words']}ත් අතර විය යුතුය."
        )
    else:
        line = LEGACY_LINE
    return (
        "### Instruction:\n"
        "ඔබ සිංහල පුවත් සංස්කාරකයෙකි.\n"
        "පහත සිංහල පුවත් ලිපිය කියවා, ලිපිය සඳහා සංක්ෂිප්ත හා ආකර්ශනීය ශීර්ෂ පාඨයක් (headline) ලියන්න.\n"
        f"{line}\n\n"
        f"Article:\n{text}\n\n"
        "### Response:\n"
    )


# Sinhala words -> tokens under the extended SinLLaMA vocabulary, measured on
# 12 real v17 headline generations (output_tokens vs. word count): min 1.20,
# median 1.71, p90 2.67. The spread is wide because a single Sinhala compound
# can cost as much as three short words — "කොටස් වෙළෙඳපොළේ පසුබෑමක්" is 3 words
# but 8 tokens.
#
# The ceiling is deliberately over-provisioned (~1.5x the observed p90):
# generation already stops on EOS or a stop sequence the moment the model
# finishes, so a budget that's too high costs nothing, while one that's too
# low cuts a headline mid-word. The band is enforced by word count downstream,
# never by this ceiling.
TOKENS_PER_WORD_CEILING = 4.0

# The floor is the opposite trade — min_new_tokens blocks EOS until it's
# reached, so overshooting produces rambling that then has to be trimmed. Set
# to the measured median so that on a typical headline the floor lands right
# at the band's lower edge; the spread around it is what the retry loop in
# backend-api's headline_service.py absorbs. No single token floor can
# guarantee a word count when the ratio varies 1.20-2.67.
#
# This is the lever that matters most for the long band: v17 left to itself
# stops at ~7 words (9-12 tokens) and ignores a prompt asking for 8-10, which
# matches test_headline.py's v15 note that min_new_tokens=8 pushed outputs
# into the 8-10 word range where 5 did not.
TOKENS_PER_WORD_FLOOR = 1.7


def max_new_tokens(raw_text: str, prompt_token_len: int, length: str = None) -> int:
    if length in HEADLINE_LENGTHS:
        band = HEADLINE_LENGTHS[length]
        return int(band["max_words"] * TOKENS_PER_WORD_CEILING) + 12
    return 60


def min_new_tokens_for(length: str = None) -> int:
    """Per-band generation floor, read by the gateway via
    TaskSpec.min_new_tokens_for. Never returns less than MIN_NEW_TOKENS, so
    the short band and the legacy path keep the value v15 testing settled on."""
    if length in HEADLINE_LENGTHS:
        band = HEADLINE_LENGTHS[length]
        return max(MIN_NEW_TOKENS, int(band["min_words"] * TOKENS_PER_WORD_FLOOR))
    return MIN_NEW_TOKENS


REPETITION_PENALTY   = 1.1
NO_REPEAT_NGRAM_SIZE = 2
DO_SAMPLE            = True
TEMPERATURE          = 0.3
TOP_P                = 0.9
MIN_NEW_TOKENS       = 5
