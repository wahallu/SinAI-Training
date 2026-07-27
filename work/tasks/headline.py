"""Headline-generator task — owned independently of grammar/style/summarizer.
Editing this file only changes the headline adapter's prompt, token budget,
and generation behavior; it cannot affect the other three tasks."""


def prompt_headline(text: str, **_) -> str:
    return (
        "### Instruction:\n"
        "ඔබ සිංහල පුවත් සංස්කාරකයෙකි.\n"
        "පහත සිංහල පුවත් ලිපිය කියවා, ලිපිය සඳහා සංක්ෂිප්ත හා ආකර්ශනීය ශීර්ෂ පාඨයක් (headline) ලියන්න.\n"
        "ශීර්ෂ පාඨය වචන 10කට නොඉක්මවිය යුතුය.\n\n"
        f"Article:\n{text}\n\n"
        "### Response:\n"
    )


def max_new_tokens(raw_text: str, prompt_token_len: int) -> int:
    return 60


REPETITION_PENALTY   = 1.1
NO_REPEAT_NGRAM_SIZE = 2
DO_SAMPLE            = True
TEMPERATURE          = 0.3
TOP_P                = 0.9
MIN_NEW_TOKENS       = 5
