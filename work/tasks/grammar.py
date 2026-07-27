"""Grammar-checker task — owned independently of headline/style/summarizer.
Editing this file only changes the grammar adapter's prompt, token budget,
and generation behavior; it cannot affect the other three tasks."""


def prompt_grammar(text: str, **_) -> str:
    return (
        "### Instruction:\n"
        "ඔබ සිංහල භාෂා විශේෂඥයෙකි.\n"
        "පහත සිංහල පාඨයේ ඇති වාකරණ දෝෂ, අක්ෂර වින්‍යාස දෝෂ සහ විරාම ලකුණු දෝෂ නිවැරදි කරන්න.\n"
        "නිවැරදි කළ පාඨය පමණක් ලියන්න. වෙනත් කිසිදු පැහැදිලි කිරීමක් එකතු නොකරන්න.\n\n"
        f"Text:\n{text}\n\n"
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
