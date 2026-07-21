"""Style-rewriter task — owned independently of grammar/headline/summarizer.
Editing this file (including STYLE_INSTRUCTIONS — add a new style by adding
a key here, nothing else needs to change) only affects style rewriting; it
cannot affect the other three tasks."""

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


def max_new_tokens(raw_text: str, prompt_token_len: int) -> int:
    return max(64, min(600, int(prompt_token_len * 1.5)))


REPETITION_PENALTY = 1.3
