"""Style-rewriter task — owned independently of grammar/headline/summarizer.

IMPORTANT: STYLE_INSTRUCTIONS and the prompt shape built by prompt_style()
must stay byte-for-byte identical (modulo the public style-name keys, see
STYLE_ID_MAP) to STYLE_RULES / format_prompt() in
work/sinllama/scripts/train_style.py. The deployed adapter (style_sinllama_v*)
was LoRA-fine-tuned on that exact prompt shape — same "### Instruction:" /
"### FACT PRESERVATION RULES:" / "### Input:" / "### Response:" structure,
same English rule text. Serving any other prompt shape here (different
instruction wording, a different content marker, a missing FACT
PRESERVATION RULES block) puts every request out-of-distribution for the
adapter and it falls back to copying the input near-verbatim regardless of
requested style — that's what v12 was doing before this file was
synced to train_style.py.

If you add a new style, add it to STYLE_INSTRUCTIONS AND STYLE_ID_MAP here,
AND add a matching entry to STYLE_RULES in train_style.py before the next
training run — the adapter has no signal for a style it never trained on.
"""

# Public API style name -> STYLE_IDS key used in train_style.py's
# STYLE_RULES / training_config.json. Keep in sync with STYLE_IDS there.
STYLE_ID_MAP: dict[str, str] = {
    "formal": "style_1_formal_news",
    "editorial": "style_2_editorial",
    "sports": "style_3_sports",
    "youth": "style_4_youth",
    "feature": "style_5_feature",
}

# Verbatim copy of STYLE_RULES from train_style.py, keyed by the public
# style name instead of the style_id.
STYLE_INSTRUCTIONS: dict[str, str] = {
    "formal": """
Write the article as a professional Sinhala news report.

- Objective and factual
- Clear journalistic Sinhala
- Most important information first
- Formal vocabulary
- No personal opinion
- No unnecessary commentary
- Preserve all important facts
- Preserve names, numbers, dates and places
- Do not invent information
""",

    "editorial": """
Rewrite the article in a Sinhala editorial / analytical style.

- Analytical and reflective tone
- Discuss the significance of the reported facts
- Use natural editorial Sinhala
- May use phrases such as "සැලකිය යුතුය" or "සමස්තයක් ලෙස"
  only when they naturally fit
- Do not invent facts
- Do not exaggerate
- Preserve names, numbers, dates and places
- Preserve the factual basis of the original article
""",

    "sports": """
Rewrite the article in a Sinhala sports-news style.

- Energetic but professional
- Emphasize important actions, events and outcomes
- Use natural sports journalism vocabulary where appropriate
- Do not force sports terminology into non-sports stories
- Do not invent sporting events or actions
- Preserve all facts, names, numbers and dates
""",

    "youth": """
Rewrite the article in a modern, accessible Sinhala style
suitable for younger readers.

- Conversational but grammatically correct Sinhala
- Simple and clear sentences
- Engaging tone
- Natural modern vocabulary
- Do not force greetings into every article
- Do not use "යාලුවනේ" unless it naturally fits
- Do not add information
- Preserve names, numbers, dates and factual details
""",

    "feature": """
Rewrite the article in a Sinhala feature-writing style.

- Engaging narrative presentation
- More descriptive language where appropriate
- Smooth transitions
- Human-interest perspective where supported by the source
- Do not invent scenes, emotions or events
- Do not fabricate background information
- Preserve all factual information
""",
}

# Verbatim copy of the block train_style.py::format_prompt() inserts
# between the style instruction and "### Input:" on every training example.
FACT_PRESERVATION_RULES = (
    "### FACT PRESERVATION RULES:\n"
    "1. Preserve the meaning of every factual statement.\n"
    "2. Do not add facts that are absent from the source.\n"
    "3. Do not remove important factual information.\n"
    "4. Preserve all person names exactly.\n"
    "5. Preserve all organization names exactly.\n"
    "6. Preserve all locations exactly.\n"
    "7. Preserve numbers and numerical values exactly.\n"
    "8. Preserve dates exactly.\n"
    "9. Preserve quoted text exactly.\n"
    "10. Preserve gender-marked honorifics such as "
    "මහතා / මහත්මිය.\n"
    "11. Do not invent durations, measurements or statistics.\n"
    "12. Do not translate proper names unnecessarily.\n"
    "13. Do not introduce unrelated information.\n"
    "14. Change the writing STYLE, not the underlying facts.\n"
    "15. Use natural Sinhala grammar and morphology.\n"
    "16. Do not deliberately replace correct Sinhala words with "
    "different words merely to make the sentence different.\n"
)

DEFAULT_STYLE = "formal"
VALID_STYLES  = set(STYLE_INSTRUCTIONS.keys())


def prompt_style(text: str, style: str = DEFAULT_STYLE, **_) -> str:
    instruction = STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS[DEFAULT_STYLE])
    return (
        "### Instruction:\n"
        f"{instruction.strip()}\n\n"

        f"{FACT_PRESERVATION_RULES}\n"

        "### Input:\n"
        f"{text.strip()}\n\n"

        "### Response:\n"
    )


def max_new_tokens(raw_text: str, prompt_token_len: int) -> int:
    return max(64, min(600, int(prompt_token_len * 1.5)))


REPETITION_PENALTY = 1.05
