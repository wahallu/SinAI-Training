"""Central registry mapping task name -> TaskSpec.

Each task's own module (tasks/grammar.py, tasks/headline.py, tasks/style.py,
tasks/summarizer.py, tasks/base.py) owns its prompt format, token budget,
and generation tuning. Changing one task's behavior means editing only that
task's file — this registry and serve_sinai.py's generation core never need
to change, and no task module imports another. That isolation is the whole
point of this split: a change to one component can no longer silently break
another's.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from tasks import base, extractive, grammar, headline, style, summarizer


@dataclass
class TaskSpec:
    name: str
    prompt_builder: Callable[..., str]
    max_new_tokens: Callable[[str, int], int]
    repetition_penalty: float = 1.3
    no_repeat_ngram_size: int = 0
    decode: Optional[Callable] = None  # (tokenizer, outputs, prompt_len) -> (text, output_tokens); None = gateway default


TASKS: dict[str, TaskSpec] = {
    "grammar": TaskSpec(
        name="grammar",
        prompt_builder=grammar.prompt_grammar,
        max_new_tokens=grammar.max_new_tokens,
        repetition_penalty=grammar.REPETITION_PENALTY,
    ),
    "headline": TaskSpec(
        name="headline",
        prompt_builder=headline.prompt_headline,
        max_new_tokens=headline.max_new_tokens,
        repetition_penalty=headline.REPETITION_PENALTY,
    ),
    "style": TaskSpec(
        name="style",
        prompt_builder=style.prompt_style,
        max_new_tokens=style.max_new_tokens,
        repetition_penalty=style.REPETITION_PENALTY,
    ),
    "summarizer": TaskSpec(
        name="summarizer",
        prompt_builder=summarizer.prompt_summarizer,
        max_new_tokens=summarizer.max_new_tokens,
        repetition_penalty=summarizer.REPETITION_PENALTY,
        decode=summarizer.decode,
    ),
    "extractive": TaskSpec(
        name="extractive",
        prompt_builder=extractive.prompt_extractive,
        max_new_tokens=extractive.max_new_tokens,
        repetition_penalty=extractive.REPETITION_PENALTY,
    ),
    "tfidf": TaskSpec(
        name="tfidf",
        prompt_builder=extractive.prompt_extractive,
        max_new_tokens=extractive.max_new_tokens,
    ),
    "textrank": TaskSpec(
        name="textrank",
        prompt_builder=extractive.prompt_extractive,
        max_new_tokens=extractive.max_new_tokens,
    ),
    "rake": TaskSpec(
        name="rake",
        prompt_builder=extractive.prompt_extractive,
        max_new_tokens=extractive.max_new_tokens,
    ),
    "yake": TaskSpec(
        name="yake",
        prompt_builder=extractive.prompt_extractive,
        max_new_tokens=extractive.max_new_tokens,
    ),
    "keybert": TaskSpec(
        name="keybert",
        prompt_builder=extractive.prompt_extractive,
        max_new_tokens=extractive.max_new_tokens,
    ),
    "mt5": TaskSpec(
        name="mt5",
        prompt_builder=lambda text, **_: f"summarize: {text}",
        max_new_tokens=lambda raw_text, prompt_len: 180,
    ),
    "mt5-base": TaskSpec(
        name="mt5-base",
        prompt_builder=lambda text, **_: f"summarize: {text}",
        max_new_tokens=lambda raw_text, prompt_len: 180,
    ),
    "base": TaskSpec(
        name="base",
        prompt_builder=base.prompt_base,
        max_new_tokens=base.max_new_tokens,
        repetition_penalty=base.REPETITION_PENALTY,
    ),
}

# Re-exported so the gateway only needs to import task_registry, not reach
# into tasks/style.py directly for these.
VALID_STYLES  = style.VALID_STYLES
DEFAULT_STYLE = style.DEFAULT_STYLE
