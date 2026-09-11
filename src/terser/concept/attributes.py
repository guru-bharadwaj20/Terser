"""Discrete attributes describing a (prompt, operator set) instance for concept learning.

An instance has five prompt attributes followed by seven operator flags, in the order of
:data:`ATTRIBUTES`:

* ``length`` in ``{short, medium, long}``: gpt-4o (``o200k_base``) token count
  ``< SHORT_BELOW`` is short, ``<= MEDIUM_MAX`` is medium, otherwise long.
* ``has_constraints``: an explicit requirement ("must", "at most", "3 bullets", ...).
* ``has_payload``: material the prompt operates on: a code fence, inline code, a quotation of
  at least ``LONG_QUOTE_CHARS`` characters, or a code keyword pattern (``def f(``,
  ``SELECT … FROM``, ``import x``, ``#include <``, ...). Patterns require code syntax, so
  English uses of "class" or "import" do not trigger it.
* ``tone_cue``: a request about tone, style, register or audience.
* ``filler_density`` in ``{low, high}``: share of gpt-4o tokens removed by the operators
  ``F``, ``H``, ``C`` (``FILLER_OPS``) is at least ``FILLER_DENSITY_HIGH``.
* ``op_F`` … ``op_S``: whether each operator is in the applied set.

Booleans use the values ``yes``/``no``. The regex cues are deliberately simple and
inspectable; they define the instance language, not a classifier to be tuned.

``terser.operators.apply_ops`` is imported lazily, and :func:`describe_prompt` accepts a
``remover`` so callers and tests can supply the ``FHC`` rewrite themselves.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping

SHORT_BELOW = 25
"""Prompts with fewer gpt-4o tokens than this are ``short``."""
MEDIUM_MAX = 80
"""Prompts with at most this many gpt-4o tokens (and not short) are ``medium``."""
FILLER_DENSITY_HIGH = 0.15
"""Minimum share of tokens removed by :data:`FILLER_OPS` for ``filler_density=high``."""
FILLER_OPS = "FHC"
"""Operators whose removals measure filler density: filler, hedging, meta-talk."""
LENGTH_TARGET = "gpt-4o"
"""Tokenizer used for ``length`` and ``filler_density``."""
LONG_QUOTE_CHARS = 40
"""Minimum characters inside quotes for a quotation to count as payload."""

OP_CODES = "FHDILCS"
"""Operator codes, in flag order."""
YES, NO = "yes", "no"
BOOL: tuple[str, str] = (YES, NO)

PROMPT_ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "length": ("short", "medium", "long"),
    "has_constraints": BOOL,
    "has_payload": BOOL,
    "tone_cue": BOOL,
    "filler_density": ("low", "high"),
}
"""Prompt-level attributes and their domains."""
OP_ATTRIBUTES: dict[str, tuple[str, ...]] = {f"op_{c}": BOOL for c in OP_CODES}
"""Operator flag attributes and their domains."""
ATTRIBUTES: dict[str, tuple[str, ...]] = {**PROMPT_ATTRIBUTES, **OP_ATTRIBUTES}
"""All attributes, in instance order, mapped to their domains."""
ATTRIBUTE_NAMES: tuple[str, ...] = tuple(ATTRIBUTES)
DOMAINS: tuple[tuple[str, ...], ...] = tuple(ATTRIBUTES.values())

Remover = Callable[[str], str]
"""Returns the text with the :data:`FILLER_OPS` operators applied."""

_CONSTRAINT = re.compile(
    r"\b(?:must|should not|shouldn'?t|don'?t|do not|never|only|at least|at most|exactly|"
    r"no more than|no longer than|fewer than|less than|under \d+|avoid|without|make sure|"
    r"keep it|limit(?:ed)? to|maximum|minimum|max|"
    r"\d+\s*(?:words?|lines?|bullets?|bullet points?|sentences?|paragraphs?|characters?|items?))\b",
    re.IGNORECASE,
)
_TONE = re.compile(
    r"\b(?:tone|style|voice|register|funny|humou?r(?:ous)?|formal|informal|casual|friendly|"
    r"polite|professional|sarcastic|playful|warm|enthusiastic|persuasive|poem|story|lyrics|"
    r"like a (?:pirate|kid|child|teacher|friend)|"
    r"for (?:a )?(?:kids?|children|child|beginners?|\d+[- ]year[- ]old))\b",
    re.IGNORECASE,
)
_PAYLOAD = [
    re.compile(r"```"),
    re.compile(r"`[^`\n]+`"),
    re.compile(r"[\"“][^\"”]{" + str(LONG_QUOTE_CHARS) + r",}[\"”]"),
    re.compile(r"^\s*(?:def|class)\s+\w+\s*[(:]", re.MULTILINE),
    re.compile(r"\bfunction\s*\w*\s*\([^)]*\)\s*\{"),
    re.compile(r"\bSELECT\b[\s\S]+?\bFROM\b"),
    re.compile(r"^\s*(?:import\s+[\w.]+|from\s+[\w.]+\s+import\b)", re.MULTILINE),
    re.compile(r"#include\s*<"),
    re.compile(r"\b(?:console\.log|System\.out\.println|printf)\s*\("),
    re.compile(r"\w+\s*\([^()\n]*\)\s*(?:\{|=>)"),
]


def length_bucket(n_tokens: int) -> str:
    """Map a gpt-4o token count to ``short``/``medium``/``long``."""
    if n_tokens < SHORT_BELOW:
        return "short"
    return "medium" if n_tokens <= MEDIUM_MAX else "long"


def has_constraints(text: str) -> bool:
    """Whether the prompt states an explicit requirement."""
    return _CONSTRAINT.search(text) is not None


def has_payload(text: str) -> bool:
    """Whether the prompt carries code or quoted material to operate on."""
    return any(p.search(text) for p in _PAYLOAD)


def has_tone_cue(text: str) -> bool:
    """Whether the prompt asks for a tone, style, register or audience."""
    return _TONE.search(text) is not None


def _count(text: str) -> int:
    from terser.tokenizers import count_tokens

    return count_tokens(text, LENGTH_TARGET)


def default_remover(text: str) -> str:
    """Apply :data:`FILLER_OPS` with ``terser.operators.apply_ops`` (imported lazily)."""
    try:
        from terser.operators import apply_ops  # type: ignore[attr-defined,unused-ignore]
    except ImportError as exc:  # operators land in rows 1.2.x
        raise ImportError(
            "terser.operators.apply_ops is unavailable; pass remover= to describe_prompt"
        ) from exc
    out: str = apply_ops(text, FILLER_OPS, LENGTH_TARGET)
    return out


def filler_ratio(text: str, remover: Remover | None = None) -> float:
    """Share of gpt-4o tokens removed by ``remover`` (clipped to ``[0, 1]``; 0 for empty text)."""
    n = _count(text)
    if n == 0:
        return 0.0
    after = _count((remover or default_remover)(text))
    return min(1.0, max(0.0, (n - after) / n))


def describe_prompt(text: str, remover: Remover | None = None) -> dict[str, str]:
    """Compute the prompt attributes of ``text``, keyed in :data:`PROMPT_ATTRIBUTES` order."""
    ratio = filler_ratio(text, remover)
    return {
        "length": length_bucket(_count(text)),
        "has_constraints": YES if has_constraints(text) else NO,
        "has_payload": YES if has_payload(text) else NO,
        "tone_cue": YES if has_tone_cue(text) else NO,
        "filler_density": "high" if ratio >= FILLER_DENSITY_HIGH else "low",
    }


def instance(prompt_attrs: Mapping[str, str], ops: str | Iterable[str]) -> tuple[str, ...]:
    """Build the full instance tuple from prompt attributes and the applied operator codes.

    ``ops`` is a string such as ``"FHC"`` or an iterable of codes; order and repeats are
    ignored. Raises ``ValueError`` on unknown codes, missing/extra attributes or bad values.
    """
    codes = set(ops)
    unknown = codes - set(OP_CODES)
    if unknown:
        raise ValueError(f"unknown operator codes {sorted(unknown)}; expected from {OP_CODES}")
    if set(prompt_attrs) != set(PROMPT_ATTRIBUTES):
        raise ValueError(
            f"prompt attributes must be exactly {list(PROMPT_ATTRIBUTES)}, got {list(prompt_attrs)}"
        )
    for name, value in prompt_attrs.items():
        if value not in PROMPT_ATTRIBUTES[name]:
            raise ValueError(f"{name}={value!r} not in {PROMPT_ATTRIBUTES[name]}")
    flags = tuple(YES if c in codes else NO for c in OP_CODES)
    return tuple(prompt_attrs[name] for name in PROMPT_ATTRIBUTES) + flags
