"""Position-swapped pairwise LLM judge and the pilot label rule.

Plan rows **1.3.6** (pilot judge) and **2.3.3 / 2.3.4** (the same judge at full scale). The judge
sees the user's *original* request and two target-model outputs, one produced from the original
prompt and one from the rewrite, and answers ``{"verdict": "A" | "B" | "tie", "reason": ...}``.
Every pair is judged twice with the positions swapped:

* order ``original_first``: A = output for the original prompt, B = output for the rewrite;
* order ``rewrite_first``: A = output for the rewrite, B = output for the original prompt.

Verdicts are mapped from positions to *sides* (``original`` / ``rewrite`` / ``tie``) so the two
orders can be compared. The label rule (:func:`label_from_sides`) is: ``preserved`` iff the judge
says tie-or-rewrite-better in **both** orders; ``degraded`` if either order prefers the original;
no label (``None``) if either verdict could not be parsed after all retries.

When the item's category is known, the rendered category rubric (:func:`rubric_section`, row
2.3.2) is inserted at the template's ``{rubric}`` placeholder; without a category the placeholder
renders empty, so the prompt (and its cache key) is unchanged.

All calls go through :class:`terser.llm.LLMClient` (budget check, disk cache) at temperature 0.
Because identical requests hit the cache, a retry after unparseable output appends a corrective
follow-up turn, which changes the request and therefore the cache key.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from terser.judge.rubrics import render_rubric

if TYPE_CHECKING:
    from terser.data.categories import Category
    from terser.llm import LLMClient, TokenLogprobs

__all__ = [
    "DEFAULT_PROMPT_VERSION",
    "JudgeCall",
    "PairwiseJudge",
    "PairwiseResult",
    "Side",
    "Verdict",
    "label_from_sides",
    "load_prompt_template",
    "parse_verdict",
    "render_prompt",
    "rubric_section",
    "to_side",
    "verdict_probs",
]

Verdict = Literal["A", "B", "tie"]
Side = Literal["original", "rewrite", "tie"]
Order = Literal["original_first", "rewrite_first"]
Label = Literal["preserved", "degraded"]

DEFAULT_PROMPT_VERSION = "pairwise_v1"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
RETRY_MESSAGE = (
    "Your previous reply could not be parsed. Reply again with ONLY a JSON object of the form "
    '{"verdict": "A" | "B" | "tie", "reason": "<one sentence>"} and no other text.'
)

_PLACEHOLDER_RE = re.compile(r"\{(prompt|response_a|response_b|rubric)\}")
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_VERDICT_KEYS = ("verdict", "winner", "choice", "answer", "better")
_REASON_KEYS = ("reason", "reasoning", "explanation", "rationale")
_TIE_WORDS = {"tie", "draw", "equal", "same", "both", "neither", "equivalent"}
_BARE_VERDICT_RE = re.compile(
    r"""["']?verdict["']?\s*[:=]\s*["']?(A|B|tie)["']?\s*(?:[,}.]|$)""",
    re.IGNORECASE | re.MULTILINE,
)


@cache
def load_prompt_template(version: str = DEFAULT_PROMPT_VERSION) -> str:
    """Return the judge prompt template ``src/terser/judge/prompts/<version>.txt``."""
    path = PROMPTS_DIR / f"{version}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"judge prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def rubric_section(category: Category | str | None, mode: str = "pairwise") -> str:
    """The ``{rubric}`` block for a category: rendered rubric plus markers, or ``""`` for None.

    ``mode`` is ``"pairwise"`` (two responses, one verdict) or ``"pointwise"`` (one response,
    1-5 scores); the instructions are worded for what the judge actually sees.
    """
    if category is None:
        return ""
    intro = (
        "Apply this rubric to the response when scoring it.\n"
        if mode == "pointwise"
        else "Apply this rubric to both responses when deciding your verdict.\n"
    )
    return (
        "[Rubric for this request]\n"
        f"{intro}"
        f"{render_rubric(category, mode=mode)}\n"
        "[End of rubric]\n\n"
    )


def render_prompt(
    template: str, prompt: str, response_a: str, response_b: str, rubric: str = ""
) -> str:
    """Fill ``{prompt}``, ``{response_a}``, ``{response_b}`` and ``{rubric}`` in a single pass.

    A single regex pass means placeholder-like text *inside* a response is never substituted, and
    the literal JSON braces in the template are left alone.

    Raises:
        ValueError: If ``rubric`` is non-empty but the template has no ``{rubric}`` placeholder.
    """
    if rubric and "{rubric}" not in template:
        raise ValueError("judge template has no {rubric} placeholder for the category rubric")
    values = {
        "prompt": prompt,
        "response_a": response_a,
        "response_b": response_b,
        "rubric": rubric,
    }
    return _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], template)


def _normalise_verdict(raw: Any) -> Verdict | None:
    """Map a raw verdict value (``"A"``, ``"response b"``, ``"Tie"``...) to A|B|tie or None."""
    if not isinstance(raw, str):
        return None
    key = raw.strip().strip(".\"'").lower()
    key = re.sub(r"^(response|output|assistant|option)\s+", "", key)
    if key == "a":
        return "A"
    if key == "b":
        return "B"
    if key in _TIE_WORDS:
        return "tie"
    return None


def _json_candidates(text: str) -> list[str]:
    """Substrings of ``text`` that may be a JSON object: fenced blocks, then balanced ``{...}``."""
    candidates = [m.group(1).strip() for m in _FENCE_RE.finditer(text)]
    depth, start = 0, -1
    in_str, escape = False, False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and depth > 0:
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                candidates.append(text[start : i + 1])
    return candidates


def parse_verdict(text: str) -> tuple[Verdict, str] | None:
    """Parse a judge reply into ``(verdict, reason)``; return None if no valid verdict is found.

    Tolerates code fences, prose around the object, alternative key names (``winner``,
    ``choice``...), case differences and values like ``"Response A"``. As a last resort accepts a
    bare ``verdict: A`` line (which also rescues JSON truncated after the verdict). Anything else
    (missing verdict, ``"A or B"``, empty text) is None.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    for candidate in _json_candidates(text):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        lowered = {str(k).strip().lower(): v for k, v in obj.items()}
        for key in _VERDICT_KEYS:
            if key in lowered:
                verdict = _normalise_verdict(lowered[key])
                if verdict is None:
                    break
                reason = next(
                    (str(lowered[k]) for k in _REASON_KEYS if lowered.get(k) is not None), ""
                )
                return verdict, reason.strip()
    match = _BARE_VERDICT_RE.search(text)
    if match:
        verdict = _normalise_verdict(match.group(1))
        if verdict is not None:
            return verdict, ""
    return None


_VERDICT_VALUE_RE = re.compile(r"""["']?verdict["']?\s*:\s*["']?""", re.IGNORECASE)


def _token_verdict(token: str) -> Verdict | None:
    """Map one generated token (``"A"``, ``' "B'``, ``"tie"``...) to a verdict, else None."""
    key = token.strip().strip("\"'").strip().lower()
    return {"a": "A", "b": "B", "tie": "tie"}.get(key)  # type: ignore[return-value]


def verdict_probs(text: str, logprobs: TokenLogprobs | None) -> dict[str, float] | None:
    """Probability of each verdict (``A``, ``B``, ``tie``) at the reply's verdict token.

    Finds the first token at or after the end of ``"verdict": "`` in the generated text and
    renormalises the ``top_logprobs`` mass of the candidates that spell a verdict. Returns None
    when there are no logprobs, no verdict key, or no verdict candidate in the top list.
    """
    if not logprobs:
        return None
    joined = "".join(tok for tok, _ in logprobs)
    match = _VERDICT_VALUE_RE.search(joined)
    if match is None:
        return None
    pos = 0
    for token, top in logprobs:
        end = pos + len(token)
        if end > match.end() and token.strip().strip("\"'"):
            mass: dict[str, float] = {"A": 0.0, "B": 0.0, "tie": 0.0}
            for cand, lp in top:
                verdict = _token_verdict(cand)
                if verdict is not None:
                    mass[verdict] += math.exp(lp)
            total = sum(mass.values())
            return {k: v / total for k, v in mass.items()} if total > 0 else None
        pos = end
    return None


def to_side(verdict: Verdict | None, order: Order) -> Side | None:
    """Translate a positional verdict into the side it favours for the given order."""
    if verdict is None:
        return None
    if verdict == "tie":
        return "tie"
    a_side: Side = "original" if order == "original_first" else "rewrite"
    b_side: Side = "rewrite" if order == "original_first" else "original"
    return a_side if verdict == "A" else b_side


def label_from_sides(original_first: Side | None, rewrite_first: Side | None) -> Label | None:
    """Label rule (row 2.3.4, judge part): preserved iff tie-or-rewrite-better in both orders.

    Truth table (``None`` = unparseable verdict):

    ============== ============== ===========
    original_first rewrite_first  label
    ============== ============== ===========
    rewrite/tie    rewrite/tie    preserved
    original       anything       degraded
    anything       original       degraded
    None           not original   None
    not original   None           None
    ============== ============== ===========

    A single ``original`` verdict is sufficient evidence of degradation, so it wins over a
    missing verdict in the other order.
    """
    from terser.judge.labels import label_pair  # labels.py is the single home of the rule

    return label_pair(None, original_first, rewrite_first).label


@dataclass(frozen=True)
class JudgeCall:
    """One judge call in one order, after retries."""

    order: Order
    verdict: Verdict | None
    side: Side | None
    reason: str
    raw: str
    attempts: int
    cached: bool
    output_tokens: int
    verdict_probs: dict[str, float] | None = None  # positional A/B/tie, with top_logprobs only

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return asdict(self)


@dataclass(frozen=True)
class PairwiseResult:
    """Both orders of one pairwise comparison plus the derived label."""

    original_first: JudgeCall
    rewrite_first: JudgeCall
    label: Label | None

    @property
    def position_consistent(self) -> bool:
        """True when both orders favour the same side (both parsed)."""
        a, b = self.original_first.side, self.rewrite_first.side
        return a is not None and a == b

    @property
    def parse_failed(self) -> bool:
        """True when at least one order has no valid verdict after retries."""
        return self.original_first.verdict is None or self.rewrite_first.verdict is None

    def verdicts(self) -> dict[str, Any]:
        """The ``verdicts`` field of a label record."""
        return {
            "original_first": self.original_first.to_dict(),
            "rewrite_first": self.rewrite_first.to_dict(),
            "position_consistent": self.position_consistent,
        }


class PairwiseJudge:
    """A pairwise judge bound to one provider/model and prompt template."""

    def __init__(
        self,
        client: LLMClient,
        *,
        provider: str,
        model: str,
        max_tokens: int = 256,
        retries: int = 2,
        temperature: float = 0.0,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        top_logprobs: int | None = None,
    ) -> None:
        """``top_logprobs=k`` requests the top-k logprobs per token so each call also carries
        :func:`verdict_probs` (OpenAI-compatible servers such as vLLM; changes the cache key)."""
        if retries < 0:
            raise ValueError("retries must be >= 0")
        if top_logprobs is not None and top_logprobs < 1:
            raise ValueError("top_logprobs must be >= 1 or None")
        self.top_logprobs = top_logprobs
        self.client = client
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.retries = retries
        self.temperature = temperature
        self.prompt_version = prompt_version
        self.template = load_prompt_template(prompt_version)

    def judge_once(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        order: Order,
        category: Category | str | None = None,
    ) -> JudgeCall:
        """Judge one ordering, retrying with a corrective turn when the reply is unparseable.

        ``category`` adds that category's rubric to the prompt; None leaves it out.
        """
        rubric = rubric_section(category)
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": render_prompt(self.template, prompt, response_a, response_b, rubric),
            }
        ]
        raw = ""
        cached_all = True
        out_tokens = 0
        extra: dict[str, Any] = (
            {"logprobs": True, "top_logprobs": self.top_logprobs} if self.top_logprobs else {}
        )
        for attempt in range(1, self.retries + 2):
            resp = self.client.complete(
                self.provider,
                self.model,
                messages,
                self.max_tokens,
                temperature=self.temperature,
                **extra,
            )
            raw = resp.text
            cached_all = cached_all and resp.cached
            out_tokens += 0 if resp.cached else resp.output_tokens
            parsed = parse_verdict(raw)
            if parsed is not None:
                verdict, reason = parsed
                return JudgeCall(
                    order=order,
                    verdict=verdict,
                    side=to_side(verdict, order),
                    reason=reason,
                    raw=raw,
                    attempts=attempt,
                    cached=cached_all,
                    output_tokens=out_tokens,
                    verdict_probs=verdict_probs(raw, resp.logprobs),
                )
            messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {"role": "user", "content": RETRY_MESSAGE},
            ]
        return JudgeCall(order, None, None, "", raw, self.retries + 1, cached_all, out_tokens)

    def judge_pair(
        self,
        prompt: str,
        original_output: str,
        rewrite_output: str,
        category: Category | str | None = None,
    ) -> PairwiseResult:
        """Judge (original output, rewrite output) in both A/B orders and derive the label.

        ``category`` adds that category's rubric (row 2.3.2) to both judge prompts.
        """
        first = self.judge_once(prompt, original_output, rewrite_output, "original_first", category)
        second = self.judge_once(prompt, rewrite_output, original_output, "rewrite_first", category)
        return PairwiseResult(first, second, label_from_sides(first.side, second.side))
