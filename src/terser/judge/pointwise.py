"""Pointwise (absolute) rubric judge: each target output is scored on its own.

Plan rows **2.3.3 / 2.3.4**, protocol comparison in ADR-0005. The pairwise judge
(:mod:`terser.judge.pairwise`) shows both outputs at once, so its verdict can depend on which one
is shown first. The pointwise judge never shows two outputs together: it scores one output
against the category rubric (row 2.3.2), dimension by dimension on the rubric's 1-5 anchors, and
reports whether a hard-fail condition applies. Two outputs are then compared through their
scores (:func:`side_from_scores`), which yields the same ``original`` / ``rewrite`` / ``tie``
sides as the pairwise judge, so the row 2.3.4 label rule applies unchanged: ``preserved`` iff the
rewrite's output scores tie-or-better.

Because a score depends only on (request, output), the comparison is order-free by construction,
and an output that is byte-identical to the original's always ties.

All calls go through :class:`terser.llm.LLMClient` (budget check, disk cache) at temperature 0;
a retry after an unparseable reply appends a corrective turn, as in the pairwise judge.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from terser.judge.pairwise import _json_candidates, load_prompt_template, rubric_section
from terser.judge.rubrics import ANCHOR_SCORES, load_rubric

if TYPE_CHECKING:
    from terser.data.categories import Category
    from terser.judge.pairwise import Side
    from terser.llm import LLMClient

__all__ = [
    "DEFAULT_POINTWISE_VERSION",
    "PointScore",
    "PointwiseJudge",
    "parse_scores",
    "render_pointwise_prompt",
    "side_from_scores",
    "weighted_score",
]

DEFAULT_POINTWISE_VERSION = "pointwise_v1"
RETRY_MESSAGE = (
    "Your previous reply could not be parsed. Reply again with ONLY a JSON object of the form "
    '{"scores": {"<dimension name>": <integer 1-5>, ...}, "hard_fail": true | false, '
    '"reason": "<one sentence>"} with every rubric dimension scored, and no other text.'
)
SCORE_TOLERANCE = 1e-9

_PLACEHOLDER_RE = re.compile(r"\{(prompt|response|rubric)\}")
_MIN_SCORE, _MAX_SCORE = min(ANCHOR_SCORES), max(ANCHOR_SCORES)


def render_pointwise_prompt(template: str, prompt: str, response: str, rubric: str) -> str:
    """Fill ``{prompt}``, ``{response}`` and ``{rubric}`` in a single regex pass."""
    values = {"prompt": prompt, "response": response, "rubric": rubric}
    return _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], template)


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def parse_scores(text: str, dimensions: tuple[str, ...]) -> tuple[dict[str, int], bool, str] | None:
    """Parse a pointwise reply into ``(scores, hard_fail, reason)``; None if invalid.

    Every name in ``dimensions`` must have an integer score in 1-5 (names are matched
    case- and separator-insensitively; extra keys are ignored). ``hard_fail`` defaults to False
    when absent.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    wanted = {_norm_name(d): d for d in dimensions}
    for candidate in _json_candidates(text):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("scores"), dict):
            continue
        raw = {_norm_name(str(k)): v for k, v in obj["scores"].items()}
        scores: dict[str, int] = {}
        for key, name in wanted.items():
            value = raw.get(key)
            if isinstance(value, bool) or not isinstance(value, int | float):
                break
            if value != int(value) or not _MIN_SCORE <= value <= _MAX_SCORE:
                break
            scores[name] = int(value)
        else:
            hard_fail = obj.get("hard_fail", False)
            if not isinstance(hard_fail, bool):
                hard_fail = str(hard_fail).strip().lower() in ("true", "yes", "1")
            return scores, hard_fail, str(obj.get("reason", "") or "").strip()
    return None


def weighted_score(scores: dict[str, int], weights: dict[str, float]) -> float:
    """Rubric-weighted mean of the dimension scores (weights sum to 1 in every rubric)."""
    return sum(weights[name] * scores[name] for name in weights)


@dataclass(frozen=True)
class PointScore:
    """One output scored once (after retries); ``score`` is None when unparseable."""

    scores: dict[str, int] | None
    hard_fail: bool | None
    score: float | None
    reason: str
    raw: str
    attempts: int
    cached: bool
    output_tokens: int

    @property
    def parsed(self) -> bool:
        """True when the reply gave a score for every rubric dimension."""
        return self.score is not None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return asdict(self)


def side_from_scores(original: PointScore, rewrite: PointScore) -> Side | None:
    """Compare two scored outputs: the side with the better score, ``tie``, or None.

    A hard fail loses unless both outputs have one (the rubric's hard-fail rule); otherwise the
    higher weighted score wins and equal scores tie. None if either output is unparsed.
    """
    if not original.parsed or not rewrite.parsed:
        return None
    if original.hard_fail != rewrite.hard_fail:
        return "rewrite" if original.hard_fail else "original"
    assert original.score is not None and rewrite.score is not None
    diff = rewrite.score - original.score
    if abs(diff) <= SCORE_TOLERANCE:
        return "tie"
    return "rewrite" if diff > 0 else "original"


class PointwiseJudge:
    """A pointwise rubric judge bound to one provider/model and prompt template."""

    def __init__(
        self,
        client: LLMClient,
        *,
        provider: str,
        model: str,
        max_tokens: int = 256,
        retries: int = 2,
        temperature: float = 0.0,
        prompt_version: str = DEFAULT_POINTWISE_VERSION,
    ) -> None:
        if retries < 0:
            raise ValueError("retries must be >= 0")
        self.client = client
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.retries = retries
        self.temperature = temperature
        self.prompt_version = prompt_version
        self.template = load_prompt_template(prompt_version)

    def score(self, prompt: str, response: str, category: Category | str) -> PointScore:
        """Score one output for ``prompt`` against ``category``'s rubric."""
        rubric = load_rubric(category)
        weights = {d.name: d.weight for d in rubric.dimensions}
        names = tuple(weights)
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": render_pointwise_prompt(
                    self.template, prompt, response, rubric_section(category)
                ),
            }
        ]
        raw = ""
        cached_all = True
        out_tokens = 0
        for attempt in range(1, self.retries + 2):
            resp = self.client.complete(
                self.provider, self.model, messages, self.max_tokens, temperature=self.temperature
            )
            raw = resp.text
            cached_all = cached_all and resp.cached
            out_tokens += 0 if resp.cached else resp.output_tokens
            parsed = parse_scores(raw, names)
            if parsed is not None:
                scores, hard_fail, reason = parsed
                return PointScore(
                    scores=scores,
                    hard_fail=hard_fail,
                    score=weighted_score(scores, weights),
                    reason=reason,
                    raw=raw,
                    attempts=attempt,
                    cached=cached_all,
                    output_tokens=out_tokens,
                )
            messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {"role": "user", "content": RETRY_MESSAGE},
            ]
        return PointScore(None, None, None, "", raw, self.retries + 1, cached_all, out_tokens)

    def compare(
        self, prompt: str, original_output: str, rewrite_output: str, category: Category | str
    ) -> tuple[PointScore, PointScore, Side | None]:
        """Score both outputs independently and return ``(original, rewrite, side)``."""
        original = self.score(prompt, original_output, category)
        rewrite = self.score(prompt, rewrite_output, category)
        return original, rewrite, side_from_scores(original, rewrite)
