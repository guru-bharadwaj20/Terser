"""Automatic task metrics for ``fact``, ``reason`` and ``code`` (plan row 2.3.1).

Metric names follow :data:`terser.data.categories.CATEGORY_METRICS`. ``explain``, ``creative`` and
``transform`` have no automatic metric here: :func:`task_metric` returns a result with
``value=None`` for them and the rubric + pairwise judge (rows 2.3.2, 2.3.3) decide.

* **fact** (``exact_match``, ``token_f1``): SQuAD-style normalisation (NFKC, lowercase, strip
  punctuation and the articles *a/an/the*, collapse whitespace); EM and token F1, max over
  references. Because target models answer in sentences, a result also records whether a
  reference appears verbatim (after normalisation) inside the answer and whether every quantity
  in the reference appears in the answer up to unit conversion (:func:`numeric_match`, e.g.
  "212 °F" for "100 °C"). ``passed`` is EM or containment or numeric match. Two pure quantity
  expressions that are numerically equivalent ("212 F" vs "100 °C") score EM = F1 = 1.
* **reason** (``final_answer_accuracy``): :func:`extract_final_answer` finds the answer span
  (``\\boxed{}``, GSM8K ``####``, "the answer is X", else the last line with a number) and
  :func:`answer_accuracy` compares it with the reference: numbers, currency, fractions, percents,
  clock times and quantities with units are compared numerically with rounding tolerance;
  option letters and words ("Carol", "Friday") by normalised containment.
* **code** (``unit_test_pass_rate``): extract the code and run its tests in the sandboxed runner
  (:mod:`terser.judge.code_runner`). ``value`` is the share of test cases passed.

:func:`not_worse` implements the metric half of the 2.3.4 label rule, using the "not worse"
definitions of ``docs/categories.md`` section 2.
"""

from __future__ import annotations

import math
import re
import string
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from terser.data.categories import CATEGORY_METRICS, Category, parse_category
from terser.judge.code_runner import CodeTests, SandboxLimits, run_output_tests

__all__ = [
    "FACT_F1_TOLERANCE",
    "MetricResult",
    "Quantity",
    "answer_accuracy",
    "code_metric",
    "contains_answer",
    "exact_match",
    "extract_final_answer",
    "fact_metric",
    "normalize_answer",
    "not_worse",
    "numeric_match",
    "parse_quantities",
    "quantities_equal",
    "reason_metric",
    "task_metric",
    "token_f1",
]

FACT_F1_TOLERANCE = 0.05
"""A ``fact`` rewrite is not worse if its F1 is at most this much below the original's."""

References = str | Sequence[str]


# --------------------------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricResult:
    """The task metric for one target output.

    Attributes:
        category: The prompt's category.
        name: Metric name from ``CATEGORY_METRICS`` (``"token_f1"``, ``"final_answer_accuracy"``,
            ``"unit_test_pass_rate"``), or ``None`` when the category has no automatic metric.
        value: Metric value in [0, 1], or ``None`` if not computable (rubric-only category, no
            reference, non-executable language).
        passed: Whether the output counts as correct, or ``None`` when ``value`` is ``None``.
        details: Per-metric diagnostics (JSON-serialisable).
    """

    category: Category
    name: str | None
    value: float | None
    passed: bool | None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_null(self) -> bool:
        """True when no automatic metric was computed."""
        return self.value is None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "category": self.category.value,
            "name": self.name,
            "value": self.value,
            "passed": self.passed,
            "details": self.details,
        }


def _null(category: Category, reason: str, **details: Any) -> MetricResult:
    return MetricResult(
        category,
        None,
        None,
        None,
        {"reason": reason, "judged_by": list(CATEGORY_METRICS[category]), **details},
    )


# --------------------------------------------------------------------------------------------
# fact: normalisation, EM, F1
# --------------------------------------------------------------------------------------------

_ASCII_PUNCT = frozenset(string.punctuation)
_ARTICLES_RE = re.compile(r"\b(?:a|an|the)\b")
_NUMBER_RE = re.compile(r"(?<![\w.])[-−]?\d+(?:\.\d+)?")


def _is_punct(ch: str) -> bool:
    return ch in _ASCII_PUNCT or unicodedata.category(ch).startswith("P")


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: NFKC, lowercase, drop punctuation and articles, squash spaces.

    Unicode punctuation (curly quotes, dashes) is removed like ASCII punctuation; symbols such as
    ``°`` are kept.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    # Protect a number's sign and decimal point: "-5" must not normalise to "5", nor "3.5" to "35".
    text = _NUMBER_RE.sub(
        lambda m: m.group(0).replace("-", "").replace("−", "").replace(".", ""),
        text,
    )
    text = "".join(" " if _is_punct(ch) and ch in "-–—/" else ch for ch in text)
    text = "".join(ch for ch in text if not _is_punct(ch))
    text = text.replace("", "-").replace("", ".")
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def _as_references(references: References) -> list[str]:
    if isinstance(references, str):
        refs = [references]
    else:
        refs = [r for r in references if isinstance(r, str)]
    if not refs:
        raise ValueError("at least one reference answer is required")
    return refs


def _em_single(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def _f1_single(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _contains_single(prediction: str, reference: str) -> float:
    ref = normalize_answer(reference).split()
    pred = normalize_answer(prediction).split()
    if not ref:
        return 0.0
    n = len(ref)
    return float(any(pred[i : i + n] == ref for i in range(len(pred) - n + 1)))


def exact_match(prediction: str, references: References) -> float:
    """1.0 if the normalised prediction equals any normalised reference, else 0.0."""
    return max(_em_single(prediction, r) for r in _as_references(references))


def token_f1(prediction: str, references: References) -> float:
    """SQuAD token-level F1 against the best-matching reference."""
    return max(_f1_single(prediction, r) for r in _as_references(references))


def contains_answer(prediction: str, references: References) -> float:
    """1.0 if some normalised reference occurs as a contiguous token run in the prediction."""
    return max(_contains_single(prediction, r) for r in _as_references(references))


# --------------------------------------------------------------------------------------------
# Quantities: numbers, currency, fractions, percents, times, units
# --------------------------------------------------------------------------------------------

_Converter = Callable[[float], float]


def _scale(factor: float) -> _Converter:
    return lambda v: v * factor


# (pattern, unit key, dimension, converter to the dimension's base unit). Order matters: longer
# and more specific patterns first. Base units: kelvin, degree, m/s, m^2, m^3, m, kg, s.
_UNIT_TABLE: tuple[tuple[str, str, str, _Converter], ...] = (
    (
        r"°\s*c|º\s*c|degrees?\s+c(?:elsius)?|celsius|centigrade|c",
        "celsius",
        "temperature",
        lambda v: v + 273.15,
    ),
    (
        r"°\s*f|º\s*f|degrees?\s+f(?:ahrenheit)?|fahrenheit|f",
        "fahrenheit",
        "temperature",
        lambda v: (v - 32.0) * 5.0 / 9.0 + 273.15,
    ),
    (r"°\s*k|kelvins?|(?-i:K)", "kelvin", "temperature", lambda v: v),
    (
        r"km\s*/\s*h|kmh|kph|km\s+per\s+hour|kilomet(?:er|re)s?\s+(?:per|an)\s+hour",
        "km/h",
        "speed",
        _scale(1 / 3.6),
    ),
    (r"mph|miles?\s+(?:per|an)\s+hour", "mph", "speed", _scale(0.44704)),
    (r"m\s*/\s*s|met(?:er|re)s?\s+per\s+second", "m/s", "speed", _scale(1.0)),
    (r"cm\s*²|cm\s*\^\s*2|square\s+centimet(?:er|re)s?|sq\.?\s*cm", "cm2", "area", _scale(1e-4)),
    (r"m\s*²|m\s*\^\s*2|square\s+met(?:er|re)s?|sq\.?\s*m\b", "m2", "area", _scale(1.0)),
    (
        r"cm\s*³|cm\s*\^\s*3|cubic\s+centimet(?:er|re)s?|cc",
        "cm3",
        "volume",
        _scale(1e-6),
    ),
    (r"m\s*³|m\s*\^\s*3|cubic\s+met(?:er|re)s?", "m3", "volume", _scale(1.0)),
    (r"ml|millilit(?:er|re)s?", "ml", "volume", _scale(1e-6)),
    (r"l|lit(?:er|re)s?", "l", "volume", _scale(1e-3)),
    (r"gal(?:lons?)?", "gal", "volume", _scale(3.785411784e-3)),
    (r"degrees?|°|deg", "degree", "angle", _scale(1.0)),
    (r"km|kilomet(?:er|re)s?", "km", "length", _scale(1000.0)),
    (r"cm|centimet(?:er|re)s?", "cm", "length", _scale(0.01)),
    (r"mm|millimet(?:er|re)s?", "mm", "length", _scale(0.001)),
    (r"m|met(?:er|re)s?", "m", "length", _scale(1.0)),
    (r"mi|miles?", "mi", "length", _scale(1609.344)),
    (r"yd|yards?", "yd", "length", _scale(0.9144)),
    (r"ft|feet|foot", "ft", "length", _scale(0.3048)),
    (r"inch(?:es)?", "in", "length", _scale(0.0254)),
    (r"kg|kilograms?|kilos?", "kg", "mass", _scale(1.0)),
    (r"mg|milligrams?", "mg", "mass", _scale(1e-6)),
    (r"g|grams?", "g", "mass", _scale(1e-3)),
    (r"lbs?|pounds?", "lb", "mass", _scale(0.45359237)),
    (r"oz|ounces?", "oz", "mass", _scale(0.028349523125)),
    (r"ms|milliseconds?", "ms", "duration", _scale(1e-3)),
    (r"s|secs?|seconds?", "s", "duration", _scale(1.0)),
    (r"mins?|minutes?", "min", "duration", _scale(60.0)),
    (r"h|hrs?|hours?", "h", "duration", _scale(3600.0)),
    (r"days?", "day", "duration", _scale(86400.0)),
    (r"weeks?", "week", "duration", _scale(604800.0)),
)
_UNIT_RES: tuple[tuple[re.Pattern[str], str, str, _Converter], ...] = tuple(
    (re.compile(rf"[ \t]*(?:{pat})(?![A-Za-z0-9²³])", re.IGNORECASE), key, dim, conv)
    for pat, key, dim, conv in _UNIT_TABLE
)
_PERCENT_RE = re.compile(r"[ \t]*(?:%|percent\b|per\s*cent\b)", re.IGNORECASE)
_MERIDIEM = r"(?:[ap]\.?\s?m\b\.?)"
_QTY_RE = re.compile(
    rf"""
    (?P<clock>(?<![\d.:])(?P<ch>\d{{1,2}}):(?P<cm>\d{{2}})(?::\d{{2}})?(?![\d:])
        (?:\s*(?P<cmer>{_MERIDIEM}))?)
    |(?P<hour>(?<![\d.:,])(?P<hh>\d{{1,2}})\s*(?P<hmer>{_MERIDIEM}))
    |(?P<frac>(?<![\w./])(?P<fneg>[-−](?=\d))?(?P<fn>\d+)\s*/\s*(?P<fd>\d+)(?![\d/]|\.\d))
    |(?P<num>(?<![\w.])(?P<neg1>[-−](?=[$€£¥]?\d))?(?P<cur>[$€£¥]\s*)?(?P<neg2>[-−](?=\d))?
        (?P<val>(?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.\d+)?|\.\d+))
    """,
    re.VERBOSE | re.IGNORECASE,
)


@dataclass(frozen=True)
class Quantity:
    """A number found in text.

    ``value`` is as written (``30`` for "30%", ``19.0`` for "7pm", ``0.3571`` for "5/14").
    ``canonical`` is the value in the dimension's base unit when a unit was recognised.
    ``decimals`` and ``sig_digits`` describe the written precision (used for rounding tolerance).
    """

    value: float
    kind: str  # "number" | "currency" | "fraction" | "percent" | "time"
    text: str
    unit: str | None = None
    dimension: str | None = None
    canonical: float | None = None
    decimals: int = 0
    sig_digits: int = 0
    meridiem: bool = False
    start: int = 0
    end: int = 0


def _precision(digits: str) -> tuple[int, int]:
    """(decimal places, significant digits) of a written number like ``"0.3570"``."""
    body = digits.replace(",", "").lstrip("-−")
    decimals = len(body.split(".", 1)[1]) if "." in body else 0
    sig = body.replace(".", "").lstrip("0")
    return decimals, len(sig)


def parse_quantities(text: str) -> list[Quantity]:
    """Find all quantities in ``text``, in order of appearance.

    Recognises plain numbers (``1,440``, ``-3``, ``.5``), currency (``$2,315.25``), fractions
    (``5/14``), percents (``30%``, ``30 percent``), clock times (``7pm``, ``7:15 AM``, ``19:00``,
    ``5:15``, stored as hours) and units for temperature, speed, area, volume, angle, length, mass
    and duration. Spelled-out numbers ("twelve") are not recognised.
    """
    if not isinstance(text, str):
        return []
    text = unicodedata.normalize("NFC", text)
    out: list[Quantity] = []
    pos = 0
    while True:
        m = _QTY_RE.search(text, pos)
        if m is None:
            break
        end = m.end()
        if m.group("clock") is not None:
            hours, minutes = int(m.group("ch")), int(m.group("cm"))
            mer = m.group("cmer")
            value = _to_24h(hours, mer) + minutes / 60.0
            out.append(
                Quantity(value, "time", m.group(0), meridiem=mer is not None, start=m.start())
            )
        elif m.group("hour") is not None:
            hours, mer = int(m.group("hh")), m.group("hmer")
            out.append(Quantity(float(_to_24h(hours, mer)), "time", m.group(0), meridiem=True))
        elif m.group("frac") is not None:
            den = int(m.group("fd"))
            if den == 0:
                pos = end
                continue
            sign = -1.0 if m.group("fneg") else 1.0
            value = sign * int(m.group("fn")) / den
            unit = _match_unit(text, end)
            q = Quantity(value, "fraction", m.group(0), start=m.start())
            if unit is not None:
                (key, dim, conv), end = unit
                q = Quantity(value, "fraction", text[m.start() : end], key, dim, conv(value))
            out.append(q)
        else:
            raw = m.group("val")
            sign = -1.0 if (m.group("neg1") or m.group("neg2")) else 1.0
            value = sign * float(raw.replace(",", ""))
            decimals, sig = _precision(raw)
            kind = "currency" if m.group("cur") else "number"
            pct = _PERCENT_RE.match(text, end)
            if pct and kind == "number":
                end = pct.end()
                out.append(
                    Quantity(
                        value, "percent", text[m.start() : end], None, None, None, decimals, sig
                    )
                )
            else:
                unit = _match_unit(text, end) if kind == "number" else None
                if unit is not None:
                    (key, dim, conv), end = unit
                    out.append(
                        Quantity(
                            value,
                            kind,
                            text[m.start() : end],
                            key,
                            dim,
                            conv(value),
                            decimals,
                            sig,
                        )
                    )
                else:
                    out.append(Quantity(value, kind, m.group(0), None, None, None, decimals, sig))
        last = out[-1]
        out[-1] = Quantity(
            last.value,
            last.kind,
            last.text,
            last.unit,
            last.dimension,
            last.canonical,
            last.decimals,
            last.sig_digits,
            last.meridiem,
            m.start(),
            end,
        )
        pos = max(end, m.start() + 1)
    return out


def _to_24h(hours: int, meridiem: str | None) -> int:
    if meridiem is None:
        return hours
    is_pm = meridiem.strip().lower().startswith("p")
    hours %= 12
    return hours + 12 if is_pm else hours


def _match_unit(text: str, pos: int) -> tuple[tuple[str, str, _Converter], int] | None:
    for regex, key, dim, conv in _UNIT_RES:
        um = regex.match(text, pos)
        if um is not None:
            return (key, dim, conv), um.end()
    return None


def _close(a: float, b: float, a_q: Quantity | None = None, b_q: Quantity | None = None) -> bool:
    """Numeric equality with rounding tolerance.

    Exact up to 1e-6 relative error, or equal after rounding to the precision one side was written
    with, provided that side has decimals and at least two significant digits ("0.357" matches
    5/14, "11.3" matches 11.31; "0.4" does not match 0.357).
    """
    if math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9):
        return True
    for written, other in ((a_q, b), (b_q, a)):
        if written is None or written.decimals == 0 or written.sig_digits < 2:
            continue
        if abs(written.value - other) <= 0.5 * 10 ** (-written.decimals) + 1e-12:
            return True
    return False


def quantities_equal(pred: Quantity, ref: Quantity) -> bool:
    """Whether two quantities denote the same value.

    * Times compare as hours; if either side has no am/pm, equality modulo 12 is accepted ("7:00"
      vs "7pm"). A plain number equal to the clock hour also matches ("7" vs "7pm").
    * Both with units: same dimension required; compared in base units (0.5 % tolerance when the
      units differ, to absorb rounded conversions like "98.6 F" vs "37 C").
    * Percent vs plain number: "30%" matches 30 and 0.3.
    * Otherwise numbers (incl. currency and fractions) compare with rounding tolerance; a unit on
      only one side is ignored.
    """
    if pred.kind == "time" or ref.kind == "time":
        if pred.kind == "time" and ref.kind == "time":
            if math.isclose(pred.value, ref.value, abs_tol=1e-6):
                return True
            if not (pred.meridiem and ref.meridiem):
                return math.isclose(pred.value % 12, ref.value % 12, abs_tol=1e-6)
            return False
        time_q, other = (pred, ref) if pred.kind == "time" else (ref, pred)
        if other.kind != "number" or other.unit is not None:
            return False
        return math.isclose(time_q.value % 12, other.value % 12, abs_tol=1e-6) and (
            other.value == int(other.value)
        )
    if pred.dimension is not None and ref.dimension is not None:
        if pred.dimension != ref.dimension:
            return False
        assert pred.canonical is not None and ref.canonical is not None
        if pred.unit == ref.unit:
            return _close(pred.value, ref.value, pred, ref)
        return math.isclose(pred.canonical, ref.canonical, rel_tol=5e-3, abs_tol=1e-9)
    if pred.kind == "percent" and ref.kind != "percent":
        return _close(pred.value, ref.value, pred, ref) or _close(pred.value / 100, ref.value)
    if ref.kind == "percent" and pred.kind != "percent":
        return _close(pred.value, ref.value, pred, ref) or _close(pred.value, ref.value / 100, pred)
    return _close(pred.value, ref.value, pred, ref)


def numeric_match(prediction: str, reference: str) -> bool:
    """True if every quantity in ``reference`` has an equal quantity in ``prediction``.

    Unit-aware: ``numeric_match("It boils at 212 °F", "100 °C")`` is True. False when the reference
    contains no quantity.
    """
    ref_q = parse_quantities(reference)
    if not ref_q:
        return False
    pred_q = parse_quantities(prediction)
    return all(any(quantities_equal(p, r) for p in pred_q) for r in ref_q)


_CONNECTOR_WORDS = frozenset({"and", "or", "about", "approximately", "around", "roughly", "is"})


def _quantity_only(text: str) -> bool:
    """True if ``text`` is nothing but quantities and connector words ("100 °C and 212 °F")."""
    quantities = parse_quantities(text)
    if not quantities:
        return False
    rest = text
    for q in reversed(quantities):
        rest = rest[: q.start] + " " + rest[q.end :]
    return all(tok in _CONNECTOR_WORDS for tok in normalize_answer(rest).split())


def fact_metric(output: str, references: References) -> MetricResult:
    """``fact`` metric: EM and token F1 (max over references), plus containment and numeric match.

    ``value`` is token F1; ``passed`` is EM or containment or numeric match. Numerically equivalent
    pure-quantity answers score EM = F1 = 1.
    """
    refs = _as_references(references)
    em = exact_match(output, refs)
    f1 = token_f1(output, refs)
    contains = contains_answer(output, refs)
    numeric = any(numeric_match(output, r) for r in refs)
    if numeric and em < 1.0 and _quantity_only(output):
        if any(_quantity_only(r) and numeric_match(output, r) for r in refs):
            em, f1 = 1.0, 1.0
    passed = em == 1.0 or contains == 1.0 or numeric
    return MetricResult(
        Category.FACT,
        "token_f1",
        f1,
        passed,
        {
            "exact_match": em,
            "token_f1": f1,
            "contains": contains,
            "numeric_match": numeric,
            "n_references": len(refs),
        },
    )


# --------------------------------------------------------------------------------------------
# reason: final-answer extraction and accuracy
# --------------------------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\(?:boxed|fbox)\s*\{")
_HASH_RE = re.compile(r"^[ \t]*#{4}[ \t]*(.+?)[ \t]*$", re.MULTILINE)
_MARKER_RE = re.compile(
    r"\b(?:final\s+answer|answer)\b[ \t]*(?:\*{1,2}[ \t]*)?"
    r"(?:is\b|was\b|would\s+be\b|will\s+be\b|should\s+be\b|[:=])"
    r"(?:[ \t]*\*{1,2})?[ \t]*[:=]?",
    re.IGNORECASE,
)
_HEADER_MARKER_RE = re.compile(r"^[ \t#*]*final\s+answer[ \t*:]*$", re.IGNORECASE | re.MULTILINE)
_CLOSER_RE = re.compile(
    r"^\W*(?:let me know|i hope|hope (?:this|that) helps|feel free|good luck|happy to help|"
    r"if you have any|any (?:other|more) questions)",
    re.IGNORECASE,
)
_SENTENCE_END_RE = re.compile(r"[.!?;](?=\s|$)")
_LETTER_REF_RE = re.compile(r"^\(?([A-Ea-e])\)?$")


def _balanced(text: str, open_idx: int) -> str | None:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i]
    return None


def _clean_latex(text: str) -> str:
    for _ in range(3):
        text = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1/\2", text)
        text = re.sub(r"\\(?:text|textbf|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\^\s*\{?\\circ\}?", "°", text)
    text = text.replace("\\%", "%").replace("\\$", "$").replace("\\,", "").replace("\\!", "")
    text = (
        text.replace("\\;", " ").replace("\\ ", " ").replace("\\cdot", "*").replace("\\times", "*")
    )
    text = re.sub(r"\\(?:left|right)", "", text)
    return text.strip()


def _strip_markdown(text: str) -> str:
    text = re.sub(r"[*_`]{1,3}", "", text)
    return text.strip().strip("$").strip() if text.count("$") == 2 else text.strip()


def _first_line(text: str) -> str:
    for line in text.splitlines():
        cleaned = _strip_markdown(line)
        if cleaned.strip(" :=-"):
            return cleaned
    return ""


def _cut_sentence(text: str) -> str:
    m = _SENTENCE_END_RE.search(text)
    return text[: m.start()] if m else text


def extract_final_answer(text: str) -> tuple[str, str] | None:
    """Locate the final-answer span of a free-form solution.

    Priority: last ``\\boxed{...}``; last GSM8K ``#### X`` line; last "(final) answer is/:/= X"
    (rest of that sentence, or the next non-empty line after a bare "Final answer" header); else
    the last non-closing line ("Hope this helps!" lines are skipped).

    Returns:
        ``(span, method)`` with method ``"boxed"``, ``"hash"``, ``"marker"`` or ``"last_line"``,
        or None for empty text.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    boxes = list(_BOXED_RE.finditer(text))
    for box in reversed(boxes):
        inner = _balanced(text, box.end() - 1)
        if inner is not None and inner.strip():
            return _clean_latex(inner), "boxed"
    hashes = list(_HASH_RE.finditer(text))
    if hashes:
        return _strip_markdown(hashes[-1].group(1)), "hash"
    markers = list(_MARKER_RE.finditer(text))
    if markers:
        rest = text[markers[-1].end() :]
        line = _first_line(rest)
        if line:
            return _clean_latex(_cut_sentence(line).strip()), "marker"
    headers = list(_HEADER_MARKER_RE.finditer(text))
    if headers:
        line = _first_line(text[headers[-1].end() :])
        if line:
            return _clean_latex(_cut_sentence(line).strip()), "marker"
    lines = [_strip_markdown(line) for line in text.splitlines()]
    lines = [line for line in lines if line and not _CLOSER_RE.match(line)]
    if not lines:
        return None
    return _clean_latex(lines[-1]), "last_line"


def _compatible(candidates: list[Quantity], ref: Quantity) -> list[Quantity]:
    if ref.kind == "time":
        pool = [q for q in candidates if q.kind == "time"]
    elif ref.dimension is not None:
        pool = [q for q in candidates if q.dimension == ref.dimension]
        pool = pool or [q for q in candidates if q.dimension is None and q.kind != "time"]
    else:
        pool = [q for q in candidates if q.kind != "time"]
    return pool or candidates


def _numeric_answer(text: str, method: str, ref: Quantity) -> tuple[Quantity | None, str]:
    """Pick the answer quantity from a span; for last-line fallback scan lines backwards."""
    if method != "last_line":
        span = text.rsplit("=", 1)[-1] if "=" in text else text
        quantities = parse_quantities(span) or parse_quantities(text)
        pool = _compatible(quantities, ref)
        return (pool[0] if pool else None), span
    return None, text


def _last_line_quantity(output: str, ref: Quantity) -> tuple[Quantity | None, str]:
    lines = [_clean_latex(_strip_markdown(line)) for line in output.splitlines()]
    lines = [line for line in lines if line and not _CLOSER_RE.match(line)]
    for line in reversed(lines):
        if not parse_quantities(line):
            continue
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", line) if parse_quantities(s)]
        sentence = sentences[-1]
        no_parens = re.sub(r"\([^)]*\)", " ", sentence)
        chosen = no_parens if parse_quantities(no_parens) else sentence
        if "=" in chosen and parse_quantities(chosen.rsplit("=", 1)[-1]):
            chosen = chosen.rsplit("=", 1)[-1]
        pool = _compatible(parse_quantities(chosen), ref)
        return (pool[-1] if pool else None), chosen
    return None, ""


def reason_metric(output: str, reference: str) -> MetricResult:
    """``reason`` metric: final-answer accuracy of one output against one reference answer."""
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("reason metric needs a non-empty reference answer")
    found = extract_final_answer(output) if isinstance(output, str) else None
    details: dict[str, Any] = {"reference": reference, "extracted": None, "method": None}
    if found is None:
        return MetricResult(Category.REASON, "final_answer_accuracy", 0.0, False, details)
    span, method = found
    details.update({"extracted": span, "method": method})
    ref_quantities = parse_quantities(reference)
    letter = _LETTER_REF_RE.match(reference.strip())
    correct = False
    if letter:
        wanted = letter.group(1).upper()
        pattern = (
            r"(?<![A-Za-z])\(?([A-E])\)?(?![A-Za-z])"
            if method != "last_line"
            else r"(?:option|choice|answer)\s*(?:is\s*)?[:\-]?\s*\(?([A-E])\)?(?![A-Za-z])"
        )
        m = re.search(pattern, span if method != "last_line" else output, re.IGNORECASE)
        chosen = m.group(1).upper() if m else None
        details["answer_letter"] = chosen
        correct = chosen == wanted
    elif ref_quantities and _quantity_only(reference):
        ref_q = ref_quantities[0]
        if method == "last_line":
            answer_q, chosen_span = _last_line_quantity(output, ref_q)
        else:
            answer_q, chosen_span = _numeric_answer(span, method, ref_q)
        details["extracted"] = chosen_span.strip()
        details["answer_quantity"] = answer_q.text if answer_q else None
        correct = answer_q is not None and quantities_equal(answer_q, ref_q)
    else:
        # Word answers ("Carol", "apples and oranges") or mixed text ("5:15 per km").
        correct = bool(contains_answer(span, reference))
        if not correct and ref_quantities:
            correct = numeric_match(span, reference)
    value = 1.0 if correct else 0.0
    return MetricResult(Category.REASON, "final_answer_accuracy", value, correct, details)


def answer_accuracy(output: str, reference: str) -> float:
    """1.0 if the final answer extracted from ``output`` matches ``reference``, else 0.0."""
    result = reason_metric(output, reference)
    return result.value if result.value is not None else 0.0


# --------------------------------------------------------------------------------------------
# code
# --------------------------------------------------------------------------------------------


def code_metric(
    output: str,
    tests: CodeTests | Mapping[str, Any],
    language: str = "python",
    limits: SandboxLimits | None = None,
) -> MetricResult:
    """``code`` metric: unit-test pass rate of the code in ``output``.

    Non-Python languages return a null result (value None) with ``reason="unsupported_language"``.
    """
    spec = tests if isinstance(tests, CodeTests) else CodeTests.from_record(tests)
    run = run_output_tests(output, spec, language, limits)
    if run is None:
        return _null(Category.CODE, "unsupported_language", language=language)
    return MetricResult(
        Category.CODE,
        "unit_test_pass_rate",
        run.pass_rate,
        run.passed,
        {"language": "python", **run.to_dict()},
    )


# --------------------------------------------------------------------------------------------
# Dispatcher and "not worse"
# --------------------------------------------------------------------------------------------


def task_metric(
    category: Category | str,
    output: str,
    reference: Any,
    *,
    language: str = "python",
    limits: SandboxLimits | None = None,
) -> MetricResult:
    """Compute the automatic task metric for one output.

    Args:
        category: Prompt category.
        output: Target-model output.
        reference: ``fact``: answer string or list of strings. ``reason``: answer string.
            ``code``: :class:`CodeTests` or a HumanEval/MBPP record. Ignored for ``explain``,
            ``creative`` and ``transform``. ``None`` yields a null result.
        language: ``code`` only: the language the prompt asked for.
        limits: ``code`` only: sandbox limits.

    Returns:
        A :class:`MetricResult`; ``value`` is None for rubric-judged categories, missing
        references and non-executable languages.
    """
    cat = parse_category(category)
    if cat in (Category.EXPLAIN, Category.CREATIVE, Category.TRANSFORM):
        return _null(cat, "judged_by_rubric")
    if reference is None or (isinstance(reference, str | list | tuple) and not reference):
        return _null(cat, "no_reference")
    if cat is Category.FACT:
        return fact_metric(output, reference)
    if cat is Category.REASON:
        if not isinstance(reference, str):
            refs = _as_references(reference)
            results = [reason_metric(output, r) for r in refs]
            return max(results, key=lambda r: r.value or 0.0)
        return reason_metric(output, reference)
    return code_metric(output, reference, language, limits)


def not_worse(original: MetricResult, rewrite: MetricResult) -> bool:
    """Metric half of the 2.3.4 label rule: is the rewrite's output not worse than the original's?

    * Null metric on both sides (rubric-only categories, no reference): True; the judge decides.
    * Original null, rewrite scored: True. Original scored, rewrite null (e.g. the rewrite's
      answer switched to a non-executable language): False.
    * ``code``: rewrite pass rate >= original pass rate.
    * ``fact``: EM not lower, F1 not lower by more than :data:`FACT_F1_TOLERANCE`, and correct
      (``passed``) whenever the original was.
    * ``reason``: correct whenever the original was correct.

    Raises:
        ValueError: If the two results belong to different categories.
    """
    if original.category != rewrite.category:
        raise ValueError(
            f"cannot compare {original.category.value!r} with {rewrite.category.value!r}"
        )
    if original.value is None:
        return True
    if rewrite.value is None:
        return False
    if original.passed and not rewrite.passed:
        return False
    if original.category is Category.CODE:
        return rewrite.value >= original.value - 1e-12
    if original.category is Category.FACT:
        em_o = float(original.details.get("exact_match", 0.0))
        em_r = float(rewrite.details.get("exact_match", 0.0))
        return em_r >= em_o and rewrite.value >= original.value - FACT_F1_TOLERANCE - 1e-12
    return True
