"""Tests for terser.judge.metrics (plan row 2.3.1): fact EM/F1, reason final answers, dispatch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from terser.data.categories import Category
from terser.judge.code_runner import CodeTests
from terser.judge.metrics import (
    MetricResult,
    answer_accuracy,
    exact_match,
    extract_final_answer,
    fact_metric,
    normalize_answer,
    not_worse,
    numeric_match,
    parse_quantities,
    reason_metric,
    task_metric,
    token_f1,
)

REASON_PROMPTS = (
    Path(__file__).resolve().parent.parent / "data" / "pilot" / "prompts" / "reason.jsonl"
)


# ----------------------------------------------------------------------------------------------
# fact
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The Eiffel Tower!", "eiffel tower"),
        ("  An   apple, a day ", "apple day"),
        ("“Pride and Prejudice”", "pride and prejudice"),
        ("well-known", "well known"),
        ("ＣＡＮＢＥＲＲＡ", "canberra"),
        ("100 °C", "100 °c"),
    ],
)
def test_normalize_answer(text: str, expected: str) -> None:
    assert normalize_answer(text) == expected


def test_exact_match_and_f1_max_over_references() -> None:
    refs = ["Jane Austen", "Austen"]
    assert exact_match("austen.", refs) == 1.0
    assert exact_match("Charlotte Bronte", refs) == 0.0
    assert token_f1("Jane Austen", refs) == 1.0
    # "the author Jane Austen" -> tokens [author, jane, austen] vs [jane, austen]: P=2/3, R=1.
    assert token_f1("the author Jane Austen", "Jane Austen") == pytest.approx(0.8)
    assert token_f1("the author Jane Austen", refs) == pytest.approx(0.8)
    assert token_f1("Bronte", refs) == 0.0


def test_references_must_be_non_empty() -> None:
    with pytest.raises(ValueError):
        exact_match("x", [])


def test_fact_metric_containment_passes_sentence_answers() -> None:
    result = fact_metric("The capital of Australia is Canberra.", ["Canberra"])
    assert result.name == "token_f1" and result.category is Category.FACT
    assert result.details["exact_match"] == 0.0 and result.details["contains"] == 1.0
    assert result.passed is True
    assert 0.0 < (result.value or 0.0) < 1.0
    assert fact_metric("It is Sydney.", ["Canberra"]).passed is False


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("Water boils at 212 °F at sea level.", "100 °C", True),
        ("about 212 degrees Fahrenheit", "100 degrees Celsius", True),
        ("It is 3.5 kg", "3500 g", True),
        ("roughly 26.2 miles", "42.195 km", True),
        ("5 feet", "5 meters", False),
        ("100 °C", "212 °F and 100 °C", True),  # same temperature: one quantity covers both
        ("100 °C", "100 °C and 30 kg", False),
        ("1,440 minutes", "1440 minutes", True),
        ("no numbers here", "Canberra", False),
    ],
)
def test_numeric_match_with_units(prediction: str, reference: str, expected: bool) -> None:
    assert numeric_match(prediction, reference) is expected


def test_fact_numeric_equivalents_score_full() -> None:
    result = fact_metric("212 F", "100 °C")
    assert result.details["exact_match"] == 1.0 and result.value == 1.0 and result.passed


# ----------------------------------------------------------------------------------------------
# quantities
# ----------------------------------------------------------------------------------------------


def test_parse_quantities_kinds() -> None:
    qs = parse_quantities("Pay $2,315.25 at 7:15 AM, 30% off, 5/14 of 72 cm² and 1818.")
    kinds = [(q.kind, q.value) for q in qs]
    assert kinds[0] == ("currency", 2315.25)
    assert kinds[1] == ("time", pytest.approx(7.25))
    assert kinds[2] == ("percent", 30.0)
    assert kinds[3] == ("fraction", pytest.approx(5 / 14))
    assert qs[4].dimension == "area" and qs[4].canonical == pytest.approx(72e-4)
    assert kinds[5] == ("number", 1818.0)


def test_fraction_at_sentence_end_is_a_fraction() -> None:
    (q,) = parse_quantities("Therefore, 3/8.")
    assert q.kind == "fraction" and q.value == pytest.approx(0.375)


# ----------------------------------------------------------------------------------------------
# reason
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "span", "method"),
    [
        (r"so x = 4 and \boxed{\frac{5}{14}}", "5/14", "boxed"),
        ("Natalia sold 48 clips.\n#### 72", "72", "hash"),
        ("Working...\nThe answer is $50. Let me know!", "$50", "marker"),
        ("Steps\n\n**Final answer:** 7pm", "7pm", "marker"),
        ("Final Answer\n\n42", "42", "marker"),
        (
            "First 3 + 4 = 7.\nSo Tom has 6 apples.\nHope this helps!",
            "So Tom has 6 apples.",
            "last_line",
        ),
    ],
)
def test_extract_final_answer(text: str, span: str, method: str) -> None:
    assert extract_final_answer(text) == (span, method)


def test_extract_final_answer_empty() -> None:
    assert extract_final_answer("   ") is None


@pytest.mark.parametrize(
    ("output", "reference", "correct"),
    [
        ("Distance gap is 60 miles, closing at 20 mph, so 3 hours after 4pm: 7 PM.", "7pm", True),
        ("The second train catches up at 7:00 p.m.", "7pm", True),
        ("The answer is 7am.", "7pm", False),
        ("P = 5/8 * 4/7 = 20/56 = 5/14", "5/14", True),
        ("The probability is about 0.357.", "5/14", True),
        ("The probability is about 0.4.", "5/14", False),
        ("Answer: 6/16", "3/8", True),
        ("original price = 40 / 0.8 = $50.00", "$50", True),
        ("The answer is 50 dollars.", "$50", True),
        ("So the answer is 45.", "$50", False),
        ("That is 0.3 of the total, i.e. the answer is 30 percent.", "30%", True),
        ("The answer is 2,315.25", "$2,315.25", True),
        ("Total: 120 minutes", "2 hours", True),
        ("Final answer: 8 km", "8.0 km", True),
        ("Final answer: 77 °F", "25 °C", True),
        ("The answer is 11.3 feet", "11.3 ft", True),
        ("It is 11.31 ft.", "11.3 ft", True),
        ("She finishes at 7:15 in the morning. The answer is 7:15 AM.", "7:15 AM", True),
        ("Bob > Carol > Dan, so the second shortest is Carol.", "Carol", True),
        ("The answer is Bob.", "Carol", False),
        ("You should answer: (B)", "B", True),
        ("The answer is (C).", "B", False),
        ("My pace is 5:15 per km.", "5:15 per km", True),
        ("", "42", False),
    ],
)
def test_answer_accuracy(output: str, reference: str, correct: bool) -> None:
    assert answer_accuracy(output, reference) == (1.0 if correct else 0.0)


def _pilot_reason_answers() -> list[str]:
    with REASON_PROMPTS.open(encoding="utf-8") as fh:
        return sorted({json.loads(line)["answer"] for line in fh if line.strip()})


PILOT_ANSWERS = _pilot_reason_answers()


@pytest.mark.parametrize(
    "template",
    [
        "Step 1: 3 + 4 = 7.\nStep 2: 7 * 12 = 84.\nSo the answer is {a}.",
        "Some working with 999.\n\n**Final answer:** {a}",
        "Let x = 5. Then 2x = 10.\nTherefore, {a}.\n\nHope this helps!",
        "\\boxed{{{a}}}",
    ],
)
def test_pilot_reason_answer_formats(template: str) -> None:
    failures = [a for a in PILOT_ANSWERS if not reason_metric(template.format(a=a), a).passed]
    assert failures == []


def test_pilot_reason_wrong_answers_fail() -> None:
    wrong = "Working: 1 + 2 = 3.\nThe answer is 987654."
    assert [a for a in PILOT_ANSWERS if reason_metric(wrong, a).passed] == []


def test_reason_metric_requires_reference() -> None:
    with pytest.raises(ValueError):
        reason_metric("42", "")


# ----------------------------------------------------------------------------------------------
# dispatcher and not_worse
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("category", ["explain", "creative", "transform"])
def test_rubric_only_categories_return_null_metric(category: str) -> None:
    result = task_metric(category, "anything", "ref")
    assert result.name is None and result.value is None and result.passed is None
    assert result.is_null and result.details["reason"] == "judged_by_rubric"
    json.dumps(result.to_dict())


def test_task_metric_dispatch() -> None:
    assert task_metric("fact", "Canberra", ["Canberra"]).name == "token_f1"
    reason = task_metric(" Reason ", "The answer is 12.", "12")
    assert reason.name == "final_answer_accuracy" and reason.value == 1.0
    assert task_metric("reason", "The answer is 12.", ["13", "12"]).value == 1.0
    assert task_metric("fact", "x", None).details["reason"] == "no_reference"
    with pytest.raises(ValueError):
        task_metric("poetry", "x", "y")


def test_task_metric_code_non_python_is_null() -> None:
    tests = CodeTests.mbpp(["assert True"])
    result = task_metric(Category.CODE, "```sql\nSELECT 1;\n```", tests, language="sql")
    assert result.value is None and result.details["reason"] == "unsupported_language"


def test_task_metric_code_runs_tests() -> None:
    output = "Here you go:\n```python\ndef add(a, b):\n    return a + b\n```"
    record = {"test_list": ["assert add(1, 2) == 3", "assert add(2, 2) == 5"]}
    result = task_metric("code", output, record)
    assert result.name == "unit_test_pass_rate"
    assert result.value == 0.5 and result.passed is False
    assert result.details["per_test"] == [True, False]


def _r(cat: Category, value: float | None, passed: bool | None, **details: float) -> MetricResult:
    return MetricResult(cat, "m", value, passed, dict(details))


def test_not_worse_rules() -> None:
    code, fact, reason = Category.CODE, Category.FACT, Category.REASON
    assert not_worse(_r(code, 0.5, False), _r(code, 0.5, False))
    assert not not_worse(_r(code, 1.0, True), _r(code, 0.8, False))
    assert not_worse(_r(code, 0.2, False), _r(code, 1.0, True))
    # fact: EM not lower and F1 within 0.05.
    assert not_worse(_r(fact, 0.80, True, exact_match=0.0), _r(fact, 0.76, True, exact_match=0.0))
    assert not not_worse(
        _r(fact, 0.80, True, exact_match=0.0), _r(fact, 0.70, True, exact_match=0.0)
    )
    assert not not_worse(_r(fact, 1.0, True, exact_match=1.0), _r(fact, 1.0, True, exact_match=0.0))
    assert not not_worse(
        _r(fact, 0.5, True, exact_match=0.0), _r(fact, 0.9, False, exact_match=0.0)
    )
    # reason: correct whenever the original was.
    assert not not_worse(_r(reason, 1.0, True), _r(reason, 0.0, False))
    assert not_worse(_r(reason, 0.0, False), _r(reason, 0.0, False))
    # null handling
    assert not_worse(_r(Category.EXPLAIN, None, None), _r(Category.EXPLAIN, None, None))
    assert not not_worse(_r(code, 1.0, True), _r(code, None, None))
    assert not_worse(_r(code, None, None), _r(code, 0.0, False))
    with pytest.raises(ValueError):
        not_worse(_r(code, 1.0, True), _r(fact, 1.0, True))


def test_normalize_answer_keeps_sign_and_decimal_point() -> None:
    from terser.judge.metrics import fact_metric

    assert normalize_answer("-5") == "-5"
    assert normalize_answer("3.5 kg.") == "3.5 kg"
    assert not fact_metric("-5", "5").passed
    assert not fact_metric("3.5", "35").passed
    assert fact_metric("The answer is -5.", "-5").passed


def test_one_letter_units_need_to_touch_the_number() -> None:
    from terser.judge.metrics import fact_metric

    assert fact_metric("30c", "30 °C").passed
    assert fact_metric("Answer: 3 f", "3").passed  # "f" here is not Fahrenheit
