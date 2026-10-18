"""Regressions: operator rewrites that used to change the meaning of a prompt."""

from terser.operators.dedupe import dedupe
from terser.operators.filler import filler
from terser.operators.hedging import hedging
from terser.operators.substitute import substitute


def test_dedupe_keeps_a_negation_with_different_scope() -> None:
    text = (
        "Use numpy for math and do not use matplotlib for plotting. "
        "Do not use numpy for plotting."
    )
    assert "Do not use numpy for plotting" in dedupe(text)


def test_dedupe_still_drops_a_repeated_negation() -> None:
    text = "Do not use global variables in the module. Never use global variables in the module."
    assert dedupe(text).count("global variables") == 1


def test_substitute_majority_keeps_of() -> None:
    out = substitute("The majority of the votes were counted.")
    assert out == "Most of the votes were counted."
    assert substitute("The majority of people agree.") == "Most people agree."


def test_substitute_required_to_is_not_rewritten() -> None:
    text = "What is required to install Python?"
    assert substitute(text) == text


def test_substitute_keeps_decision_tree() -> None:
    assert "decision tree" in substitute("Help me make a decision tree for churn.")
    assert substitute("We must make a decision now.") == "We must decide now."


def test_filler_keeps_micrometre_unit() -> None:
    assert "10 um wide" in filler("The channel is 10 um wide.")
    assert "um" not in filler("How do I, um, sort a list?").lower().split()


def test_filler_keeps_intensifier_as_topic() -> None:
    assert "literally" in filler("What does literally mean here?")
    assert "totally ordered set" in filler("Is this a totally ordered set?")
    assert "literally" not in filler("I literally cannot get this to compile.")


def test_hedging_keeps_please_as_verb() -> None:
    assert "please my boss" in hedging("How do I please my boss?")
    assert "please" not in hedging("Please sort this list.").lower()
