"""Operator base: identity passthrough and tidy()."""

from terser.operators.base import Operator, tidy


def test_unchanged_text_is_returned_verbatim() -> None:
    op = Operator("X", "identity", lambda text, target: text)
    raw = "  keep   my  spacing ,exactly "
    assert op(raw) == raw


def test_changed_text_is_tidied() -> None:
    op = Operator("X", "drop um", lambda text, target: text.replace("um", ""))
    assert op("um , explain recursion .") == "Explain recursion."


def test_tidy_repairs_deletion_artifacts() -> None:
    assert tidy("so ,  , explain this . then  show code") == "So, explain this. Then show code"
    assert tidy("line one\n\n\n\nline two") == "Line one\n\nLine two"


def test_tidy_never_touches_payloads() -> None:
    op = Operator("X", "drop um", lambda text, target: text.replace("um ", ""))
    raw = 'um fix this:\n```\nx  =  1 ,y\n```\nand "keep  , this"'
    assert op(raw) == 'Fix this:\n```\nx  =  1 ,y\n```\nAnd "keep  , this"'
