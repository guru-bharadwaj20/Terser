"""``prompt_has_code`` detects code, not prose that happens to contain "def"."""

from __future__ import annotations

import pytest

from terser.phase3.build_features import has_code


@pytest.mark.parametrize(
    "text",
    [
        "fix this:\n```python\nprint(1)\n```",
        "write def add(a, b): that returns the sum",
        "here is my loop\n    for i in range(3):",
        "class Stack: with push and pop",
        "#include <stdio.h> please explain",
    ],
)
def test_code(text: str) -> None:
    assert has_code(text)


@pytest.mark.parametrize(
    "text",
    [
        "please define photosynthesis",
        "I'm def gonna need a recipe",
        "what is the default setting for brightness",
    ],
)
def test_prose(text: str) -> None:
    assert not has_code(text)
