"""Row 3.1.1: alignment labels against 20 hand-aligned (original, rewrite) pairs.

Each expected label string was written by hand, one letter per original token
(K = KEEP, D = DROP, R = REPLACE), before running the aligner.
"""

from __future__ import annotations

import pytest

from terser.features.alignment import DROP, KEEP, REPLACE, align, tokenize

_CODE = {"K": KEEP, "D": DROP, "R": REPLACE}

# (original, rewrite, hand labels)
HAND_ALIGNED: list[tuple[str, str, str]] = [
    # 1. pure filler deletion at the start
    ("hey so can you sort this list", "can you sort this list", "D D K K K K K"),
    # 2. identical
    ("Sort the list.", "Sort the list.", "K K K K"),
    # 3. case-only change keeps tokens
    ("please Sort the list", "sort the list", "D K K K"),
    # 4. trailing politeness dropped
    ("write a haiku about rain thanks so much!", "write a haiku about rain", "K K K K K D D D D"),
    # 5. hedge dropped in the middle
    ("explain recursion if possible simply", "explain recursion simply", "K K D D K"),
    # 6. phrase replaced by a shorter word
    ("write it in a short amount of time", "write it quickly", "K K R R R R R R"),
    # 7. replacement with a single synonym
    ("make a list of fruits", "list fruits", "D D K D K"),
    # 8. imperative rewrite of a question
    ("could you translate this to French?", "translate this to French.", "D D K K K K R"),
    # 9. meta talk removed
    ("I want you to summarise the article", "summarise the article", "D D D D K K K"),
    # 10. duplicate requirement removed
    (
        "use python and make sure to use python",
        "use python",
        "K K D D D D D D",
    ),
    # 11. everything dropped but the core noun phrase
    ("um basically just fix the bug lol", "fix the bug", "D D D K K K D"),
    # 12. number kept, hedges dropped
    ("give me like five examples", "give five examples", "K D D K K"),
    # 13. contraction kept as one token
    ("don't use recursion please", "don't use recursion", "K K K D"),
    # 14. punctuation dropped
    ("so, what is a monad?", "what is a monad?", "D D K K K K K"),
    # 15. adjective dropped
    ("compute the total sum", "compute the sum", "K K D K"),
    # 16. two separated deletions
    ("hi can you really quickly list primes thanks", "list primes", "D D D D D K K D"),
    # 17. "in order to" shortened to "to"
    ("in order to sort it use merge sort", "to sort it use merge sort", "D D K K K K K K"),
    # 18. negation preserved while hedge dropped
    ("I think do not use globals", "do not use globals", "D D K K K K"),
    # 19. replace a phrase in the middle
    ("return the answer in the form of json", "return the answer as json", "K K K R R R R K"),
    # 20. empty rewrite drops all
    ("thanks!", "", "D D"),
]


def test_twenty_hand_aligned_pairs() -> None:
    assert len(HAND_ALIGNED) == 20


@pytest.mark.parametrize(("original", "rewrite", "expected"), HAND_ALIGNED)
def test_matches_hand_alignment(original: str, rewrite: str, expected: str) -> None:
    result = align(original, rewrite)
    want = [_CODE[c] for c in expected.split()]
    assert len(want) == len(result.original), result.original
    assert result.labels == want


def test_tokenize_splits_punctuation_keeps_apostrophes() -> None:
    assert tokenize("don't stop, ok?") == ["don't", "stop", ",", "ok", "?"]


def test_replacements_and_insertions_recorded() -> None:
    r = align("write it in a short amount of time", "write it quickly")
    assert r.replacements == {2: ["quickly"]}
    ins = align("sort list", "sort the list")
    assert ins.labels == [KEEP, KEEP]
    assert ins.inserted == ["the"]


def test_drop_ratio() -> None:
    assert align("a b c d", "a b").drop_ratio() == 0.5
    assert align("", "").drop_ratio() == 0.0
