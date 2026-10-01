"""Corpus counts for the Bayes-theorem worked example (plan row **5.2.1**).

Silver labels: a word occurrence is *filler* when the deterministic ``F`` operator
(:mod:`terser.operators.filler`) deletes it. Deletions are found by aligning the word sequences
before and after ``F`` with :class:`difflib.SequenceMatcher`. Context is a binary feature:
``initial`` (first word of the prompt or of a sentence, or right after a comma) vs ``medial``.

For each phrase ``w`` and context ``c`` the module reports the counts that Bayes' theorem needs,

    P(filler | w, c) = P(w, c | filler) P(filler) / P(w, c),

and checks the identity numerically (both sides computed from the same counts).
``python -m terser.phase5.bayes_counts --prompts data/pilot/prompts.jsonl`` prints the table used
in ``docs/derivations/bayes.md``.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from terser.operators.filler import filler

__all__ = ["PHRASES", "BayesRow", "bayes_table", "count_occurrences", "main"]

PHRASES: tuple[str, ...] = ("like", "so", "um", "basically", "literally", "honestly", "well")
_WORD = re.compile(r"[A-Za-z']+|[.,!?;:]")


def _words(text: str) -> list[str]:
    return _WORD.findall(text)


def count_occurrences(texts: Iterable[str]) -> tuple[Counter[tuple[str, str, bool]], int, int]:
    """Count ``(word, context, is_filler)`` over all word tokens.

    Returns the counter, the total number of word tokens and the number of filler tokens.
    """
    counts: Counter[tuple[str, str, bool]] = Counter()
    total = n_filler = 0
    for text in texts:
        before, after = _words(text), _words(filler(text))
        deleted: set[int] = set()
        sm = difflib.SequenceMatcher(a=[w.lower() for w in before], b=[w.lower() for w in after])
        for op, i1, i2, _j1, _j2 in sm.get_opcodes():
            if op in ("delete", "replace"):
                deleted.update(range(i1, i2))
        prev = None
        for i, w in enumerate(before):
            if not w[0].isalpha():
                prev = w
                continue
            ctx = "initial" if prev is None or prev in {".", "!", "?", ","} else "medial"
            is_f = i in deleted
            counts[(w.lower(), ctx, is_f)] += 1
            total += 1
            n_filler += is_f
            prev = w
    return counts, total, n_filler


@dataclass(frozen=True)
class BayesRow:
    phrase: str
    context: str
    n_phrase_ctx: int  # count(w, c)
    n_phrase_ctx_filler: int  # count(w, c, filler)
    p_wc_given_filler: float  # P(w, c | filler)
    p_filler: float  # P(filler)
    p_wc: float  # P(w, c)
    posterior: float  # Bayes: P(w,c|F) P(F) / P(w,c)
    direct: float  # count(w, c, filler) / count(w, c)


def bayes_table(texts: Iterable[str], phrases: Sequence[str] = PHRASES) -> list[BayesRow]:
    """One row per (phrase, context) with non-zero count."""
    counts, total, n_f = count_occurrences(texts)
    if total == 0 or n_f == 0:
        raise ValueError("corpus has no words or no filler deletions")
    rows = []
    for w in phrases:
        for ctx in ("initial", "medial"):
            n_wc = counts[(w, ctx, True)] + counts[(w, ctx, False)]
            if n_wc == 0:
                continue
            n_wcf = counts[(w, ctx, True)]
            p_wc_f, p_f, p_wc = n_wcf / n_f, n_f / total, n_wc / total
            rows.append(
                BayesRow(w, ctx, n_wc, n_wcf, p_wc_f, p_f, p_wc, p_wc_f * p_f / p_wc, n_wcf / n_wc)
            )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m terser.phase5.bayes_counts", description=__doc__)
    ap.add_argument("--prompts", type=Path, default=Path("data/pilot/prompts.jsonl"))
    args = ap.parse_args(argv)
    texts = [
        json.loads(line)["prompt"]
        for line in args.prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts, total, n_f = count_occurrences(texts)
    print(f"prompts={len(texts)} word_tokens={total} filler_tokens={n_f} P(filler)={n_f/total:.5f}")
    print(
        "| phrase | context | count(w,c) | count(w,c,F) | P(w,c given F) | P(w,c) "
        "| P(F given w,c) |"
    )
    print("|---|---|---:|---:|---:|---:|---:|")
    for r in bayes_table(texts):
        print(
            f"| {r.phrase} | {r.context} | {r.n_phrase_ctx} | {r.n_phrase_ctx_filler} | "
            f"{r.p_wc_given_filler:.5f} | {r.p_wc:.5f} | {r.posterior:.3f} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
