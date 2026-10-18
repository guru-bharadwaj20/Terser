"""Tokenizer drift alert (plan row **11.5.3**, checkpoint C11.6).

Every reported saving is a token count on a target's tokenizer. If a tokenizer changes (a new
``tiktoken`` encoding, an updated HF ``tokenizer.json``, a library upgrade that changes
pre-tokenisation), every stored count and every model trained on token features silently goes
stale. This job re-counts the fixed fixture set (``data/monitoring/tokenizer_strings.jsonl``, 200
strings) with the tokenizers installed *now* and compares them with the pinned snapshot
(``data/monitoring/tokenizer_counts.json``). **Any** changed count on any target raises an alert.

The alert is a ``CRITICAL`` record on the ``terser.alerts`` logger (picked up by the log shipper /
Alertmanager) and exit code 1 from the CLI, which fails the scheduled ``drift_flow``
(:mod:`terser.flows`)::

    python -m terser.monitoring.tokenizer_drift               # all local targets
    python -m terser.monitoring.tokenizer_drift --target gpt-4o
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from terser.config import project_root

__all__ = [
    "ALERT_LOGGER",
    "DriftReport",
    "check_tokenizer_drift",
    "load_fixture",
]

ALERT_LOGGER = "terser.alerts"
# The reference set lives with the monitoring data, not in tests/ (the tests read it here too).
_FIXTURES = project_root() / "data" / "monitoring"
STRINGS_PATH = _FIXTURES / "tokenizer_strings.jsonl"
SNAPSHOT_PATH = _FIXTURES / "tokenizer_counts.json"


@dataclass
class DriftReport:
    """Per-target count differences between the snapshot and the current tokenizers."""

    checked: dict[str, int] = field(default_factory=dict)
    changed: dict[str, list[tuple[int, int, int]]] = field(default_factory=dict)

    @property
    def alert(self) -> bool:
        return any(self.changed.values())

    def summary(self) -> str:
        if not self.alert:
            return f"no tokenizer drift on {sorted(self.checked)}"
        parts = [
            f"{t}: {len(c)}/{self.checked[t]} counts changed (e.g. string {c[0][0]}: "
            f"{c[0][1]} -> {c[0][2]})"
            for t, c in sorted(self.changed.items())
            if c
        ]
        return "TOKENIZER DRIFT " + "; ".join(parts)


def load_fixture(
    strings_path: Path = STRINGS_PATH, snapshot_path: Path = SNAPSHOT_PATH
) -> tuple[list[str], dict[str, list[int]]]:
    """Fixture strings and the pinned per-target counts."""
    lines = strings_path.read_text(encoding="utf-8").split("\n")
    strings = [json.loads(x)["text"] for x in lines if x]
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    return strings, {k: [int(v) for v in vs] for k, vs in snapshot.items()}


def check_tokenizer_drift(
    strings: Sequence[str],
    snapshot: Mapping[str, Sequence[int]],
    counters: Mapping[str, Callable[[str], int]],
    logger: logging.Logger | None = None,
) -> DriftReport:
    """Re-count ``strings`` with ``counters`` and compare to ``snapshot``; log an alert on change.

    A target missing from the snapshot, or a snapshot of the wrong length, is itself drift.
    """
    report = DriftReport()
    for target, count in counters.items():
        pinned = list(snapshot.get(target, []))
        now = [count(s) for s in strings]
        report.checked[target] = len(strings)
        if len(pinned) != len(now):
            report.changed[target] = [(-1, len(pinned), len(now))]
            continue
        report.changed[target] = [
            (i, a, b) for i, (a, b) in enumerate(zip(pinned, now, strict=True)) if a != b
        ]
    log = logger or logging.getLogger(ALERT_LOGGER)
    if report.alert:
        log.critical(report.summary())
    else:
        log.info(report.summary())
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: exit 1 when any count changed."""
    from terser.tokenizers import count_tokens

    ap = argparse.ArgumentParser(description="Tokenizer drift check (row 11.5.3)")
    ap.add_argument("--target", action="append", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    strings, snapshot = load_fixture()
    targets = args.target or sorted(snapshot)
    counters: dict[str, Callable[[str], int]] = {
        t: (lambda s, t=t: count_tokens(s, t)) for t in targets  # type: ignore[misc]
    }
    report = check_tokenizer_drift(strings, snapshot, counters)
    print(report.summary())
    return 1 if report.alert else 0


if __name__ == "__main__":
    raise SystemExit(main())
