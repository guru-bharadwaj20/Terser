"""Tokenizer drift alert (row 11.5.3): fires on a pinned-version swap, silent otherwise."""

from __future__ import annotations

import logging

import pytest

from terser.monitoring.tokenizer_drift import (
    ALERT_LOGGER,
    check_tokenizer_drift,
    load_fixture,
    main,
)
from terser.tokenizers import count_tokens

STRINGS, SNAPSHOT = load_fixture()


def test_fake_counters() -> None:
    strings = ["a b", "c"]
    same = check_tokenizer_drift(strings, {"t": [2, 1]}, {"t": lambda s: len(s.split())})
    assert not same.alert
    diff = check_tokenizer_drift(strings, {"t": [2, 1]}, {"t": len})
    assert diff.alert and diff.changed["t"] == [(0, 2, 3)]
    missing = check_tokenizer_drift(strings, {}, {"t": len})
    assert missing.alert


@pytest.mark.network
def test_current_gpt4o_matches_snapshot() -> None:
    rep = check_tokenizer_drift(STRINGS, SNAPSHOT, {"gpt-4o": lambda s: count_tokens(s, "gpt-4o")})
    assert not rep.alert and rep.checked == {"gpt-4o": 200}
    assert main(["--target", "gpt-4o"]) == 0


@pytest.mark.network
def test_alert_fires_on_pinned_version_swap(caplog: pytest.LogCaptureFixture) -> None:
    """Swap gpt-4o's pinned o200k_base encoding for its predecessor cl100k_base."""
    import tiktoken

    old = tiktoken.get_encoding("cl100k_base")
    with caplog.at_level(logging.INFO, logger=ALERT_LOGGER):
        rep = check_tokenizer_drift(
            STRINGS, SNAPSHOT, {"gpt-4o": lambda s: len(old.encode(s, disallowed_special=()))}
        )
    assert rep.alert and len(rep.changed["gpt-4o"]) > 0
    crit = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert len(crit) == 1 and "TOKENIZER DRIFT gpt-4o" in crit[0].getMessage()
