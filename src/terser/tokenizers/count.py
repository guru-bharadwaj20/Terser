"""Ground-truth token counting per target model.

Counts are never approximated: each target uses its real tokenizer.

* ``gpt-4o``    -> tiktoken ``o200k_base``
* ``qwen2.5``   -> Hugging Face ``Qwen/Qwen2.5-7B-Instruct`` tokenizer
* ``llama-3.1`` -> official ``meta-llama/Llama-3.1-8B-Instruct`` when ``HF_TOKEN`` grants access,
  otherwise the ungated mirror ``unsloth/Meta-Llama-3.1-8B-Instruct`` (identical files)
* ``phi3.5``    -> Hugging Face ``microsoft/Phi-3.5-mini-instruct`` tokenizer (ADR-0011)
* ``claude``    -> Anthropic token-counting endpoint (needs ``ANTHROPIC_API_KEY``); model from
  ``TERSER_CLAUDE_MODEL`` (default ``claude-sonnet-5``)

Counts exclude special tokens and chat-template framing, so savings ratios reflect the prompt
text only. For Claude the endpoint counts a full user message, so the framing overhead is
calibrated once (see :func:`calibrate_overhead`) and subtracted. Claude counts are cached on
disk under ``$TERSER_CACHE/token_counts`` (default ``.cache``).

CLI: ``python -m terser.tokenizers.count "some text"``
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

LLAMA_OFFICIAL = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_MIRROR = "unsloth/Meta-Llama-3.1-8B-Instruct"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5"
CALIBRATION_UNIT = "hello\n"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Target:
    """A countable target: ``kind`` selects the backend, ``ref`` its encoding/repo."""

    name: str
    kind: str  # "tiktoken" | "hf" | "anthropic"
    ref: str


TARGETS: dict[str, Target] = {
    "gpt-4o": Target("gpt-4o", "tiktoken", "o200k_base"),
    "qwen2.5": Target("qwen2.5", "hf", "Qwen/Qwen2.5-7B-Instruct"),
    "llama-3.1": Target("llama-3.1", "hf", LLAMA_MIRROR),
    "phi3.5": Target("phi3.5", "hf", "microsoft/Phi-3.5-mini-instruct"),
    "claude": Target("claude", "anthropic", DEFAULT_CLAUDE_MODEL),
}


def cache_dir() -> Path:
    """Directory for cached Claude counts (read from ``TERSER_CACHE`` at call time)."""
    return Path(os.environ.get("TERSER_CACHE", ".cache")) / "token_counts"


def claude_model() -> str:
    """Claude model used for counting (``TERSER_CLAUDE_MODEL``)."""
    return os.environ.get("TERSER_CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)


def load_hf_tokenizer(repo: str) -> Any:
    """Load an HF tokenizer; for Llama prefer the official gated repo when accessible."""
    from tokenizers import Tokenizer

    token = os.environ.get("HF_TOKEN")
    if repo == LLAMA_MIRROR and token:
        try:
            return Tokenizer.from_pretrained(LLAMA_OFFICIAL, token=token)
        except Exception as exc:  # noqa: BLE001 - gated/no access/offline: use the mirror
            log.info("official Llama tokenizer unavailable (%s); using %s", exc, LLAMA_MIRROR)
    return Tokenizer.from_pretrained(repo, token=token)


def calibrate_overhead(raw: Callable[[str], int], unit: str = CALIBRATION_UNIT) -> int:
    """Estimate the fixed framing tokens the endpoint adds around a user message.

    The endpoint returns ``raw(text) = F + T(text)`` and rejects empty content, so ``F`` cannot
    be read off directly, and assuming some probe (e.g. ``"a"``) is exactly one token is
    unverifiable. Instead we probe ``unit * k`` for ``k = 1, 2, 3``. BPE tokenizers pre-split on
    word/newline boundaries, so repeating a self-delimiting unit (``"hello\\n"``) gives
    ``T(unit * k) = k * d``: the counts must form an arithmetic sequence, and extrapolating the
    line to ``k = 0`` yields ``F = raw(unit) - d``. Linearity is checked (equal successive
    differences); if it fails, tokens merged across units and we refuse to guess. This holds
    with zero intercept for o200k, Qwen2.5 and Llama-3.1 (2, 4, 6 tokens), which is the evidence
    the unit is self-delimiting.
    """
    r1, r2, r3 = (raw(unit * k) for k in (1, 2, 3))
    d = r2 - r1
    if d <= 0 or r3 - r2 != d:
        raise RuntimeError(f"Claude overhead calibration not linear: {r1}, {r2}, {r3}")
    overhead = r1 - d
    if overhead < 0:
        raise RuntimeError(f"Claude overhead calibration negative: {overhead}")
    return overhead


def _claude_counter(model: str) -> Callable[[str], int]:
    client: Any = None

    def raw(text: str) -> int:
        nonlocal client
        key = hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()
        path = cache_dir() / f"{key}.json"
        if path.exists():
            return int(json.loads(path.read_text(encoding="utf-8"))["input_tokens"])
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        resp = client.messages.count_tokens(
            model=model, messages=[{"role": "user", "content": text}]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"input_tokens": resp.input_tokens}), encoding="utf-8")
        return int(resp.input_tokens)

    overhead = calibrate_overhead(raw)
    return lambda text: raw(text) - overhead if text else 0


@cache
def _encoder(name: str, ref: str) -> Callable[[str], int]:
    kind = TARGETS[name].kind
    if kind == "tiktoken":
        import tiktoken

        enc = tiktoken.get_encoding(ref)
        return lambda text: len(enc.encode(text, disallowed_special=()))
    if kind == "hf":
        tok = load_hf_tokenizer(ref)
        return lambda text: len(tok.encode(text, add_special_tokens=False).ids)
    if kind == "anthropic":
        return _claude_counter(ref)
    raise ValueError(f"unknown tokenizer kind: {kind}")


def available_targets() -> list[str]:
    """Targets usable in this environment (Claude needs ``ANTHROPIC_API_KEY``)."""
    names = [n for n, t in TARGETS.items() if t.kind != "anthropic"]
    if os.environ.get("ANTHROPIC_API_KEY"):
        names.append("claude")
    return names


def count_tokens(text: str, target: str) -> int:
    """Number of tokens ``text`` costs on ``target``, excluding special/framing tokens."""
    if target not in TARGETS:
        raise KeyError(f"unknown target {target!r}; choose from {sorted(TARGETS)}")
    ref = claude_model() if TARGETS[target].kind == "anthropic" else TARGETS[target].ref
    return _encoder(target, ref)(text)


def main(argv: list[str] | None = None) -> None:
    """Print the token count of the given text for every available target."""
    from dotenv import load_dotenv

    load_dotenv()
    args = sys.argv[1:] if argv is None else argv
    sample = " ".join(args) or "hey so can you like explain recursion to me"
    for name in available_targets():
        print(f"{name:10s} {count_tokens(sample, name)}")


if __name__ == "__main__":
    main()
