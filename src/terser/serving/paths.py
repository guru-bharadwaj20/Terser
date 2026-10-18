"""Fast and quality inference paths and the quality gates they share (rows 11.3.2, 11.3.3).

* **Fast path** (CPU, no GPU ever): a span rewriter followed by the gate. Until the Phase 5/8 span
  tagger is trained, the rewriter is the rule-based operator pipeline
  (:func:`terser.operators.registry.apply_ops` with every operator); a trained tagger exported to
  ONNX plugs in through the same :class:`Rewriter` protocol.
* **Quality path** (GPU): the generative rewriter behind an OpenAI-compatible server (vLLM with
  continuous batching), followed by the same gate.

Gates return ``P(degraded)``; a rewrite whose score exceeds the threshold is rejected and the
original prompt is returned. :class:`ConstraintGate` is a transparent rule-based stand-in (the
share of constraint keywords the rewrite dropped) used until a trained gate is exported;
:class:`OnnxGate` serves a trained gate through ONNX Runtime
(see :mod:`terser.serving.onnx_export`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from terser.features.shared import gate_vector

__all__ = [
    "ConstraintGate",
    "FastPath",
    "Gate",
    "OnnxGate",
    "OpenAICompatibleRewriter",
    "OperatorRewriter",
    "Rewriter",
    "RewriteResult",
]


@dataclass(frozen=True)
class RewriteResult:
    """A rewrite and the rewriter's own token usage (for net-savings accounting)."""

    text: str
    tokens_in: int = 0
    tokens_out: int = 0


class Rewriter(Protocol):
    """Anything that rewrites a prompt for a target."""

    def __call__(self, prompt: str, target: str) -> RewriteResult: ...


class Gate(Protocol):
    """Scores ``P(degraded)`` from original and rewrite feature dicts."""

    def score(self, orig: Mapping[str, float], rew: Mapping[str, float]) -> float: ...


class OperatorRewriter:
    """Rule-based span rewriter: all Phase 1 operators in canonical order."""

    def __init__(self, codes: str | None = None) -> None:
        from terser.operators.registry import CANONICAL_ORDER

        self.codes = codes if codes is not None else CANONICAL_ORDER

    def __call__(self, prompt: str, target: str) -> RewriteResult:
        from terser.operators.registry import apply_ops

        return RewriteResult(apply_ops(prompt, self.codes, target))


class ConstraintGate:
    """Rule-based stand-in gate: fraction of constraint keywords the rewrite lost."""

    def score(self, orig: Mapping[str, float], rew: Mapping[str, float]) -> float:
        before = orig["n_constraints"]
        if before <= 0:
            return 0.0
        return max(0.0, min(1.0, (before - rew["n_constraints"]) / before))


class OnnxGate:
    """A trained gate exported to ONNX; input is :func:`gate_vector`, output ``P(degraded)``."""

    def __init__(self, path: str | Path) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def score_vectors(self, rows: Sequence[Sequence[float]]) -> list[float]:
        """``P(class 1)`` for a batch of gate vectors."""
        import numpy as np

        x = np.asarray(rows, dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: x})
        probs = outputs[1]
        if isinstance(probs, list):  # ZipMap output: list of {label: prob}
            return [float(p[1]) for p in probs]
        return [float(p) for p in np.asarray(probs)[:, 1]]

    def score(self, orig: Mapping[str, float], rew: Mapping[str, float]) -> float:
        return self.score_vectors([gate_vector(orig, rew)])[0]


class FastPath:
    """CPU fast path: span rewriter (operators or ONNX tagger)."""

    def __init__(self, rewriter: Rewriter | None = None) -> None:
        self.rewriter: Rewriter = rewriter if rewriter is not None else OperatorRewriter()

    def __call__(self, prompt: str, target: str) -> RewriteResult:
        return self.rewriter(prompt, target)


class OpenAICompatibleRewriter:
    """Quality path: the generative rewriter served by vLLM's OpenAI-compatible API.

    ``prompt_format="sft"`` sends the SFT training format of :mod:`terser.models.rewriter.train`
    (``<prompt>`` + :data:`SFT_SEP` as a raw completion; the server stops at EOS), which is what
    the Phase 8 rungs were trained on. ``"chat"`` sends an instruction as a system message, for
    a zero-shot instruct model.
    """

    SYSTEM = (
        "Rewrite the user's prompt to use as few tokens as possible while keeping every "
        "requirement, constraint, code block and piece of data. Output only the rewritten prompt."
    )
    SFT_SEP = "\n### Rewrite:\n"

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 2.0,
        max_tokens: int = 512,
        prompt_format: str = "chat",
    ) -> None:
        if prompt_format not in ("chat", "sft"):
            raise ValueError(f"prompt_format must be 'chat' or 'sft', got {prompt_format!r}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.prompt_format = prompt_format
        self._client: Any = None

    def _post(self, url: str, body: dict[str, Any]) -> Any:
        """POST on one pooled client, so requests reuse connections instead of opening one each."""
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout_s)
        return self._client.post(url, json=body, timeout=self.timeout_s)

    def close(self) -> None:
        """Close the pooled HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __call__(self, prompt: str, target: str) -> RewriteResult:
        if self.prompt_format == "sft":
            sft: dict[str, Any] = {
                "model": self.model,
                "temperature": 0.0,
                "max_tokens": self.max_tokens,
                "prompt": prompt + self.SFT_SEP,
            }
            resp = self._post(f"{self.base_url}/completions", sft)
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage") or {}
            return RewriteResult(
                str(data["choices"][0]["text"]).strip(),
                int(usage.get("prompt_tokens", 0)),
                int(usage.get("completion_tokens", 0)),
            )
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": f"{self.SYSTEM} Target model: {target}."},
                {"role": "user", "content": prompt},
            ],
        }
        resp = self._post(f"{self.base_url}/chat/completions", body)
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage") or {}
        return RewriteResult(
            str(data["choices"][0]["message"]["content"]).strip(),
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
        )
