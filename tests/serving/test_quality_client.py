"""Quality-path client request formats (row 11.3.3)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from terser.serving.api import build_service, load_config
from terser.serving.paths import OpenAICompatibleRewriter


class _Resp:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._data


def test_sft_format_posts_raw_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, Any] = {}

    def post(url: str, json: dict[str, Any], timeout: float) -> _Resp:
        sent.update(url=url, body=json, timeout=timeout)
        usage = {"prompt_tokens": 9, "completion_tokens": 3}
        return _Resp({"choices": [{"text": " Sort the list. "}], "usage": usage})

    monkeypatch.setattr(httpx, "post", post)
    rw = OpenAICompatibleRewriter("http://w:8001/v1/", "terser-rewriter", prompt_format="sft")
    out = rw("hey could you sort this list pls", "qwen2.5")
    assert sent["url"] == "http://w:8001/v1/completions"
    assert sent["body"]["prompt"] == "hey could you sort this list pls\n### Rewrite:\n"
    assert sent["body"]["temperature"] == 0.0 and "messages" not in sent["body"]
    assert (out.text, out.tokens_in, out.tokens_out) == ("Sort the list.", 9, 3)


def test_chat_format_is_default(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, Any] = {}

    def post(url: str, json: dict[str, Any], timeout: float) -> _Resp:
        sent.update(url=url, body=json)
        return _Resp({"choices": [{"message": {"content": "x"}}]})

    monkeypatch.setattr(httpx, "post", post)
    OpenAICompatibleRewriter("http://w/v1", "m")("p", "gpt-4o")
    assert (
        sent["url"].endswith("/chat/completions") and sent["body"]["messages"][1]["content"] == "p"
    )


def test_bad_format_rejected() -> None:
    with pytest.raises(ValueError):
        OpenAICompatibleRewriter("http://w/v1", "m", prompt_format="raw")


def test_quality_config_wires_sft_rewriter(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("onnxruntime")
    from terser.serving.api import REPO_ROOT

    svc = build_service(load_config(REPO_ROOT / "configs/serving/quality.yaml"))
    assert isinstance(svc.quality, OpenAICompatibleRewriter)
    assert svc.quality.prompt_format == "sft" and svc.quality.model == "terser-rewriter"
