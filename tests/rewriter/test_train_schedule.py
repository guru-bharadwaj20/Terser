"""Row 10.7 trainer additions: warmup + linear-decay schedule and the validation-loss hook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from terser.models.rewriter.train import (
    ByteTokenizer,
    build_tiny_lm,
    encode_pair,
    eval_loss,
    load_pairs,
    lr_factor,
    train_sft,
)

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "samples" / "casual_prompts.jsonl"


def test_lr_factor_warmup_then_linear_decay() -> None:
    f = [lr_factor(s, 10, 4, "linear") for s in range(10)]
    assert f[:4] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert f[4] == pytest.approx(1.0) and f[-1] == pytest.approx(1 / 6)
    assert all(a >= b for a, b in zip(f[3:], f[4:], strict=False))
    assert [lr_factor(s, 10, 0, "linear") for s in (0, 9)] == pytest.approx([1.0, 0.1])
    assert [lr_factor(s, 10, 2, "constant") for s in (0, 1, 2, 9)] == [0.5, 1.0, 1.0, 1.0]
    assert all(lr_factor(s, 10) == 1.0 for s in range(10))
    with pytest.raises(ValueError):
        lr_factor(0, 10, 0, "cosine")
    with pytest.raises(ValueError):
        lr_factor(0, 10, -1)


def _examples(n: int = 8) -> list[tuple[list[int], list[int]]]:
    tok = ByteTokenizer()
    pairs = load_pairs(FIXTURE)[:n]
    return [
        encode_pair(tok.encode(p.prompt), tok.encode(p.rewrite), tok.sep_id, tok.eos_id, 256)
        for p in pairs
    ]


def _train(**kw: Any) -> list[float]:
    import torch

    torch.manual_seed(0)
    model = build_tiny_lm(ByteTokenizer.vocab_size, d_model=32, n_layers=1, max_len=256)
    pad = ByteTokenizer.pad_id
    return train_sft(model, _examples(), steps=8, lr=3e-3, batch_size=2, pad_id=pad, **kw)


def test_default_schedule_is_unchanged_and_linear_differs() -> None:
    pytest.importorskip("torch")
    default = _train()
    assert _train(schedule="constant", warmup_steps=0) == default
    linear = _train(schedule="linear", warmup_steps=3)
    assert linear[0] == pytest.approx(default[0])  # the first loss is before any update
    assert linear[1:] != default[1:]


def test_eval_hook_steps_and_stop() -> None:
    pytest.importorskip("torch")
    ex = _examples()
    calls: list[tuple[int, float]] = []
    _train(eval_examples=ex[:4], eval_every=3, on_eval=lambda s, v: calls.append((s, v)))
    assert [s for s, _ in calls] == [3, 6, 8]
    assert all(v > 0 for _, v in calls)
    _train(eval_examples=ex[:4], eval_every=0, on_eval=lambda s, v: calls.append((s, v)))
    assert len(calls) == 3  # eval_every 0 disables the hook

    class Stop(Exception):
        pass

    def stop(step: int, loss: float) -> None:
        raise Stop(step)

    with pytest.raises(Stop) as exc:
        _train(eval_examples=ex[:4], eval_every=2, on_eval=stop)
    assert exc.value.args == (2,)


def test_eval_loss_is_token_weighted_and_keeps_mode() -> None:
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    model = build_tiny_lm(ByteTokenizer.vocab_size, d_model=32, n_layers=1, max_len=256)
    model.train()
    ex = _examples(4)
    whole = eval_loss(model, ex, batch_size=4, pad_id=ByteTokenizer.pad_id)
    chunked = eval_loss(model, ex, batch_size=1, pad_id=ByteTokenizer.pad_id)
    assert whole == pytest.approx(chunked, rel=1e-4)
    assert model.training
    with pytest.raises(ValueError):
        eval_loss(model, [], batch_size=1, pad_id=0)


def test_run_writes_val_loss(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from terser.config import load_config
    from terser.models.rewriter.train import run

    cfg = load_config(
        [
            "experiment=sft_tiny",
            "hardware=cpu",
            f"experiment.params.output_dir={tmp_path.as_posix()}",
            "experiment.params.steps=6",
            f"++experiment.params.val_data={FIXTURE.as_posix()}",
            "++experiment.params.eval_every=3",
            "++experiment.params.schedule=linear",
            "++experiment.params.warmup_ratio=0.5",
        ]
    )
    seen: list[int] = []
    result = run(cfg, on_eval=lambda s, v: seen.append(s))
    assert seen == [3, 6]
    saved = json.loads((tmp_path / "loss.json").read_text(encoding="utf-8"))
    assert [s for s, _ in saved["val_loss"]] == [3, 6] and len(result["loss"]) == 6
