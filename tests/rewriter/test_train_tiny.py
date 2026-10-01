"""Row 8.3.2: the full SFT loop on a tiny model on CPU (smoke test for the GPU rungs)."""

from __future__ import annotations

from pathlib import Path

import pytest

from terser.config import load_config
from terser.models.rewriter.train import IGNORE, ByteTokenizer, encode_pair, load_pairs, run

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "casual_prompts.jsonl"


def test_encode_pair_masks_prompt() -> None:
    ids, labels = encode_pair([1, 2, 3], [7, 8], sep=9, eos=10, max_len=16)
    assert ids == [1, 2, 3, 9, 7, 8, 10]
    assert labels == [IGNORE] * 4 + [7, 8, 10]
    ids, labels = encode_pair(list(range(20)), [7, 8], sep=99, eos=98, max_len=8)
    assert len(ids) == 8 and ids[-3:] == [7, 8, 98] and ids[:4] == [16, 17, 18, 19]


def test_load_pairs_and_byte_tokenizer(tmp_path: Path) -> None:
    pairs = load_pairs(FIXTURE)
    assert len(pairs) >= 20 and all(p.rewrite for p in pairs)
    tok = ByteTokenizer()
    assert tok.decode(tok.encode("héllo")) == "héllo"
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "x", "prompt": "p"}\n', encoding="utf-8")
    with pytest.raises(KeyError):
        load_pairs(bad)


def test_tiny_sft_loss_decreases(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    cfg = load_config(
        [
            "experiment=sft_tiny",
            "hardware=cpu",
            f"experiment.params.output_dir={tmp_path.as_posix()}",
            "experiment.params.steps=30",
        ]
    )
    result = run(cfg)
    loss = result["loss"]
    assert len(loss) == 30
    first, last = sum(loss[:5]) / 5, sum(loss[-5:]) / 5
    assert last < 0.8 * first
    assert (tmp_path / "loss.json").exists()
