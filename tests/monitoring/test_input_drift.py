"""Input drift (row 11.5.2): silent on same-distribution windows, fires on synthetic shifts."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from terser.features.embeddings import EmbeddingCache, Encoder
from terser.monitoring.input_drift import (
    ALERT_LOGGER,
    DriftResult,
    PromptRecord,
    categorical_psi,
    detect_input_drift,
    load_prompts,
    main,
    mmd_permutation_test,
    psi,
    resolve_embeddings,
)

CATS = ["code", "fact", "reason", "explain", "creative", "transform"]
DIM = 32


def _window(
    n: int, seed: int, shift: float = 0.0, long: bool = False, cats: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    scales = np.linspace(3.0, 0.5, DIM)
    emb = rng.normal(size=(n, DIM)) * scales
    emb[:, 0] += shift * scales[0]
    lengths = rng.lognormal(mean=4.0 if long else 3.0, sigma=0.5, size=n).round()
    categories = list(rng.choice(cats or CATS, size=n))
    return emb, lengths, categories


def _detect(cur: tuple[np.ndarray, np.ndarray, list[str]]) -> DriftResult:
    ref = _window(1000, seed=1)
    result, _, _ = detect_input_drift(
        ref[0], cur[0], ref[1], cur[1], ref[2], cur[2], n_components=8, n_permutations=200
    )
    return result


def test_psi_basics() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=5000), rng.normal(size=5000)
    assert psi(a, b) < 0.02
    assert psi(a, b + 1.0) > 0.5
    assert categorical_psi(["a"] * 50 + ["b"] * 50, ["a"] * 50 + ["b"] * 50) == pytest.approx(0)
    assert categorical_psi(["a"] * 50 + ["b"] * 50, ["a"] * 95 + ["b"] * 5) > 0.2
    # an unseen level is floored, not a division by zero
    assert np.isfinite(categorical_psi(["a", "b"] * 10, ["c"] * 20))


def test_mmd_same_vs_shifted() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 5))
    _, p_same, _ = mmd_permutation_test(x, rng.normal(size=(300, 5)), n_permutations=200)
    mmd2, p_shift, sigma = mmd_permutation_test(
        x, rng.normal(size=(300, 5)) + 0.5, n_permutations=200
    )
    assert p_same > 0.05
    assert p_shift < 0.01 and mmd2 > 0 and sigma > 0


def test_same_distribution_no_drift() -> None:
    result = _detect(_window(500, seed=2))
    assert not result.alert, result.reasons
    assert max(result.psi.values()) < 0.1 and result.mmd_p > 0.01
    assert "no input drift" in result.summary()


def test_embedding_mean_shift_flagged() -> None:
    result = _detect(_window(500, seed=2, shift=1.0))
    assert result.alert
    assert any(k.startswith("pc_") for k in result.drifted) and result.mmd_p < 0.01
    assert "length_words" not in result.drifted and "category" not in result.drifted


def test_small_window_without_drift_does_not_alert() -> None:
    """At n=50 sampling noise alone is near PSI 0.2; the null correction absorbs it."""
    alerts = sum(_detect(_window(50, seed=s)).alert for s in range(20))
    assert alerts <= 3  # was 20 of 20 before the correction


def test_longer_prompts_flagged() -> None:
    result = _detect(_window(500, seed=2, long=True))
    assert result.drifted == ["length_words"]
    assert result.length["current"]["median"] > 2 * result.length["reference"]["median"]


def test_category_mix_flagged() -> None:
    result = _detect(_window(500, seed=2, cats=["code", "code", "code", "fact"]))
    assert result.drifted == ["category"]
    assert result.category["current"]["code"] > 0.6
    assert "category" in result.markdown("t")


def test_shape_checks() -> None:
    ref = _window(100, seed=1)
    with pytest.raises(ValueError):
        detect_input_drift(ref[0], ref[0][:, :5], ref[1], ref[1], ref[2], ref[2])
    with pytest.raises(ValueError):
        detect_input_drift(ref[0], ref[0][:10], ref[1], ref[1][:10], ref[2], ref[2][:10])


def test_load_prompts_and_text_cache(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    rows = [
        {"id": "a", "prompt": "write a python function to sort a list", "category": "code"},
        {"request_id": "r1", "prompt_sha256": "x"},  # hashed-only request log record
        {"request_id": "r2", "prompt": "translate this sentence into french: hi"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
    recs, skipped = load_prompts(path)
    assert skipped == 1 and [r.id for r in recs] == ["a", "r2"]
    assert recs[0].category == "code" and recs[0].length == 8
    assert recs[1].category != ""

    calls: list[int] = []

    def factory() -> Encoder:
        def enc(texts: Sequence[str]) -> np.ndarray:
            calls.append(len(texts))
            return np.full((len(texts), 4), float(len(texts)), dtype=np.float32)

        return enc

    cache = EmbeddingCache(["a"], np.ones((1, 4), dtype=np.float32))
    emb = resolve_embeddings(recs, [cache], factory, tmp_path / "tc")
    assert emb.shape == (2, 4) and calls == [1] and emb[0, 0] == 1.0
    again = resolve_embeddings(recs, [cache], None, tmp_path / "tc")  # now fully cached
    np.testing.assert_array_equal(emb, again)
    with pytest.raises(KeyError):
        resolve_embeddings(recs, [cache], None, None)


def _write_window(path: Path, records: list[PromptRecord]) -> None:
    lines = [json.dumps({"id": r.id, "prompt": r.text, "category": r.category}) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.parametrize("shifted", [False, True])
def test_cli_exit_code(tmp_path: Path, shifted: bool, caplog: pytest.LogCaptureFixture) -> None:
    ref_e, ref_l, ref_c = _window(400, seed=1)
    cur_e, cur_l, cur_c = _window(200, seed=2, shift=1.5 if shifted else 0.0, long=shifted)
    ref = [
        PromptRecord(f"r{i}", "w " * int(n), c)
        for i, (n, c) in enumerate(zip(ref_l, ref_c, strict=True))
    ]
    cur = [
        PromptRecord(f"c{i}", "w " * int(n), c)
        for i, (n, c) in enumerate(zip(cur_l, cur_c, strict=True))
    ]
    _write_window(tmp_path / "ref.jsonl", ref)
    _write_window(tmp_path / "cur.jsonl", cur)
    cache = EmbeddingCache([r.id for r in ref + cur], np.vstack([ref_e, cur_e]).astype(np.float32))
    cache.save(tmp_path / "emb", "e")
    argv = [
        "--reference", str(tmp_path / "ref.jsonl"),
        "--current", str(tmp_path / "cur.jsonl"),
        "--embeddings-dir", str(tmp_path / "emb"),
        "--embeddings-name", "e",
        "--text-cache-dir", str(tmp_path / "tc"),
        "--out-dir", str(tmp_path / "out"),
        "--components", "8",
        "--no-encode",
        "--no-evidently",
    ]  # fmt: skip
    with caplog.at_level(logging.INFO, logger=ALERT_LOGGER):
        code = main(argv)
    assert code == (1 if shifted else 0)
    summary = json.loads((tmp_path / "out" / "input_drift_cur.json").read_text(encoding="utf-8"))
    assert summary["alert"] is shifted
    crit = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert len(crit) == (1 if shifted else 0)
    assert (tmp_path / "out" / "input_drift_cur.md").exists()


def test_evidently_report_agrees(tmp_path: Path) -> None:
    pytest.importorskip("evidently")
    from terser.monitoring.input_drift import _frame, evidently_report

    ref = _window(600, seed=1)
    cur = _window(300, seed=2, shift=1.0)
    res, ref_pc, cur_pc = detect_input_drift(
        ref[0], cur[0], ref[1], cur[1], ref[2], cur[2], n_components=4, n_permutations=50
    )
    values, artifacts = evidently_report(
        _frame(ref_pc, ref[1], ref[2]), _frame(cur_pc, cur[1], cur[2]), tmp_path, "t"
    )
    assert set(values["psi"]) == set(res.psi)
    assert values["psi"]["pc_1"] > 0.2 and values["psi"]["length_words"] < 0.2
    assert values["drifted_count"] >= 1
    assert Path(tmp_path / "input_drift_t.evidently.json").exists()
    assert "evidently_html" in artifacts
