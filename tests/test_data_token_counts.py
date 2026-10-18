"""Tests for row 2.2.2 token counts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest

from terser.data.token_counts import (
    COLUMNS,
    BatchCounter,
    MissingTokenCountsError,
    build_table,
    check_complete,
    load_texts,
    main,
    make_batch_counter,
    write_table,
)

PROMPTS = [
    {"id": "p1", "prompt": "um can you please write a sort function"},
    {"id": "p2", "prompt": "what is the capital of France"},
]
CANDIDATES = [
    {"prompt_id": "p1", "candidate_id": "p1:ops_FHC", "text": "write a sort function"},
    {"prompt_id": "p1", "candidate_id": "p1:llmlingua2", "text": "write sort function"},
    # identical to its original: counted once but listed under both ids
    {"prompt_id": "p2", "candidate_id": "p2:ops_FHC", "text": "what is the capital of France"},
]


def write_inputs(
    tmp_path: Path, candidates: list[dict[str, str]] = CANDIDATES
) -> tuple[Path, Path]:
    prompts = tmp_path / "prompts.jsonl"
    cands = tmp_path / "candidates.jsonl"
    prompts.write_text("".join(json.dumps(r) + "\n" for r in PROMPTS), encoding="utf-8")
    cands.write_text("".join(json.dumps(r) + "\n" for r in candidates), encoding="utf-8")
    return prompts, cands


class WordCounter:
    """Fake batched counter: words (target 'w') or characters (target 'c'); records batches."""

    def __init__(self) -> None:
        self.batches: list[tuple[str, int]] = []

    def factory(self, target: str) -> BatchCounter:
        def count(texts: Sequence[str]) -> list[int]:
            self.batches.append((target, len(texts)))
            return [len(t.split()) if target == "w" else len(t) for t in texts]

        return count


def test_load_texts(tmp_path: Path) -> None:
    texts = load_texts(*write_inputs(tmp_path))
    assert list(texts.columns) == ["prompt_id", "candidate_id", "text"]
    assert texts["candidate_id"].tolist() == [
        "p1:original",
        "p2:original",
        "p1:ops_FHC",
        "p1:llmlingua2",
        "p2:ops_FHC",
    ]


def test_load_texts_rejects_bad_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not in prompts"):
        load_texts(
            *write_inputs(tmp_path, [{"prompt_id": "zz", "candidate_id": "zz:x", "text": ""}])
        )
    with pytest.raises(ValueError, match="duplicate"):
        load_texts(*write_inputs(tmp_path, CANDIDATES + CANDIDATES[:1]))
    with pytest.raises(FileNotFoundError):
        load_texts(tmp_path / "nope.jsonl", tmp_path / "nope2.jsonl")


def test_build_table_counts_unique_texts_once_and_caches(tmp_path: Path) -> None:
    texts = load_texts(*write_inputs(tmp_path))
    fake = WordCounter()
    table = build_table(texts, ["w", "c"], counter_factory=fake.factory, cache_dir=tmp_path / "c")
    assert list(table.columns) == list(COLUMNS)
    assert len(table) == 5 * 2
    assert fake.batches == [("w", 4), ("c", 4)]  # 5 texts, 4 unique
    tokens = table.set_index(["candidate_id", "target"])["tokens"]
    assert tokens[("p1:ops_FHC", "w")] == 4
    assert tokens[("p2:ops_FHC", "c")] == tokens[("p2:original", "c")] == len(PROMPTS[1]["prompt"])
    assert table["tokens"].dtype == "int64"
    assert check_complete(table, texts, ["w", "c"]) == 0

    again = build_table(texts, ["w", "c"], counter_factory=fake.factory, cache_dir=tmp_path / "c")
    assert len(fake.batches) == 2  # everything from the cache
    pd.testing.assert_frame_equal(table, again)


def test_batching(tmp_path: Path) -> None:
    texts = load_texts(*write_inputs(tmp_path))
    fake = WordCounter()
    build_table(texts, ["w"], counter_factory=fake.factory, batch_size=3)
    assert fake.batches == [("w", 3), ("w", 1)]


def test_check_complete_fails_loudly(tmp_path: Path) -> None:
    texts = load_texts(*write_inputs(tmp_path))
    table = build_table(texts, ["w"], counter_factory=WordCounter().factory)
    with pytest.raises(MissingTokenCountsError, match="5 missing"):
        check_complete(table, texts, ["w", "c"])
    with pytest.raises(MissingTokenCountsError, match="1 missing"):
        check_complete(table.iloc[1:], texts, ["w"])
    broken = table.copy().astype({"tokens": "object"})
    broken.loc[0, "tokens"] = None
    with pytest.raises(MissingTokenCountsError):
        check_complete(broken, texts, ["w"])


def test_write_table_parquet_or_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    texts = load_texts(*write_inputs(tmp_path))
    table = build_table(texts, ["w"], counter_factory=WordCounter().factory)
    path = write_table(table, tmp_path / "out" / "counts.parquet")
    pytest.importorskip("pyarrow")
    assert path.suffix == ".parquet"
    pd.testing.assert_frame_equal(pd.read_parquet(path), table)

    import builtins

    real_import = builtins.__import__

    def no_pyarrow(name: str, *args: object, **kwargs: object) -> object:
        if name == "pyarrow":
            raise ImportError("no pyarrow")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", no_pyarrow)
    csv_path = write_table(table, tmp_path / "out" / "counts.parquet")
    monkeypatch.undo()
    assert csv_path.suffix == ".csv"
    assert pd.read_csv(csv_path).shape == table.shape


def test_tiktoken_batch_matches_count_tokens() -> None:
    from terser.tokenizers import count_tokens

    texts = ["hey so um", "", "<|endoftext|> special text", "naïve café ✓\n\n```py\nx=1\n```"]
    counts = make_batch_counter("gpt-4o")(texts)
    assert counts == [count_tokens(t, "gpt-4o") for t in texts]
    with pytest.raises(KeyError):
        make_batch_counter("gpt-2")


@pytest.mark.network
@pytest.mark.parametrize("target", ["qwen2.5", "llama-3.1"])
def test_hf_batch_matches_count_tokens(target: str) -> None:
    from terser.tokenizers import count_tokens

    texts = ["hey so um can you like help", "", "<|im_start|> x", "naïve café ✓\n  indent"]
    assert make_batch_counter(target)(texts) == [count_tokens(t, target) for t in texts]


def test_cli_end_to_end_gpt4o(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    prompts, cands = write_inputs(tmp_path)
    out = tmp_path / "counts.parquet"
    code = main(
        [
            "--prompts", str(prompts),
            "--candidates", str(cands),
            "--out", str(out),
            "--targets", "gpt-4o",
            "--cache-dir", str(tmp_path / "cache"),
        ]
    )  # fmt: skip
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["missing"] == 0 and summary["rows"] == 5 and summary["unique_texts"] == 4
    assert Path(summary["output"]).is_file()


def test_default_targets_never_include_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    from terser.data.token_counts import default_targets

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    targets = default_targets()
    assert "claude" not in targets and "gpt-4o" in targets


def test_count_cache_file_is_keyed_by_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from terser.data.token_counts import _CountCache

    monkeypatch.setenv("TERSER_CLAUDE_MODEL", "claude-a")
    a = _CountCache(tmp_path, "claude").path
    monkeypatch.setenv("TERSER_CLAUDE_MODEL", "claude-b")
    b = _CountCache(tmp_path, "claude").path
    assert a != b and a is not None and a.name.startswith("claude-")
