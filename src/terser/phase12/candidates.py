"""Final-evaluation candidates for every CPU-producible system (plan row **12.2**).

``build`` writes, for one split, the prompt file and the candidate file that
:mod:`terser.phase2.respond` and :mod:`terser.phase2.label_stage` read, the SFT input file that
:mod:`terser.models.rewriter.generate` reads, and a meta file with per-method latency.

=================  ==========================================================================
method             system (row 12.2)
=================  ==========================================================================
``ops_FHC``        best operator stack, chosen on train in row 8.3.9 (``apply_ops(.., "FHC")``)
``llmlingua2``     LLMLingua-2 exactly as Phase 2 made it (:class:`LLMLingua2Compressor`,
                   bert-base multilingual meetingbank, rate 0.5, CPU, ``.cache/llmlingua2``)
``rl_greedy``      greedy rollout of the row 9.2.3 Q-table (:mod:`terser.models.rl.verify` recipe:
                   ``BEST_HP``, seed 0, 30,000 episodes, 300 train prompts, LightGBM gate reward)
``hmm``            row 5.3.6 HMM compressor (supervised HMM, Viterbi DROP deletion)
``gbm_tagger``     row 8.2.4 LightGBM span tagger (:mod:`terser.phase12.taggers`); gated later
``textcnn_tagger`` row 8.2.4 TextCNN (concat) span tagger; gated later
``fast_path``      deployed CPU fast path rewrite (``models/fast/tagger.onnx``); gated later
=================  ==========================================================================

Gates are applied in :mod:`terser.phase12.results` on the judged candidates, so the gate-off
ablation needs no extra judging. Seed variants (``--textcnn-seeds 0,1,2``, ``--rl-seeds``) add
methods ``<method>_s<k>`` for the >= 3 seeds of row 12.4.

Splits: ``val`` reads :data:`~terser.phase12.common.VAL_PROMPTS` (the 285 judged val prompts) and
never the test split. ``test`` reads the frozen test split **and** the real-prompt slice through
:func:`terser.phase12.unlock.read_locked`, so it fails until the row 12.1 unlock and logs each read.
Real-slice prompts carry ``slice: "real"``.

CLI (after the unlock; CPU only)::

    CUDA_VISIBLE_DEVICES= python -m terser.phase12.candidates build --split test \\
        --out-dir data/interim/final --textcnn-seeds 0,1,2 --rl-seeds 0,1,2

writes ``data/interim/final/test_prompts.jsonl``, ``test_candidates_cpu.jsonl``,
``test_sft_input.jsonl`` and ``test_cpu_meta.json``. GPU systems are then added by the main
session (see :mod:`terser.phase12.results` for the full sequence), e.g. the SFT rewrites::

    python -m terser.models.rewriter.generate --base Qwen/Qwen2.5-3B-Instruct \\
        --model models/rewriter/sft-qwen3b-lora-v2 --data data/interim/final/test_sft_input.jsonl \\
        --out data/interim/final/test_sft3bv2.jsonl
    python -m terser.phase12.candidates import-sft --tag sft3bv2 \\
        --generate data/interim/final/test_sft3bv2.jsonl \\
        --out data/interim/final/test_candidates_gpu.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import cache, cached_property, lru_cache
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from terser.phase12.common import (
    MAIN_SLICE,
    REAL_SLICE,
    TOKEN_TARGET,
    VAL_PROMPTS,
    read_jsonl,
    write_jsonl,
)

__all__ = [
    "CPU_METHODS",
    "Resources",
    "build",
    "candidate_record",
    "import_sft",
    "load_split_prompts",
    "main",
    "merge_candidates",
    "timed_rewrites",
]

CPU_METHODS: Final[tuple[str, ...]] = (
    "ops_FHC",
    "fast_path",
    "hmm",
    "llmlingua2",
    "gbm_tagger",
    "textcnn_tagger",
    "rl_greedy",
)
"""CPU systems in build order (cheapest first)."""

BEST_OPERATOR: Final = "FHC"
SPANS: Final = Path("data/processed/span_features.parquet")
PAIRS: Final = Path("data/processed/p4_pairs.parquet")
ORACLE: Final = Path("data/processed/oracle.csv")
TRAIN: Final = Path("data/processed/splits/train.jsonl")
FAST_TAGGER: Final = Path("models/fast/tagger.onnx")
LLMLINGUA_CACHE: Final = Path(".cache/llmlingua2")
RL_EPISODES: Final = 30_000
RL_TRAIN_PROMPTS: Final = 300
LATENCY_SAMPLE: Final = 100


# --------------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------------


def _usable(r: Mapping[str, Any]) -> bool:
    return r.get("stand_in") not in (True, "True", "true") and not r.get("rejected")


def load_split_prompts(
    split: str,
    *,
    val_prompts: Path = VAL_PROMPTS,
    reader: Callable[[str, str], list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Usable prompt records of ``split`` with ``split`` and ``slice`` fields set.

    ``val``: ``val_prompts`` (refused if it is a frozen test file). ``test``: the test split
    plus the real-prompt slice via ``reader`` (default :func:`terser.phase12.unlock.read_locked`).
    """
    from terser.data.splits import assert_no_test_access

    if split == "val":
        assert_no_test_access(val_prompts)
        if "test" in val_prompts.name:
            raise ValueError(f"{val_prompts} looks like a test file; val must not read test")
        parts = [(MAIN_SLICE, read_jsonl(val_prompts))]
    elif split == "test":
        if reader is None:
            from terser.phase12.unlock import read_locked

            reader = read_locked
        purpose = "row 12.2 candidate build"
        parts = [(MAIN_SLICE, reader("test", purpose)), (REAL_SLICE, reader("real", purpose))]
    else:
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sl, recs in parts:
        for r in recs:
            if not _usable(r):
                continue
            pid = str(r["id"])
            if pid in seen:
                raise ValueError(f"duplicate prompt id {pid!r}")
            seen.add(pid)
            out.append({**r, "split": split, "slice": sl})
    return out


# --------------------------------------------------------------------------------------------
# Records and timing
# --------------------------------------------------------------------------------------------


def candidate_record(
    pid: str, method: str, text: str, generator: str, **extra: Any
) -> dict[str, Any]:
    """A candidate in the Phase 2 :data:`~terser.data.candidates.CANDIDATE_FIELDS` format."""
    return {
        "prompt_id": pid,
        "candidate_id": f"{pid}:{method}",
        "method": method,
        "text": text,
        "generator": generator,
        "stand_in": False,
        "rejected": False,
        "rejection_reason": None,
        **extra,
    }


def timed_rewrites(
    prompts: Sequence[Mapping[str, Any]], rewrite: Callable[[str], str]
) -> tuple[dict[str, str], list[float]]:
    """``{prompt_id: rewrite}`` and per-prompt wall seconds (models loaded before the call)."""
    out, secs = {}, []
    for p in prompts:
        t0 = time.perf_counter()
        out[str(p["id"])] = rewrite(str(p["prompt"]))
        secs.append(time.perf_counter() - t0)
    return out, secs


def _latency(secs: Sequence[float]) -> dict[str, Any]:
    if not secs:
        return {"latency_mean_s": None, "latency_p50_s": None, "latency_p95_s": None, "n_timed": 0}
    a = np.asarray(secs, dtype=np.float64)
    return {
        "latency_mean_s": float(a.mean()),
        "latency_p50_s": float(np.percentile(a, 50)),
        "latency_p95_s": float(np.percentile(a, 95)),
        "n_timed": len(secs),
    }


def merge_candidates(path: Path, new: Sequence[Mapping[str, Any]]) -> int:
    """Replace ``path``'s records of the methods in ``new`` by ``new``; keep the others."""
    methods = {str(r["method"]) for r in new}
    old = read_jsonl(path) if path.is_file() else []
    kept = [r for r in old if str(r["method"]) not in methods]
    return write_jsonl(path, [*kept, *new])


# --------------------------------------------------------------------------------------------
# Lazy CPU resources
# --------------------------------------------------------------------------------------------


class Resources:
    """Models and tables shared by the builders, loaded on first use."""

    def __init__(self, spans: Path = SPANS, pairs: Path = PAIRS, oracle: Path = ORACLE) -> None:
        self.spans_path, self.pairs_path, self.oracle_path = spans, pairs, oracle

    @cached_property
    def cost(self) -> Callable[[str], int]:
        from terser.tokenizers.count import count_tokens

        @cache
        def count(text: str) -> int:
            return count_tokens(text, TOKEN_TARGET)

        return count

    @cached_property
    def spans(self) -> Any:
        import pandas as pd

        return pd.read_parquet(self.spans_path)

    @cached_property
    def train_ids(self) -> set[str]:
        return {str(x) for x in self.spans.prompt_id.unique()}

    @cached_property
    def surprisal(self) -> Callable[[Sequence[str]], Sequence[float]]:
        from terser.features.span import gpt2_surprisal

        fn = gpt2_surprisal("gpt2")

        @cache
        def bits(toks: tuple[str, ...]) -> tuple[float, ...]:
            return tuple(fn(list(toks))) if toks else ()

        return lambda toks: list(bits(tuple(toks)))

    @cached_property
    def pos(self) -> Callable[[Sequence[str]], Sequence[str]]:
        from terser.features.span import spacy_pos

        return spacy_pos()

    @cached_property
    def oracle(self) -> Any:
        """Oracle rows of the span-table (train) prompts only."""
        import pandas as pd

        df = pd.read_csv(self.oracle_path)
        return df[df.prompt_id.astype(str).isin(self.train_ids)]


# --------------------------------------------------------------------------------------------
# Builders: each returns ({prompt_id: text}, info)
# --------------------------------------------------------------------------------------------

Built = tuple[dict[str, str], dict[str, Any]]


def build_ops(prompts: Sequence[Mapping[str, Any]], res: Resources) -> Built:
    from terser.operators.registry import apply_ops

    texts, secs = timed_rewrites(prompts, lambda t: apply_ops(t, BEST_OPERATOR, None))
    return texts, {"generator": f"operators:{BEST_OPERATOR}", **_latency(secs)}


def build_fast_path(prompts: Sequence[Mapping[str, Any]], res: Resources) -> Built:
    import yaml

    from terser.serving.fast_models import OnnxSpanRewriter

    cfg = yaml.safe_load(Path("configs/serving/fast_onnx.yaml").read_text(encoding="utf-8"))
    rw = OnnxSpanRewriter(FAST_TAGGER, res.cost, float(cfg.get("tagger_threshold", 0.5)))
    texts, secs = timed_rewrites(prompts, lambda t: rw(t, TOKEN_TARGET).text)
    return texts, {
        "generator": f"{FAST_TAGGER.as_posix()}@{cfg.get('tagger_threshold', 0.5)}",
        "gate": "onnx",
        **_latency(secs),
    }


def build_hmm(prompts: Sequence[Mapping[str, Any]], res: Resources) -> Built:
    from terser.models.classical.hmm import TokenClassEncoder

    enc = TokenClassEncoder()
    try:  # row 5.3.6 module (being written in parallel); fall back to the HMM itself
        from terser.phase5.hmm_compressor import fit_span_hmm, hmm_rewrite

        hmm = fit_span_hmm(res.spans, enc)

        def rewrite(t: str) -> str:
            return hmm_rewrite(hmm, t, enc)[0]

        source = "terser.phase5.hmm_compressor"
    except ImportError:  # pragma: no cover - the module ships with the repo
        from terser.features.alignment import TOKEN_RE
        from terser.models.classical.hmm import TAGS, fit_supervised
        from terser.phase12.taggers import delete_tokens

        tag_id = {t: i for i, t in enumerate(TAGS)}
        obs, tags = [], []
        for _, g in res.spans.sort_values(["prompt_id", "position"]).groupby("prompt_id"):
            obs.append(enc.encode(g.token.tolist()))
            tags.append(np.array([tag_id[t] for t in g.label], dtype=np.int64))
        model = fit_supervised(obs, tags, len(TAGS), enc.n_obs, smoothing=1.0)
        drop_id = TAGS.index("DROP")

        def rewrite(t: str) -> str:
            toks = [m.group(0) for m in TOKEN_RE.finditer(t)]
            if not toks:
                return t
            path, _ = model.viterbi(enc.encode(toks))
            mask = [int(x) == drop_id for x in path.tolist()]
            return delete_tokens(t, mask, protect_lexicon=False)

        source = "terser.models.classical.hmm (direct)"
    texts, secs = timed_rewrites(prompts, rewrite)
    return texts, {"generator": f"hmm:supervised-mle-laplace1 via {source}", **_latency(secs)}


def build_llmlingua(prompts: Sequence[Mapping[str, Any]], res: Resources) -> Built:
    from terser.data.candidates import DEFAULT_LLMLINGUA_MODEL, LLMLingua2Compressor

    lingua = LLMLingua2Compressor(
        DEFAULT_LLMLINGUA_MODEL, rate=0.5, device="cpu", cache_dir=LLMLINGUA_CACHE
    )
    texts: dict[str, str] = {}
    secs: list[float] = []
    cached = 0
    for p in prompts:
        text = str(p["prompt"])
        hit = (
            lingua.cache is not None
            and bool(text.strip())
            and lingua.cache.get(lingua._key(text)) is not None
        )
        t0 = time.perf_counter()
        texts[str(p["id"])] = lingua(text)
        if hit:
            cached += 1
        else:
            secs.append(time.perf_counter() - t0)
    info = {"generator": lingua.generator, "cache_hits": cached, **_latency(secs)}
    if not secs:
        info["latency_note"] = "every output came from the Phase 2 disk cache; latency n/r"
    return texts, info


def build_gbm_tagger(prompts: Sequence[Mapping[str, Any]], res: Resources) -> Built:
    from terser.phase12.taggers import delete_tokens, fit_gbm_tagger, gbm_drop_mask

    model = fit_gbm_tagger(res.spans)

    def rewrite(t: str) -> str:
        return delete_tokens(t, gbm_drop_mask(model, t, res.cost, res.surprisal, res.pos))

    texts, secs = timed_rewrites(prompts, rewrite)
    return texts, {
        "generator": "lightgbm-span-tagger(n=200,lr=0.1)+hard-keep",
        "gate": "pair",
        **_latency(secs),
    }


def build_textcnn(prompts: Sequence[Mapping[str, Any]], res: Resources, seed: int = 0) -> Built:
    from terser.features.alignment import TOKEN_RE
    from terser.phase8.textcnn_study import MAX_WORDS
    from terser.phase12.taggers import delete_tokens, textcnn_drop_masks

    texts_in = [str(p["prompt"]) for p in prompts]
    t0 = time.perf_counter()
    masks = textcnn_drop_masks(res.spans, res.oracle, texts_in, seed=seed)
    train_s = time.perf_counter() - t0
    texts = {
        str(p["id"]): delete_tokens(t, m) for p, t, m in zip(prompts, texts_in, masks, strict=True)
    }
    # per-request latency estimate: the bge-small forward of one prompt with the encoder already
    # loaded (``word_states`` reloads the model per call, so it is not timed); the CNN is < 1 ms
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    enc = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").eval()
    secs = []
    for t in texts_in[: min(LATENCY_SAMPLE, len(texts_in))]:
        words = [m.group(0) for m in TOKEN_RE.finditer(t)][:MAX_WORDS]
        if words:
            t1 = time.perf_counter()
            batch = tok(
                words,
                is_split_into_words=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                enc(**batch)
            secs.append(time.perf_counter() - t1)
    return texts, {
        "generator": f"textcnn-span-tagger(concat,12 epochs,seed={seed})+hard-keep",
        "gate": "pair",
        "train_and_predict_s": train_s,
        "latency_note": "bge-small forward per prompt (encoder loaded); CNN forward excluded",
        **_latency(secs),
    }


def _qtable_key(seed: int, n_train: int, episodes: int) -> dict[str, Any]:
    from terser.models.rl.verify import BEST_HP

    return {"hp": BEST_HP, "seed": seed, "n_train": n_train, "episodes": episodes}


def build_rl(
    prompts: Sequence[Mapping[str, Any]],
    res: Resources,
    seed: int = 0,
    qtable_cache: Path | None = None,
    episodes: int = RL_EPISODES,
) -> Built:
    """Greedy rollouts of the row 9.2.3 Q-table (retrained deterministically, or from cache)."""
    from terser.models.rl.env import RewriteEnv
    from terser.models.rl.scratch_qlearning import QLearner
    from terser.models.rl.verify import BEST_HP, episode_of, greedy_rollout, train_prompts
    from terser.operators.registry import apply_ops
    from terser.phase9.qlearning_study import build_gate

    gate, cost, category_of = build_gate(res.pairs_path, TOKEN_TARGET)
    env = RewriteEnv(
        count=cost, gate=gate, target=TOKEN_TARGET, apply=lru_cache(maxsize=None)(apply_ops)
    )
    rows = train_prompts(read_jsonl(TRAIN), RL_TRAIN_PROMPTS, seed)
    for r in rows:
        category_of[str(r["prompt"])] = str(r["category"])
    episodes_ = [episode_of(r) for r in rows]
    key = _qtable_key(seed, len(rows), episodes)
    ql = QLearner(
        alpha=BEST_HP["alpha"],
        gamma=BEST_HP["gamma"],
        decay=BEST_HP["decay"],
        eps_min=BEST_HP["eps_min"],
        seed=seed,
    )
    loaded = False
    if qtable_cache is not None and qtable_cache.is_file():
        blob = json.loads(qtable_cache.read_text(encoding="utf-8"))
        if blob.get("key") == json.loads(json.dumps(key)):
            for s, a, q, v in blob["table"]:
                ql.q[(tuple(s), a)] = float(q)
                ql.visits[(tuple(s), a)] = int(v)
            loaded = True
    t0 = time.perf_counter()
    if not loaded:
        ql.fit(env, episodes_, episodes)
        if qtable_cache is not None:
            keys = sorted(set(ql.q) | set(ql.visits), key=repr)
            table = [
                [list(cast(tuple[Any, ...], s)), a, ql.q.get((s, a), 0.0), ql.visits.get((s, a), 0)]
                for s, a in keys
            ]
            qtable_cache.parent.mkdir(parents=True, exist_ok=True)
            qtable_cache.write_text(json.dumps({"key": key, "table": table}), encoding="utf-8")
    train_s = time.perf_counter() - t0

    texts: dict[str, str] = {}
    seqs: dict[str, str] = {}
    unseen = 0
    for p in prompts:
        category_of[str(p["prompt"])] = str(p["category"])
        seq, _, draft, n_unseen = greedy_rollout(ql, env, episode_of(p))
        texts[str(p["id"])] = draft
        seqs[str(p["id"])] = seq
        unseen += n_unseen > 0
    # latency: second, warm pass (operator outputs and the reward gate are cached, so this is
    # the policy's own cost: state features + greedy lookups + operator application)
    secs = []
    for p in prompts[: min(LATENCY_SAMPLE, len(prompts))]:
        t1 = time.perf_counter()
        greedy_rollout(ql, env, episode_of(p))
        secs.append(time.perf_counter() - t1)
    return texts, {
        "generator": f"qlearning:{json.dumps(BEST_HP, sort_keys=True)}:seed={seed}",
        "qtable_from_cache": loaded,
        "train_s": train_s,
        "q_entries": len(ql.q),
        "prompts_with_unseen_state": unseen,
        "sequences": dict(sorted(_count(seqs.values()).items(), key=lambda kv: -kv[1])[:10]),
        "latency_note": "warm second pass; excludes the training-time reward gate",
        **_latency(secs),
    }


def _count(xs: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


# --------------------------------------------------------------------------------------------
# build / import-sft
# --------------------------------------------------------------------------------------------


def _seeds(spec: str) -> list[int]:
    return [int(s) for s in spec.split(",") if s.strip()] or [0]


def _method_name(base: str, seed: int) -> str:
    return base if seed == 0 else f"{base}_s{seed}"


def build(
    split: str,
    out_dir: Path,
    methods: Sequence[str] = CPU_METHODS,
    *,
    textcnn_seeds: Sequence[int] = (0,),
    rl_seeds: Sequence[int] = (0,),
    val_prompts: Path = VAL_PROMPTS,
    resources: Resources | None = None,
    builders: Mapping[str, Callable[..., Built]] | None = None,
    reader: Callable[[str, str], list[dict[str, Any]]] | None = None,
    qtable_dir: Path | None = None,
) -> dict[str, Any]:
    """Write the prompt, CPU-candidate, SFT-input and meta files of ``split`` into ``out_dir``."""
    unknown = [m for m in methods if m not in CPU_METHODS]
    if unknown:
        raise ValueError(f"unknown method(s) {unknown}; CPU methods: {CPU_METHODS}")
    res = resources or Resources()
    prompts = load_split_prompts(split, val_prompts=val_prompts, reader=reader)
    if "hmm" in methods or "gbm_tagger" in methods or "textcnn_tagger" in methods:
        leak = {str(p["id"]) for p in prompts} & res.train_ids
        if leak:
            raise ValueError(f"{len(leak)} {split} prompts are in the train span table")
    fns: dict[str, Callable[..., Built]] = {
        "ops_FHC": build_ops,
        "fast_path": build_fast_path,
        "hmm": build_hmm,
        "llmlingua2": build_llmlingua,
        "gbm_tagger": build_gbm_tagger,
        "textcnn_tagger": build_textcnn,
        "rl_greedy": build_rl,
        **(builders or {}),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / f"{split}_prompts.jsonl", prompts)
    write_jsonl(
        out_dir / f"{split}_sft_input.jsonl",
        (
            {"id": p["id"], "category": p["category"], "prompt": p["prompt"], "savings": None}
            for p in prompts
        ),
    )
    meta_path = out_dir / f"{split}_cpu_meta.json"
    meta: dict[str, Any] = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    )
    meta.setdefault("methods", {})
    meta.update(
        split=split,
        n_prompts=len(prompts),
        n_by_slice=_count(str(p["slice"]) for p in prompts),
        token_target=TOKEN_TARGET,
    )
    cand_path = out_dir / f"{split}_candidates_cpu.jsonl"
    for base in [m for m in CPU_METHODS if m in methods]:
        seeds = (
            textcnn_seeds if base == "textcnn_tagger" else rl_seeds if base == "rl_greedy" else (0,)
        )
        for seed in seeds:
            name = _method_name(base, seed)
            t0 = time.perf_counter()
            if base == "textcnn_tagger":
                texts, info = fns[base](prompts, res, seed=seed)
            elif base == "rl_greedy":
                qcache = (qtable_dir or out_dir) / f"rl_qtable_s{seed}.json"
                texts, info = fns[base](prompts, res, seed=seed, qtable_cache=qcache)
            else:
                texts, info = fns[base](prompts, res)
            rows = [
                candidate_record(str(p["id"]), name, texts[str(p["id"])], str(info["generator"]))
                for p in prompts
            ]
            merge_candidates(cand_path, rows)
            identity = sum(texts[str(p["id"])] == p["prompt"] for p in prompts)
            meta["methods"][name] = {
                **info,
                "base": base,
                "seed": seed,
                "n": len(rows),
                "identity": identity,
                "path": "fast",
                "wall_s": time.perf_counter() - t0,
            }
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"method": name, "n": len(rows), "identity": identity}), flush=True)
    return meta


def import_sft(tag: str, generate_out: Path, out: Path, count: Callable[[str], int]) -> int:
    """SFT generate output -> candidates holding the **raw** rewrite (safeguard applied later).

    An empty raw rewrite becomes the original prompt (nothing to send). Each record keeps the
    rewriter's token usage (prompt + separator in, rewrite + EOS out, Qwen2.5 tokenizer) and the
    measured generation latency for the net-savings and latency columns.
    """
    rows = []
    for r in read_jsonl(generate_out):
        raw = str(r.get("raw_rewrite", r.get("rewrite", "")))
        text = raw if raw.strip() else str(r["prompt"])
        rows.append(
            candidate_record(
                str(r["id"]),
                tag,
                text,
                f"sft:{r.get('model', tag)}",
                latency_s=float(r.get("latency_s", 0.0)),
                rewriter_tokens_in=count(str(r["prompt"])) + 1,
                rewriter_tokens_out=count(raw) + 1,
                empty_raw=not raw.strip(),
            )
        )
    return merge_candidates(out, rows)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="python -m terser.phase12.candidates", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="CPU candidates for a split")
    b.add_argument("--split", choices=["val", "test"], required=True)
    b.add_argument("--out-dir", type=Path, default=Path("data/interim/final"))
    b.add_argument("--methods", default=",".join(CPU_METHODS))
    b.add_argument("--textcnn-seeds", default="0")
    b.add_argument("--rl-seeds", default="0")
    b.add_argument("--val-prompts", type=Path, default=VAL_PROMPTS)
    b.add_argument(
        "--qtable-dir", type=Path, default=None, help="Q-table cache dir (default: --out-dir)"
    )
    s = sub.add_parser("import-sft", help="SFT generate output -> raw-rewrite candidates")
    s.add_argument("--tag", required=True)
    s.add_argument("--generate", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        meta = build(
            args.split,
            args.out_dir,
            [m.strip() for m in args.methods.split(",") if m.strip()],
            textcnn_seeds=_seeds(args.textcnn_seeds),
            rl_seeds=_seeds(args.rl_seeds),
            val_prompts=args.val_prompts,
            qtable_dir=args.qtable_dir,
        )
        summary = {
            m: {k: v.get(k) for k in ("n", "identity", "latency_mean_s")}
            for m, v in meta["methods"].items()
        }
        print(json.dumps({"split": meta["split"], "n_prompts": meta["n_prompts"], **summary}))
        return 0
    n = import_sft(args.tag, args.generate, args.out, Resources().cost)
    print(json.dumps({"tag": args.tag, "records_in_file": n}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
