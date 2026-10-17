"""Benchmark release packaging (plan row **12.11**).

``build`` writes a self-contained release directory (default ``release/terser-bench-v1``):

=====================================  ======================================================
path                                   content
=====================================  ======================================================
``prompts/{train,val,test}.jsonl``     casual prompt, clean reference, category, style,
                                       ``source``, ``source_id`` (one record per prompt)
``prompts/real_ids.jsonl``             WildChat real-slice **ids only** (no text), with the
                                       sha256 of each prompt so a rebuild can be checked
``candidates/candidates.jsonl``        the Phase 2 compressed candidates (row 2.2.1)
``labels/labels.jsonl``                the Phase 2 judge labels (row 2.3.3, ``labels.jsonl``)
``rubrics/``                           the judge rubrics (row 2.3.2) and judge prompt templates
``LICENSE`` / ``NOTICE``               CC-BY-SA-4.0 for the data; attribution per upstream
                                       source and per generating model, with the changes made
``README.md``                          the data card (``docs/data_cards/_template.md`` layout)
``MANIFEST.json`` / ``SHA256SUMS``     sha256, bytes and record count of every file
=====================================  ======================================================

The release copies no evaluation code: ``MANIFEST.json`` and the card point to the repository
path and commit instead. Licence conditions come from ``reports/license_audit.md``: Dolly's
ShareAlike clause puts every file that carries Dolly-derived text under CC-BY-SA-4.0 (option (b)
of § dolly), and a Llama-generated text in the inputs is refused (the audit requires a separate
bundle under the Llama 3.1 licence).

**WildChat.** Real-slice prompts are user-written chat and the manual PII check of row 2.1.3 is
open, so their text is never written. ``rebuild-real`` recreates it locally from the pinned
``allenai/WildChat-1M`` revision with the row 2.1.8 pipeline (:mod:`terser.data.real_slice`) and
checks each prompt against the released sha256.

**Locked splits.** The frozen test split and the real slice are read only through
:func:`terser.phase12.unlock.read_locked` with purpose :data:`PURPOSE`, so each read is logged.

CLI (CPU only)::

    python -m terser.phase12.benchmark build --out release/terser-bench-v1
    python -m terser.phase12.benchmark verify --release release/terser-bench-v1
    python -m terser.phase12.benchmark rebuild-real --release release/terser-bench-v1 \\
        --out /tmp/real_prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from terser.data.sources import SOURCES, sha256_file
from terser.phase12.common import read_jsonl, text_sha, write_jsonl

__all__ = [
    "DATA_LICENSE",
    "PURPOSE",
    "RELEASE_NAME",
    "BuildInputs",
    "build",
    "main",
    "rebuild_real",
    "render_license",
    "render_notice",
    "verify",
]

RELEASE_NAME: Final = "terser-bench-v1"
RELEASE_VERSION: Final = "1.0"
REPO_URL: Final = "https://github.com/guru-bharadwaj20/Terser"
PURPOSE: Final = "12.11 benchmark release"
"""``purpose`` of every :func:`~terser.phase12.unlock.read_locked` call (it is logged)."""

DATA_LICENSE: Final = "CC-BY-SA-4.0"
DATA_LICENSE_URL: Final = "https://creativecommons.org/licenses/by-sa/4.0/legalcode"

SPLITS_DIR: Final = Path("data/processed/splits")
CANDIDATES: Final = Path("data/interim/candidates.jsonl")
LABELS: Final = Path("data/processed/labels.jsonl")
OPEN_SPLITS: Final[tuple[str, ...]] = ("train", "val")
LOCKED_MAIN: Final = "test"
LOCKED_REAL: Final = "real"
SPLIT_ORDER: Final[tuple[str, ...]] = (*OPEN_SPLITS, LOCKED_MAIN)
CATEGORIES: Final[tuple[str, ...]] = ("code", "fact", "reason", "explain", "creative", "transform")
STYLES: Final[tuple[str, ...]] = ("rambling", "hedging", "texting", "over_polite", "repetitive")
WILDCHAT: Final = "wildchat"

REAL_IDS_FILE: Final = "prompts/real_ids.jsonl"
CANDIDATES_FILE: Final = "candidates/candidates.jsonl"
LABELS_FILE: Final = "labels/labels.jsonl"
MANIFEST_FILE: Final = "MANIFEST.json"
SUMS_FILE: Final = "SHA256SUMS"

PROMPT_FIELDS: Final[tuple[str, ...]] = (
    "id",
    "source",
    "source_id",
    "category",
    "style",
    "prompt",
    "clean",
    "generator_model",
    "prompt_version",
)
"""Fields kept from a split record (``prompt`` = casual prompt, ``clean`` = clean reference)."""

REAL_ID_FIELDS: Final[tuple[str, ...]] = (
    "id",
    "source",
    "source_id",
    "category",
    "category_source",
    "pii_scrubbed",
    "use",
)
"""Fields kept from a real-slice record: never ``prompt`` or any other text field."""

CODE_PATHS: Final[tuple[str, ...]] = (
    "src/terser/phase12/benchmark.py",
    "src/terser/data/",
    "src/terser/judge/",
    "src/terser/phase2/",
    "src/terser/phase12/",
    "dvc.yaml",
)
"""Repository paths of the code that built and evaluates the benchmark (not copied)."""

# --------------------------------------------------------------------------------------------------
# Attribution tables (reports/license_audit.md, ADR-0003, ADR-0011)
# --------------------------------------------------------------------------------------------------

_MIT_PERMISSION: Final = """\
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."""

_COMMON_CHANGES: Final = (
    "PII-scrubbed at collection with the row 2.1.3 regex policy (typed placeholders)",
    "rewritten into casual styles (rambling, hedging, texting, over-polite, repetitive) by "
    "Qwen/Qwen2.5-32B-Instruct-AWQ with a meaning check (row 2.1.4)",
    "near-duplicates removed (row 2.1.5) and split by source prompt (row 2.1.6)",
    "compressed candidates generated from the casual prompts by LLM rewrites, rule operators and "
    "LLMLingua-2 (row 2.2.1); target-model outputs judged pairwise (row 2.3.3) and combined with "
    "task metrics into labels (rows 2.3.1, 2.3.4)",
    "upstream reference answers are not included",
)


@dataclass(frozen=True)
class SourceNotice:
    """Attribution for one upstream dataset."""

    licence_url: str
    attribution: str
    citation: str
    changes: tuple[str, ...]
    mit_copyright: str | None = None


SOURCE_NOTICES: Final[dict[str, SourceNotice]] = {
    "dolly": SourceNotice(
        licence_url="https://creativecommons.org/licenses/by-sa/3.0/",
        attribution=(
            "databricks-dolly-15k was developed at Databricks (https://www.databricks.com) and its "
            "use is subject to the CC BY-SA 3.0 license. Certain categories of material include "
            "materials from Wikipedia (various pages, https://www.wikipedia.org/), Copyright (c) "
            "Wikipedia editors and contributors, licensed under CC BY-SA 3.0."
        ),
        citation="Databricks, Inc., databricks-dolly-15k (dataset card at the pinned revision).",
        changes=(
            "the clean prompt is the Dolly `instruction`, plus `context` when present",
            *_COMMON_CHANGES,
        ),
    ),
    "oasst1": SourceNotice(
        licence_url="https://www.apache.org/licenses/LICENSE-2.0",
        attribution=(
            "OpenAssistant Conversations (oasst1), licensed under the Apache License 2.0; a copy "
            "of the licence must accompany redistributed files."
        ),
        citation="OpenAssistant, oasst1 (dataset card at the pinned revision).",
        changes=("root prompter messages only; user ids and metadata dropped", *_COMMON_CHANGES),
    ),
    "humaneval": SourceNotice(
        licence_url="https://opensource.org/licenses/MIT",
        attribution="HumanEval, OpenAI, MIT License (upstream https://github.com/openai/human-eval).",
        citation="Chen et al. (2021), Evaluating Large Language Models Trained on Code.",
        changes=(
            'the clean prompt is "Complete the following Python function." plus the signature and '
            "docstring",
            *_COMMON_CHANGES,
        ),
        mit_copyright="Copyright (c) OpenAI (https://openai.com)",
    ),
    "mbpp": SourceNotice(
        licence_url="https://creativecommons.org/licenses/by/4.0/",
        attribution=(
            "MBPP (Mostly Basic Python Problems), Google Research, licensed under CC BY 4.0."
        ),
        citation="Austin et al. (2021), Program Synthesis with Large Language Models.",
        changes=(
            "the clean prompt is the MBPP `text` plus its `test_list` asserts",
            *_COMMON_CHANGES,
        ),
    ),
    "gsm8k": SourceNotice(
        licence_url="https://opensource.org/licenses/MIT",
        attribution="GSM8K, OpenAI, MIT License (upstream https://github.com/openai/grade-school-math).",
        citation="Cobbe et al. (2021), Training Verifiers to Solve Math Word Problems.",
        changes=("the clean prompt is the GSM8K `question`", *_COMMON_CHANGES),
        mit_copyright="Copyright (c) 2021 OpenAI",
    ),
}

WILDCHAT_NOTICE: Final = SourceNotice(
    licence_url="https://opendatacommons.org/licenses/by/1-0/",
    attribution=(
        "WildChat-1M, Allen Institute for AI, made available under the ODC Attribution License "
        "(ODC-By 1.0)."
    ),
    citation="Zhao et al. (2024), WildChat: 1M ChatGPT Interaction Logs in the Wild.",
    changes=(
        "no WildChat text is included: only conversation ids, heuristic category tags and the "
        "sha256 of each (PII-scrubbed) first user turn",
        "the text is rebuilt locally by `python -m terser.phase12.benchmark rebuild-real`",
    ),
)


@dataclass(frozen=True)
class ModelNotice:
    """Licence of one generating, judging or target model (``None``: not recorded in the repo)."""

    licence: str | None
    note: str


MODEL_NOTICES: Final[dict[str, ModelNotice]] = {
    "Qwen/Qwen2.5-32B-Instruct-AWQ": ModelNotice(
        "Apache-2.0", "no conditions on outputs (license_audit.md § Generating models)"
    ),
    "Qwen/Qwen2.5-14B-Instruct-AWQ": ModelNotice("Apache-2.0", "no conditions on outputs"),
    "Qwen/Qwen2.5-7B-Instruct": ModelNotice("Apache-2.0", "no conditions on outputs"),
    "microsoft/Phi-3.5-mini-instruct": ModelNotice(
        "MIT",
        "keep the copyright and permission notice if its outputs are redistributed (ADR-0003)",
    ),
    "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank": ModelNotice(
        "Apache-2.0",
        "no conditions on outputs (model card license: apache-2.0 at revision "
        "5f0c82792b7ea14c6484e015b6a072009496b7f2)",
    ),
}

ROLE_TEXT: Final[dict[str, str]] = {
    "casualisation": "wrote the casual prompts (row 2.1.4); its text is in prompts/",
    "candidates": "wrote compressed candidates (row 2.2.1); its text is in candidates/",
    "judge": "judged target outputs pairwise (row 2.3.3); its verdicts and reasons are in labels/",
    "target": (
        "answered prompts and candidates (row 2.2.3); its outputs are NOT in this release, only "
        "the judge's verdicts and short reasons about them"
    ),
}


def model_of(generator: str) -> str | None:
    """Model id in a generator string (``model|prompt``, ``model@rev?rate=..``); None for rules."""
    if generator.startswith("operators:"):
        return None
    return generator.split("|", 1)[0].split("?", 1)[0].split("@", 1)[0]


# --------------------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------------------


def is_wildchat(record: Mapping[str, Any]) -> bool:
    """True for a record whose provenance is WildChat (its text must never be released)."""
    return str(record.get("source") or "") == WILDCHAT or str(
        record.get("source_id") or ""
    ).startswith(f"{WILDCHAT}:")


def prompt_record(record: Mapping[str, Any], split: str) -> dict[str, Any]:
    """Release record of one split prompt (raises KeyError on a missing provenance field)."""
    out: dict[str, Any] = {"id": record["id"], "split": split}
    for k in PROMPT_FIELDS[1:]:
        out[k] = record.get(k)
    for k in ("source", "source_id", "prompt", "clean", "category"):
        if not out.get(k):
            raise KeyError(f"prompt {record.get('id')!r} has no {k!r}")
    return out


def real_id_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Text-free release record of one real-slice prompt, with the sha256 of its text."""
    out: dict[str, Any] = {k: record.get(k) for k in REAL_ID_FIELDS}
    out["split"] = LOCKED_REAL
    out["prompt_sha256"] = text_sha(str(record["prompt"]))
    if not out["source_id"]:
        raise KeyError(f"real-slice record {record.get('id')!r} has no source_id")
    return out


def _with_provenance(
    rec: Mapping[str, Any], index: Mapping[str, Mapping[str, Any]], kind: str
) -> dict[str, Any]:
    p = index.get(str(rec["prompt_id"]))
    if p is None:
        raise KeyError(f"{kind} {rec.get('id') or rec.get('candidate_id')!r}: unknown prompt_id")
    out = dict(rec)
    out["source"] = p["source"]
    out["source_id"] = p["source_id"]
    out.setdefault("split", p["split"])
    return out


# --------------------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------------------

Reader = Callable[[str, str], list[dict[str, Any]]]


@dataclass
class BuildInputs:
    """Where :func:`build` reads from; defaults are the repository's DVC outputs."""

    splits_dir: Path = SPLITS_DIR
    candidates: Path = CANDIDATES
    labels: Path = LABELS
    rubrics_dir: Path | None = None
    judge_prompts_dir: Path | None = None
    reader: Reader | None = None
    """``(split_name, purpose) -> records`` for the locked splits; default ``read_locked``."""
    lock_dirs: Mapping[str, Path] | None = None
    """``{"test": dir, "real": dir}`` holding the ``test.lock.json`` files (sha256 only)."""
    repo: Path = Path(".")
    code_ref: tuple[str | None, bool | None] | None = None
    """``(commit, dirty)``; default :func:`terser.phase12.unlock.git_state` of ``repo``."""
    built: datetime | None = None


@dataclass
class _Ctx:
    stats: dict[str, Any] = field(default_factory=dict)
    files: list[dict[str, Any]] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)
    models: dict[str, set[str]] = field(default_factory=dict)

    def model(self, model_id: str | None, role: str) -> None:
        if model_id:
            self.models.setdefault(model_id, set()).add(role)


def _default_dirs(inputs: BuildInputs) -> tuple[Path, Path]:
    from terser.judge.rubrics import RUBRICS_DIR

    rub = inputs.rubrics_dir or RUBRICS_DIR
    return rub, inputs.judge_prompts_dir or rub.parent / "prompts"


def _lock_info(inputs: BuildInputs) -> dict[str, Any]:
    from terser.data.splits import LOCK_NAME
    from terser.phase12.unlock import LOCKED_SPLITS

    dirs = inputs.lock_dirs if inputs.lock_dirs is not None else LOCKED_SPLITS
    out = {}
    for name, d in dirs.items():
        p = Path(d) / LOCK_NAME
        if p.is_file():
            lock = json.loads(p.read_text(encoding="utf-8"))
            out[name] = {"sha256": lock.get("sha256"), "n_records": lock.get("n_records")}
    return out


def _splits_meta(splits_dir: Path) -> dict[str, Any]:
    """Seed and leakage result recorded by the ``split`` stage (``splits.json``), if present."""
    p = splits_dir / "splits.json"
    if not p.is_file():
        return {}
    sj = json.loads(p.read_text(encoding="utf-8"))
    leak = sj.get("leakage", {})
    return {
        "seed": sj.get("summary", {}).get("seed"),
        "leakage_ok": leak.get("ok"),
        "n_shared_groups": leak.get("n_shared_groups"),
    }


def _counter(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(r.get(key)) for r in rows).items()))


def _split_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "source_prompts": len({r["source_id"] for r in rows}),
        "by_category": _counter(rows, "category"),
        "by_style": _counter(rows, "style"),
        "by_source": _counter(rows, "source"),
        "by_source_category": _counter(
            [{"k": f"{r['source']}/{r['category']}"} for r in rows], "k"
        ),
    }


def _check_no_llama(models: Iterable[str]) -> None:
    bad = sorted(m for m in models if "llama" in m.lower())
    if bad:
        raise ValueError(
            f"Llama-generated text in the inputs ({bad}): license_audit.md requires a separate "
            "bundle under the Llama 3.1 Community Licence; not supported by this release"
        )


def _project_licence(repo: Path) -> str | None:
    hits = sorted(p.name for p in repo.glob("LICEN[CS]E*") if p.is_file())
    return hits[0] if hits else None


def _prepare_out(out: Path, force: bool) -> None:
    if out.exists() and any(out.iterdir()):
        if not force:
            raise FileExistsError(f"{out} is not empty (pass force=True / --force to rebuild)")
        if not (out / MANIFEST_FILE).is_file():
            raise FileExistsError(f"{out} has no {MANIFEST_FILE}; refusing to delete it")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def _record_file(ctx: _Ctx, out: Path, rel: str, records: int | None) -> None:
    p = out / rel
    ctx.files.append(
        {"path": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p), "records": records}
    )


def build(out: Path, inputs: BuildInputs | None = None, *, force: bool = False) -> dict[str, Any]:
    """Write the release into ``out``; returns the manifest dict (also ``MANIFEST.json``)."""
    inputs = inputs or BuildInputs()
    reader = inputs.reader
    if reader is None:
        from terser.phase12.unlock import read_locked

        reader = read_locked
    if inputs.code_ref is None:
        from terser.phase12.unlock import git_state

        commit, dirty = git_state(inputs.repo)
    else:
        commit, dirty = inputs.code_ref
    built = (inputs.built or datetime.now(UTC)).astimezone(UTC)
    rubrics_dir, prompts_dir = _default_dirs(inputs)
    _prepare_out(out, force)
    ctx = _Ctx()

    # Prompts per split: train/val from the split files, test only through the logged reader.
    by_split: dict[str, list[dict[str, Any]]] = {}
    dropped_wildchat = 0
    for split in SPLIT_ORDER:
        if split in OPEN_SPLITS:
            raw = read_jsonl(inputs.splits_dir / f"{split}.jsonl")
        else:
            raw = reader(LOCKED_MAIN, PURPOSE)
        keep = [r for r in raw if not is_wildchat(r) and r.get("stand_in") is not True]
        dropped_wildchat += sum(1 for r in raw if is_wildchat(r))
        by_split[split] = [prompt_record(r, split) for r in keep]
    index: dict[str, dict[str, Any]] = {}
    for split, rows in by_split.items():
        for r in rows:
            if r["id"] in index:
                raise ValueError(f"prompt id {r['id']!r} is in two splits")
            index[r["id"]] = r
            ctx.sources.add(r["source"])
            ctx.model(r.get("generator_model"), "casualisation")
        n = write_jsonl(out / f"prompts/{split}.jsonl", rows)
        _record_file(ctx, out, f"prompts/{split}.jsonl", n)
        ctx.stats.setdefault("prompts", {})[split] = _split_stats(rows)

    # Real slice: ids only, never text.
    real = [real_id_record(r) for r in reader(LOCKED_REAL, PURPOSE)]
    n = write_jsonl(out / REAL_IDS_FILE, real)
    _record_file(ctx, out, REAL_IDS_FILE, n)
    real_ids = {r["id"] for r in real}
    ctx.stats["real_ids"] = {
        "rows": len(real),
        "by_category": _counter(real, "category"),
        "pii_scrubbed": sum(1 for r in real if r.get("pii_scrubbed")),
        "by_source": _counter(real, "source"),
    }

    # Candidates and labels, joined to their prompt for source / source_id / split.
    def usable(r: Mapping[str, Any]) -> bool:
        pid = str(r.get("prompt_id"))
        return r.get("stand_in") is not True and pid not in real_ids and not pid.startswith("real-")

    cand_raw = read_jsonl(inputs.candidates)
    cands = [_with_provenance(r, index, "candidate") for r in cand_raw if usable(r)]
    for c in cands:
        ctx.model(model_of(str(c.get("generator") or "")), "candidates")
    n = write_jsonl(out / CANDIDATES_FILE, cands)
    _record_file(ctx, out, CANDIDATES_FILE, n)
    ctx.stats["candidates"] = {
        "rows": len(cands),
        "dropped": len(cand_raw) - len(cands),
        "prompts": len({c["prompt_id"] for c in cands}),
        "by_method": _counter(cands, "method"),
        "by_split": _counter(cands, "split"),
        "rejected": sum(1 for c in cands if c.get("rejected")),
    }

    lab_raw = read_jsonl(inputs.labels)
    labels = [_with_provenance(r, index, "label") for r in lab_raw if usable(r)]
    mismatch = sum(1 for lab in labels if lab["split"] != index[str(lab["prompt_id"])]["split"])
    if mismatch:
        raise ValueError(f"{mismatch} labels disagree with their prompt's split")
    for lab in labels:
        ctx.model(str(lab.get("judge_model") or "") or None, "judge")
        ctx.model(str(lab.get("target_model") or lab.get("target") or "") or None, "target")
    n = write_jsonl(out / LABELS_FILE, labels)
    _record_file(ctx, out, LABELS_FILE, n)
    by_target: dict[str, dict[str, int]] = {}
    for lab in labels:
        t = by_target.setdefault(str(lab.get("target_model")), {})
        t[str(lab.get("label"))] = t.get(str(lab.get("label")), 0) + 1
    ctx.stats["labels"] = {
        "rows": len(labels),
        "dropped": len(lab_raw) - len(labels),
        "by_split": _counter(labels, "split"),
        "by_target_label": {k: dict(sorted(v.items())) for k, v in sorted(by_target.items())},
        "judge_models": _counter(labels, "judge_model"),
        "judge_protocols": _counter(labels, "judge_protocol"),
        "judge_prompt_versions": _counter(labels, "judge_prompt_version"),
        "needs_human_review": sum(1 for lab in labels if lab.get("needs_human_review")),
        "position_inconsistent": sum(
            1 for lab in labels if lab.get("position_consistent") is False
        ),
    }
    ctx.stats["dropped_wildchat_prompts"] = dropped_wildchat
    _check_no_llama(ctx.models)

    # Rubrics and judge prompt templates (project-authored).
    for src_dir, rel_dir, pattern in (
        (rubrics_dir, "rubrics", "*.yaml"),
        (prompts_dir, "rubrics/judge_prompts", "*.txt"),
    ):
        for f in sorted(src_dir.glob(pattern)):
            rel = f"{rel_dir}/{f.name}"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            (out / rel).write_bytes(f.read_bytes())
            _record_file(ctx, out, rel, None)

    sources = sorted(ctx.sources)
    unknown = [s for s in sources if s not in SOURCE_NOTICES]
    if unknown:
        raise ValueError(f"no attribution recorded for source(s) {unknown}")
    models = {m: sorted(r) for m, r in sorted(ctx.models.items())}
    project_licence = _project_licence(inputs.repo)
    code = {
        "repo": REPO_URL,
        "commit": commit,
        "dirty": dirty,
        "paths": list(CODE_PATHS),
        "project_licence_file": project_licence,
    }
    meta = {
        "name": RELEASE_NAME,
        "version": RELEASE_VERSION,
        "built_utc": built.isoformat(timespec="seconds"),
        "data_license": DATA_LICENSE,
        "code": code,
        "sources": sources,
        "real_slice": {"source": WILDCHAT, **_wildchat_pin()},
        "models": models,
        "locked_inputs": _lock_info(inputs),
        "splits_meta": _splits_meta(inputs.splits_dir),
        "read_purpose": PURPOSE,
        "stats": ctx.stats,
    }
    for rel, text in (
        ("LICENSE", render_license(project_licence)),
        ("NOTICE", render_notice(sources, models, real_included=bool(real))),
        ("README.md", render_card(meta, ctx.files)),
    ):
        (out / rel).write_text(text, encoding="utf-8", newline="\n")
        _record_file(ctx, out, rel, None)

    manifest = {**meta, "files": ctx.files}
    (out / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    sums = [*ctx.files, {"path": MANIFEST_FILE, "sha256": sha256_file(out / MANIFEST_FILE)}]
    (out / SUMS_FILE).write_text(
        "".join(f"{f['sha256']}  {f['path']}\n" for f in sums), encoding="utf-8", newline="\n"
    )
    return manifest


def _wildchat_pin() -> dict[str, str]:
    from terser.data.real_slice import LICENSE_SPDX, REPO_ID, REVISION, SHARDS

    return {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "shards": ", ".join(SHARDS),
        "license": LICENSE_SPDX,
    }


# --------------------------------------------------------------------------------------------------
# LICENSE, NOTICE, data card
# --------------------------------------------------------------------------------------------------


def render_license(project_licence: str | None) -> str:
    """The release ``LICENSE``: CC-BY-SA-4.0 for every data file; code is not included."""
    code = (
        f"governed by the repository's `{project_licence}`"
        if project_licence
        else "governed by the Terser repository's licence (no LICENSE file was present in the "
        "repository when this release was built; one must be added before a public release)"
    )
    return f"""\
{RELEASE_NAME}: licence
{"=" * (len(RELEASE_NAME) + 9)}

DATA (every file in this directory: prompts/, candidates/, labels/, rubrics/, README.md, NOTICE,
MANIFEST.json, SHA256SUMS)

  Licensed under the Creative Commons Attribution-ShareAlike 4.0 International License
  (SPDX: {DATA_LICENSE}). Full legal code: {DATA_LICENSE_URL}

  Why ShareAlike: the prompts and candidates are adaptations of databricks-dolly-15k (CC BY-SA 3.0)
  and of text derived from it. CC BY-SA 4.0 is a compatible later version
  (reports/license_audit.md, section "dolly", option (b)). The other upstream sources (MIT,
  CC BY 4.0, Apache-2.0) only require attribution and notices, which NOTICE carries. Labels and
  judge reasons sit next to Dolly-derived ids and may quote prompt text, so they are released
  under the same licence; the rubrics are project-authored and released under it too.

  You may share and adapt this material for any purpose, provided you give appropriate credit
  (see NOTICE), indicate changes, and distribute your contributions under the same licence. No
  additional restrictions, DRM or access gating may be applied to these files.

  Per-record provenance is machine-readable: every prompt, candidate and label record carries
  `source` and `source_id`.

CODE

  No code is included. The code that built and evaluates this benchmark lives in the Terser
  repository at the commit recorded in MANIFEST.json (`code.commit`) and is {code}.

NOTICE lists the attribution required for every upstream source and generating model, and the
changes made to each.
"""


def _section(title: str) -> str:
    return f"\n{title}\n{'-' * len(title)}\n"


def render_notice(
    sources: Sequence[str], models: Mapping[str, Sequence[str]], *, real_included: bool
) -> str:
    """The release ``NOTICE``: attribution per upstream source and per model, with changes."""
    parts = [
        f"{RELEASE_NAME}\nCopyright (c) the Terser authors ({REPO_URL}).\n"
        f"Data licensed under {DATA_LICENSE}; see LICENSE.\n\n"
        "This release contains material derived from the datasets and models below. Each entry\n"
        "gives the upstream, its pinned revision, its licence, the required attribution and the\n"
        "changes made.\n"
    ]
    parts.append(_section("UPSTREAM DATASETS"))
    for name in sources:
        src, note = SOURCES[name], SOURCE_NOTICES[name]
        parts.append(_notice_entry(name, src.repo_id, src.revision, src.license_spdx, note))
    if real_included:
        pin = _wildchat_pin()
        parts.append(
            _notice_entry(
                WILDCHAT, pin["repo_id"], pin["revision"], pin["license"], WILDCHAT_NOTICE
            )
        )
    absent = sorted(set(SOURCE_NOTICES) - set(sources))
    if absent:
        parts.append(
            f"Registered sources with no record in this release (no notice needed): "
            f"{', '.join(absent)}.\n"
        )
    parts.append(_section("GENERATING, JUDGING AND TARGET MODELS"))
    for model_id, roles in models.items():
        m = MODEL_NOTICES.get(model_id, ModelNotice(None, "licence not recorded in the repository"))
        lines = [
            f"* {model_id}",
            f"  Licence: {m.licence or 'NOT RECORDED - check before release'}",
        ]
        lines += [f"  Role: {ROLE_TEXT.get(r, r)}" for r in roles]
        lines.append(f"  Condition: {m.note}")
        parts.append("\n".join(lines) + "\n\n")
    parts.append(
        "* Rule operators (`operators:<stack>` candidates) are Terser code; they add no\n"
        "  third-party terms.\n"
        "* No Llama output is included, so the Llama 3.1 Community Licence does not apply.\n"
        "* Outputs of the target models are not included.\n"
    )
    return "".join(parts)


def _notice_entry(name: str, repo_id: str, revision: str, spdx: str, note: SourceNotice) -> str:
    lines = [
        f"* {name}: https://huggingface.co/datasets/{repo_id} @ {revision}",
        f"  Licence: {spdx} ({note.licence_url})",
        f"  Attribution: {note.attribution}",
        f"  Cite: {note.citation}",
        "  Changes made:",
        *[f"    - {c}" for c in note.changes],
    ]
    text = "\n".join(lines) + "\n"
    if note.mit_copyright:
        indented = "\n".join(f"    {ln}" if ln else "" for ln in _MIT_PERMISSION.splitlines())
        text += f"  MIT notice:\n    {note.mit_copyright}\n\n{indented}\n"
    return text + "\n"


def _fmt(n: Any) -> str:
    return f"{n:,}" if isinstance(n, int) else str(n)


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]], left: int = 1) -> str:
    align = [":---"] * left + ["---:"] * (len(header) - left)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(_fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def render_card(meta: Mapping[str, Any], files: Sequence[Mapping[str, Any]]) -> str:
    """The release ``README.md``: the data card, in the ``docs/data_cards/_template.md`` layout."""
    st = meta["stats"]
    pr = st["prompts"]
    code = meta["code"]
    locks = meta["locked_inputs"]
    real = st["real_ids"]
    cand = st["candidates"]
    lab = st["labels"]
    pin = meta["real_slice"]
    sm = meta.get("splits_meta", {})
    total_rows = sum(pr[s]["rows"] for s in SPLIT_ORDER)
    total_groups = sum(pr[s]["source_prompts"] for s in SPLIT_ORDER)
    commit = code["commit"] or "unknown"
    dirty = " (built from a working tree with uncommitted changes)" if code["dirty"] else ""
    test_lock = locks.get(LOCKED_MAIN, {}).get("sha256", "unknown")
    real_lock = locks.get(LOCKED_REAL, {}).get("sha256", "unknown")

    src_rows = []
    for name in meta["sources"]:
        s = SOURCES[name]
        src_rows.append(
            [f"`{name}`", f"`{s.revision}`", f"https://huggingface.co/datasets/{s.repo_id}", name]
        )
    src_rows.append(
        [
            f"`{WILDCHAT}` (ids only)",
            f"`{pin['revision']}`",
            f"https://huggingface.co/datasets/{pin['repo_id']}",
            "`wildchat` (evaluation-only section)",
        ]
    )
    lic_rows = [
        [f"`{n}`", f"`{SOURCES[n].license_spdx}`", "yes", "yes", f"license_audit.md#{n}"]
        for n in meta["sources"]
    ]
    lic_rows.append(
        [f"`{WILDCHAT}`", f"`{pin['license']}`", "yes", "ids only", "license_audit.md#wildchat"]
    )
    model_rows = [
        [
            f"`{m}`",
            ", ".join(roles),
            (
                MODEL_NOTICES[m].licence
                if m in MODEL_NOTICES and MODEL_NOTICES[m].licence
                else "**not recorded**"
            ),
        ]
        for m, roles in meta["models"].items()
    ]
    split_rows = [
        [
            s if s != LOCKED_MAIN else "test (frozen, unlocked row 12.1)",
            pr[s]["rows"],
            pr[s]["source_prompts"],
            f"{pr[s]['source_prompts'] / total_groups:.1%}" if total_groups else "-",
            f"`{test_lock}`" if s == LOCKED_MAIN else "file sha256 in MANIFEST.json",
        ]
        for s in SPLIT_ORDER
    ]
    split_rows.append(
        ["real (WildChat, ids only)", real["rows"], real["rows"], "-", f"`{real_lock}`"]
    )
    cat_rows = []
    for c in CATEGORIES:
        vals = [pr[s]["by_category"].get(c, 0) for s in SPLIT_ORDER]
        cat_rows.append([c, *vals, sum(vals), real["by_category"].get(c, 0)])
    style_rows = []
    for sty in STYLES:
        n = sum(pr[s]["by_style"].get(sty, 0) for s in SPLIT_ORDER)
        style_rows.append([sty, n, f"{n / total_rows:.1%}" if total_rows else "-"])
    method_rows = [[f"`{m}`", n] for m, n in cand["by_method"].items()]
    label_rows = []
    for t, counts in lab["by_target_label"].items():
        label_rows.append([f"`{t}`", *[counts.get(k, 0) for k in ("preserved", "degraded")]])
    file_rows = [[f"`{f['path']}`", f["bytes"], f["records"] or "-"] for f in files]
    lic_file = code["project_licence_file"]
    by_src = _sum_by(pr, "by_source")
    src_cat = _sum_by(pr, "by_source_category")
    src_cat_rows = []
    for name in sorted(by_src):
        vals = [src_cat.get(f"{name}/{c}", 0) for c in CATEGORIES]
        src_cat_rows.append([f"`{name}`", *vals, sum(vals)])
    judges = ", ".join(f"`{k}` ({_fmt(v)})" for k, v in lab["judge_models"].items())
    protocols = ", ".join(f"`{k}`" for k in lab["judge_protocols"])
    head_rows = [
        ["Card version", f"`{meta['version']}`"],
        ["Plan row(s)", "`12.11` (data from 2.1.1-2.1.8, 2.2.1, 2.3.2, 2.3.3)"],
        ["Owner", "`guru-bharadwaj20`"],
        ["Last updated", f"`{meta['built_utc'][:10]}`"],
        ["Built (UTC)", f"`{meta['built_utc']}`"],
        ["Data licence", f"`{meta['data_license']}` (see `LICENSE`, `NOTICE`)"],
        ["Code", f"{code['repo']} @ `{commit}`{dirty}; not copied into the release"],
        ["Code paths", ", ".join(f"`{p}`" for p in code["paths"])],
        [
            "DVC path",
            "inputs: `data/processed/splits/`, `data/interim/candidates.jsonl`, "
            "`data/processed/labels.jsonl`",
        ],
        ["DVC hash", "see `dvc.lock` at the code commit; sha256 in `MANIFEST.json`, `SHA256SUMS`"],
    ]
    pii_rows = [
        ["Detector", "`terser.data.pii` regex layer, strict policy (row 2.1.3); NER off"],
        ["Action on detection", "Typed placeholder replacement at collection"],
        [
            "Rows affected",
            "See `reports/p2_pii_scrub_smoke.md` and `reports/p2_pii_pilot.md` at the code commit",
        ],
        [
            "Manual sample size / PII found",
            "500 rows reviewed by hand, 0 residual PII (`reports/p2_pii_pilot_sample.csv`)",
        ],
        [
            "Real user prompts included",
            f"**No text.** {_fmt(real['rows'])} WildChat ids only "
            f"(prompts changed by the PII scrub at slice build: {_fmt(real['pii_scrubbed'])})",
        ],
    ]
    bias_rows = [
        [
            "Labels are one judge's estimates",
            f"Judge(s): {judges}; protocol {protocols}; {_fmt(lab['needs_human_review'])} of "
            f"{_fmt(lab['rows'])} labels flagged `needs_human_review`, "
            f"{_fmt(lab['position_inconsistent'])} position-inconsistent",
            "Preservation rates inherit judge bias; no human validation yet (rows 2.3.5, 2.3.6)",
        ],
        [
            "Casual styles are model-generated",
            "All casual prompts come from the casualisation model in section 1",
            "A rewriter can score well by undoing that model's verbosity; the WildChat slice is "
            "the check",
        ],
        [
            "Source mix",
            "Rows by source: " + ", ".join(f"{k} {_fmt(v)}" for k, v in by_src.items()),
            "Each category comes from few sources (section 9), so category and source effects "
            "are confounded",
        ],
        [
            "Contamination",
            "HumanEval, MBPP and GSM8K are public evaluation sets",
            "Affects target-output quality, not the savings / preservation comparison",
        ],
        ["English only", "All sources are English", "No claims about other languages"],
    ]
    size_rows = [[s, pr[s]["rows"], pr[s]["source_prompts"]] for s in SPLIT_ORDER]
    size_rows.append(["**Total**", total_rows, total_groups])
    maint_rows = [
        [
            meta["version"],
            meta["built_utc"][:10],
            "see `dvc.lock`",
            f"`{commit}`",
            "Initial release build (row 12.11)",
        ]
    ]
    lic_header = ["Source", "Licence", "Research use", "Redistribution", "Notes / audit link"]

    return f"""\
# Data Card: `{meta['name']}` (Terser benchmark release)

Casual prompts, compressed candidates, judge labels and rubrics from the Terser project:
tokenizer-aware rewriting of casual prompts into shorter prompts that keep the answer. Built by
`python -m terser.phase12.benchmark build` (plan row **12.11**).

> **Known limitations:**
>
> 1. **PII (row 2.1.3).** The text was scrubbed automatically (regex policy,
>    `docs/pii_policy.md`); the 500-row manual sample (`reports/p2_pii_pilot_sample.csv`) found
>    0 residual PII. Report any residual PII as a repository issue.
> 2. **Labels are judge estimates (rows 2.3.5, 2.3.6).** Human annotation of 300 pairs and the
>    judge-human agreement analysis are open, so no label in this release has been validated
>    against humans.
> 3. **WildChat text is withheld.** The real-slice prompts are user-written chat and are not
>    included (ids and sha256 only, section 7); rebuild them locally with `rebuild-real`.
> 4. **Project licence.** Repository licence file at build time:
>    {f"`{lic_file}`" if lic_file else "**none**"}.

{_table(["Field", "Value"], head_rows, left=2)}

---

## 1. Source

| Upstream | Version / revision | URL | `SOURCES.md` entry |
| :--- | :--- | :--- | :--- |
{chr(10).join("| " + " | ".join(r) + " |" for r in src_rows)}

Generated text and labels:

{_table(["Model", "Role in this release", "Licence"], model_rows, left=3)}

Rule operators (`ops_*` candidates) are Terser code. Prompt templates and model settings are in
the code at the commit above (`src/terser/data/casualise.py`, `src/terser/data/candidates.py`,
`src/terser/judge/`).

## 2. Licence and redistribution

{_table(lic_header, lic_rows, left=5)}

All data files are released under **{meta['data_license']}** because Dolly's ShareAlike clause
covers the Dolly-derived prompts and candidates (`reports/license_audit.md`, option (b)). The MIT,
CC-BY-4.0 and Apache-2.0 conditions are met by the notices in `NOTICE`. No Llama output is
included.

## 3. Size

{_table(["Split", "Rows", "Source prompts"], size_rows)}

| Item | Rows |
| :--- | ---: |
| Candidates (`{CANDIDATES_FILE}`) | {_fmt(cand['rows'])} for {_fmt(cand['prompts'])} prompts |
| of which rejected by the generator checks (`rejected: true`) | {_fmt(cand['rejected'])} |
| Labels (`{LABELS_FILE}`) | {_fmt(lab['rows'])} |
| Real-slice ids (`{REAL_IDS_FILE}`) | {_fmt(real['rows'])} |

Candidates per method:

{_table(["Method", "Candidates"], method_rows)}

Token statistics are not part of this card; the repository's `token_counts` stage has them.

Files:

{_table(["File", "Bytes", "Records"], file_rows)}

## 4. Collection method

Upstream rows were downloaded at the pinned revisions with `python -m terser.data.sources
download`, parsed to a common schema and PII-scrubbed at the collection boundary
(`terser.data.collect`). See `data/SOURCES.md` and `docs/data_pipeline.md` at the code commit.
The DVC stages `collect -> casualise -> dedupe -> split -> label_subset -> candidates -> respond ->
label` produce the inputs of this release.

Record fields:

- `prompts/<split>.jsonl`: `id`, `split`, `source`, `source_id`, `category`, `style`, `prompt`
  (the casual prompt), `clean` (the clean reference prompt), `generator_model`, `prompt_version`.
- `{CANDIDATES_FILE}`: the Phase 2 candidate record (`prompt_id`, `candidate_id`, `method`,
  `text`, `generator`, `rejected`, ...) plus `source`, `source_id`, `split` of its prompt.
- `{LABELS_FILE}`: the `label_v1` record of `data/processed/labels.jsonl` (verdicts, `label`,
  `label_int`, judge probabilities and reasons, review flags) plus `source`, `source_id`.
- `{REAL_IDS_FILE}`: `id`, `source`, `source_id`, `category`, `category_source`,
  `pii_scrubbed`, `use`, `split`, `prompt_sha256`; **no text**.

## 5. Casualisation process

Casual prompts were written by the casualisation model in section 1 from the clean prompt, one
record per (source prompt, style), and checked for meaning (`meaning_check`, row 2.1.4). Rows
the check rejected went to `casual_rejected.jsonl`, which feeds neither the splits nor this
release.

{_table(["Style", "Rows", "Share"], style_rows)}

## 6. Splits

{_table(["Split", "Rows", "Source prompts", "Share of source prompts", "Hash"], split_rows)}

Grouping key: `source_id` (all casual variants of one source prompt share a split, row 2.1.6) ·
Seed: `{sm.get('seed', 'unknown')}` · Leakage test: `tests/test_data_splits.py`; `splits.json`
records `leakage.ok = {sm.get('leakage_ok', 'unknown')}` with
{sm.get('n_shared_groups', 'unknown')} shared source prompts. The test split was read for this
release through `terser.phase12.unlock.read_locked` (purpose `"{meta['read_purpose']}"`),
which logs the read in `docs/checkpoints/C12/test_access_log.jsonl`.

Labels per split: {", ".join(f"{k} {_fmt(v)}" for k, v in lab['by_split'].items())}. Candidates
per split: {", ".join(f"{k} {_fmt(v)}" for k, v in cand['by_split'].items())}.

## 7. PII handling

{_table(["Item", "Value"], pii_rows, left=2)}

**WildChat real slice.** `{REAL_IDS_FILE}` lists the {_fmt(real['rows'])} real-slice prompts by
WildChat conversation hash (`source_id`). To recreate the text locally from
`{pin['repo_id']}` @ `{pin['revision']}` (shard `{pin['shards']}`, {pin['license']}), run from a
checkout of the code commit:

```bash
python -m terser.phase12.benchmark rebuild-real --release <this dir> --out real_prompts.jsonl
```

It downloads the pinned shard if needed, re-runs the row 2.1.8 filters, dedupe and PII scrub
(`terser.data.real_slice`) and checks every prompt against `prompt_sha256`. The hash pins the
text of the frozen slice. When the current scrubber would change a frozen prompt, the command
restores the frozen text and marks the record `scrub_drift: true`. Review those prompts by hand.
Keep the rebuilt file local; it is real user text.

## 8. Known biases

{_table(["Bias", "Evidence", "Likely effect on results"], bias_rows, left=3)}

## 9. Task-category balance

{_table(["Category", "train", "val", "test", "Total", "real (ids)"], cat_rows)}

Labels per target model:

{_table(["Target", "preserved", "degraded"], label_rows)}

Rows per source and category:

{_table(["Source", *CATEGORIES, "Total"], src_cat_rows)}

## 10. Intended use

- Intended:
  - Research on prompt compression: measuring token savings against answer preservation.
  - Reproducing the Terser Phase 2 labels and the Phase 12 evaluation with the code commit above.
- Out of scope:
  - Redistribution that ignores the known limitations at the top of this card.
  - Treating labels as human ground truth.
  - Training on the test split when reporting results on it.
  - Anything that tries to identify WildChat users or other authors of upstream text.

## 11. Maintenance and version

{_table(["Version", "Date", "DVC hash", "Git SHA", "Change"], maint_rows, left=5)}

Check the files with `python -m terser.phase12.benchmark verify --release <this directory>` or
`sha256sum -c SHA256SUMS`. A new version is built when the inputs change (new DVC version, PII
fixes from the 2.1.3 review, human-validated labels).
"""


def _sum_by(pr: Mapping[str, Any], key: str) -> dict[str, int]:
    total: Counter[str] = Counter()
    for s in SPLIT_ORDER:
        total.update(pr[s][key])
    return dict(sorted(total.items()))


# --------------------------------------------------------------------------------------------------
# Verify and WildChat rebuild
# --------------------------------------------------------------------------------------------------


def verify(release: Path) -> dict[str, Any]:
    """Re-hash every manifest file; ``{"ok", "problems", "extra"}`` (extra files are allowed)."""
    manifest = json.loads((release / MANIFEST_FILE).read_text(encoding="utf-8"))
    problems = []
    listed = {f["path"] for f in manifest["files"]}
    for f in manifest["files"]:
        p = release / f["path"]
        if not p.is_file():
            problems.append(f"missing {f['path']}")
        elif p.stat().st_size != f["bytes"] or sha256_file(p) != f["sha256"]:
            problems.append(f"changed {f['path']}")
    present = {p.relative_to(release).as_posix() for p in release.rglob("*") if p.is_file()}
    extra = sorted(present - listed - {MANIFEST_FILE, SUMS_FILE})
    return {"ok": not problems, "problems": problems, "extra": extra, "files": len(listed)}


def rebuild_real(
    release: Path,
    out: Path,
    *,
    raw_dir: Path | None = None,
    rows: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Recreate the real-slice prompt text from the pinned WildChat revision and check it.

    ``rows`` overrides reading the raw shard (tests). Without it the shard under
    ``raw_dir/wildchat/`` is verified against its manifest, and downloaded first if missing.
    A prompt whose current scrub no longer matches the released hash, but whose unscrubbed first
    turn does, is restored as frozen and listed under ``scrub_drift``. The frozen slice predates
    a later scrubber change.
    """
    from terser.data import real_slice as rs
    from terser.data.sources import MANIFEST_NAME, verify_manifest

    ids = read_jsonl(release / REAL_IDS_FILE)
    if rows is None:
        raw = raw_dir or rs.DEFAULT_RAW_DIR
        if not (raw / rs.SOURCE_NAME / MANIFEST_NAME).is_file():
            rs.download(raw)
        problems = verify_manifest(raw / rs.SOURCE_NAME)
        if problems:
            raise RuntimeError("raw WildChat shard failed verification: " + "; ".join(problems))
        rows = rs.read_raw_rows(raw)
    wanted = {str(r["source_id"]) for r in ids}
    raw_turns: dict[str, list[str]] = {}

    def capture(src: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
        """Pass rows through, keeping the unscrubbed first turn of every released id."""
        for row in src:
            sid = f"{rs.SOURCE_NAME}:{row.get('conversation_hash')}"
            conv = row.get("conversation")
            if sid in wanted and conv is not None and len(conv) > 0:
                raw_turns.setdefault(sid, []).append(str(conv[0].get("content") or "").strip())
            yield row

    pool: dict[str, dict[str, Any]] = {}
    for rec in rs.build_records(capture(rows), rs.SliceSummary()):
        pool.setdefault(str(rec["source_id"]), rec)
    rebuilt: list[dict[str, Any]] = []
    missing: list[str] = []
    mismatch: list[str] = []
    drift: list[str] = []
    for r in ids:
        sid = str(r["source_id"])
        found = pool.get(sid)
        if found is not None and text_sha(str(found["prompt"])) == r["prompt_sha256"]:
            text, drifted = str(found["prompt"]), False
        else:
            # The frozen slice was scrubbed by the PII code of its build date; when today's
            # scrubber changes a prompt, the released hash still pins the frozen text.
            same = [t for t in raw_turns.get(sid, []) if text_sha(t) == r["prompt_sha256"]]
            if not same:
                (mismatch if found is not None else missing).append(r["id"])
                continue
            text, drifted = same[0], True
            drift.append(r["id"])
        rebuilt.append(
            {**{k: r.get(k) for k in REAL_ID_FIELDS}, "prompt": text, "scrub_drift": drifted}
        )
    write_jsonl(out, rebuilt)
    return {
        "ids": len(ids),
        "rebuilt": len(rebuilt),
        "missing": missing,
        "sha256_mismatch": mismatch,
        "scrub_drift": drift,
        "out": out.as_posix(),
    }


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (``build`` / ``verify`` / ``rebuild-real``)."""
    ap = argparse.ArgumentParser(prog="python -m terser.phase12.benchmark", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write the release directory")
    b.add_argument("--out", type=Path, default=Path("release") / RELEASE_NAME)
    b.add_argument("--force", action="store_true", help="replace an existing release build")
    v = sub.add_parser("verify", help="re-hash the files against MANIFEST.json")
    v.add_argument("--release", type=Path, default=Path("release") / RELEASE_NAME)
    r = sub.add_parser("rebuild-real", help="recreate the WildChat slice text from its ids")
    r.add_argument("--release", type=Path, default=Path("release") / RELEASE_NAME)
    r.add_argument("--out", type=Path, required=True, help="output JSONL (real user text)")
    r.add_argument("--raw-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        manifest = build(args.out, force=args.force)
        summary = {
            "out": args.out.as_posix(),
            "code": manifest["code"],
            "files": [{k: f[k] for k in ("path", "bytes", "records")} for f in manifest["files"]],
            "stats": manifest["stats"],
        }
        print(json.dumps(summary, indent=2))
        return 0
    if args.cmd == "verify":
        res = verify(args.release)
        print(json.dumps(res, indent=2))
        return 0 if res["ok"] else 1
    res = rebuild_real(args.release, args.out, raw_dir=args.raw_dir)
    print(json.dumps(res, indent=2))
    return 0 if not res["missing"] and not res["sha256_mismatch"] else 1


if __name__ == "__main__":
    sys.exit(main())
