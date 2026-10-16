"""Populate the model registry with lineage (rows **11.2.2**, **11.2.3**).

Registers every served component under :data:`terser.tracking.lineage.MODEL_NAMES` and walks it
through the stage aliases ``None -> Staging -> Production -> Archived``:

* ``gate`` and ``tagger`` (``models/fast/*.onnx``, row 11.3.2): Staging, then Production; they
  serve the CPU fast path.
* ``rewriter``: the SFT rungs of rows 8.3.3 / 8.3.4 (``models/rewriter/*``). None passes C8.3,
  so the best rung is held at Staging and the others are Archived. No rewriter is in Production
  until one passes C8.3 (the promotion guard, row 11.4.4, would refuse it).

Each version carries the lineage tags of row 11.2.3 (git SHA, DVC data hash, config + sha256,
checkpoints passed, source run); metrics come from ``reports/p11_fast_path.json`` and
``reports/p8_rewriter_val.json``. :func:`terser.tracking.lineage.trace` is run on every
Production / Staging alias as the check. Output: ``reports/p11_registry.md``.

CLI::

    python -m terser.phase11.register
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlflow

from terser.tracking.lineage import MODEL_NAMES, register_with_lineage, trace
from terser.tracking.runs import promote, start_run

__all__ = ["COMPONENTS", "load_metrics", "main"]

REWRITER_METRICS = {"median_savings": "median_savings", "preserved": "preserved"}
# Each component's metrics: (json file, key or None, {logged name: field}). The files are
# written by the runs that measured them (row 11.3.2 trainer, row 8.3.5 val report).
COMPONENTS: list[dict[str, Any]] = [
    {
        "component": "gate",
        "path": "models/fast/gate.onnx",
        "config": {"model": "GradientBoostingClassifier", "n_estimators": 200, "max_depth": 3},
        "checkpoints": ["C3.1"],
        "stages": ["Staging", "Production"],
        "metrics": ("reports/p11_fast_path.json", None, {"grouped_auc": "gate_auc"}),
    },
    {
        "component": "tagger",
        "path": "models/fast/tagger.onnx",
        "config": {"model": "DecisionTreeClassifier", "max_depth": 10, "min_samples_leaf": 50},
        "checkpoints": [],
        "stages": ["Staging", "Production"],
        "metrics": ("reports/p11_fast_path.json", None, {"droppable_macro_f1": "tagger_macro_f1"}),
    },
    {
        "component": "rewriter",
        "path": "models/rewriter/sft-qwen0.5b-full",
        "config": {"experiment": "sft_qwen05b_full", "row": "8.3.3"},
        "checkpoints": [],
        "stages": ["Staging", "Archived"],
        "metrics": ("reports/p8_rewriter_val.json", "sft05", REWRITER_METRICS),
    },
    {
        "component": "rewriter",
        "path": "models/rewriter/sft-qwen1.5b-lora",
        "config": {"experiment": "sft_qwen15b_lora", "row": "8.3.4"},
        "checkpoints": [],
        "stages": ["Staging", "Archived"],
        "metrics": ("reports/p8_rewriter_val.json", "sft15", REWRITER_METRICS),
    },
    {
        "component": "rewriter",
        "path": "models/rewriter/sft-qwen3b-lora",
        "config": {"experiment": "sft_qwen3b_lora", "row": "8.3.4"},
        "checkpoints": [],
        "stages": ["Staging"],
        "metrics": ("reports/p8_rewriter_val.json", "sft3b", REWRITER_METRICS),
    },
]


class _Artifacts(mlflow.pyfunc.PythonModel):  # type: ignore[misc]
    """Registry wrapper: the version's files live under ``artifacts['weights']``; serving loads
    them with ONNX Runtime / vLLM, not through ``predict``."""

    def predict(self, context: Any, model_input: Any, params: Any = None) -> Any:
        """Return the artifact path; the input is ignored (the model is a file bundle)."""
        del model_input, params
        return context.artifacts["weights"]


def load_metrics(src: str, key: str | None, fields: dict[str, str]) -> dict[str, float]:
    """Read ``fields`` (logged name -> json field) from ``src`` (under ``key`` if given)."""
    d = json.loads(Path(src).read_text(encoding="utf-8"))
    if key is not None:
        d = d[key]
    return {name: float(d[field]) for name, field in fields.items()}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Register served models with lineage.")
    ap.add_argument("--report", type=Path, default=Path("reports/p11_registry.md"))
    args = ap.parse_args(argv)
    rows = []
    for c in COMPONENTS:
        p = Path(c["path"])
        name = p.stem if p.is_file() else p.name
        with start_run(
            phase=11,
            row="11.2.2",
            task=f"register_{c['component']}",
            run_name=f"11-register-{c['component']}-{name.replace('.', '')}",
            params=c["config"],
        ) as run:
            mlflow.log_metrics(load_metrics(*c["metrics"]))
            # MLflow 3 registers logged models only, so the files ride as pyfunc artifacts.
            mlflow.pyfunc.log_model(
                name="model", python_model=_Artifacts(), artifacts={"weights": str(p)}
            )
            run_id = run.info.run_id
        mv = register_with_lineage(run_id, "model", c["component"], c["config"], c["checkpoints"])
        for stage in c["stages"]:
            promote(MODEL_NAMES[c["component"]], mv.version, stage)
        rows.append(
            {
                "component": c["component"],
                "registry name": MODEL_NAMES[c["component"]],
                "version": mv.version,
                "source": c["path"],
                "stage history": " -> ".join(["None", *c["stages"]]),
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    traced = {
        "gate (production)": trace("gate", "production"),
        "tagger (production)": trace("tagger", "production"),
        "rewriter (staging)": trace("rewriter", "staging"),
    }
    lines = [
        "# Model registry and lineage (rows 11.2.2, 11.2.3)",
        "",
        "Generated by `python -m terser.phase11.register` against the repo-local MLflow store.",
        "",
        "| " + " | ".join(rows[0]) + " |",
        "|---" * len(rows[0]) + "|",
        *("| " + " | ".join(str(v) for v in r.values()) + " |" for r in rows),
        "",
        "No rewriter is in Production: every SFT rung fails C8.3, so the best-preserving rung "
        "(3B LoRA) is held at Staging and the earlier rungs are Archived.",
        "",
        "## Lineage traces (`terser.tracking.lineage.trace`)",
        "",
    ]
    for k, t in traced.items():
        lines += [f"**{k}**", ""]
        lines += [
            f"- `{f}`: `{str(t.get(f))[:80]}`"
            for f in (
                "version",
                "git_sha",
                "data_hash",
                "config_sha256",
                "checkpoints_passed",
                "source_run_id",
            )
        ]
        lines.append("")
    args.report.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
