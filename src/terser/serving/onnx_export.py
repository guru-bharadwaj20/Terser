"""Export a scikit-learn gate (or span classifier) to ONNX for the CPU fast path (row 11.3.2).

``export_sklearn(model, n_features, path)`` converts with ``skl2onnx`` (float32 input named
``input``, ZipMap disabled so probabilities come back as a plain ``[n, 2]`` array) and
:func:`check_parity` asserts the ONNX Runtime probabilities match ``model.predict_proba`` within
``atol``. Exported files are loaded by :class:`terser.serving.paths.OnnxGate`.

CLI (once the trained gate exists as a joblib/pickle file)::

    python -m terser.serving.onnx_export --model models/gate/gate.joblib --out models/gate/gate.onnx
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from terser.features.shared import GATE_FEATURE_NAMES

__all__ = ["check_parity", "export_sklearn"]


def export_sklearn(model: Any, n_features: int, path: str | Path) -> Path:
    """Write ``model`` as ONNX to ``path`` and return the path."""
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    onx = to_onnx(
        model,
        initial_types=[("input", FloatTensorType([None, n_features]))],
        options={id(model): {"zipmap": False}},
        target_opset=17,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(onx.SerializeToString())
    return out


def check_parity(model: Any, path: str | Path, x: np.ndarray, atol: float = 1e-4) -> float:
    """Max |P_sklearn - P_onnx| over ``x``; raises ``AssertionError`` if it exceeds ``atol``."""
    from terser.serving.paths import OnnxGate

    onnx_p = np.asarray(OnnxGate(path).score_vectors(x.tolist()))
    ref = model.predict_proba(x)[:, 1]
    diff = float(np.max(np.abs(onnx_p - ref)))
    if diff > atol:
        raise AssertionError(f"ONNX/sklearn probability mismatch {diff:.2e} > {atol}")
    return diff


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: export a pickled/joblib sklearn gate."""
    import joblib

    ap = argparse.ArgumentParser(description="Export a trained gate to ONNX")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-features", type=int, default=len(GATE_FEATURE_NAMES))
    args = ap.parse_args(argv)
    model = joblib.load(args.model)
    export_sklearn(model, args.n_features, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
