"""PyTorch MLP quality gate scored under the grouped CV protocol (plan row **4.1.4**).

The ``1536 -> 256 -> 64 -> 1`` network of :func:`~terser.models.neural.mlp.build_quality_gate_mlp`
is trained with Adam on ``BCEWithLogitsLoss`` inside every fold of
:func:`~terser.eval.cv.out_of_fold_scores` (row 3.2.4), with standardisation fitted on the fold's
training part only. The out-of-fold scores give ROC-AUC, specificity at sensitivity 0.95
(row 3.2.3) and the per-category table every performance report carries.

``python -m terser.phase4.quality_gate --table <meta> --features data/features/pair_emb.npy``.
``torch`` is imported lazily (CPU is enough for 410k parameters).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from terser.eval.cv import FitPredict

__all__ = ["main", "make_torch_fit_predict"]

FloatArray = npt.NDArray[np.float64]


def make_torch_fit_predict(
    sizes: Sequence[int] | None = None,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 64,
    dropout: float = 0.1,
    seed: int = 0,
) -> FitPredict:
    """Return a ``fit_predict(X_train, y_train, X_val) -> scores`` closure for the CV helper."""

    def fit_predict(X: FloatArray, y: npt.NDArray[np.int64], Xv: FloatArray) -> FloatArray:
        import torch

        from terser.models.neural.mlp import GATE_SIZES, build_quality_gate_mlp

        torch.manual_seed(seed)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        net = build_quality_gate_mlp(
            tuple(sizes) if sizes else (X.shape[1], *GATE_SIZES[1:]), dropout
        )
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        gen = torch.Generator().manual_seed(seed)
        net.train()
        for _ in range(epochs):
            perm = torch.randperm(len(yt), generator=gen)
            for i in range(0, len(yt), batch_size):
                b = perm[i : i + batch_size]
                opt.zero_grad()
                loss = loss_fn(net(Xt[b]).squeeze(-1), yt[b])
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            Xvt = torch.tensor((Xv - mu) / sd, dtype=torch.float32)
            return np.asarray(torch.sigmoid(net(Xvt).squeeze(-1)).numpy(), dtype=np.float64)

    return fit_predict


def main(argv: Sequence[str] | None = None) -> int:
    from terser.eval.curves import roc_auc
    from terser.eval.cv import out_of_fold_scores
    from terser.eval.threshold import choose_threshold
    from terser.phase4.tables import load_table

    ap = argparse.ArgumentParser(prog="python -m terser.phase4.quality_gate", description=__doc__)
    ap.add_argument(
        "--table", type=Path, required=True, help="metadata: id, group, category, label"
    )
    ap.add_argument("--features", type=Path, default=Path("data/features/pair_emb.npy"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args(argv)
    t = load_table(args.table, args.features)
    for seed in range(args.seeds):
        fp = make_torch_fit_predict(epochs=args.epochs, seed=seed)
        oof = out_of_fold_scores(fp, t.X, t.y, t.categories, t.groups, seed=seed)
        choice = choose_threshold(t.y, oof.scores)
        print(f"seed {seed}: AUC {roc_auc(t.y, oof.scores):.4f}  {choice}")
        cats = np.asarray(t.categories)
        for c in sorted(set(t.categories)):
            mask = cats == c
            if len(np.unique(t.y[mask])) == 2:
                print(f"  {c}: AUC {roc_auc(t.y[mask], oof.scores[mask]):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
