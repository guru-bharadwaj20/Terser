"""0.1.2: the core numerical and classical ML stack imports and works."""

import numpy as np
import pytest


def test_numpy_pandas_scipy() -> None:
    import pandas as pd
    from scipy import stats

    frame = pd.DataFrame({"x": np.arange(10, dtype=float)})
    assert frame.x.mean() == 4.5
    assert stats.spearmanr(frame.x, frame.x).statistic == pytest.approx(1.0)


def test_sklearn_and_lightgbm_fit() -> None:
    import lightgbm as lgb
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    X, y = make_classification(n_samples=200, n_features=5, random_state=0)
    assert LogisticRegression().fit(X, y).score(X, y) > 0.8
    model = lgb.LGBMClassifier(n_estimators=20, verbose=-1).fit(X, y)
    assert model.score(X, y) > 0.8


def test_cvxopt_solves_qp() -> None:
    from cvxopt import matrix, solvers

    # minimise x^2 + y^2 subject to x + y = 1  ->  x = y = 0.5
    solvers.options["show_progress"] = False
    sol = solvers.qp(
        matrix([[2.0, 0.0], [0.0, 2.0]]),
        matrix([0.0, 0.0]),
        A=matrix([[1.0], [1.0]]),
        b=matrix([1.0]),
    )
    assert np.allclose(np.array(sol["x"]).ravel(), [0.5, 0.5], atol=1e-6)


def test_matplotlib_headless(tmp_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = tmp_path / "plot.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.stat().st_size > 0
