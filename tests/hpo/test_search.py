"""Tests for terser.hpo.search (row 10.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("optuna")

from terser.hpo import Categorical, Float, Int, SearchSpace, run_search

SPACE = SearchSpace({"x": Float(-3.0, 3.0), "k": Int(1, 5), "c": Categorical(("a", "b"))})


def quad(p: dict[str, object]) -> float:
    """Maximum 0 at x=1, k=3, c='b'."""
    x, k = float(p["x"]), int(p["k"])  # type: ignore[arg-type]
    return -((x - 1.0) ** 2) - (k - 3) ** 2 - (0.0 if p["c"] == "b" else 1.0)


def test_grid_discretisation() -> None:
    g = SPACE.grid(4)
    assert g["x"] == pytest.approx([-3.0, -1.0, 1.0, 3.0])
    assert g["k"] == [1, 2, 4, 5] and g["c"] == ["a", "b"]
    assert SPACE.grid_size(4) == 32
    log = SearchSpace({"lr": Float(1e-4, 1e-1, log=True)}).grid(4)["lr"]
    assert log == pytest.approx([1e-4, 1e-3, 1e-2, 1e-1])
    assert 1e-4 <= min(log) and max(log) <= 1e-1  # endpoints never step outside the bounds


def test_gp_survives_failed_startup_trials() -> None:
    calls = {"n": 0}

    def flaky(p: dict[str, object]) -> float:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise ValueError("startup trial failed")
        return quad(p)

    res = run_search(flaky, SPACE, method="gp", n_trials=4, n_startup=2, catch=(ValueError,))
    assert len(res.values) == 2


def test_invalid_space() -> None:
    with pytest.raises(ValueError):
        SearchSpace({})
    with pytest.raises(ValueError):
        SearchSpace({"lr": Float(0.0, 1.0, log=True)})
    with pytest.raises(ValueError):
        run_search(quad, SPACE, "anneal", 3)  # type: ignore[arg-type]


@pytest.mark.parametrize("method", ["grid", "random", "tpe", "gp"])
def test_methods_find_good_point(method: str) -> None:
    res = run_search(quad, SPACE, method, n_trials=30, seed=0, n_startup=8)  # type: ignore[arg-type]
    assert len(res.values) == 30 and res.method == method
    assert res.best_value == max(res.values) == res.best_so_far[-1]
    assert res.best_so_far == sorted(res.best_so_far)  # monotone for maximize
    assert res.best_value > -1.5


def test_grid_caps_at_grid_size_and_covers_it() -> None:
    space = SearchSpace({"a": Categorical((1, 2, 3)), "b": Int(0, 1)})
    res = run_search(lambda p: float(p["a"] + p["b"]), space, "grid", n_trials=100)
    assert len(res.values) == 6 and res.best_value == 4.0
    assert {(p["a"], p["b"]) for p in res.params} == {(a, b) for a in (1, 2, 3) for b in (0, 1)}


def test_seeded_reproducible_and_minimize() -> None:
    a = run_search(quad, SPACE, "tpe", 12, seed=3, direction="minimize")
    b = run_search(quad, SPACE, "tpe", 12, seed=3, direction="minimize")
    assert a.values == b.values
    assert a.best_so_far == sorted(a.best_so_far, reverse=True)


def test_gp_beats_random_on_smooth_objective() -> None:
    space = SearchSpace({"x": Float(-5.0, 5.0), "y": Float(-5.0, 5.0)})

    def f(p: dict[str, float]) -> float:
        return -((p["x"] - 1.3) ** 2) - (p["y"] + 0.7) ** 2

    gp = [run_search(f, space, "gp", 25, seed=s).best_value for s in range(3)]
    rnd = [run_search(f, space, "random", 25, seed=s).best_value for s in range(3)]
    assert sum(gp) > sum(rnd)


def test_mlflow_parent_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    res = run_search(
        quad,
        SPACE,
        "random",
        4,
        mlflow=True,
        mlflow_tags={"phase": 10, "row": "10.1", "task": "test", "run_name": "10-hpo-q-rnd-0"},
    )
    assert res.mlflow_run_id is not None
    client = mlflow.MlflowClient()
    parent = client.get_run(res.mlflow_run_id)
    assert parent.data.tags["row"] == "10.1"
    assert parent.data.metrics["best_value"] == pytest.approx(res.best_value)
    children = client.search_runs(
        [parent.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{res.mlflow_run_id}'",
    )
    assert len(children) == 4
    hist = client.get_metric_history(res.mlflow_run_id, "best_so_far")
    assert [m.value for m in sorted(hist, key=lambda m: m.step)] == res.best_so_far
