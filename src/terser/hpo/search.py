"""Grid, random, TPE and GP hyperparameter search behind one interface (plan row **10.1**).

All four methods run as an Optuna study so their trials, best-so-far curves and MLflow logging are
identical; only the proposal mechanism differs:

* ``grid``   -> :class:`optuna.samplers.GridSampler` over the space discretised by
  :meth:`SearchSpace.grid` (``Int``/``Float`` get ``grid_points`` evenly spaced values).
* ``random`` -> :class:`optuna.samplers.RandomSampler`.
* ``tpe``    -> :class:`optuna.samplers.TPESampler` (Tree-structured Parzen Estimator).
* ``gp``     -> Gaussian-process Bayesian optimisation implemented here with scikit-learn
  (Matern 5/2 kernel, expected improvement maximised over random candidates). Optuna's own
  ``GPSampler`` needs torch, which the CPU install does not have. The first ``n_startup`` trials
  are random, then every proposal is enqueued as a fixed trial.

Every search maximises or minimises ``objective(params) -> float``. With ``mlflow=True`` the study
is one MLflow parent run (tags ``phase``/``row``/... from :func:`terser.tracking.start_run`) and
each trial a nested child run with its params and ``objective``; the parent logs ``best_value``
and the per-trial ``best_so_far`` curve.

Data access: an objective that loads data should read it through a
:class:`terser.hpo.audit.DataAccessLog` passed as ``data_log`` so row **10.8** can prove the
search never read the test split.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from terser.hpo.audit import DataAccessLog

__all__ = [
    "METHODS",
    "Categorical",
    "Float",
    "Int",
    "SearchResult",
    "SearchSpace",
    "run_search",
]

Method = Literal["grid", "random", "tpe", "gp"]
METHODS: tuple[str, ...] = ("grid", "random", "tpe", "gp")
Params = dict[str, Any]


@dataclass(frozen=True)
class Int:
    """Integer in ``[low, high]`` (inclusive), optionally log-scaled."""

    low: int
    high: int
    log: bool = False


@dataclass(frozen=True)
class Float:
    """Float in ``[low, high]``, optionally log-scaled."""

    low: float
    high: float
    log: bool = False


@dataclass(frozen=True)
class Categorical:
    """One of a fixed list of choices."""

    choices: tuple[Any, ...]


Dim = Int | Float | Categorical


@dataclass(frozen=True)
class SearchSpace:
    """Named dimensions of a search space."""

    dims: Mapping[str, Dim]

    def __post_init__(self) -> None:
        if not self.dims:
            raise ValueError("search space is empty")
        for name, d in self.dims.items():
            if isinstance(d, Int | Float) and d.low > d.high:
                raise ValueError(f"{name}: low > high")
            if isinstance(d, Int | Float) and d.log and d.low <= 0:
                raise ValueError(f"{name}: log scale needs low > 0")
            if isinstance(d, Categorical) and not d.choices:
                raise ValueError(f"{name}: no choices")

    def suggest(self, trial: Any) -> Params:
        """Draw parameters from an Optuna trial."""
        out: Params = {}
        for name, d in self.dims.items():
            if isinstance(d, Int):
                out[name] = trial.suggest_int(name, d.low, d.high, log=d.log)
            elif isinstance(d, Float):
                out[name] = trial.suggest_float(name, d.low, d.high, log=d.log)
            else:
                out[name] = trial.suggest_categorical(name, list(d.choices))
        return out

    def grid(self, grid_points: int) -> dict[str, list[Any]]:
        """Discretise every dimension (``grid_points`` values for numeric ones)."""
        if grid_points < 1:
            raise ValueError("grid_points must be >= 1")
        out: dict[str, list[Any]] = {}
        for name, d in self.dims.items():
            if isinstance(d, Categorical):
                out[name] = list(d.choices)
                continue
            lo, hi = (math.log(d.low), math.log(d.high)) if d.log else (d.low, d.high)
            raw = np.linspace(lo, hi, grid_points)
            vals = np.exp(raw) if d.log else raw
            if isinstance(d, Int):
                out[name] = sorted({round(float(v)) for v in vals})
            else:
                out[name] = [float(v) for v in vals]
        return out

    def grid_size(self, grid_points: int) -> int:
        """Number of points in :meth:`grid`."""
        return int(np.prod([len(v) for v in self.grid(grid_points).values()]))

    # ---- encoding used by the GP sampler -------------------------------------------------------
    def encode(self, params: Mapping[str, Any]) -> np.ndarray:
        """Map params to the unit cube (categoricals one-hot)."""
        parts: list[float] = []
        for name, d in self.dims.items():
            v = params[name]
            if isinstance(d, Categorical):
                parts.extend(1.0 if v == c else 0.0 for c in d.choices)
                continue
            lo, hi, x = (
                (math.log(d.low), math.log(d.high), math.log(v)) if d.log else (d.low, d.high, v)
            )
            parts.append(0.0 if hi == lo else (float(x) - lo) / (hi - lo))
        return np.asarray(parts, dtype=float)

    def sample(self, rng: np.random.Generator) -> Params:
        """Uniform (log-uniform where ``log``) random point."""
        out: Params = {}
        for name, d in self.dims.items():
            if isinstance(d, Categorical):
                out[name] = d.choices[int(rng.integers(len(d.choices)))]
                continue
            lo, hi = (math.log(d.low), math.log(d.high)) if d.log else (d.low, d.high)
            x = float(rng.uniform(lo, hi))
            v = math.exp(x) if d.log else x
            out[name] = int(min(max(round(v), d.low), d.high)) if isinstance(d, Int) else v
        return out


@dataclass
class SearchResult:
    """Outcome of one search."""

    method: str
    best_params: Params
    best_value: float
    values: list[float]
    params: list[Params]
    direction: str
    mlflow_run_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def best_so_far(self) -> list[float]:
        """Running best objective after each trial (the curve compared in row 10.5)."""
        better = max if self.direction == "maximize" else min
        out: list[float] = []
        for v in self.values:
            out.append(v if not out else better(out[-1], v))
        return out


def _expected_improvement(
    mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.01
) -> np.ndarray:
    from scipy.stats import norm

    sigma = np.maximum(sigma, 1e-12)
    z = (mu - best - xi) / sigma
    return np.asarray((mu - best - xi) * norm.cdf(z) + sigma * norm.pdf(z))


def _gp_propose(
    space: SearchSpace,
    history: Sequence[tuple[Params, float]],
    rng: np.random.Generator,
    maximize: bool,
    n_candidates: int = 512,
) -> Params:
    """Next point by expected improvement under a GP fitted to ``history``."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    x = np.stack([space.encode(p) for p, _ in history])
    y = np.asarray([v if maximize else -v for _, v in history], dtype=float)
    mean, std = float(y.mean()), float(y.std()) or 1.0
    yn = (y - mean) / std
    kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(1e-3)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, random_state=0)
    with warnings.catch_warnings():  # tiny histories often hit kernel-bound warnings
        warnings.simplefilter("ignore")
        gp.fit(x, yn)
    cands = [space.sample(rng) for _ in range(n_candidates)]
    xc = np.stack([space.encode(p) for p in cands])
    mu, sigma = gp.predict(xc, return_std=True)
    ei = _expected_improvement(np.asarray(mu), np.asarray(sigma), float(yn.max()))
    return cands[int(np.argmax(ei))]


def _log_child(mlf: Any, trial: Any, params: Params, value: float | None, parent_exp: Any) -> None:
    """One nested MLflow run per trial: params, objective, numeric user attrs, pruned flag."""
    with mlf.start_run(run_name=f"trial-{trial.number}", nested=True, experiment_id=parent_exp):
        mlf.log_params(params)
        if value is not None:
            mlf.log_metric("objective", value)
        else:
            mlf.set_tag("pruned", "true")
        for k, v in trial.user_attrs.items():
            if isinstance(v, bool):
                mlf.log_metric(k, float(v))
            elif isinstance(v, int | float) and math.isfinite(float(v)):
                mlf.log_metric(k, float(v))
            elif isinstance(v, list) and v and all(isinstance(x, list | tuple) for x in v):
                for step, x in v:  # [[step, value], ...] curves, e.g. a val-loss history
                    mlf.log_metric(k, float(x), step=int(step))


def run_search(
    objective: Callable[..., float],
    space: SearchSpace,
    method: Method,
    n_trials: int,
    *,
    seed: int = 0,
    direction: Literal["maximize", "minimize"] = "maximize",
    grid_points: int = 4,
    n_startup: int = 5,
    pruner: Any | None = None,
    mlflow: bool = False,
    mlflow_tags: Mapping[str, Any] | None = None,
    data_log: DataAccessLog | None = None,
    with_trial: bool = False,
    storage: str | None = None,
    study_name: str | None = None,
    enqueue: Sequence[Params] = (),
    catch: tuple[type[Exception], ...] = (),
) -> SearchResult:
    """Run ``n_trials`` of ``method`` on ``objective`` over ``space``.

    ``grid`` evaluates at most ``n_trials`` points of the grid (all of them if the grid is
    smaller). ``mlflow_tags`` are passed to :func:`terser.tracking.start_run` (needs ``phase``,
    ``row``, ``task``, ``run_name``). ``data_log`` is attached to the result as ``extra``
    ``data_access`` so callers can audit it.

    Pruning and long studies (row 10.7):

    * ``with_trial=True`` calls ``objective(params, trial)`` so the objective can
      ``trial.report`` intermediate values, raise ``optuna.TrialPruned`` and set user attrs
      (logged to the trial's MLflow child run).
    * ``storage`` (e.g. ``sqlite:///study.db``) with ``study_name`` persists the study and
      resumes it: only ``n_trials`` minus the already finished trials are run.
    * ``enqueue`` fixes the parameters of the first trials of a new study.
    * ``catch`` exception types mark a trial FAILED instead of stopping the study.

    Only COMPLETE trials enter ``values`` / ``params`` / ``best_*``: Optuna stores a pruned
    trial's last intermediate value as its value, which is not on the objective's scale. The
    study is returned in ``extra["study"]``. With no completed trial ``best_value`` is NaN.
    """
    import optuna

    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rng = np.random.default_rng(seed)
    sampler: Any
    if method == "grid":
        grid = space.grid(grid_points)
        sampler = optuna.samplers.GridSampler(grid, seed=seed)
        n_trials = min(n_trials, space.grid_size(grid_points))
    elif method == "tpe":
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup)
    else:
        sampler = optuna.samplers.RandomSampler(seed=seed)
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=storage is not None,
    )
    finished = [t for t in study.trials if t.state.is_finished()]
    if not study.trials:
        for p in enqueue:
            study.enqueue_trial(dict(p))
    n_trials = max(0, n_trials - len(finished))
    maximize = direction == "maximize"

    parent_id: str | None = None
    parent_exp: str | None = None
    ctx: Any = None
    if mlflow:
        from terser.tracking import start_run

        tags = dict(mlflow_tags or {})
        ctx = start_run(
            phase=int(tags.pop("phase", 10)),
            row=str(tags.pop("row", "10.1")),
            task=str(tags.pop("task", "hpo")),
            run_name=str(tags.pop("run_name", f"10-hpo-search-{method}-{seed}")),
            params={"method": method, "n_trials": n_trials, "seed": seed, **tags},
        )
        parent = ctx.__enter__()
        parent_id, parent_exp = parent.info.run_id, parent.info.experiment_id

    def wrapped(trial: Any) -> float:
        params = space.suggest(trial)
        try:
            value = float(objective(params, trial) if with_trial else objective(params))
        except optuna.TrialPruned:
            if mlflow:
                import mlflow as mlf

                _log_child(mlf, trial, params, None, parent_exp)
            raise
        if mlflow:
            import mlflow as mlf

            _log_child(mlf, trial, params, value, parent_exp)
        return value

    try:
        if method == "gp":
            history: list[tuple[Params, float]] = []
            for i in range(n_trials):
                if i >= n_startup:
                    study.enqueue_trial(_gp_propose(space, history, rng, maximize))
                study.optimize(wrapped, n_trials=1, catch=catch)
                t = study.trials[-1]
                if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE:
                    history.append((dict(t.params), float(t.value)))
        elif n_trials:
            study.optimize(wrapped, n_trials=n_trials, catch=catch)
        done = [
            t
            for t in study.trials
            if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE
        ]
        result = SearchResult(
            method=method,
            best_params=dict(study.best_params) if done else {},
            best_value=float(study.best_value) if done else float("nan"),
            values=[float(t.value) for t in done if t.value is not None],
            params=[dict(t.params) for t in done],
            direction=direction,
            mlflow_run_id=parent_id,
        )
        result.extra["study"] = study
        if data_log is not None:
            result.extra["data_access"] = data_log.records()
        if mlflow and done:
            import mlflow as mlf

            mlf.log_metric("best_value", result.best_value)
            for step, v in enumerate(result.best_so_far):
                mlf.log_metric("best_so_far", v, step=step)
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    return result
