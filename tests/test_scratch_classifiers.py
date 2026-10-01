"""Rows 4.1.1 (perceptron), 5.1.1 (AdaBoost), 5.2.3 (multinomial NB), 5.2.4 (Bayes optimal)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron as SkPerceptron
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from terser.models.classical.bayes_optimal import BayesOptimalClassifier
from terser.models.classical.scratch_naive_bayes import MultinomialNaiveBayes
from terser.models.ensembles.scratch_adaboost import AdaBoostStumps, fit_stump
from terser.models.neural.scratch_perceptron import Perceptron


def _separable(seed: int, n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 3))
    w = np.array([1.0, -2.0, 0.5])
    s = X @ w + 0.3
    keep = np.abs(s) > 0.3  # enforce a margin
    return X[keep], (s[keep] > 0).astype(int)


# ------------------------------------------------------------------ perceptron (4.1.1)


@pytest.mark.parametrize("seed", range(3))
def test_perceptron_converges_on_separable(seed: int) -> None:
    X, y = _separable(seed)
    p = Perceptron(max_epochs=1000).fit(X, y)
    assert p.converged_ and p.errors_[-1] == 0
    assert (p.predict(X) == y).all()


def test_perceptron_fails_on_xor() -> None:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 5, dtype=float)
    y = np.array([0, 1, 1, 0] * 5)
    p = Perceptron(max_epochs=50).fit(X, y)
    assert not p.converged_ and len(p.errors_) == 50 and min(p.errors_) > 0


@pytest.mark.parametrize("seed", range(3))
def test_perceptron_matches_sklearn(seed: int) -> None:
    X, y = _separable(seed)
    ours = Perceptron(max_epochs=20).fit(X, y)
    sk = SkPerceptron(
        eta0=1.0, alpha=0.0, shuffle=False, max_iter=20, tol=None, n_iter_no_change=1000
    )
    sk.fit(X, y)
    np.testing.assert_allclose(ours.coef_, sk.coef_[0])
    assert ours.intercept_ == pytest.approx(sk.intercept_[0])


def test_perceptron_rejects_bad_labels() -> None:
    with pytest.raises(ValueError):
        Perceptron().fit(np.zeros((2, 1)), np.array([-1, 1]))


# ------------------------------------------------------------------ AdaBoost (5.1.1)


def _noisy(seed: int, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 5))
    logit = 1.5 * X[:, 0] - X[:, 1] + 0.8 * X[:, 2] * X[:, 3]
    return X, (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)


def test_stump_matches_sklearn_depth1() -> None:
    X, y = _noisy(0)
    w = np.random.default_rng(1).random(len(y))
    stump, err = fit_stump(X, np.where(y == 1, 1.0, -1.0), w)
    sk = DecisionTreeClassifier(max_depth=1, criterion="gini").fit(X, y, sample_weight=w)
    ours_err = np.average(stump.predict(X) != np.where(y == 1, 1, -1), weights=w)
    sk_err = np.average(sk.predict(X) != y, weights=w)
    assert err == pytest.approx(ours_err)
    assert ours_err <= sk_err + 1e-12  # exhaustive 0-1 search is never worse than gini


@pytest.mark.parametrize("seed", range(3))
def test_adaboost_matches_sklearn_auc(seed: int) -> None:
    X, y = _noisy(seed)
    Xte, yte = _noisy(seed + 50)
    ours = AdaBoostStumps(n_estimators=60).fit(X, y)
    sk = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1), n_estimators=60, random_state=0
    ).fit(X, y)
    auc_ours = roc_auc_score(yte, ours.decision_function(Xte))
    auc_sk = roc_auc_score(yte, sk.decision_function(Xte))
    assert abs(auc_ours - auc_sk) < 0.01


def test_adaboost_training_error_and_weights() -> None:
    X, y = _noisy(4)
    m = AdaBoostStumps(n_estimators=40).fit(X, y)
    assert len(m.train_error_) == len(m.alphas_) == 40
    assert m.train_error_[-1] <= m.train_error_[0]
    assert m.sample_weight_.sum() == pytest.approx(1.0)
    hard = m.hard_examples(5)
    assert (m.sample_weight_[hard] >= np.sort(m.sample_weight_)[-5] - 1e-15).all()
    proba = m.predict_proba(X)
    assert ((proba > 0.5).astype(int) == m.predict(X)).all()


def test_adaboost_stops_on_perfect_stump() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    m = AdaBoostStumps(n_estimators=10).fit(X, np.array([0, 0, 1, 1]))
    assert len(m.stumps_) == 1 and (m.predict(X) == [0, 0, 1, 1]).all()


# ------------------------------------------------------------------ naive Bayes (5.2.3)

_DOCS = [
    "um so like can you help",
    "basically just write the code",
    "write a function that sorts a list",
    "please kindly maybe help me",
    "return the sorted list of numbers",
    "honestly like um whatever",
    "compute the mean of the column",
    "so um yeah thanks",
]
_LABELS = [1, 1, 0, 1, 0, 1, 0, 1]


@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.01])
def test_nb_matches_sklearn(alpha: float) -> None:
    X = CountVectorizer(ngram_range=(1, 3)).fit_transform(_DOCS)
    ours = MultinomialNaiveBayes(alpha=alpha).fit(X, _LABELS)
    sk = MultinomialNB(alpha=alpha).fit(X, _LABELS)
    np.testing.assert_allclose(ours.feature_log_prob_, sk.feature_log_prob_, atol=1e-12)
    np.testing.assert_allclose(ours.class_log_prior_, sk.class_log_prior_, atol=1e-12)
    np.testing.assert_allclose(ours.predict_proba(X), sk.predict_proba(X), atol=1e-10)
    np.testing.assert_array_equal(ours.predict(X), sk.predict(X))


def test_nb_laplace_avoids_zero_posterior() -> None:
    X = np.array([[2, 0], [0, 2]])
    y = [0, 1]
    mle = MultinomialNaiveBayes(alpha=0.0).fit(X, y)
    lap = MultinomialNaiveBayes(alpha=1.0).fit(X, y)
    probe = np.array([[1, 1]])  # contains a word unseen in each class
    assert np.isclose(np.exp(mle.feature_log_prob_[0, 1]), 0.0)
    assert (lap.predict_proba(probe) > 0).all()
    # Laplace = (count + 1) / (total + |V|): class 0 is (3/4, 1/4).
    np.testing.assert_allclose(np.exp(lap.feature_log_prob_[0]), [0.75, 0.25])
    with pytest.raises(ValueError):
        MultinomialNaiveBayes().fit(-X, y)


# ------------------------------------------------------------------ Bayes optimal (5.2.4)


def test_bayes_optimal_textbook_example() -> None:
    # Mitchell 6.7: posteriors .4/.3/.3; h1 says +, h2 and h3 say -  ->  MAP +, Bayes optimal -.
    preds_val = np.array([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1]])
    y_val = np.array([1, 1, 1, 1])
    m = BayesOptimalClassifier(noise=0.25, prior=[0.4, 0.3, 0.3]).fit(preds_val, y_val)
    post = m.posterior_
    assert post.sum() == pytest.approx(1.0) and m.map_index_ == 0
    # With one mistake each, h2/h3 lose a factor of noise/(1-noise) = 1/3.
    assert post[1] / post[0] == pytest.approx(0.3 / 0.4 / 3)
    query = np.array([[1], [0], [0]])
    assert m.predict_map(query)[0] == 1
    m2 = BayesOptimalClassifier(noise=0.25, prior=[0.4, 0.3, 0.3]).fit(preds_val[:, :0], y_val[:0])
    assert m2.predict(query)[0] == 0 and m2.predict_map(query)[0] == 1


def test_bayes_optimal_validation() -> None:
    with pytest.raises(ValueError):
        BayesOptimalClassifier(noise=0.6).fit(np.zeros((2, 3)), np.zeros(3))
    with pytest.raises(ValueError):
        BayesOptimalClassifier().fit(np.zeros((2, 3)), np.zeros(4))
    with pytest.raises(ValueError):
        BayesOptimalClassifier(prior=[1.0]).fit(np.zeros((2, 3)), np.zeros(3))
