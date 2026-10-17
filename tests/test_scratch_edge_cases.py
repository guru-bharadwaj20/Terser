"""Edge cases of the from-scratch models found by the repository audit."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from terser.models.classical.hmm import TokenClassEncoder
from terser.models.classical.scratch_id3 import DecisionTree
from terser.models.classical.scratch_knn import KNNClassifier, pairwise_distances
from terser.models.clustering.scratch_kmeans import BisectingKMeans
from terser.models.ensembles.scratch_adaboost import AdaBoostStumps


def test_c45_min_samples_leaf_finds_non_boundary_cut() -> None:
    X = np.arange(10, dtype=float)[:, None]
    y = np.array([1] + [0] * 9)
    ours = DecisionTree(splitter="c45", min_samples_leaf=3).fit(X, y)
    sk = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=3, random_state=0).fit(X, y)
    assert ours.n_leaves == sk.get_n_leaves() == 2


def test_hmm_encoder_pos_ignores_trailing_punctuation() -> None:
    enc = TokenClassEncoder()
    assert enc.encode_word("and,") == enc.encode_word("and")
    assert enc.encode_word("the.") == enc.encode_word("the")


def test_euclidean_duplicates_are_at_distance_zero() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5)) * 100
    d = pairwise_distances(X, X, "euclidean")
    assert np.all(np.diag(d) == 0.0)
    y = (X[:, 0] > 0).astype(int)
    proba = KNNClassifier(k=5, weights="distance").fit(X, y).predict_proba(X)
    assert set(np.unique(proba)) <= {0.0, 1.0}


def test_bisecting_kmeans_with_identical_points_has_no_empty_cluster() -> None:
    X = np.zeros((6, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        km = BisectingKMeans(n_clusters=3).fit(X)
    assert np.isfinite(km.cluster_centers_).all()
    assert sorted(np.bincount(km.labels_, minlength=3).tolist())[0] >= 1


def test_adaboost_train_error_matches_predict() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] + 0.3 * rng.normal(size=60) > 0).astype(int)
    ab = AdaBoostStumps(n_estimators=5).fit(X, y)
    assert ab.train_error_[-1] == float(np.mean(ab.predict(X) != y))
