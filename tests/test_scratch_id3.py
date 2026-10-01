"""Rows 3.3.1, 3.3.2, 3.3.4, 3.3.7: from-scratch decision trees and pruning."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from terser.features.continuous import EqualFrequencyBinner
from terser.models.classical.scratch_id3 import (
    DecisionTree,
    candidate_thresholds,
    entropy,
    export_rules,
    information_gain,
    rule_post_prune,
    tree_to_rules,
)

# Mitchell (1997) Table 3.2 PlayTennis. Outlook: 0 Sunny, 1 Overcast, 2 Rain. Temperature:
# 0 Hot, 1 Mild, 2 Cool. Humidity: 0 High, 1 Normal. Wind: 0 Weak, 1 Strong.
PLAY_TENNIS = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1],
        [2, 1, 0, 0, 1],
        [2, 2, 1, 0, 1],
        [2, 2, 1, 1, 0],
        [1, 2, 1, 1, 1],
        [0, 1, 0, 0, 0],
        [0, 2, 1, 0, 1],
        [2, 1, 1, 0, 1],
        [0, 1, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 0, 1, 0, 1],
        [2, 1, 0, 1, 0],
    ]
)
X_PT, Y_PT = PLAY_TENNIS[:, :4], PLAY_TENNIS[:, 4]


def test_entropy_hand_values() -> None:
    assert entropy([9, 5]) == pytest.approx(0.9403, abs=1e-4)
    assert entropy([7, 7]) == pytest.approx(1.0)
    assert entropy([4, 0]) == 0.0
    assert entropy([]) == 0.0


def test_information_gain_play_tennis() -> None:
    # Hand-computed (Mitchell 3.4.1.2): Outlook .246, Humidity .151, Wind .048, Temperature .029.
    gains = [information_gain(X_PT[:, j], Y_PT) for j in range(4)]
    assert gains == pytest.approx([0.2467, 0.0292, 0.1518, 0.0481], abs=1e-4)


def test_id3_builds_the_textbook_tree() -> None:
    tree = DecisionTree(splitter="id3").fit(X_PT, Y_PT)
    root = tree.root_
    assert root.feature == 0  # Outlook
    assert root.children[1].is_leaf and root.children[1].prediction == 1  # Overcast -> Yes
    assert root.children[0].feature == 2  # Sunny -> Humidity
    assert root.children[2].feature == 3  # Rain -> Wind
    assert tree.n_leaves == 5 and tree.depth == 2
    assert (tree.predict(X_PT) == Y_PT).all()


def test_id3_unseen_value_falls_back_to_node_majority() -> None:
    tree = DecisionTree(splitter="id3").fit(X_PT, Y_PT)
    assert tree.predict([[7, 0, 0, 0]])[0] == 1  # root majority is Yes (9 vs 5)


def test_id3_rejects_continuous_features() -> None:
    with pytest.raises(ValueError):
        DecisionTree(splitter="id3").fit([[0.5], [1.5]], [0, 1])


def test_pre_pruning_limits_depth_and_leaf_size() -> None:
    stump = DecisionTree(splitter="id3", max_depth=1).fit(X_PT, Y_PT)
    assert stump.depth == 1 and stump.n_leaves == 3
    big_leaf = DecisionTree(splitter="id3", min_samples_leaf=5).fit(X_PT, Y_PT)
    assert big_leaf.root_.feature != 0 or big_leaf.root_.is_leaf  # Overcast branch has 4


def _data(seed: int) -> tuple[np.ndarray, ...]:
    X, y = make_classification(
        n_samples=1200,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        flip_y=0.05,
        random_state=seed,
    )
    return tuple(train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y))


def _id3_vs_sklearn_gap(seed: int, depth: int | None, n_bins: int) -> float:
    Xtr, Xte, ytr, yte = _data(seed)
    binner = EqualFrequencyBinner(n_bins=n_bins).fit(Xtr)
    Btr, Bte = binner.transform(Xtr), binner.transform(Xte)
    ours = DecisionTree(splitter="id3", max_depth=depth).fit(Btr, ytr)
    ref = DecisionTreeClassifier(criterion="entropy", max_depth=depth, random_state=0)
    ref.fit(Btr, ytr)
    return float(f1_score(yte, ref.predict(Bte)) - f1_score(yte, ours.predict(Bte)))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("depth", [2, 3, 4, 5, 6, None])
def test_id3_within_two_f1_of_sklearn_on_binary_attributes(seed: int, depth: int | None) -> None:
    # With two bins every attribute is binary, so a multiway ID3 split and sklearn's binary
    # split coincide and equal depth means equal capacity.
    assert abs(_id3_vs_sklearn_gap(seed, depth, n_bins=2)) <= 0.02


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_id3_within_two_f1_of_sklearn_on_four_bins_full_depth(seed: int) -> None:
    # With 4 bins and a depth cap, a multiway node is not comparable to a binary one (a depth-d
    # ID3 tree can have 4^d leaves, sklearn's 2^d), and gaps of up to ~8 F1 occur at depth 5-6
    # on these fixtures. Grown fully, both trees fit the same partitions.
    assert _id3_vs_sklearn_gap(seed, None, n_bins=4) <= 0.02


def test_candidate_thresholds_only_at_class_boundaries() -> None:
    x = [1.0, 2.0, 3.0, 4.0, 4.0, 5.0]
    y = [0, 0, 1, 1, 0, 0]
    # boundaries: 2|3 (0->1), 3|4 (value 4 is mixed), 4|5 (mixed); 1|2 is not.
    assert candidate_thresholds(x, y).tolist() == [2.5, 3.5, 4.5]
    assert candidate_thresholds([1.0, 1.0], [0, 1]).size == 0


def test_c45_one_dimensional_separable_fixture() -> None:
    x = np.array([[0.3], [1.2], [1.9], [2.2], [4.0], [5.5], [6.1]])
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    tree = DecisionTree(splitter="c45").fit(x, y)
    assert tree.root_.threshold == pytest.approx(3.1)  # midpoint of 2.2 and 4.0
    assert tree.n_leaves == 2
    assert tree.predict([[3.0], [3.2]]).tolist() == [0, 1]


@pytest.mark.parametrize("depth", [1, 3, 5, None])
def test_c45_matches_sklearn_entropy_tree(depth: int | None) -> None:
    Xtr, Xte, ytr, _ = _data(3)
    ours = DecisionTree(splitter="c45", max_depth=depth).fit(Xtr, ytr)
    ref = DecisionTreeClassifier(criterion="entropy", max_depth=depth, random_state=0)
    ref.fit(Xtr, ytr)
    agree = np.mean(ours.predict(Xte) == ref.predict(Xte))
    # Fully grown, both trees have 83 leaves and depth 11; deep equal-gain ties are broken
    # differently (sklearn permutes features), so a few test points differ.
    assert agree >= (0.95 if depth is None else 0.98)
    assert ours.n_leaves == ref.get_n_leaves()
    assert ours.root_.feature == ref.tree_.feature[0]
    assert ours.root_.threshold == pytest.approx(ref.tree_.threshold[0], abs=1e-6)


def test_cost_complexity_path_matches_sklearn() -> None:
    X, y = make_classification(
        n_samples=150, n_features=3, n_informative=3, n_redundant=0, flip_y=0.1, random_state=7
    )
    ours = DecisionTree(splitter="c45").fit(X, y)
    ref = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(X, y)
    assert ours.n_leaves == ref.get_n_leaves()
    alphas, trees = ours.cost_complexity_path("impurity")
    ref_alphas = ref.cost_complexity_pruning_path(X, y).ccp_alphas
    np.testing.assert_allclose(alphas, ref_alphas, atol=1e-9)
    assert trees[-1].root_.is_leaf
    mid = alphas[len(alphas) // 2]
    ref_mid = DecisionTreeClassifier(criterion="entropy", ccp_alpha=mid, random_state=0).fit(X, y)
    assert ours.prune_cost_complexity(mid).n_leaves == ref_mid.get_n_leaves()


def test_reduced_error_pruning_hand_example() -> None:
    # Root split on x0; the right subtree splits on x1 over noise. Validation says the
    # right subtree is worse than a leaf, the left split is useful.
    X = np.array([[0, 0], [0, 1], [0, 0], [1, 0], [1, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1, 0, 1, 1])
    tree = DecisionTree(splitter="id3").fit(X, y)
    assert tree.n_leaves == 3
    Xv = np.array([[0, 0], [1, 1], [1, 1], [1, 0]])
    yv = np.array([0, 1, 1, 1])
    pruned = tree.reduced_error_prune(Xv, yv)
    assert pruned.n_leaves == 2
    assert pruned.root_.feature == 0
    assert tree.n_leaves == 3  # original untouched
    assert (pruned.predict(Xv) == yv).all()


def test_reduced_error_pruning_never_hurts_validation_accuracy() -> None:
    Xtr, Xva, ytr, yva = _data(4)
    full = DecisionTree(splitter="c45").fit(Xtr, ytr)
    pruned = full.reduced_error_prune(Xva, yva)
    assert pruned.n_leaves < full.n_leaves
    assert np.mean(pruned.predict(Xva) == yva) >= np.mean(full.predict(Xva) == yva)


def test_rules_reproduce_tree_and_post_pruning() -> None:
    tree = DecisionTree(splitter="id3").fit(X_PT, Y_PT)
    rules = tree_to_rules(tree)
    assert len(rules) == tree.n_leaves
    lines = export_rules(rules, ["outlook", "temp", "humidity", "wind"], ["no", "yes"])
    assert "IF outlook == 1 THEN yes  (acc=1.000, n=4)" in lines
    ruleset = rule_post_prune(tree, X_PT, Y_PT)
    # The Laplace estimate favours general rules: "humidity == normal -> yes" (7 covered, 6
    # right) outranks "rain and strong wind -> no" (2 covered), so day 6 is misclassified.
    assert np.mean(ruleset.predict(X_PT) == Y_PT) == pytest.approx(13 / 14)
    assert (
        "IF humidity == 1 THEN yes"
        in export_rules(ruleset.rules, ["outlook", "temp", "humidity", "wind"], ["no", "yes"])[3]
    )
    # every pruned rule is no longer than its source rule
    assert sum(len(r.conditions) for r in ruleset.rules) <= sum(len(r.conditions) for r in rules)


def test_rule_post_pruning_drops_useless_precondition() -> None:
    # x1 is noise; the tree tests it under x0 == 1 but validation shows x0 alone decides.
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [0, 0]])
    y = np.array([0, 0, 1, 1, 0, 0])
    tree = DecisionTree(splitter="id3").fit(X, y)
    Xv = np.array([[1, 0], [1, 1], [0, 1], [1, 1]])
    yv = np.array([1, 1, 0, 1])
    rs = rule_post_prune(tree, Xv, yv)
    assert (rs.predict(Xv) == yv).all()
    assert min(len(r.conditions) for r in rs.rules) <= 1
