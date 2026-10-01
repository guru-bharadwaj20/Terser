"""Decision trees from scratch: ID3, C4.5 thresholds, and four pruning methods.

Plan rows **3.3.1** (ID3), **3.3.2** (continuous values), **3.3.4** (overfitting solutions) and
**3.3.7** (rule export). Derivation: ``docs/derivations/decision_trees.md``.

* **ID3** (``splitter="id3"``): features are small non-negative integer codes (e.g. from
  :class:`terser.features.continuous.EqualFrequencyBinner`). Each node does a multiway split on the
  attribute with the largest information gain ``IG(S, A) = H(S) - sum_v |S_v|/|S| H(S_v)``, with
  ``H`` the Shannon entropy in bits; an attribute is used at most once on any root-to-leaf path.
* **C4.5 thresholds** (``splitter="c45"``): features are continuous. Each node does a binary split
  ``x_j <= t`` with ``t`` a midpoint between adjacent sorted distinct values whose classes differ
  (:func:`candidate_thresholds`). Fayyad and Irani (1992) show the entropy-optimal cut always lies
  on such a boundary point, so this finds the same best split as trying every midpoint (which is
  what scikit-learn does) while evaluating fewer cuts. Features may be reused deeper in the tree.
  The split criterion is information gain (not C4.5's gain ratio) so the tree is comparable with
  scikit-learn ``criterion="entropy"``.

Pruning (row 3.3.4):

* **pre-pruning**: ``max_depth``, ``min_samples_split``, ``min_samples_leaf``, ``min_gain``;
* **reduced-error pruning** (:meth:`DecisionTree.reduced_error_prune`): bottom-up, replace a
  subtree by a leaf whenever that does not increase errors on a held-out validation set
  (Mitchell 1997, section 3.7.1.1);
* **rule post-pruning** (:func:`rule_post_prune`): one rule per leaf, greedily drop preconditions
  while validation accuracy does not fall, then sort rules by accuracy (Mitchell 3.7.1.2);
* **minimal cost-complexity pruning** (:meth:`DecisionTree.cost_complexity_path`): Breiman's
  weakest-link sequence ``R_alpha(T) = R(T) + alpha |leaves(T)|``, with ``R`` either training
  misclassification rate or weighted entropy (scikit-learn's choice).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

__all__ = [
    "Condition",
    "DecisionTree",
    "Node",
    "Rule",
    "RuleSet",
    "candidate_thresholds",
    "entropy",
    "export_rules",
    "information_gain",
    "rule_post_prune",
    "tree_to_rules",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
Splitter = Literal["id3", "c45"]
CcpMeasure = Literal["error", "impurity"]


def entropy(counts: npt.ArrayLike) -> float:
    """Shannon entropy in bits of a vector of class counts."""
    c = np.asarray(counts, dtype=np.float64)
    n = c.sum()
    if n <= 0:
        return 0.0
    p = c[c > 0] / n
    return float(-(p * np.log2(p)).sum())


def _counts(y: IntArray, n_classes: int) -> FloatArray:
    return np.bincount(y, minlength=n_classes).astype(np.float64)


def information_gain(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Gain of a multiway split of labels ``y`` on the categorical attribute ``x``."""
    xa, ya = np.asarray(x), np.asarray(y)
    _, yi = np.unique(ya, return_inverse=True)
    k = int(yi.max()) + 1 if len(yi) else 1
    total = entropy(np.bincount(yi, minlength=k))
    rem = 0.0
    for v in np.unique(xa):
        m = xa == v
        rem += m.mean() * entropy(np.bincount(yi[m], minlength=k))
    return total - rem


def candidate_thresholds(x: npt.ArrayLike, y: npt.ArrayLike) -> FloatArray:
    """Midpoints between adjacent distinct values of ``x`` at class boundaries.

    A boundary lies between consecutive distinct values ``a < b`` unless every example with value
    ``a`` and every example with value ``b`` share one single class.
    """
    xa = np.asarray(x, dtype=np.float64)
    order = np.argsort(xa, kind="stable")
    xs = xa[order]
    _, yc = np.unique(np.asarray(y)[order], return_inverse=True)
    values, start = np.unique(xs, return_index=True)
    if len(values) < 2:
        return np.empty(0)
    lo = np.minimum.reduceat(yc, start)
    hi = np.maximum.reduceat(yc, start)
    pure = lo == hi
    same = pure[:-1] & pure[1:] & (lo[:-1] == lo[1:])
    mids = (values[:-1] + values[1:]) / 2
    return np.asarray(mids[~same], dtype=np.float64)


@dataclass
class Node:
    """A tree node. ``feature < 0`` marks a leaf.

    ``counts`` are the training class counts reaching the node. For ID3 nodes ``children`` maps an
    attribute value to a child; for C4.5 nodes key 0 is ``x <= threshold`` and key 1 is ``>``.
    """

    counts: FloatArray
    depth: int
    feature: int = -1
    threshold: float | None = None
    children: dict[int, Node] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return self.feature < 0

    @property
    def prediction(self) -> int:
        return int(np.argmax(self.counts))

    @property
    def n(self) -> float:
        return float(self.counts.sum())

    def make_leaf(self) -> None:
        self.feature = -1
        self.threshold = None
        self.children = {}

    def leaves(self) -> list[Node]:
        if self.is_leaf:
            return [self]
        return [leaf for c in self.children.values() for leaf in c.leaves()]

    def internal_nodes(self) -> list[Node]:
        if self.is_leaf:
            return []
        return [self, *(n for c in self.children.values() for n in c.internal_nodes())]

    def route(self, x: npt.NDArray[np.float64]) -> Node | None:
        """Child for example ``x``; ``None`` for an ID3 value unseen in training."""
        if self.threshold is not None:
            return self.children[0 if x[self.feature] <= self.threshold else 1]
        return self.children.get(int(x[self.feature]))


@dataclass
class DecisionTree:
    """Entropy-based decision tree classifier (ID3 multiway or C4.5-threshold binary splits).

    Attributes:
        splitter: ``"id3"`` for integer-coded categorical features, ``"c45"`` for continuous ones.
        max_depth: Maximum depth (root has depth 0); ``None`` for unlimited.
        min_samples_split: A node with fewer examples becomes a leaf.
        min_samples_leaf: A split that leaves any child with fewer examples is rejected.
        min_gain: A split must gain strictly more than this many bits.
    """

    splitter: Splitter = "id3"
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_gain: float = 0.0
    classes_: npt.NDArray[np.generic] = field(init=False, repr=False)
    root_: Node = field(init=False, repr=False)
    n_train_: int = field(init=False, default=0)

    # ------------------------------------------------------------------ fit
    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> DecisionTree:
        Xa = np.asarray(X, dtype=np.float64)
        if Xa.ndim != 2:
            raise ValueError("X must be 2-D")
        self.classes_, yi = np.unique(np.asarray(y), return_inverse=True)
        yi = yi.astype(np.int64)
        if len(yi) != len(Xa):
            raise ValueError("X and y lengths differ")
        if self.splitter == "id3" and (np.any(Xa < 0) or np.any(Xa != np.round(Xa))):
            raise ValueError("ID3 needs non-negative integer-coded features; bin them first")
        self.n_train_ = len(yi)
        self.root_ = self._grow(Xa, yi, depth=0, used=frozenset())
        return self

    def _grow(self, X: FloatArray, y: IntArray, depth: int, used: frozenset[int]) -> Node:
        k = len(self.classes_)
        node = Node(counts=_counts(y, k), depth=depth)
        if (
            len(y) < self.min_samples_split
            or np.count_nonzero(node.counts) <= 1
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return node
        parent_h = entropy(node.counts)
        best_gain, best_feat, best_thr = self.min_gain, -1, None
        for j in range(X.shape[1]):
            if self.splitter == "id3":
                if j in used:
                    continue
                gain = self._id3_gain(X[:, j], y, parent_h)
                if gain > best_gain + 1e-12:
                    best_gain, best_feat = gain, j
            else:
                gain, thr = self._best_threshold(X[:, j], y, parent_h)
                if thr is not None and gain > best_gain + 1e-12:
                    best_gain, best_feat, best_thr = gain, j, thr
        if best_feat < 0:
            return node
        node.feature = best_feat
        if self.splitter == "id3":
            col = X[:, best_feat]
            for v in np.unique(col):
                m = col == v
                node.children[int(v)] = self._grow(X[m], y[m], depth + 1, used | {best_feat})
        else:
            assert best_thr is not None
            node.threshold = best_thr
            m = X[:, best_feat] <= best_thr
            node.children[0] = self._grow(X[m], y[m], depth + 1, used)
            node.children[1] = self._grow(X[~m], y[~m], depth + 1, used)
        return node

    def _id3_gain(self, col: FloatArray, y: IntArray, parent_h: float) -> float:
        k = len(self.classes_)
        values = np.unique(col)
        if len(values) < 2:
            return -math.inf
        rem = 0.0
        for v in values:
            m = col == v
            if m.sum() < self.min_samples_leaf:
                return -math.inf
            rem += m.mean() * entropy(_counts(y[m], k))
        return parent_h - rem

    def _best_threshold(
        self, col: FloatArray, y: IntArray, parent_h: float
    ) -> tuple[float, float | None]:
        k = len(self.classes_)
        order = np.argsort(col, kind="stable")
        xs, ys = col[order], y[order]
        total = _counts(ys, k)
        left = np.cumsum(np.eye(k)[ys], axis=0)  # left[i] = counts of ys[: i + 1]
        n = len(ys)
        best_gain, best_thr = -math.inf, None
        for t in candidate_thresholds(xs, ys):
            i = int(np.searchsorted(xs, t, side="right"))  # examples with x <= t
            if i < self.min_samples_leaf or n - i < self.min_samples_leaf:
                continue
            lc = left[i - 1]
            rem = (i / n) * entropy(lc) + ((n - i) / n) * entropy(total - lc)
            gain = parent_h - rem
            if gain > best_gain + 1e-12:
                best_gain, best_thr = gain, float(t)
        return best_gain, best_thr

    # -------------------------------------------------------------- predict
    def _leaf_for(self, x: FloatArray) -> Node:
        node = self.root_
        while not node.is_leaf:
            nxt = node.route(x)
            if nxt is None:
                return node
            node = nxt
        return node

    def predict_proba(self, X: npt.ArrayLike) -> FloatArray:
        Xa = np.asarray(X, dtype=np.float64)
        out = np.empty((len(Xa), len(self.classes_)))
        for i, x in enumerate(Xa):
            c = self._leaf_for(x).counts
            out[i] = c / c.sum()
        return out

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.generic]:
        Xa = np.asarray(X, dtype=np.float64)
        idx = [self._leaf_for(x).prediction for x in Xa]
        return self.classes_[np.asarray(idx, dtype=np.int64)]

    # ------------------------------------------------------------ structure
    @property
    def n_leaves(self) -> int:
        return len(self.root_.leaves())

    @property
    def depth(self) -> int:
        return max(leaf.depth for leaf in self.root_.leaves()) - self.root_.depth

    def _encode(self, y: npt.ArrayLike) -> IntArray:
        ya = np.asarray(y)
        idx = np.searchsorted(self.classes_, ya)
        idx = np.clip(idx, 0, len(self.classes_) - 1)
        if not np.all(self.classes_[idx] == ya):
            raise ValueError("labels contain classes unseen in training")
        return idx.astype(np.int64)

    # -------------------------------------------------------------- pruning
    def reduced_error_prune(self, X_val: npt.ArrayLike, y_val: npt.ArrayLike) -> DecisionTree:
        """Copy of the tree after reduced-error pruning on a validation set.

        Post-order: each internal node becomes a leaf (predicting its training majority) when that
        makes no more validation errors than its already-pruned subtree. Nodes no validation
        example reaches are pruned too (0 errors either way), as in Mitchell's formulation.
        """
        out = copy.deepcopy(self)
        Xv = np.asarray(X_val, dtype=np.float64)
        yv = self._encode(y_val)

        def prune(node: Node, idx: IntArray) -> int:
            leaf_err = int(np.sum(yv[idx] != node.prediction))
            if node.is_leaf:
                return leaf_err
            groups: dict[int, list[int]] = {}
            stuck = 0
            for i in idx:
                nxt = node.route(Xv[i])
                if nxt is None:
                    stuck += int(yv[i] != node.prediction)
                    continue
                key = next(k for k, c in node.children.items() if c is nxt)
                groups.setdefault(key, []).append(int(i))
            sub_err = stuck + sum(
                prune(c, np.asarray(groups.get(k, []), dtype=np.int64))
                for k, c in node.children.items()
            )
            if leaf_err <= sub_err:
                node.make_leaf()
                return leaf_err
            return sub_err

        prune(out.root_, np.arange(len(yv), dtype=np.int64))
        return out

    def _risk(self, node: Node, measure: CcpMeasure) -> float:
        if measure == "error":
            return (node.n - node.counts.max()) / self.n_train_
        return node.n / self.n_train_ * entropy(node.counts)

    def cost_complexity_path(
        self, measure: CcpMeasure = "impurity"
    ) -> tuple[list[float], list[DecisionTree]]:
        """Weakest-link pruning sequence: effective alphas and the matching pruned trees.

        ``alphas[0] = 0`` is the full tree; each next tree collapses every internal node ``t`` with
        the smallest ``g(t) = (R(t) - R(T_t)) / (|leaves(T_t)| - 1)``; the last tree is the root.
        """
        cur = copy.deepcopy(self)
        alphas, trees = [0.0], [copy.deepcopy(cur)]
        while not cur.root_.is_leaf:
            g: list[tuple[float, Node]] = []
            for t in cur.root_.internal_nodes():
                leaves = t.leaves()
                r_sub = sum(cur._risk(leaf, measure) for leaf in leaves)
                g.append(((cur._risk(t, measure) - r_sub) / (len(leaves) - 1), t))
            alpha = min(v for v, _ in g)
            for v, t in g:
                if v <= alpha + 1e-12:
                    t.make_leaf()
            alphas.append(max(alpha, 0.0))
            trees.append(copy.deepcopy(cur))
        return alphas, trees

    def prune_cost_complexity(self, alpha: float, measure: CcpMeasure = "impurity") -> DecisionTree:
        """Smallest tree of the weakest-link sequence whose effective alpha is ``<= alpha``."""
        alphas, trees = self.cost_complexity_path(measure)
        best = trees[0]
        for a, t in zip(alphas, trees, strict=True):
            if a <= alpha + 1e-12:
                best = t
        return best


# ------------------------------------------------------------------ rules
@dataclass(frozen=True)
class Condition:
    """``x[feature] op value`` with ``op`` in ``==``, ``<=``, ``>``."""

    feature: int
    op: Literal["==", "<=", ">"]
    value: float

    def holds(self, X: FloatArray) -> npt.NDArray[np.bool_]:
        col = X[:, self.feature]
        if self.op == "==":
            return col == self.value
        if self.op == "<=":
            return col <= self.value
        return col > self.value

    def text(self, names: Sequence[str] | None = None) -> str:
        name = names[self.feature] if names is not None else f"x[{self.feature}]"
        value = int(self.value) if self.op == "==" else round(self.value, 4)
        return f"{name} {self.op} {value}"


@dataclass(frozen=True)
class Rule:
    """``IF all conditions THEN class index prediction``, with an accuracy estimate."""

    conditions: tuple[Condition, ...]
    prediction: int
    accuracy: float = 0.0
    coverage: int = 0

    def covers(self, X: FloatArray) -> npt.NDArray[np.bool_]:
        m = np.ones(len(X), dtype=bool)
        for c in self.conditions:
            m &= c.holds(X)
        return m


def tree_to_rules(tree: DecisionTree) -> list[Rule]:
    """One rule per leaf: the path's tests as a conjunction, the leaf majority as conclusion."""
    rules: list[Rule] = []

    def walk(node: Node, conds: tuple[Condition, ...]) -> None:
        if node.is_leaf:
            acc = float(node.counts.max() / node.n) if node.n else 0.0
            rules.append(Rule(conds, node.prediction, acc, int(node.n)))
            return
        for key, child in node.children.items():
            if node.threshold is not None:
                op: Literal["==", "<=", ">"] = "<=" if key == 0 else ">"
                cond = Condition(node.feature, op, node.threshold)
            else:
                cond = Condition(node.feature, "==", float(key))
            walk(child, (*conds, cond))

    walk(tree.root_, ())
    return rules


def _laplace(rule: Rule, X: FloatArray, y: IntArray) -> tuple[float, int]:
    m = rule.covers(X)
    n = int(m.sum())
    correct = int(np.sum(y[m] == rule.prediction))
    return (correct + 1) / (n + 2), n


@dataclass
class RuleSet:
    """Ordered rules: the first rule whose conditions hold decides; else ``default``."""

    rules: list[Rule]
    default: int
    classes_: npt.NDArray[np.generic]

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.generic]:
        Xa = np.asarray(X, dtype=np.float64)
        out = np.full(len(Xa), self.default, dtype=np.int64)
        done = np.zeros(len(Xa), dtype=bool)
        for r in self.rules:
            m = r.covers(Xa) & ~done
            out[m] = r.prediction
            done |= m
        return self.classes_[out]


def rule_post_prune(tree: DecisionTree, X_val: npt.ArrayLike, y_val: npt.ArrayLike) -> RuleSet:
    """Rule post-pruning on a validation set.

    Each rule's accuracy is the Laplace estimate ``(correct + 1) / (covered + 2)`` on validation
    examples it covers. A precondition is removed whenever that does not lower the estimate
    (best removal first, repeated until none qualifies). Rules are then sorted by estimate,
    highest first; the default class is the training majority.
    """
    Xv = np.asarray(X_val, dtype=np.float64)
    yv = tree._encode(y_val)
    pruned: list[Rule] = []
    for rule in tree_to_rules(tree):
        cur = rule
        cur_acc, cur_n = _laplace(cur, Xv, yv)
        while cur.conditions:
            trials = []
            for i in range(len(cur.conditions)):
                cand = Rule(cur.conditions[:i] + cur.conditions[i + 1 :], cur.prediction)
                acc, n = _laplace(cand, Xv, yv)
                trials.append((acc, n, cand))
            acc, n, cand = max(trials, key=lambda t: t[0])
            if acc < cur_acc:
                break
            cur, cur_acc, cur_n = cand, acc, n
        pruned.append(Rule(cur.conditions, cur.prediction, cur_acc, cur_n))
    pruned.sort(key=lambda r: -r.accuracy)
    return RuleSet(pruned, tree.root_.prediction, tree.classes_)


def export_rules(
    rules: Sequence[Rule],
    feature_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
) -> list[str]:
    """Human-readable ``IF ... THEN ...`` lines, e.g. for ``reports/p3_tree_rules.md``."""
    lines = []
    for r in rules:
        cond = " AND ".join(c.text(feature_names) for c in r.conditions) or "TRUE"
        cls = class_names[r.prediction] if class_names is not None else str(r.prediction)
        lines.append(f"IF {cond} THEN {cls}  (acc={r.accuracy:.3f}, n={r.coverage})")
    return lines
