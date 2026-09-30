# Support vector machines: margin, primal, dual, kernels

Row **4.4.1** of `CONTRIBUTING.md`. It derives the objects that the from-scratch dual solver
(row 4.4.4) optimises. The running example is recomputed by `tests/test_svm_doc.py`.

Notation: training pairs $(x_i, y_i)$, $i = 1..n$, with $x_i \in \mathbb{R}^d$ and
$y_i \in \{-1, +1\}$ (``degraded`` $= +1$). The classifier is $\hat y = \operatorname{sign}(w^\top x + b)$.

**Running example.** $x_1 = (1, 1)$ with $y_1 = +1$, and $x_2 = (-1, -1)$ with $y_2 = -1$.

## 1. Margin $= 2 / \lVert w \rVert$

The distance from a point $x$ to the hyperplane $H = \{z : w^\top z + b = 0\}$ is
$|w^\top x + b| / \lVert w \rVert$. To see this, let $x_\perp$ be the projection of $x$ onto $H$.
Then $x = x_\perp + r\, w / \lVert w \rVert$, and applying $w^\top \cdot + b$ gives
$w^\top x + b = r \lVert w \rVert$.

The pair $(w, b)$ is defined only up to a positive scale, since $(cw, cb)$ describes the same
hyperplane. We fix the scale **canonically**: the points closest to $H$ satisfy
$y_i (w^\top x_i + b) = 1$. Those points lie at distance $1 / \lVert w \rVert$ on each side, so

$$
\text{margin} = \frac{2}{\lVert w \rVert}.
$$

Maximising the margin is therefore the same as minimising $\lVert w \rVert$, or equivalently the
smooth $\tfrac12 \lVert w \rVert^2$.

*Example.* By symmetry, $H$ is the perpendicular bisector $x^{(1)} + x^{(2)} = 0$, so
$w \propto (1, 1)$ and $b = 0$. The canonical scaling $w^\top x_1 = 1$ gives $w = (0.5, 0.5)$, so
$\lVert w \rVert = 1/\sqrt 2$ and the margin is $2\sqrt 2 \approx 2.828427$. This equals
$\lVert x_1 - x_2 \rVert$, as it must for two points.

## 2. Primal problems

**Hard margin** (the data must be linearly separable):

$$
\min_{w, b} \; \tfrac12 \lVert w \rVert^2
\quad \text{s.t.} \quad y_i (w^\top x_i + b) \ge 1 \;\; \forall i.
$$

This is a convex quadratic programme with linear constraints. If the data are separable, it has a
unique optimal $w$.

**Soft margin.** Casual-prompt data are not separable (row 4.4.6), so each constraint gets a slack
variable $\xi_i \ge 0$, and the total slack is priced by $C > 0$:

$$
\min_{w, b, \xi} \; \tfrac12 \lVert w \rVert^2 + C \sum_i \xi_i
\quad \text{s.t.} \quad y_i (w^\top x_i + b) \ge 1 - \xi_i, \;\; \xi_i \ge 0.
$$

At the optimum $\xi_i = \max(0,\, 1 - y_i f(x_i))$, so the soft-margin primal is exactly the
unconstrained **hinge-loss** problem
$\min \tfrac12 \lVert w \rVert^2 + C \sum_i \max(0, 1 - y_i f(x_i))$.

The two terms trade off against each other:

- **Large $C$** punishes violations. The margin narrows and the result approaches hard margin
  (low bias, higher variance).
- **Small $C$** tolerates violations. The margin widens and more points become support vectors
  (row 4.4.6 sweeps $C$).
