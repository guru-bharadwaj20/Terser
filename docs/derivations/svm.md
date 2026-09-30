# Support vector machines: margin, primal, dual, kernels

Rows **4.4.1–4.4.3** of `CONTRIBUTING.md`. It derives the objects that the from-scratch dual solver
(row 4.4.4) optimises. The running example is recomputed by `tests/test_svm_doc.py`.

Notation: training pairs $(x_i, y_i)$, $i = 1..n$, with $x_i \in \mathbb{R}^d$ and
$y_i \in \{-1, +1\}$ (`degraded` $= +1$). The classifier is $\hat y = \operatorname{sign}(w^\top x + b)$.

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

## 3. Lagrangian, KKT conditions and the dual

We take the soft-margin primal and introduce multipliers $\alpha_i \ge 0$ for the margin
constraints and $\mu_i \ge 0$ for $\xi_i \ge 0$:

$$
\mathcal{L}(w, b, \xi, \alpha, \mu) = \tfrac12 \lVert w \rVert^2 + C \sum_i \xi_i
- \sum_i \alpha_i \big[ y_i (w^\top x_i + b) - 1 + \xi_i \big] - \sum_i \mu_i \xi_i .
$$

**Stationarity.** Setting the partial derivatives of the primal variables to zero gives

$$
\frac{\partial \mathcal{L}}{\partial w} = 0 \Rightarrow w = \sum_i \alpha_i y_i x_i, \qquad
\frac{\partial \mathcal{L}}{\partial b} = 0 \Rightarrow \sum_i \alpha_i y_i = 0, \qquad
\frac{\partial \mathcal{L}}{\partial \xi_i} = 0 \Rightarrow \alpha_i + \mu_i = C .
$$

**KKT conditions** (they are necessary and sufficient here: the problem is convex and Slater's
condition holds):

| Condition | Statement |
| :--- | :--- |
| Primal feasibility | $y_i (w^\top x_i + b) \ge 1 - \xi_i$ and $\xi_i \ge 0$ |
| Dual feasibility | $\alpha_i \ge 0$ and $\mu_i \ge 0$, hence $0 \le \alpha_i \le C$ |
| Complementary slackness | $\alpha_i [y_i f(x_i) - 1 + \xi_i] = 0$ and $\mu_i \xi_i = 0$ |
| Stationarity | the three equations above |

Complementary slackness sorts the training points into three groups:

- $\alpha_i = 0$: the point is outside the margin and does not affect $w$.
- $0 < \alpha_i < C$: then $\mu_i > 0$, so $\xi_i = 0$ and $y_i f(x_i) = 1$. The point is a
  **free support vector** exactly on the margin.
- $\alpha_i = C$: the point is inside the margin or misclassified ($\xi_i \ge 0$). It is a
  **bound support vector**.

**The dual.** Substituting stationarity back into $\mathcal{L}$ makes the $b$ and $\xi$ terms
vanish and leaves

$$
\max_{\alpha} \; \sum_i \alpha_i - \tfrac12 \sum_{i,j} \alpha_i \alpha_j y_i y_j \, x_i^\top x_j
\quad \text{s.t.} \quad 0 \le \alpha_i \le C, \;\; \sum_i \alpha_i y_i = 0 .
$$

This is a QP in $n$ variables with a box constraint and one equality constraint. It is the form
`cvxopt.solvers.qp` solves in row 4.4.4, as the minimisation of
$\tfrac12 \alpha^\top P \alpha - \mathbf 1^\top \alpha$ with $P_{ij} = y_i y_j x_i^\top x_j$.
The data enter **only through inner products**, which is what makes kernels possible
(section 5). Hard margin is the special case $C = \infty$, where only $\alpha_i \ge 0$ remains.

## 4. Recovering $w$ and $b$

- $w = \sum_i \alpha_i y_i x_i$. Only support vectors ($\alpha_i > 0$) contribute.
- For any free support vector $s$, $y_s (w^\top x_s + b) = 1$, so $b = y_s - w^\top x_s$. For
  numerical stability, average this over all free support vectors.

*Example.* The constraint $\sum \alpha_i y_i = 0$ forces $\alpha_1 = \alpha_2 = a$. With
$x_1^\top x_1 = x_2^\top x_2 = 2$, $x_1^\top x_2 = -2$ and $y_1 y_2 = -1$, the dual objective is
$2a - \tfrac12 (2a^2 + 2a^2 + 2 \cdot 2a^2) = 2a - 4a^2$. It is maximised at
$a = \alpha_1 = \alpha_2 = 0.25$, with value $0.25$. Then
$w = 0.25 (1, 1) - 0.25 (-1, -1) = (0.5, 0.5)$, and $b = y_1 - w^\top x_1 = 0$. This matches
section 1. The dual optimum $0.25$ equals the primal optimum $\tfrac12 \lVert w \rVert^2 = 0.25$
(strong duality).

## 5. The linear kernel in the dual

A **kernel** is a function $K(x, z) = \phi(x)^\top \phi(z)$ for some feature map $\phi$. The
**linear kernel** uses the identity map $\phi(x) = x$:

$$
K(x, z) = x^\top z .
$$

Substituting it into the dual of section 3 gives

$$
\max_{\alpha} \; \sum_i \alpha_i - \tfrac12 \sum_{i,j} \alpha_i \alpha_j y_i y_j \, K(x_i, x_j)
\quad \text{s.t.} \quad 0 \le \alpha_i \le C, \;\; \sum_i \alpha_i y_i = 0 .
$$

The QP matrix becomes $P = (y y^\top) \odot G$, where $G_{ij} = K(x_i, x_j)$ is the **Gram matrix**.
The decision function needs only kernel evaluations against the support vectors:

$$
f(x) = w^\top x + b = \Big(\sum_i \alpha_i y_i x_i\Big)^\top x + b
= \sum_{i \in \text{SV}} \alpha_i y_i \, K(x_i, x) + b ,
$$

and $b = y_s - \sum_i \alpha_i y_i K(x_i, x_s)$ for a free support vector $s$.

Two properties follow:

- **Validity.** $G = X X^\top$ is positive semidefinite, since $v^\top G v = \lVert X^\top v \rVert^2 \ge 0$.
  So $P$ is PSD, the dual is concave, and every local optimum is global.
- **Why derive it at all.** With the linear kernel, the primal ($d + 1$ variables) and the dual ($n$
  variables) give the same classifier, and $w$ can be formed explicitly. The kernel form matters
  because any other PSD kernel drops into the *same* QP unchanged.

  The polynomial kernel of row 4.4.7, $K(x, z) = (\gamma x^\top z + r)^p$, corresponds to all
  monomials of degree $\le p$. For the 1536-d pair embedding with $p = 3$, that is about
  $6 \times 10^8$ features, which are never materialised. The cost moves from $d$ to $n^2$ kernel
  evaluations.

*Example.* $G = \begin{pmatrix} 2 & -2 \\ -2 & 2 \end{pmatrix}$ and
$f(x) = 0.25 \, K(x_1, x) - 0.25 \, K(x_2, x) = 0.5\, x^{(1)} + 0.5\, x^{(2)}$, the same
classifier as section 4.
