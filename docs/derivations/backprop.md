# Backpropagation for the quality-gate MLP

Plan row **4.1.7**. It derives the gradients that `terser.models.neural.numpy_mlp`
(row 4.1.3) implements without autograd. The worked example in section 4 is recomputed by
`tests/test_backprop_doc.py`, so this document and the code cannot drift apart.

## 1. Setup

The network has layers $l = 1, \dots, L$ with weights $W_l \in \mathbb{R}^{n_{l-1} \times n_l}$ and
biases $b_l \in \mathbb{R}^{n_l}$. For a mini-batch $X \in \mathbb{R}^{n \times d}$ (one example
per row):

$$
a_0 = X, \qquad z_l = a_{l-1} W_l + b_l, \qquad
a_l = \begin{cases} f(z_l) & l < L \\ \sigma(z_L) & l = L \end{cases}
$$

Here $f$ is the hidden activation (ReLU, tanh or sigmoid) and $\sigma(z) = 1 / (1 + e^{-z})$. The
output $p = a_L \in (0, 1)^{n}$ is $P(\text{degraded})$. The loss is the mean binary
cross-entropy:

$$
\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \big[ y_i \log p_i + (1 - y_i) \log (1 - p_i) \big]
$$

We define the **error** of layer $l$ as $\delta_l = \partial \mathcal{L} / \partial z_l$, which has
the same shape as $z_l$.

## 2. Output layer: sigmoid + BCE

For one example, $\partial \mathcal{L} / \partial p = \frac{1}{n}\left(-\frac{y}{p} + \frac{1-y}{1-p}\right)
= \frac{1}{n} \cdot \frac{p - y}{p(1-p)}$. The sigmoid's derivative is $\sigma'(z) = \sigma(z)(1 - \sigma(z)) = p(1-p)$.
By the chain rule the two $p(1-p)$ factors cancel:

$$
\boxed{\delta_L = \frac{1}{n}(p - y)}
$$

This cancellation is why the implementation computes the loss from logits
($\log(1 + e^{z}) - y z$). That form is numerically stable and gives the same gradient.

## 3. Hidden layers: the recursion

$z_{l+1} = a_l W_{l+1} + b_{l+1}$ and $a_l = f(z_l)$ element-wise, so by the chain rule:

$$
\boxed{\delta_l = \big(\delta_{l+1} W_{l+1}^{\top}\big) \odot f'(z_l)}
$$

Here $\odot$ is the element-wise product. The activation derivatives are:

| $f$ | $f'(z)$ |
| :--- | :--- |
| ReLU | $\mathbb{1}[z > 0]$ (0 at the kink by convention) |
| tanh | $1 - \tanh^2 z$ |
| sigmoid | $\sigma(z)(1 - \sigma(z))$ |

Given $\delta_l$, the parameter gradients follow from $z_l = a_{l-1} W_l + b_l$:

$$
\boxed{\frac{\partial \mathcal{L}}{\partial W_l} = a_{l-1}^{\top} \delta_l, \qquad
\frac{\partial \mathcal{L}}{\partial b_l} = \mathbf{1}^{\top} \delta_l}
$$

The batch sum lives inside the matrix product. The $1/n$ factor entered once, in $\delta_L$.

**Algorithm.** Run a forward pass and cache every $z_l$ and $a_l$. Set $\delta_L = (p - y)/n$. For
$l = L, \dots, 1$, emit the gradients of $W_l$ and $b_l$, then propagate $\delta_{l-1}$. The backward
pass costs about as much as the forward pass: one extra matrix product per layer.

**Why sigmoid and tanh hidden units saturate (row 4.2.1).** $\sigma'(z) \le 1/4$, so every sigmoid
layer multiplies the error by at most $1/4$ times $\lVert W \rVert$. Deep sigmoid stacks therefore
shrink gradients geometrically. ReLU passes the error through unchanged where $z > 0$, but a unit
whose $z \le 0$ on every input never receives gradient again (a "dead" unit).

## 4. Worked example

This is a 2-2-1 ReLU network with one example: $x = (1, 2)$, $y = 1$.

$$
W_1 = \begin{pmatrix} 0.5 & -0.3 \\ 0.2 & 0.8 \end{pmatrix},\;
b_1 = (0.1, -0.1),\;
W_2 = \begin{pmatrix} 1.0 \\ -1.5 \end{pmatrix},\;
b_2 = 0.2
$$

| Quantity | Computation | Value |
| :--- | :--- | :--- |
| $z_1$ | $(0.5 + 0.4 + 0.1,\; -0.3 + 1.6 - 0.1)$ | $(1.0, 1.2)$ |
| $a_1$ | ReLU, both positive | $(1.0, 1.2)$ |
| $z_2$ | $1.0 - 1.8 + 0.2$ | $-0.6$ |
| $p$ | $\sigma(-0.6)$ | $0.354344$ |
| $\mathcal{L}$ | $-\log p$ | $1.037488$ |
| $\delta_2$ | $p - y$ | $-0.645656$ |
| $\partial \mathcal{L} / \partial W_2$ | $a_1^{\top} \delta_2$ | $(-0.645656, -0.774788)^{\top}$ |
| $\partial \mathcal{L} / \partial b_2$ | $\delta_2$ | $-0.645656$ |
| $\delta_1$ | $\delta_2 W_2^{\top} \odot (1, 1)$ | $(-0.645656, 0.968484)$ |
| $\partial \mathcal{L} / \partial W_1$ | $x^{\top} \delta_1$ | $\begin{pmatrix} -0.645656 & 0.968484 \\ -1.291313 & 1.936969 \end{pmatrix}$ |
| $\partial \mathcal{L} / \partial b_1$ | $\delta_1$ | $(-0.645656, 0.968484)$ |

The signs make sense. The example is degraded ($y = 1$) but $p < 0.5$, so gradient descent
($\theta \leftarrow \theta - \eta \nabla$) raises $b_2$ and both output weights, which makes $-1.5$
less negative. It also lowers the input weights and bias of the second hidden unit, whose activation
currently pushes $p$ down through that negative weight.

## 5. Verification in code

- **Finite differences.** `gradient_check` compares every analytic partial with
  $(\mathcal{L}(\theta + h) - \mathcal{L}(\theta - h)) / 2h$ at $h = 10^{-5}$. The maximum relative
  error is below $10^{-6}$ for ReLU, tanh and sigmoid networks.
- **Autograd.** The same weights in PyTorch give gradients equal to within $10^{-10}$.
