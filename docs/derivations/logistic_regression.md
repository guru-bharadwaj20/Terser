# Logistic regression: penalties, proximal gradient, calibration

Code: [`logistic.py`](../../src/terser/models/classical/logistic.py) (3.4.1, L2 batch GD),
[`scratch_penalized_lr.py`](../../src/terser/models/classical/scratch_penalized_lr.py) (3.4.2, 3.4.5),
[`eval/calibration.py`](../../src/terser/eval/calibration.py) (3.4.3).

## 1. Objective

With `p_i = sigmoid(w . x_i + b)` and mixing `r` in [0, 1]:

    J(w, b) = (1/n) sum_i -[y_i log p_i + (1 - y_i) log(1 - p_i)]
              + [ (1 - r)/2 ||w||_2^2 + r ||w||_1 ] / (C n)

Multiplying by `C n` gives scikit-learn's `C sum log-loss + (1 - r)/2 ||w||^2 + r ||w||_1`, so both
have the same minimiser. `r = 0` is ridge (L2), `r = 1` lasso (L1), in between elastic net.

## 2. Proximal gradient

Split `J = f + g` with `f` = log-loss + L2 term (smooth) and `g = lambda_1 ||w||_1`,
`lambda_1 = r/(C n)`. The gradient of `f` is

    grad_w f = X^T (p - y) / n + (1 - r) w / (C n),   grad_b f = mean(p - y).

The Hessian of the mean log-loss is `Z^T D Z / n` with `D = diag(p(1-p)) <= 1/4`, where `Z = [X, 1]`,
so `grad f` is Lipschitz with `L = sigma_max(Z)^2 / (4n) + (1 - r)/(C n)`. One ISTA step is

    theta <- prox_{g/L}(theta - grad f(theta) / L),   prox(z)_j = sign(z_j) max(|z_j| - lambda_1/L, 0)

applied to `w` only (the intercept is not penalised). The soft-threshold sets small coefficients to
exactly zero, which is why L1 selects features and L2 only shrinks them. FISTA adds Nesterov
momentum `t_{k+1} = (1 + sqrt(1 + 4 t_k^2))/2`, `m = theta_{k+1} + ((t_k - 1)/t_{k+1})(theta_{k+1} - theta_k)`,
improving the rate from `O(1/k)` to `O(1/k^2)`. Tested against scikit-learn `saga` to 1e-3.

## 3. Odds ratios

`log(p/(1-p)) = w . x + b`, so increasing `x_j` by one unit multiplies the odds by `exp(w_j)`. On
z-scored features (row 3.1.4) the unit is one training standard deviation, which makes odds ratios
comparable across features.

## 4. Calibration

A model is calibrated when `P(y = 1 | p_hat = q) = q`. The reliability diagram bins `p_hat` into
equal-width bins and plots the observed positive rate against the mean prediction in each bin;
`ECE = sum_b (n_b/n) |acc_b - conf_b|`. The Brier score `mean (p_hat - y)^2` is a proper scoring rule
that mixes calibration and refinement (sharpness).
