# Reinforcement learning for operator ordering (row 9.1.1)

Code: [`models/rl/env.py`](../../src/terser/models/rl/env.py),
[`models/rl/scratch_qlearning.py`](../../src/terser/models/rl/scratch_qlearning.py) · Tests:
`tests/test_rl.py`

## 1. Agent and environment

- **Agent**: the rewriter. Each step it picks an operator (`F H D I L C S`) or `STOP`.
- **Environment**: the current draft, the target tokenizer T (counts tokens) and the quality gate
  (estimates P(degraded)). It applies the operator set to the original prompt and returns the next
  state and a reward.
- **State** $s$: (remaining-filler bucket, kept-constraint bucket, length bucket, category,
  operators applied). It is a lossy summary of the draft, so the process is only approximately
  Markov; the buckets keep the Q-table small.
- **Actions** $a \in \{F,H,D,I,L,C,S,\text{STOP}\}$, each operator usable once.
- **Reward** (row 9.1.4): $-0.01$ per operator step; on STOP,
  $r = 100\,(1 - |T(\text{draft})| / |T(\text{original})|) - \lambda\,\mathbb 1[\text{degraded}]$,
  $\lambda = 100$. A degraded rewrite therefore always scores below keeping the original (0).
- **Episode**: ends at STOP, or after all seven operators.

## 2. Policy and value functions

A policy $\pi(a \mid s)$ is a distribution over actions per state. With return
$G_t = \sum_{k \ge 0} \gamma^k r_{t+k+1}$ and discount $\gamma \in [0, 1]$:

$$V^\pi(s) = \mathbb E_\pi[G_t \mid s_t = s], \qquad
Q^\pi(s, a) = \mathbb E_\pi[G_t \mid s_t = s, a_t = a],$$

and $V^\pi(s) = \sum_a \pi(a \mid s)\, Q^\pi(s, a)$.

## 3. Bellman expectation equations

Split the return into the first reward and the rest, $G_t = r_{t+1} + \gamma G_{t+1}$, and condition
on the next state:

$$Q^\pi(s,a) = \sum_{s', r} p(s', r \mid s, a)\,\big[r + \gamma V^\pi(s')\big]
= \mathbb E\big[r + \gamma \textstyle\sum_{a'} \pi(a' \mid s')\, Q^\pi(s', a')\big],$$

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\,\big[r + \gamma V^\pi(s')\big].$$

Terminal states have value 0.

## 4. Bellman optimality equations

The optimal values $V^*(s) = \max_\pi V^\pi(s)$ and $Q^*(s,a) = \max_\pi Q^\pi(s,a)$ satisfy
$V^*(s) = \max_a Q^*(s,a)$ (an optimal policy puts all its mass on a best action), hence

$$Q^*(s,a) = \mathbb E\big[r + \gamma \max_{a'} Q^*(s', a') \,\big|\, s, a\big].$$

A greedy policy $\pi^*(s) = \arg\max_a Q^*(s,a)$ is optimal.

## 5. Q-learning from the optimality equation

$Q^*$ is the fixed point of the operator
$(\mathcal T Q)(s,a) = \mathbb E[r + \gamma \max_{a'} Q(s',a')]$, a $\gamma$-contraction in the max
norm. We cannot compute the expectation (no model of $p(s', r \mid s,a)$), but each transition
$(s, a, r, s')$ gives an unbiased sample $y = r + \gamma \max_{a'} Q(s', a')$ of $(\mathcal T Q)(s,a)$
(with $y = r$ at terminal steps). Moving $Q(s,a)$ a step $\alpha$ toward the sample is stochastic
approximation of the fixed point:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\,\big[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\big].$$

The target uses the *max* (the optimal policy), not the action actually taken next, so Q-learning is
**off-policy**: the behaviour policy can be ε-greedy for exploration while $Q$ converges to $Q^*$,
provided every state-action pair is visited infinitely often and the step sizes satisfy
$\sum \alpha_t = \infty$, $\sum \alpha_t^2 < \infty$ (Watkins and Dayan, 1992). In practice a small
constant $\alpha$ and a decaying ε are used, and convergence is checked empirically (C9.1: Q-table
change below 1e-3 over the last 1,000 episodes).

## 6. Why the gate is the reward model

Calling the 32B judge in every episode would cost thousands of GPU generations. The quality gate is a
cheap classifier of the same label, so learning uses it, and the learnt greedy policy is verified
once with the real judge on 300 validation prompts (row 9.2.4).
