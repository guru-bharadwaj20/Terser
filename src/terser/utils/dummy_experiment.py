"""Tiny CPU experiment used to verify the determinism harness.

Run: ``python -m terser.utils.dummy_experiment --seed 0``. Prints metrics as one JSON line.

Randomness sources: ``random``, NumPy global RNG, an explicit Generator, string set iteration
(hash-seed dependent), a train/test split and a scikit-learn model. The set iteration is sorted
before use because ``set_seed`` cannot change hash randomization of the *current* interpreter
(``PYTHONHASHSEED`` is read only at start-up), so output is identical for any inherited
``PYTHONHASHSEED``.
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from terser.utils.seed import rng, set_seed

VOCAB = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"}


def run(seed: int) -> dict[str, Any]:
    """Run the experiment and return a JSON-serializable metrics dict."""
    set_seed(seed)
    gen = rng(seed)

    # Hash-order dependent container: sort to be independent of PYTHONHASHSEED.
    words = sorted(VOCAB)
    random.shuffle(words)
    word_to_idx = {w: i for i, w in enumerate(words)}

    n = 200
    x = np.random.normal(size=(n, 4))
    tokens = gen.choice(words, size=n)
    tok_feat = np.array([word_to_idx[t] for t in tokens], dtype=float)[:, None]
    features = np.hstack([x, tok_feat])
    y = (x[:, 0] + 0.5 * x[:, 1] + 0.1 * tok_feat[:, 0] + gen.normal(0, 0.5, n) > 0).astype(int)

    x_tr, x_te, y_tr, y_te = train_test_split(features, y, test_size=0.3, random_state=seed)
    model = RandomForestClassifier(n_estimators=20, max_features=2, random_state=seed)
    model.fit(x_tr, y_tr)
    proba = model.predict_proba(x_te)

    return {
        "seed": seed,
        "word_order": words,
        "accuracy": round(float(accuracy_score(y_te, proba.argmax(axis=1))), 10),
        "log_loss": round(float(log_loss(y_te, proba, labels=[0, 1])), 10),
        "test_idx_checksum": round(float(x_te[:, 0].sum()), 10),
        "random_draw": random.random(),
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.seed), sort_keys=True))


if __name__ == "__main__":
    main()
