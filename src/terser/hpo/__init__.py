"""Hyperparameter optimisation: grid, random, and Bayesian search."""

from terser.hpo.audit import DataAccessLog, audit_access_log, scan_sources
from terser.hpo.search import (
    METHODS,
    Categorical,
    Float,
    Int,
    SearchResult,
    SearchSpace,
    run_search,
)

__all__ = [
    "METHODS",
    "Categorical",
    "DataAccessLog",
    "Float",
    "Int",
    "SearchResult",
    "SearchSpace",
    "audit_access_log",
    "run_search",
    "scan_sources",
]
