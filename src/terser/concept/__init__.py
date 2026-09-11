"""Concept learning: hypotheses, Find-S, Candidate-Elimination, version spaces."""

from terser.concept.attributes import (
    ATTRIBUTE_NAMES,
    ATTRIBUTES,
    DOMAINS,
    OP_CODES,
    PROMPT_ATTRIBUTES,
    describe_prompt,
    instance,
)
from terser.concept.candidate_elimination import (
    BoundaryTooLargeError,
    CandidateEliminationResult,
    Collapse,
    candidate_elimination,
    minimal_conflict,
)
from terser.concept.find_s import covered_negatives, find_s
from terser.concept.hypothesis import (
    ANY,
    NONE,
    Domains,
    Example,
    Hypothesis,
    Instance,
    covers,
    format_hypothesis,
    hypothesis_space,
    is_consistent,
    min_generalise,
    min_specialisations,
    more_general_or_equal,
)
from terser.concept.version_space import (
    brute_force_version_space,
    iter_version_space,
    version_space,
)

__all__ = [
    "ANY",
    "ATTRIBUTES",
    "ATTRIBUTE_NAMES",
    "DOMAINS",
    "NONE",
    "OP_CODES",
    "PROMPT_ATTRIBUTES",
    "BoundaryTooLargeError",
    "CandidateEliminationResult",
    "Collapse",
    "Domains",
    "Example",
    "Hypothesis",
    "Instance",
    "brute_force_version_space",
    "candidate_elimination",
    "covered_negatives",
    "covers",
    "describe_prompt",
    "find_s",
    "format_hypothesis",
    "hypothesis_space",
    "instance",
    "is_consistent",
    "iter_version_space",
    "min_generalise",
    "min_specialisations",
    "minimal_conflict",
    "more_general_or_equal",
    "version_space",
]
