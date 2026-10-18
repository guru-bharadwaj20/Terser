"""Promotion guard for continuous training (plan row **11.4.4**, checkpoint C11.7).

CT (row 11.4.3) retrains a component and asks to register it as ``Staging``. The guard compares
the candidate's validation summary with the current champion and the fixed quality floors, and
refuses promotion on any regression. It never looks at test data: summaries come from the
validation split.

Rules (all must hold; thresholds in :class:`GuardConfig`):

1. overall preservation ``>= min_preservation`` (0.92, the dashboard alert floor of row 11.5.5);
2. every category's preservation ``>= min_category_preservation`` (0.90, criterion S3), and no
   category that the champion has is missing;
3. no drop versus the champion larger than ``max_preservation_drop`` overall, or
   ``max_category_drop`` in any category;
4. savings not lower than the champion's by more than ``max_savings_drop``;
5. gate AUC (when both report it) not lower than the champion's by more than ``max_auc_drop``;
6. the candidate passed every required checkpoint id;
7. the candidate has full lineage (row 11.2.3).

:func:`guarded_promote` calls the registry only when :func:`decide` allows it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

__all__ = ["Decision", "EvalSummary", "GuardConfig", "decide", "guarded_promote"]


@dataclass(frozen=True)
class EvalSummary:
    """Validation-split summary of one model version."""

    preservation: float
    savings: float
    per_category_preservation: Mapping[str, float]
    gate_auc: float | None = None
    checkpoints_passed: Sequence[str] = ()
    has_lineage: bool = False
    """Registry lineage is complete (rule 7); False unless a caller has checked it."""


@dataclass(frozen=True)
class GuardConfig:
    """Promotion thresholds."""

    min_preservation: float = 0.92
    min_category_preservation: float = 0.90
    max_preservation_drop: float = 0.005
    max_category_drop: float = 0.02
    max_savings_drop: float = 0.01
    max_auc_drop: float = 0.005
    required_checkpoints: Sequence[str] = ()


@dataclass
class Decision:
    """Outcome; ``promote`` is True only with no reasons to refuse."""

    reasons: list[str] = field(default_factory=list)

    @property
    def promote(self) -> bool:
        return not self.reasons


def decide(
    candidate: EvalSummary, champion: EvalSummary | None, cfg: GuardConfig | None = None
) -> Decision:
    """Apply every rule; collect all failures (not just the first)."""
    cfg = cfg or GuardConfig()
    d = Decision()
    c = candidate
    if c.preservation < cfg.min_preservation:
        d.reasons.append(f"preservation {c.preservation:.3f} < floor {cfg.min_preservation}")
    for cat, p in sorted(c.per_category_preservation.items()):
        if p < cfg.min_category_preservation:
            d.reasons.append(f"{cat} preservation {p:.3f} < floor {cfg.min_category_preservation}")
    missing = [ck for ck in cfg.required_checkpoints if ck not in c.checkpoints_passed]
    if missing:
        d.reasons.append(f"checkpoints not passed: {missing}")
    if not c.has_lineage:
        d.reasons.append("no lineage (git SHA / data hash / config / checkpoints)")
    if champion is not None:
        ch = champion
        if c.preservation < ch.preservation - cfg.max_preservation_drop:
            d.reasons.append(
                f"preservation regressed {ch.preservation:.3f} -> {c.preservation:.3f}"
            )
        for cat, p0 in sorted(ch.per_category_preservation.items()):
            p1 = c.per_category_preservation.get(cat)
            if p1 is None:
                d.reasons.append(f"category {cat} missing from candidate evaluation")
            elif p1 < p0 - cfg.max_category_drop:
                d.reasons.append(f"{cat} preservation regressed {p0:.3f} -> {p1:.3f}")
        if c.savings < ch.savings - cfg.max_savings_drop:
            d.reasons.append(f"savings regressed {ch.savings:.3f} -> {c.savings:.3f}")
        if (
            c.gate_auc is not None
            and ch.gate_auc is not None
            and c.gate_auc < ch.gate_auc - cfg.max_auc_drop
        ):
            d.reasons.append(f"gate AUC regressed {ch.gate_auc:.3f} -> {c.gate_auc:.3f}")
    return d


def guarded_promote(
    candidate: EvalSummary,
    champion: EvalSummary | None,
    promote: Callable[[], None],
    cfg: GuardConfig | None = None,
) -> Decision:
    """Call ``promote()`` (e.g. ``terser.tracking.promote(name, v, "Staging")``) only if allowed."""
    decision = decide(candidate, champion, cfg)
    if decision.promote:
        promote()
    return decision
