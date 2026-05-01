"""Quality flags + confidence scoring helpers. Filled out across M3–M5."""

from __future__ import annotations


def linear_confidence(value: float, threshold: float, margin: float) -> float:
    """Piecewise-linear confidence: 0.5 at threshold, 1.0 once `margin` past it.

    Used by the rules engine (PRD §7.5). `value` and `threshold` should be
    pre-aligned so positive `(value - threshold)` means "more confident".
    """
    if margin <= 0:
        return 1.0 if value >= threshold else 0.0
    delta = (value - threshold) / margin
    if delta >= 1.0:
        return 1.0
    if delta <= -1.0:
        return 0.0
    return 0.5 + 0.5 * delta
