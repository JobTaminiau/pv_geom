"""Confidence helper covers the M5 rules engine."""

import math

from pv_geom.quality import linear_confidence


def test_at_threshold_is_half() -> None:
    assert linear_confidence(5.0, threshold=5.0, margin=1.0) == 0.5


def test_full_margin_past_is_one() -> None:
    assert linear_confidence(6.0, threshold=5.0, margin=1.0) == 1.0


def test_full_margin_short_is_zero() -> None:
    assert linear_confidence(4.0, threshold=5.0, margin=1.0) == 0.0


def test_monotone_in_value() -> None:
    xs = [linear_confidence(v, threshold=5.0, margin=2.0) for v in [3, 4, 5, 6, 7]]
    assert all(b >= a for a, b in zip(xs, xs[1:], strict=False))


def test_zero_margin_is_step() -> None:
    assert linear_confidence(5.0, 5.0, 0.0) == 1.0
    assert linear_confidence(4.999, 5.0, 0.0) == 0.0


def test_clamped_to_unit_interval() -> None:
    c = linear_confidence(100.0, 5.0, 1.0)
    assert math.isclose(c, 1.0)
