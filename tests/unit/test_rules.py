"""Unit tests for ``classify.rules``. M5 (PRD §7.5)."""

from __future__ import annotations

import math

import pytest

from pv_geom.classify.interface import MountingFeatures, MountingResult
from pv_geom.classify.rules import (
    RulesMountingClassifier,
    _conf_ge,
    _conf_le,
    classify_mounting,
)
from pv_geom.config import MountingRulesConfig


# --------------------------------------------------------------------------- #
# Confidence helpers
# --------------------------------------------------------------------------- #


def test_conf_le_at_threshold_is_half() -> None:
    assert _conf_le(5.0, threshold=5.0, margin=0.5) == pytest.approx(0.5)


def test_conf_le_far_below_is_one() -> None:
    assert _conf_le(2.0, threshold=5.0, margin=0.5) == 1.0   # full_pass = 2.5


def test_conf_le_far_above_is_zero() -> None:
    assert _conf_le(8.0, threshold=5.0, margin=0.5) == 0.0   # full_fail = 7.5


def test_conf_le_monotonic() -> None:
    xs = [_conf_le(v, 5.0, 0.5) for v in [2, 3, 4, 5, 6, 7, 8]]
    assert all(b <= a for a, b in zip(xs, xs[1:]))


def test_conf_ge_symmetric() -> None:
    """For value <-> threshold mirroring, ge and le are dual operations."""
    assert _conf_ge(7.0, 5.0, 0.5) == pytest.approx(_conf_le(3.0, 5.0, 0.5))
    assert _conf_ge(5.0, 5.0, 0.5) == pytest.approx(0.5)


def test_conf_zero_margin_is_step() -> None:
    assert _conf_le(5.0, 5.0, 0.0) == 1.0
    assert _conf_le(5.001, 5.0, 0.0) == 0.0


def test_conf_nan_input_is_zero() -> None:
    assert _conf_le(float("nan"), 5.0, 0.5) == 0.0
    assert _conf_ge(float("nan"), 5.0, 0.5) == 0.0


# --------------------------------------------------------------------------- #
# Rules engine — happy paths
# --------------------------------------------------------------------------- #


def _features(**overrides) -> MountingFeatures:
    base = dict(
        on_building=True,
        panel_tilt_deg=5.0,
        panel_azimuth_deg=180.0,
        panel_roof_angle_deg=2.0,
        height_above_roof_m=0.2,
        height_above_ground_m=4.0,
        area_m2=50.0,
        aspect_ratio=1.5,
        roof_plane_available=True,
    )
    base.update(overrides)
    return MountingFeatures(**base)


def test_R1_flush_mount_clear_match() -> None:
    f = _features(panel_roof_angle_deg=1.0, height_above_roof_m=0.1)
    r = classify_mounting(f, MountingRulesConfig())
    assert r.label == "flush_mount_rooftop"
    assert r.triggered_rule == "R1"
    assert r.confidence > 0.9


def test_R2_tilted_rack_with_roof() -> None:
    f = _features(
        panel_roof_angle_deg=15.0,   # >> 5 deg, very far past threshold
        height_above_roof_m=0.5,
    )
    r = classify_mounting(f, MountingRulesConfig())
    assert r.label == "tilted_rack_rooftop"
    assert r.triggered_rule == "R2"
    assert r.confidence > 0.5


def test_R2_tilted_rack_fallback_no_roof() -> None:
    f = _features(
        roof_plane_available=False,
        panel_roof_angle_deg=float("nan"),
        height_above_roof_m=float("nan"),
        panel_tilt_deg=20.0,
        height_above_ground_m=4.0,
    )
    r = classify_mounting(f, MountingRulesConfig())
    assert r.label == "tilted_rack_rooftop"
    assert r.triggered_rule == "R2"


def test_R3_carport() -> None:
    f = _features(
        on_building=False, roof_plane_available=False,
        panel_roof_angle_deg=float("nan"), height_above_roof_m=float("nan"),
        height_above_ground_m=3.0, aspect_ratio=4.0,
        panel_tilt_deg=10.0,
    )
    r = classify_mounting(f, MountingRulesConfig())
    assert r.label == "carport"
    assert r.triggered_rule == "R3"


def test_R4_tracker() -> None:
    f = _features(
        on_building=False, roof_plane_available=False,
        panel_roof_angle_deg=float("nan"), height_above_roof_m=float("nan"),
        height_above_ground_m=1.0, aspect_ratio=10.0,
        panel_tilt_deg=15.0,
    )
    r = classify_mounting(f, MountingRulesConfig())
    assert r.label == "ground_mount_tracker_suspected"
    assert r.triggered_rule == "R4"


def test_R5_ground_mount_fixed() -> None:
    f = _features(
        on_building=False, roof_plane_available=False,
        panel_roof_angle_deg=float("nan"), height_above_roof_m=float("nan"),
        height_above_ground_m=1.0, aspect_ratio=2.0,
        panel_tilt_deg=20.0,
    )
    r = classify_mounting(f, MountingRulesConfig())
    assert r.label == "ground_mount_fixed"
    assert r.triggered_rule == "R5"


def test_R6_ambiguous_default() -> None:
    """On-building, roof-plane-unavailable, panel-near-horizontal — R1/R2 fail."""
    f = _features(
        on_building=True, roof_plane_available=False,
        panel_roof_angle_deg=float("nan"), height_above_roof_m=float("nan"),
        panel_tilt_deg=2.0, height_above_ground_m=2.0,
    )
    r = classify_mounting(f, MountingRulesConfig())
    assert r.label == "ambiguous"
    assert r.triggered_rule == "R6"
    # confidence ~ 1 - best_near_miss ∈ [0, 1]
    assert 0.0 <= r.confidence <= 1.0


# --------------------------------------------------------------------------- #
# Order semantics: R3 (carport) wins over R4 (tracker) when both off-building.
# --------------------------------------------------------------------------- #


def test_R3_beats_R4_when_both_could_match() -> None:
    """Off-building, high HAG, very elongated, low tilt: technically R4-shaped
    but HAG is in carport range (>= 2 m). R3 should fire first."""
    f = _features(
        on_building=False, roof_plane_available=False,
        panel_roof_angle_deg=float("nan"), height_above_roof_m=float("nan"),
        height_above_ground_m=2.5,    # in carport range; R4 needs HAG < 2.0
        aspect_ratio=10.0,
        panel_tilt_deg=20.0,
    )
    r = classify_mounting(f, MountingRulesConfig())
    assert r.triggered_rule == "R3"
    assert r.label == "carport"


# --------------------------------------------------------------------------- #
# Confidence semantics
# --------------------------------------------------------------------------- #


def test_confidence_at_threshold_is_half() -> None:
    """At the threshold exactly, the rule fires with confidence 0.5."""
    cfg = MountingRulesConfig()
    f = _features(
        panel_roof_angle_deg=cfg.R1.panel_roof_angle_deg_max,   # exactly at threshold
        height_above_roof_m=0.0,                                # well past threshold
    )
    r = classify_mounting(f, cfg)
    assert r.label == "flush_mount_rooftop"
    assert r.confidence == pytest.approx(0.5, abs=1e-6)


def test_ambiguous_confidence_inverse_of_near_miss() -> None:
    """Sit in the R1/R2 height-above-roof gap (R1 needs <=0.5, R2 needs <=1.5;
    set 1.6 so both fail) — should fall through to R6 with non-zero confidence."""
    cfg = MountingRulesConfig()
    f = _features(
        panel_roof_angle_deg=6.0,    # R2-leaning angle
        height_above_roof_m=1.6,     # past both R1 and R2 height caps
        panel_tilt_deg=2.0,
    )
    r = classify_mounting(f, cfg)
    assert r.label == "ambiguous"
    # Best near-miss is R2 with confidence ~0.43; ambiguous = 1 - 0.43 ~= 0.57
    assert 0.5 < r.confidence < 0.7


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #


def test_classifier_returns_mounting_result_type() -> None:
    f = _features()
    r = classify_mounting(f, MountingRulesConfig())
    assert isinstance(r, MountingResult)
    assert not math.isnan(r.confidence)


def test_classifier_class_can_be_reused() -> None:
    clf = RulesMountingClassifier(MountingRulesConfig())
    f1 = _features()
    f2 = _features(on_building=False, roof_plane_available=False,
                   panel_roof_angle_deg=float("nan"),
                   height_above_roof_m=float("nan"),
                   height_above_ground_m=1.0, panel_tilt_deg=20.0)
    r1 = clf.classify(f1)
    r2 = clf.classify(f2)
    assert r1.triggered_rule == "R1"
    assert r2.triggered_rule == "R5"
