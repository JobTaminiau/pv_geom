"""Unit tests for ``classify.rules``. M5 (PRD §7.5)."""

from __future__ import annotations

import math

import pytest

from pv_geom.config import MountingRulesConfig
from pv_geom.experimental.mounting.interface import MountingFeatures, MountingResult
from pv_geom.experimental.mounting.rules import (
    RulesMountingClassifier,
    _conf_ge,
    _conf_le,
    classify_mounting,
)

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
        roof_tilt_deg=18.0,        # pitched residential roof
        east_west_rack=False,
    )
    base.update(overrides)
    return MountingFeatures(**base)


def test_R1_flush_mount_pitched_roof() -> None:
    f = _features(panel_roof_angle_deg=1.0, height_above_roof_m=0.1, roof_tilt_deg=18.0)
    r = classify_mounting(f, MountingRulesConfig())
    assert r.label == "flush_mount_pitched_roof"
    assert r.triggered_rule == "R1"
    assert r.confidence > 0.9


def test_R1_flush_mount_flat_roof() -> None:
    """Same R1 predicate; roof tilt at/below flat_roof_tilt_deg_max flips the label."""
    f = _features(panel_roof_angle_deg=1.0, height_above_roof_m=0.1, roof_tilt_deg=2.0)
    r = classify_mounting(f, MountingRulesConfig())
    assert r.label == "flush_mount_flat_roof"
    assert r.triggered_rule == "R1"
    assert r.confidence > 0.9


def test_R1_label_split_does_not_change_confidence() -> None:
    """The flat/pitched split is descriptive only — identical features apart
    from roof tilt must fire R1 with identical confidence."""
    flat = classify_mounting(_features(roof_tilt_deg=2.0), MountingRulesConfig())
    pitched = classify_mounting(_features(roof_tilt_deg=18.0), MountingRulesConfig())
    assert flat.label == "flush_mount_flat_roof"
    assert pitched.label == "flush_mount_pitched_roof"
    assert flat.confidence == pytest.approx(pitched.confidence)


def test_R7_east_west_rack_on_flat_roof() -> None:
    """EW flag + flat roof + modest tilt fires R7 even though the primary
    plane alone would have matched R2 as a generic tilted rack."""
    f = _features(
        east_west_rack=True,
        roof_tilt_deg=2.0,
        panel_tilt_deg=10.0,
        panel_roof_angle_deg=10.0,   # would satisfy R2
        height_above_roof_m=0.5,
    )
    r = classify_mounting(f, MountingRulesConfig())
    assert r.label == "east_west_rack_rooftop"
    assert r.triggered_rule == "R7"
    assert r.confidence > 0.5


def test_R7_gable_facet_guard() -> None:
    """Two facets of a pitched roof also face ~180 deg apart, so the EW flag
    on a clearly pitched roof must NOT fire R7; it falls through to R1."""
    f = _features(
        east_west_rack=True,
        roof_tilt_deg=20.0,          # pitched -> EW signature is facet bleed
        panel_tilt_deg=20.0,
        panel_roof_angle_deg=1.0,    # panel parallel to its facet
        height_above_roof_m=0.1,
    )
    r = classify_mounting(f, MountingRulesConfig())
    assert r.triggered_rule == "R1"
    assert r.label == "flush_mount_pitched_roof"


def test_R7_fires_without_roof_plane() -> None:
    """No roof fit (e.g. roof_insufficient on a commercial roof): R7 still
    fires on the EW flag + modest tilt, beating R2's no-roof fallback."""
    f = _features(
        east_west_rack=True,
        roof_plane_available=False,
        roof_tilt_deg=None,
        panel_roof_angle_deg=float("nan"),
        height_above_roof_m=float("nan"),
        panel_tilt_deg=8.0,
        height_above_ground_m=5.0,   # would satisfy R2's fallback
    )
    r = classify_mounting(f, MountingRulesConfig())
    assert r.label == "east_west_rack_rooftop"
    assert r.triggered_rule == "R7"


def test_R8_pole_mount() -> None:
    f = _features(
        on_building=False, roof_plane_available=False,
        panel_roof_angle_deg=float("nan"), height_above_roof_m=float("nan"),
        roof_tilt_deg=None,
        height_above_ground_m=4.0, area_m2=8.0, aspect_ratio=1.2,
        panel_tilt_deg=25.0,
    )
    r = classify_mounting(f, MountingRulesConfig())
    assert r.label == "pole_mount"
    assert r.triggered_rule == "R8"
    assert r.confidence > 0.5


def test_R8_does_not_steal_carports() -> None:
    """Large elongated elevated arrays stay carports: R8's area cap rejects
    them long before R3 is consulted."""
    f = _features(
        on_building=False, roof_plane_available=False,
        panel_roof_angle_deg=float("nan"), height_above_roof_m=float("nan"),
        roof_tilt_deg=None,
        height_above_ground_m=3.0, area_m2=60.0, aspect_ratio=4.0,
        panel_tilt_deg=10.0,
    )
    r = classify_mounting(f, MountingRulesConfig())
    assert r.label == "carport"
    assert r.triggered_rule == "R3"


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
    assert r.label == "flush_mount_pitched_roof"
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
# Canopy evidence + NaN-height semantics (0.3.0)
# --------------------------------------------------------------------------- #


def _off_building(**overrides) -> MountingFeatures:
    base = dict(
        on_building=False,
        roof_plane_available=False,
        panel_roof_angle_deg=float("nan"),
        height_above_roof_m=float("nan"),
        roof_tilt_deg=None,
    )
    base.update(overrides)
    return _features(**base)


def test_nan_hag_off_building_is_ambiguous() -> None:
    """Unknown height must NOT classify as ground mount. Pre-0.3 the NaN was
    coerced to 0.0 upstream, so a canopy with no nearby ground reference fired
    R4/R5 with full margin confidence."""
    f = _off_building(
        height_above_ground_m=float("nan"), aspect_ratio=4.0, panel_tilt_deg=10.0
    )
    r = classify_mounting(f, MountingRulesConfig())
    assert r.label == "ambiguous"


def test_canopy_evidence_rescues_carport_with_nan_hag() -> None:
    """No HAG measurement, but plenty of ground returns under the panels with
    a 3 m gap: the LiDAR canopy signature alone supports carport."""
    f = _off_building(
        height_above_ground_m=float("nan"), aspect_ratio=4.0, panel_tilt_deg=8.0,
        n_ground_under=50, ground_under_gap_m=3.0,
    )
    r = classify_mounting(f, MountingRulesConfig())
    assert r.label == "carport"
    assert r.triggered_rule == "R3"
    assert r.confidence >= 0.5


def test_canopy_evidence_overrides_on_building() -> None:
    """Carport mapped as a building by the footprint layer: the features look
    flush-mount (panel parallel to the 'roof' it IS), but ground returns under
    the panels reroute it to carport."""
    f = _features(
        panel_roof_angle_deg=1.0, height_above_roof_m=0.1, roof_tilt_deg=2.0,
        aspect_ratio=3.0, height_above_ground_m=3.5,
        n_ground_under=80, ground_under_gap_m=3.5,
    )
    r = classify_mounting(f, MountingRulesConfig())
    assert r.label == "carport"
    assert r.triggered_rule == "R3"


def test_few_under_returns_do_not_reroute_rooftop() -> None:
    """A handful of stray ground returns inside a rooftop polygon (edge
    effects, misclassification) must not trip the canopy detector."""
    f = _features(
        panel_roof_angle_deg=1.0, height_above_roof_m=0.1,
        n_ground_under=5, ground_under_gap_m=4.0,
    )
    r = classify_mounting(f, MountingRulesConfig())
    assert r.triggered_rule == "R1"


def test_high_hag_off_building_is_ambiguous_not_carport() -> None:
    """Off-building at 9 m with no canopy evidence is almost certainly a
    rooftop whose building is missing from the footprint layer — pre-0.3 this
    was a confident carport."""
    f = _off_building(
        height_above_ground_m=9.0, aspect_ratio=4.0, panel_tilt_deg=10.0
    )
    r = classify_mounting(f, MountingRulesConfig())
    assert r.label == "ambiguous"


def test_r3_height_cap_grades_confidence() -> None:
    cfg = MountingRulesConfig()
    lo = classify_mounting(_off_building(height_above_ground_m=3.0, aspect_ratio=4.0), cfg)
    hi = classify_mounting(_off_building(height_above_ground_m=5.5, aspect_ratio=4.0), cfg)
    assert lo.label == "carport"
    assert hi.label == "carport"
    assert hi.confidence < lo.confidence


def test_ground_rules_suppressed_by_canopy_evidence() -> None:
    """Contradiction: measured HAG says ground level but there is a 2.5 m gap
    under the panels — trust the direct under-panel signal, never emit
    ground_mount alongside canopy evidence."""
    f = _off_building(
        height_above_ground_m=0.5, aspect_ratio=5.0, panel_tilt_deg=10.0,
        n_ground_under=40, ground_under_gap_m=2.5,
    )
    r = classify_mounting(f, MountingRulesConfig())
    assert r.label == "carport"


def test_pole_mount_via_canopy_evidence() -> None:
    f = _off_building(
        height_above_ground_m=float("nan"), area_m2=8.0, aspect_ratio=1.2,
        panel_tilt_deg=25.0, n_ground_under=30, ground_under_gap_m=3.0,
    )
    r = classify_mounting(f, MountingRulesConfig())
    assert r.label == "pole_mount"
    assert r.triggered_rule == "R8"


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


# --------------------------------------------------------------------------- #
# Panel standoff (input/LiDAR vintage) confidence cap
# --------------------------------------------------------------------------- #


def test_no_panel_standoff_caps_confidence() -> None:
    """A row with no resolvable panel standoff satisfies R1 *maximally* — panel
    -roof angle ~0, height above roof ~0 — which is also exactly what an array
    missing from the point cloud looks like. The label stands; the confidence
    must not."""
    cfg = MountingRulesConfig()
    clean = _features(panel_roof_angle_deg=0.5, height_above_roof_m=0.10)
    flat = classify_mounting(clean, cfg)
    assert flat.triggered_rule == "R1"
    assert flat.confidence == pytest.approx(1.0)

    suspect = _features(panel_roof_angle_deg=0.0, height_above_roof_m=0.0,
                        no_panel_standoff=True)
    r = classify_mounting(suspect, cfg)
    assert r.triggered_rule == "R1"
    assert r.label == flat.label                       # same reading of the evidence
    assert r.confidence == pytest.approx(cfg.no_panel_standoff_confidence_max)


def test_no_panel_standoff_cap_is_configurable_and_never_raises_confidence() -> None:
    disabled = MountingRulesConfig(no_panel_standoff_confidence_max=1.0)
    f = _features(panel_roof_angle_deg=0.0, height_above_roof_m=0.0,
                  no_panel_standoff=True)
    assert classify_mounting(f, disabled).confidence == pytest.approx(1.0)

    # A row that was already below the cap keeps its (lower) confidence.
    low = _features(panel_roof_angle_deg=4.5, height_above_roof_m=0.0,
                    no_panel_standoff=True)
    cfg = MountingRulesConfig()
    assert classify_mounting(low, cfg).confidence <= cfg.no_panel_standoff_confidence_max


def test_no_panel_standoff_does_not_touch_ambiguous() -> None:
    """`ambiguous` confidence means "how far from any rule firing", which the
    standoff says nothing about — capping it would be meaningless."""
    f = _features(on_building=False, roof_plane_available=False,
                  panel_roof_angle_deg=float("nan"),
                  height_above_roof_m=float("nan"),
                  height_above_ground_m=float("nan"),
                  panel_tilt_deg=float("nan"), no_panel_standoff=True)
    r = classify_mounting(f, MountingRulesConfig())
    assert r.label == "ambiguous"
    assert r.confidence == pytest.approx(1.0)
