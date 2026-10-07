"""Rules-based mounting classifier (PRD §7.5 + extended taxonomy). M5.

Rules evaluated in order; first match wins. Rule IDs are stable identifiers,
not evaluation order: more-specific rules added after the PRD's R1-R6 run
*before* the broader rule they would otherwise lose polygons to (R7 east-west
rack before R1/R2; R8 pole mount before R3 carport). Confidence is a
piecewise-linear function of the margin past the deciding threshold. The
``ambiguous`` default returns ``1 - best_near_miss_confidence``.

The fractional ``confidence_margin`` (default 0.5) scales each threshold
absolutely: for threshold T, full confidence is reached at distance
``|T| * margin`` past T (or at distance ``|T| * margin`` short of T for
opposing direction). At T exactly, confidence is 0.5.

Canopy evidence (0.3.0): ground-class returns inside the polygon with a
multi-metre vertical gap to the panel plane are the LiDAR signature of an
open-sided canopy. When present, the polygon is routed down the canopy rules
(R8/R3) even if a footprint layer says ``on_building`` (carports are often
mapped as buildings), and the ground-mount rules (R4/R5) are suppressed.
Unknown (NaN) heights never satisfy evidence conditions — they propagate to
``ambiguous`` — but an unknown height does not veto an *upper-cap* guard.

Panel standoff (0.4.0): when the panel plane is not resolvably above the roof
plane, every panel-vs-roof feature is uninformative — the row is equally
consistent with a low-profile flush mount and with an array that was not in the
point cloud at all. The label still stands, but its confidence is capped at
``no_panel_standoff_confidence_max``.
"""

from __future__ import annotations

import math

from pv_geom.classify.interface import (
    MountingClassifier,
    MountingFeatures,
    MountingResult,
)
from pv_geom.config import MountingRulesConfig


def _is_nan(x: float | None) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _conf_le(value: float | None, threshold: float, margin: float) -> float:
    """Confidence that ``value <= threshold``, smoothed by ``margin`` (fractional)."""
    if _is_nan(value):
        return 0.0
    abs_margin = abs(threshold) * margin
    if abs_margin <= 1e-12:
        return 1.0 if value <= threshold else 0.0
    full_pass = threshold - abs_margin
    full_fail = threshold + abs_margin
    if value <= full_pass:
        return 1.0
    if value >= full_fail:
        return 0.0
    return float((full_fail - value) / (full_fail - full_pass))


def _conf_ge(value: float | None, threshold: float, margin: float) -> float:
    """Confidence that ``value >= threshold``, smoothed by ``margin`` (fractional)."""
    if _is_nan(value):
        return 0.0
    abs_margin = abs(threshold) * margin
    if abs_margin <= 1e-12:
        return 1.0 if value >= threshold else 0.0
    full_pass = threshold + abs_margin
    full_fail = threshold - abs_margin
    if value >= full_pass:
        return 1.0
    if value <= full_fail:
        return 0.0
    return float((value - full_fail) / (full_pass - full_fail))


def _conf_le_unknown_ok(value: float | None, threshold: float, margin: float) -> float:
    """``_conf_le`` where an unknown value does NOT veto.

    Used for upper-cap *guards* (e.g. "carports are under ~6 m"): the cap
    should only bind when the value was actually measured; the rule's positive
    evidence conditions still require real measurements to fire.
    """
    if _is_nan(value):
        return 1.0
    return _conf_le(value, threshold, margin)


class RulesMountingClassifier(MountingClassifier):
    """PRD §7.5's R1-R6 plus the extended taxonomy: R7 east_west_rack_rooftop,
    R8 pole_mount, and R1's flat/pitched label split (logged in STATUS.md)."""

    def __init__(self, cfg: MountingRulesConfig) -> None:
        self.cfg = cfg

    def classify(self, f: MountingFeatures) -> MountingResult:
        result = self._classify(f)
        # A row whose panel plane is indistinguishable from the roof beneath it
        # scores *maximally* on R1's two conditions while being exactly what an
        # array missing from the point cloud looks like. Cap the confidence so
        # the label survives but cannot be mistaken for strong evidence.
        # `ambiguous` is exempt: its confidence measures how far the row was
        # from any rule firing, which the standoff tells us nothing about.
        if f.no_panel_standoff and result.label != "ambiguous":
            capped = min(result.confidence, self.cfg.no_panel_standoff_confidence_max)
            if capped != result.confidence:
                return MountingResult(result.label, float(capped), result.triggered_rule)
        return result

    def _classify(self, f: MountingFeatures) -> MountingResult:
        cfg = self.cfg
        m = cfg.confidence_margin
        near_misses: list[float] = []

        # ------------------------------------------------------------------
        # Canopy evidence — ground returns under the panels. Rooftops have
        # none (the building blocks them); open-sided canopies let LiDAR
        # through around the edges and gaps. Footprint-independent, so it
        # corrects both failure modes of ``on_building``: carports mapped as
        # buildings, and rooftops on unmapped buildings.
        # ------------------------------------------------------------------
        canopy_gap_conf = 0.0
        if f.n_ground_under >= cfg.canopy_min_ground_points_under:
            canopy_gap_conf = _conf_ge(f.ground_under_gap_m, cfg.canopy_gap_m_min, m)
        canopy_evidence = canopy_gap_conf >= 0.5

        rooftop_context = f.on_building and not canopy_evidence
        canopy_context = (not f.on_building) or canopy_evidence

        # ------------------------------------------------------------------
        # R7 — east_west_rack_rooftop (before R1/R2: the two-plane EW
        # signature is more specific than either, and R2 would otherwise
        # absorb the primary plane as a generic tilted rack). Caveat: two
        # facets of a gable roof also face ~180 deg apart with similar tilts,
        # so on a pitched roof the EW flag is most likely facet bleed —
        # confidence is graded on the roof being flat when a roof fit exists.
        # ------------------------------------------------------------------
        if rooftop_context and f.east_west_rack:
            r = cfg.R7
            c = _conf_le(f.panel_tilt_deg, r.tilt_deg_max, m)
            if f.roof_plane_available:
                c = min(c, _conf_le(f.roof_tilt_deg, cfg.flat_roof_tilt_deg_max, m))
            if c >= 0.5:
                return MountingResult("east_west_rack_rooftop", c, "R7")
            near_misses.append(c)

        # ------------------------------------------------------------------
        # R1 — flush mount, split by roof type. Firing/confidence semantics
        # are the PRD's R1; the label refines on roof tilt (hard threshold —
        # the split is descriptive, not a separate gate).
        # ------------------------------------------------------------------
        if rooftop_context and f.roof_plane_available:
            r = cfg.R1
            c = min(
                _conf_le(f.panel_roof_angle_deg, r.panel_roof_angle_deg_max, m),
                _conf_le(f.height_above_roof_m, r.height_above_roof_m_max, m),
            )
            if c >= 0.5:
                flat = (
                    not _is_nan(f.roof_tilt_deg)
                    and f.roof_tilt_deg <= cfg.flat_roof_tilt_deg_max
                )
                label = "flush_mount_flat_roof" if flat else "flush_mount_pitched_roof"
                return MountingResult(label, c, "R1")
            near_misses.append(c)

        # ------------------------------------------------------------------
        # R2 — tilted_rack_rooftop (with/without roof-plane fallback)
        # ------------------------------------------------------------------
        if rooftop_context:
            r = cfg.R2
            c_with = c_without = 0.0
            if f.roof_plane_available:
                c_with = min(
                    _conf_ge(f.panel_roof_angle_deg, r.panel_roof_angle_deg_min, m),
                    _conf_le(f.height_above_roof_m, r.height_above_roof_m_max, m),
                )
            else:
                c_without = min(
                    _conf_ge(f.panel_tilt_deg, r.fallback_tilt_deg_min, m),
                    _conf_ge(f.height_above_ground_m, r.fallback_height_above_ground_m_min, m),
                )
            c = max(c_with, c_without)
            if c >= 0.5:
                return MountingResult("tilted_rack_rooftop", c, "R2")
            near_misses.append(c)

        # ------------------------------------------------------------------
        # R8 — pole_mount (before R3: a small near-square elevated array is a
        # pole mount; R3's aspect_ratio_min only catches the elongated ones,
        # so these used to land in ambiguous or worse). Elevation is
        # satisfiable by measured HAG *or* the under-panel gap; the height cap
        # guards against rooftops on unmapped buildings and only binds when
        # HAG was actually measured.
        # ------------------------------------------------------------------
        if canopy_context:
            r = cfg.R8
            c = min(
                max(
                    _conf_ge(f.height_above_ground_m, r.height_above_ground_m_min, m),
                    canopy_gap_conf,
                ),
                _conf_le_unknown_ok(f.height_above_ground_m, r.height_above_ground_m_max, m),
                _conf_le(f.area_m2, r.area_m2_max, m),
                _conf_le(f.aspect_ratio, r.aspect_ratio_max, m),
            )
            if c >= 0.5:
                return MountingResult("pole_mount", c, "R8")
            near_misses.append(c)

        # ------------------------------------------------------------------
        # R3 — carport (same elevation/cap semantics as R8)
        # ------------------------------------------------------------------
        if canopy_context:
            r = cfg.R3
            c = min(
                max(
                    _conf_ge(f.height_above_ground_m, r.height_above_ground_m_min, m),
                    canopy_gap_conf,
                ),
                _conf_le_unknown_ok(f.height_above_ground_m, r.height_above_ground_m_max, m),
                _conf_ge(f.aspect_ratio, r.aspect_ratio_min, m),
            )
            if c >= 0.5:
                return MountingResult("carport", c, "R3")
            near_misses.append(c)

        # ------------------------------------------------------------------
        # R4 — ground_mount_tracker_suspected (suppressed by canopy evidence:
        # a multi-metre gap under the panels contradicts "at ground level"
        # even when a stray HAG measurement says otherwise)
        # ------------------------------------------------------------------
        if not f.on_building and not canopy_evidence:
            r = cfg.R4
            c = min(
                _conf_le(f.height_above_ground_m, r.height_above_ground_m_max, m),
                _conf_ge(f.aspect_ratio, r.aspect_ratio_min, m),
                _conf_le(f.panel_tilt_deg, r.tilt_deg_max, m),
            )
            if c >= 0.5:
                return MountingResult("ground_mount_tracker_suspected", c, "R4")
            near_misses.append(c)

        # ------------------------------------------------------------------
        # R5 — ground_mount_fixed (suppressed by canopy evidence, as R4)
        # ------------------------------------------------------------------
        if not f.on_building and not canopy_evidence:
            r = cfg.R5
            c = min(
                _conf_le(f.height_above_ground_m, r.height_above_ground_m_max, m),
                _conf_ge(f.panel_tilt_deg, r.tilt_deg_min, m),
            )
            if c >= 0.5:
                return MountingResult("ground_mount_fixed", c, "R5")
            near_misses.append(c)

        # ------------------------------------------------------------------
        # R6 — ambiguous (default)
        # ------------------------------------------------------------------
        best_near = max(near_misses) if near_misses else 0.0
        return MountingResult("ambiguous", float(1.0 - best_near), "R6")


def classify_mounting(features: MountingFeatures, cfg: MountingRulesConfig) -> MountingResult:
    """Convenience wrapper around the default rules classifier."""
    return RulesMountingClassifier(cfg).classify(features)
