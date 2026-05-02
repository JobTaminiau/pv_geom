"""Rules-based mounting classifier (PRD §7.5). M5.

Rules evaluated in order; first match wins. Confidence is a piecewise-linear
function of the margin past the deciding threshold. The ``ambiguous`` default
returns ``1 - best_near_miss_confidence``.

The fractional ``confidence_margin`` (default 0.5) scales each threshold
absolutely: for threshold T, full confidence is reached at distance
``|T| * margin`` past T (or at distance ``|T| * margin`` short of T for
opposing direction). At T exactly, confidence is 0.5.
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


class RulesMountingClassifier(MountingClassifier):
    """Implements the v1 mounting rules from PRD §7.5."""

    def __init__(self, cfg: MountingRulesConfig) -> None:
        self.cfg = cfg

    def classify(self, f: MountingFeatures) -> MountingResult:
        cfg = self.cfg
        m = cfg.confidence_margin
        near_misses: list[float] = []

        # ------------------------------------------------------------------
        # R1 — flush_mount_rooftop
        # ------------------------------------------------------------------
        if f.on_building and f.roof_plane_available:
            r = cfg.R1
            c = min(
                _conf_le(f.panel_roof_angle_deg, r.panel_roof_angle_deg_max, m),
                _conf_le(f.height_above_roof_m, r.height_above_roof_m_max, m),
            )
            if c >= 0.5:
                return MountingResult("flush_mount_rooftop", c, "R1")
            near_misses.append(c)

        # ------------------------------------------------------------------
        # R2 — tilted_rack_rooftop (with/without roof-plane fallback)
        # ------------------------------------------------------------------
        if f.on_building:
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
        # R3 — carport
        # ------------------------------------------------------------------
        if not f.on_building:
            r = cfg.R3
            c = min(
                _conf_ge(f.height_above_ground_m, r.height_above_ground_m_min, m),
                _conf_ge(f.aspect_ratio, r.aspect_ratio_min, m),
            )
            if c >= 0.5:
                return MountingResult("carport", c, "R3")
            near_misses.append(c)

        # ------------------------------------------------------------------
        # R4 — ground_mount_tracker_suspected
        # ------------------------------------------------------------------
        if not f.on_building:
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
        # R5 — ground_mount_fixed
        # ------------------------------------------------------------------
        if not f.on_building:
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
