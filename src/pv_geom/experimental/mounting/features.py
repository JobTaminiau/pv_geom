"""Mounting features from a measurement, and the columns the classifier adds."""

from __future__ import annotations

from typing import Any

import numpy as np

from pv_geom.config import PVGeomConfig
from pv_geom.experimental.mounting.interface import MountingFeatures
from pv_geom.experimental.mounting.rules import classify_mounting
from pv_geom.pipeline.measure import LocalPoints, Measurement


def ground_under_polygon(
    panel_inlier_z: np.ndarray,
    ground_xyz: np.ndarray,
    polygon,
) -> tuple[int, float]:
    """Ground-class returns inside the polygon + vertical gap to the panel plane.

    The canopy discriminator: rooftop arrays have (near) zero ground returns
    inside their polygon because the building blocks the pulse; open-sided
    canopies (carports, pole mounts) let LiDAR reach the ground through and
    around the panels; ground mounts have ground directly below with a
    sub-metre gap. Returns ``(n_under, gap_m)`` where ``gap_m`` is median
    panel-inlier z minus median under-polygon ground z (NaN when either side
    is empty).
    """
    from shapely import contains_xy

    if len(ground_xyz) == 0 or polygon is None or polygon.is_empty:
        return 0, float("nan")
    minx, miny, maxx, maxy = polygon.bounds
    bbox_mask = (
        (ground_xyz[:, 0] >= minx) & (ground_xyz[:, 0] <= maxx)
        & (ground_xyz[:, 1] >= miny) & (ground_xyz[:, 1] <= maxy)
    )
    if not bbox_mask.any():
        return 0, float("nan")
    candidates = ground_xyz[bbox_mask]
    inside = contains_xy(polygon, candidates[:, 0], candidates[:, 1])
    n_under = int(inside.sum())
    if n_under == 0 or len(panel_inlier_z) == 0:
        return n_under, float("nan")
    return n_under, float(np.median(panel_inlier_z) - np.median(candidates[inside, 2]))


def is_tracker_suspected(
    *,
    on_building: bool,
    aspect_ratio: float,
    height_above_ground_m: float,
    tilt_deg: float,
    aspect_min: float = 4.0,
    height_above_ground_max_m: float = 2.0,
    tilt_max_deg: float = 35.0,
) -> bool:
    """Per-polygon tracker heuristic (PRD §7.2 v1).

    Off-building, elongated, low height-above-ground, low tilt. Sets the
    ``tracker_suspected`` flag on the row; the more robust spatial-clustering
    version is deferred to v1.1.
    """
    if on_building:
        return False
    if np.isnan(aspect_ratio) or np.isnan(tilt_deg) or np.isnan(height_above_ground_m):
        return False
    return (
        aspect_ratio >= aspect_min
        and height_above_ground_m < height_above_ground_max_m
        and tilt_deg < tilt_max_deg
    )


def mounting_columns(m: Measurement, pts: LocalPoints, cfg: PVGeomConfig) -> dict[str, Any]:
    """Classify one measurement. Returns the ``mounting_*`` columns and appends
    the classifier's own flags to ``m.flags``."""
    rules = cfg.mounting_rules
    fit = m.panel.fit
    hag = m.height_above_ground_m
    on_building = bool(m.roof.on_building)

    # Canopy evidence: ground-class returns under the panel polygon (NaN HAG
    # or footprint errors must not silently become "ground level"; the gap
    # under the panels is the direct LiDAR signal).
    n_ground_under, ground_under_gap = ground_under_polygon(
        m.panel_z, pts.ground, m.task.polygon)
    canopy_evidence = (
        n_ground_under >= rules.canopy_min_ground_points_under
        and not np.isnan(ground_under_gap)
        and ground_under_gap >= rules.canopy_gap_m_min
    )

    # Off-building + high HAG + no canopy evidence: most likely a rooftop on a
    # building the footprint layer is missing. The R3/R8 height caps route it
    # toward ambiguous; the flag makes the population auditable.
    if (
        not on_building
        and not canopy_evidence
        and not np.isnan(hag)
        and hag > rules.missing_footprint_hag_m
    ):
        m.flags.append("possible_missing_footprint")

    # NaN HAG/tilt propagate: is_tracker_suspected returns False on NaN, which
    # is the correct "unknown" behaviour.
    if is_tracker_suspected(
        on_building=on_building,
        aspect_ratio=m.aspect_ratio,
        height_above_ground_m=hag,
        tilt_deg=fit.tilt_deg,
    ):
        m.flags.append("tracker_suspected")

    # The rules only use roof features from a footprint ring: an open-ring
    # reference exists for the standoff screen, not as evidence of a building.
    use_roof = m.roof.usable and on_building
    feats = MountingFeatures(
        on_building=on_building,
        tilt_deg=fit.tilt_deg,
        azimuth_deg=fit.azimuth_deg,
        angle_to_roof_deg=m.angle_to_roof_deg if use_roof else float("nan"),
        height_above_roof_m=m.height_above_roof_m if use_roof else float("nan"),
        height_above_ground_m=hag,
        area_m2=m.area_m2,
        aspect_ratio=m.aspect_ratio,
        roof_plane_available=use_roof,
        roof_tilt_deg=float(m.roof.fit.tilt_deg) if use_roof else None,
        east_west_rack="east_west_rack" in m.flags,
        n_ground_under=n_ground_under,
        ground_under_gap_m=ground_under_gap,
        no_standoff=use_roof and "no_standoff" in m.flags,
    )
    result = classify_mounting(feats, rules)
    return {
        "mounting_type": str(result.label),
        "mounting_confidence": np.float32(result.confidence),
        "mounting_rule": str(result.triggered_rule),
    }
