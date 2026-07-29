"""Height-above-ground / height-above-roof helpers. M4 (PRD §7.4)."""

from __future__ import annotations

import numpy as np

from pv_geom.geometry.plane_fit import PlaneFit


def height_above_ground(
    panel_inlier_z: np.ndarray,
    ground_xyz: np.ndarray,
    polygon_centroid_xy: tuple[float, float],
    search_radius_m: float = 10.0,
    fallback_k: int = 50,
    fallback_max_radius_m: float = 100.0,
) -> float:
    """Median panel-inlier z minus median class-2 z within ``search_radius_m``.

    PRD §7.4. When the disk is empty (large buildings: class 2 stops at the
    wall, so a roof centroid can be >10 m from any ground return), fall back
    to the ``fallback_k`` nearest ground points within ``fallback_max_radius_m``
    so big commercial rooftops get a HAG instead of NaN. Returns NaN when the
    panel inliers are empty or no ground point qualifies (``fallback_k=0``
    disables the fallback).
    """
    if len(panel_inlier_z) == 0 or len(ground_xyz) == 0:
        return float("nan")
    cx, cy = polygon_centroid_xy
    dx = ground_xyz[:, 0] - cx
    dy = ground_xyz[:, 1] - cy
    d2 = dx * dx + dy * dy
    mask = d2 <= search_radius_m ** 2
    if mask.any():
        return float(np.median(panel_inlier_z) - np.median(ground_xyz[mask, 2]))

    if fallback_k <= 0:
        return float("nan")
    within = d2 <= fallback_max_radius_m ** 2
    if not within.any():
        return float("nan")
    d2_within = d2[within]
    k = min(fallback_k, len(d2_within))
    nearest = np.argpartition(d2_within, k - 1)[:k]
    return float(np.median(panel_inlier_z) - np.median(ground_xyz[within][nearest, 2]))


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


def height_above_roof(
    panel_inlier_xyz: np.ndarray,
    roof_plane: PlaneFit,
) -> float:
    """Vertical offset from the panel inlier centroid to the fitted roof plane.

    PRD §7.4. Roof z at (x, y) is solved from the plane equation
    ``n . (p - centroid) = 0``. Returns NaN if the roof fit is degenerate
    (n_inliers < 3, NaN tilt) or if the plane is near-vertical (|nz| < 1e-9,
    physically implausible for a roof).
    """
    if roof_plane.n_inliers < 3 or np.isnan(roof_plane.tilt_deg):
        return float("nan")
    if len(panel_inlier_xyz) == 0:
        return float("nan")

    px = float(np.median(panel_inlier_xyz[:, 0]))
    py = float(np.median(panel_inlier_xyz[:, 1]))
    pz = float(np.median(panel_inlier_xyz[:, 2]))

    nx, ny, nz = roof_plane.normal
    cx, cy, cz = roof_plane.centroid
    if abs(nz) < 1e-9:
        return float("nan")
    z_roof = cz - (nx * (px - cx) + ny * (py - cy)) / nz
    return pz - z_roof


def panel_roof_angle_deg(panel: PlaneFit, roof: PlaneFit) -> float:
    """Angle between the panel and roof normals, in degrees [0, 180].

    Used by the M5 mounting classifier (PRD §7.5: ``panel_roof_angle_deg``).
    Returns NaN if either fit is degenerate.
    """
    if (
        panel.n_inliers < 3 or roof.n_inliers < 3
        or np.isnan(panel.tilt_deg) or np.isnan(roof.tilt_deg)
    ):
        return float("nan")
    cos_a = float(np.clip(np.dot(panel.normal, roof.normal), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))
