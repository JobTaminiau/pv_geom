"""Height-above-ground / height-above-roof helpers. M4 (PRD §7.4)."""

from __future__ import annotations

import numpy as np

from pv_geom.geometry.plane_fit import PlaneFit


def height_above_ground(
    panel_inlier_z: np.ndarray,
    ground_xyz: np.ndarray,
    polygon_centroid_xy: tuple[float, float],
    search_radius_m: float = 10.0,
) -> float:
    """Median panel-inlier z minus median class-2 z within ``search_radius_m``.

    PRD §7.4. Returns NaN when the panel inliers are empty or no ground points
    fall inside the search disk.
    """
    if len(panel_inlier_z) == 0 or len(ground_xyz) == 0:
        return float("nan")
    cx, cy = polygon_centroid_xy
    dx = ground_xyz[:, 0] - cx
    dy = ground_xyz[:, 1] - cy
    mask = (dx * dx + dy * dy) <= search_radius_m ** 2
    if not mask.any():
        return float("nan")
    return float(np.median(panel_inlier_z) - np.median(ground_xyz[mask, 2]))


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
