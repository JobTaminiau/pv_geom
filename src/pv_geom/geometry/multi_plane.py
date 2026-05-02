"""Multi-plane detection (e.g. east-west racks) + tracker heuristic. M5 (PRD §7.2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon

from pv_geom.config import MultiPlaneConfig
from pv_geom.geometry.plane_fit import PlaneFit, fit_plane_ransac


@dataclass(frozen=True)
class MultiPlaneResult:
    primary: PlaneFit
    secondary: PlaneFit | None
    flags: tuple[str, ...]


def polygon_aspect_ratio(polygon: Polygon) -> float:
    """Long-to-short axis ratio of the minimum rotated rectangle around polygon.

    A square gives 1.0; a long thin tracker row gives a large number. Returns
    NaN for degenerate polygons (zero short axis or invalid geometry).
    """
    if polygon is None or polygon.is_empty:
        return float("nan")
    mrr = polygon.minimum_rotated_rectangle
    coords = np.asarray(mrr.exterior.coords)
    sides = np.linalg.norm(np.diff(coords, axis=0), axis=1)[:4]
    long_side = float(sides.max())
    short_side = float(sides.min())
    if short_side < 1e-9:
        return float("nan")
    return long_side / short_side


def detect_multi_plane(
    points: np.ndarray,
    primary: PlaneFit,
    cfg: MultiPlaneConfig,
    *,
    seed: int | None = None,
) -> MultiPlaneResult:
    """Look for a coherent secondary plane in the non-inliers of ``primary``.

    PRD §7.2. ``points`` must be the SAME (N, 3) array used to compute
    ``primary`` so we can index by ``primary.inlier_mask``. Returns the primary
    fit unchanged plus a secondary ``PlaneFit`` (or ``None``) and a ``flags``
    tuple. The only flag set here is ``"east_west_rack"`` when the two planes
    are roughly 180 deg apart in azimuth and have similar tilts.
    """
    if not cfg.enabled or primary.n_inliers < 10 or np.isnan(primary.tilt_deg):
        return MultiPlaneResult(primary=primary, secondary=None, flags=())

    pts = np.asarray(points, dtype=np.float64)
    if len(pts) != primary.n_total:
        # caller passed something other than the original point set — can't index
        return MultiPlaneResult(primary=primary, secondary=None, flags=())

    non_inliers = pts[~primary.inlier_mask]
    secondary_min = max(int(round(cfg.secondary_min_frac * primary.n_total)), 10)
    if len(non_inliers) < secondary_min:
        return MultiPlaneResult(primary=primary, secondary=None, flags=())

    secondary = fit_plane_ransac(
        non_inliers,
        ransac_threshold=0.10,
        min_inlier_frac=0.5,
        max_iter=200,
        seed=seed,
    )
    if secondary.n_inliers < secondary_min or np.isnan(secondary.tilt_deg):
        return MultiPlaneResult(primary=primary, secondary=None, flags=())

    flags: list[str] = []
    if _is_east_west_rack(primary, secondary, cfg):
        flags.append("east_west_rack")
    return MultiPlaneResult(primary=primary, secondary=secondary, flags=tuple(flags))


def _is_east_west_rack(
    primary: PlaneFit, secondary: PlaneFit, cfg: MultiPlaneConfig
) -> bool:
    if np.isnan(primary.azimuth_deg) or np.isnan(secondary.azimuth_deg):
        return False
    raw = (primary.azimuth_deg - secondary.azimuth_deg) % 360.0
    az_offset = abs(raw - 180.0)         # how far from "exactly opposite"
    az_close = az_offset < cfg.ew_rack_azimuth_tol_deg
    tilt_close = abs(primary.tilt_deg - secondary.tilt_deg) < cfg.ew_rack_tilt_tol_deg
    return bool(az_close and tilt_close)


def is_tracker_suspected(
    *,
    on_building: bool,
    aspect_ratio: float,
    height_above_ground_m: float,
    panel_tilt_deg: float,
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
    if np.isnan(aspect_ratio) or np.isnan(panel_tilt_deg) or np.isnan(height_above_ground_m):
        return False
    return (
        aspect_ratio >= aspect_min
        and height_above_ground_m < height_above_ground_max_m
        and panel_tilt_deg < tilt_max_deg
    )
