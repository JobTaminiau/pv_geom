"""Ring-buffer roof plane extraction. M4 (PRD §7.3)."""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from shapely import contains_xy
from shapely.geometry import Polygon
from shapely.ops import unary_union

from pv_geom.config import RoofPlaneConfig
from pv_geom.geometry.plane_fit import PlaneFit, fit_plane_ransac


@dataclass(frozen=True)
class RoofPlaneResult:
    """Outcome of one roof-plane extraction attempt.

    ``fit`` is the underlying ``PlaneFit`` (may itself signal NaN tilt on
    poor consensus). ``flag`` mirrors the per-row quality flag we'll emit:
    ``None`` on success, ``"roof_insufficient"`` if the ring never accumulated
    enough points within ``buffer_max_m``, ``"roof_complex"`` if the fit
    succeeded but post-RANSAC RMSE exceeded ``rmse_max_m``.
    """

    fit: PlaneFit | None
    on_building: bool
    building_id: str | None
    flag: str | None
    used_buffer_m: float | None


def _build_ring(
    pv_polygon: Polygon,
    footprint: Polygon,
    other_pv_union,
    buffer_m: float,
):
    """Buffer minus PV minus other PVs, intersected with the footprint."""
    ring = pv_polygon.buffer(buffer_m).difference(pv_polygon).intersection(footprint)
    if other_pv_union is not None and not other_pv_union.is_empty:
        ring = ring.difference(other_pv_union)
    return ring


def extract_roof_plane(
    pv_polygon: Polygon,
    building_footprints: gpd.GeoDataFrame,
    other_pv_polygons: gpd.GeoDataFrame,
    panel_class_points: np.ndarray,
    cfg: RoofPlaneConfig,
    *,
    building_id_col: str = "building_id",
    seed: int | None = None,
) -> RoofPlaneResult:
    """Fit the roof plane in a ring buffer around the PV polygon (PRD §7.3).

    Parameters
    ----------
    pv_polygon
        The single PV polygon (caller must explode MultiPolygons before this).
        Must share the CRS of ``panel_class_points`` and ``building_footprints``.
    building_footprints
        GeoDataFrame of footprints; only those intersecting ``pv_polygon`` matter.
    other_pv_polygons
        Other PV polygons that may sit on the same building; their geometry is
        subtracted from the ring so adjacent panel arrays don't pollute the fit.
    panel_class_points
        ``(N, 3)`` panel-class returns (pre-filtered by the caller — typically
        class 6, or class 1 above ground when class 6 is absent).
    cfg
        Roof-plane settings (buffer, min_points, RANSAC threshold, RMSE max).

    Returns
    -------
    RoofPlaneResult
    """
    pts = np.asarray(panel_class_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"panel_class_points must be (N, 3); got {pts.shape}")

    # 1) Find building footprint(s) intersecting the polygon.
    if len(building_footprints):
        idx = list(building_footprints.sindex.query(pv_polygon, predicate="intersects"))
        candidates = building_footprints.iloc[idx]
    else:
        candidates = building_footprints.iloc[0:0]

    if len(candidates) == 0:
        return RoofPlaneResult(
            fit=None, on_building=False, building_id=None,
            flag=None, used_buffer_m=None,
        )

    # If multiple footprints overlap, pick the one with maximum intersection area.
    if len(candidates) == 1:
        chosen = candidates.iloc[0]
    else:
        inter_areas = candidates.geometry.intersection(pv_polygon).area
        chosen = candidates.loc[inter_areas.idxmax()]
    footprint = chosen.geometry
    bid = (
        str(chosen[building_id_col])
        if building_id_col in candidates.columns and chosen[building_id_col] is not None
        else None
    )

    # 2-3) Iteratively expand the buffer until we have enough ring points.
    other_union = (
        unary_union(other_pv_polygons.geometry.tolist())
        if len(other_pv_polygons)
        else None
    )

    buf = float(cfg.buffer_m)
    ring = None
    points_in_ring: np.ndarray | None = None
    while True:
        ring = _build_ring(pv_polygon, footprint, other_union, buf)
        if not ring.is_empty:
            mask = contains_xy(ring, pts[:, 0], pts[:, 1])
            points_in_ring = pts[mask]
            if len(points_in_ring) >= cfg.min_points:
                break
        if buf >= cfg.buffer_max_m - 1e-9:
            break
        buf = min(buf + cfg.buffer_step_m, cfg.buffer_max_m)

    if (
        ring is None
        or ring.is_empty
        or points_in_ring is None
        or len(points_in_ring) < cfg.min_points
    ):
        return RoofPlaneResult(
            fit=None, on_building=True, building_id=bid,
            flag="roof_insufficient", used_buffer_m=buf,
        )

    # 4) RANSAC + LSQ fit.
    fit = fit_plane_ransac(
        points_in_ring,
        ransac_threshold=cfg.ransac_threshold_m,
        max_iter=200,
        seed=seed,
    )

    # 5) Reject if the post-fit inlier RMSE is too noisy (complex roof, dormer, etc.).
    if np.isnan(fit.tilt_deg) or fit.rmse > cfg.rmse_max_m:
        return RoofPlaneResult(
            fit=fit, on_building=True, building_id=bid,
            flag="roof_complex", used_buffer_m=buf,
        )

    return RoofPlaneResult(
        fit=fit, on_building=True, building_id=bid,
        flag=None, used_buffer_m=buf,
    )
