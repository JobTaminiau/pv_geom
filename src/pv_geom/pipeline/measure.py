"""Measure one polygon: the steps, and the record they produce.

Each step is a small function over explicit inputs so it can be tested and
reasoned about alone; :func:`measure_polygon` runs them in order and returns a
:class:`Measurement`. Turning a measurement into an output row is a separate
concern (:mod:`pv_geom.pipeline.rows`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import geopandas as gpd
import numpy as np
from shapely import contains_xy

from pv_geom.config import PanelPlaneConfig, PVGeomConfig
from pv_geom.geometry.heights import (
    height_above_ground,
    height_above_roof,
    panel_roof_angle_deg,
)
from pv_geom.geometry.multi_plane import detect_multi_plane, polygon_aspect_ratio
from pv_geom.geometry.plane_fit import (
    PlaneFit,
    bootstrap_uncertainty,
    failed_fit,
    fit_plane_ransac,
)
from pv_geom.geometry.roof_plane import RoofPlaneResult, extract_roof_plane
from pv_geom.schema import MEASURED, NO_FIT
from pv_geom.vintage import geometry_basis, vintage_gap_days

NAN = float("nan")


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PolygonTask:
    """One polygon to measure and what is known about it up front."""

    polygon: Any                              # shapely Polygon, in the run CRS
    polygon_id: str
    parent_polygon_id: str | None = None      # defaults to polygon_id
    input_row: int = 0
    polygon_vintage: date | None = None       # imagery capture date
    input_flags: tuple[str, ...] = ()         # quality flags settled when the layer was read
    # True-north bearing of grid north at the polygon (see pv_geom.utils.north).
    # Fits are made in grid coordinates; this turns their azimuths into true ones.
    grid_convergence_deg: float = 0.0

    @property
    def parent_id(self) -> str:
        return self.polygon_id if self.parent_polygon_id is None else self.parent_polygon_id


@dataclass(frozen=True)
class LocalPoints:
    """The LiDAR returns and context layers around one polygon."""

    panel: np.ndarray                         # (N, 3) candidates inside the eroded polygon
    ground: np.ndarray                        # (G, 3) ground returns within the search radius
    surroundings: np.ndarray                  # (P, 3) candidates around the polygon (roof ring source)
    footprints: gpd.GeoDataFrame | None       # None = the run has no footprint layer
    other_polygons: gpd.GeoDataFrame          # neighbouring PV polygons, kept out of the ring


def seed_for_polygon(polygon_id: str) -> int:
    """Stable RNG seed derived from the polygon id.

    Must not use the builtin ``hash()``: string hashing is salted per process
    (PYTHONHASHSEED), so every Dask worker would draw different RANSAC and
    bootstrap seeds and runs would not be reproducible.
    """
    digest = hashlib.sha256(polygon_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PanelFit:
    """The accepted plane fit, the inlier tolerance it used, and whether that
    tolerance had to be widened beyond the configured base."""

    fit: PlaneFit
    tolerance_m: float
    widened: bool = False

    @property
    def ok(self) -> bool:
        return not np.isnan(self.fit.tilt_deg)


def fit_panel(points: np.ndarray, cfg: PanelPlaneConfig, seed: int) -> PanelFit:
    """Fit the plane inside the polygon, widening the tolerance if the returns
    are too noisy for the base one.

    No consensus at the base tolerance may only mean the returns scatter more
    than it assumes (Delaware: ~7.5 cm about a roof plane, against ~2 cm in
    Phoenix). In that case their scatter about the best plane in the widest
    allowed band is measured and the fit repeated at twice that.
    """
    def _fit_at(tol: float) -> PlaneFit:
        return fit_plane_ransac(
            points, ransac_threshold=tol, min_inlier_frac=cfg.min_inlier_frac,
            max_iter=cfg.max_iter, tilt_floor_deg=cfg.tilt_floor_deg, seed=seed,
        )

    base = cfg.ransac_threshold_m
    fit = _fit_at(base) if len(points) >= 3 else failed_fit(len(points))
    cap = cfg.ransac_threshold_max_m
    if not np.isnan(fit.tilt_deg) or cap <= base or len(points) < cfg.min_points:
        return PanelFit(fit, base)

    wide = _fit_at(cap)
    if np.isnan(wide.tilt_deg):
        return PanelFit(fit, base)
    resid = (points[wide.inlier_mask] - wide.centroid) @ wide.normal
    sigma = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    tol = float(np.clip(2.0 * sigma, base, cap))
    refit = _fit_at(tol) if tol < cap else wide
    if np.isnan(refit.tilt_deg):
        return PanelFit(wide, cap, widened=True)
    return PanelFit(refit, tol, widened=True)


def roof_reference(task: PolygonTask, pts: LocalPoints, cfg: PVGeomConfig,
                   seed: int) -> RoofPlaneResult:
    """The plane of the roof around the polygon, or a result saying why not."""
    if not cfg.roof_plane.enabled:
        return RoofPlaneResult(
            fit=None, on_building=None if pts.footprints is None else False,
            building_id=None, flag=None, used_buffer_m=None,
        )
    return extract_roof_plane(
        task.polygon, pts.footprints, pts.other_polygons, pts.surroundings,
        cfg.roof_plane, seed=seed,
    )


@dataclass(frozen=True)
class StandoffScreen:
    """Outcome of testing whether the fitted plane stands above the roof.

    Three states, and a row must say which: *passed* (panels were physically
    in the point cloud), *failed* (the plane is not resolvably above the roof —
    a low-profile flush mount, or a polygon whose panels postdate the LiDAR, in
    which case the "panel" plane is the bare roof), or *not screened* (no
    trustworthy roof reference to measure against, which says nothing either
    way).
    """

    screened: bool
    passed: bool

    @property
    def flag(self) -> str | None:
        if not self.screened:
            return "standoff_unscreenable"
        return None if self.passed else "no_panel_standoff"


def screen_standoff(height_above_roof_m: float, roof_usable: bool,
                    min_standoff_m: float) -> StandoffScreen:
    screened = roof_usable and not np.isnan(height_above_roof_m)
    return StandoffScreen(screened, screened and height_above_roof_m >= min_standoff_m)


# Ground returns inside a polygon that mark it as lying at ground level.
_GROUND_LEVEL_MIN_RETURNS = 5


def why_no_fit(task: PolygonTask, pts: LocalPoints, cfg: PanelPlaneConfig) -> str:
    """Name the reason a polygon has no fit.

    ``ground_level_only``: nothing stands above the ground there — the polygon
    holds ground returns but (almost) no candidates, the signature of a
    ground-level false detection or a misplaced polygon. ``too_few_points``:
    something is there, but too little of it. ``no_consensus``: plenty of
    returns, no single plane.
    """
    n = len(pts.panel)
    if n < 3 and len(pts.ground):
        poly = task.polygon
        minx, miny, maxx, maxy = poly.bounds
        g = pts.ground
        near = g[(g[:, 0] >= minx) & (g[:, 0] <= maxx) & (g[:, 1] >= miny) & (g[:, 1] <= maxy)]
        if len(near) and int(contains_xy(poly, near[:, 0], near[:, 1]).sum()) >= _GROUND_LEVEL_MIN_RETURNS:
            return "ground_level_only"
    return "too_few_points" if n < cfg.min_points else "no_consensus"


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #

@dataclass
class Measurement:
    """Everything measured for one polygon."""

    task: PolygonTask
    area_m2: float
    aspect_ratio: float
    point_density: float
    panel: PanelFit
    tilt_unc_deg: float
    azimuth_unc_deg: float
    secondary: PlaneFit | None
    roof: RoofPlaneResult
    height_above_roof_m: float
    panel_roof_angle_deg: float
    height_above_ground_m: float
    screen: StandoffScreen
    lidar_date: date | None
    lidar_date_source: str | None
    gap_days: int | None
    basis: str
    fit_failure: str | None = None            # why there is no fit, when there is none
    flags: list[str] = field(default_factory=list)
    # Heights of the returns the panel plane was fitted to (kept for the
    # experimental mounting classifier, which looks at what lies beneath them).
    panel_z: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)

    @property
    def fit_ok(self) -> bool:
        return self.panel.ok

    @property
    def status(self) -> str:
        return MEASURED if self.panel.ok else NO_FIT

    @property
    def surface_area_m2(self) -> float | None:
        """Area along the fitted plane, or None without a usable fit."""
        tilt = self.panel.fit.tilt_deg
        if not self.fit_ok or tilt >= 89.0:
            return None
        return self.area_m2 / float(np.cos(np.radians(tilt)))


def measure_polygon(
    task: PolygonTask,
    pts: LocalPoints,
    cfg: PVGeomConfig,
    *,
    lidar_date: date | None = None,
    lidar_date_source: str | None = None,
) -> Measurement:
    """Run every measurement step for one polygon."""
    flags: list[str] = list(task.input_flags)
    polygon = task.polygon
    area_m2 = float(polygon.area)
    centroid = polygon.centroid
    seed = seed_for_polygon(task.polygon_id)

    density = len(pts.panel) / area_m2 if area_m2 > 0 else 0.0
    if density < cfg.panel_plane.min_density_pts_per_m2 or len(pts.panel) < cfg.panel_plane.min_points:
        flags.append("low_density")

    # Panel plane (attempted even at low density; the flag says so).
    panel = fit_panel(pts.panel, cfg.panel_plane, seed)
    fit = panel.fit
    if panel.widened:
        flags.append("wide_tolerance_fit")
    if not panel.ok:
        flags.append("poor_fit")
    elif np.isnan(fit.azimuth_deg):
        flags.append("near_horizontal")

    if cfg.panel_plane.uncertainty_method == "bootstrap" and fit.n_inliers >= 3:
        tilt_unc, az_unc = bootstrap_uncertainty(
            pts.panel, fit, n_samples=cfg.panel_plane.bootstrap_samples, seed=seed)
    else:
        tilt_unc, az_unc = NAN, NAN

    secondary = None
    if cfg.multi_plane.enabled and fit.n_inliers >= 10 and panel.ok:
        multi = detect_multi_plane(pts.panel, fit, cfg.multi_plane, seed=seed)
        flags.extend(multi.flags)
        secondary = multi.secondary

    # Roof reference. A flagged result still reports its fit for QC, but only
    # an unflagged one is precise enough to measure a panel against.
    roof = roof_reference(task, pts, cfg, seed)
    if roof.flag:
        flags.append(roof.flag)

    inliers = pts.panel[fit.inlier_mask] if fit.n_inliers > 0 else pts.panel[:0]
    panel_z = inliers[:, 2] if len(inliers) else pts.panel[:, 2]
    hag = height_above_ground(
        panel_z, pts.ground, (float(centroid.x), float(centroid.y)),
        cfg.heights.ground_search_radius_m,
        fallback_k=cfg.heights.ground_fallback_k,
        fallback_max_radius_m=cfg.heights.ground_fallback_max_radius_m,
    )
    if roof.usable and roof.fit is not None:
        har = height_above_roof(inliers, roof.fit) if len(inliers) else NAN
        angle = panel_roof_angle_deg(fit, roof.fit)
    else:
        har, angle = NAN, NAN

    screen = screen_standoff(har, roof.usable, cfg.heights.min_panel_standoff_m)
    if screen.flag:
        flags.append(screen.flag)

    # What the tilt and azimuth rest on: the standoff screen is direct evidence
    # that panels were in the cloud; failing that, the two input dates decide.
    gap_days = vintage_gap_days(task.polygon_vintage, lidar_date)
    basis = geometry_basis(
        fit_ok=panel.ok, standoff_passed=screen.passed,
        standoff_screened=screen.screened, gap_days=gap_days,
    )

    return Measurement(
        task=task, area_m2=area_m2, aspect_ratio=polygon_aspect_ratio(polygon),
        point_density=density, panel=panel, tilt_unc_deg=tilt_unc, azimuth_unc_deg=az_unc,
        secondary=secondary, roof=roof, height_above_roof_m=har,
        panel_roof_angle_deg=angle, height_above_ground_m=hag, screen=screen,
        lidar_date=lidar_date, lidar_date_source=lidar_date_source, gap_days=gap_days,
        basis=basis, fit_failure=None if panel.ok else why_no_fit(task, pts, cfg.panel_plane),
        flags=flags, panel_z=panel_z,
    )
