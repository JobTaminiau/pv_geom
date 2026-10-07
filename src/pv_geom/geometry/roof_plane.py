"""Ring-buffer roof plane extraction. M4 (PRD §7.3)."""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from shapely import contains_xy
from shapely.geometry import Polygon
from shapely.ops import unary_union

from pv_geom.config import RoofPlaneConfig
from pv_geom.geometry.plane_fit import PlaneFit, fit_plane_ransac, fit_planes_sequential


@dataclass(frozen=True)
class RoofPlaneResult:
    """Outcome of one roof-plane extraction attempt.

    ``fit`` is the underlying ``PlaneFit`` (may itself signal NaN tilt on
    poor consensus). ``flag`` mirrors the per-row quality flag we'll emit:

    - ``None`` on success;
    - ``"roof_insufficient"`` if the ring never accumulated enough points
      within ``buffer_max_m``;
    - ``"roof_no_consensus"`` if RANSAC found no plane holding
      ``min_inlier_frac`` of the ring (tilt is NaN — the ring spans several
      surfaces, so there is no single roof plane to report);
    - ``"roof_complex"`` if a plane *was* agreed on but its post-fit inlier
      RMSE exceeded ``rmse_max_m`` (a genuinely rough or cluttered roof).

    The two failure modes were one flag before 0.4.0, which made a consensus
    failure indistinguishable from a quality failure and hid the fact that
    nearly all of them were well within the RMSE gate.
    """

    fit: PlaneFit | None
    on_building: bool | None            # None = no footprint layer to test against
    building_id: str | None
    flag: str | None
    used_buffer_m: float | None
    # How the plane was chosen within the ring (DOMINANT, COLLAR_FACET,
    # PARALLEL_FACET, COLLAR_ONLY), or None without a usable fit.
    method: str | None = None
    # How the reference ring was built: "footprint_ring" (clipped to the
    # overlapped building footprint), "open_ring" (no footprint available, so
    # just the elevated returns around the array) or "none" (no fit attempted
    # or, for an open ring, none that held up).
    source: str = "none"

    @property
    def usable(self) -> bool:
        """True when the fit is a trustworthy reference for panel-vs-roof work.

        A flagged result still carries its ``fit`` (it is informative for QC and
        is written to the ``roof_*`` columns), but it must not feed
        ``height_above_roof_m``, ``panel_roof_angle_deg``, the mounting rules,
        or the panel-standoff screen: a ``roof_complex`` fit has RMSE above
        ``rmse_max_m``, which is coarser than the 5 cm standoff those consumers
        need to resolve.
        """
        return self.fit is not None and self.flag is None and not np.isnan(self.fit.tilt_deg)


def _build_ring(
    pv_polygon: Polygon,
    footprint: Polygon | None,
    other_pv_union,
    buffer_m: float,
):
    """Buffer minus PV minus other PVs, intersected with the footprint (when
    there is one — an open ring is left unclipped)."""
    ring = pv_polygon.buffer(buffer_m).difference(pv_polygon)
    if footprint is not None:
        ring = ring.intersection(footprint)
    if other_pv_union is not None and not other_pv_union.is_empty:
        ring = ring.difference(other_pv_union)
    return ring


def _collar_agreement(fit: PlaneFit, collar_pts: np.ndarray, threshold: float) -> float:
    """Share of the collar that lies on the plane."""
    if len(collar_pts) == 0:
        return 0.0
    return float((np.abs((collar_pts - fit.centroid) @ fit.normal) < threshold).mean())


# How a reference plane was chosen (the `roof_ref_method` column).
DOMINANT = "dominant_plane"          # the ring's main plane, and the collar agrees
COLLAR_FACET = "collar_facet"        # the ring facet a clear majority of the collar lies on
COLLAR_ONLY = "collar_only"          # a plane fitted to the collar alone
PARALLEL_FACET = "panel_parallel_facet"   # the ring facet parallel to the array plane


def _reference_plane(
    pv_polygon: Polygon,
    points_in_ring: np.ndarray,
    cfg: RoofPlaneConfig,
    *,
    panel_normal: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[PlaneFit, str | None]:
    """The plane of the roof the array rests on, and how it was chosen.

    The *collar* — ring points within ``cfg.collar_m`` of the array — is the
    authority on which facet that is. Attempts, in order:

    1. **The ring's dominant plane**, if it holds ``min_inlier_frac`` of the
       ring and the collar agrees with it. The simple case: one facet.
    2. **Facet search.** The ring spans a ridge or a hip, so either no plane
       dominates or the dominant one is the facet *across* the ridge (RANSAC
       reports whichever has more points, which for an array set near one edge
       is the other one — measuring a panel against that is worse than not
       measuring it). The ring's planes are peeled off one at a time and the
       one a clear majority (``facet_agreement_min``) of the collar lies on is
       taken.
    3. **The facet parallel to the array plane**, when the collar is genuinely
       split (an array at a ridge or hip, a polygon covering two facets) and
       ``panel_normal`` is known: the ring facet within
       ``parallel_angle_max_deg`` of the array's plane that also holds
       ``parallel_collar_min`` of the collar.
    4. **The collar alone**.

    Returns a fit with NaN tilt (and no method) when none succeeds, which the
    caller reports as ``roof_no_consensus``.
    """
    def _fit(pts: np.ndarray, floor: float) -> PlaneFit:
        return fit_plane_ransac(
            pts, ransac_threshold=cfg.ransac_threshold_m, min_inlier_frac=floor,
            max_iter=200, seed=seed,
        )

    def _done(fit: PlaneFit, method: str) -> tuple[PlaneFit, str | None]:
        return fit, (None if np.isnan(fit.tilt_deg) else method)

    ring_fit = _fit(points_in_ring, cfg.min_inlier_frac)
    collar_zone = pv_polygon.buffer(cfg.collar_m)
    collar_pts = points_in_ring[
        contains_xy(collar_zone, points_in_ring[:, 0], points_in_ring[:, 1])]
    if len(collar_pts) < cfg.collar_min_points:
        return _done(ring_fit, DOMINANT)        # no collar to consult

    ring_ok = not np.isnan(ring_fit.tilt_deg)
    if ring_ok and (_collar_agreement(ring_fit, collar_pts, cfg.ransac_threshold_m)
                    >= cfg.collar_agreement_min):
        return ring_fit, DOMINANT

    facets: list[PlaneFit] = []
    scores: list[float] = []
    if cfg.facet_search:
        facets = fit_planes_sequential(
            points_in_ring, ransac_threshold=cfg.ransac_threshold_m,
            min_points=cfg.collar_min_points, max_planes=cfg.max_facets, seed=seed,
        )
        scores = [_collar_agreement(f, collar_pts, cfg.ransac_threshold_m) for f in facets]
        if facets and max(scores) >= cfg.facet_agreement_min:
            return facets[int(np.argmax(scores))], COLLAR_FACET

    # The ring facet parallel to the array plane that also touches the array.
    parallel: PlaneFit | None = None
    if panel_normal is not None and cfg.parallel_angle_max_deg > 0:
        cos_max = np.cos(np.radians(cfg.parallel_angle_max_deg))
        touching = [
            (score, f) for score, f in zip(scores, facets, strict=True)
            if score >= cfg.parallel_collar_min
            and abs(float(np.dot(f.normal, panel_normal))) >= cos_max
        ]
        if touching:
            parallel = max(touching, key=lambda sf: sf[0])[1]

    if ring_ok:
        # The ring has a dominant plane, just not the collar's. The collar is
        # refitted at the ordinary consensus floor, as it always was, and that
        # result stands — unless it is grossly unlike the array's plane while a
        # parallel facet is at hand, which is the signature of the collar fit
        # having landed across the ridge.
        collar_fit = _fit(collar_pts, cfg.min_inlier_frac)
        if np.isnan(collar_fit.tilt_deg):
            return (parallel, PARALLEL_FACET) if parallel is not None else (collar_fit, None)
        if parallel is not None and panel_normal is not None:
            gross = np.cos(np.radians(2.0 * cfg.parallel_angle_max_deg))
            if abs(float(np.dot(collar_fit.normal, panel_normal))) < gross:
                return parallel, PARALLEL_FACET
        return collar_fit, COLLAR_ONLY

    # The ring has no dominant plane: new ground. The parallel facet first; else
    # the collar alone, which must show the clear majority a searched facet needs.
    if parallel is not None:
        return parallel, PARALLEL_FACET
    return _done(_fit(collar_pts, max(cfg.min_inlier_frac, cfg.facet_agreement_min)),
                 COLLAR_ONLY)


def _overlapped_footprint(
    pv_polygon: Polygon,
    building_footprints: gpd.GeoDataFrame | None,
    min_overlap_frac: float,
    building_id_col: str,
) -> tuple[Polygon | None, str | None]:
    """The footprint the polygon sits on and its id, or ``(None, None)``.

    Among footprints the polygon intersects, the one with the largest overlap
    is taken — and only if it covers enough of the polygon. A sliver touch is
    not "on the building": an edge-clipping ground mount or a carport beside a
    wall must not be treated as rooftop.
    """
    if building_footprints is None or not len(building_footprints):
        return None, None
    idx = list(building_footprints.sindex.query(pv_polygon, predicate="intersects"))
    candidates = building_footprints.iloc[idx]
    if not len(candidates):
        return None, None
    if len(candidates) == 1:
        chosen = candidates.iloc[0]
    else:
        chosen = candidates.loc[candidates.geometry.intersection(pv_polygon).area.idxmax()]
    poly_area = pv_polygon.area
    overlap = chosen.geometry.intersection(pv_polygon).area / poly_area if poly_area > 0 else 0.0
    if overlap < min_overlap_frac:
        return None, None
    has_id = building_id_col in candidates.columns and chosen[building_id_col] is not None
    return chosen.geometry, str(chosen[building_id_col]) if has_id else None


def _ring_points(
    pv_polygon: Polygon,
    footprint: Polygon | None,
    other_pv_polygons: gpd.GeoDataFrame,
    pts: np.ndarray,
    cfg: RoofPlaneConfig,
) -> tuple[np.ndarray, float]:
    """Returns in the ring around the polygon, and the buffer that gathered them.

    The ring starts at ``buffer_m`` and widens in ``buffer_step_m`` steps up to
    ``buffer_max_m`` until it holds ``min_points``; the caller checks whether it
    ever did.
    """
    other_union = (
        unary_union(other_pv_polygons.geometry.tolist()) if len(other_pv_polygons) else None
    )
    buf = float(cfg.buffer_m)
    in_ring = pts[:0]
    while True:
        ring = _build_ring(pv_polygon, footprint, other_union, buf)
        if not ring.is_empty:
            in_ring = pts[contains_xy(ring, pts[:, 0], pts[:, 1])]
            if len(in_ring) >= cfg.min_points:
                break
        if buf >= cfg.buffer_max_m - 1e-9:
            break
        buf = min(buf + cfg.buffer_step_m, cfg.buffer_max_m)
    return in_ring, buf


def extract_roof_plane(
    pv_polygon: Polygon,
    building_footprints: gpd.GeoDataFrame | None,
    other_pv_polygons: gpd.GeoDataFrame,
    panel_class_points: np.ndarray,
    cfg: RoofPlaneConfig,
    *,
    building_id_col: str = "building_id",
    panel_normal: np.ndarray | None = None,
    seed: int | None = None,
) -> RoofPlaneResult:
    """Fit the roof plane in a ring around the PV polygon.

    Parameters
    ----------
    pv_polygon
        The single PV polygon (caller must explode MultiPolygons before this).
        Must share the CRS of ``panel_class_points`` and ``building_footprints``.
    building_footprints
        GeoDataFrame of footprints; only those intersecting ``pv_polygon`` matter.
        ``None`` means the run has no footprint layer: ``on_building`` is then
        unknown (None) and the reference comes from an open ring.
    other_pv_polygons
        Other PV polygons that may sit on the same building; their geometry is
        subtracted from the ring so adjacent panel arrays don't pollute the fit.
    panel_class_points
        ``(N, 3)`` panel-candidate returns around the polygon.
    cfg
        Roof-plane settings (buffer, min_points, RANSAC threshold, RMSE max).
    panel_normal
        Unit normal of the plane fitted inside the polygon, when there is one.
        Used only as a last resort, to pick the parallel facet where the collar
        is split between facets.
    """
    pts = np.asarray(panel_class_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"panel_class_points must be (N, 3); got {pts.shape}")

    footprint, bid = _overlapped_footprint(
        pv_polygon, building_footprints, cfg.min_overlap_frac, building_id_col)
    on_building: bool | None = None if building_footprints is None else footprint is not None
    open_ring = footprint is None
    source = "open_ring" if open_ring else "footprint_ring"
    no_reference = RoofPlaneResult(
        fit=None, on_building=on_building, building_id=None, flag=None, used_buffer_m=None)
    if open_ring and not cfg.open_ring:
        return no_reference

    def _rejected(flag: str, fit: PlaneFit | None, buf: float) -> RoofPlaneResult:
        # An open ring that yields nothing usable is the normal case for a
        # ground mount, not a roof problem: no reference, rather than a flag.
        if open_ring:
            return no_reference
        return RoofPlaneResult(fit=fit, on_building=True, building_id=bid, flag=flag,
                               used_buffer_m=buf, source=source)

    in_ring, buf = _ring_points(pv_polygon, footprint, other_pv_polygons, pts, cfg)
    if len(in_ring) < cfg.min_points:
        return _rejected("roof_insufficient", None, buf)

    # The consensus floor is the ring-specific `min_inlier_frac`, not the
    # panel-tuned default — see RoofPlaneConfig.min_inlier_frac. The collar
    # guard then makes sure the facet fitted is the one the array is on.
    fit, method = _reference_plane(pv_polygon, in_ring, cfg, panel_normal=panel_normal,
                                   seed=seed)

    # No plane held enough of the ring: there is no single roof surface to
    # measure against (and the RMSE describes only the minority that agreed).
    if np.isnan(fit.tilt_deg):
        return _rejected("roof_no_consensus", fit, buf)
    # A plane was agreed on but is too rough (uneven roof, clutter).
    if fit.rmse > cfg.rmse_max_m:
        return _rejected("roof_complex", fit, buf)

    return RoofPlaneResult(fit=fit, on_building=on_building, building_id=bid, flag=None,
                           used_buffer_m=buf, source=source, method=method)
