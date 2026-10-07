"""Facets within one polygon.

A polygon is one detected array outline, but not always one plane: a detection
can run over a ridge, a dissolved cluster can merge the arrays on two faces of a
roof, and an east-west rack is two planes by design. Fitting a single plane to
such a polygon either fails (no plane holds enough of the returns) or reports
one face and silently drops the other.

Here the returns are split into **segments**: planes that are each well
supported and distinct in orientation from one another. A noisy single surface
does not qualify — sequential fitting would slice it into parallel layers, and
those are the same facet.
"""

from __future__ import annotations

import numpy as np

from pv_geom.config import MultiPlaneConfig, PanelPlaneConfig
from pv_geom.geometry.plane_fit import PlaneFit, fit_planes_sequential


def angle_between_deg(a: PlaneFit, b: PlaneFit) -> float:
    """Angle between two planes, in degrees (0 = parallel)."""
    cos = abs(float(np.clip(np.dot(a.normal, b.normal), -1.0, 1.0)))
    return float(np.degrees(np.arccos(cos)))


def _distinct(candidates: list[PlaneFit], kept: list[PlaneFit],
              multi: MultiPlaneConfig) -> list[PlaneFit]:
    """``kept`` extended by the candidates that could be array facets — not too
    steep — and differ in orientation from every plane already kept."""
    out = list(kept)
    for plane in candidates:
        if len(out) >= multi.max_segments:
            break
        if plane.tilt_deg > multi.segment_max_tilt_deg:
            continue
        if all(angle_between_deg(plane, k) >= multi.segment_min_angle_deg for k in out):
            out.append(plane)
    return out


def _min_support(n_points: int, panel: PanelPlaneConfig, multi: MultiPlaneConfig) -> int:
    """Returns a plane needs to count as a facet: a share of the polygon's
    returns, and never fewer than a robust fit needs."""
    return max(int(panel.min_points), round(multi.secondary_min_frac * n_points))


def split_into_facets(points: np.ndarray, panel: PanelPlaneConfig, multi: MultiPlaneConfig,
                      *, tolerance_m: float, seed: int) -> list[PlaneFit] | None:
    """Try to explain a polygon with no dominant plane as several facets.

    Returns the facets, largest first, when at least two distinct planes
    together hold ``panel.min_inlier_frac`` of the returns — the same share a
    single plane would need. Returns None otherwise.
    """
    n = len(points)
    if not multi.enabled or multi.max_segments < 2 or n < 2 * panel.min_points:
        return None
    planes = fit_planes_sequential(
        points, ransac_threshold=tolerance_m, min_points=_min_support(n, panel, multi),
        max_planes=multi.max_segments, max_iter=panel.max_iter,
        tilt_floor_deg=panel.tilt_floor_deg, seed=seed,
    )
    facets = _distinct(planes, [], multi)
    if len(facets) < 2:
        return None
    if sum(f.n_inliers for f in facets) < panel.min_inlier_frac * n:
        return None
    return sorted(facets, key=lambda f: -f.n_inliers)


def further_facets(points: np.ndarray, primary: PlaneFit, panel: PanelPlaneConfig,
                   multi: MultiPlaneConfig, *, tolerance_m: float, seed: int) -> list[PlaneFit]:
    """Facets beside an accepted primary plane: distinct, well-supported planes
    among the returns the primary does not explain. The primary comes first."""
    if not multi.enabled or multi.max_segments < 2 or np.isnan(primary.tilt_deg):
        return [primary]
    n = len(points)
    rest_idx = np.flatnonzero(~primary.inlier_mask)
    support = _min_support(n, panel, multi)
    if len(rest_idx) < support:
        return [primary]
    planes = fit_planes_sequential(
        points[rest_idx], ransac_threshold=tolerance_m, min_points=support,
        max_planes=multi.max_segments - 1, max_iter=panel.max_iter,
        tilt_floor_deg=panel.tilt_floor_deg, seed=seed + 1,
    )
    lifted = []
    for p in planes:                       # masks back onto the full point set
        mask = np.zeros(n, dtype=bool)
        mask[rest_idx[p.inlier_mask]] = True
        lifted.append(PlaneFit(
            normal=p.normal, centroid=p.centroid, tilt_deg=p.tilt_deg,
            azimuth_deg=p.azimuth_deg, rmse=p.rmse, n_inliers=p.n_inliers,
            n_total=n, inlier_mask=mask,
        ))
    return _distinct(lifted, [primary], multi)


def is_east_west_pair(a: PlaneFit, b: PlaneFit, multi: MultiPlaneConfig) -> bool:
    """Two facets facing about 180 degrees apart at similar tilt: the signature
    of an east-west rack (and, on a pitched roof, of a gable)."""
    if np.isnan(a.azimuth_deg) or np.isnan(b.azimuth_deg):
        return False
    opposite = abs((a.azimuth_deg - b.azimuth_deg) % 360.0 - 180.0)
    return bool(opposite < multi.ew_rack_azimuth_tol_deg
                and abs(a.tilt_deg - b.tilt_deg) < multi.ew_rack_tilt_tol_deg)
