"""RANSAC + LSQ plane fitting. Tilt/azimuth from the unit normal. M3 (PRD §7.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlaneFit:
    normal: np.ndarray          # unit vector, length 3, pointing into upper half-space (nz >= 0)
    centroid: np.ndarray        # length 3
    tilt_deg: float             # 0 = horizontal; NaN on RANSAC failure
    azimuth_deg: float          # 0 = N, clockwise; NaN if tilt < tilt_floor or on failure
    rmse: float                 # m, perpendicular RMSE on inliers; NaN on failure
    n_inliers: int
    n_total: int
    inlier_mask: np.ndarray     # bool, length n_total


def tilt_azimuth_from_normal(
    normal: np.ndarray, tilt_floor_deg: float = 1.0
) -> tuple[float, float]:
    """Return (tilt_deg, azimuth_deg) from a unit normal in (x=East, y=North, z=Up).

    Tilt is the angle between the normal and +z (0 = horizontal).
    Azimuth (compass) is the direction the panel faces — the horizontal
    projection of the outward normal — measured 0=N clockwise to 360.
    Below ``tilt_floor_deg`` the azimuth is undefined and returned as NaN.
    """
    nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
    if nz < 0:                              # canonicalize: outward normal points up
        nx, ny, nz = -nx, -ny, -nz
    nz_clip = min(max(nz, -1.0), 1.0)
    tilt_deg = float(np.degrees(np.arccos(nz_clip)))
    if tilt_deg < tilt_floor_deg:
        return tilt_deg, float("nan")
    azimuth_deg = float(np.degrees(np.arctan2(nx, ny))) % 360.0
    return tilt_deg, azimuth_deg


def _failed_fit(n_total: int) -> PlaneFit:
    return PlaneFit(
        normal=np.array([0.0, 0.0, 1.0]),
        centroid=np.zeros(3),
        tilt_deg=float("nan"),
        azimuth_deg=float("nan"),
        rmse=float("nan"),
        n_inliers=0,
        n_total=int(n_total),
        inlier_mask=np.zeros(int(n_total), dtype=bool),
    )


def _pca_normal(centered: np.ndarray) -> np.ndarray:
    """Smallest-variance direction (unit normal). ``centered`` already mean-removed."""
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    n = vt[-1]
    if n[2] < 0:
        n = -n
    return n


def fit_plane_ransac(
    points: np.ndarray,
    ransac_threshold: float = 0.05,
    min_inlier_frac: float = 0.6,
    max_iter: int = 200,
    refine_with_lsq: bool = True,
    tilt_floor_deg: float = 1.0,
    seed: int | None = None,
) -> PlaneFit:
    """Fit a plane via RANSAC, refine on inliers via PCA-LSQ. PRD §7.1.

    Returns a ``PlaneFit``. On failure (too few points, no consensus, or inlier
    fraction below ``min_inlier_frac``), tilt/azimuth/rmse are NaN and
    ``n_inliers`` reflects what was actually found. Caller decides which
    quality flag to set.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3); got shape {pts.shape}")
    n = len(pts)
    if n < 3:
        return _failed_fit(n)

    rng = np.random.default_rng(seed)
    best_inliers: np.ndarray | None = None
    best_count = 0

    for _ in range(int(max_iter)):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = pts[idx]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        nlen = np.linalg.norm(normal)
        if nlen < 1e-9:                     # collinear sample, skip
            continue
        normal = normal / nlen
        dists = np.abs((pts - p0) @ normal)
        in_mask = dists < ransac_threshold
        cnt = int(in_mask.sum())
        if cnt > best_count:
            best_count = cnt
            best_inliers = in_mask

    if best_inliers is None or best_count < 3:
        return _failed_fit(n)

    # Refine plane on inliers via PCA, then re-classify inliers and recompute RMSE.
    inliers = pts[best_inliers]
    if refine_with_lsq:
        centroid = inliers.mean(axis=0)
        centered = inliers - centroid
        normal = _pca_normal(centered)
        dists = np.abs((pts - centroid) @ normal)
        best_inliers = dists < ransac_threshold
        if best_inliers.sum() < 3:
            return _failed_fit(n)
        inliers = pts[best_inliers]
        residuals = (inliers - centroid) @ normal
    else:
        centroid = inliers.mean(axis=0)
        residuals = (inliers - centroid) @ normal

    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    n_inliers = int(best_inliers.sum())

    if n_inliers / n < min_inlier_frac:
        # Plane found but consensus too weak; return geometry (informative for QC)
        # while signalling failure via NaN tilt/azimuth.
        return PlaneFit(
            normal=normal,
            centroid=centroid,
            tilt_deg=float("nan"),
            azimuth_deg=float("nan"),
            rmse=rmse,
            n_inliers=n_inliers,
            n_total=n,
            inlier_mask=best_inliers,
        )

    tilt_deg, azimuth_deg = tilt_azimuth_from_normal(normal, tilt_floor_deg)
    return PlaneFit(
        normal=normal,
        centroid=centroid,
        tilt_deg=tilt_deg,
        azimuth_deg=azimuth_deg,
        rmse=rmse,
        n_inliers=n_inliers,
        n_total=n,
        inlier_mask=best_inliers,
    )


def bootstrap_uncertainty(
    points: np.ndarray,
    fit: PlaneFit,
    n_samples: int = 50,
    seed: int | None = None,
) -> tuple[float, float]:
    """1-σ uncertainty in (tilt, azimuth) from non-parametric bootstrap on inliers.

    Sampling with replacement from the RANSAC inlier set, refits via PCA-LSQ,
    and returns the std of tilt and the circular std (Mardia) of azimuth in
    degrees. Both are NaN if the fit had no inliers; azimuth std is NaN if
    every bootstrap landed below the tilt floor.
    """
    if fit.n_inliers < 3:
        return float("nan"), float("nan")

    pts = np.asarray(points, dtype=np.float64)
    inliers = pts[fit.inlier_mask]
    n = len(inliers)
    rng = np.random.default_rng(seed)

    tilts: list[float] = []
    sin_az: list[float] = []
    cos_az: list[float] = []
    for _ in range(int(n_samples)):
        idx = rng.integers(0, n, size=n)
        sample = inliers[idx]
        centred = sample - sample.mean(axis=0)
        try:
            normal = _pca_normal(centred)
        except np.linalg.LinAlgError:
            continue
        tilt, az = tilt_azimuth_from_normal(normal, tilt_floor_deg=0.0)
        tilts.append(tilt)
        if not np.isnan(az):
            ar = np.radians(az)
            sin_az.append(np.sin(ar))
            cos_az.append(np.cos(ar))

    tilt_unc = float(np.std(tilts, ddof=1)) if len(tilts) > 1 else float("nan")
    if len(sin_az) > 1:
        # Mardia circular std: σ = sqrt(-2 * ln R), R is mean resultant length.
        r = float(np.hypot(np.mean(cos_az), np.mean(sin_az)))
        if r > 0:
            azimuth_unc = float(np.degrees(np.sqrt(-2.0 * np.log(r))))
        else:
            azimuth_unc = float("nan")
    else:
        azimuth_unc = float("nan")
    return tilt_unc, azimuth_unc
