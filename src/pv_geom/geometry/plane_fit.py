"""RANSAC + LSQ plane fitting. Tilt/azimuth from the unit normal. M3 (PRD §7.1)."""

from __future__ import annotations

import itertools
import math
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
    # The strongest *different* plane found in the same points: its inlier
    # count as a share of this plane's, and the angle between the two. A rival
    # share near 1 means the data support two planes about equally and the
    # choice between them is fragile. NaN when no rival was found.
    rival_share: float = float("nan")
    rival_angle_deg: float = float("nan")


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


def failed_fit(n_total: int) -> PlaneFit:
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


# Hypotheses refined to convergence before the best is chosen. Refinement is
# what makes the answer stable: many different starting triples settle on the
# same plane, so which triples happened to be drawn stops mattering.
_REFINE_CANDIDATES = 12
_REFINE_ROUNDS = 25
_POLISH_ROUNDS = 200
# Hypotheses scored per block (bounds the points-by-hypotheses matrix).
_BLOCK_CELLS = 4_000_000
# How many hypotheses a fit of n points gets. Small sets are where two planes
# can tie to within an inlier, and missing one of them is what made results
# depend on the seed, so they are searched hard (every triple up to about 80
# points). Large sets have one clear plane and need few: the count falls with
# n squared, down to the configured floor.
_SCORE_CELLS = 6_000_000          # at most this many point-hypothesis pairs ...
_SMALL_SET = 1_500_000_000        # ... and at most this / n^2 hypotheses
_ALL_TRIPLES = 300_000            # try every triple when there are no more than this
                                  # (about 120 points): nothing is then left to chance
# A rival plane counts as different when it is at least this far from the winner.
_RIVAL_MIN_ANGLE_DEG = 2.0
# Two hypotheses closer than this in angle and within the inlier threshold in
# height are the same plane.
_SAME_PLANE_DEG = 2.0


def _refine(pts: np.ndarray, mask: np.ndarray, threshold: float
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Refit a plane to its inliers and reclassify, until the inlier set stops
    changing. Returns ``(normal, centroid, inlier mask)`` or None if it empties."""
    seen: set[bytes] = set()
    normal = centroid = None
    for _ in range(_REFINE_ROUNDS):
        if int(mask.sum()) < 3:
            return None
        inliers = pts[mask]
        centroid = inliers.mean(axis=0)
        normal = _pca_normal(inliers - centroid)
        new_mask = np.abs((pts - centroid) @ normal) < threshold
        key = np.packbits(new_mask).tobytes()
        if np.array_equal(new_mask, mask) or key in seen:      # settled (or cycling)
            mask = new_mask
            break
        seen.add(key)
        mask = new_mask
    if normal is None or int(mask.sum()) < 3:
        return None
    # The plane reported is the least-squares plane of the inliers reported.
    inliers = pts[mask]
    centroid = inliers.mean(axis=0)
    normal = _pca_normal(inliers - centroid)
    return normal, centroid, mask


def _polish(pts: np.ndarray, normal: np.ndarray, centroid: np.ndarray, threshold: float
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Settle a plane with a smooth robust fit (Tukey biweight, scale
    ``threshold``), iterated to convergence.

    Refitting to a hard inlier set has many fixed points a fraction of a degree
    apart, and which one a search lands on is luck. Weights that fall smoothly
    to zero at the threshold have one optimum nearby, so every start in its
    basin ends on the same plane.
    """
    for _ in range(_POLISH_ROUNDS):
        r = (pts - centroid) @ normal
        w = np.clip(1.0 - (r / threshold) ** 2, 0.0, None) ** 2
        total = float(w.sum())
        if total <= 0 or int((w > 0).sum()) < 3:
            return None
        new_centroid = (w[:, None] * pts).sum(axis=0) / total
        centred = (pts - new_centroid) * np.sqrt(w)[:, None]
        new_normal = _pca_normal(centred)
        moved = 1.0 - abs(float(new_normal @ normal))
        shift = abs(float((new_centroid - centroid) @ new_normal))
        normal, centroid = new_normal, new_centroid
        if moved < 1e-12 and shift < 1e-7:
            break
    mask = np.abs((pts - centroid) @ normal) < threshold
    if int(mask.sum()) < 3:
        return None
    return normal, centroid, mask


def _consensus(pts: np.ndarray, threshold: float, n_hypotheses: int, seed: int | None,
               refine: bool, thorough: bool = True
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float] | None:
    """The plane with the largest consensus among ``pts``.

    Planes through point triples (all of them for a small set, otherwise at
    least ``n_hypotheses`` random ones) are scored by how many points lie
    within ``threshold``; the best distinct few are each refined to convergence and
    the one that ends with the most inliers wins (ties: the smaller residual,
    then the normal's components, so the choice never depends on draw order).
    """
    n = len(pts)
    # How many hypotheses: every triple when there are few enough points to
    # afford it (then nothing is random at all), otherwise as many as the
    # scoring budget allows and never fewer than asked for.
    affordable = int(n_hypotheses)
    if thorough:
        affordable = max(affordable, min(_SCORE_CELLS // n, _SMALL_SET // (n * n)))
    if math.comb(n, 3) <= (max(affordable, _ALL_TRIPLES) if thorough else affordable):
        idx = np.array(list(itertools.combinations(range(n), 3)), dtype=np.int64)
    else:
        idx = np.random.default_rng(seed).integers(0, n, size=(affordable, 3))
    p0 = pts[idx[:, 0]]
    normals = np.cross(pts[idx[:, 1]] - p0, pts[idx[:, 2]] - p0)
    length = np.linalg.norm(normals, axis=1)
    ok = length > 1e-9                        # drops repeated and collinear triples
    if not ok.any():
        return None
    normals, p0 = normals[ok] / length[ok, None], p0[ok]
    normals[normals[:, 2] < 0] *= -1.0
    offsets = np.einsum("ij,ij->i", normals, p0)

    # Scored in single precision about the data's centre: half the memory
    # traffic, and millimetres are still exact at any coordinate size.
    centre = pts.mean(axis=0)
    local = (pts - centre).astype(np.float32)
    normals32 = normals.astype(np.float32)
    offsets32 = (offsets - normals @ centre).astype(np.float32)
    counts = np.empty(len(normals), dtype=np.int64)
    block = max(1, _BLOCK_CELLS // n)
    for start in range(0, len(normals), block):
        stop = start + block
        dist = np.abs(local @ normals32[start:stop].T - offsets32[start:stop])
        counts[start:stop] = np.count_nonzero(dist < threshold, axis=0)

    # Candidates to refine: the best-supported hypotheses that are different
    # planes. Without the "different", all of them are usually one plane and a
    # rival with one inlier fewer is never looked at.
    height = offsets - normals @ centre           # signed distance of the data centre
    picked: list[int] = []
    cos_same = np.cos(np.radians(_SAME_PLANE_DEG))
    alive = counts.astype(np.float64)
    for _ in range(_REFINE_CANDIDATES if refine else 1):
        h = int(np.argmax(alive))                 # first of equals: order-independent
        if alive[h] < 3:
            break
        picked.append(h)
        same = (normals @ normals[h] >= cos_same) & (np.abs(height - height[h]) <= threshold)
        alive[same] = -1.0

    best: tuple[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
    refined: list[tuple[int, np.ndarray]] = []
    for h in picked:
        mask = np.abs(pts @ normals[h] - offsets[h]) < threshold
        if refine:
            plane = _refine(pts, mask, threshold)
            if plane is None:
                continue
        else:
            if int(mask.sum()) < 3:
                continue
            plane = (normals[h], p0[h], mask)
        normal, centroid, mask = plane
        rmse = float(np.sqrt(np.mean(((pts[mask] - centroid) @ normal) ** 2)))
        key = (int(mask.sum()), -round(rmse, 9), *(round(float(v), 9) for v in normal))
        refined.append((int(mask.sum()), normal))
        if best is None or key > best[0]:
            best = (key, plane)
    if best is None:
        return None
    # The strongest refined candidate that is a different plane from the winner.
    win_count, win_normal = best[0][0], best[1][0]
    rival_share, rival_angle = float("nan"), float("nan")
    for count, normal in refined:
        angle = float(np.degrees(np.arccos(min(1.0, abs(float(normal @ win_normal))))))
        if angle >= _RIVAL_MIN_ANGLE_DEG and (np.isnan(rival_share)
                                               or count / win_count > rival_share):
            rival_share, rival_angle = count / win_count, angle
    plane = best[1]
    if refine:
        plane = _polish(pts, plane[0], plane[1], threshold) or plane
    return (*plane, rival_share, rival_angle)


def fit_plane_ransac(
    points: np.ndarray,
    ransac_threshold: float = 0.05,
    min_inlier_frac: float = 0.6,
    max_iter: int = 300,
    refine_with_lsq: bool = True,
    thorough: bool = True,
    tilt_floor_deg: float = 1.0,
    seed: int | None = None,
) -> PlaneFit:
    """Fit a plane via RANSAC, refine on inliers via PCA-LSQ. PRD §7.1.

    ``thorough`` searches small point sets much harder (every triple, up to
    about 80 points), which is what keeps the answer from depending on the
    seed when two planes are nearly tied. Turn it off where only the rough
    layout matters.

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
        return failed_fit(n)

    best = _consensus(pts, ransac_threshold, int(max_iter), seed, refine_with_lsq, thorough)
    if best is None:
        return failed_fit(n)
    normal, centroid, best_inliers, rival_share, rival_angle = best
    inliers = pts[best_inliers]
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
            rival_share=rival_share,
            rival_angle_deg=rival_angle,
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
        rival_share=rival_share,
        rival_angle_deg=rival_angle,
    )


def fit_planes_sequential(
    points: np.ndarray,
    *,
    ransac_threshold: float,
    min_points: int,
    max_planes: int = 4,
    max_iter: int = 300,
    tilt_floor_deg: float = 1.0,
    thorough: bool = True,
    seed: int | None = None,
) -> list[PlaneFit]:
    """Peel planes off a point set one at a time, largest first.

    RANSAC finds the plane with the most support; its inliers are set aside and
    the search repeats on what is left, until a plane would have fewer than
    ``min_points`` inliers or ``max_planes`` have been found. No consensus
    floor is applied — the caller decides what each plane is worth.

    Every returned ``PlaneFit`` has ``n_total = len(points)`` and an
    ``inlier_mask`` over *all* the points, and the masks are disjoint.
    """
    pts = np.asarray(points, dtype=np.float64)
    remaining = np.ones(len(pts), dtype=bool)
    planes: list[PlaneFit] = []
    for k in range(int(max_planes)):
        idx = np.flatnonzero(remaining)
        if len(idx) < max(3, min_points):
            break
        fit = fit_plane_ransac(
            pts[idx], ransac_threshold=ransac_threshold, min_inlier_frac=0.0,
            max_iter=max_iter, tilt_floor_deg=tilt_floor_deg, thorough=thorough,
            seed=None if seed is None else seed + k,
        )
        if fit.n_inliers < min_points or np.isnan(fit.tilt_deg):
            break
        mask = np.zeros(len(pts), dtype=bool)
        mask[idx[fit.inlier_mask]] = True
        planes.append(PlaneFit(
            normal=fit.normal, centroid=fit.centroid, tilt_deg=fit.tilt_deg,
            azimuth_deg=fit.azimuth_deg, rmse=fit.rmse, n_inliers=fit.n_inliers,
            n_total=len(pts), inlier_mask=mask,
        ))
        remaining &= ~mask
    return planes


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
        azimuth_unc = float(np.degrees(np.sqrt(-2.0 * np.log(r)))) if r > 0 else float("nan")
    else:
        azimuth_unc = float("nan")
    return tilt_unc, azimuth_unc
