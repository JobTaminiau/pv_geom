"""Synthetic-plane unit tests for ``geometry.plane_fit``. M3 (PRD §11.1)."""

from __future__ import annotations

import numpy as np
import pytest

from pv_geom.geometry.plane_fit import (
    PlaneFit,
    bootstrap_uncertainty,
    fit_plane_ransac,
    tilt_azimuth_from_normal,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def normal_from_tilt_azimuth(tilt_deg: float, azimuth_deg: float) -> np.ndarray:
    """Unit normal (x=East, y=North, z=Up) for a panel facing ``azimuth_deg``."""
    t = np.radians(tilt_deg)
    a = np.radians(azimuth_deg)
    return np.array([np.sin(t) * np.sin(a), np.sin(t) * np.cos(a), np.cos(t)])


def synthesize_plane(
    tilt_deg: float,
    azimuth_deg: float,
    n: int = 200,
    extent_m: float = 5.0,
    noise_m: float = 0.0,
    outlier_frac: float = 0.0,
    outlier_scale_m: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize ``n`` points lying on a plane at known orientation.

    Returns (points, normal_truth). Optionally adds Gaussian noise normal to
    the plane and replaces ``outlier_frac`` of points with uniformly sampled
    outliers in a thick slab.
    """
    rng = np.random.default_rng(seed)
    n_truth = normal_from_tilt_azimuth(tilt_deg, azimuth_deg)
    # Two in-plane orthonormal directions (Gram-Schmidt against +x then +y).
    seed_v = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed_v, n_truth)) > 0.95:
        seed_v = np.array([0.0, 1.0, 0.0])
    u = seed_v - np.dot(seed_v, n_truth) * n_truth
    u /= np.linalg.norm(u)
    v = np.cross(n_truth, u)

    uv = rng.uniform(-extent_m / 2, extent_m / 2, size=(n, 2))
    pts = uv[:, 0:1] * u + uv[:, 1:2] * v   # (n, 3) on the plane through origin
    if noise_m > 0:
        pts = pts + rng.normal(0.0, noise_m, size=n)[:, None] * n_truth

    if outlier_frac > 0:
        n_out = round(n * outlier_frac)
        out_idx = rng.choice(n, size=n_out, replace=False)
        pts[out_idx] = rng.uniform(
            -outlier_scale_m, outlier_scale_m, size=(n_out, 3)
        )
    return pts.astype(np.float64), n_truth


def assert_close_normals(n1: np.ndarray, n2: np.ndarray, deg_tol: float) -> None:
    """Compare two unit normals modulo sign, asserting angle <= ``deg_tol``."""
    cos_angle = abs(float(np.dot(n1, n2)))
    cos_angle = min(max(cos_angle, -1.0), 1.0)
    angle_deg = np.degrees(np.arccos(cos_angle))
    assert angle_deg <= deg_tol, f"normals differ by {angle_deg:.3f} deg (tol {deg_tol})"


# --------------------------------------------------------------------------- #
# tilt_azimuth_from_normal
# --------------------------------------------------------------------------- #


def test_tilt_azimuth_horizontal_returns_nan_azimuth() -> None:
    tilt, az = tilt_azimuth_from_normal(np.array([0.0, 0.0, 1.0]))
    assert tilt == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(az)


@pytest.mark.parametrize(
    "az_in", [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
)
def test_tilt_azimuth_round_trip(az_in: float) -> None:
    n = normal_from_tilt_azimuth(20.0, az_in)
    tilt, az_out = tilt_azimuth_from_normal(n)
    assert tilt == pytest.approx(20.0, abs=1e-6)
    # azimuth wraps at 360 — compare on the circle
    diff = ((az_out - az_in + 180) % 360) - 180
    assert abs(diff) < 1e-6


def test_tilt_azimuth_canonicalizes_downward_normal() -> None:
    n_up = normal_from_tilt_azimuth(15.0, 200.0)
    n_down = -n_up
    tilt_up, az_up = tilt_azimuth_from_normal(n_up)
    tilt_down, az_down = tilt_azimuth_from_normal(n_down)
    assert tilt_up == pytest.approx(tilt_down, abs=1e-9)
    assert az_up == pytest.approx(az_down, abs=1e-9)


def test_tilt_azimuth_below_floor_is_nan() -> None:
    _, az = tilt_azimuth_from_normal(
        normal_from_tilt_azimuth(0.5, 180.0), tilt_floor_deg=1.0
    )
    assert np.isnan(az)


# --------------------------------------------------------------------------- #
# fit_plane_ransac — clean cases
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tilt_deg", [0.0, 5.0, 20.0, 45.0])
@pytest.mark.parametrize("azimuth_deg", [0.0, 90.0, 180.0, 270.0])
def test_recovers_known_orientation_clean(tilt_deg: float, azimuth_deg: float) -> None:
    if tilt_deg == 0.0 and azimuth_deg != 0.0:
        pytest.skip("azimuth ill-defined at tilt=0")
    pts, n_truth = synthesize_plane(tilt_deg, azimuth_deg, n=300, noise_m=0.0, seed=42)
    fit = fit_plane_ransac(pts, ransac_threshold=0.01, seed=0)
    assert fit.n_inliers >= 0.95 * fit.n_total
    assert_close_normals(fit.normal, n_truth, deg_tol=0.5)
    if tilt_deg >= 1.0:
        assert fit.tilt_deg == pytest.approx(tilt_deg, abs=0.5)
        diff = ((fit.azimuth_deg - azimuth_deg + 180) % 360) - 180
        assert abs(diff) < 1.5
    else:
        assert np.isnan(fit.azimuth_deg)


def test_recovers_orientation_with_noise() -> None:
    pts, n_truth = synthesize_plane(
        tilt_deg=18.0, azimuth_deg=200.0, n=400, noise_m=0.02, seed=7
    )
    fit = fit_plane_ransac(pts, ransac_threshold=0.05, seed=0)
    assert fit.n_inliers >= 0.85 * fit.n_total
    assert_close_normals(fit.normal, n_truth, deg_tol=1.0)
    assert fit.rmse < 0.05
    assert fit.tilt_deg == pytest.approx(18.0, abs=1.0)


def test_rejects_outliers() -> None:
    pts, n_truth = synthesize_plane(
        tilt_deg=10.0,
        azimuth_deg=180.0,
        n=400,
        noise_m=0.01,
        outlier_frac=0.30,
        outlier_scale_m=2.0,
        seed=11,
    )
    fit = fit_plane_ransac(
        pts, ransac_threshold=0.05, max_iter=400, seed=0
    )
    # Should recover ~70% inliers (the non-outlier plane points)
    assert 0.6 < fit.n_inliers / fit.n_total < 0.85
    assert_close_normals(fit.normal, n_truth, deg_tol=1.5)
    assert fit.tilt_deg == pytest.approx(10.0, abs=1.5)


# --------------------------------------------------------------------------- #
# fit_plane_ransac — degenerate cases
# --------------------------------------------------------------------------- #


def test_too_few_points_returnsfailed_fit() -> None:
    pts = np.zeros((2, 3))
    fit = fit_plane_ransac(pts)
    assert fit.n_inliers == 0
    assert fit.n_total == 2
    assert np.isnan(fit.tilt_deg)
    assert np.isnan(fit.azimuth_deg)


def test_collinear_points_are_handled() -> None:
    rng = np.random.default_rng(0)
    t = rng.uniform(-5, 5, size=50)
    pts = np.column_stack([t, 2 * t, 3 * t])     # all on a line
    fit = fit_plane_ransac(pts, ransac_threshold=0.05, seed=0)
    # Either: (a) RANSAC fails to find any non-degenerate sample → failed_fit; or
    # (b) a plane is found but inlier_frac < 0.6 → tilt/az NaN. Both acceptable.
    assert np.isnan(fit.tilt_deg)
    assert np.isnan(fit.azimuth_deg)


def test_low_inlier_fraction_yields_nan_tilt() -> None:
    # A plane embedded in mostly-outlier noise: inlier frac will fall below 0.6
    pts, _ = synthesize_plane(
        tilt_deg=15.0, azimuth_deg=180.0, n=400,
        noise_m=0.005, outlier_frac=0.55, outlier_scale_m=3.0, seed=3
    )
    fit = fit_plane_ransac(pts, ransac_threshold=0.05, min_inlier_frac=0.6, seed=0)
    # With >50% outliers, the inlier majority is borderline. Check the contract:
    # if n_inliers/n < min_inlier_frac, tilt/azimuth must be NaN.
    if fit.n_inliers / fit.n_total < 0.6:
        assert np.isnan(fit.tilt_deg)
        assert np.isnan(fit.azimuth_deg)
    else:
        assert not np.isnan(fit.tilt_deg)


def test_invalid_shape_raises() -> None:
    with pytest.raises(ValueError):
        fit_plane_ransac(np.zeros((10, 2)))


# --------------------------------------------------------------------------- #
# Determinism + bootstrap
# --------------------------------------------------------------------------- #


def test_seed_determinism() -> None:
    pts, _ = synthesize_plane(20.0, 180.0, n=200, noise_m=0.02, seed=1)
    f1 = fit_plane_ransac(pts, seed=42)
    f2 = fit_plane_ransac(pts, seed=42)
    assert np.allclose(f1.normal, f2.normal)
    assert f1.n_inliers == f2.n_inliers


def test_bootstrap_uncertainty_sane_clean_plane() -> None:
    pts, _ = synthesize_plane(20.0, 180.0, n=400, noise_m=0.01, seed=5)
    fit = fit_plane_ransac(pts, seed=0)
    tilt_unc, az_unc = bootstrap_uncertainty(pts, fit, n_samples=50, seed=0)
    # Tight noise → tight uncertainty. Loose tol because bootstrap is sample-noisy.
    assert 0.0 < tilt_unc < 0.5
    assert 0.0 < az_unc < 1.5


def test_bootstrap_uncertainty_grows_with_noise() -> None:
    pts_low, _ = synthesize_plane(20.0, 180.0, n=400, noise_m=0.005, seed=5)
    pts_hi, _ = synthesize_plane(20.0, 180.0, n=400, noise_m=0.05, seed=5)
    fit_low = fit_plane_ransac(pts_low, ransac_threshold=0.02, seed=0)
    fit_hi = fit_plane_ransac(pts_hi, ransac_threshold=0.15, seed=0)
    tu_low, _ = bootstrap_uncertainty(pts_low, fit_low, n_samples=50, seed=0)
    tu_hi, _ = bootstrap_uncertainty(pts_hi, fit_hi, n_samples=50, seed=0)
    assert tu_hi > tu_low


def test_bootstrapfailed_fit_returns_nan() -> None:
    fit = PlaneFit(
        normal=np.array([0.0, 0.0, 1.0]),
        centroid=np.zeros(3),
        tilt_deg=float("nan"),
        azimuth_deg=float("nan"),
        rmse=float("nan"),
        n_inliers=0,
        n_total=10,
        inlier_mask=np.zeros(10, dtype=bool),
    )
    tu, au = bootstrap_uncertainty(np.zeros((10, 3)), fit)
    assert np.isnan(tu) and np.isnan(au)


# --------------------------------------------------------------------------- #
# E9: the fitted plane does not depend on the random seed
# --------------------------------------------------------------------------- #

def _plane_points(tilt: float, azimuth: float, *, n: int, extent: float = 6.0,
                  noise_m: float = 0.0, seed: int = 0) -> np.ndarray:
    """``n`` points on a plane of the given tilt and azimuth through the origin,
    spread over an ``extent`` metre square, with vertical noise."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-extent / 2, extent / 2, size=(n, 2))
    nx, ny, nz = normal_from_tilt_azimuth(tilt, azimuth)
    z = -(nx * xy[:, 0] + ny * xy[:, 1]) / nz + rng.normal(0.0, noise_m, n)
    return np.column_stack([xy, z])


def _two_rival_planes(n_each: int = 40, seed: int = 0) -> np.ndarray:
    """Two planes 6 degrees apart crossing in the middle of the patch, with
    almost the same number of points near each: the case a random search
    settles differently from run to run."""
    rng = np.random.default_rng(seed)
    a = _plane_points(20.0, 180.0, n=n_each + 2, extent=6.0, noise_m=0.004, seed=seed)
    b = _plane_points(26.0, 180.0, n=n_each, extent=6.0, noise_m=0.004, seed=seed + 1)
    return rng.permutation(np.vstack([a, b]))


def test_small_sets_are_searched_exhaustively_so_the_seed_is_irrelevant() -> None:
    pts = _two_rival_planes()
    fits = [fit_plane_ransac(pts, ransac_threshold=0.03, min_inlier_frac=0.3, seed=s)
            for s in range(8)]
    assert len({round(f.tilt_deg, 6) for f in fits}) == 1
    assert len({f.n_inliers for f in fits}) == 1
    # ...and the order the points arrive in does not matter either.
    shuffled = fit_plane_ransac(pts[::-1].copy(), ransac_threshold=0.03, min_inlier_frac=0.3,
                                seed=99)
    assert shuffled.tilt_deg == pytest.approx(fits[0].tilt_deg, abs=1e-6)


def test_a_nearly_tied_rival_plane_is_reported() -> None:
    fit = fit_plane_ransac(_two_rival_planes(), ransac_threshold=0.03, min_inlier_frac=0.3,
                           seed=0)
    assert fit.rival_share > 0.8 and 4.0 < fit.rival_angle_deg < 8.0
    clean = fit_plane_ransac(_plane_points(20.0, 180.0, n=120, noise_m=0.004, seed=3),
                             ransac_threshold=0.03, seed=0)
    assert np.isnan(clean.rival_share) or clean.rival_share < 0.5


@pytest.mark.parametrize("n", [60, 400, 3000])
def test_noisy_plane_is_stable_across_seeds_at_every_size(n: int) -> None:
    """Small sets by exhaustive search, large ones by the smooth final polish."""
    pts = _plane_points(27.0, 215.0, n=n, extent=8.0, noise_m=0.03, seed=n)
    fits = [fit_plane_ransac(pts, ransac_threshold=0.05, seed=s) for s in range(6)]
    tilts = np.array([f.tilt_deg for f in fits])
    assert np.ptp(tilts) < 1e-3
    assert abs(tilts[0] - 27.0) < (1.0 if n == 60 else 0.3)
    az = np.array([f.azimuth_deg for f in fits])
    assert np.ptp(az) < 1e-2


def test_fitted_plane_is_not_biased_by_the_robust_polish() -> None:
    """Outliers on one side (a roof under a raised array) must not drag it."""
    plane = _plane_points(18.0, 180.0, n=500, extent=8.0, noise_m=0.01, seed=5)
    below = _plane_points(18.0, 180.0, n=150, extent=8.0, noise_m=0.01, seed=6)
    below[:, 2] -= 0.20
    fit = fit_plane_ransac(np.vstack([plane, below]), ransac_threshold=0.05, seed=1)
    assert fit.tilt_deg == pytest.approx(18.0, abs=0.15)
    assert fit.n_inliers == pytest.approx(500, abs=10)
