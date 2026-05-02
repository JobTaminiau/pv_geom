"""Unit tests for ``geometry.multi_plane``. M5 (PRD §7.2)."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from pv_geom.config import MultiPlaneConfig
from pv_geom.geometry.multi_plane import (
    detect_multi_plane,
    is_tracker_suspected,
    polygon_aspect_ratio,
)
from pv_geom.geometry.plane_fit import fit_plane_ransac


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _plane(tilt_deg: float, az_deg: float, n: int = 200, seed: int = 0,
           noise_m: float = 0.005) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.radians(tilt_deg)
    a = np.radians(az_deg)
    nx, ny, nz = np.sin(t) * np.sin(a), np.sin(t) * np.cos(a), np.cos(t)
    seed_v = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed_v, [nx, ny, nz])) > 0.95:
        seed_v = np.array([0.0, 1.0, 0.0])
    n_truth = np.array([nx, ny, nz])
    u = seed_v - np.dot(seed_v, n_truth) * n_truth
    u /= np.linalg.norm(u)
    v = np.cross(n_truth, u)
    uv = rng.uniform(-3.0, 3.0, size=(n, 2))
    pts = uv[:, 0:1] * u + uv[:, 1:2] * v
    pts = pts + rng.normal(0.0, noise_m, size=n)[:, None] * n_truth
    return pts


# --------------------------------------------------------------------------- #
# polygon_aspect_ratio
# --------------------------------------------------------------------------- #


def test_aspect_ratio_square() -> None:
    sq = box(0, 0, 4, 4)
    assert polygon_aspect_ratio(sq) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("long,short", [(4.0, 2.0), (10.0, 1.0), (50.0, 2.5)])
def test_aspect_ratio_rectangle(long: float, short: float) -> None:
    rect = box(0, 0, long, short)
    assert polygon_aspect_ratio(rect) == pytest.approx(long / short, abs=1e-6)


def test_aspect_ratio_invariant_under_rotation() -> None:
    from shapely.affinity import rotate
    rect = box(0, 0, 8, 2)
    base = polygon_aspect_ratio(rect)
    for angle in (15, 37, 90, 123):
        r = rotate(rect, angle, origin="center")
        assert polygon_aspect_ratio(r) == pytest.approx(base, rel=1e-6)


def test_aspect_ratio_empty_returns_nan() -> None:
    assert np.isnan(polygon_aspect_ratio(Polygon()))
    assert np.isnan(polygon_aspect_ratio(None))


# --------------------------------------------------------------------------- #
# detect_multi_plane
# --------------------------------------------------------------------------- #


def test_single_plane_no_secondary() -> None:
    pts = _plane(20.0, 180.0, n=400, seed=1)
    primary = fit_plane_ransac(pts, ransac_threshold=0.05, seed=0)
    res = detect_multi_plane(pts, primary, MultiPlaneConfig(), seed=0)
    assert res.secondary is None
    assert res.flags == ()


def test_east_west_rack_flagged() -> None:
    """Two planes 180 deg apart in azimuth at similar tilts → ew_rack.
    Use a 2:1 mix so the primary clears the default min_inlier_frac."""
    east = _plane(20.0, 90.0, n=400, seed=1)
    east[:, 0] += 5.0
    west = _plane(20.0, 270.0, n=200, seed=2)
    west[:, 0] -= 5.0
    pts = np.concatenate([east, west])
    primary = fit_plane_ransac(
        pts, ransac_threshold=0.05, min_inlier_frac=0.5, seed=0
    )
    assert not np.isnan(primary.tilt_deg)
    cfg = MultiPlaneConfig(secondary_min_frac=0.20)
    res = detect_multi_plane(pts, primary, cfg, seed=0)
    assert res.secondary is not None
    assert "east_west_rack" in res.flags
    assert res.primary.tilt_deg == pytest.approx(20.0, abs=2.0)
    assert res.secondary.tilt_deg == pytest.approx(20.0, abs=2.0)


def test_disabled_short_circuits() -> None:
    pts = _plane(20.0, 180.0, n=200, seed=1)
    primary = fit_plane_ransac(pts, seed=0)
    cfg = MultiPlaneConfig(enabled=False)
    res = detect_multi_plane(pts, primary, cfg, seed=0)
    assert res.secondary is None
    assert res.flags == ()


def test_failed_primary_skips() -> None:
    from pv_geom.geometry.plane_fit import _failed_fit
    primary = _failed_fit(0)
    res = detect_multi_plane(np.zeros((0, 3)), primary, MultiPlaneConfig(), seed=0)
    assert res.secondary is None


# --------------------------------------------------------------------------- #
# is_tracker_suspected
# --------------------------------------------------------------------------- #


def test_tracker_clear_yes() -> None:
    assert is_tracker_suspected(
        on_building=False, aspect_ratio=8.0,
        height_above_ground_m=1.0, panel_tilt_deg=20.0,
    ) is True


def test_tracker_on_building_is_no() -> None:
    assert is_tracker_suspected(
        on_building=True, aspect_ratio=8.0,
        height_above_ground_m=1.0, panel_tilt_deg=20.0,
    ) is False


def test_tracker_too_short_is_no() -> None:
    assert is_tracker_suspected(
        on_building=False, aspect_ratio=2.0,
        height_above_ground_m=1.0, panel_tilt_deg=20.0,
    ) is False


def test_tracker_too_high_is_no() -> None:
    """High height-above-ground → carport-like, not tracker."""
    assert is_tracker_suspected(
        on_building=False, aspect_ratio=8.0,
        height_above_ground_m=3.0, panel_tilt_deg=20.0,
    ) is False


def test_tracker_high_tilt_is_no() -> None:
    """A 60 deg tilt isn't physically tracker-like."""
    assert is_tracker_suspected(
        on_building=False, aspect_ratio=8.0,
        height_above_ground_m=1.0, panel_tilt_deg=60.0,
    ) is False


def test_tracker_nan_returns_false() -> None:
    assert is_tracker_suspected(
        on_building=False, aspect_ratio=float("nan"),
        height_above_ground_m=1.0, panel_tilt_deg=20.0,
    ) is False
