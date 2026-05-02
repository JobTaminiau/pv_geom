"""Unit tests for ``geometry.heights``. M4 (PRD §7.4)."""

from __future__ import annotations

import numpy as np
import pytest

from pv_geom.geometry.heights import (
    height_above_ground,
    height_above_roof,
    panel_roof_angle_deg,
)
from pv_geom.geometry.plane_fit import PlaneFit


def _fit(normal=(0.0, 0.0, 1.0), centroid=(0.0, 0.0, 5.0), n_inliers=100,
         tilt=0.0, az=float("nan"), rmse=0.01) -> PlaneFit:
    return PlaneFit(
        normal=np.asarray(normal, dtype=float),
        centroid=np.asarray(centroid, dtype=float),
        tilt_deg=tilt,
        azimuth_deg=az,
        rmse=rmse,
        n_inliers=n_inliers,
        n_total=n_inliers,
        inlier_mask=np.ones(n_inliers, dtype=bool),
    )


# --------------------------------------------------------------------------- #
# height_above_ground
# --------------------------------------------------------------------------- #


def test_hag_simple_case() -> None:
    panel_z = np.full(50, 12.0)
    rng = np.random.default_rng(0)
    ground_xy = rng.uniform(-5, 5, size=(200, 2))
    ground_z = np.full(200, 10.0)
    ground = np.column_stack([ground_xy, ground_z])
    h = height_above_ground(panel_z, ground, (0.0, 0.0), search_radius_m=10.0)
    assert h == pytest.approx(2.0, abs=1e-6)


def test_hag_uses_only_points_within_radius() -> None:
    panel_z = np.full(50, 15.0)
    # Half the ground points 1 m away, half 50 m away — only near ones should count
    near = np.column_stack([
        np.linspace(-1, 1, 50), np.zeros(50), np.full(50, 10.0)
    ])
    far = np.column_stack([
        np.full(50, 50.0), np.zeros(50), np.full(50, 0.0)
    ])
    ground = np.concatenate([near, far])
    h = height_above_ground(panel_z, ground, (0.0, 0.0), search_radius_m=5.0)
    # Should be 15 - 10 = 5, not 15 - mean(0, 10) = 10
    assert h == pytest.approx(5.0, abs=1e-6)


def test_hag_no_ground_in_radius_returns_nan() -> None:
    panel_z = np.array([12.0, 12.0])
    ground = np.array([[100.0, 100.0, 10.0]])
    h = height_above_ground(panel_z, ground, (0.0, 0.0), search_radius_m=5.0)
    assert np.isnan(h)


def test_hag_empty_inputs_return_nan() -> None:
    assert np.isnan(height_above_ground(np.array([]), np.zeros((10, 3)), (0, 0)))
    assert np.isnan(height_above_ground(np.array([1.0]), np.zeros((0, 3)), (0, 0)))


# --------------------------------------------------------------------------- #
# height_above_roof
# --------------------------------------------------------------------------- #


def test_hroof_horizontal_roof() -> None:
    panel = np.column_stack([
        np.linspace(-1, 1, 20), np.linspace(-1, 1, 20), np.full(20, 12.5)
    ])
    roof = _fit(normal=(0, 0, 1), centroid=(0, 0, 10), tilt=0.0, az=float("nan"))
    h = height_above_roof(panel, roof)
    assert h == pytest.approx(2.5, abs=1e-6)


def test_hroof_tilted_roof_offset_panel() -> None:
    """Roof tilted 30 deg facing south. Panel sits 0.5 m above the roof at its
    own (median) location. Expected HAR = 0.5 m regardless of where the panel is."""
    tilt = 30.0
    az = 180.0
    t = np.radians(tilt)
    a = np.radians(az)
    nx, ny, nz = np.sin(t) * np.sin(a), np.sin(t) * np.cos(a), np.cos(t)
    cx, cy, cz = 0.0, 0.0, 10.0
    # Panel centroid at (3, 4)
    px, py = 3.0, 4.0
    z_roof_at_panel = cz - (nx * (px - cx) + ny * (py - cy)) / nz
    panel = np.array([[px, py, z_roof_at_panel + 0.5]])
    roof = _fit(normal=(nx, ny, nz), centroid=(cx, cy, cz), tilt=tilt, az=az)
    h = height_above_roof(panel, roof)
    assert h == pytest.approx(0.5, abs=1e-9)


def test_hroof_failed_fit_returns_nan() -> None:
    panel = np.zeros((10, 3))
    failed = _fit(n_inliers=0, tilt=float("nan"))
    assert np.isnan(height_above_roof(panel, failed))


# --------------------------------------------------------------------------- #
# panel_roof_angle_deg
# --------------------------------------------------------------------------- #


def test_panel_roof_angle_zero_when_aligned() -> None:
    n = np.array([0.1, 0.2, np.sqrt(1 - 0.05)])
    n /= np.linalg.norm(n)
    f = _fit(normal=n, tilt=10.0, az=180.0)
    g = _fit(normal=n, tilt=10.0, az=180.0)
    assert panel_roof_angle_deg(f, g) == pytest.approx(0.0, abs=1e-6)


def test_panel_roof_angle_known() -> None:
    # 0 deg vs 30 deg tilt facing same direction → 30 deg between normals
    n_flat = np.array([0.0, 0.0, 1.0])
    t = np.radians(30.0)
    n_tilt = np.array([0.0, np.sin(t), np.cos(t)])
    f = _fit(normal=n_flat, tilt=0.0, az=float("nan"))
    g = _fit(normal=n_tilt, tilt=30.0, az=180.0)
    assert panel_roof_angle_deg(f, g) == pytest.approx(30.0, abs=1e-6)


def test_panel_roof_angle_failed_fit_is_nan() -> None:
    f = _fit(normal=np.array([0.0, 0.0, 1.0]), n_inliers=100)
    bad = _fit(normal=np.array([0.0, 0.0, 1.0]), n_inliers=0, tilt=float("nan"))
    assert np.isnan(panel_roof_angle_deg(f, bad))
    assert np.isnan(panel_roof_angle_deg(bad, f))
