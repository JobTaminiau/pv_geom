"""Unit tests for ``geometry.roof_plane``. M4 (PRD §11.1)."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from pv_geom.config import RoofPlaneConfig
from pv_geom.geometry.roof_plane import extract_roof_plane


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _plane_z(xy: np.ndarray, tilt_deg: float, azimuth_deg: float, z0: float) -> np.ndarray:
    """z(x, y) for a plane through (0, 0, z0) at given tilt and compass azimuth."""
    t = np.radians(tilt_deg)
    a = np.radians(azimuth_deg)
    nx, ny, nz = np.sin(t) * np.sin(a), np.sin(t) * np.cos(a), np.cos(t)
    return z0 - (nx * xy[:, 0] + ny * xy[:, 1]) / nz


def _grid(extent: tuple[float, float, float, float], step: float = 0.3, seed: int = 0) -> np.ndarray:
    """Random-ish (x, y) grid at ~``step`` density inside ``extent``."""
    rng = np.random.default_rng(seed)
    xmin, ymin, xmax, ymax = extent
    nx = max(2, int((xmax - xmin) / step))
    ny = max(2, int((ymax - ymin) / step))
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    pts += rng.normal(0.0, step / 4, size=pts.shape)
    return pts


def _make_scene(
    *,
    pv_extent: tuple[float, float, float, float],
    building_extent: tuple[float, float, float, float],
    roof_tilt_deg: float = 25.0,
    roof_az_deg: float = 180.0,
    z0: float = 10.0,
    noise_m: float = 0.005,
    extra_pvs: list[Polygon] | None = None,
):
    """Synthesize a building, a PV polygon, and class-6-style roof returns."""
    pv = box(*pv_extent)
    building = box(*building_extent)
    footprints = gpd.GeoDataFrame(
        {"building_id": ["b1"]}, geometry=[building], crs="EPSG:6341"
    )
    extras = extra_pvs or []
    others = gpd.GeoDataFrame(geometry=extras, crs="EPSG:6341")
    # Roof returns: cover the whole building footprint, planar
    xy = _grid(building_extent, step=0.25, seed=0)
    z = _plane_z(xy, roof_tilt_deg, roof_az_deg, z0)
    rng = np.random.default_rng(1)
    z = z + rng.normal(0.0, noise_m, size=z.shape)
    pts = np.column_stack([xy, z])
    return pv, footprints, others, pts


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_off_building_returns_none() -> None:
    pv = box(100.0, 100.0, 110.0, 110.0)
    footprints = gpd.GeoDataFrame(
        {"building_id": ["b1"]},
        geometry=[box(0, 0, 20, 20)],
        crs="EPSG:6341",
    )
    others = gpd.GeoDataFrame(geometry=[], crs="EPSG:6341")
    pts = np.zeros((10, 3))
    result = extract_roof_plane(pv, footprints, others, pts, RoofPlaneConfig())
    assert result.fit is None
    assert result.on_building is False
    assert result.flag is None


def test_recovers_known_roof_orientation() -> None:
    pv, footprints, others, pts = _make_scene(
        pv_extent=(2.0, 2.0, 6.0, 6.0),
        building_extent=(0.0, 0.0, 10.0, 10.0),
        roof_tilt_deg=22.5,
        roof_az_deg=180.0,
    )
    result = extract_roof_plane(pv, footprints, others, pts, RoofPlaneConfig(), seed=0)
    assert result.on_building is True
    assert result.flag is None
    assert result.building_id == "b1"
    assert result.fit is not None
    assert result.fit.tilt_deg == pytest.approx(22.5, abs=0.5)
    diff = ((result.fit.azimuth_deg - 180.0 + 180) % 360) - 180
    assert abs(diff) < 1.5


def test_ring_excludes_pv_polygon() -> None:
    """Confirm the fit comes from points outside the PV (ring), not inside it.
    We seed PV-area returns with a contradictory orientation; if the ring is
    leaking, the fit will be biased toward the contradictory plane."""
    pv, footprints, others, pts_roof = _make_scene(
        pv_extent=(2.0, 2.0, 6.0, 6.0),
        building_extent=(0.0, 0.0, 10.0, 10.0),
        roof_tilt_deg=20.0,
        roof_az_deg=180.0,
    )
    rng = np.random.default_rng(2)
    pv_xy = rng.uniform(2.5, 5.5, size=(400, 2))
    # Contradictory plane inside the PV: tilt 20 deg facing NORTH (0 deg)
    pv_z = _plane_z(pv_xy, 20.0, 0.0, z0=20.0)
    pts_pv = np.column_stack([pv_xy, pv_z])
    pts = np.concatenate([pts_roof, pts_pv])
    result = extract_roof_plane(pv, footprints, others, pts, RoofPlaneConfig(), seed=0)
    assert result.fit is not None
    # If the ring is correct, we recover ~180 deg, NOT ~0 deg.
    diff = ((result.fit.azimuth_deg - 180.0 + 180) % 360) - 180
    assert abs(diff) < 2.0
    assert result.fit.tilt_deg == pytest.approx(20.0, abs=1.0)


def test_ring_subtracts_other_pvs_on_same_building() -> None:
    """A second PV array on the same building should be subtracted from the ring."""
    pv, footprints, _, pts_roof = _make_scene(
        pv_extent=(2.0, 2.0, 6.0, 6.0),
        building_extent=(0.0, 0.0, 10.0, 10.0),
    )
    other = box(7.0, 2.0, 9.0, 6.0)
    others = gpd.GeoDataFrame(geometry=[other], crs="EPSG:6341")
    # Add contradictory points within the OTHER PV polygon to detect leakage
    rng = np.random.default_rng(3)
    other_xy = rng.uniform(7.2, 8.8, size=(200, 2))
    other_xy[:, 1] = rng.uniform(2.2, 5.8, size=200)
    other_z = _plane_z(other_xy, 20.0, 0.0, z0=20.0)
    pts = np.concatenate([pts_roof, np.column_stack([other_xy, other_z])])
    result = extract_roof_plane(pv, footprints, others, pts, RoofPlaneConfig(), seed=0)
    assert result.fit is not None
    diff = ((result.fit.azimuth_deg - 180.0 + 180) % 360) - 180
    assert abs(diff) < 2.0


def test_buffer_expansion_kicks_in() -> None:
    """PV occupies most of the footprint — initial buffer is too small;
    expansion up to ``buffer_max_m`` should still recover a fit."""
    # 8x8 PV inside a 10x10 footprint: at buffer=1 m the ring is mostly outside
    # the footprint, so few in-footprint ring points exist.
    pv, footprints, others, pts = _make_scene(
        pv_extent=(1.0, 1.0, 9.0, 9.0),
        building_extent=(0.0, 0.0, 10.0, 10.0),
    )
    # Initial 0.2 m ring around an 8x8 PV (clipped to a 10x10 footprint) has
    # area ~6 m^2; at the synthetic point density we get ~100 points. With a
    # 400-point floor only the wider buffers will suffice.
    cfg = RoofPlaneConfig(buffer_m=0.2, buffer_max_m=2.0, buffer_step_m=0.4,
                          min_points=400)
    result = extract_roof_plane(pv, footprints, others, pts, cfg, seed=0)
    assert result.fit is not None
    assert result.used_buffer_m > cfg.buffer_m


def test_insufficient_points_flag() -> None:
    """Tiny footprint, large PV → ring is essentially empty even at buffer_max."""
    pv = box(0.0, 0.0, 9.5, 9.5)
    footprint = box(0.0, 0.0, 10.0, 10.0)
    footprints = gpd.GeoDataFrame(
        {"building_id": ["b1"]}, geometry=[footprint], crs="EPSG:6341"
    )
    others = gpd.GeoDataFrame(geometry=[], crs="EPSG:6341")
    rng = np.random.default_rng(4)
    pts_xy = rng.uniform(0, 10, size=(20, 2))
    pts_z = _plane_z(pts_xy, 20.0, 180.0, 10.0)
    pts = np.column_stack([pts_xy, pts_z])
    cfg = RoofPlaneConfig(buffer_m=0.1, buffer_max_m=0.2, min_points=200)
    result = extract_roof_plane(pv, footprints, others, pts, cfg, seed=0)
    assert result.flag == "roof_insufficient"
    assert result.fit is None


def test_complex_roof_flag_when_rmse_too_high() -> None:
    """Two-facet hip roof in the ring: the single-plane fit should have
    too much residual scatter to pass ``rmse_max_m`` and get flagged."""
    pv = box(4.0, 4.0, 6.0, 6.0)
    footprint = box(0.0, 0.0, 10.0, 10.0)
    footprints = gpd.GeoDataFrame(
        {"building_id": ["b1"]}, geometry=[footprint], crs="EPSG:6341"
    )
    others = gpd.GeoDataFrame(geometry=[], crs="EPSG:6341")
    # Hip: x<5 facet faces west (az=270, tilt 30), x>=5 faces east (az=90, tilt 30)
    xy = _grid((0, 0, 10, 10), step=0.25, seed=0)
    z_west = _plane_z(xy, 30.0, 270.0, 10.0)
    z_east = _plane_z(xy, 30.0, 90.0, 10.0)
    z = np.where(xy[:, 0] < 5.0, z_west, z_east)
    pts = np.column_stack([xy, z])
    # Tight RANSAC threshold so the fit can't span both facets cleanly.
    cfg = RoofPlaneConfig(buffer_m=2.0, buffer_max_m=3.0, ransac_threshold_m=0.10,
                          rmse_max_m=0.05, min_points=50)
    result = extract_roof_plane(pv, footprints, others, pts, cfg, seed=0)
    assert result.flag == "roof_complex"
    # We still report the (failed) fit for QC
    assert result.fit is not None


def test_picks_largest_overlapping_footprint() -> None:
    """Two overlapping footprints — pick the one with the bigger PV intersection."""
    pv = box(2.0, 2.0, 6.0, 6.0)
    big = box(0.0, 0.0, 10.0, 10.0)
    tiny = box(2.5, 2.5, 3.5, 3.5)              # also overlaps but smaller
    footprints = gpd.GeoDataFrame(
        {"building_id": ["big", "tiny"]},
        geometry=[big, tiny],
        crs="EPSG:6341",
    )
    others = gpd.GeoDataFrame(geometry=[], crs="EPSG:6341")
    xy = _grid((0, 0, 10, 10), step=0.25, seed=0)
    z = _plane_z(xy, 15.0, 200.0, 10.0)
    pts = np.column_stack([xy, z])
    result = extract_roof_plane(pv, footprints, others, pts, RoofPlaneConfig(), seed=0)
    assert result.building_id == "big"
