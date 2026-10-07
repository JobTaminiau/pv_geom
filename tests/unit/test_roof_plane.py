"""Unit tests for ``geometry.roof_plane``. M4 (PRD §11.1)."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from pv_geom.config import RoofPlaneConfig
from pv_geom.geometry.plane_fit import fit_plane_ransac
from pv_geom.geometry.roof_plane import _enforce_collar_agreement, extract_roof_plane

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
    """A single but rough roof surface: consensus is fine, residual scatter is
    not. That is what ``roof_complex`` means since 0.4.0 — a *quality* failure,
    distinct from the consensus failure below."""
    pv = box(4.0, 4.0, 6.0, 6.0)
    footprints = gpd.GeoDataFrame(
        {"building_id": ["b1"]}, geometry=[box(0.0, 0.0, 10.0, 10.0)], crs="EPSG:6341"
    )
    others = gpd.GeoDataFrame(geometry=[], crs="EPSG:6341")
    xy = _grid((0, 0, 10, 10), step=0.25, seed=0)
    rng = np.random.default_rng(7)
    # One plane, but 8 cm of scatter — a loose RANSAC threshold keeps nearly
    # every point an inlier (consensus passes) while RMSE lands above the gate.
    z = _plane_z(xy, 20.0, 180.0, 10.0) + rng.normal(0.0, 0.08, size=len(xy))
    pts = np.column_stack([xy, z])
    cfg = RoofPlaneConfig(buffer_m=2.0, buffer_max_m=3.0, ransac_threshold_m=0.30,
                          rmse_max_m=0.05, min_points=50)
    result = extract_roof_plane(pv, footprints, others, pts, cfg, seed=0)
    assert result.flag == "roof_complex"
    # The fit is still reported for QC, but must not be used as a reference.
    assert result.fit is not None
    assert result.usable is False


def test_no_consensus_flag_when_array_straddles_the_ridge() -> None:
    """An array sitting across a gable ridge has no single roof plane beneath
    it. Neither the ring nor the collar can agree, and the honest answer is
    ``roof_no_consensus`` — not a confident fit to whichever facet is bigger."""
    pv = box(4.0, 4.0, 6.0, 6.0)                # centred on the ridge at x=5
    footprints = gpd.GeoDataFrame(
        {"building_id": ["b1"]}, geometry=[box(0.0, 0.0, 10.0, 10.0)], crs="EPSG:6341"
    )
    others = gpd.GeoDataFrame(geometry=[], crs="EPSG:6341")
    xy = _grid((0, 0, 10, 10), step=0.25, seed=0)
    z = np.where(
        xy[:, 0] < 5.0,
        _plane_z(xy, 35.0, 270.0, 10.0),
        _plane_z(xy, 35.0, 90.0, 10.0),
    )
    pts = np.column_stack([xy, z])
    cfg = RoofPlaneConfig(buffer_m=2.0, buffer_max_m=3.0, ransac_threshold_m=0.05,
                          min_points=50, min_inlier_frac=0.6)
    result = extract_roof_plane(pv, footprints, others, pts, cfg, seed=0)
    assert result.flag == "roof_no_consensus"
    assert result.usable is False


def test_collar_overrides_a_ring_fit_from_the_far_facet() -> None:
    """The guard that makes the looser consensus floor safe.

    Ring points are mostly the *far* facet of a gable; the collar — the roof the
    array actually rests on — is the near facet. RANSAC on the ring alone would
    report the far facet (measured on real Phoenix polygons: 3 of 7 recovered
    rows were 20-45 deg off). The collar must win.
    """
    pv = box(0.0, 0.0, 2.0, 2.0)
    # Far facet: 70% of the ring, facing west. Near facet: the collar, east.
    rng = np.random.default_rng(11)
    far_xy = rng.uniform([-6.0, -1.0], [-2.0, 3.0], size=(700, 2))
    near_xy = rng.uniform([2.05, -1.0], [3.2, 3.0], size=(300, 2))
    far = np.column_stack([far_xy, _plane_z(far_xy, 30.0, 270.0, 12.0)])
    near = np.column_stack([near_xy, _plane_z(near_xy, 30.0, 90.0, 10.0)])
    ring = np.concatenate([far, near])

    cfg = RoofPlaneConfig(collar_m=1.2, collar_min_points=40, ransac_threshold_m=0.15)
    ring_only = fit_plane_ransac(ring, ransac_threshold=cfg.ransac_threshold_m,
                                 min_inlier_frac=cfg.min_inlier_frac, max_iter=200, seed=0)
    guarded = _enforce_collar_agreement(ring_only, pv, ring, cfg, seed=0)

    # Unguarded, RANSAC follows the majority to the far (west-facing) facet.
    assert abs(((ring_only.azimuth_deg - 270.0 + 180) % 360) - 180) < 5.0
    # Guarded, we get the facet the array is resting on.
    assert abs(((guarded.azimuth_deg - 90.0 + 180) % 360) - 180) < 5.0
    assert guarded.tilt_deg == pytest.approx(30.0, abs=1.0)


def test_looser_consensus_floor_recovers_a_gable_adjacent_array() -> None:
    """The change this whole guard exists to make safe: an array whose ring
    catches a slice of the neighbouring facet used to be rejected outright at
    the panel fit's 0.6 floor, losing its roof reference (and with it any
    panel-standoff vintage screen). At 0.4 it is recovered — on the correct
    facet, because the collar is clean."""
    # Ridge at x=5; the east facet is the short one, so a 3 m ring around an
    # array sitting on it pulls in more of the west facet than the east.
    pv = box(5.3, 4.0, 7.2, 6.0)
    footprints = gpd.GeoDataFrame(
        {"building_id": ["b1"]}, geometry=[box(0.0, 0.0, 7.5, 10.0)], crs="EPSG:6341"
    )
    others = gpd.GeoDataFrame(geometry=[], crs="EPSG:6341")
    xy = _grid((0, 0, 7.5, 10), step=0.2, seed=0)
    z = np.where(
        xy[:, 0] < 5.0,
        _plane_z(xy, 30.0, 270.0, 10.0),
        _plane_z(xy, 30.0, 90.0, 10.0),
    )
    pts = np.column_stack([xy, z])
    base = dict(buffer_m=3.0, buffer_max_m=5.0, ransac_threshold_m=0.05, min_points=50)

    strict = extract_roof_plane(pv, footprints, others, pts,
                                RoofPlaneConfig(**base, min_inlier_frac=0.6), seed=0)
    loose = extract_roof_plane(pv, footprints, others, pts,
                               RoofPlaneConfig(**base, min_inlier_frac=0.4), seed=0)

    assert strict.usable is False
    assert strict.flag == "roof_no_consensus"
    assert loose.usable is True
    assert loose.fit.tilt_deg == pytest.approx(30.0, abs=1.0)
    assert abs(((loose.fit.azimuth_deg - 90.0 + 180) % 360) - 180) < 5.0


def test_sliver_overlap_is_off_building() -> None:
    """Edge-clipping a footprint by a few percent must NOT set on_building —
    routing edge-adjacent ground mounts / carports down the rooftop rules was
    a top rooftop-vs-canopy confusion channel."""
    pv = box(0.0, 0.0, 10.0, 10.0)              # 5% overlap with the footprint
    footprints = gpd.GeoDataFrame(
        {"building_id": ["b1"]},
        geometry=[box(9.5, 0.0, 20.0, 10.0)],
        crs="EPSG:6341",
    )
    others = gpd.GeoDataFrame(geometry=[], crs="EPSG:6341")
    result = extract_roof_plane(pv, footprints, others, np.zeros((10, 3)), RoofPlaneConfig())
    assert result.on_building is False
    assert result.fit is None
    assert result.building_id is None


def test_majority_overlap_is_on_building() -> None:
    """60% overlap clears the default 0.5 gate (tolerates ML-footprint
    misregistration); with no usable ring points the result is still
    on_building + roof_insufficient."""
    pv = box(0.0, 0.0, 10.0, 10.0)
    footprints = gpd.GeoDataFrame(
        {"building_id": ["b1"]},
        geometry=[box(4.0, 0.0, 20.0, 10.0)],   # covers 60% of the PV
        crs="EPSG:6341",
    )
    others = gpd.GeoDataFrame(geometry=[], crs="EPSG:6341")
    result = extract_roof_plane(pv, footprints, others, np.zeros((0, 3)), RoofPlaneConfig())
    assert result.on_building is True
    assert result.flag == "roof_insufficient"


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
