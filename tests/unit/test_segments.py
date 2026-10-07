"""Milestone 0.5 (E1): polygons that cover more than one roof face."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

import pv_geom
from pv_geom.config import PVGeomConfig
from pv_geom.pipeline.worker import build_row
from pv_geom.sample import write_demo

CRS = "EPSG:6341"


def _plane_z(xy: np.ndarray, tilt: float, az: float, z0: float, at: tuple[float, float]):
    t, a = np.radians(tilt), np.radians(az)
    nx, ny, nz = np.sin(t) * np.sin(a), np.sin(t) * np.cos(a), np.cos(t)
    return z0 - (nx * (xy[:, 0] - at[0]) + ny * (xy[:, 1] - at[1])) / nz


def _row(pts: np.ndarray, polygon=None, cfg: PVGeomConfig | None = None) -> dict:
    return build_row(
        polygon=polygon or box(0, 0, 10, 6), polygon_id="p", cfg=cfg or PVGeomConfig(),
        config_hash="x", run_id="r", partition_id=0,
        panel_pts=pts, ground_xyz=np.zeros((0, 3)), roof_input_pts=pts, footprints=None,
        other_pv_polygons=gpd.GeoDataFrame(geometry=[], crs=CRS), contributing_tile_ids=("t",),
    )


def _gable(split_x: float, n: int = 700, noise: float = 0.01, seed: int = 4) -> np.ndarray:
    """Returns over a polygon lying across a ridge at ``x = split_x``: the west
    part faces west, the east part faces east, both at 25 degrees."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform([0.3, 0.3], [9.7, 5.7], size=(n, 2))
    west = xy[:, 0] < split_x
    z = np.where(west, _plane_z(xy, 25.0, 270.0, 8.0, (split_x, 3.0)),
                 _plane_z(xy, 25.0, 90.0, 8.0, (split_x, 3.0)))
    return np.column_stack([xy, z + rng.normal(0, noise, n)])


def test_polygon_over_a_ridge_is_measured_as_two_facets() -> None:
    """An even split has no dominant plane. It used to be a `no_fit`; it is now
    two facets, each with its own tilt and azimuth."""
    row = _row(_gable(split_x=5.0))
    assert row["status"] == "measured" and row["n_planes_detected"] == 2
    assert "multi_facet" in row["flags"] and "poor_fit" not in row["flags"]

    segs = row["segments"]
    assert [s["segment_index"] for s in segs] == [0, 1]
    assert sum(s["area_share"] for s in segs) == pytest.approx(1.0)
    assert sum(s["area_m2"] for s in segs) == pytest.approx(row["area_m2"], rel=1e-4)
    assert sorted(round(float(s["azimuth_deg"])) for s in segs) == [90, 270]
    for s in segs:
        assert s["tilt_deg"] == pytest.approx(25.0, abs=0.5)
        assert 0.4 < s["area_share"] < 0.6
        assert s["tilt_unc_deg"] < 1.0 and s["n_points"] > 200
    # The polygon-level columns describe the primary (larger) facet.
    assert row["panel_tilt_deg"] == pytest.approx(segs[0]["tilt_deg"])
    assert segs[0]["n_points"] >= segs[1]["n_points"]
    # Facing opposite ways at equal tilt is also the east-west signature.
    assert "east_west_rack" in row["flags"]


def test_facet_footprints_partition_the_polygon() -> None:
    from shapely import from_wkb

    row = _row(_gable(split_x=5.0))
    west, east = sorted((from_wkb(s["geometry"]) for s in row["segments"]),
                        key=lambda g: g.centroid.x)
    assert west.centroid.x < 4.0 < 6.0 < east.centroid.x
    assert west.within(box(0, 0, 10, 6).buffer(1e-6))
    assert west.area + east.area == pytest.approx(60.0, rel=0.15)


def test_dominant_facet_with_a_minor_one() -> None:
    """A 75/25 split: the big face is the primary at the base tolerance, and the
    minor face is still found."""
    row = _row(_gable(split_x=7.5))
    assert row["n_planes_detected"] == 2
    shares = [s["area_share"] for s in row["segments"]]
    assert shares[0] == pytest.approx(0.75, abs=0.05) and shares[1] == pytest.approx(0.25, abs=0.05)
    assert round(float(row["panel_azimuth_deg"])) == 270


def test_one_noisy_plane_is_not_sliced_into_parallel_facets() -> None:
    """Sequential fitting would peel a noisy surface into layers. They share an
    orientation, so they are one facet — fitted at a wider tolerance."""
    rng = np.random.default_rng(1)
    xy = rng.uniform([0.3, 0.3], [9.7, 5.7], size=(600, 2))
    z = _plane_z(xy, 30.0, 180.0, 8.0, (5.0, 3.0)) + rng.normal(0, 0.075, 600)
    row = _row(np.column_stack([xy, z]))
    assert row["n_planes_detected"] == 1 and "multi_facet" not in row["flags"]
    assert "wide_tolerance_fit" in row["flags"]
    assert row["panel_tilt_deg"] == pytest.approx(30.0, abs=1.5)
    assert len(row["segments"]) == 1 and row["segments"][0]["area_share"] == pytest.approx(1.0)


def test_a_wall_inside_the_polygon_is_not_a_facet() -> None:
    """Returns off a wall or parapet inside an oversized polygon form a
    near-vertical plane. It is not an array and must not become a segment."""
    rng = np.random.default_rng(2)
    xy = rng.uniform([0.3, 0.3], [9.7, 5.7], size=(500, 2))
    roof = np.column_stack([xy, _plane_z(xy, 20.0, 180.0, 8.0, (5.0, 3.0))])
    wall_y = rng.uniform(0.3, 5.7, 200)
    wall = np.column_stack([np.full(200, 9.5) + rng.normal(0, 0.005, 200), wall_y,
                            rng.uniform(4.0, 7.5, 200)])
    row = _row(np.concatenate([roof, wall]))
    assert row["n_planes_detected"] == 1
    assert row["panel_tilt_deg"] == pytest.approx(20.0, abs=0.5)


def test_single_plane_array_has_exactly_one_segment() -> None:
    rng = np.random.default_rng(3)
    xy = rng.uniform([0.3, 0.3], [9.7, 5.7], size=(500, 2))
    row = _row(np.column_stack([xy, _plane_z(xy, 18.0, 200.0, 8.0, (5.0, 3.0))]))
    (seg,) = row["segments"]
    assert seg["segment_index"] == 0 and seg["area_share"] == pytest.approx(1.0)
    assert seg["tilt_deg"] == pytest.approx(row["panel_tilt_deg"])
    assert seg["azimuth_deg"] == pytest.approx(row["panel_azimuth_deg"])
    assert seg["surface_area_m2"] == pytest.approx(row["surface_area_m2"], rel=1e-4)


def test_segments_can_be_switched_off() -> None:
    cfg = PVGeomConfig()
    cfg.multi_plane.enabled = False
    row = _row(_gable(split_x=5.0), cfg=cfg)
    assert row["n_planes_detected"] <= 1 and "multi_facet" not in row["flags"]


def test_unfitted_polygon_has_no_segments() -> None:
    rng = np.random.default_rng(5)
    clutter = rng.uniform([0.3, 0.3, 0.0], [9.7, 5.7, 6.0], size=(400, 3))
    row = _row(clutter)
    assert row["status"] == "no_fit" and row["segments"] is None
    assert row["n_planes_detected"] == 0


def test_release_dataset_carries_a_flat_segments_table(tmp_path: Path) -> None:
    result = pv_geom.run(write_demo(tmp_path), use_dask=False)
    report = result.report()
    gdf = result.load()
    measured = gdf[gdf["status"] == "measured"]

    segs = gpd.read_parquet(report.dataset_dir / "pv_geom_segments.parquet")
    assert len(segs) == int(measured["n_planes_detected"].sum())
    assert segs["segment_id"].is_unique and segs.crs == gdf.crs
    assert set(segs["polygon_id"]) == set(measured["polygon_id"])
    assert {"tilt_deg", "azimuth_deg", "area_share", "geometry_basis", "status"} <= set(segs.columns)
    # Facet areas add back up to their polygons.
    per_polygon = segs.groupby("polygon_id")["area_m2"].sum()
    assert np.allclose(per_polygon.to_numpy(),
                       measured.set_index("polygon_id").loc[per_polygon.index, "area_m2"],
                       rtol=1e-4)
    assert (report.dataset_dir / "pv_geom_segments.csv").exists()
    header = (report.dataset_dir / "pv_geom.csv").read_text().splitlines()[0]
    assert "segments" not in header.split(",")
    assert (report.tables_dir / "facets.csv").exists()
