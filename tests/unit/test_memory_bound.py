"""Milestone 0.5b (H5): worker memory is bounded whatever the tile."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pv_geom
from pv_geom.config import PVGeomConfig
from pv_geom.io.lidar import TileStream, read_tile
from pv_geom.pipeline import pointpool
from pv_geom.pipeline.pointpool import GroupOverBudget, budget_points, load_group_points
from pv_geom.sample import write_demo, write_scene

_COMPARE = ["polygon_id", "status", "geometry_basis", "tilt_deg", "azimuth_deg",
            "fit_rmse_m", "n_points", "n_inliers", "roof_tilt_deg",
            "height_above_roof_m", "height_above_ground_m", "lidar_date", "n_facets"]


def test_a_tile_read_in_chunks_is_the_same_tile(tmp_path: Path) -> None:
    tile = write_scene(tmp_path) / "bench.laz"
    whole = read_tile(tile)
    stream = TileStream(tile, chunk_points=5_000)
    blocks = list(stream)
    assert len(blocks) > 3 and max(len(b) for b in blocks) <= 5_000
    assert np.array_equal(np.concatenate(blocks), whole.points)
    assert stream.crs == whole.crs and stream.n_points == len(whole.points)
    assert stream.vintage.flight_end == whole.vintage.flight_end
    assert stream.vintage.flight_start == whole.vintage.flight_start


def test_missing_tile_is_reported_as_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TileStream(tmp_path / "nope.laz")


def test_budget_follows_the_configured_memory() -> None:
    cfg = PVGeomConfig()
    cfg.compute.memory_budget_gb = 6.0
    six = budget_points(cfg)
    cfg.compute.memory_budget_gb = 12.0
    assert budget_points(cfg) > 2 * six * 0.95
    cfg.compute.memory_budget_gb = None
    assert budget_points(cfg) is None


def test_loading_stops_as_soon_as_the_budget_is_passed(tmp_path: Path) -> None:
    import geopandas as gpd

    scene = write_scene(tmp_path)
    polys = gpd.read_parquet(scene / "polygons.parquet")
    cfg = PVGeomConfig()
    cfg.crs.target = str(polys.crs)
    cfg.compute.lidar_chunk_points = 5_000
    args = ({"bench": str(scene / "bench.laz")}, "bench", ("bench",), polys.geometry.to_numpy(),
            cfg)
    assert load_group_points(*args) is not None
    with pytest.raises(GroupOverBudget):
        load_group_points(*args, max_points=2_000)


@pytest.mark.parametrize("use_dask", [False])
def test_results_do_not_depend_on_the_memory_budget(tmp_path: Path, monkeypatch,
                                                    use_dask: bool) -> None:
    """The acceptance test for H5. A budget far below what the demo's one tile
    group holds forces it to be measured in many spatial batches; every row
    must come out exactly as it does unbounded."""
    free = pv_geom.run(write_demo(tmp_path / "free"), use_dask=use_dask)

    splits = []
    original = pointpool.load_group_points

    def counting(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except GroupOverBudget:
            splits.append(1)
            raise

    monkeypatch.setattr("pv_geom.pipeline.worker.load_group_points", counting)
    monkeypatch.setattr("pv_geom.pipeline.worker.budget_points", lambda cfg: 4_000)
    tight = pv_geom.run(write_demo(tmp_path / "tight"), use_dask=use_dask)
    assert len(splits) >= 3                       # the group really was split, repeatedly

    a = pd.DataFrame(free.load().drop(columns="geometry"))[_COMPARE]
    b = pd.DataFrame(tight.load().drop(columns="geometry"))[_COMPARE]
    pd.testing.assert_frame_equal(a, b)
    assert list(free.load()["flags"].map(list)) == list(tight.load()["flags"].map(list))


def test_one_polygon_is_measured_even_if_it_alone_exceeds_the_budget(tmp_path: Path,
                                                                    monkeypatch) -> None:
    monkeypatch.setattr("pv_geom.pipeline.worker.budget_points", lambda cfg: 1)
    run = pv_geom.run(write_demo(tmp_path), use_dask=False)
    assert (run.load()["status"] == "measured").sum() >= 5
