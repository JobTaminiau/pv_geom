"""0.2.0 input model: vintages, geometry basis, optional footprints/ids, CRS."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq
import pytest
from shapely.geometry import box

from pv_geom.config import PVGeomConfig
from pv_geom.geometry.point_index import GroundModel, PointGrid
from pv_geom.pipeline.runner import run_pipeline
from pv_geom.schema import OUTPUT_SCHEMA, data_dictionary
from pv_geom.utils.crs import assert_metric_projected, resolve_target_crs
from pv_geom.vintage import (
    NO_FIT,
    PANEL_BY_VINTAGE,
    PANEL_CONFIRMED,
    SURFACE_UNRESOLVED,
    UNSCREENED,
    geometry_basis,
    parse_vintage,
    vintage_gap_days,
)
from .test_runner_smoke import _laz_flown_on, _rooftop_scene, _write_synthetic_laz

# --------------------------------------------------------------------------- #
# Vintage parsing + geometry basis
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2024", date(2024, 12, 31)),          # a year is a window: take its last day
        (2024, date(2024, 12, 31)),
        ("2024-02", date(2024, 2, 29)),        # leap year
        ("2024-04-01", date(2024, 4, 1)),
        (date(2023, 3, 31), date(2023, 3, 31)),
        (None, None),
        ("", None),
    ],
)
def test_parse_vintage(value, expected) -> None:
    assert parse_vintage(value) == expected


def test_parse_vintage_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="cannot parse vintage"):
        parse_vintage("spring 2024")


def test_vintage_gap_sign() -> None:
    """Positive = polygons newer than the LiDAR = installations may be missing."""
    assert vintage_gap_days(date(2024, 4, 1), date(2020, 11, 28)) == 1220
    assert vintage_gap_days(date(2020, 1, 1), date(2020, 11, 28)) < 0
    assert vintage_gap_days(None, date(2020, 11, 28)) is None


@pytest.mark.parametrize(
    "fit_ok, passed, screened, gap, expected",
    [
        (False, False, False, 100, NO_FIT),
        (True, True, True, 1000, PANEL_CONFIRMED),      # physical evidence wins over dates
        (True, False, True, -5, PANEL_BY_VINTAGE),      # flush array, LiDAR is the newer input
        (True, False, False, 0, PANEL_BY_VINTAGE),      # no roof reference, but co-temporal
        (True, False, True, 1000, SURFACE_UNRESOLVED),  # could be the bare roof
        (True, False, False, 1000, UNSCREENED),
        (True, False, True, None, SURFACE_UNRESOLVED),  # unknown dates earn no credit
        (True, False, False, None, UNSCREENED),
    ],
)
def test_geometry_basis(fit_ok, passed, screened, gap, expected) -> None:
    assert geometry_basis(fit_ok=fit_ok, standoff_passed=passed,
                          standoff_screened=screened, gap_days=gap) == expected


def test_input_epoch_is_still_accepted() -> None:
    """Configs written before 0.2.0 call the polygon vintage `input_epoch`."""
    cfg = PVGeomConfig(vintage={"input_epoch": "2024-04-01"})
    assert parse_vintage(cfg.vintage.polygon_vintage) == date(2024, 4, 1)


def test_old_config_blocks_are_ignored() -> None:
    PVGeomConfig(output={"partition_size": 100000}, heights={"use_whitebox_dem": False})


# --------------------------------------------------------------------------- #
# Row-level basis from a physical scene
# --------------------------------------------------------------------------- #


def _row(standoff_m: float, **kw):
    from pv_geom.pipeline.tile_task import _build_row

    return _build_row(polygon_id="p", cfg=PVGeomConfig(), config_hash="x", run_id="r",
                      partition_id=0, contributing_tile_ids=("t1",),
                      **_rooftop_scene(standoff_m=standoff_m), **kw)


def test_row_panel_confirmed_despite_newer_polygons() -> None:
    row = _row(0.10, polygon_vintage=date(2024, 4, 1), lidar_date=date(2020, 11, 28))
    assert row["geometry_basis"] == PANEL_CONFIRMED
    assert row["vintage_gap_days"] == 1220
    assert row["roof_ref_source"] == "footprint_ring"


def test_row_bare_roof_is_unresolved_when_polygons_are_newer() -> None:
    """The motivating case: imagery after the LiDAR, no panel in the cloud."""
    row = _row(0.0, polygon_vintage=date(2024, 4, 1), lidar_date=date(2020, 11, 28))
    assert row["geometry_basis"] == SURFACE_UNRESOLVED
    assert "no_panel_standoff" in row["flags"]


def test_row_flush_array_credited_when_lidar_is_newer() -> None:
    """Same geometry, but the LiDAR postdates the imagery: the array was there."""
    row = _row(0.0, polygon_vintage=date(2020, 1, 1), lidar_date=date(2023, 3, 31))
    assert row["geometry_basis"] == PANEL_BY_VINTAGE
    assert row["vintage_gap_days"] < 0


def test_row_without_dates_gets_no_vintage_credit() -> None:
    row = _row(0.0)
    assert row["geometry_basis"] == SURFACE_UNRESOLVED
    assert row["polygon_vintage"] is None and row["vintage_gap_days"] is None


def test_row_has_no_mounting_columns_by_default() -> None:
    row = _row(0.10)
    assert not {"mounting_type", "mounting_confidence", "mounting_rule"} & set(row)
    assert "tracker_suspected" not in row["flags"]


def test_row_surface_area_exceeds_plan_area_by_cos_tilt() -> None:
    row = _row(0.10)
    expected = row["area_m2"] / np.cos(np.radians(row["panel_tilt_deg"]))
    assert row["surface_area_m2"] == pytest.approx(expected, rel=1e-5)


# --------------------------------------------------------------------------- #
# Footprints optional: open-ring roof reference
# --------------------------------------------------------------------------- #


def test_open_ring_gives_a_roof_reference_without_footprints() -> None:
    from pv_geom.pipeline.tile_task import _build_row

    scene = _rooftop_scene(standoff_m=0.10)
    scene["footprints"] = None
    row = _build_row(polygon_id="p", cfg=PVGeomConfig(), config_hash="x", run_id="r",
                     partition_id=0, contributing_tile_ids=("t1",), **scene)
    assert row["on_building"] is None            # unknown, not False
    assert row["roof_ref_source"] == "open_ring"
    assert row["height_above_roof_m"] == pytest.approx(0.10, abs=0.03)
    assert row["geometry_basis"] == PANEL_CONFIRMED


def test_open_ring_can_be_disabled() -> None:
    from pv_geom.pipeline.tile_task import _build_row

    cfg = PVGeomConfig()
    cfg.roof_plane.open_ring = False
    scene = _rooftop_scene(standoff_m=0.10)
    scene["footprints"] = None
    row = _build_row(polygon_id="p", cfg=cfg, config_hash="x", run_id="r",
                     partition_id=0, contributing_tile_ids=("t1",), **scene)
    assert row["roof_ref_source"] == "none"
    assert "standoff_unscreenable" in row["flags"]


def test_open_ring_with_nothing_around_is_not_a_roof_flag() -> None:
    """A ground mount has no elevated returns around it; that is 'no reference',
    not a roof-quality problem."""
    from pv_geom.pipeline.tile_task import _build_row

    rng = np.random.default_rng(0)
    xy = rng.uniform([0, 0], [10, 4], size=(400, 2))
    pts = np.column_stack([xy, 1.5 + 0.3 * xy[:, 1] + rng.normal(0, 0.01, 400)])
    row = _build_row(polygon=box(0, 0, 10, 4), polygon_id="g", cfg=PVGeomConfig(),
                     config_hash="x", run_id="r", partition_id=0,
                     panel_pts=pts, ground_xyz=np.zeros((0, 3)), roof_input_pts=pts,
                     footprints=None,
                     other_pv_polygons=gpd.GeoDataFrame(geometry=[], crs="EPSG:6341"),
                     contributing_tile_ids=("t1",))
    assert row["roof_ref_source"] == "none"
    assert not [f for f in row["flags"] if f.startswith("roof_")]


# --------------------------------------------------------------------------- #
# Spatial index + ground model
# --------------------------------------------------------------------------- #


def test_point_grid_matches_brute_force() -> None:
    rng = np.random.default_rng(1)
    pts = rng.uniform([1000, 5000, 0], [1400, 5300, 30], size=(20000, 3))
    grid = PointGrid(pts, cell_m=10.0)
    for bbox in [(1100, 5100, 1130, 5125), (990, 4990, 1005, 5003), (1399, 5299, 1500, 5400),
                 (2000, 2000, 2100, 2100)]:
        got = grid.query_bbox(*bbox)
        m = ((pts[:, 0] >= bbox[0]) & (pts[:, 0] <= bbox[2])
             & (pts[:, 1] >= bbox[1]) & (pts[:, 1] <= bbox[3]))
        assert len(got) == int(m.sum())
        assert np.allclose(np.sort(got[:, 2]), np.sort(pts[m, 2]))


def test_point_grid_empty() -> None:
    assert len(PointGrid(np.zeros((0, 3))).query_bbox(0, 0, 1, 1)) == 0


def test_ground_model_follows_a_slope() -> None:
    """A single tile-wide ground median fails on sloping terrain; the local
    model must track it and fill a gap (a building) from its surroundings."""
    rng = np.random.default_rng(2)
    xy = rng.uniform([0, 0], [500, 500], size=(60000, 2))
    xy = xy[~((xy[:, 0] > 200) & (xy[:, 0] < 240) & (xy[:, 1] > 200) & (xy[:, 1] < 240))]
    ground = np.column_stack([xy, 0.1 * xy[:, 0]])          # 10% grade: 50 m of relief
    model = GroundModel(ground, cell_m=5.0)
    z = model.ground_z(np.array([50.0, 450.0, 220.0]), np.array([250.0, 250.0, 220.0]))
    assert z[0] == pytest.approx(5.0, abs=0.5)
    assert z[1] == pytest.approx(45.0, abs=0.5)
    assert z[2] == pytest.approx(22.0, abs=3.0)             # under the building: coarse fill


def test_fallback_keeps_panels_on_a_hillside() -> None:
    """With no building class, panel candidates are class-1 returns above
    *local* ground — a rooftop at the low end of a sloping tile must survive."""
    from pv_geom.pipeline.tile_task import _split_classes_for_tile_group

    rng = np.random.default_rng(3)
    gxy = rng.uniform([0, 0], [500, 500], size=(50000, 2))
    ground = np.column_stack([gxy, 0.1 * gxy[:, 0], np.full(len(gxy), 2)])
    low = np.column_stack([rng.uniform([20, 20], [30, 30], size=(300, 2)),
                           np.full(300, 2.5 + 3.0), np.full(300, 1)])     # 3 m above ground
    weeds = np.column_stack([rng.uniform([400, 400], [410, 410], size=(300, 2)),
                             np.full(300, 40.5 + 0.2), np.full(300, 1)])  # 0.2 m above ground
    _, panel, used = _split_classes_for_tile_group(
        np.concatenate([ground, low, weeds]), PVGeomConfig())
    assert used == 1
    assert len(panel) == 300 and panel[:, 0].max() < 100     # roof kept, weeds dropped


# --------------------------------------------------------------------------- #
# CRS
# --------------------------------------------------------------------------- #


def test_auto_crs_comes_from_the_tile_index() -> None:
    assert resolve_target_crs("auto", "EPSG:6347") == "EPSG:6347"
    assert resolve_target_crs("EPSG:6341", "EPSG:6347") == "EPSG:6341"   # explicit wins


def test_auto_crs_needs_a_projected_source() -> None:
    with pytest.raises(ValueError, match="set crs.target explicitly"):
        resolve_target_crs("auto", "EPSG:4326")


def test_non_metric_crs_is_refused() -> None:
    with pytest.raises(NotImplementedError, match="metric"):
        assert_metric_projected("EPSG:2223")       # Arizona Central, international feet


# --------------------------------------------------------------------------- #
# End to end: two inputs, no ids, no footprints, GeoJSON polygons
# --------------------------------------------------------------------------- #


@pytest.fixture
def two_input_run(tmp_path: Path) -> dict:
    laz = tmp_path / "tile.laz"
    _write_synthetic_laz(laz)
    polys = gpd.GeoDataFrame(geometry=[box(40.0, 40.0, 50.0, 50.0)], crs="EPSG:6341")
    polys_path = tmp_path / "polys.geojson"
    polys.to_crs("EPSG:4326").to_file(polys_path, driver="GeoJSON")
    tindex = gpd.GeoDataFrame({"NAME": ["t1"]}, geometry=[box(0, 0, 100, 100)],
                              crs="EPSG:6341")
    tindex_path = tmp_path / "tindex.gpkg"
    tindex.to_file(tindex_path, driver="GPKG")
    return {"polygons": polys_path, "tindex": tindex_path, "laz_dir": tmp_path,
            "out": tmp_path / "out"}


def test_runs_on_polygons_and_lidar_alone(two_input_run: dict) -> None:
    cfg = PVGeomConfig()
    cfg.vintage.polygon_vintage = "2024"
    cfg.vintage.lidar_date = "2023-03-31"
    manifest_path = run_pipeline(
        polygons_uri=str(two_input_run["polygons"]),
        tile_index_uri=str(two_input_run["tindex"]),
        lidar_prefix=str(two_input_run["laz_dir"]),
        output_uri=str(two_input_run["out"]),
        cfg=cfg, name_template="tile.laz", use_dask=False,
    )
    part = sorted(two_input_run["out"].glob("part-*.parquet"))[0]
    table = pq.read_table(part)
    assert table.schema.names == OUTPUT_SCHEMA.names
    row = table.to_pylist()[0]
    assert row["polygon_id"] == "poly_0000000" and row["input_row"] == 0
    assert row["on_building"] is None
    assert row["polygon_vintage"] == date(2024, 12, 31)
    assert row["lidar_date"] == date(2023, 3, 31)
    assert row["lidar_date_source"] == "declared"
    assert row["vintage_gap_days"] == 641
    assert row["panel_tilt_deg"] == pytest.approx(20.0, abs=1.0)
    assert row["geometry_basis"] == PANEL_CONFIRMED       # 0.5 m above the roof

    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest["crs"] == "EPSG:6341"                  # resolved from the tile index
    assert manifest["inputs"]["footprints"] is None
    assert manifest["vintage"]["vintage_gap_days"] == 641
    assert manifest["aggregate_stats"]["geometry_basis_counts"][PANEL_CONFIRMED] == 1
    assert "mounting_type_counts" not in manifest["aggregate_stats"]


def test_partitions_are_valid_geoparquet(two_input_run: dict) -> None:
    run_pipeline(
        polygons_uri=str(two_input_run["polygons"]),
        tile_index_uri=str(two_input_run["tindex"]),
        lidar_prefix=str(two_input_run["laz_dir"]),
        output_uri=str(two_input_run["out"]),
        cfg=PVGeomConfig(), name_template="tile.laz", use_dask=False,
    )
    part = sorted(two_input_run["out"].glob("part-*.parquet"))[0]
    gdf = gpd.read_parquet(part)                           # fails without `geo` metadata
    assert gdf.crs.to_epsg() == 6341
    assert gdf.geometry.iloc[0].area == pytest.approx(100.0, rel=1e-3)


def test_lidar_date_is_measured_from_gps_time_per_row(tmp_path: Path) -> None:
    """No declared LiDAR date: each row gets its own tile's flight date."""
    from pv_geom.pipeline.tile_task import process_tile_group

    laz = tmp_path / "t.laz"
    _laz_flown_on(laz, date(2020, 11, 26), n=2000)
    polys = gpd.GeoDataFrame({"polygon_id": ["a"]}, geometry=[box(1, 1, 9, 9)],
                             crs="EPSG:6341")
    table = process_tile_group(
        tile_uri_map={"t1": str(laz)}, primary_tile_id="t1", polygons=polys,
        fetch_tile_ids=("t1",), footprints=None, cfg=PVGeomConfig(),
        config_hash="x", run_id="r", partition_id=0,
        polygon_vintage=date(2024, 4, 1),
    )
    row = table.to_pylist()[0]
    assert row["lidar_date"] == date(2020, 11, 26)
    assert row["lidar_date_source"] == "gps_time"
    assert row["vintage_gap_days"] == (date(2024, 4, 1) - date(2020, 11, 26)).days


def test_per_polygon_vintage_column_overrides_run_vintage(tmp_path: Path) -> None:
    from pv_geom.io.polygons import read_polygons

    gdf = gpd.GeoDataFrame({"seen": ["2019", None]},
                           geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)], crs="EPSG:6341")
    p = tmp_path / "p.parquet"
    gdf.to_parquet(p)
    out = read_polygons(p, target_crs="EPSG:6341", vintage_col="seen")
    assert out["polygon_vintage"].tolist() == [date(2019, 12, 31), None]


def test_data_dictionary_describes_every_column() -> None:
    rows = data_dictionary()
    assert [r["column"] for r in rows] == OUTPUT_SCHEMA.names
    assert all(r["description"] for r in rows)
