"""Unit tests for ``pipeline.partition``. M6 (PRD §6.2)."""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import box

from pv_geom.pipeline.partition import (
    assign_polygons_to_tiles,
    build_tile_groups,
)


def _tindex(tiles: list[tuple[str, tuple[float, float, float, float]]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"Name": [n for n, _ in tiles]},
        geometry=[box(*b) for _, b in tiles],
        crs="EPSG:6341",
    )


def _polys(rows: list[tuple[str, tuple[float, float, float, float]]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"polygon_id": [n for n, _ in rows]},
        geometry=[box(*b) for _, b in rows],
        crs="EPSG:6341",
    )


def test_simple_inside_one_tile() -> None:
    tindex = _tindex([("t1", (0, 0, 100, 100)), ("t2", (100, 0, 200, 100))])
    polys = _polys([("p1", (10, 10, 12, 12)), ("p2", (110, 10, 112, 12))])
    out = assign_polygons_to_tiles(polys, tindex)
    pri = dict(zip(out["polygon_id"], out["primary_tile_id"]))
    assert pri == {"p1": "t1", "p2": "t2"}
    overlaps = dict(zip(out["polygon_id"], out["overlapping_tile_ids"]))
    assert overlaps == {"p1": ("t1",), "p2": ("t2",)}


def test_polygon_spans_two_tiles() -> None:
    tindex = _tindex([("t1", (0, 0, 100, 100)), ("t2", (100, 0, 200, 100))])
    # PV polygon centered in t1 but extending into t2
    polys = _polys([("p1", (90, 10, 130, 50))])
    out = assign_polygons_to_tiles(polys, tindex)
    # centroid is x=110 -> t2 is primary
    assert out.iloc[0]["primary_tile_id"] == "t2"
    # both tiles intersect -> overlap union has both
    assert set(out.iloc[0]["overlapping_tile_ids"]) == {"t1", "t2"}


def test_polygon_off_the_grid_is_dropped() -> None:
    tindex = _tindex([("t1", (0, 0, 100, 100))])
    polys = _polys([("p1", (10, 10, 12, 12)), ("p2", (500, 500, 502, 502))])
    out = assign_polygons_to_tiles(polys, tindex)
    assert list(out["polygon_id"]) == ["p1"]


def test_crs_mismatch_raises() -> None:
    t = gpd.GeoDataFrame({"Name": ["t"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:6341")
    p = gpd.GeoDataFrame({"polygon_id": ["p"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="CRS"):
        assign_polygons_to_tiles(p, t)


def test_build_tile_groups_groups_by_primary() -> None:
    tindex = _tindex([
        ("t1", (0, 0, 100, 100)), ("t2", (100, 0, 200, 100)),
        ("t3", (0, 100, 100, 200)),
    ])
    polys = _polys([
        ("p1", (10, 10, 12, 12)),     # primary t1
        ("p2", (15, 15, 17, 17)),     # primary t1
        ("p3", (110, 10, 112, 12)),   # primary t2
        ("p4", (85, 10, 105, 12)),    # primary t1 (centroid x=95), overlaps t1+t2
    ])
    out = assign_polygons_to_tiles(polys, tindex)
    groups = build_tile_groups(out)
    by_pri = {g.primary_tile_id: g for g in groups}
    assert set(by_pri) == {"t1", "t2"}
    assert set(by_pri["t1"].polygon_ids) == {"p1", "p2", "p4"}
    # p4 spans t1+t2, so the t1 group fetches both
    assert set(by_pri["t1"].fetch_tile_ids) == {"t1", "t2"}
    assert set(by_pri["t2"].polygon_ids) == {"p3"}


def test_centroid_just_inside_one_tile() -> None:
    """A polygon nearly straddling the boundary still gets a single primary tile,
    determined by where its centroid actually falls. (Centroids exactly on a
    boundary are measure-zero in continuous coordinates and are dropped — that's
    expected behaviour for ``within``.)"""
    tindex = _tindex([("t1", (0, 0, 100, 100)), ("t2", (100, 0, 200, 100))])
    polys = _polys([("p1", (97, 0, 103, 4))])    # centroid at (100, 2) — boundary
    boundary = assign_polygons_to_tiles(polys, tindex)
    assert len(boundary) == 0          # centroid exactly on boundary -> dropped

    polys = _polys([("p2", (98, 0, 103, 4))])    # centroid at (100.5, 2) -> t2
    inside = assign_polygons_to_tiles(polys, tindex)
    assert len(inside) == 1
    assert inside.iloc[0]["primary_tile_id"] == "t2"
    # overlap captures both tiles since the polygon spans the boundary
    assert set(inside.iloc[0]["overlapping_tile_ids"]) == {"t1", "t2"}
