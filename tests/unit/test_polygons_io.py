"""Unit tests for ``io.polygons``. M2."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from pv_geom.io.polygons import read_polygons


def _write_parquet(tmp_path: Path, gdf: gpd.GeoDataFrame, name: str = "polys.parquet") -> Path:
    p = tmp_path / name
    gdf.to_parquet(p)
    return p


def test_reads_with_polygon_id_column(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame(
        {"polygon_id": ["a", "b"], "extra": [1, 2]},
        geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)],
        crs="EPSG:4326",
    )
    out = read_polygons(_write_parquet(tmp_path, gdf), target_crs="EPSG:6341")
    assert list(out["polygon_id"]) == ["a", "b"]
    assert out.crs.to_epsg() == 6341
    # passthrough column survives
    assert list(out["extra"]) == [1, 2]


def test_aliases_detection_id(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame(
        {"detection_id": ["det1", "det2"]},
        geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)],
        crs="EPSG:4326",
    )
    out = read_polygons(_write_parquet(tmp_path, gdf), target_crs="EPSG:4326")
    assert list(out["polygon_id"]) == ["det1", "det2"]


def test_missing_id_raises(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="polygon_id"):
        read_polygons(_write_parquet(tmp_path, gdf), target_crs="EPSG:4326")


def test_explodes_multipolygon(tmp_path: Path) -> None:
    multi = MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])
    single = box(5, 5, 6, 6)
    gdf = gpd.GeoDataFrame(
        {"polygon_id": ["multi", "single"]},
        geometry=[multi, single],
        crs="EPSG:4326",
    )
    out = read_polygons(_write_parquet(tmp_path, gdf), target_crs="EPSG:4326")
    # 2 parts of `multi` + 1 single = 3 rows
    assert len(out) == 3
    # Multi parts get __p0 / __p1; single keeps its id
    assert set(out["polygon_id"]) == {"multi__p0", "multi__p1", "single"}
    # parent_polygon_id preserves the original id
    assert set(out["parent_polygon_id"]) == {"multi", "single"}
    assert all(g.geom_type == "Polygon" for g in out.geometry)


def test_explode_off_keeps_multipolygon(tmp_path: Path) -> None:
    multi = MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])
    gdf = gpd.GeoDataFrame(
        {"polygon_id": ["m"]}, geometry=[multi], crs="EPSG:4326"
    )
    out = read_polygons(
        _write_parquet(tmp_path, gdf), target_crs="EPSG:4326",
        explode_multipolygons=False,
    )
    assert len(out) == 1
    assert out.iloc[0].geometry.geom_type == "MultiPolygon"
    assert out.iloc[0]["polygon_id"] == "m"
    assert out.iloc[0]["parent_polygon_id"] == "m"


def test_bbox_filter(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame(
        {"polygon_id": ["a", "b", "c"]},
        geometry=[box(0, 0, 1, 1), box(50, 50, 51, 51), box(100, 100, 101, 101)],
        crs="EPSG:6341",
    )
    out = read_polygons(
        _write_parquet(tmp_path, gdf),
        target_crs="EPSG:6341",
        bbox=(40, 40, 60, 60),
    )
    assert list(out["polygon_id"]) == ["b"]


def test_max_polygons(tmp_path: Path) -> None:
    geoms = [box(i, 0, i + 1, 1) for i in range(10)]
    gdf = gpd.GeoDataFrame(
        {"polygon_id": [f"p{i}" for i in range(10)]},
        geometry=geoms, crs="EPSG:4326",
    )
    out = read_polygons(
        _write_parquet(tmp_path, gdf), target_crs="EPSG:4326", max_polygons=3
    )
    assert len(out) == 3
