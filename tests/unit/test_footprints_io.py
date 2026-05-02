"""Unit tests for ``io.footprints``. M2."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from pv_geom.io.footprints import read_footprints


def _write_parquet(tmp_path: Path, gdf: gpd.GeoDataFrame, name: str = "fp.parquet") -> Path:
    p = tmp_path / name
    gdf.to_parquet(p)
    return p


def test_uses_existing_building_id(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame(
        {"building_id": ["b_existing"]}, geometry=[box(0, 0, 5, 5)], crs="EPSG:4326"
    )
    out = read_footprints(_write_parquet(tmp_path, gdf), target_crs="EPSG:6341")
    assert list(out["building_id"]) == ["b_existing"]
    assert out.crs.to_epsg() == 6341


def test_normalizes_fema_build_id(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame(
        {"build_id": [12345, 67890]},
        geometry=[box(0, 0, 5, 5), box(10, 10, 15, 15)],
        crs="EPSG:4326",
    )
    out = read_footprints(_write_parquet(tmp_path, gdf), target_crs="EPSG:4326")
    # FEMA's build_id mapped to canonical building_id, as strings
    assert list(out["building_id"]) == ["12345", "67890"]


def test_auto_generates_when_absent(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame(
        geometry=[box(i, 0, i + 1, 1) for i in range(3)], crs="EPSG:4326"
    )
    out = read_footprints(_write_parquet(tmp_path, gdf), target_crs="EPSG:4326")
    assert list(out["building_id"]) == ["auto_0", "auto_1", "auto_2"]


def test_no_id_no_auto_raises(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="building_id"):
        read_footprints(
            _write_parquet(tmp_path, gdf), target_crs="EPSG:4326", auto_id=False
        )


def test_bbox_filter_in_target_crs(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame(
        {"build_id": ["a", "b", "c"]},
        geometry=[box(0, 0, 1, 1), box(50, 50, 51, 51), box(100, 100, 101, 101)],
        crs="EPSG:6341",
    )
    out = read_footprints(
        _write_parquet(tmp_path, gdf),
        target_crs="EPSG:6341",
        bbox=(40, 40, 60, 60),
    )
    assert list(out["building_id"]) == ["b"]


def test_reads_geopackage(tmp_path: Path) -> None:
    p = tmp_path / "fp.gpkg"
    gdf = gpd.GeoDataFrame(
        {"building_id": ["x"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326"
    )
    gdf.to_file(p, driver="GPKG")
    out = read_footprints(p, target_crs="EPSG:4326")
    assert list(out["building_id"]) == ["x"]
