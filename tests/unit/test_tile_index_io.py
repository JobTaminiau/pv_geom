"""Unit tests for ``io.tile_index``. M2."""

from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from pv_geom.io.tile_index import build_tile_uris, load_tile_index


def _make_tindex(rows: int = 3) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"Name": [f"w{432+i:04d}n3719" for i in range(rows)]},
        geometry=[box(i * 1000, 0, (i + 1) * 1000, 1000) for i in range(rows)],
        crs="EPSG:6341",
    )


def test_loads_geoparquet(tmp_path: Path) -> None:
    p = tmp_path / "tindex.parquet"
    _make_tindex().to_parquet(p)
    out = load_tile_index(p, target_crs="EPSG:6341")
    assert len(out) == 3
    assert "Name" in out.columns


def test_loads_geopackage(tmp_path: Path) -> None:
    p = tmp_path / "tindex.gpkg"
    _make_tindex().to_file(p, driver="GPKG")
    out = load_tile_index(p, target_crs="EPSG:6341")
    assert len(out) == 3


def test_loads_shapefile(tmp_path: Path) -> None:
    p = tmp_path / "tindex.shp"
    _make_tindex().to_file(p)
    out = load_tile_index(p, target_crs="EPSG:6341")
    assert len(out) == 3


def test_loads_zipped_shapefile(tmp_path: Path) -> None:
    """Phoenix's index ships as a zipped SHP — must read via zip:// virtual path."""
    shp_dir = tmp_path / "shp"
    shp_dir.mkdir()
    _make_tindex().to_file(shp_dir / "tindex.shp")
    zip_path = tmp_path / "tindex.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for f in shp_dir.iterdir():
            z.write(f, arcname=f.name)
    out = load_tile_index(zip_path, target_crs="EPSG:6341")
    assert len(out) == 3
    assert "Name" in out.columns


def test_reprojects(tmp_path: Path) -> None:
    p = tmp_path / "tindex.parquet"
    _make_tindex().to_parquet(p)
    out = load_tile_index(p, target_crs="EPSG:4326")
    assert out.crs.to_epsg() == 4326


def test_build_tile_uris_phoenix_pattern() -> None:
    tindex = _make_tindex(2)
    out = build_tile_uris(
        tindex,
        base_uri="s3://asu-nsf-phoenix/data/lidar_data",
        name_template="USGS_LPC_AZ_MaricopaPinal_2020_B20_{name}.laz",
    )
    assert list(out["tile_path"]) == [
        "s3://asu-nsf-phoenix/data/lidar_data/USGS_LPC_AZ_MaricopaPinal_2020_B20_w0432n3719.laz",
        "s3://asu-nsf-phoenix/data/lidar_data/USGS_LPC_AZ_MaricopaPinal_2020_B20_w0433n3719.laz",
    ]


def test_build_tile_uris_strips_trailing_slash() -> None:
    tindex = _make_tindex(1)
    out = build_tile_uris(tindex, base_uri="s3://b/p/", name_template="{name}.laz")
    assert out.iloc[0]["tile_path"] == "s3://b/p/w0432n3719.laz"
