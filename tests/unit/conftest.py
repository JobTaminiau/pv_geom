"""Fixtures shared by the unit tests."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from .scenes import write_synthetic_laz


@pytest.fixture
def synth_inputs(tmp_path: Path) -> dict[str, Path]:
    """Build an end-to-end synthetic dataset under tmp_path."""
    laz = tmp_path / "tile.laz"
    write_synthetic_laz(laz)

    polygons = gpd.GeoDataFrame(
        {"polygon_id": ["panel_1"]},
        geometry=[box(40.0, 40.0, 50.0, 50.0)],
        crs="EPSG:6341",
    )
    polygons_path = tmp_path / "polys.parquet"
    polygons.to_parquet(polygons_path)

    footprints = gpd.GeoDataFrame(
        {"building_id": ["b1"]},
        geometry=[box(35.0, 35.0, 65.0, 65.0)],
        crs="EPSG:6341",
    )
    footprints_path = tmp_path / "fp.parquet"
    footprints.to_parquet(footprints_path)

    tindex = gpd.GeoDataFrame(
        {"Name": ["t1"]},
        geometry=[box(0.0, 0.0, 100.0, 100.0)],
        crs="EPSG:6341",
    )
    tindex_path = tmp_path / "tindex.parquet"
    tindex.to_parquet(tindex_path)

    out_dir = tmp_path / "out"
    return {
        "polygons": polygons_path,
        "footprints": footprints_path,
        "tindex": tindex_path,
        "laz_dir": tmp_path,
        "out": out_dir,
    }
