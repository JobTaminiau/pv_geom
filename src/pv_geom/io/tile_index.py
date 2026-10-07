"""LiDAR tile-index loader. M2.

Auto-detects GeoParquet / GPKG / SHP / zipped-SHP. The Phoenix dataset ships
the index as a zipped shapefile (``USGS_AZ_MaricopaPinal_1_2020_TileIndex.zip``);
GDAL handles ``zip://`` virtual paths natively when the local file is on
disk. PRD §3.2 requires either a ``tile_path`` column with full S3 URIs OR a
``Name``/``filename`` column you can compose into URIs via :func:`build_tile_uris`.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from pv_geom.errors import TileIndexError
from pv_geom.io.vector import read_vector, reproject


def load_tile_index(
    uri: str | Path,
    target_crs: str | None = None,
) -> gpd.GeoDataFrame:
    """Load the tile-index dataset, reprojecting to ``target_crs`` when given
    (``None`` keeps its native CRS, which is how ``crs.target: auto`` finds it)."""
    return reproject(read_vector(uri), target_crs).reset_index(drop=True)


_TILE_ID_ALIASES = ("name", "tile_id", "tilename", "tile_name", "filename", "location")


def resolve_tile_id_col(tindex: gpd.GeoDataFrame, requested: str | None = None) -> str:
    """Find the tile-id column: the requested name (case-insensitively), else
    the first of the usual suspects. Tile indexes disagree on capitalisation
    (USGS ships ``Name``, the Delaware state index ``NAME``)."""
    lower = {c.lower(): c for c in tindex.columns}
    if requested:
        if requested in tindex.columns:
            return requested
        if requested.lower() in lower:
            return lower[requested.lower()]
        raise TileIndexError(
            f"tile index has no '{requested}' column",
            f"columns are {list(tindex.columns)}; pass one with --tile-id-col",
        )
    for alias in _TILE_ID_ALIASES:
        if alias in lower:
            return lower[alias]
    raise TileIndexError(
        f"cannot find a tile-id column among {list(tindex.columns)}",
        "name it with --tile-id-col",
    )


def build_tile_uris(
    tindex: gpd.GeoDataFrame,
    *,
    base_uri: str,
    name_col: str = "Name",
    name_template: str = "{name}.laz",
    out_col: str = "tile_path",
) -> gpd.GeoDataFrame:
    """Compose a per-row ``tile_path`` URI from ``base_uri`` + a name column.

    For Phoenix's USGS LPC bucket, call:
        build_tile_uris(
            tindex,
            base_uri="s3://free-research-data-raw/US/arizona/top-level/lidar/lidar_data",
            name_template="USGS_LPC_AZ_MaricopaPinal_2020_B20_{name}.laz",
        )
    """
    base = base_uri.rstrip("/")
    out = tindex.copy()
    out[out_col] = [
        f"{base}/{name_template.format(name=v)}" for v in tindex[name_col]
    ]
    return out
