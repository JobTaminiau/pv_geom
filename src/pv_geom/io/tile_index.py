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

from pv_geom.io._localize import is_remote, localize


def load_tile_index(
    uri: str | Path,
    target_crs: str,
) -> gpd.GeoDataFrame:
    """Load the tile-index dataset and reproject to ``target_crs``."""
    s = str(uri)
    suffix = s.lower().split("?", 1)[0]

    if suffix.endswith((".parquet", ".geoparquet")):
        # Parquet reads s3:// transparently via fsspec.
        gdf = gpd.read_parquet(s)
    elif suffix.endswith(".zip"):
        local = localize(s) if is_remote(s) else Path(s)
        gdf = gpd.read_file(f"zip://{local}")
    elif is_remote(s):
        local = localize(s)
        gdf = gpd.read_file(local)
    else:
        gdf = gpd.read_file(s)

    if gdf.crs is not None and str(gdf.crs).lower() != str(target_crs).lower():
        gdf = gdf.to_crs(target_crs)
    return gdf.reset_index(drop=True)


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
            base_uri="s3://asu-nsf-phoenix/data/lidar_data",
            name_template="USGS_LPC_AZ_MaricopaPinal_2020_B20_{name}.laz",
        )
    """
    base = base_uri.rstrip("/")
    out = tindex.copy()
    out[out_col] = [
        f"{base}/{name_template.format(name=v)}" for v in tindex[name_col]
    ]
    return out
