"""One reader for every vector layer the pipeline takes.

Polygons, building footprints and the LiDAR tile index all arrive as
GeoParquet, GeoPackage, GeoJSON, Shapefile (optionally zipped) or FlatGeobuf,
locally or on S3. They are read here, the same way.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from pv_geom.errors import require
from pv_geom.io.storage import is_remote, localize

_PARQUET_SUFFIXES = (".parquet", ".geoparquet")


def read_vector(uri: str | Path) -> gpd.GeoDataFrame:
    """Read a vector layer from a local path or remote URI, in its own CRS."""
    s = str(uri)
    suffix = s.lower().split("?", 1)[0]
    if s.startswith("s3://"):
        require("s3fs", "cloud", "reading from S3")
    if suffix.endswith(_PARQUET_SUFFIXES):
        return gpd.read_parquet(s)              # remote URIs go through fsspec
    local = localize(s) if is_remote(s) else Path(s)
    if suffix.endswith(".zip"):
        return gpd.read_file(f"zip://{local}")  # zipped shapefile
    return gpd.read_file(local)


def reproject(gdf: gpd.GeoDataFrame, target_crs: str | None) -> gpd.GeoDataFrame:
    """``gdf`` in ``target_crs`` (unchanged when it already is, has no CRS, or
    no target is given)."""
    if target_crs is None or gdf.crs is None:
        return gdf
    if str(gdf.crs).lower() == str(target_crs).lower():
        return gdf
    return gdf.to_crs(target_crs)
