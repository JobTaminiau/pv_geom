"""PV polygon reader + reprojector.

Reads a polygon layer in any vector format GeoPandas can open (GeoParquet,
GeoPackage, GeoJSON, Shapefile, FlatGeobuf), settles a row id, reprojects to
the run CRS, and (by default) explodes MultiPolygons into one row per part with
a ``parent_polygon_id`` link. MultiPolygon parts are treated as independent
arrays since they typically sit on distinct facets.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from pv_geom.io.vector import read_vector, reproject
from pv_geom.vintage import parse_vintage

# Tried in order when the caller does not name the id column.
_ID_ALIASES = ("polygon_id", "detection_id", "id", "fid", "objectid")


def read_polygons(
    uri: str | Path,
    target_crs: str,
    *,
    id_col: str | None = None,
    explode_multipolygons: bool = True,
    bbox: tuple[float, float, float, float] | None = None,
    max_polygons: int | None = None,
    vintage_col: str | None = None,
) -> gpd.GeoDataFrame:
    """Read PV polygons (path or ``s3://``) into the run CRS.

    The result always has ``polygon_id`` (unique per row), ``parent_polygon_id``
    (the input feature's id) and ``input_row`` (the input feature's 0-based row
    position). With ``vintage_col`` it also has ``polygon_vintage`` (a
    ``datetime.date`` or None per row).

    Parameters
    ----------
    id_col
        Column holding the input ids. ``None`` tries ``polygon_id``,
        ``detection_id``, ``id``, ``fid``, ``objectid`` (case-insensitive) and,
        failing those, synthesizes ``poly_<input_row>`` ids — detector outputs
        often have none, and ``input_row`` still joins the output back.
    explode_multipolygons
        Explode each MultiPolygon into one row per part; parts of a multi-part
        parent get ``__p<i>`` suffixes on ``polygon_id``.
    bbox
        Optional ``(xmin, ymin, xmax, ymax)`` filter in ``target_crs`` units,
        applied after reprojection.
    max_polygons
        Optional row limit (for dev/smoke runs).
    vintage_col
        Optional per-polygon capture-date column (dates, timestamps, or
        ``YYYY`` / ``YYYY-MM`` / ``YYYY-MM-DD`` strings).
    """
    gdf = read_vector(uri)
    if gdf.crs is None:
        raise ValueError(f"polygon layer {uri} has no CRS; cannot place it on the LiDAR")
    gdf = gdf.reset_index(drop=True)
    gdf["input_row"] = np.arange(len(gdf), dtype=np.int32)

    # Settle the id column.
    lower = {c.lower(): c for c in gdf.columns}
    source: str | None
    if id_col is not None:
        if id_col not in gdf.columns:
            raise ValueError(f"polygon layer has no '{id_col}' column")
        source = id_col
    else:
        source = next((lower[a] for a in _ID_ALIASES if a in lower), None)
    if source is None:
        ids = pd.Series([f"poly_{i:07d}" for i in gdf["input_row"]], index=gdf.index)
    else:
        ids = gdf[source].astype(str)
        if ids.duplicated().any():
            raise ValueError(
                f"polygon id column '{source}' has duplicate values; pass a unique "
                f"column via id_col, or drop it to have ids synthesized"
            )
    gdf["polygon_id"] = ids

    if vintage_col is not None:
        if vintage_col not in gdf.columns:
            raise ValueError(f"polygon layer has no '{vintage_col}' vintage column")
        gdf["polygon_vintage"] = [
            None if pd.isna(v) else parse_vintage(v) for v in gdf[vintage_col]
        ]

    gdf = reproject(gdf, target_crs)

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        gdf = gdf.cx[slice(x0, x1), slice(y0, y1)]

    gdf = gdf.assign(parent_polygon_id=gdf["polygon_id"])
    if explode_multipolygons:
        gdf = gdf.explode(index_parts=False, ignore_index=True)
        # Suffix ids that now repeat (i.e. came from a MultiPolygon parent).
        is_dup = gdf["parent_polygon_id"].duplicated(keep=False)
        cumcount = gdf.groupby("parent_polygon_id").cumcount()
        new_ids = np.where(
            is_dup,
            gdf["parent_polygon_id"] + "__p" + cumcount.astype(str),
            gdf["parent_polygon_id"],
        )
        gdf["polygon_id"] = new_ids.astype(str)

    if max_polygons is not None:
        gdf = gdf.head(max_polygons)
    return gdf.reset_index(drop=True)
