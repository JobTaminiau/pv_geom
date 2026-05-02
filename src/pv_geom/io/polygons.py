"""PV polygon reader + reprojector. M2.

Reads a GeoParquet of PV polygons, normalizes the polygon-id column,
reprojects to the LiDAR target CRS, and (by default) explodes any
MultiPolygons into one row per part with a ``parent_polygon_id`` link
column. PRD §3.1 accepts both Polygon and MultiPolygon input; we treat
MultiPolygon parts as independent panel arrays since they typically
correspond to distinct facets / panel orientations.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np


def read_polygons(
    uri: str | Path,
    target_crs: str,
    *,
    id_col: str = "polygon_id",
    explode_multipolygons: bool = True,
    bbox: tuple[float, float, float, float] | None = None,
    max_polygons: int | None = None,
) -> gpd.GeoDataFrame:
    """Read PV polygons from GeoParquet (path or s3://).

    Parameters
    ----------
    uri
        Local path or ``s3://`` URI to a GeoParquet file.
    target_crs
        EPSG-string CRS to reproject to (e.g. ``"EPSG:6341"``).
    id_col
        Canonical id column name. If absent in the input, ``detection_id``
        is accepted as an alias and copied to ``id_col``.
    explode_multipolygons
        If True (default), explode each MultiPolygon into one row per part
        and append ``parent_polygon_id`` (the original id). Per-part ids get
        ``__p<i>`` suffixes when more than one part shares a parent.
    bbox
        Optional ``(xmin, ymin, xmax, ymax)`` filter, in ``target_crs`` units,
        applied after reprojection.
    max_polygons
        Optional row limit (for dev/smoke runs).
    """
    gdf = gpd.read_parquet(str(uri))

    # Normalize id column.
    if id_col not in gdf.columns:
        if "detection_id" in gdf.columns:
            gdf = gdf.assign(**{id_col: gdf["detection_id"].astype(str)})
        else:
            raise ValueError(
                f"polygon parquet has neither '{id_col}' nor 'detection_id' column"
            )
    else:
        gdf[id_col] = gdf[id_col].astype(str)

    # Reproject (cheap no-op if already in target).
    if gdf.crs is not None and str(gdf.crs).lower() != str(target_crs).lower():
        gdf = gdf.to_crs(target_crs)

    if bbox is not None:
        x0, y0, x1, y1 = bbox
        gdf = gdf.cx[x0:x1, y0:y1]

    if explode_multipolygons:
        gdf = gdf.assign(parent_polygon_id=gdf[id_col])
        gdf = gdf.explode(index_parts=False, ignore_index=True)
        # Suffix ids that now repeat (i.e. came from a MultiPolygon parent).
        is_dup = gdf["parent_polygon_id"].duplicated(keep=False)
        cumcount = gdf.groupby("parent_polygon_id").cumcount()
        new_ids = np.where(
            is_dup,
            gdf["parent_polygon_id"] + "__p" + cumcount.astype(str),
            gdf["parent_polygon_id"],
        )
        gdf[id_col] = new_ids.astype(str)
    else:
        gdf = gdf.assign(parent_polygon_id=gdf[id_col])

    if max_polygons is not None:
        gdf = gdf.head(max_polygons).reset_index(drop=True)
    else:
        gdf = gdf.reset_index(drop=True)

    return gdf
