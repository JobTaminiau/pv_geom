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
import shapely

from pv_geom.errors import InputError, MissingCRSError, PolygonIdError
from pv_geom.io.vector import read_vector, reproject
from pv_geom.vintage import parse_vintage


def _record_date(value, *, earliest: bool):
    """A per-polygon installation date. A bare year or month is a window: its
    last day for "complete by", its first for "not before", so that neither
    claims more than the record says."""
    s = str(value).strip()
    if earliest and len(s) in (4, 7) and s[:4].isdigit():
        return parse_vintage(f"{s}-01-01" if len(s) == 4 else f"{s}-01")
    return parse_vintage(value)


# Tried in order when the caller does not name the id column.
_ID_ALIASES = ("polygon_id", "detection_id", "cluster_id", "id", "fid", "objectid")


def read_polygons(
    uri: str | Path,
    target_crs: str,
    *,
    id_col: str | None = None,
    explode_multipolygons: bool = True,
    bbox: tuple[float, float, float, float] | None = None,
    max_polygons: int | None = None,
    vintage_col: str | None = None,
    installed_by_col: str | None = None,
    not_installed_before_col: str | None = None,
    min_area_m2: float = 0.0,
    overlap_flag_frac: float = 0.0,
) -> gpd.GeoDataFrame:
    """Read PV polygons (path or ``s3://``) into the run CRS.

    The result always has ``polygon_id`` (unique per row), ``parent_polygon_id``
    (the input feature's id) and ``input_row`` (the input feature's 0-based row
    position). With ``vintage_col`` it also has ``polygon_vintage`` (a
    ``datetime.date`` or None per row).

    No feature is dropped for being unusable: ``input_issue`` says why a feature
    cannot be measured (None when it can), and ``input_flags`` carries quality
    flags for those that can (see :func:`_flag_inputs`).

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
        raise MissingCRSError(
            f"polygon layer {uri} has no CRS, so it cannot be placed on the LiDAR",
            "assign the layer its CRS (e.g. gdf.set_crs(...)) and save it again",
        )
    gdf = gdf.reset_index(drop=True)
    gdf["input_row"] = np.arange(len(gdf), dtype=np.int32)

    # Settle the id column.
    lower = {c.lower(): c for c in gdf.columns}
    source: str | None
    if id_col is not None:
        if id_col not in gdf.columns:
            raise PolygonIdError(
                f"polygon layer has no '{id_col}' column",
                f"choose one of {list(gdf.columns)[:12]}, or omit the option to have "
                f"ids synthesized",
            )
        source = id_col
    else:
        source = next((lower[a] for a in _ID_ALIASES if a in lower), None)
    if source is None:
        ids = pd.Series([f"poly_{i:07d}" for i in gdf["input_row"]], index=gdf.index)
    else:
        ids = gdf[source].astype(str)
        if ids.duplicated().any():
            raise PolygonIdError(
                f"polygon id column '{source}' has duplicate values",
                "pass a unique column with --polygon-id-col, or remove the column to "
                "have ids synthesized",
            )
    gdf["polygon_id"] = ids

    if vintage_col is not None:
        if vintage_col not in gdf.columns:
            raise InputError(
                f"polygon layer has no '{vintage_col}' vintage column",
                f"columns are {list(gdf.columns)[:12]}",
            )
        gdf["polygon_vintage"] = [
            None if pd.isna(v) else parse_vintage(v) for v in gdf[vintage_col]
        ]
    for out_col, src_col in (("installed_by", installed_by_col),
                             ("not_installed_before", not_installed_before_col)):
        if src_col is None:
            continue
        if src_col not in gdf.columns:
            raise InputError(
                f"polygon layer has no '{src_col}' column (vintage.{out_col}_column)",
                f"columns are {list(gdf.columns)[:12]}",
            )
        # Read before any output-named column is written: the source column may
        # itself be called installed_by.
        earliest = out_col == "not_installed_before"
        gdf[out_col] = [None if pd.isna(v) else _record_date(v, earliest=earliest)
                        for v in gdf[src_col]]

    gdf = reproject(gdf, target_crs).assign(parent_polygon_id=gdf["polygon_id"])
    if explode_multipolygons:
        # Explode before judging validity: a MultiPolygon whose parts share an
        # edge is "invalid" as a whole, yet each part is a perfectly good array.
        arr = gdf.geometry.to_numpy()
        has_geom = ~(shapely.is_missing(arr) | shapely.is_empty(arr))
        parts = gdf[has_geom].explode(index_parts=False, ignore_index=True)
        # A dissolved geometry can carry stray line or point fragments beside
        # its polygons. They are debris, not arrays: drop them wherever the
        # feature has a polygonal part to stand for it.
        polygonal = parts.geometry.geom_type.isin(_POLYGONAL)
        keeps_a_polygon = polygonal.groupby(parts["parent_polygon_id"]).transform("any")
        parts = parts[polygonal | ~keeps_a_polygon].reset_index(drop=True)
        # Suffix ids that now repeat (i.e. came from a MultiPolygon parent).
        is_dup = parts["parent_polygon_id"].duplicated(keep=False)
        cumcount = parts.groupby("parent_polygon_id").cumcount()
        parts["polygon_id"] = np.where(
            is_dup,
            parts["parent_polygon_id"] + "__p" + cumcount.astype(str),
            parts["parent_polygon_id"],
        ).astype(str)
        gdf = gpd.GeoDataFrame(
            pd.concat([parts, gdf[~has_geom]], ignore_index=True), geometry="geometry",
            crs=gdf.crs,
        ).sort_values("input_row", kind="stable")

    gdf = _screen_geometries(gdf.reset_index(drop=True))
    if bbox is not None:
        # A feature without usable geometry cannot be placed, so it is out of
        # scope for a bounded run.
        x0, y0, x1, y1 = bbox
        usable = gdf["input_issue"].isna()
        gdf = gdf.loc[gdf[usable].cx[slice(x0, x1), slice(y0, y1)].index]

    gdf = gdf.reset_index(drop=True)
    _flag_inputs(gdf, min_area_m2, overlap_flag_frac)
    if max_polygons is not None:
        gdf = gdf.head(max_polygons)
    return gdf.reset_index(drop=True)


_POLYGONAL = ("Polygon", "MultiPolygon")


def _polygonal_part(geom):
    """The polygonal content of a geometry (a repair can return a collection
    holding stray lines and points), or None when there is none."""
    parts = [g for g in shapely.get_parts(geom) if g.geom_type in _POLYGONAL and not g.is_empty]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else shapely.union_all(parts)


def _screen_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Decide which features can be measured. Nothing is dropped.

    Adds ``input_issue`` — why a feature has no usable polygon (``null_geometry``,
    ``empty_geometry``, ``not_polygonal``, ``zero_area``), or None — and
    ``input_flags``. Invalid polygons (self-intersections and the like) are
    repaired and flagged ``geometry_repaired`` rather than rejected.
    """
    gdf = gdf.copy()
    geoms = gdf.geometry.to_numpy().copy()
    n = len(geoms)
    issue = np.full(n, None, dtype=object)
    flags = np.empty(n, dtype=object)
    flags[:] = [()] * n

    missing = shapely.is_missing(geoms)
    empty = ~missing & shapely.is_empty(geoms)
    # 3 = Polygon, 6 = MultiPolygon, 7 = GeometryCollection. A collection may
    # hold polygons beside stray lines and points (a dissolve can produce one),
    # so it is sifted for its polygonal content rather than rejected outright.
    type_id = shapely.get_type_id(geoms)
    polygonal = np.isin(type_id, (3, 6))
    collection = type_id == 7
    issue[missing] = "null_geometry"
    issue[empty] = "empty_geometry"
    issue[~missing & ~empty & ~polygonal & ~collection] = "not_polygonal"

    # Repairs are rare, so only those features are handled one at a time.
    candidates = ~missing & ~empty & (polygonal | collection)
    for i in np.flatnonzero(candidates & (collection | ~shapely.is_valid(geoms))):
        fixed = _polygonal_part(shapely.make_valid(geoms[i]))
        if fixed is None:
            issue[i] = "not_polygonal"
        else:
            geoms[i] = fixed
            flags[i] = ("geometry_repaired",)

    usable = np.array([v is None for v in issue])
    area = np.zeros(n)
    area[usable] = shapely.area(geoms[usable])
    issue[usable & (area <= 0)] = "zero_area"
    gdf = gdf.set_geometry(gpd.GeoSeries(geoms, index=gdf.index, crs=gdf.crs))
    gdf["input_issue"] = pd.Series(issue, index=gdf.index, dtype=object)
    gdf["input_flags"] = pd.Series(list(flags), index=gdf.index, dtype=object)
    return gdf


def _flag_inputs(gdf: gpd.GeoDataFrame, min_area_m2: float, overlap_flag_frac: float) -> None:
    """Add the input-quality flags (in place): ``below_min_area``,
    ``duplicate_geometry`` and ``overlaps_polygon``. Flagged polygons are still
    measured; the flags travel to their output rows."""
    usable = gdf["input_issue"].isna().to_numpy()
    idx = np.flatnonzero(usable)
    if not len(idx):
        return
    geoms = gdf.geometry.to_numpy()[idx]
    area = shapely.area(geoms)
    extra: dict[int, list[str]] = {}

    for k in np.flatnonzero(area < min_area_m2):
        extra.setdefault(int(idx[k]), []).append("below_min_area")

    wkb = pd.Series(shapely.to_wkb(shapely.normalize(geoms)))
    for k in np.flatnonzero(wkb.duplicated(keep="first").to_numpy()):
        extra.setdefault(int(idx[k]), []).append("duplicate_geometry")

    if overlap_flag_frac > 0 and len(geoms) > 1:
        left, right = shapely.STRtree(geoms).query(geoms, predicate="intersects")
        pairs = left < right
        left, right = left[pairs], right[pairs]
        if len(left):
            shared = shapely.area(shapely.intersection(geoms[left], geoms[right]))
            for side in (left, right):
                hit = shared >= overlap_flag_frac * area[side]
                for k in np.unique(side[hit]):
                    extra.setdefault(int(idx[k]), []).append("overlaps_polygon")

    col = gdf.columns.get_loc("input_flags")
    for pos, names in extra.items():
        gdf.iat[pos, col] = (*gdf.iat[pos, col], *names)
