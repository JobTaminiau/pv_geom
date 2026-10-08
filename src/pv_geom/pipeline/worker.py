"""Per-tile-group worker: one group's tiles and polygons in, one table out.

The unit of work the executor dispatches. It loads the group's points
(:mod:`pointpool`), measures each polygon (:mod:`measure`) and converts the
results to rows in the run's schema (:mod:`rows`).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import geopandas as gpd
import numpy as np
import pyarrow as pa
import shapely

from pv_geom.config import PVGeomConfig
from pv_geom.io.lidar import clip_points_to_polygon
from pv_geom.pipeline.measure import LocalPoints, Measurement, PolygonTask, measure_polygon
from pv_geom.pipeline.pointpool import (
    GroupOverBudget,
    budget_points,
    lidar_date_for,
    load_group_points,
)
from pv_geom.pipeline.rows import Provenance, rows_to_table, to_row
from pv_geom.schema import output_schema
from pv_geom.utils.north import meridian_convergence_deg

log = logging.getLogger(__name__)


def _row(m: Measurement, pts: LocalPoints, cfg: PVGeomConfig, prov: Provenance) -> dict[str, Any]:
    """A measurement as an output row, with the experimental mounting columns
    when that classifier is switched on."""
    extra: dict[str, Any] = {}
    if cfg.mounting_rules.enabled:
        from pv_geom.experimental.mounting.features import mounting_columns

        extra = mounting_columns(m, pts, cfg)       # may append to m.flags
    return {**to_row(m, prov), **extra}


def build_row(
    polygon: Any,
    polygon_id: str,
    cfg: PVGeomConfig,
    config_hash: str,
    run_id: str,
    partition_id: int,
    *,
    panel_pts: np.ndarray,
    ground_xyz: np.ndarray,
    roof_input_pts: np.ndarray,
    footprints: gpd.GeoDataFrame | None,
    other_pv_polygons: gpd.GeoDataFrame,
    contributing_tile_ids: tuple[str, ...],
    parent_polygon_id: str | None = None,
    input_row: int = 0,
    polygon_vintage: date | None = None,
    lidar_date: date | None = None,
    lidar_date_source: str | None = None,
    grid_convergence_deg: float = 0.0,
) -> dict[str, Any]:
    """Measure one polygon from point arrays and return its output row.

    A convenience over :func:`measure_polygon` + :func:`to_row` for callers
    that already hold the points: ``panel_pts`` are the candidates inside the
    polygon, ``roof_input_pts`` those around it, ``ground_xyz`` the ground
    returns nearby.
    """
    task = PolygonTask(polygon, str(polygon_id), parent_polygon_id, input_row, polygon_vintage,
                       grid_convergence_deg=grid_convergence_deg)
    pts = LocalPoints(panel_pts, ground_xyz, roof_input_pts, footprints, other_pv_polygons)
    m = measure_polygon(task, pts, cfg, lidar_date=lidar_date,
                        lidar_date_source=lidar_date_source)
    return _row(m, pts, cfg,
                Provenance(config_hash, run_id, partition_id, tuple(contributing_tile_ids)))


def _is_missing(v: Any) -> bool:
    """None, NaN or NaT (the only values unequal to themselves)."""
    if v is None:
        return True
    try:
        return bool(v != v)
    except Exception:
        return False


def process_tile_group(
    tile_uri_map: dict[str, str],
    primary_tile_id: str,
    polygons: gpd.GeoDataFrame,
    fetch_tile_ids: tuple[str, ...],
    footprints: gpd.GeoDataFrame | None,
    cfg: PVGeomConfig,
    *,
    config_hash: str,
    run_id: str,
    partition_id: int,
    polygon_id_col: str = "polygon_id",
    polygon_vintage: date | None = None,
    lidar_date: date | None = None,
) -> pa.Table:
    """Fetch the group's tiles, measure every polygon, return a table.

    ``polygon_vintage`` is the run-level imagery date (a per-row
    ``polygon_vintage`` column on ``polygons`` overrides it). ``lidar_date`` is
    a *declared* capture date; when None each row gets the flight date measured
    from its primary tile's GPS time.
    """
    schema = output_schema(cfg.mounting_rules.enabled)
    prov = Provenance(config_hash, run_id, partition_id,
                      tuple(t for t in fetch_tile_ids if tile_uri_map.get(t)))
    geoms = polygons.geometry.to_numpy()
    neighbours = shapely.STRtree(geoms)
    # Fits are made in grid coordinates; azimuths are reported from true north.
    centroids = shapely.centroid(geoms)
    cx, cy = shapely.get_x(centroids), shapely.get_y(centroids)
    convergence = meridian_convergence_deg(polygons.crs, cx, cy)
    has_parent = "parent_polygon_id" in polygons.columns
    has_row = "input_row" in polygons.columns
    has_vintage = "polygon_vintage" in polygons.columns
    has_flags = "input_flags" in polygons.columns
    records = list(polygons.to_dict("records"))

    # The whole group at once when its returns fit the memory budget; otherwise
    # in spatial batches, each with its own (smaller) point pools. A polygon's
    # neighbours come from the whole group either way, so the rows are the same.
    limit = budget_points(cfg)
    measured: dict[int, dict[str, Any]] = {}
    batches = [np.arange(len(geoms))]
    while batches:
        batch = batches.pop()
        try:
            pool = load_group_points(tile_uri_map, primary_tile_id, fetch_tile_ids,
                                     geoms[batch], cfg,
                                     max_points=limit if len(batch) > 1 else None)
        except GroupOverBudget as exc:
            along = cx[batch] if np.ptp(cx[batch]) >= np.ptp(cy[batch]) else cy[batch]
            order = batch[np.argsort(along, kind="stable")]
            half = len(order) // 2
            log.info("partition %d: %s; measuring %d polygons as two batches",
                     partition_id, exc, len(batch))
            batches += [order[half:], order[:half]]
            continue
        if pool is None:
            log.warning("primary tile %s unavailable; partition %d has no rows",
                        primary_tile_id, partition_id)
            return rows_to_table([], schema)
        row_lidar_date, row_lidar_source = lidar_date_for(pool.primary_vintage, lidar_date)

        for i in batch:
            rec = records[i]
            poly = geoms[i]
            vintage = polygon_vintage
            if has_vintage and not _is_missing(rec["polygon_vintage"]):
                vintage = rec["polygon_vintage"]
            task = PolygonTask(
                polygon=poly, polygon_id=str(rec[polygon_id_col]),
                parent_polygon_id=str(rec["parent_polygon_id"]) if has_parent else None,
                input_row=int(rec["input_row"]) if has_row else int(i),
                polygon_vintage=vintage,
                input_flags=tuple(rec["input_flags"]) if has_flags else (),
                grid_convergence_deg=float(convergence[i]),
            )
            # Other PV polygons close enough to intrude on this one's roof ring.
            near = neighbours.query(poly.buffer(pool.roof_pad_m))
            pts = LocalPoints(
                panel=clip_points_to_polygon(pool.panel.query_polygon_bounds(poly), poly,
                                             erosion_m=cfg.panel_plane.erosion_m),
                ground=pool.ground.query_polygon_bounds(poly, pad_m=pool.ground_pad_m),
                surroundings=pool.panel.query_polygon_bounds(poly, pad_m=pool.roof_pad_m),
                footprints=footprints,
                other_polygons=gpd.GeoDataFrame(
                    geometry=[geoms[j] for j in near if j != i], crs=polygons.crs),
            )
            m = measure_polygon(task, pts, cfg, lidar_date=row_lidar_date,
                                lidar_date_source=row_lidar_source)
            measured[int(i)] = _row(m, pts, cfg, prov)
        del pool

    rows = [measured[i] for i in range(len(geoms))]
    return rows_to_table(rows, schema)
