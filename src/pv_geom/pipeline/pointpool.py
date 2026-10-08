"""Load one tile group's LiDAR into the two point pools the measurement needs.

A tile group can hold tens of millions of returns, of which a sparse inventory
uses a sliver. This module reads the group's tiles one at a time, keeps only
ground and panel-candidate returns near a polygon, and hands back two spatial
indexes — so peak memory tracks what is used rather than what the tiles hold.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from pv_geom.config import PVGeomConfig
from pv_geom.geometry.point_index import GroundModel, PointGrid
from pv_geom.io.lidar import TileStream, TileVintage, read_tile
from pv_geom.utils.crs import to_run_crs

log = logging.getLogger(__name__)


class NearMask:
    """Which (x, y) lie within ``pad_m`` of any polygon's bounding box, answered
    from a coarse raster so it costs one lookup per point."""

    def __init__(self, geometries: Any, pad_m: float, cell_m: float = 10.0) -> None:
        bounds = np.asarray([g.bounds for g in geometries], dtype=float).reshape(-1, 4)
        self.cell = float(cell_m)
        self.empty = len(bounds) == 0
        if self.empty:
            return
        lo = bounds[:, :2] - pad_m
        hi = bounds[:, 2:] + pad_m
        # On absolute multiples of the cell, so which returns are kept near a
        # polygon does not depend on the other polygons in the batch.
        self.x0, self.y0 = np.floor(lo.min(axis=0) / self.cell) * self.cell
        nx = int(np.floor((hi[:, 0].max() - self.x0) / self.cell)) + 1
        ny = int(np.floor((hi[:, 1].max() - self.y0) / self.cell)) + 1
        self.mask = np.zeros((nx, ny), dtype=bool)
        i0 = np.floor((lo[:, 0] - self.x0) / self.cell).astype(int)
        i1 = np.floor((hi[:, 0] - self.x0) / self.cell).astype(int)
        j0 = np.floor((lo[:, 1] - self.y0) / self.cell).astype(int)
        j1 = np.floor((hi[:, 1] - self.y0) / self.cell).astype(int)
        for a, b, c, d in zip(i0, i1, j0, j1, strict=True):
            self.mask[a:b + 1, c:d + 1] = True

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.empty:
            return np.zeros(len(x), dtype=bool)
        ix = np.floor((x - self.x0) / self.cell).astype(np.int64)
        iy = np.floor((y - self.y0) / self.cell).astype(np.int64)
        nx, ny = self.mask.shape
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        out = np.zeros(len(x), dtype=bool)
        out[inside] = self.mask[ix[inside], iy[inside]]
        return out


def above_ground(candidates: np.ndarray, ground_xyz: np.ndarray, cfg: PVGeomConfig) -> np.ndarray:
    """Keep candidate returns above the local-ground cutoff (all of them when
    there is no ground reference at all)."""
    if len(candidates) == 0 or len(ground_xyz) == 0:
        return candidates
    model = GroundModel(ground_xyz, cell_m=cfg.io.classification.ground_grid_cell_m)
    gz = model.ground_z(candidates[:, 0], candidates[:, 1])
    return candidates[candidates[:, 2] > gz + cfg.io.classification.fallback_height_above_ground_m]


def split_classes(pts: np.ndarray, cfg: PVGeomConfig) -> tuple[np.ndarray, np.ndarray, int]:
    """Split ``(N, 4)`` ``[x, y, z, class]`` returns into ground and panel candidates.

    Returns ``(ground_xyz, panel_xyz, class_used)``. When the primary panel
    class (ASPRS 6, building) is present it wins. Most public LiDAR does not
    carry it — neither the Phoenix USGS tiles nor the Delaware state collection
    do — so the usual path is the fallback: unclassified returns more than
    ``fallback_height_above_ground_m`` above *local* ground, read from a grid of
    ground elevations so the cut holds on sloping terrain.
    """
    classes = cfg.io.classification
    if pts.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), classes.panel_class_primary
    cls = pts[:, 3].astype(np.int16)
    ground_xyz = pts[cls == classes.ground_class][:, :3]
    if (cls == classes.panel_class_primary).any():
        return ground_xyz, pts[cls == classes.panel_class_primary][:, :3], classes.panel_class_primary
    fallback = pts[cls == classes.panel_class_fallback][:, :3]
    return ground_xyz, above_ground(fallback, ground_xyz, cfg), classes.panel_class_fallback


def lidar_date_for(vintage: TileVintage | None,
                   declared: date | None) -> tuple[date | None, str | None]:
    """The capture date to stamp on a tile's rows, and where it came from."""
    if declared is not None:
        return declared, "declared"
    if vintage is None:
        return None, None
    if vintage.flight_end is not None:
        return vintage.flight_end, "gps_time"
    if vintage.creation_date is not None:
        return vintage.creation_date, "header_date"
    return None, None


@dataclass
class GroupPoints:
    """Indexed point pools for one tile group."""

    panel: PointGrid                    # panel-candidate returns
    ground: PointGrid                   # ground returns
    primary_vintage: TileVintage | None
    roof_pad_m: float                   # how far around a polygon the roof ring can reach
    ground_pad_m: float                 # how far the ground search can reach


# What a kept return costs while a group is measured: its coordinates, the
# copy made when the chunks are joined, and the spatial index over them.
BYTES_PER_KEPT_POINT = 120


class GroupOverBudget(Exception):
    """A tile group's kept returns would exceed the memory budget."""


def budget_points(cfg: PVGeomConfig) -> int | None:
    """How many returns one task may keep, from ``compute.memory_budget_gb``
    (less what a chunk needs while it is being filtered); None = unbounded."""
    gb = cfg.compute.memory_budget_gb
    if gb is None:
        return None
    decode = 150 * int(cfg.compute.lidar_chunk_points)
    return max(int((gb * 1e9 - decode) / BYTES_PER_KEPT_POINT), 100_000)


def _tile_chunks(uri: str, cfg: PVGeomConfig) -> tuple[Iterator[np.ndarray], Any]:
    """A tile as ``(chunks, source)``; ``source`` has ``.crs`` and, once the
    chunks are consumed, ``.vintage``. PDAL reads whole tiles."""
    if cfg.io.lidar_reader == "pdal":
        data = read_tile(uri, reader="pdal")
        return iter([data.points]), data
    stream = TileStream(uri, chunk_points=cfg.compute.lidar_chunk_points)
    return iter(stream), stream


def _cat(chunks: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3))


def load_group_points(
    tile_uri_map: dict[str, str],
    primary_tile_id: str,
    fetch_tile_ids: tuple[str, ...],
    geometries: Any,
    cfg: PVGeomConfig,
    *,
    max_points: int | None = None,
) -> GroupPoints | None:
    """Read a tile group's tiles into indexed ground and panel-candidate pools.

    Tiles are decoded a chunk at a time and only returns near a polygon are
    kept. Raises ``GroupOverBudget`` as soon as more than ``max_points`` have
    been kept, so the caller can measure fewer polygons at once.

    Returns ``None`` when the *primary* tile is unavailable: a group's polygons
    live on its primary tile by construction, so there is nothing to measure.
    Other missing tiles are tolerated, so the runner need not pre-screen the
    whole bucket.
    """
    classes = cfg.io.classification
    roof_pad = cfg.roof_plane.buffer_max_m + 1.0
    ground_pad = max(cfg.heights.ground_search_radius_m,
                     cfg.heights.ground_fallback_max_radius_m)
    # Nothing farther from a polygon than the ground-search radius is ever
    # used; dropping it as each tile is read is what keeps a sparse inventory
    # on big, dense tiles (Delaware: 44M returns per tile) inside worker memory.
    near_ground = NearMask(geometries, pad_m=ground_pad)
    near_panel = NearMask(geometries, pad_m=roof_pad)

    primary_vintage: TileVintage | None = None
    primary_loaded = False
    ground_chunks: list[np.ndarray] = []
    primary_chunks: list[np.ndarray] = []
    fallback_chunks: list[np.ndarray] = []
    kept = 0
    for tid in fetch_tile_ids:
        uri = tile_uri_map.get(tid)
        if uri is None:
            continue
        try:
            blocks, source = _tile_chunks(uri, cfg)
        except FileNotFoundError:           # includes RemoteFileMissing
            log.warning("%s missing; skipping", uri)
            continue
        told = False
        for block in blocks:
            pts, converted = to_run_crs(block, source.crs, str(cfg.crs.target))
            if converted and not told:
                log.info("%s: %s", tid, converted)
                told = True
            pts = pts[near_ground(pts[:, 0], pts[:, 1])]
            cls = pts[:, 3].astype(np.int16)
            ground_chunks.append(pts[cls == classes.ground_class][:, :3])
            kept += len(ground_chunks[-1])
            for chunks, klass in ((primary_chunks, classes.panel_class_primary),
                                  (fallback_chunks, classes.panel_class_fallback)):
                cand = pts[cls == klass][:, :3]
                chunks.append(cand[near_panel(cand[:, 0], cand[:, 1])])
                kept += len(chunks[-1])
            del pts, cls, block
            if max_points is not None and kept > max_points:
                close = getattr(blocks, "close", None)
                if close is not None:
                    close()                      # stop decoding; releases the file
                raise GroupOverBudget(f"{kept:,} returns kept, budget {max_points:,}")
        if tid == primary_tile_id:
            primary_loaded = True
            primary_vintage = source.vintage

    if not primary_loaded:
        return None

    # Choose the panel-candidate pool ONCE for the whole group: the primary
    # class when the tiles carry it, else the fallback class above local ground.
    ground_xyz = _cat(ground_chunks)
    panel_xyz = _cat(primary_chunks)
    if len(panel_xyz) == 0:
        panel_xyz = above_ground(_cat(fallback_chunks), ground_xyz, cfg)
    return GroupPoints(
        panel=PointGrid(panel_xyz), ground=PointGrid(ground_xyz),
        primary_vintage=primary_vintage, roof_pad_m=roof_pad, ground_pad_m=ground_pad,
    )
