"""From inputs to a work plan: what will be measured, from which tiles.

Planning reads the polygon layer, the tile index and (optionally) footprints,
settles the run CRS and dates, and groups polygons by the tile that holds them.
It touches no point data, so it is cheap, deterministic and testable alone —
and it is all ``--dry-run`` needs.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import geopandas as gpd

from pv_geom.config import PVGeomConfig
from pv_geom.io.footprints import read_footprints
from pv_geom.io.polygons import read_polygons
from pv_geom.io.storage import RemoteFileMissing, is_remote, list_s3_uris, localize
from pv_geom.io.tile_index import build_tile_uris, load_tile_index, resolve_tile_id_col
from pv_geom.io.tile_scan import scan_tile_index
from pv_geom.pipeline.partition import TileGroup, assign_polygons_to_tiles, build_tile_groups
from pv_geom.utils.crs import resolve_target_crs
from pv_geom.vintage import parse_vintage

log = logging.getLogger(__name__)

# Columns of the polygon layer the workers need. Detector outputs drag along
# wide attribute columns that would otherwise be pickled into every task.
_WORKER_COLUMNS = ("polygon_id", "parent_polygon_id", "input_row", "polygon_vintage",
                   "input_flags", "input_issue")

PendingGroup = tuple[int, TileGroup]          # (partition_id, group)


@dataclass(frozen=True)
class RunInputs:
    """Where a run's inputs are and how to read them."""

    polygons_uri: str
    tile_index_uri: str | None               # None = build the index from tile headers
    lidar_prefix: str
    footprints_uri: str | None = None
    name_template: str = "{name}.laz"
    tile_id_col: str | None = None
    polygon_id_col: str | None = None
    max_polygons: int | None = None
    bbox: tuple[float, float, float, float] | None = None

    def as_manifest(self) -> dict[str, str | None]:
        return {
            "polygons": str(self.polygons_uri),
            "tile_index": str(self.tile_index_uri) if self.tile_index_uri else None,
            "lidar_prefix": str(self.lidar_prefix),
            "footprints": str(self.footprints_uri) if self.footprints_uri else None,
        }


@dataclass
class Plan:
    """Everything decided before any LiDAR is read."""

    inputs: RunInputs
    cfg: PVGeomConfig                         # with crs.target resolved
    config_hash: str
    crs: str
    polygons: gpd.GeoDataFrame                # measurable polygons, indexed by polygon_id
    invalid: gpd.GeoDataFrame                 # features with no usable polygon geometry
    outside: gpd.GeoDataFrame                 # polygons outside every tile of the index
    footprints: gpd.GeoDataFrame | None
    tile_uri_map: dict[str, str]
    groups: list[TileGroup]
    polygon_vintage: date | None
    lidar_date: date | None                   # declared; None = measure per tile

    @property
    def n_polygons(self) -> int:
        """Every input polygon part in scope, measurable or not."""
        return len(self.polygons) + len(self.invalid) + len(self.outside)

    @property
    def fingerprint(self) -> str:
        """Identifies *what* is being measured and how it is partitioned: the
        polygon ids of every tile group, in order. Two plans with the same
        fingerprint assign the same polygons to the same partition ids, which
        is what ``--resume`` needs to be true."""
        h = hashlib.sha256()
        for pid, g in enumerate(self.groups):
            h.update(f"{pid}|{g.primary_tile_id}|".encode())
            h.update(",".join(g.polygon_ids).encode())
            h.update(b"\n")
        return h.hexdigest()

    @property
    def n_attempted(self) -> int:
        return sum(len(g.polygon_ids) for g in self.groups)

    @property
    def tiles_touched(self) -> list[str]:
        return sorted({t for g in self.groups for t in g.fetch_tile_ids})

    @property
    def primary_tile_uris(self) -> list[str]:
        return [uri for tid in {g.primary_tile_id for g in self.groups}
                if (uri := self.tile_uri_map.get(tid))]

    def task_kwargs(self, partition_id: int, group: TileGroup, run_id: str) -> dict[str, Any]:
        """Arguments for ``process_tile_group`` for one group, carrying only
        that group's polygons, nearby footprints and tile URIs. (Embedding the
        full tile index in every task inflates the graph to GBs at atlas scale.)"""
        sub = self.polygons.loc[list(group.polygon_ids)].copy()
        footprints = None
        if self.footprints is not None:
            x0, y0, x1, y1 = sub.total_bounds
            margin = max(50.0, self.cfg.roof_plane.buffer_max_m * 4.0)
            footprints = self.footprints.cx[x0 - margin: x1 + margin, y0 - margin: y1 + margin]
        return dict(
            tile_uri_map={t: self.tile_uri_map[t] for t in group.fetch_tile_ids
                          if t in self.tile_uri_map},
            primary_tile_id=group.primary_tile_id,
            polygons=sub,
            fetch_tile_ids=group.fetch_tile_ids,
            footprints=footprints,
            cfg=self.cfg,
            config_hash=self.config_hash,
            run_id=run_id,
            partition_id=partition_id,
            polygon_id_col="polygon_id",
            polygon_vintage=self.polygon_vintage,
            lidar_date=self.lidar_date,
        )


def build_plan(inputs: RunInputs, cfg: PVGeomConfig) -> Plan:
    """Read the inputs and decide the work."""
    # The tile index is read first: with ``crs.target: auto`` it is what says
    # which CRS the LiDAR — and therefore the whole run — lives in.
    if inputs.tile_index_uri:
        tindex = load_tile_index(inputs.tile_index_uri)
    else:
        tindex = scan_tile_index(inputs.lidar_prefix)
    cfg = cfg.model_copy(deep=True)
    cfg.crs.target = resolve_target_crs(cfg.crs.target, tindex.crs)
    crs = cfg.crs.target
    tindex = tindex.to_crs(crs) if tindex.crs is not None else tindex.set_crs(crs)
    tile_id_col = resolve_tile_id_col(tindex, inputs.tile_id_col)

    polygons = read_polygons(
        inputs.polygons_uri, target_crs=crs, id_col=inputs.polygon_id_col,
        bbox=inputs.bbox, max_polygons=inputs.max_polygons,
        vintage_col=cfg.vintage.polygon_vintage_column,
        min_area_m2=cfg.polygons.min_area_m2,
        overlap_flag_frac=cfg.polygons.overlap_flag_frac,
    )
    polygons = polygons[[c for c in _WORKER_COLUMNS if c in polygons.columns] + ["geometry"]]
    usable = polygons["input_issue"].isna()
    invalid = polygons[~usable]
    polygons = polygons[usable].drop(columns="input_issue")
    if len(invalid):
        log.warning("%d input features have no usable polygon geometry (%s); they are "
                    "kept as rows with status invalid_geometry", len(invalid),
                    ", ".join(f"{k}: {v}" for k, v in
                              invalid["input_issue"].value_counts().items()))

    footprints = None
    if inputs.footprints_uri:
        footprints = read_footprints(inputs.footprints_uri, target_crs=crs)
    else:
        log.info("no footprint layer: on_building will be null and roof "
                 "references come from open rings")

    if "tile_path" not in tindex.columns:
        tindex = build_tile_uris(tindex, base_uri=inputs.lidar_prefix,
                                 name_col=tile_id_col, name_template=inputs.name_template)
    tile_uri_map = dict(zip(tindex[tile_id_col].astype(str), tindex["tile_path"], strict=True))

    assignments = assign_polygons_to_tiles(
        polygons, tindex, polygon_id_col="polygon_id", tile_id_col=tile_id_col)
    groups = build_tile_groups(assignments)
    outside = polygons[~polygons["polygon_id"].isin(assignments["polygon_id"])]
    polygons = polygons[polygons["polygon_id"].isin(assignments["polygon_id"])]
    log.info("%d polygons -> %d tile groups", len(polygons), len(groups))
    if len(outside):
        log.warning("%d polygons lie outside the LiDAR tile index; they are kept as rows "
                    "with status outside_tile_index", len(outside))
    if len(outside) and not groups:
        log.warning("no polygon falls on any tile: check that the tile index and the "
                    "polygon layer cover the same area (and that --bbox is in the run CRS)")

    return Plan(
        inputs=inputs, cfg=cfg, config_hash=cfg.hash(), crs=crs,
        polygons=polygons.set_index("polygon_id", drop=False),
        invalid=invalid, outside=outside,
        footprints=footprints, tile_uri_map=tile_uri_map, groups=groups,
        polygon_vintage=parse_vintage(cfg.vintage.polygon_vintage),
        lidar_date=parse_vintage(cfg.vintage.lidar_date),
    )


def find_missing_tiles(pending: list[PendingGroup], plan: Plan, *, prewarm: bool) -> set[str]:
    """URIs of indexed tiles that are not in storage.

    With ``prewarm`` the remote tiles are downloaded to the local cache, one at
    a time, in this process: parallel workers on Windows stomp on the AWS SSO
    token cache, and the downloads double as the existence check. Otherwise
    (cloud runs, where workers sit beside the bucket) one paginated listing of
    the prefix finds the gaps without a 404 round-trip per tile per worker.
    """
    uri_map = plan.tile_uri_map
    missing: set[str] = set()
    if not pending:
        return missing
    # Local tiles: a file that is not there is simply not there.
    for _, g in pending:
        for t in g.fetch_tile_ids:
            local_uri = uri_map.get(t)
            if local_uri and not is_remote(local_uri) and not Path(local_uri).exists():
                missing.add(local_uri)

    if prewarm:
        uris = sorted({uri_map[t] for _, g in pending for t in g.fetch_tile_ids
                       if uri_map.get(t) and is_remote(uri_map[t])})
        if uris:
            log.info("pre-warming %d LAZ tiles into the local cache", len(uris))
        for i, uri in enumerate(uris, start=1):
            try:
                local = localize(uri)
                size_mb = local.stat().st_size / 1e6 if local.exists() else 0
                log.info("  [%d/%d] %s -> %.0f MB local", i, len(uris), Path(uri).name, size_mb)
            except RemoteFileMissing:
                log.info("  [%d/%d] %s -> MISSING (will be skipped)", i, len(uris), Path(uri).name)
                missing.add(uri)
        return missing

    prefix = str(plan.inputs.lidar_prefix)
    if not prefix.startswith("s3://"):
        return missing
    try:
        available = list_s3_uris(prefix)
    except Exception as exc:
        # No client-side credentials or network: workers still 404 gracefully.
        log.warning("listing %s failed (%r); missing tiles will be handled by workers",
                    prefix, exc)
        return missing
    root = prefix.rstrip("/") + "/"
    for _, g in pending:
        for t in g.fetch_tile_ids:
            tile_uri = uri_map.get(t)
            if tile_uri and tile_uri.startswith(root) and tile_uri not in available:
                missing.add(tile_uri)
    if missing:
        log.info("%d indexed tiles have no LAZ under %s; dropping them up front",
                 len(missing), prefix)
    return missing


def drop_missing_tiles(pending: list[PendingGroup], plan: Plan,
                       missing: set[str]) -> list[PendingGroup]:
    """Remove missing tiles from each group's fetch list; drop groups whose
    primary tile is missing (there are no points to fit)."""
    if not missing:
        return pending
    uri_map = plan.tile_uri_map
    kept: list[PendingGroup] = []
    n_dropped = 0
    for partition_id, g in pending:
        if uri_map.get(g.primary_tile_id) in missing:
            n_dropped += len(g.polygon_ids)
            continue
        kept.append((partition_id, TileGroup(
            primary_tile_id=g.primary_tile_id, polygon_ids=g.polygon_ids,
            fetch_tile_ids=tuple(t for t in g.fetch_tile_ids if uri_map.get(t) not in missing),
        )))
    if n_dropped:
        log.info("dropped %d polygons whose primary tile is missing", n_dropped)
    return kept
