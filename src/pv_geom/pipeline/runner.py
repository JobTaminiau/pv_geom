"""Run the pipeline: plan, execute, write, record.

The stages live in their own modules — :mod:`plan` (what to measure),
:mod:`executor` (how tasks run), :mod:`worker` (one tile group),
:mod:`sink` (where output goes) — and this module strings them together.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa

from pv_geom.config import PVGeomConfig
from pv_geom.pipeline.executor import execute
from pv_geom.pipeline.partition import TileGroup
from pv_geom.pipeline.plan import (
    PendingGroup,
    Plan,
    RunInputs,
    build_plan,
    drop_missing_tiles,
    find_missing_tiles,
)
from pv_geom.pipeline.sink import OutputSink
from pv_geom.pipeline.vintage_check import probe_lidar_vintage, row_vintage_summary
from pv_geom.provenance import write_manifest
from pv_geom.summary import summarise_table

log = logging.getLogger(__name__)


@dataclass
class _Collector:
    """Receives each finished group: writes its partition, keeps its table for
    the run statistics, and records failures."""

    sink: OutputSink
    crs: str
    tables: list[pa.Table] = field(default_factory=list)
    failed: dict[int, str] = field(default_factory=dict)
    failed_polygons: int = 0
    rows_written: int = 0

    def on_done(self, partition_id: int, table: pa.Table) -> None:
        if len(table) == 0:
            return
        self.tables.append(table)
        target = self.sink.write_part(table, partition_id, self.crs)
        self.rows_written += len(table)
        log.info("wrote %s (%d rows)", target, len(table))

    def on_error(self, partition_id: int, group: TileGroup, exc: BaseException) -> None:
        self.failed[partition_id] = repr(exc)
        self.failed_polygons += len(group.polygon_ids)
        log.error("tile group %s (partition %d, %d polygons) FAILED: %r",
                  group.primary_tile_id, partition_id, len(group.polygon_ids), exc)


def _pending_groups(plan: Plan, sink: OutputSink, resume: bool) -> tuple[list[PendingGroup], set[int]]:
    """Groups still to run, and the partition ids ``--resume`` is keeping."""
    kept = sink.existing_part_ids() & set(range(len(plan.groups))) if resume else set()
    if kept:
        log.info("resume: skipping %d already-written partitions", len(kept))
    return [(pid, g) for pid, g in enumerate(plan.groups) if pid not in kept], kept


def _concat(tables: list[pa.Table]) -> pa.Table | None:
    tables = [t.replace_schema_metadata(None) for t in tables if len(t)]
    return pa.concat_tables(tables, promote_options="default") if tables else None


def run_pipeline(
    *,
    polygons_uri: str,
    tile_index_uri: str,
    lidar_prefix: str,
    output_uri: str,
    cfg: PVGeomConfig,
    footprints_uri: str | None = None,
    name_template: str = "{name}.laz",
    tile_id_col: str | None = None,
    polygon_id_col: str | None = None,
    max_polygons: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    dry_run: bool = False,
    resume: bool = False,
    use_dask: bool = True,
) -> Path | str:
    """End-to-end pipeline. Returns the manifest path (str when output is s3://).

    Two inputs are required — the polygon layer and the LiDAR (tiles plus their
    index) — each with a capture date under ``cfg.vintage``. Building
    footprints are optional: without them ``on_building`` is null and the roof
    reference comes from an open ring around each polygon.

    Output layout::

        <output_uri>/
            part-<partition_id>.parquet
            manifest.json

    Partitions are written as each tile group completes, and a failed group is
    recorded in the manifest (``counts.failed_groups``,
    ``aggregate_stats.task_errors``) rather than aborting the run.

    ``resume`` is for crash recovery on the SAME inputs and config: partitions
    already at the output are kept and their groups skipped. Changing inputs or
    config between runs while using it produces inconsistent output.
    """
    sink = OutputSink(output_uri)
    run_id = uuid.uuid4().hex
    plan = build_plan(
        RunInputs(
            polygons_uri=polygons_uri, tile_index_uri=tile_index_uri,
            lidar_prefix=lidar_prefix, footprints_uri=footprints_uri,
            name_template=name_template, tile_id_col=tile_id_col,
            polygon_id_col=polygon_id_col, max_polygons=max_polygons, bbox=bbox,
        ),
        cfg,
    )
    cfg = plan.cfg

    # Vintage check up front: if the polygons postdate the point cloud, that is
    # worth knowing before a multi-hour run — and it is what --dry-run is for.
    vintage = probe_lidar_vintage(cfg, plan.primary_tile_uris)

    def _manifest(cluster_spec: dict, counts: dict, stats: dict) -> None:
        write_manifest(
            sink.manifest_target,
            config_dict=cfg.model_dump(mode="json"),
            config_hash=plan.config_hash,
            inputs=plan.inputs.as_manifest(),
            crs=plan.crs,
            cluster_spec=cluster_spec,
            counts=counts,
            aggregate_stats=stats,
            tiles_touched=plan.tiles_touched,
            run_id=run_id,
            vintage=vintage,
        )

    base_counts = {"polygons": plan.n_polygons, "tile_groups": len(plan.groups)}
    if dry_run:
        _manifest({"backend": "dry_run"},
                  {**base_counts, "attempted": 0, "succeeded": 0, "failed": 0},
                  {"dry_run": True})
        return sink.manifest_target

    pending, kept = _pending_groups(plan, sink, resume)
    # On Coiled the laptop is not on the data path — workers sit beside the
    # bucket — so tiles are listed rather than downloaded here.
    prewarm = bool(pending) and use_dask and cfg.compute.backend != "coiled"
    pending = drop_missing_tiles(pending, plan, find_missing_tiles(pending, plan, prewarm=prewarm))

    collector = _Collector(sink, plan.crs)
    execute(pending, lambda pid, g: plan.task_kwargs(pid, g, run_id),
            collector.on_done, collector.on_error, cfg, use_dask=use_dask)
    if collector.failed:
        log.error("%d tile-group tasks failed (%d polygons); rerun with --resume to retry",
                  len(collector.failed), collector.failed_polygons)

    # Statistics cover the whole output: this run's partitions and resumed ones.
    resumed = [sink.read_part(pid) for pid in sorted(kept)]
    full = _concat(collector.tables + resumed)
    stats = summarise_table(full) if full is not None else {}
    if collector.failed:
        stats["task_errors"] = collector.failed
    if full is not None:
        vintage.update(row_vintage_summary(full))

    succeeded = collector.rows_written + sum(len(t) for t in resumed)
    _manifest(
        {"backend": cfg.compute.backend, "use_dask": use_dask},
        {**base_counts, "attempted": plan.n_attempted, "succeeded": succeeded,
         "failed": plan.n_attempted - succeeded, "failed_groups": len(collector.failed)},
        stats,
    )
    return sink.manifest_target
