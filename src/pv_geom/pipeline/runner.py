"""Run the pipeline: plan, execute, write, record.

The stages live in their own modules — :mod:`plan` (what to measure),
:mod:`executor` (how tasks run), :mod:`worker` (one tile group),
:mod:`sink` (where output goes) — and this module strings them together.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import pyarrow as pa

from pv_geom.config import PVGeomConfig
from pv_geom.errors import ResumeMismatchError
from pv_geom.io.output import META_PLAN_FINGERPRINT, META_RESULT_HASH, META_SCHEMA_VERSION
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
from pv_geom.pipeline.rows import Provenance, rows_to_table, unmeasured_row
from pv_geom.pipeline.sink import OutputSink
from pv_geom.pipeline.vintage_check import probe_lidar_vintage, row_vintage_summary
from pv_geom.provenance import write_manifest
from pv_geom.schema import (
    INVALID_GEOMETRY,
    NO_LIDAR_TILE,
    OUTSIDE_TILE_INDEX,
    SCHEMA_VERSION,
    TILE_UNREADABLE,
    output_schema,
)
from pv_geom.summary import summarise_table

log = logging.getLogger(__name__)

# partition_id of rows that belong to no tile-group partition.
UNMEASURED_PARTITION = -1


def _clock(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


@dataclass
class _Collector:
    """Receives each finished group: writes its partition, keeps its table for
    the run statistics, records failures, and reports progress."""

    sink: OutputSink
    crs: str
    metadata: dict[bytes, str]
    total: int = 0                            # groups this run will attempt
    tables: list[pa.Table] = field(default_factory=list)
    failed: dict[int, str] = field(default_factory=dict)
    done: int = 0
    rows_written: int = 0
    started: float = field(default_factory=time.monotonic)

    def on_done(self, partition_id: int, table: pa.Table) -> None:
        self.done += 1
        if len(table):
            self.tables.append(table)
            target = self.sink.write_part(table, partition_id, self.crs, self.metadata)
            self.rows_written += len(table)
            log.debug("wrote %s (%d rows)", target, len(table))
        self._progress()

    def on_error(self, partition_id: int, group: TileGroup, exc: BaseException) -> None:
        self.done += 1
        self.failed[partition_id] = repr(exc)
        log.error("tile group %s (partition %d, %d polygons) FAILED: %r",
                  group.primary_tile_id, partition_id, len(group.polygon_ids), exc)
        self._progress()

    def _progress(self) -> None:
        elapsed = time.monotonic() - self.started
        remaining = elapsed / self.done * (self.total - self.done) if self.done else 0.0
        log.info("[%d/%d groups] %s rows, %d failed, elapsed %s, about %s left",
                 self.done, self.total, f"{self.rows_written:,}", len(self.failed),
                 _clock(elapsed), _clock(remaining))


def _check_resumable(plan: Plan, sink: OutputSink, kept: set[int], force: bool) -> None:
    """Refuse to resume into an output made from something else.

    Partition ids only mean the same polygons if the plan is the same, and rows
    are only comparable if the measurement settings and schema are. Each
    partition records all three; the first and last kept ones are checked.
    """
    if not kept or force:
        return
    expected = {
        META_SCHEMA_VERSION: (SCHEMA_VERSION, "schema version"),
        META_RESULT_HASH: (plan.cfg.result_hash(), "measurement configuration"),
        META_PLAN_FINGERPRINT: (plan.fingerprint, "inputs (polygons or tile index)"),
    }
    for pid in sorted({min(kept), max(kept)}):
        found = sink.part_metadata(pid)
        for key, (want, what) in expected.items():
            got = found.get(key, b"").decode()
            if got != want:
                was = f"was made with a different {what}" if got else \
                    f"records no {what} (it predates schema {SCHEMA_VERSION})"
                raise ResumeMismatchError(
                    f"cannot resume: partition {pid} at {sink.uri} {was}",
                    "write to a new --output, or pass --force-resume to keep the existing "
                    "partitions anyway (the output will then mix the two)",
                )


def _pending_groups(plan: Plan, sink: OutputSink, resume: bool,
                    force: bool) -> tuple[list[PendingGroup], set[int]]:
    """Groups still to run, and the partition ids ``--resume`` is keeping."""
    kept = sink.existing_part_ids() & set(range(len(plan.groups))) if resume else set()
    _check_resumable(plan, sink, kept, force)
    if kept:
        log.info("resume: keeping %d already-written partitions", len(kept))
    return [(pid, g) for pid, g in enumerate(plan.groups) if pid not in kept], kept


def _concat(tables: list[pa.Table]) -> pa.Table | None:
    tables = [t.replace_schema_metadata(None) for t in tables if len(t)]
    return pa.concat_tables(tables, promote_options="default") if tables else None


def _unmeasured_table(plan: Plan, measured_ids: set[str], failed: dict[int, str],
                      run_id: str) -> pa.Table:
    """Rows for every in-scope polygon that has no measured row, with the reason.

    Every input polygon gets exactly one row. Those a tile-group task produced
    are in the numbered partitions; the rest are accounted for here: features
    with no usable geometry, polygons off the tile index, polygons whose tile is
    not in storage, and polygons whose group failed.
    """
    prov = Provenance(plan.config_hash, run_id, UNMEASURED_PARTITION)
    rows: list[dict[str, Any]] = []

    def _add(frame: gpd.GeoDataFrame, status: str) -> None:
        has_vintage = "polygon_vintage" in frame.columns
        has_flags = "input_flags" in frame.columns
        for rec in frame.itertuples(index=False):
            vintage = getattr(rec, "polygon_vintage", None) if has_vintage else None
            if vintage is None or vintage != vintage:          # None or NaT
                vintage = plan.polygon_vintage
            rows.append(unmeasured_row(
                polygon_id=rec.polygon_id, parent_polygon_id=rec.parent_polygon_id,
                input_row=int(rec.input_row), polygon=rec.geometry, status=status,
                polygon_vintage=vintage,
                flags=tuple(rec.input_flags) if has_flags else (), prov=prov,
            ))

    _add(plan.invalid, INVALID_GEOMETRY)
    _add(plan.outside, OUTSIDE_TILE_INDEX)
    for pid, group in enumerate(plan.groups):
        missing = [p for p in group.polygon_ids if p not in measured_ids]
        if missing:
            _add(plan.polygons.loc[missing],
                 TILE_UNREADABLE if pid in failed else NO_LIDAR_TILE)
    return rows_to_table(rows, output_schema(plan.cfg.mounting_rules.enabled))


def run_pipeline(
    *,
    polygons_uri: str,
    lidar_prefix: str,
    output_uri: str,
    cfg: PVGeomConfig,
    tile_index_uri: str | None = None,
    footprints_uri: str | None = None,
    name_template: str = "{name}.laz",
    tile_id_col: str | None = None,
    polygon_id_col: str | None = None,
    max_polygons: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    dry_run: bool = False,
    resume: bool = False,
    force_resume: bool = False,
    use_dask: bool = True,
) -> Path | str:
    """End-to-end pipeline. Returns the manifest path (str when output is s3://).

    Two inputs are required — the polygon layer and the LiDAR (tiles plus their
    index, or just the tiles: without ``tile_index_uri`` the index is read
    from their headers) — each with a capture date under ``cfg.vintage``. Building
    footprints are optional: without them ``on_building`` is null and the roof
    reference comes from an open ring around each polygon.

    Output layout::

        <output_uri>/
            part-<partition_id>.parquet     one per tile group
            part-unmeasured.parquet         polygons that reached no tile group
            manifest.json

    Every in-scope input polygon has exactly one row; ``status`` says what
    happened to it. Partitions are written as each tile group completes, and a
    failed group is recorded (``counts.failed_groups``,
    ``aggregate_stats.task_errors``) rather than aborting the run.

    ``resume`` keeps the partitions already at the output and runs the rest. It
    refuses if they were made from different inputs, measurement settings or
    schema — each partition records all three — unless ``force_resume``.
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
            plan_fingerprint=plan.fingerprint,
            result_hash=cfg.result_hash(),
        )

    base_counts = {"polygons": plan.n_polygons, "tile_groups": len(plan.groups)}
    if dry_run:
        _manifest({"backend": "dry_run"},
                  {**base_counts, "attempted": 0, "succeeded": 0, "failed": 0},
                  {"dry_run": True})
        return sink.manifest_target

    pending, kept = _pending_groups(plan, sink, resume, force_resume)
    # On Coiled the laptop is not on the data path — workers sit beside the
    # bucket — so tiles are listed rather than downloaded here.
    prewarm = bool(pending) and use_dask and cfg.compute.backend != "coiled"
    pending = drop_missing_tiles(pending, plan, find_missing_tiles(pending, plan, prewarm=prewarm))

    metadata = {META_RESULT_HASH: cfg.result_hash(), META_PLAN_FINGERPRINT: plan.fingerprint}
    collector = _Collector(sink, plan.crs, metadata, total=len(pending))
    execute(pending, lambda pid, g: plan.task_kwargs(pid, g, run_id),
            collector.on_done, collector.on_error, cfg, use_dask=use_dask)
    if collector.failed:
        log.error("%d tile-group tasks failed; rerun with --resume to retry them",
                  len(collector.failed))

    # Account for every polygon no task produced a row for, then summarise the
    # whole output: this run's partitions, resumed ones, and the unmeasured.
    resumed = [sink.read_part(pid) for pid in sorted(kept)]
    produced = _concat(collector.tables + resumed)
    measured_ids = set(produced.column("polygon_id").to_pylist()) if produced is not None else set()
    unmeasured = _unmeasured_table(plan, measured_ids, collector.failed, run_id)
    target = sink.write_unmeasured(unmeasured, plan.crs, metadata)
    if target is not None:
        log.info("%d polygons were not measured; their rows are in %s",
                 len(unmeasured), Path(str(target)).name)

    full = _concat([t for t in (produced, unmeasured) if t is not None])
    stats = summarise_table(full) if full is not None else {}
    if collector.failed:
        stats["task_errors"] = collector.failed
    if produced is not None:
        vintage.update(row_vintage_summary(produced))

    succeeded = len(produced) if produced is not None else 0
    _manifest(
        {"backend": cfg.compute.backend, "use_dask": use_dask},
        {**base_counts, "rows": succeeded + len(unmeasured),
         "attempted": plan.n_attempted, "succeeded": succeeded,
         "failed": plan.n_attempted - succeeded, "failed_groups": len(collector.failed),
         "not_measured": len(unmeasured)},
        stats,
    )
    return sink.manifest_target
