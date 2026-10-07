"""Top-level pipeline orchestrator. M6.

Reads polygons, footprints, and the tile index; partitions polygons by primary
tile; dispatches per-tile-group tasks (serial today; Dask LocalCluster behind
``compute.backend == 'local'``; Coiled in M7); writes per-partition Parquet
plus a JSON run manifest.
"""

from __future__ import annotations

import contextlib
import traceback
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pv_geom.config import PVGeomConfig
from pv_geom.io._localize import RemoteFileMissing, is_remote, list_s3_uris, localize
from pv_geom.io.footprints import read_footprints
from pv_geom.io.lidar import read_tile_vintage
from pv_geom.io.polygons import read_polygons
from pv_geom.io.tile_index import build_tile_uris, load_tile_index
from pv_geom.pipeline.partition import (
    TileGroup,
    assign_polygons_to_tiles,
    build_tile_groups,
)
from pv_geom.pipeline.tile_task import process_tile_group
from pv_geom.provenance import write_manifest


def run_pipeline(
    *,
    polygons_uri: str,
    tile_index_uri: str,
    lidar_prefix: str,
    footprints_uri: str,
    output_uri: str,
    cfg: PVGeomConfig,
    name_template: str = "{name}.laz",
    tile_id_col: str = "Name",
    polygon_id_col: str = "polygon_id",
    max_polygons: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    dry_run: bool = False,
    resume: bool = False,
    use_dask: bool = True,
) -> Path | str:
    """End-to-end pipeline. Returns the manifest path (str when output is s3://).

    Output layout::

        <output_uri>/
            part-<partition_id>.parquet
            ...
            logs/
            manifest.json

    ``resume`` is for crash recovery on the SAME inputs and config — partition
    files matching ``part-<id>.parquet`` already on disk are kept as-is and
    their groups are skipped. Changing inputs/config between runs while using
    ``--resume`` will produce inconsistent output (partition_ids shift). The
    manifest's ``config_hash`` is logged so you can detect drift after the fact.

    Partitions are written incrementally as each tile group completes, and a
    failed group is logged + recorded in the manifest (``counts.failed_groups``,
    ``aggregate_stats.task_errors``) rather than aborting the run — so completed
    work survives mid-run failures and ``--resume`` retries only the failures.
    """
    # Output target: local directory or an s3:// prefix. S3 writes happen on
    # the CLIENT (per-group tables come back to the client either way), so no
    # worker-side write permissions are needed — only the client's creds.
    out_str = str(output_uri).rstrip("/")
    s3_output = out_str.startswith("s3://")
    if s3_output:
        import fsspec

        out_fs = fsspec.filesystem("s3")
        out_root: Path | None = None
    else:
        out_root = Path(output_uri)
        out_root.mkdir(parents=True, exist_ok=True)

    def _part_target(partition_id: int) -> str | Path:
        name = f"part-{partition_id:05d}.parquet"
        return f"{out_str}/{name}" if s3_output else out_root / name

    def _existing_part_ids() -> set[int]:
        """Partition ids with a non-empty part file already at the output."""
        found: set[int] = set()
        if s3_output:
            try:
                infos = out_fs.find(out_str, detail=True)
            except FileNotFoundError:
                return found
            entries = [(i.get("name", ""), i.get("size", 0)) for i in infos.values()]
        else:
            entries = [(p.name, p.stat().st_size) for p in out_root.glob("part-*.parquet")]
        for name, size in entries:
            base = str(name).rsplit("/", 1)[-1]
            if base.startswith("part-") and base.endswith(".parquet") and size > 0:
                try:
                    found.add(int(base[len("part-"):-len(".parquet")]))
                except ValueError:
                    continue
        return found

    manifest_target: str | Path = (
        f"{out_str}/manifest.json" if s3_output else out_root / "manifest.json"
    )

    run_id = uuid.uuid4().hex
    config_hash = cfg.hash()

    # 1) Inputs ---------------------------------------------------------------
    polygons = read_polygons(
        polygons_uri,
        target_crs=cfg.crs.target,
        id_col=polygon_id_col,
        bbox=bbox,
        max_polygons=max_polygons,
    )
    footprints = read_footprints(footprints_uri, target_crs=cfg.crs.target)
    tindex = load_tile_index(tile_index_uri, target_crs=cfg.crs.target)

    if "tile_path" not in tindex.columns:
        tindex = build_tile_uris(
            tindex,
            base_uri=lidar_prefix,
            name_col=tile_id_col,
            name_template=name_template,
        )
    tile_uri_map = dict(zip(tindex[tile_id_col].astype(str), tindex["tile_path"]))

    # 2) Partition ------------------------------------------------------------
    assignments = assign_polygons_to_tiles(
        polygons, tindex, polygon_id_col=polygon_id_col, tile_id_col=tile_id_col,
    )
    groups = build_tile_groups(assignments)
    print(f"[runner] {len(polygons)} polygons -> {len(groups)} tile groups")

    # Vintage check up front: if the inventory postdates the point cloud, that
    # is worth knowing before a multi-hour run, and `--dry-run` becomes a way to
    # ask the question on its own.
    vintage = _probe_lidar_vintage(
        cfg,
        [uri for tid in {t for g in groups for t in g.fetch_tile_ids}
         if (uri := tile_uri_map.get(tid))],
    )

    if dry_run:
        write_manifest(
            manifest_target,
            config_dict=cfg.model_dump(mode="json"),
            config_hash=config_hash,
            inputs={
                "polygons": str(polygons_uri),
                "tile_index": str(tile_index_uri),
                "lidar_prefix": str(lidar_prefix),
                "footprints": str(footprints_uri),
            },
            cluster_spec={"backend": "dry_run"},
            counts={"polygons": int(len(polygons)),
                    "tile_groups": int(len(groups)),
                    "attempted": 0, "succeeded": 0, "failed": 0},
            aggregate_stats={"dry_run": True},
            tiles_touched=sorted({t for g in groups for t in g.fetch_tile_ids}),
            run_id=run_id,
            vintage=vintage,
        )
        return manifest_target

    # 3) Per-group worker dispatch -------------------------------------------
    polygon_id_set_per_group = [set(g.polygon_ids) for g in groups]
    polygons_indexed = polygons.set_index(polygon_id_col, drop=False)

    # Resume support — skip groups whose partition file already exists at the
    # output (one LIST for s3 outputs rather than a HEAD per group).
    skipped_ids: set[int] = _existing_part_ids() if resume else set()
    skipped_ids &= set(range(len(groups)))
    if skipped_ids:
        print(
            f"[runner] resume: skipping {len(skipped_ids)} already-written partitions"
        )

    pending = [
        (partition_id, group)
        for partition_id, group in enumerate(groups)
        if partition_id not in skipped_ids
    ]

    # Pre-warm the LAZ cache in the main process. The original motivation was
    # the Dask + AWS SSO race on Windows: parallel workers stomp on the SSO
    # token cache file and crash with PermissionError, so we serialize the
    # downloads. The pre-warm also doubles as a cheap way to detect missing
    # tiles upfront so we can drop them from each group's fetch list before
    # dispatch (the bucket has known gaps relative to the tile index).
    #
    # On Coiled the laptop is not on the data path: workers run in the same
    # AWS region as the bucket, use IAM roles (no SSO race), and pre-warming
    # to the laptop would just transfer the data to a machine that won't
    # consume it. Skip pre-warm there; tile_task.process_tile_group catches
    # RemoteFileMissing per-tile and degrades gracefully.
    missing_uris: set[str] = set()
    do_prewarm = pending and use_dask and cfg.compute.backend != "coiled"
    if do_prewarm:
        unique_uris = sorted({
            tile_uri_map[tid] for _, g in pending
            for tid in g.fetch_tile_ids
            if tile_uri_map.get(tid) and is_remote(tile_uri_map[tid])
        })
        if unique_uris:
            print(f"[runner] pre-warming {len(unique_uris)} LAZ tiles into cache "
                  f"(serial; avoids the SSO race)")
            for i, uri in enumerate(unique_uris):
                try:
                    local = localize(uri)
                    size_mb = local.stat().st_size / 1e6 if local.exists() else 0
                    print(f"  [{i+1}/{len(unique_uris)}] {Path(uri).name} -> "
                          f"{size_mb:.0f} MB local")
                except RemoteFileMissing:
                    print(f"  [{i+1}/{len(unique_uris)}] {Path(uri).name} -> "
                          f"MISSING (will be skipped)")
                    missing_uris.add(uri)

    # When pre-warm didn't run (Coiled, or serial runs against S3), one
    # paginated LIST from the client finds tiles absent from the bucket —
    # ~75% of indexed Phoenix tiles have no LAZ — so dead groups are dropped
    # before dispatch instead of every worker paying a 404 round-trip.
    if pending and not do_prewarm and str(lidar_prefix).startswith("s3://"):
        try:
            available = list_s3_uris(lidar_prefix)
        except Exception as exc:
            # No client-side S3 creds / network: workers still 404 gracefully.
            print(f"[runner] S3 listing of {lidar_prefix} failed ({exc!r}); "
                  f"missing tiles will be handled worker-side")
        else:
            prefix = str(lidar_prefix).rstrip("/") + "/"
            for _, g in pending:
                for tid in g.fetch_tile_ids:
                    uri = tile_uri_map.get(tid)
                    if uri and uri.startswith(prefix) and uri not in available:
                        missing_uris.add(uri)
            if missing_uris:
                print(f"[runner] {len(missing_uris)} indexed tiles have no LAZ "
                      f"under {lidar_prefix}; dropping them up-front")

    # Drop missing tiles from each group's fetch list. If the primary tile
    # itself is missing, the group is dropped from `pending` (no points to fit).
    if missing_uris:
        filtered_pending: list[tuple[int, TileGroup]] = []
        n_dropped_polys = 0
        for partition_id, g in pending:
            primary_uri = tile_uri_map.get(g.primary_tile_id)
            if primary_uri in missing_uris:
                n_dropped_polys += len(g.polygon_ids)
                continue
            kept = tuple(
                tid for tid in g.fetch_tile_ids
                if tile_uri_map.get(tid) not in missing_uris
            )
            filtered_pending.append(
                (partition_id, TileGroup(
                    primary_tile_id=g.primary_tile_id,
                    polygon_ids=g.polygon_ids,
                    fetch_tile_ids=kept,
                ))
            )
        if n_dropped_polys:
            print(f"[runner] dropped {n_dropped_polys} polygons whose primary tile is missing")
        pending = filtered_pending

    def _build_task(partition_id: int, group):
        sub = polygons_indexed.loc[list(polygon_id_set_per_group[partition_id])].copy()
        sub_bbox = sub.total_bounds
        margin = max(50.0, cfg.roof_plane.buffer_max_m * 4.0)
        fp_sub = footprints.cx[
            sub_bbox[0] - margin: sub_bbox[2] + margin,
            sub_bbox[1] - margin: sub_bbox[3] + margin,
        ]
        # Subset the URI map to this group's tiles: embedding the full index
        # (13k+ entries, ~MBs serialized) in every task inflates the graph to
        # GBs at atlas scale (~1000-2000 groups).
        group_uri_map = {
            tid: tile_uri_map[tid]
            for tid in group.fetch_tile_ids
            if tid in tile_uri_map
        }
        return dict(
            tile_uri_map=group_uri_map,
            primary_tile_id=group.primary_tile_id,
            polygons=sub,
            fetch_tile_ids=group.fetch_tile_ids,
            footprints=fp_sub,
            cfg=cfg,
            config_hash=config_hash,
            run_id=run_id,
            partition_id=partition_id,
            polygon_id_col=polygon_id_col,
        )

    # 4) Dispatch + write each partition as soon as its group finishes. A
    # failed group is recorded and skipped rather than aborting the run, so a
    # multi-hour atlas run keeps every completed partition on disk and a
    # follow-up `--resume` retries only the failures. (Previously a single
    # dask.compute barrier wrote nothing until ALL tasks succeeded — one
    # KilledWorker discarded hours of completed work.)
    new_tables: list[pa.Table] = []
    failed_groups: dict[int, str] = {}
    n_succeeded_new = 0

    def _on_done(partition_id: int, table: pa.Table) -> None:
        nonlocal n_succeeded_new
        new_tables.append(table)
        if len(table) == 0:
            return
        target = _part_target(partition_id)
        if s3_output:
            with out_fs.open(str(target), "wb") as f:
                pq.write_table(table, f)
        else:
            pq.write_table(table, target)
        n_succeeded_new += len(table)
        print(f"[runner] wrote {target} ({len(table)} rows)")

    def _on_error(partition_id: int, group: TileGroup, exc: BaseException) -> None:
        failed_groups[partition_id] = repr(exc)
        print(f"[runner] tile group {group.primary_tile_id} (partition "
              f"{partition_id}, {len(group.polygon_ids)} polygons) FAILED: {exc!r}")

    if use_dask and pending:
        with _cluster_for(cfg) as client:
            from dask.distributed import as_completed

            # Bounded-in-flight submission. Building every task payload up
            # front (the pre-2026-07-29 dict comprehension) materializes and
            # serializes ~3k footprint/polygon subsets on the client before
            # the first result returns — at atlas scale that is multi-GB and
            # took the client down. A submit window a few times the worker
            # count keeps the cluster saturated at bounded client memory.
            if cfg.compute.backend == "coiled":
                n_workers = cfg.compute.coiled.n_workers
            elif cfg.compute.backend == "local":
                n_workers = cfg.compute.local.n_workers or 4
            else:
                n_workers = 8
            max_inflight = max(32, 4 * int(n_workers))

            pending_iter = iter(pending)
            inflight: dict = {}
            ac = as_completed()

            def _submit_next() -> bool:
                try:
                    pid, g = next(pending_iter)
                except StopIteration:
                    return False
                fut = client.submit(process_tile_group, **_build_task(pid, g))
                inflight[fut] = (pid, g)
                ac.add(fut)
                return True

            for _ in range(min(max_inflight, len(pending))):
                _submit_next()
            # A lost scheduler connection cancels every in-flight future;
            # FutureCancelledError is a BaseException (CancelledError), and
            # letting it propagate discards the manifest for hours of
            # already-written partitions. Drain instead: record the
            # remainder as failed and fall through to the manifest write —
            # rerunning with --resume retries exactly those groups.
            from concurrent.futures import CancelledError

            try:
                for fut in ac:
                    pid, g = inflight.pop(fut)
                    if fut.status == "error":
                        _on_error(pid, g, fut.exception())
                    elif fut.status == "cancelled":
                        _on_error(pid, g, RuntimeError("future cancelled (connection lost?)"))
                    else:
                        _on_done(pid, fut.result())
                    fut.release()
                    _submit_next()
            except (Exception, CancelledError) as exc:
                for pid, g in inflight.values():
                    _on_error(pid, g, exc)
                for pid, g in pending_iter:
                    _on_error(pid, g, RuntimeError("not dispatched (run aborted)"))
                print(f"[runner] dispatch aborted ({exc!r}); writing manifest "
                      f"for completed work — rerun with --resume to retry")
    elif pending:
        for pid, g in pending:
            try:
                table = process_tile_group(**_build_task(pid, g))
            except Exception as exc:
                traceback.print_exc()
                _on_error(pid, g, exc)
            else:
                _on_done(pid, table)

    n_attempted = sum(len(g.polygon_ids) for g in groups)
    if failed_groups:
        n_failed_polys = sum(
            len(g.polygon_ids) for pid, g in pending if pid in failed_groups
        )
        print(f"[runner] {len(failed_groups)} tile-group tasks failed "
              f"({n_failed_polys} polygons); rerun with --resume to retry them")

    # 4b) Read back skipped partitions so manifest stats include them ---------
    skipped_tables: list[pa.Table] = []
    for pid in sorted(skipped_ids):
        target = _part_target(pid)
        if s3_output:
            with out_fs.open(str(target), "rb") as f:
                skipped_tables.append(pq.read_table(f))
        else:
            skipped_tables.append(pq.read_table(target))
    n_succeeded_resumed = sum(len(t) for t in skipped_tables)
    n_succeeded = n_succeeded_new + n_succeeded_resumed
    tables = list(new_tables) + skipped_tables

    # 5) Aggregate stats + manifest ------------------------------------------
    full_table = pa.concat_tables(list(tables)) if tables else None
    aggregate_stats = _aggregate(full_table) if full_table is not None and len(full_table) else {}
    if failed_groups:
        aggregate_stats["task_errors"] = failed_groups

    write_manifest(
        manifest_target,
        config_dict=cfg.model_dump(mode="json"),
        config_hash=config_hash,
        inputs={
            "polygons": str(polygons_uri),
            "tile_index": str(tile_index_uri),
            "lidar_prefix": str(lidar_prefix),
            "footprints": str(footprints_uri),
        },
        cluster_spec={"backend": cfg.compute.backend, "use_dask": use_dask},
        counts={
            "polygons": int(len(polygons)),
            "tile_groups": int(len(groups)),
            "attempted": int(n_attempted),
            "succeeded": int(n_succeeded),
            "failed": int(n_attempted - n_succeeded),
            "failed_groups": len(failed_groups),
        },
        aggregate_stats=aggregate_stats,
        tiles_touched=sorted({t for g in groups for t in g.fetch_tile_ids}),
        run_id=run_id,
        vintage=vintage,
    )
    return manifest_target


def _probe_lidar_vintage(cfg: PVGeomConfig, tile_uris: list[str]) -> dict:
    """Measure when the LiDAR was flown and compare it to the declared inputs.

    Returns the manifest's ``vintage`` block and prints a warning when the
    inventory's imagery postdates the point cloud — the case where arrays exist
    in the input but not in the data being used to measure them, so the panel
    columns silently report bare roof. Never raises: an unreadable tile costs a
    sample, not the run.
    """
    declared = cfg.vintage.input_epoch
    out: dict = {
        "input_epoch": declared.isoformat() if declared else None,
        "lidar_tiles_sampled": 0,
    }
    n = int(cfg.vintage.sample_tiles)
    if n <= 0 or not tile_uris:
        return out

    # Even stride over the sorted URIs so the sample spans the AOI. A 3DEP work
    # unit is flown in strips over weeks, so one corner is not representative.
    ordered = sorted(set(tile_uris))
    step = max(1, len(ordered) // n)
    candidates = ordered[::step][:n]

    samples = []
    n_errors = 0
    for uri in candidates:
        try:
            samples.append(read_tile_vintage(uri))
        except Exception:
            n_errors += 1

    out["lidar_tiles_sampled"] = len(samples)
    out["lidar_tiles_unreadable"] = n_errors
    if not samples:
        print("[runner] vintage: no tile could be read; LiDAR epoch unknown")
        return out

    starts = [s.flight_start for s in samples if s.flight_start]
    ends = [s.flight_end for s in samples if s.flight_end]
    creations = [s.creation_date for s in samples if s.creation_date]
    observed = ends or creations

    out["lidar_flight_start"] = min(starts).isoformat() if starts else None
    out["lidar_flight_end"] = max(ends).isoformat() if ends else None
    out["lidar_creation_date_max"] = max(creations).isoformat() if creations else None
    out["lidar_epoch_source"] = "gps_time" if ends else "header_creation_date"
    out["per_tile"] = [s.to_dict() for s in samples]

    if not observed:
        return out
    latest = max(observed)
    print(
        f"[runner] vintage: LiDAR flown "
        f"{min(starts).isoformat() if starts else '?'} -> {latest.isoformat()} "
        f"(source: {out['lidar_epoch_source']}, {len(samples)} tiles sampled)"
    )

    if declared is None:
        print(
            "[runner] vintage: vintage.input_epoch is unset — cannot check whether "
            "the inventory postdates the LiDAR. Arrays built after "
            f"{latest.isoformat()} would be measured as bare roof."
        )
        return out

    gap_days = (declared - latest).days
    out["vintage_gap_days"] = gap_days
    if gap_days > 0:
        out["vintage_gap_warning"] = (
            f"input imagery postdates LiDAR by {gap_days} days"
        )
        print(
            f"[runner] *** VINTAGE GAP: the inventory's imagery ({declared.isoformat()}) "
            f"postdates the LiDAR ({latest.isoformat()}) by {gap_days} days "
            f"({gap_days / 365.25:.1f} years). Arrays installed in that window are "
            f"in the input but not in the point cloud: their rows will report the "
            f"bare roof under the panel columns. Screen on the `no_panel_standoff` "
            f"and `standoff_unscreenable` flags before using panel geometry. ***"
        )
    return out


@contextlib.contextmanager
def _cluster_for(cfg: PVGeomConfig):
    """Context manager that yields a Dask Client matching ``cfg.compute.backend``.

    - ``local``: spins up a ``distributed.LocalCluster`` per ``cfg.compute.local``.
    - ``coiled``: spins up a Coiled cluster per ``cfg.compute.coiled``.
    - anything else: yields ``None`` (default scheduler — threaded for sync).
    """
    backend = cfg.compute.backend
    if backend == "local":
        from dask.distributed import Client, LocalCluster

        cluster = LocalCluster(
            n_workers=cfg.compute.local.n_workers,
            threads_per_worker=cfg.compute.local.threads_per_worker,
            processes=True,
        )
        client = Client(cluster)
        try:
            print(f"[runner] LocalCluster up at {client.dashboard_link}")
            yield client
        finally:
            client.close()
            cluster.close()
    elif backend == "coiled":
        from dask.distributed import Client

        from pv_geom.coiled_env import (
            ensure_software_env,
            install_pv_geom_on_workers,
            make_cluster,
        )

        ensure_software_env()
        cluster = make_cluster(cfg)
        client = Client(cluster)
        try:
            print(f"[runner] Coiled cluster up: {client.dashboard_link}")
            install_pv_geom_on_workers(client)
            yield client
        finally:
            client.close()
            cluster.close()
    else:
        yield None


def _aggregate(table: pa.Table) -> dict:
    """Manifest-friendly summary stats."""
    df = table.to_pandas()
    out: dict = {
        "mounting_type_counts": df["mounting_type"].value_counts().to_dict(),
        "mounting_rule_counts": df["mounting_rule"].value_counts().to_dict(),
        "on_building_count": int(df["on_building"].sum()),
        "off_building_count": int((~df["on_building"]).sum()),
    }
    confidences = df["mounting_confidence"].dropna()
    if len(confidences):
        out["mounting_confidence_p50"] = float(confidences.median())
        out["mounting_confidence_p10"] = float(confidences.quantile(0.10))
    rmses = df["panel_rmse_m"].dropna()
    if len(rmses):
        out["panel_rmse_p50"] = float(rmses.median())
        out["panel_rmse_p90"] = float(rmses.quantile(0.90))
    flag_counts: dict[str, int] = {}
    flag_sets: list[set] = []
    for flags in df["flags"]:
        # `flags` may be a numpy array, list, or None; iterate uniformly.
        s = set(flags) if flags is not None else set()
        flag_sets.append(s)
        for f in s:
            flag_counts[f] = flag_counts.get(f, 0) + 1
    out["flag_counts"] = flag_counts

    # Panel-standoff (vintage) screen. The README asks callers to report the
    # flagged fraction alongside any fleet-level geometry statistic; reporting
    # it here means they cannot forget to. The cross-tab is the part that
    # matters: it shows which labels rest on unmeasured panel geometry.
    import pandas as pd

    failed = pd.Series([("no_panel_standoff" in s) for s in flag_sets], index=df.index)
    unscreenable = pd.Series([("standoff_unscreenable" in s) for s in flag_sets], index=df.index)
    state = pd.Series("passed", index=df.index)
    state[failed] = "no_panel_standoff"
    state[unscreenable] = "unscreenable"
    n = len(df)
    out["standoff_screen"] = {
        "passed": int((state == "passed").sum()),
        "no_panel_standoff": int(failed.sum()),
        "unscreenable": int(unscreenable.sum()),
        "screened_frac": float((~unscreenable).mean()) if n else 0.0,
        "failed_frac_of_screened": (
            float(failed.sum() / (~unscreenable).sum()) if (~unscreenable).sum() else None
        ),
        "by_mounting_type": {
            str(k): {str(kk): int(vv) for kk, vv in v.items()}
            for k, v in pd.crosstab(df["mounting_type"], state).to_dict("index").items()
        },
    }

    har = df["height_above_roof_m"].dropna()
    if len(har):
        out["height_above_roof_p10"] = float(har.quantile(0.10))
        out["height_above_roof_p50"] = float(har.median())
        out["height_above_roof_p90"] = float(har.quantile(0.90))
    return out
