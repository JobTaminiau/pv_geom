"""Top-level pipeline orchestrator. M6.

Reads polygons, footprints, and the tile index; partitions polygons by primary
tile; dispatches per-tile-group tasks (serial today; Dask LocalCluster behind
``compute.backend == 'local'``; Coiled in M7); writes per-partition Parquet
plus a JSON run manifest.
"""

from __future__ import annotations

import contextlib
import uuid
from pathlib import Path

import dask
import pyarrow as pa
import pyarrow.parquet as pq

from pv_geom.config import PVGeomConfig
from pv_geom.io._localize import RemoteFileMissing, is_remote, localize
from pv_geom.io.footprints import read_footprints
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
) -> Path:
    """End-to-end pipeline. Returns the manifest path.

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
    """
    out_root = Path(output_uri)
    out_root.mkdir(parents=True, exist_ok=True)

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

    if dry_run:
        manifest_path = out_root / "manifest.json"
        write_manifest(
            manifest_path,
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
        )
        return manifest_path

    # 3) Per-group worker dispatch -------------------------------------------
    polygon_id_set_per_group = [set(g.polygon_ids) for g in groups]
    polygons_indexed = polygons.set_index(polygon_id_col, drop=False)

    # Resume support — skip groups whose partition file is already on disk.
    skipped_paths: list[Path] = []
    if resume:
        for partition_id, _group in enumerate(groups):
            path = out_root / f"part-{partition_id:05d}.parquet"
            if path.exists() and path.stat().st_size > 0:
                skipped_paths.append(path)
        if skipped_paths:
            print(
                f"[runner] resume: skipping {len(skipped_paths)} already-written partitions"
            )

    pending = [
        (partition_id, group)
        for partition_id, group in enumerate(groups)
        if not (resume and (out_root / f"part-{partition_id:05d}.parquet").exists()
                and (out_root / f"part-{partition_id:05d}.parquet").stat().st_size > 0)
    ]

    # Pre-warm the LAZ cache in the main process. With Dask + AWS SSO on
    # Windows, parallel workers race on the SSO token cache file and crash with
    # PermissionError; serial pre-localizing avoids that and is essentially
    # free given workers would download the same tiles anyway. Missing tiles
    # are logged and dropped from each group's fetch list so they don't crash
    # the whole pipeline (the bucket has known gaps relative to the index).
    missing_uris: set[str] = set()
    if pending and use_dask:
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
        return dict(
            tile_uri_map=tile_uri_map,
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

    if use_dask and pending:
        delayed_tasks = [
            dask.delayed(process_tile_group)(**_build_task(pid, g)) for pid, g in pending
        ]
        with _cluster_for(cfg) as client:
            del client                        # context manager keeps it alive
            new_tables = list(dask.compute(*delayed_tasks))
    elif pending:
        new_tables = [process_tile_group(**_build_task(pid, g)) for pid, g in pending]
    else:
        new_tables = []

    # 4) Write new partitions -------------------------------------------------
    n_attempted = sum(len(g.polygon_ids) for g in groups)
    n_succeeded_new = 0
    for (partition_id, _group), table in zip(pending, new_tables, strict=True):
        if len(table) == 0:
            continue
        path = out_root / f"part-{partition_id:05d}.parquet"
        pq.write_table(table, path)
        n_succeeded_new += len(table)
        print(f"[runner] wrote {path} ({len(table)} rows)")

    # 4b) Read back skipped partitions so manifest stats include them ---------
    skipped_tables = [pq.read_table(p) for p in skipped_paths]
    n_succeeded_resumed = sum(len(t) for t in skipped_tables)
    n_succeeded = n_succeeded_new + n_succeeded_resumed
    tables = list(new_tables) + skipped_tables

    # 5) Aggregate stats + manifest ------------------------------------------
    full_table = pa.concat_tables(list(tables)) if tables else None
    aggregate_stats = _aggregate(full_table) if full_table is not None and len(full_table) else {}

    manifest_path = out_root / "manifest.json"
    write_manifest(
        manifest_path,
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
        },
        aggregate_stats=aggregate_stats,
        tiles_touched=sorted({t for g in groups for t in g.fetch_tile_ids}),
        run_id=run_id,
    )
    return manifest_path


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
    for flags in df["flags"]:
        # `flags` may be a numpy array, list, or None; iterate uniformly.
        if flags is None:
            continue
        for f in flags:
            flag_counts[f] = flag_counts.get(f, 0) + 1
    out["flag_counts"] = flag_counts
    return out
