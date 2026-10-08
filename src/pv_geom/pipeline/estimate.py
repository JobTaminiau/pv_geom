"""A rough estimate of what a run will take, before it is started.

The estimate is deliberately simple: reading and decoding tiles costs time in
proportion to their size, and measuring polygons costs time per polygon. The two
rates below were fitted to four local runs made on 2026-10-07 (the Phoenix and
Delaware test blocks, 0.2 and 0.3 code, 4-5 workers):

    Phoenix   11 tiles, 1.5 GB, 3,390 polygons   ~2 min of compute
    Delaware  19 tiles, 9.0 GB,   816 polygons   ~7-9 min of compute

Expect the truth within a factor of two either way: tile density, worker
hardware and network distance all move it. It is for deciding whether a run is
minutes, hours or days, and roughly what it costs — not for scheduling.
"""

from __future__ import annotations

import logging
from typing import Any

from pv_geom.config import PVGeomConfig
from pv_geom.io.storage import object_sizes
from pv_geom.pipeline.plan import PendingGroup, Plan

log = logging.getLogger(__name__)

WORKER_SECONDS_PER_GB = 240.0          # read + decode + index one GB of LAZ
WORKER_SECONDS_PER_POLYGON = 0.06      # fits, bootstrap, roof reference
COILED_STARTUP_SECONDS = 150.0         # cluster spin-up and worker install


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def estimate_run(plan: Plan, pending: list[PendingGroup], cfg: PVGeomConfig,
                 *, use_dask: bool = True) -> dict[str, Any]:
    """Tiles, bytes, time and (when a price is configured) cost for the pending groups."""
    uris = sorted({plan.tile_uri_map[t] for _, g in pending for t in g.fetch_tile_ids
                   if t in plan.tile_uri_map})
    try:
        sizes = object_sizes(uris)
    except Exception as exc:                       # no credentials, no network: still plan
        log.warning("could not size the tiles (%r); the estimate omits reading time", exc)
        sizes = {}
    n_polygons = sum(len(g.polygon_ids) for _, g in pending)
    n_bytes = sum(sizes.values())

    backend = cfg.compute.backend
    if not use_dask:
        workers = 1
    elif backend == "coiled":
        workers = int(cfg.compute.coiled.n_workers)
    else:
        workers = int(cfg.compute.local.n_workers or 4)
    # No more workers can be busy than there are groups to give them.
    busy = max(1, min(workers, len(pending)))

    worker_seconds = (n_bytes / 1e9 * WORKER_SECONDS_PER_GB
                      + n_polygons * WORKER_SECONDS_PER_POLYGON)
    wall = worker_seconds / busy
    if use_dask and backend == "coiled":
        wall += COILED_STARTUP_SECONDS

    out: dict[str, Any] = {
        "tile_groups": len(pending),
        "polygons": n_polygons,
        "tiles": len(uris),
        "tiles_sized": len(sizes),
        "gigabytes": round(n_bytes / 1e9, 2),
        "workers": workers,
        "worker_hours": round(worker_seconds / 3600, 2),
        "wall_seconds": round(wall),
        "wall_time": _duration(wall),
        "accuracy": "rough: expect within a factor of two",
    }
    price = cfg.compute.coiled.usd_per_worker_hour
    if use_dask and backend == "coiled" and price:
        # Workers are billed while the cluster is up, busy or not.
        out["usd"] = round(workers * wall / 3600 * price, 2)
    return out


def log_estimate(est: dict[str, Any]) -> None:
    cost = f", about ${est['usd']:,.2f}" if "usd" in est else ""
    log.info("estimate: %d tile groups, %s polygons, %d tiles (%.1f GB) on %d workers "
             "-> about %s%s (rough: within a factor of two)",
             est["tile_groups"], f"{est['polygons']:,}", est["tiles"], est["gigabytes"],
             est["workers"], est["wall_time"], cost)
