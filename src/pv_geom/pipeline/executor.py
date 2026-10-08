"""Run tile-group tasks: serially, on a local Dask cluster, or on Coiled.

Whatever the backend, results are handed back one group at a time as they
finish, and a failed group is reported rather than raised — so a long run keeps
every completed partition and ``--resume`` retries only the failures.
"""

from __future__ import annotations

import contextlib
import logging
import traceback
from collections.abc import Callable, Iterator
from typing import Any

import pyarrow as pa

from pv_geom.config import PVGeomConfig
from pv_geom.errors import require
from pv_geom.pipeline.partition import TileGroup
from pv_geom.pipeline.plan import PendingGroup
from pv_geom.pipeline.worker import process_tile_group

log = logging.getLogger(__name__)

TaskFor = Callable[[int, TileGroup], dict[str, Any]]
OnDone = Callable[[int, pa.Table], None]
OnError = Callable[[int, TileGroup, BaseException], None]


@contextlib.contextmanager
def cluster_for(cfg: PVGeomConfig) -> Iterator[Any]:
    """Yield a Dask ``Client`` for ``cfg.compute.backend`` (``local`` or
    ``coiled``), closing the cluster afterwards."""
    from dask.distributed import Client

    backend = cfg.compute.backend
    if backend == "local":
        from dask.distributed import LocalCluster

        cluster = LocalCluster(
            n_workers=cfg.compute.local.n_workers,
            threads_per_worker=cfg.compute.local.threads_per_worker,
            processes=True,
        )
    elif backend == "coiled":
        require("coiled", "coiled", "the coiled compute backend")
        from pv_geom.coiled_env import ensure_software_env, make_cluster

        ensure_software_env()
        cluster = make_cluster(cfg)
    else:
        raise ValueError(f"unknown compute backend {backend!r}")

    client = Client(cluster)
    try:
        log.info("%s cluster up: %s", backend, client.dashboard_link)
        if backend == "coiled":
            from pv_geom.coiled_env import install_pv_geom_on_workers

            install_pv_geom_on_workers(client, cfg.compute.coiled.package_source)
        yield client
    finally:
        client.close()
        cluster.close()


def _worker_count(cfg: PVGeomConfig) -> int:
    if cfg.compute.backend == "coiled":
        return int(cfg.compute.coiled.n_workers)
    return int(cfg.compute.local.n_workers or 4)


def _run_serial(pending: list[PendingGroup], task_for: TaskFor,
                on_done: OnDone, on_error: OnError) -> None:
    for pid, group in pending:
        try:
            table = process_tile_group(**task_for(pid, group))
        except Exception as exc:
            traceback.print_exc()
            on_error(pid, group, exc)
        else:
            on_done(pid, table)


def _run_distributed(client: Any, pending: list[PendingGroup], task_for: TaskFor,
                     on_done: OnDone, on_error: OnError, max_inflight: int) -> None:
    """Bounded-in-flight submission.

    Building every task payload up front materialises and serialises thousands
    of polygon/footprint subsets on the client before the first result returns
    — at atlas scale that took the client down. A submit window a few times the
    worker count keeps the cluster saturated at bounded client memory.
    """
    from concurrent.futures import CancelledError

    from dask.distributed import as_completed

    queue = iter(pending)
    inflight: dict[Any, PendingGroup] = {}
    done = as_completed()

    def _submit_next() -> None:
        try:
            pid, group = next(queue)
        except StopIteration:
            return
        fut = client.submit(process_tile_group, **task_for(pid, group))
        inflight[fut] = (pid, group)
        done.add(fut)

    for _ in range(min(max_inflight, len(pending))):
        _submit_next()

    # A lost scheduler connection cancels every in-flight future, and the
    # resulting CancelledError is a BaseException: letting it propagate would
    # discard the manifest for hours of already-written partitions. Drain
    # instead — record the remainder as failed and let the caller finish.
    try:
        for fut in done:
            pid, group = inflight.pop(fut)
            if fut.status == "error":
                on_error(pid, group, fut.exception())
            elif fut.status == "cancelled":
                on_error(pid, group, RuntimeError("future cancelled (connection lost?)"))
            else:
                on_done(pid, fut.result())
            fut.release()
            _submit_next()
    except (Exception, CancelledError) as exc:
        for pid, group in inflight.values():
            on_error(pid, group, exc)
        for pid, group in queue:
            on_error(pid, group, RuntimeError("not dispatched (run aborted)"))
        log.error("dispatch aborted (%r); finishing with completed work — "
                  "rerun with --resume to retry", exc)


def execute(pending: list[PendingGroup], task_for: TaskFor, on_done: OnDone,
            on_error: OnError, cfg: PVGeomConfig, *, use_dask: bool) -> None:
    """Run every pending group, calling ``on_done`` or ``on_error`` for each."""
    if not pending:
        return
    if not use_dask:
        _run_serial(pending, task_for, on_done, on_error)
        return
    with cluster_for(cfg) as client:
        _run_distributed(client, pending, task_for, on_done, on_error,
                         max_inflight=max(32, 4 * _worker_count(cfg)))
