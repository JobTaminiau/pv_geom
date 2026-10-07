"""Run manifest writer. Config hash lives on PVGeomConfig.hash()."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pv_geom import __version__
from pv_geom.schema import SCHEMA_VERSION
from pv_geom.utils.north import TRUE_NORTH


def write_manifest(
    output_path: str | Path,
    *,
    config_dict: dict[str, Any],
    config_hash: str,
    inputs: dict[str, str | None],
    cluster_spec: dict[str, Any],
    counts: dict[str, int],
    aggregate_stats: dict[str, Any],
    tiles_touched: list[str],
    run_id: str,
    vintage: dict[str, Any] | None = None,
    crs: str | None = None,
    plan_fingerprint: str | None = None,
    result_hash: str | None = None,
) -> None:
    """Write the run manifest JSON sidecar at the output prefix root.

    ``vintage`` records the measured LiDAR flight window, the declared epoch of
    the inventory being enriched, and the gap between them — the one property of
    a run that the output rows cannot express but that decides whether their
    panel geometry means anything. See :func:`pv_geom.pipeline.runner._probe_lidar_vintage`.
    """
    manifest = {
        "pkg_version": __version__,
        "schema_version": SCHEMA_VERSION,
        # Azimuth columns are measured from true north (grid azimuth plus the
        # meridian convergence in grid_convergence_deg). Outputs written before
        # schema 0.4 have no such key: theirs are relative to grid north.
        "azimuth_reference": TRUE_NORTH,
        "config_hash": config_hash,
        "result_hash": result_hash,
        "plan_fingerprint": plan_fingerprint,
        "config": config_dict,
        "inputs": inputs,
        "crs": crs,
        "vintage": vintage or {},
        "cluster_spec": cluster_spec,
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "counts": counts,
        "aggregate_stats": aggregate_stats,
        "tiles_touched": tiles_touched,
        "run_id": run_id,
    }
    payload = json.dumps(manifest, indent=2, default=str)
    if str(output_path).startswith("s3://"):
        import fsspec

        with fsspec.open(str(output_path), "w", encoding="utf-8") as f:
            f.write(payload)
    else:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
