"""Run manifest writer. Config hash lives on PVGeomConfig.hash()."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pv_geom import __version__


def write_manifest(
    output_path: str | Path,
    *,
    config_dict: dict[str, Any],
    config_hash: str,
    inputs: dict[str, str],
    cluster_spec: dict[str, Any],
    counts: dict[str, int],
    aggregate_stats: dict[str, Any],
    tiles_touched: list[str],
    run_id: str,
    vintage: dict[str, Any] | None = None,
    crs: str | None = None,
) -> None:
    """Write the run manifest JSON sidecar at the output prefix root.

    ``vintage`` records the measured LiDAR flight window, the declared epoch of
    the inventory being enriched, and the gap between them — the one property of
    a run that the output rows cannot express but that decides whether their
    panel geometry means anything. See :func:`pv_geom.pipeline.runner._probe_lidar_vintage`.
    """
    manifest = {
        "pkg_version": __version__,
        "config_hash": config_hash,
        "config": config_dict,
        "inputs": inputs,
        "crs": crs,
        "vintage": vintage or {},
        "cluster_spec": cluster_spec,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
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
