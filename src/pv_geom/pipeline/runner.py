"""Top-level orchestrator: builds the Dask graph, writes partitions + manifest. M6."""

from __future__ import annotations

from pathlib import Path

from pv_geom.config import PVGeomConfig


def run_pipeline(
    *,
    polygons_uri: str,
    lidar_prefix: str,
    tile_index_uri: str,
    footprints_uri: str,
    output_uri: str,
    cfg: PVGeomConfig,
    max_polygons: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    dry_run: bool = False,
    resume: bool = False,
) -> Path:
    """End-to-end pipeline. Returns the manifest path."""
    raise NotImplementedError("M6")
