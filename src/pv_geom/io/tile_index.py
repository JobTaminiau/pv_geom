"""LiDAR tile-index loader (auto-detects GeoParquet / GPKG / SHP). M2."""

from __future__ import annotations

from pathlib import Path


def load_tile_index(uri: str | Path, target_crs: str):
    """Load a tile-index dataset and reproject to ``target_crs``.

    Required column (or alias): ``tile_path`` resolvable to an S3 URI.
    """
    raise NotImplementedError("M2")
