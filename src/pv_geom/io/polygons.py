"""PV polygon reader + reprojector. M2."""

from __future__ import annotations

from pathlib import Path


def read_polygons(uri: str | Path, target_crs: str):
    """Read a GeoParquet of PV polygons and reproject to ``target_crs``."""
    raise NotImplementedError("M2")
