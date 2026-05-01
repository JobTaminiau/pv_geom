"""Building footprint reader + reprojector. M2."""

from __future__ import annotations

from pathlib import Path


def read_footprints(uri: str | Path, target_crs: str, *, auto_id: bool = True):
    """Read footprints (GeoParquet/GPKG/SHP), reproject, optionally synthesize ``building_id``."""
    raise NotImplementedError("M2")
