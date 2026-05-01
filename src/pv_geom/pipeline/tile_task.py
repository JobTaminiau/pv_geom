"""Per-tile-group worker function. M6."""

from __future__ import annotations

import pyarrow as pa

from pv_geom.config import PVGeomConfig


def process_tile_group(
    tile_ids: list[str],
    polygons,           # gpd.GeoDataFrame, already filtered to this group
    footprints,         # gpd.GeoDataFrame, spatially filtered to tile bbox
    cfg: PVGeomConfig,
) -> pa.Table:
    """Fetch LAZ for all `tile_ids`, run the per-polygon pipeline, return a partition table."""
    raise NotImplementedError("M6")
