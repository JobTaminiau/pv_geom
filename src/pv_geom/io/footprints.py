"""Building footprint reader + reprojector. M2.

For Arizona work the canonical footprint dataset is FEMA USA Structures, hosted
on the FREE Research Data Commons:

    s3://free-research-data/national/fema_footprints/az.geoparquet

Other states / sources are accepted via the same GeoParquet/GPKG/SHP interface.
FEMA's ``build_id`` column is mapped to canonical ``building_id``; if neither
exists, ids are auto-synthesized.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from pv_geom.io.vector import read_vector, reproject

# Default FEMA AZ footprint URI on the FREE Research Data Commons.
FEMA_AZ_URI = "s3://free-research-data/national/fema_footprints/az.geoparquet"


def read_footprints(
    uri: str | Path,
    target_crs: str,
    *,
    id_col: str = "building_id",
    auto_id: bool = True,
    bbox: tuple[float, float, float, float] | None = None,
) -> gpd.GeoDataFrame:
    """Read building footprints, reproject, normalize the id column.

    ``id_col`` resolution order:
      1. Use the input's ``id_col`` (e.g. ``building_id``) if present.
      2. Else copy from ``build_id`` (FEMA's column name) if present.
      3. Else, when ``auto_id=True``, synthesize ``auto_<i>`` ids; else raise.
    """
    gdf = reproject(read_vector(uri), target_crs)

    if id_col not in gdf.columns:
        if "build_id" in gdf.columns:
            gdf = gdf.assign(**{id_col: gdf["build_id"].astype(str)})
        elif auto_id:
            gdf = gdf.assign(**{id_col: [f"auto_{i}" for i in range(len(gdf))]})
        else:
            raise ValueError(
                f"footprint dataset has neither '{id_col}' nor 'build_id' "
                f"and auto_id=False"
            )
    else:
        gdf[id_col] = gdf[id_col].astype(str)

    if bbox is not None:
        x0, y0, x1, y1 = bbox
        gdf = gdf.cx[x0:x1, y0:y1]

    return gdf.reset_index(drop=True)
