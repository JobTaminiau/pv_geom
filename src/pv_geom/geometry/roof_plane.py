"""Ring-buffer roof plane extraction. M4 (PRD §7.3)."""

from __future__ import annotations

import numpy as np

from pv_geom.config import RoofPlaneConfig
from pv_geom.geometry.plane_fit import PlaneFit


def extract_roof_plane(
    pv_polygon,                        # shapely.Polygon
    building_footprints,               # gpd.GeoDataFrame (spatially indexed)
    other_pv_polygons,                 # gpd.GeoDataFrame within same building
    points_class6: np.ndarray,
    cfg: RoofPlaneConfig,
) -> PlaneFit | None:
    """Fit the roof plane *around* the PV polygon via a ring buffer.

    Returns None when polygon is not over a building, ring buffer too small, too few
    class-6 points, or RMSE exceeds ``cfg.rmse_max_m``.
    """
    raise NotImplementedError("M4")
