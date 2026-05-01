"""Height-above-ground / height-above-roof helpers. M4 (PRD §7.4)."""

from __future__ import annotations

import numpy as np

from pv_geom.geometry.plane_fit import PlaneFit


def height_above_ground(
    panel_inlier_z: np.ndarray,
    points_class2: np.ndarray,
    polygon_centroid_xy: np.ndarray,
    search_radius_m: float = 10.0,
) -> float:
    raise NotImplementedError("M4")


def height_above_roof(
    panel_inlier_xyz: np.ndarray,
    roof_plane: PlaneFit,
) -> float:
    raise NotImplementedError("M4")
