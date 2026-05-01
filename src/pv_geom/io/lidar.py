"""LAZ tile reader (laspy primary, PDAL extra) + S3 fetch with per-worker cache. M2."""

from __future__ import annotations

import numpy as np


def read_tile_points(
    tile_uri: str,
    *,
    classes: tuple[int, ...] = (2, 6),
    reader: str = "laspy",
) -> np.ndarray:
    """Read points from a LAZ tile filtered to ASPRS ``classes``. Returns (N, 4) [x, y, z, class]."""
    raise NotImplementedError("M2")


def clip_points_to_polygon(points: np.ndarray, polygon, erosion_m: float = 0.0) -> np.ndarray:
    """Clip a (N, >=3) point array to a (possibly eroded) polygon."""
    raise NotImplementedError("M2")
