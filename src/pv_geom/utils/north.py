"""Grid north versus true north.

A plane's azimuth is first obtained in the projected coordinates the LiDAR is
in, where "north" is the grid's +y axis. Grid north is not true north: the two
differ by the **meridian convergence**, which depends on where in the
projection you are — zero on a Transverse Mercator zone's central meridian,
growing toward its edges (about ±1.7° at a UTM zone edge at 33° N), and several
degrees in some State Plane zones. Left uncorrected it is a systematic error
that changes from one study area to the next.

Convention used throughout pv-geom::

    true_azimuth = grid_azimuth + convergence

with ``convergence`` the true-north bearing of the grid's +y axis, clockwise
positive, as returned by PROJ (``Proj.get_factors(...).meridian_convergence``).
Reported azimuths are true-north; ``grid_convergence_deg`` carries the value
applied so the grid azimuth can be recovered.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pyproj import Proj, Transformer

from pv_geom.utils.crs import horizontal

TRUE_NORTH = "true_north"
GRID_NORTH = "grid_north"


def meridian_convergence_deg(crs: Any, x: Any, y: Any) -> np.ndarray:
    """Meridian convergence, in degrees, at projected coordinates ``(x, y)``.

    The true-north bearing of grid north at each point: add it to a grid
    azimuth to get the true azimuth. ``crs`` is the projected CRS the
    coordinates are in (a compound CRS is reduced to its horizontal part).
    """
    proj_crs = horizontal(crs)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if len(x) == 0:
        return np.zeros(0)
    to_geographic = Transformer.from_crs(proj_crs, proj_crs.geodetic_crs, always_xy=True)
    lon, lat = to_geographic.transform(x, y)
    return np.asarray(Proj(proj_crs).get_factors(lon, lat).meridian_convergence, dtype=float)


def to_true_azimuth(grid_azimuth_deg: float | None, convergence_deg: float) -> float | None:
    """A grid azimuth as a true-north azimuth in [0, 360). None and NaN pass through."""
    if grid_azimuth_deg is None or grid_azimuth_deg != grid_azimuth_deg:
        return grid_azimuth_deg
    return float((grid_azimuth_deg + convergence_deg) % 360.0)
