"""CRS helpers: resolving the run CRS, and bringing LiDAR into it.

The run CRS is always projected and metric: every threshold in the
configuration is in metres. LiDAR delivered in another CRS or in feet (most US
State Plane collections) is converted to the run CRS as it is read.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

from pv_geom.errors import CRSResolutionError, NonMetricCRSError

log = logging.getLogger(__name__)

_METRE_NAMES = {"metre", "meter", "m"}


def horizontal(crs) -> CRS:
    """The horizontal component of a (possibly compound) CRS."""
    c = CRS.from_user_input(crs)
    return c.sub_crs_list[0] if c.is_compound else c


def crs_label(crs) -> str:
    """``EPSG:xxxx`` when the CRS has a code, else its WKT."""
    c = horizontal(crs)
    epsg = c.to_epsg()
    return f"EPSG:{epsg}" if epsg is not None else c.to_wkt()


def assert_metric_projected(crs, *, what: str = "run CRS") -> None:
    """Raise unless ``crs`` is projected with metre axes.

    LiDAR points are used in their native coordinates and every threshold in
    the configuration is in metres, so a geographic or foot-based CRS would
    silently produce wrong heights, densities and areas.
    """
    c = horizontal(crs)
    if not c.is_projected:
        raise CRSResolutionError(
            f"{what} {crs_label(c)} is not projected",
            "pv-geom needs the LiDAR's projected, metric CRS; set crs.target (or --crs) "
            "explicitly if detection failed",
        )
    unit = (c.axis_info[0].unit_name or "").lower() if c.axis_info else ""
    if unit not in _METRE_NAMES:
        raise NonMetricCRSError(
            f"{what} {crs_label(c)} uses '{unit}' units; pv-geom supports metric LiDAR only",
            "reproject the tiles to a metric CRS first",
        )


def is_metric(crs) -> bool:
    c = horizontal(crs)
    unit = (c.axis_info[0].unit_name or "").lower() if c.axis_info else ""
    return unit in _METRE_NAMES


def metric_equivalent(crs) -> str:
    """A metric projected CRS for the area a foot-based CRS covers: the UTM
    zone at the centre of its area of use, on the same datum where one exists."""
    c = horizontal(crs)
    aou = c.area_of_use
    if aou is None:
        raise CRSResolutionError(
            f"cannot choose a metric CRS for {crs_label(c)}: it declares no area of use",
            "set crs.target to a metric projected CRS (e.g. the UTM zone) explicitly")
    lon, lat = (aou.west + aou.east) / 2.0, (aou.south + aou.north) / 2.0
    area = AreaOfInterest(lon, lat, lon, lat)
    datum = c.datum.name if c.datum is not None else None
    found = (query_utm_crs_info(datum_name=datum, area_of_interest=area) if datum else []) \
        or query_utm_crs_info(datum_name="WGS 84", area_of_interest=area)
    if not found:
        raise CRSResolutionError(
            f"no UTM zone found for {crs_label(c)}",
            "set crs.target to a metric projected CRS explicitly")
    return f"{found[0].auth_name}:{found[0].code}"


@lru_cache(maxsize=32)
def _tile_transform(tile_crs: str, target_crs: str) -> tuple[Transformer | None, float, str]:
    """How to bring a tile into the run CRS: a horizontal transformer (None when
    the grids are the same), the factor taking its heights to metres, and a
    description for the log and the manifest."""
    full = CRS.from_user_input(tile_crs)
    src, dst = horizontal(full), horizontal(target_crs)
    same_grid = src == dst
    transformer = None if same_grid else Transformer.from_crs(src, dst, always_xy=True)
    if full.is_compound and len(full.sub_crs_list) > 1 and full.sub_crs_list[1].axis_info:
        vertical = full.sub_crs_list[1].axis_info[0]
        z_factor, z_unit, assumed = vertical.unit_conversion_factor, vertical.unit_name, ""
    else:
        # No vertical CRS declared: heights are taken to be in the horizontal unit,
        # which is how foot-based collections are delivered.
        axis = src.axis_info[0]
        z_factor, z_unit = axis.unit_conversion_factor, axis.unit_name
        assumed = " (no vertical CRS declared; assumed same unit as horizontal)"
    what = []
    if not same_grid:
        what.append(f"horizontal {crs_label(src)} -> {crs_label(dst)}")
    if abs(z_factor - 1.0) > 1e-12:
        what.append(f"heights {z_unit} -> metre x{z_factor:.10g}{assumed}")
    return transformer, float(z_factor), "; ".join(what)


def to_run_crs(points: np.ndarray, tile_crs: str | None, target_crs: str
               ) -> tuple[np.ndarray, str | None]:
    """``points`` (N, >=3: x, y, z, ...) in the run CRS, and what was done.

    A tile with no CRS, or already in the run CRS in metres, is returned
    untouched with ``None``.
    """
    if not tile_crs or str(target_crs).lower() == "auto":
        return points, None
    transformer, z_factor, what = _tile_transform(str(tile_crs), str(target_crs))
    if not what:
        return points, None
    out = np.array(points, dtype=np.float64, copy=True)
    if transformer is not None:
        out[:, 0], out[:, 1] = transformer.transform(out[:, 0], out[:, 1])
    if abs(z_factor - 1.0) > 1e-12:
        out[:, 2] *= z_factor
    return out, what


def resolve_target_crs(configured: str, tile_index_crs=None, sample_tile_crs=None) -> str:
    """Resolve ``crs.target``: an explicit value wins; ``auto`` takes the tile
    index's CRS when it is projected, else the CRS of a LAZ header."""
    if str(configured).lower() != "auto":
        assert_metric_projected(configured)
        return str(configured)
    for cand in (tile_index_crs, sample_tile_crs):
        if cand is None:
            continue
        c = horizontal(cand)
        if c.is_projected:
            if is_metric(c):
                return crs_label(c)
            metric = metric_equivalent(c)
            log.info("LiDAR CRS %s is in '%s'; working in %s and converting tiles on read",
                     crs_label(c), c.axis_info[0].unit_name, metric)
            return metric
    raise CRSResolutionError(
        "crs.target is 'auto' but neither the tile index nor a LAZ header declares a "
        "projected CRS",
        "set crs.target explicitly (or pass --crs EPSG:xxxx)",
    )
