"""CRS helpers: resolving the run CRS and refusing non-metric ones."""

from __future__ import annotations

from pyproj import CRS

from pv_geom.errors import CRSResolutionError, NonMetricCRSError

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
            assert_metric_projected(c)
            return crs_label(c)
    raise CRSResolutionError(
        "crs.target is 'auto' but neither the tile index nor a LAZ header declares a "
        "projected CRS",
        "set crs.target explicitly (or pass --crs EPSG:xxxx)",
    )
