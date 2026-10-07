"""Input vintages and what they imply for each row's geometry.

The polygon set (from aerial or satellite imagery) and the LiDAR are rarely
captured at the same time. When the imagery is the later of the two, some
polygons mark installations that did not yet exist when the LiDAR was flown, and
the plane fitted inside such a polygon is the bare roof or ground. Nothing in a
single row's fit statistics reveals this — the fit is tight either way — so the
two dates are carried as run inputs and combined here with the per-row
panel-standoff screen into one column, ``geometry_basis``.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime

from pv_geom.errors import VintageFormatError

# The values ``geometry_basis`` can take, most to least trustworthy as a
# measurement of the *panel* surface.
PANEL_CONFIRMED = "panel_confirmed"
PANEL_BY_VINTAGE = "panel_by_vintage"
SURFACE_UNRESOLVED = "surface_unresolved"
UNSCREENED = "unscreened"
NO_FIT = "no_fit"
NOT_MEASURED = "not_measured"

GEOMETRY_BASIS: tuple[str, ...] = (
    PANEL_CONFIRMED,
    PANEL_BY_VINTAGE,
    SURFACE_UNRESOLVED,
    UNSCREENED,
    NO_FIT,
    NOT_MEASURED,
)

# Bases under which the fitted plane is known to be the panel itself.
PANEL_BASES: frozenset[str] = frozenset({PANEL_CONFIRMED, PANEL_BY_VINTAGE})

BASIS_DESCRIPTIONS: dict[str, str] = {
    PANEL_CONFIRMED: (
        "The fitted plane sits resolvably above the surrounding roof plane, so "
        "panels were physically present in the point cloud."
    ),
    PANEL_BY_VINTAGE: (
        "Not separable from the roof by height, or no roof reference, but the "
        "polygon vintage is on or before the LiDAR date, so the installation "
        "existed when the LiDAR was flown."
    ),
    SURFACE_UNRESOLVED: (
        "The polygons postdate the LiDAR (or a date is unknown) and the fitted "
        "plane coincides with the roof plane: either a flush-mounted array or "
        "the bare roof before installation. Tilt and azimuth are valid for the "
        "array only if it is flush-mounted."
    ),
    UNSCREENED: (
        "The polygons postdate the LiDAR (or a date is unknown) and there is no "
        "usable roof reference to test against, so panel presence is unverified."
    ),
    NO_FIT: "No plane could be fitted (too few points or no consensus).",
    NOT_MEASURED: (
        "The polygon never reached measurement: outside the LiDAR, on a tile that "
        "is missing or unreadable, or without a usable geometry. See status."
    ),
}

_YEAR = re.compile(r"^\d{4}$")
_YEAR_MONTH = re.compile(r"^(\d{4})-(\d{1,2})$")


def parse_vintage(value: str | int | date | datetime | None) -> date | None:
    """Parse a declared vintage to the latest day it could refer to.

    Accepts a ``date``, ``"YYYY-MM-DD"``, ``"YYYY-MM"`` or a year. A year or
    month is a window; its last day is returned so that a gap computed against
    the LiDAR date errs toward "the polygons may be newer", never the reverse.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    if not s or s.lower() in {"none", "null", "nan", "nat"}:
        return None
    if _YEAR.match(s):
        return date(int(s), 12, 31)
    m = _YEAR_MONTH.match(s)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        return date(y, mo, calendar.monthrange(y, mo)[1])
    try:
        return date.fromisoformat(s[:10])
    except ValueError as exc:
        raise VintageFormatError(
            f"cannot parse vintage {value!r}", "use YYYY, YYYY-MM or YYYY-MM-DD"
        ) from exc


def vintage_gap_days(polygon_vintage: date | None, lidar_date: date | None) -> int | None:
    """Days by which the polygon vintage postdates the LiDAR (positive = newer
    polygons, i.e. installations may be missing from the point cloud)."""
    if polygon_vintage is None or lidar_date is None:
        return None
    return (polygon_vintage - lidar_date).days


def geometry_basis(
    *,
    fit_ok: bool,
    standoff_passed: bool,
    standoff_screened: bool,
    gap_days: int | None,
) -> str:
    """Classify what a row's tilt and azimuth actually rest on.

    ``standoff_passed`` is direct physical evidence and wins outright. Failing
    that, the declared dates decide: if the polygons are no newer than the
    LiDAR the installation was there to be measured. Otherwise the row is only
    as good as the screen could make it.
    """
    if not fit_ok:
        return NO_FIT
    if standoff_passed:
        return PANEL_CONFIRMED
    if gap_days is not None and gap_days <= 0:
        return PANEL_BY_VINTAGE
    if standoff_screened:
        return SURFACE_UNRESOLVED
    return UNSCREENED
