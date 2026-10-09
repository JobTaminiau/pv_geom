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
PANEL_BY_INSTALL_DATE = "panel_by_install_date"
SURFACE_BEFORE_INSTALL = "surface_before_install"
FREE_STANDING = "free_standing"
SURFACE_UNRESOLVED = "surface_unresolved"
UNSCREENED = "unscreened"
NO_FIT = "no_fit"
NOT_MEASURED = "not_measured"

GEOMETRY_BASIS: tuple[str, ...] = (
    PANEL_CONFIRMED,
    PANEL_BY_INSTALL_DATE,
    PANEL_BY_VINTAGE,
    FREE_STANDING,
    SURFACE_UNRESOLVED,
    SURFACE_BEFORE_INSTALL,
    UNSCREENED,
    NO_FIT,
    NOT_MEASURED,
)

# Bases under which the fitted plane is known to be the panel itself.
PANEL_BASES: frozenset[str] = frozenset(
    {PANEL_CONFIRMED, PANEL_BY_INSTALL_DATE, PANEL_BY_VINTAGE})
# Bases under which the fitted plane is the array's geometry: the panel itself,
# or the free-standing structure that carries it.
ARRAY_BASES: frozenset[str] = PANEL_BASES | {FREE_STANDING}

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
    PANEL_BY_INSTALL_DATE: (
        "Not separable from the roof by height, or no roof reference, but the "
        "layer says this installation was complete on or before the LiDAR date "
        "(the installed_by column), so it was there when the LiDAR was flown."
    ),
    SURFACE_BEFORE_INSTALL: (
        "The layer says this installation did not exist until after the LiDAR "
        "was flown (the not_installed_before column), and the LiDAR shows "
        "nothing standing above the roof: the fitted plane is the roof before "
        "the array. Tilt and azimuth are valid for the array only if it was "
        "then mounted flush."
    ),
    FREE_STANDING: (
        "The fitted plane is an elevated structure standing in open ground, not on "
        "a building: a ground mount or a canopy. It was there when the LiDAR was "
        "flown. There is no roof beneath it to test a standoff against, so whether "
        "it already carried modules is not established; since modules lie flush on "
        "such structures, its tilt and azimuth are the array's either way."
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


def installed_at_lidar(installed_by: date | None, not_installed_before: date | None,
                       lidar_date: date | None) -> bool | None:
    """Whether a per-polygon installation record places the array on the roof
    when the LiDAR was flown.

    ``installed_by`` is a date by which the installation is known to have been
    complete (an inspection sign-off); ``not_installed_before`` a date before
    which it is known not to have existed (a first permit). Either may be
    missing, and between the two the record says nothing.
    """
    if lidar_date is None:
        return None
    if installed_by is not None and installed_by <= lidar_date:
        return True
    if not_installed_before is not None and not_installed_before > lidar_date:
        return False
    return None


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
    free_standing: bool = False,
    installed_at_lidar: bool | None = None,
) -> str:
    """Classify what a row's tilt and azimuth actually rest on.

    ``standoff_passed`` is direct physical evidence and wins outright. Failing
    that, dates decide. A per-polygon installation record (``installed_at_lidar``:
    True, False, or None when the layer carries none or it does not settle the
    matter) speaks first, since it is about this array; then the imagery date,
    which is about the whole layer. Otherwise the row is only as good as the
    screen could make it.
    """
    if not fit_ok:
        return NO_FIT
    if standoff_passed:
        return PANEL_CONFIRMED
    if installed_at_lidar is True:
        return PANEL_BY_INSTALL_DATE
    if installed_at_lidar is False:
        return SURFACE_BEFORE_INSTALL
    if gap_days is not None and gap_days <= 0:
        return PANEL_BY_VINTAGE
    if free_standing:
        return FREE_STANDING
    if standoff_screened:
        return SURFACE_UNRESOLVED
    return UNSCREENED
