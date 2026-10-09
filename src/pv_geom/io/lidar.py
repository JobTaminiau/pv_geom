"""LAZ tile reader (laspy primary) + S3 cache + polygon clipping. M2.

PDAL is supported as an optional extra (`pip install pv-geom[pdal]`) but the
default code path uses laspy + lazrs which is portable across Windows / Linux
without system libraries. The PRD calls PDAL the primary; in practice laspy
is sufficient at our scales (tiles ~150 MB, ~25M points, decoded fully in RAM
on a 4 GB worker).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from pv_geom.io.storage import is_remote, localize


def read_tile_points(
    tile_uri: str | Path,
    *,
    classes: tuple[int, ...] | None = None,
    reader: str = "laspy",
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, str | None]:
    """Read all points from a LAZ tile.

    Returns ``((N, 4) [x, y, z, classification], crs_wkt | None)``. If
    ``classes`` is given, the array is filtered to those ASPRS classes.
    Caches remote (``s3://``) tiles to local disk; local paths are read in
    place.
    """
    data = read_tile(tile_uri, classes=classes, reader=reader, cache_dir=cache_dir)
    return data.points, data.crs


def read_tile(
    tile_uri: str | Path,
    *,
    classes: tuple[int, ...] | None = None,
    reader: str = "laspy",
    cache_dir: Path | None = None,
) -> TileData:
    """Read a LAZ tile: points, CRS, and the dates it was flown.

    The flight dates come from the GPS time of *every* point (the tile is being
    decoded anyway), so they are exact for the tile rather than a sample.
    """
    s = str(tile_uri)
    local = localize(s, cache_dir) if is_remote(s) else Path(s)

    if reader == "pdal":
        try:
            pts, crs = _read_via_pdal(local, classes)
        except ImportError:
            pass                      # PDAL not installed; fall through to laspy
        else:
            # PDAL path does not carry GPS time; take the dates from the header.
            return TileData(points=pts, crs=crs, vintage=read_tile_vintage(local))

    pts, crs, vintage = _read_via_laspy(local, classes, tile_uri=s)
    return TileData(points=pts, crs=crs, vintage=vintage)


class TileStream:
    """A LAZ tile decoded a chunk at a time.

    Iterating yields ``(N, 4)`` ``[x, y, z, classification]`` blocks of at most
    ``chunk_points`` returns, so a tile of any size passes through a bounded
    amount of memory. ``vintage`` is complete once iteration has finished: the
    flight dates are taken from the GPS time of every return on the way past.
    """

    def __init__(self, tile_uri: str | Path, *, chunk_points: int = 2_000_000,
                 cache_dir: Path | None = None) -> None:
        import laspy

        self.uri = str(tile_uri)
        local = localize(self.uri, cache_dir) if is_remote(self.uri) else Path(self.uri)
        if not local.exists():
            raise FileNotFoundError(self.uri)
        self._chunk_points = int(chunk_points)
        self._reader = laspy.open(str(local))
        header = self._reader.header
        try:
            crs = header.parse_crs()
            self.crs: str | None = crs.to_string() if crs else None
        except Exception:
            self.crs = None
        self._has_gps = "gps_time" in header.point_format.dimension_names
        self._start: date | None = None
        self._end: date | None = None
        self.n_points = int(header.point_count)

    def __iter__(self) -> Iterator[np.ndarray]:
        header = self._reader.header
        try:
            for chunk in self._reader.chunk_iterator(self._chunk_points):
                if not len(chunk):
                    continue
                if self._has_gps:
                    start, end = _gps_range_to_dates(np.asarray(chunk.gps_time), header)
                    if start is not None and (self._start is None or start < self._start):
                        self._start = start
                    if end is not None and (self._end is None or end > self._end):
                        self._end = end
                yield np.column_stack([
                    np.asarray(chunk.x, dtype=np.float64),
                    np.asarray(chunk.y, dtype=np.float64),
                    np.asarray(chunk.z, dtype=np.float64),
                    np.asarray(chunk.classification, dtype=np.int16),
                ])
        finally:
            self._reader.close()

    @property
    def vintage(self) -> TileVintage:
        return TileVintage(tile_uri=self.uri,
                           creation_date=self._reader.header.creation_date,
                           flight_start=self._start, flight_end=self._end)


def _read_via_laspy(
    local: Path, classes: tuple[int, ...] | None, *, tile_uri: str | None = None
) -> tuple[np.ndarray, str | None, TileVintage]:
    import laspy

    with laspy.open(str(local)) as src:
        header = src.header
        try:
            crs = header.parse_crs()
            crs_str = crs.to_string() if crs else None
        except Exception:
            crs_str = None
        las = src.read()

    start = end = None
    if "gps_time" in header.point_format.dimension_names and len(las.points):
        start, end = _gps_range_to_dates(np.asarray(las.gps_time), header)
    vintage = TileVintage(
        tile_uri=tile_uri or str(local),
        creation_date=header.creation_date,
        flight_start=start,
        flight_end=end,
    )

    pts = np.column_stack(
        [
            np.asarray(las.x, dtype=np.float64),
            np.asarray(las.y, dtype=np.float64),
            np.asarray(las.z, dtype=np.float64),
            np.asarray(las.classification, dtype=np.int16),
        ]
    )
    if classes is not None:
        mask = np.isin(pts[:, 3].astype(int), list(classes))
        pts = pts[mask]
    return pts, crs_str, vintage


def _read_via_pdal(
    local: Path, classes: tuple[int, ...] | None
) -> tuple[np.ndarray, str | None]:
    """Optional PDAL backend. Requires the ``[pdal]`` extra."""
    import json

    import pdal

    pipeline_spec: list[dict] = [{"type": "readers.las", "filename": str(local)}]
    if classes is not None:
        clist = ",".join(str(c) for c in classes)
        pipeline_spec.append(
            {"type": "filters.range", "limits": f"Classification[{clist}:{clist}]"}
        )
    pipeline = pdal.Pipeline(json.dumps(pipeline_spec))
    pipeline.execute()
    arr = pipeline.arrays[0]
    pts = np.column_stack(
        [
            arr["X"].astype(np.float64),
            arr["Y"].astype(np.float64),
            arr["Z"].astype(np.float64),
            arr["Classification"].astype(np.int16),
        ]
    )
    # CRS extraction from PDAL metadata
    try:
        meta = json.loads(pipeline.metadata)
        crs_str = meta["metadata"]["readers.las"]["comp_spatialreference"]
    except Exception:
        crs_str = None
    return pts, crs_str


@dataclass(frozen=True)
class TileVintage:
    """When a LiDAR tile was actually flown.

    ``flight_start`` / ``flight_end`` come from per-point GPS time and are the
    authoritative answer. ``creation_date`` is the LAS header field, which is
    the *delivery* date and can lag the flight badly — the Phoenix
    ``MaricopaPinal_2020`` tiles were flown 2020-11-26/28 but stamped
    2021-06-30, a seven-month overstatement. Prefer the GPS dates and fall back
    to ``creation_date`` only when the point format carries no GPS time.
    """

    tile_uri: str
    creation_date: date | None
    flight_start: date | None
    flight_end: date | None
    # From the same leading sample of points, when the tile was probed rather
    # than read in full: the ASPRS classes seen, and the tile's horizontal CRS.
    classes_seen: tuple[int, ...] = ()
    crs: str | None = None

    @property
    def best_estimate(self) -> date | None:
        """Latest date at which this tile could have observed the ground."""
        return self.flight_end or self.creation_date

    def to_dict(self) -> dict[str, str | None]:
        return {
            "tile_uri": self.tile_uri,
            "creation_date": self.creation_date.isoformat() if self.creation_date else None,
            "flight_start": self.flight_start.isoformat() if self.flight_start else None,
            "flight_end": self.flight_end.isoformat() if self.flight_end else None,
        }


@dataclass(frozen=True)
class TileData:
    """One decoded tile: ``(N, 4)`` ``[x, y, z, classification]``, its CRS
    string (or None), and when it was flown."""

    points: np.ndarray
    crs: str | None
    vintage: TileVintage


# GPS week zero. LAS "Adjusted Standard GPS Time" is GPS seconds minus 1e9;
# GPS time does not observe leap seconds, so converting to UTC subtracts the
# offset in force. 18 s has held since 2017-01-01 and covers every 3DEP
# collection to date; a stale value shifts the result by seconds, not days.
_GPS_EPOCH = datetime(1980, 1, 6)
_GPS_UTC_LEAP_SECONDS = 18
_LAS_GPS_TIME_OFFSET = 1_000_000_000


def _adjusted_gps_to_date(adjusted: float) -> date:
    delta = timedelta(seconds=float(adjusted) + _LAS_GPS_TIME_OFFSET - _GPS_UTC_LEAP_SECONDS)
    return (_GPS_EPOCH + delta).date()


_SECONDS_PER_GPS_WEEK = 604_800


def _gps_range_to_dates(gps: np.ndarray, header) -> tuple[date | None, date | None]:
    """First and last flight date in a block of LAS GPS times, or (None, None).

    LAS stores either *adjusted standard* GPS time (convertible to a date) or
    *GPS week* time (seconds into an unnamed week — not convertible). The header
    bit says which, but is mis-set often enough that a value larger than a week
    is also accepted as standard time.
    """
    gps = np.asarray(gps, dtype=np.float64)
    gps = gps[np.isfinite(gps) & (gps != 0.0)]
    if not gps.size:
        return None, None
    lo, hi = float(gps.min()), float(gps.max())
    try:
        is_standard = int(header.global_encoding.gps_time_type) == 1
    except Exception:
        is_standard = False
    if not is_standard and max(abs(lo), abs(hi)) <= _SECONDS_PER_GPS_WEEK:
        return None, None
    return _adjusted_gps_to_date(lo), _adjusted_gps_to_date(hi)


def _vintage_from_open(src, tile_uri: str) -> TileVintage:
    header = src.header
    start = end = None
    classes: tuple[int, ...] = ()
    if header.point_count:
        try:
            chunk = next(src.chunk_iterator(min(1_000_000, header.point_count)))
        except StopIteration:
            chunk = None
        if chunk is not None:
            classes = tuple(int(c) for c in np.unique(np.asarray(chunk.classification)))
            if "gps_time" in header.point_format.dimension_names:
                start, end = _gps_range_to_dates(np.asarray(chunk.gps_time), header)
    crs = None
    with contextlib.suppress(Exception):
        parsed = header.parse_crs()
        if parsed is not None:
            from pv_geom.utils.crs import crs_label

            crs = crs_label(parsed)
    return TileVintage(
        tile_uri=tile_uri,
        creation_date=header.creation_date,
        flight_start=start,
        flight_end=end,
        classes_seen=classes,
        crs=crs,
    )


def read_tile_vintage(tile_uri: str | Path, cache_dir: Path | None = None) -> TileVintage:
    """Read acquisition dates from a LAZ tile without decoding all of it.

    Reads the header plus a single point chunk — enough for the GPS-time range
    of a contiguous flight strip, which is what a 3DEP tile is. A remote tile is
    streamed through fsspec so only those leading bytes cross the network; if
    the object store won't support the ranged reads laspy wants, we fall back to
    the same local cache :func:`read_tile_points` uses.
    """
    import laspy

    s = str(tile_uri)

    if is_remote(s):
        try:
            import fsspec

            with fsspec.open(s, "rb") as f, laspy.open(f) as src:
                return _vintage_from_open(src, s)
        except Exception:
            pass  # fall through to the full download

    local = localize(s, cache_dir) if is_remote(s) else Path(s)
    with laspy.open(str(local)) as src:
        return _vintage_from_open(src, s)


def inspect_tile(tile_uri: str | Path, cache_dir: Path | None = None) -> dict:
    """Summarise one LAZ tile: what a new LiDAR source has to be checked for
    before a run can be trusted (classes present, density, CRS, units, dates)."""
    import laspy

    s = str(tile_uri)
    local = localize(s, cache_dir) if is_remote(s) else Path(s)
    with laspy.open(str(local)) as src:
        header = src.header
        try:
            crs = header.parse_crs()
        except Exception:
            crs = None
        las = src.read()

    cls, counts = np.unique(np.asarray(las.classification), return_counts=True)
    n = int(header.point_count)
    dx = float(header.maxs[0] - header.mins[0])
    dy = float(header.maxs[1] - header.mins[1])
    start = end = None
    if "gps_time" in header.point_format.dimension_names and n:
        start, end = _gps_range_to_dates(np.asarray(las.gps_time), header)

    horiz = units = None
    to_metre = None                       # None: the extent cannot be put in metres
    if crs is not None:
        h = crs.sub_crs_list[0] if crs.is_compound else crs
        horiz = h.to_string() if h.to_epsg() is None else f"EPSG:{h.to_epsg()}"
        units = h.axis_info[0].unit_name if h.axis_info else None
        if h.is_projected and h.axis_info:
            to_metre = float(h.axis_info[0].unit_conversion_factor)
    # Extent and density are reported in metres whatever the tile is delivered
    # in: a State Plane tile in feet would otherwise look a tenth as dense.
    extent_m = [dx * to_metre, dy * to_metre] if to_metre is not None else None
    area_m2 = extent_m[0] * extent_m[1] if extent_m else 0.0
    return {
        "tile": s,
        "las_version": str(header.version),
        "point_format": int(header.point_format.id),
        "n_points": n,
        "extent_native": [dx, dy],
        "extent_m": extent_m,
        "density_pts_per_m2": n / area_m2 if area_m2 > 0 else None,
        "z_range": [float(header.mins[2]), float(header.maxs[2])],
        "classes": {int(c): int(k) for c, k in zip(cls, counts, strict=True)},
        "has_building_class_6": bool((cls == 6).any()),
        "has_ground_class_2": bool((cls == 2).any()),
        "horizontal_crs": horiz,
        "horizontal_units": units,
        "flight_start": start.isoformat() if start else None,
        "flight_end": end.isoformat() if end else None,
        "header_creation_date": (
            header.creation_date.isoformat() if header.creation_date else None
        ),
    }


def clip_points_to_polygon(
    pts: np.ndarray,
    polygon,                              # shapely Polygon or MultiPolygon
    erosion_m: float = 0.0,
) -> np.ndarray:
    """Clip ``pts`` (N, ≥2) to a (possibly eroded) polygon. Returns the kept rows."""
    from shapely import contains_xy

    if pts.size == 0:
        return pts

    if erosion_m > 0:
        polygon = polygon.buffer(-erosion_m)
        if polygon.is_empty:
            return pts[:0]

    mask = contains_xy(polygon, pts[:, 0], pts[:, 1])
    return pts[mask]
