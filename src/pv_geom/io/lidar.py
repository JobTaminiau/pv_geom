"""LAZ tile reader (laspy primary) + S3 cache + polygon clipping. M2.

PDAL is supported as an optional extra (`pip install pv-geom[pdal]`) but the
default code path uses laspy + lazrs which is portable across Windows / Linux
without system libraries. The PRD calls PDAL the primary; in practice laspy
is sufficient at our scales (tiles ~150 MB, ~25M points, decoded fully in RAM
on a 4 GB worker).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from pv_geom.io._localize import is_remote, localize


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
    s = str(tile_uri)
    local = localize(s, cache_dir) if is_remote(s) else Path(s)

    if reader == "pdal":
        try:
            return _read_via_pdal(local, classes)
        except ImportError:
            # PDAL not installed; fall through to laspy.
            pass

    return _read_via_laspy(local, classes)


def _read_via_laspy(
    local: Path, classes: tuple[int, ...] | None
) -> tuple[np.ndarray, str | None]:
    import laspy

    with laspy.open(str(local)) as src:
        try:
            crs = src.header.parse_crs()
            crs_str = crs.to_string() if crs else None
        except Exception:
            crs_str = None
        las = src.read()

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
    return pts, crs_str


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


def _vintage_from_open(src, tile_uri: str) -> TileVintage:
    header = src.header
    start = end = None
    if "gps_time" in header.point_format.dimension_names and header.point_count:
        try:
            chunk = next(src.chunk_iterator(min(1_000_000, header.point_count)))
        except StopIteration:
            chunk = None
        if chunk is not None:
            gps = np.asarray(chunk.gps_time, dtype=np.float64)
            gps = gps[np.isfinite(gps) & (gps != 0.0)]
            if gps.size:
                start = _adjusted_gps_to_date(gps.min())
                end = _adjusted_gps_to_date(gps.max())
    return TileVintage(
        tile_uri=tile_uri,
        creation_date=header.creation_date,
        flight_start=start,
        flight_end=end,
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
