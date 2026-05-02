"""LAZ tile reader (laspy primary) + S3 cache + polygon clipping. M2.

PDAL is supported as an optional extra (`pip install pv-geom[pdal]`) but the
default code path uses laspy + lazrs which is portable across Windows / Linux
without system libraries. The PRD calls PDAL the primary; in practice laspy
is sufficient at our scales (tiles ~150 MB, ~25M points, decoded fully in RAM
on a 4 GB worker).
"""

from __future__ import annotations

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
