"""Build a tile index from the tiles themselves.

A LiDAR collection usually ships with an index layer, but not always, and a
folder of clipped tiles never does. Each LAZ header already states the tile's
extent and CRS, so the index can be read off the files: headers only, no point
data, in parallel.
"""

from __future__ import annotations

import contextlib
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import geopandas as gpd
from shapely.geometry import box

from pv_geom.errors import TileIndexError
from pv_geom.io.storage import is_remote, list_s3_uris
from pv_geom.utils.crs import crs_label

log = logging.getLogger(__name__)

_SUFFIXES = (".laz", ".las")


def list_tiles(prefix: str | Path) -> list[str]:
    """Every LAZ/LAS file directly under a local folder or an ``s3://`` prefix."""
    s = str(prefix)
    if s.startswith("s3://"):
        root = s.rstrip("/") + "/"
        return sorted(u for u in list_s3_uris(s)
                      if u.lower().endswith(_SUFFIXES) and "/" not in u[len(root):])
    if is_remote(s):
        raise TileIndexError(
            f"cannot list tiles under {s}",
            "pass a tile index with --tile-index, or use a local folder or s3:// prefix",
        )
    folder = Path(s)
    return sorted(str(p) for p in folder.iterdir() if p.suffix.lower() in _SUFFIXES) \
        if folder.is_dir() else []


def _header_record(uri: str) -> dict:
    import laspy

    def _read(src) -> dict:
        h = src.header
        crs = None
        with contextlib.suppress(Exception):
            crs = h.parse_crs()
        return {
            "Name": PurePosixPath(uri.replace("\\", "/")).stem,
            "tile_path": uri,
            "crs": crs_label(crs) if crs is not None else None,
            "geometry": box(float(h.mins[0]), float(h.mins[1]),
                            float(h.maxs[0]), float(h.maxs[1])),
        }

    if is_remote(uri):
        import fsspec

        # Only the header bytes cross the network.
        with fsspec.open(uri, "rb") as f, laspy.open(f) as src:
            return _read(src)
    with laspy.open(uri) as src:
        return _read(src)


def scan_tile_index(prefix: str | Path, max_workers: int = 16) -> gpd.GeoDataFrame:
    """A tile index (``Name``, ``tile_path``, ``geometry``) read from the headers
    of every tile under ``prefix``, in the tiles' own CRS."""
    tiles = list_tiles(prefix)
    if not tiles:
        raise TileIndexError(
            f"no .laz or .las tiles found under {prefix}",
            "check --lidar-prefix, or pass a tile index with --tile-index",
        )
    log.info("no tile index given: reading %d tile headers under %s", len(tiles), prefix)
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tiles))) as pool:
        records = list(pool.map(_header_record, tiles))

    declared = {r["crs"] for r in records if r["crs"]}
    if not declared:
        raise TileIndexError(
            f"the tiles under {prefix} declare no CRS, so they cannot be indexed",
            "pass a tile index with --tile-index and set crs.target (--crs)",
        )
    if len(declared) > 1:
        raise TileIndexError(
            f"the tiles under {prefix} are in more than one CRS: {sorted(declared)}",
            "index them yourself per CRS and pass --tile-index, or reproject the tiles",
        )
    crs = declared.pop()
    return gpd.GeoDataFrame(
        {"Name": [r["Name"] for r in records], "tile_path": [r["tile_path"] for r in records]},
        geometry=[r["geometry"] for r in records], crs=crs,
    )
