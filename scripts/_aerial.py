"""Maricopa County 2024 aerial-basemap fetcher for spike visualizations.

Source: phoenix-pv-atlas/lib/constants.ts
  https://gis.maricopa.gov/imagery/rest/services/Aerial2024Sep2024OctOrtho/MapServer

ArcGIS MapServer LOD encoding: MC LOD 0 == Web Mercator zoom 8; MC LOD 13
(zoom 21, ~6 cm/px in Phoenix) is the max. Tiles are 256x256 JPEG, served
as `tile/{lod}/{row}/{col}` (row before col). CORS is open.
"""

from __future__ import annotations

import io
import math
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

TILE_URL = (
    "https://gis.maricopa.gov/imagery/rest/services/"
    "Aerial2024Sep2024OctOrtho/MapServer/tile/{lod}/{row}/{col}"
)
LOD_OFFSET = 8        # MC LOD == WM zoom - 8
MAX_LOD = 13          # MC LOD 13 == WM zoom 21
TILE_PX = 256
CACHE = Path.home() / ".cache" / "pv_geom_aerial"


def lonlat_to_tile_xy(lon: float, lat: float, z: int) -> tuple[float, float]:
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tile_to_lonlat(x: float, y: float, z: int) -> tuple[float, float]:
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    return lon, math.degrees(lat_rad)


def _fetch_tile(lod: int, row: int, col: int, max_retries: int = 4) -> Image.Image | None:
    """Fetch one aerial tile with disk cache + retry. Returns None on failure.

    ArcGIS MapServer tiles can be built on-demand: a fresh request may 404
    while the server builds the tile, then succeed on a follow-up call.
    Retry with linear backoff to absorb that.
    """
    import time

    cache_path = CACHE / f"{lod}" / f"{row}" / f"{col}.jpg"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            cache_path.unlink(missing_ok=True)

    url = TILE_URL.format(lod=lod, row=row, col=col)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pv-geom-spike/0.1"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            cache_path.write_bytes(data)
            return Image.open(io.BytesIO(data)).convert("RGB")
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 404 and attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))   # 0.5s, 1s, 1.5s
                continue
            break
        except Exception as exc:
            last_exc = exc
            break
    print(f"  aerial tile {lod}/{row}/{col} failed after {attempt + 1} tries: {last_exc}")
    return None


def aerial_basemap(
    bbox_wgs84: tuple[float, float, float, float],
    target_lod: int = MAX_LOD,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Stitch the aerial covering bbox=(lon_min, lat_min, lon_max, lat_max).

    Returns (image_rgb, extent) where extent=(lon_min, lon_max, lat_min, lat_max)
    is the actual canvas extent (rounded up to whole tiles), ready for
    matplotlib's ``imshow(..., extent=extent)``.
    """
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    z = target_lod + LOD_OFFSET

    x0, y0 = lonlat_to_tile_xy(lon_min, lat_max, z)
    x1, y1 = lonlat_to_tile_xy(lon_max, lat_min, z)
    col_lo, col_hi = int(math.floor(x0)), int(math.floor(x1))
    row_lo, row_hi = int(math.floor(y0)), int(math.floor(y1))

    n_cols = col_hi - col_lo + 1
    n_rows = row_hi - row_lo + 1
    canvas = Image.new("RGB", (n_cols * TILE_PX, n_rows * TILE_PX), color=(60, 60, 60))
    for r in range(row_lo, row_hi + 1):
        for c in range(col_lo, col_hi + 1):
            t = _fetch_tile(target_lod, r, c)
            if t is None:
                continue
            canvas.paste(t, ((c - col_lo) * TILE_PX, (r - row_lo) * TILE_PX))

    lon_left, lat_top = tile_to_lonlat(col_lo, row_lo, z)
    lon_right, lat_bottom = tile_to_lonlat(col_hi + 1, row_hi + 1, z)
    extent = (lon_left, lon_right, lat_bottom, lat_top)
    return np.asarray(canvas), extent
