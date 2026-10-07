"""A deterministic synthetic study area with known geometry.

Fifteen arrays chosen to walk every branch the pipeline has: flush and racked
arrays on pitched and flat roofs, clean and noisy returns, a bare roof under a
polygon (the vintage case), ground mounts, an array with too few returns and a
polygon over nothing. The terrain slopes, so "above local ground" is exercised.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
from shapely.geometry import box

CRS = "EPSG:6341"
FLOWN = date(2022, 6, 15)
POLYGON_VINTAGE = "2023"
GROUND_SLOPE = 0.02                      # 2% grade along x


@dataclass(frozen=True)
class Array:
    name: str
    x: float                 # building / array origin (south-west corner)
    y: float
    roof_tilt: float         # None-equivalent: use kind="ground"
    roof_az: float
    panel_tilt: float
    panel_az: float
    standoff: float          # panel height above the roof plane (m)
    noise: float             # 1-sigma vertical scatter of the returns (m)
    kind: str = "roof"       # roof | ground | sparse | empty | bare_roof


ARRAYS: tuple[Array, ...] = (
    Array("flush_south_20", 20, 20, 20, 180, 20, 180, 0.10, 0.010),
    Array("flush_west_25", 60, 20, 25, 270, 25, 270, 0.12, 0.010),
    Array("flush_east_18", 100, 20, 18, 90, 18, 90, 0.08, 0.015),
    Array("flush_north_30", 140, 20, 30, 0, 30, 0, 0.10, 0.010),
    Array("flush_steep_40", 180, 20, 40, 135, 40, 135, 0.10, 0.010),
    Array("rack_on_flat_15", 220, 20, 2, 180, 15, 180, 0.60, 0.010),
    Array("rack_on_flat_30", 20, 80, 2, 200, 30, 200, 0.80, 0.010),
    Array("bare_roof_22", 60, 80, 22, 180, 22, 180, 0.00, 0.010, "bare_roof"),
    Array("bare_flat_roof", 100, 80, 2, 90, 2, 90, 0.00, 0.010, "bare_roof"),
    Array("noisy_flush_28", 140, 80, 28, 225, 28, 225, 0.10, 0.070),
    Array("noisy_bare_32", 180, 80, 32, 160, 32, 160, 0.00, 0.070, "bare_roof"),
    Array("ground_mount_25", 220, 80, 0, 0, 25, 180, 1.20, 0.010, "ground"),
    Array("ground_mount_10", 20, 140, 0, 0, 10, 190, 1.00, 0.010, "ground"),
    Array("sparse_array", 60, 140, 20, 180, 20, 180, 0.10, 0.010, "sparse"),
    Array("nothing_there", 100, 140, 0, 0, 0, 0, 0.00, 0.010, "empty"),
)

BUILDING = (24.0, 16.0)      # roof extent
PANEL = (6.0, 4.0, 10.0, 4.0)  # offset x, offset y, width, depth within the roof


def _ground_z(x: np.ndarray) -> np.ndarray:
    return 100.0 + GROUND_SLOPE * x


def _plane(xy: np.ndarray, tilt: float, az: float, z0: float, centre: tuple[float, float]) -> np.ndarray:
    """z of a plane with the given tilt and facing azimuth, equal to z0 at centre."""
    t, a = np.radians(tilt), np.radians(az)
    nx, ny, nz = np.sin(t) * np.sin(a), np.sin(t) * np.cos(a), np.cos(t)
    return z0 - (nx * (xy[:, 0] - centre[0]) + ny * (xy[:, 1] - centre[1])) / nz


def _scene(rng: np.random.Generator) -> tuple[np.ndarray, list]:
    pts: list[np.ndarray] = []
    polygons = []
    roofs = []

    for a in ARRAYS:
        px0, py0 = a.x + PANEL[0], a.y + PANEL[1]
        panel_box = box(px0, py0, px0 + PANEL[2], py0 + PANEL[3])
        polygons.append(panel_box)
        centre = (a.x + BUILDING[0] / 2, a.y + BUILDING[1] / 2)
        base = float(_ground_z(np.array([centre[0]]))[0])
        if a.kind == "empty":
            continue

        if a.kind == "ground":
            n = 500
            xy = rng.uniform([px0, py0], [px0 + PANEL[2], py0 + PANEL[3]], size=(n, 2))
            z = _plane(xy, a.panel_tilt, a.panel_az, base + a.standoff + 0.8, panel_box.centroid.coords[0])
            pts.append(np.column_stack([xy, z + rng.normal(0, a.noise, n), np.full(n, 1)]))
            continue

        roofs.append(box(a.x, a.y, a.x + BUILDING[0], a.y + BUILDING[1]))
        n_roof = int(BUILDING[0] * BUILDING[1] * 10)
        rxy = rng.uniform([a.x, a.y], [a.x + BUILDING[0], a.y + BUILDING[1]], size=(n_roof, 2))
        in_panel = ((rxy[:, 0] >= px0) & (rxy[:, 0] <= px0 + PANEL[2])
                    & (rxy[:, 1] >= py0) & (rxy[:, 1] <= py0 + PANEL[3]))
        roof_z = _plane(rxy, a.roof_tilt, a.roof_az, base + 6.0, centre)
        # A bare roof has returns under the polygon too; an array hides the roof beneath it.
        keep = np.ones(n_roof, dtype=bool) if a.kind == "bare_roof" else ~in_panel
        pts.append(np.column_stack([rxy[keep], roof_z[keep] + rng.normal(0, a.noise, int(keep.sum())),
                                    np.full(int(keep.sum()), 1)]))
        if a.kind == "bare_roof":
            continue

        n = 12 if a.kind == "sparse" else int(PANEL[2] * PANEL[3] * 10)
        xy = rng.uniform([px0, py0], [px0 + PANEL[2], py0 + PANEL[3]], size=(n, 2))
        if abs(a.panel_tilt - a.roof_tilt) < 0.5 and abs(a.panel_az - a.roof_az) < 0.5:
            z = _plane(xy, a.roof_tilt, a.roof_az, base + 6.0, centre) + a.standoff
        else:
            c = panel_box.centroid.coords[0]
            z0 = float(_plane(np.array([c]), a.roof_tilt, a.roof_az, base + 6.0, centre)[0])
            z = _plane(xy, a.panel_tilt, a.panel_az, z0 + a.standoff, c)
        pts.append(np.column_stack([xy, z + rng.normal(0, a.noise, n), np.full(n, 1)]))

    # Ground: everywhere except under a roof (LiDAR does not see through one).
    gxy = rng.uniform([0, 0], [260, 180], size=(260 * 180 * 2, 2))
    from shapely import contains_xy
    from shapely.ops import unary_union

    hidden = contains_xy(unary_union(roofs), gxy[:, 0], gxy[:, 1])
    gxy = gxy[~hidden]
    pts.append(np.column_stack([gxy, _ground_z(gxy[:, 0]) + rng.normal(0, 0.02, len(gxy)),
                                np.full(len(gxy), 2)]))
    return np.concatenate(pts), polygons


def write_scene(out: Path, seed: int = 7) -> Path:
    """Write the benchmark directory layout ``pv_geom.testing.run_benchmark`` reads."""
    out.mkdir(parents=True, exist_ok=True)
    pts, polygons = _scene(np.random.default_rng(seed))

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([0.0, 0.0, 0.0])
    las = laspy.LasData(header)
    las.x, las.y, las.z = pts[:, 0], pts[:, 1], pts[:, 2]
    las.classification = pts[:, 3].astype(np.uint8)
    from pv_geom.io.lidar import _GPS_EPOCH, _GPS_UTC_LEAP_SECONDS, _LAS_GPS_TIME_OFFSET

    secs = ((datetime.combine(FLOWN, datetime.min.time()) - _GPS_EPOCH).total_seconds()
            + _GPS_UTC_LEAP_SECONDS - _LAS_GPS_TIME_OFFSET)
    las.gps_time = np.full(len(pts), secs + 43_200.0)       # midday, clear of date edges
    las.write(str(out / "bench.laz"))

    gpd.GeoDataFrame({"label": [a.name for a in ARRAYS]}, geometry=polygons, crs=CRS) \
        .to_parquet(out / "polygons.parquet")
    gpd.GeoDataFrame({"Name": ["bench"]}, geometry=[box(0, 0, 260, 180)], crs=CRS) \
        .to_parquet(out / "tile_index.parquet")
    (out / "meta.json").write_text(json.dumps({"polygon_vintage": POLYGON_VINTAGE}),
                                   encoding="utf-8")
    return out
