"""Synthetic scenes shared by the unit tests: LAZ tiles, rooftop geometry, configs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
from shapely.geometry import box

from pv_geom.config import PVGeomConfig


def mounting_cfg() -> PVGeomConfig:
    """Config with the archived (experimental) mounting classifier switched on."""
    cfg = PVGeomConfig()
    cfg.mounting_rules.enabled = True
    return cfg

def write_synthetic_laz(
    path: Path,
    *,
    tile_extent: tuple[float, float, float, float] = (0.0, 0.0, 100.0, 100.0),
    panel_extent: tuple[float, float, float, float] = (40.0, 40.0, 50.0, 50.0),
    panel_tilt_deg: float = 20.0,
    panel_az_deg: float = 180.0,
    roof_z0: float = 5.0,
    panel_z0: float = 5.5,
    seed: int = 0,
) -> None:
    """Synthesize a LAZ tile with class-2 ground + class-6 roof + class-6 panel.
    The panel is a planar surface tilted as specified, parked above the roof."""
    rng = np.random.default_rng(seed)

    def _plane_z(xy, tilt_deg, az_deg, z_at_centroid, centroid_xy):
        """Plane z(x,y) such that z(centroid_xy) == z_at_centroid."""
        t = np.radians(tilt_deg)
        a = np.radians(az_deg)
        nx, ny, nz = np.sin(t) * np.sin(a), np.sin(t) * np.cos(a), np.cos(t)
        cx, cy = centroid_xy
        return z_at_centroid - (nx * (xy[:, 0] - cx) + ny * (xy[:, 1] - cy)) / nz

    # Ground (class 2): scattered points across the tile at z=0, but NONE
    # inside the building footprint — LiDAR cannot see the ground through a
    # roof, and unphysical under-roof ground returns would (correctly) trip
    # the canopy-evidence detector and reroute the rooftop label.
    bx0, by0, bx1, by1 = 35.0, 35.0, 65.0, 65.0
    n_ground = 5_000
    g_xy = rng.uniform(
        [tile_extent[0], tile_extent[1]],
        [tile_extent[2], tile_extent[3]],
        size=(n_ground, 2),
    )
    under_building = (
        (g_xy[:, 0] >= bx0) & (g_xy[:, 0] <= bx1)
        & (g_xy[:, 1] >= by0) & (g_xy[:, 1] <= by1)
    )
    g_xy = g_xy[~under_building]
    ground = np.column_stack([g_xy, np.zeros(len(g_xy)), np.full(len(g_xy), 2)])

    # Roof (class 6) inside a 30x30m building; planar, low-tilt south
    bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
    n_roof = 4_000
    r_xy = rng.uniform([bx0, by0], [bx1, by1], size=(n_roof, 2))
    r_z = _plane_z(r_xy, 5.0, 180.0, roof_z0, (bcx, bcy)) + rng.normal(0, 0.01, n_roof)
    roof = np.column_stack([r_xy, r_z, np.full(n_roof, 6)])

    # Panel (class 6) above the roof; tilted at panel_tilt_deg facing panel_az_deg
    n_panel = 800
    pcx, pcy = (panel_extent[0] + panel_extent[2]) / 2, (panel_extent[1] + panel_extent[3]) / 2
    p_xy = rng.uniform(
        [panel_extent[0], panel_extent[1]],
        [panel_extent[2], panel_extent[3]],
        size=(n_panel, 2),
    )
    p_z = _plane_z(p_xy, panel_tilt_deg, panel_az_deg, panel_z0, (pcx, pcy)) + rng.normal(0, 0.01, n_panel)
    panel = np.column_stack([p_xy, p_z, np.full(n_panel, 6)])

    pts = np.concatenate([ground, roof, panel])

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x = pts[:, 0]
    las.y = pts[:, 1]
    las.z = pts[:, 2]
    las.classification = pts[:, 3].astype(np.uint8)
    las.write(str(path))

def rooftop_scene(standoff_m: float) -> dict:
    """A pitched roof with a polygon on it, panels ``standoff_m`` above the
    roof surface. standoff 0 reproduces the case where the detection's panels
    were absent from the point cloud and RANSAC can only fit the bare roof.
    """
    rng = np.random.default_rng(7)
    footprint = box(-10.0, -10.0, 20.0, 20.0)
    panel_poly = box(0.0, 0.0, 6.0, 4.0)
    slope = np.tan(np.radians(20.0))          # 20 deg, facing -y (south)

    def roof_z(xy: np.ndarray) -> np.ndarray:
        return 6.0 - slope * (xy[:, 1] - 5.0)

    r_xy = rng.uniform([-10.0, -10.0], [20.0, 20.0], size=(6000, 2))
    roof_pts = np.column_stack([r_xy, roof_z(r_xy) + rng.normal(0, 0.01, len(r_xy))])
    p_xy = rng.uniform([0.0, 0.0], [6.0, 4.0], size=(600, 2))
    panel_pts = np.column_stack(
        [p_xy, roof_z(p_xy) + standoff_m + rng.normal(0, 0.01, len(p_xy))]
    )
    # Ground only outside the building — LiDAR cannot see through a roof.
    g_xy = rng.uniform([-60.0, -60.0], [60.0, 60.0], size=(4000, 2))
    keep = (g_xy[:, 0] < -12) | (g_xy[:, 0] > 22) | (g_xy[:, 1] < -12) | (g_xy[:, 1] > 22)
    ground = np.column_stack([g_xy[keep], np.zeros(keep.sum())])
    return {
        "polygon": panel_poly,
        "panel_pts": panel_pts,
        "roof_input_pts": np.vstack([roof_pts, panel_pts]),
        "ground_xyz": ground,
        "footprints": gpd.GeoDataFrame(
            {"building_id": ["b1"]}, geometry=[footprint], crs="EPSG:6341"
        ),
        "other_pv_polygons": gpd.GeoDataFrame(geometry=[], crs="EPSG:6341"),
    }

def laz_flown_on(path, flight, n: int = 200) -> None:
    import laspy

    from pv_geom.io.lidar import (
        _GPS_EPOCH,
        _GPS_UTC_LEAP_SECONDS,
        _LAS_GPS_TIME_OFFSET,
    )

    rng = np.random.default_rng(0)
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([0.0, 0.0, 0.0])
    las = laspy.LasData(header)
    las.x = rng.uniform(0, 10, size=n)
    las.y = rng.uniform(0, 10, size=n)
    las.z = rng.uniform(0, 5, size=n)
    # Half ground, half building: a tile with no ground class is refused.
    las.classification = np.where(np.arange(n) % 2 == 0, 2, 6).astype(np.uint8)
    secs = (
        (datetime.combine(flight, datetime.min.time()) - _GPS_EPOCH).total_seconds()
        + _GPS_UTC_LEAP_SECONDS - _LAS_GPS_TIME_OFFSET
    )
    las.gps_time = np.full(n, secs)
    las.write(str(path))
