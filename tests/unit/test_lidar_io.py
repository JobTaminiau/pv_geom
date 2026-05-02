"""Unit tests for ``io.lidar``. M2."""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np
import pytest
from shapely.geometry import box

from pv_geom.io.lidar import clip_points_to_polygon, read_tile_points


def _write_synthetic_laz(
    path: Path,
    n: int = 200,
    x_extent: tuple[float, float] = (0.0, 10.0),
    y_extent: tuple[float, float] = (0.0, 10.0),
    z_extent: tuple[float, float] = (0.0, 5.0),
    classes: tuple[int, ...] = (1, 2, 6),
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([0.0, 0.0, 0.0])
    las = laspy.LasData(header)
    las.x = rng.uniform(*x_extent, size=n)
    las.y = rng.uniform(*y_extent, size=n)
    las.z = rng.uniform(*z_extent, size=n)
    las.classification = rng.choice(classes, size=n).astype(np.uint8)
    las.write(str(path))


def test_reads_local_laz(tmp_path: Path) -> None:
    p = tmp_path / "tile.laz"
    _write_synthetic_laz(p, n=500)
    pts, crs_str = read_tile_points(p)
    assert pts.shape == (500, 4)
    assert pts.dtype == np.float64
    assert pts[:, :3].max() <= 10.0
    # Synthetic LAZ has no CRS in header
    assert crs_str is None or isinstance(crs_str, str)


def test_filters_by_class(tmp_path: Path) -> None:
    p = tmp_path / "tile.laz"
    _write_synthetic_laz(p, n=500, classes=(1, 2, 6), seed=42)
    full, _ = read_tile_points(p)
    cls6, _ = read_tile_points(p, classes=(6,))
    cls1_2, _ = read_tile_points(p, classes=(1, 2))
    assert len(cls6) + len(cls1_2) == len(full)
    assert set(cls6[:, 3].astype(int)) <= {6}
    assert set(cls1_2[:, 3].astype(int)) <= {1, 2}


def test_clip_to_polygon_keeps_only_inside(tmp_path: Path) -> None:
    p = tmp_path / "tile.laz"
    _write_synthetic_laz(p, n=2000, x_extent=(0, 10), y_extent=(0, 10), seed=7)
    pts, _ = read_tile_points(p)
    poly = box(2.0, 2.0, 5.0, 5.0)
    clipped = clip_points_to_polygon(pts, poly)
    assert len(clipped) > 0
    assert ((clipped[:, 0] >= 2) & (clipped[:, 0] <= 5)).all()
    assert ((clipped[:, 1] >= 2) & (clipped[:, 1] <= 5)).all()
    # Sanity: roughly 9% of a 100 m^2 area is the 9 m^2 box; expect ~180 ± noise
    assert 100 < len(clipped) < 300


def test_clip_with_erosion_smaller_set(tmp_path: Path) -> None:
    p = tmp_path / "tile.laz"
    _write_synthetic_laz(p, n=2000)
    pts, _ = read_tile_points(p)
    poly = box(2.0, 2.0, 5.0, 5.0)
    full = clip_points_to_polygon(pts, poly, erosion_m=0.0)
    eroded = clip_points_to_polygon(pts, poly, erosion_m=0.5)
    assert len(eroded) < len(full)


def test_clip_full_erosion_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "tile.laz"
    _write_synthetic_laz(p, n=200)
    pts, _ = read_tile_points(p)
    poly = box(0, 0, 1, 1)
    out = clip_points_to_polygon(pts, poly, erosion_m=2.0)   # erodes past polygon
    assert len(out) == 0


def test_clip_empty_input() -> None:
    out = clip_points_to_polygon(np.zeros((0, 4)), box(0, 0, 1, 1))
    assert out.shape == (0, 4)
