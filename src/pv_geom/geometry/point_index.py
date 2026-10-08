"""Spatial lookups over a tile group's point cloud.

A tile group holds tens of millions of returns and hundreds of polygons. Testing
every polygon against every point made the per-polygon cost scale with the tile
rather than with the polygon; binning the points into a uniform grid once makes
each lookup proportional to the handful of cells the polygon touches.
"""

from __future__ import annotations

import numpy as np


class PointGrid:
    """Points sorted into square cells for fast bounding-box retrieval."""

    def __init__(self, pts: np.ndarray, cell_m: float = 10.0) -> None:
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 2:
            raise ValueError(f"pts must be (N, >=2); got {pts.shape}")
        self.cell_m = float(cell_m)
        if len(pts) == 0:
            self._pts = pts
            self._x0 = self._y0 = 0.0
            self._nx = self._ny = 0
            self._offsets = np.zeros(1, dtype=np.int64)
            return
        self._x0 = float(pts[:, 0].min())
        self._y0 = float(pts[:, 1].min())
        ix = ((pts[:, 0] - self._x0) / self.cell_m).astype(np.int64)
        iy = ((pts[:, 1] - self._y0) / self.cell_m).astype(np.int64)
        self._nx = int(ix.max()) + 1
        self._ny = int(iy.max()) + 1
        key = ix * self._ny + iy
        order = np.argsort(key, kind="stable")
        self._pts = pts[order]
        counts = np.bincount(key, minlength=self._nx * self._ny)
        self._offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    def __len__(self) -> int:
        return len(self._pts)

    @property
    def points(self) -> np.ndarray:
        return self._pts

    def query_bbox(self, minx: float, miny: float, maxx: float, maxy: float) -> np.ndarray:
        """All points with ``minx <= x <= maxx`` and ``miny <= y <= maxy``."""
        if len(self._pts) == 0:
            return self._pts
        ix0 = max(int(np.floor((minx - self._x0) / self.cell_m)), 0)
        ix1 = min(int(np.floor((maxx - self._x0) / self.cell_m)), self._nx - 1)
        iy0 = max(int(np.floor((miny - self._y0) / self.cell_m)), 0)
        iy1 = min(int(np.floor((maxy - self._y0) / self.cell_m)), self._ny - 1)
        if ix0 > ix1 or iy0 > iy1:
            return self._pts[:0]
        # Within one column of cells the iy range is contiguous in sort order.
        chunks = [
            self._pts[self._offsets[ix * self._ny + iy0]: self._offsets[ix * self._ny + iy1 + 1]]
            for ix in range(ix0, ix1 + 1)
        ]
        cand = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        if len(cand) == 0:
            return cand
        keep = (
            (cand[:, 0] >= minx) & (cand[:, 0] <= maxx)
            & (cand[:, 1] >= miny) & (cand[:, 1] <= maxy)
        )
        return cand[keep]

    def query_polygon_bounds(self, polygon, pad_m: float = 0.0) -> np.ndarray:
        minx, miny, maxx, maxy = polygon.bounds
        return self.query_bbox(minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m)


class GroundModel:
    """Local ground elevation from ground-class returns.

    A fine grid of per-cell mean elevation, backed by a coarse grid for cells
    with no ground return (under buildings and dense canopy) and by the global
    median as a last resort. Coarse enough to be cheap, local enough that a
    "height above ground" filter holds on terrain that is not flat.
    """

    def __init__(self, ground_xyz: np.ndarray, cell_m: float = 5.0,
                 coarse_factor: int = 10) -> None:
        g = np.asarray(ground_xyz, dtype=np.float64)
        self.empty = len(g) == 0
        if self.empty:
            return
        self._x0 = float(g[:, 0].min())
        self._y0 = float(g[:, 1].min())
        self._global = float(np.median(g[:, 2]))
        self._fine = self._grid(g, float(cell_m))
        self._coarse = self._grid(g, float(cell_m) * coarse_factor)

    def _grid(self, g: np.ndarray, cell: float) -> tuple[float, int, int, np.ndarray]:
        ix = ((g[:, 0] - self._x0) / cell).astype(np.int64)
        iy = ((g[:, 1] - self._y0) / cell).astype(np.int64)
        nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
        key = ix * ny + iy
        n = np.bincount(key, minlength=nx * ny)
        total = np.bincount(key, weights=g[:, 2], minlength=nx * ny)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(n > 0, total / n, np.nan)
        return cell, nx, ny, mean

    def _lookup(self, grid: tuple[float, int, int, np.ndarray],
                x: np.ndarray, y: np.ndarray) -> np.ndarray:
        cell, nx, ny, mean = grid
        ix = np.floor((x - self._x0) / cell).astype(np.int64)
        iy = np.floor((y - self._y0) / cell).astype(np.int64)
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        out = np.full(len(x), np.nan)
        out[inside] = mean[ix[inside] * ny + iy[inside]]
        return out

    def ground_z(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Ground elevation under each (x, y); NaN everywhere if no ground."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if self.empty:
            return np.full(len(x), np.nan)
        z = self._lookup(self._fine, x, y)
        miss = np.isnan(z)
        if miss.any():
            z[miss] = self._lookup(self._coarse, x[miss], y[miss])
            z[np.isnan(z)] = self._global
        return z
