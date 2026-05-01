"""RANSAC + LSQ plane fitting. Tilt/azimuth from the unit normal. M3 (PRD §7.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlaneFit:
    normal: np.ndarray          # unit vector, length 3
    centroid: np.ndarray        # length 3
    tilt_deg: float             # 0 = horizontal
    azimuth_deg: float          # 0 = N, clockwise; NaN if tilt < tilt_floor
    rmse: float                 # m
    n_inliers: int
    n_total: int
    inlier_mask: np.ndarray     # bool, length n_total


def fit_plane_ransac(
    points: np.ndarray,
    ransac_threshold: float = 0.05,
    min_inlier_frac: float = 0.6,
    max_iter: int = 200,
    refine_with_lsq: bool = True,
    tilt_floor_deg: float = 1.0,
    seed: int | None = None,
) -> PlaneFit:
    """Fit a plane to ``points`` (N, 3) via RANSAC, optionally refined by PCA-LSQ on inliers."""
    raise NotImplementedError("M3")


def tilt_azimuth_from_normal(
    normal: np.ndarray, tilt_floor_deg: float = 1.0
) -> tuple[float, float]:
    """Return (tilt_deg, azimuth_deg) from a unit normal. Azimuth is NaN below tilt_floor."""
    raise NotImplementedError("M3")
