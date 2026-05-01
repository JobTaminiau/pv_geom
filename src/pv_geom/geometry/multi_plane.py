"""Multi-plane detection (e.g. east-west racks) + tracker heuristic. M5 (PRD §7.2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pv_geom.config import MultiPlaneConfig
from pv_geom.geometry.plane_fit import PlaneFit


@dataclass(frozen=True)
class MultiPlaneResult:
    primary: PlaneFit
    secondary: PlaneFit | None
    flags: tuple[str, ...]


def detect_multi_plane(
    points: np.ndarray,
    primary: PlaneFit,
    cfg: MultiPlaneConfig,
) -> MultiPlaneResult:
    raise NotImplementedError("M5")
