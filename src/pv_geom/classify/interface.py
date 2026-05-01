"""ABC for swappable mounting classifiers (rules-based v1; learned v2+)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class MountingFeatures:
    on_building: bool
    panel_tilt_deg: float
    panel_azimuth_deg: float
    panel_roof_angle_deg: float | None     # None / NaN if no roof plane
    height_above_roof_m: float | None
    height_above_ground_m: float
    area_m2: float
    aspect_ratio: float
    roof_plane_available: bool


@dataclass(frozen=True)
class MountingResult:
    label: str           # one of pv_geom.schema.MOUNTING_LABELS
    confidence: float    # [0, 1]
    triggered_rule: str  # e.g. "R1", "R2", ..., "R6"


class MountingClassifier(ABC):
    @abstractmethod
    def classify(self, features: MountingFeatures) -> MountingResult: ...
