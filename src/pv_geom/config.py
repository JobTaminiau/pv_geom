"""Pydantic settings models for pv_geom. Mirrors configs/default.yaml (PRD §9.2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class CRSConfig(BaseModel):
    target: str = "EPSG:6341"           # NAD83(2011) / UTM 12N (metres); USGS LPC AZ


class PanelPlaneConfig(BaseModel):
    erosion_m: float = 0.15
    ransac_threshold_m: float = 0.05
    min_inlier_frac: float = 0.6
    max_iter: int = 200
    min_density_pts_per_m2: float = 3.0
    min_points: int = 30                # flat floor; size-sweep showed ~5 is the precision
                                        # floor but RANSAC robustness needs ~30 inliers.
    tilt_floor_deg: float = 1.0
    uncertainty_method: Literal["bootstrap", "covariance"] = "bootstrap"
    bootstrap_samples: int = 50


class MultiPlaneConfig(BaseModel):
    enabled: bool = True
    secondary_min_frac: float = 0.20
    ew_rack_azimuth_tol_deg: float = 25.0
    ew_rack_tilt_tol_deg: float = 5.0


class RoofPlaneConfig(BaseModel):
    enabled: bool = True
    buffer_m: float = 3.0
    buffer_max_m: float = 5.0
    buffer_step_m: float = 0.5          # iterative expansion step when ring is too sparse
    min_points: int = 100
    ransac_threshold_m: float = 0.15    # RANSAC inlier distance for the roof fit
    rmse_max_m: float = 0.10            # rejection threshold on the post-fit inlier RMSE


class HeightsConfig(BaseModel):
    ground_search_radius_m: float = 10.0
    use_whitebox_dem: bool = False


class MountingRule1(BaseModel):
    panel_roof_angle_deg_max: float = 5.0
    height_above_roof_m_max: float = 0.5


class MountingRule2(BaseModel):
    panel_roof_angle_deg_min: float = 5.0
    height_above_roof_m_max: float = 1.5
    fallback_tilt_deg_min: float = 5.0
    fallback_height_above_ground_m_min: float = 2.5


class MountingRule3(BaseModel):
    height_above_ground_m_min: float = 2.0
    aspect_ratio_min: float = 2.0


class MountingRule4(BaseModel):
    aspect_ratio_min: float = 4.0
    tilt_deg_max: float = 35.0
    height_above_ground_m_max: float = 2.0


class MountingRule5(BaseModel):
    tilt_deg_min: float = 5.0
    height_above_ground_m_max: float = 2.0


class MountingRulesConfig(BaseModel):
    R1: MountingRule1 = Field(default_factory=MountingRule1)
    R2: MountingRule2 = Field(default_factory=MountingRule2)
    R3: MountingRule3 = Field(default_factory=MountingRule3)
    R4: MountingRule4 = Field(default_factory=MountingRule4)
    R5: MountingRule5 = Field(default_factory=MountingRule5)
    confidence_margin: float = 0.5


class S3Config(BaseModel):
    requester_pays: bool = False
    region: str = "us-west-2"


class ClassificationConfig(BaseModel):
    """ASPRS class assignments. USGS LPC tiles often lack class 6; we fall back
    to class 1 returns above ground when class 6 is absent."""
    panel_class_primary: int = 6        # building
    panel_class_fallback: int = 1       # unclassified
    ground_class: int = 2
    fallback_height_above_ground_m: float = 1.5


class IOConfig(BaseModel):
    lidar_reader: Literal["pdal", "laspy"] = "laspy"
    s3: S3Config = Field(default_factory=S3Config)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)


class CoiledConfig(BaseModel):
    name: str = "pv-geom"
    n_workers: int = 40
    worker_memory: str = "8GiB"
    worker_cpu: int = 4
    software: str = "pv-geom-2026-05"


class LocalConfig(BaseModel):
    n_workers: int = 8
    threads_per_worker: int = 2


class ComputeConfig(BaseModel):
    backend: Literal["coiled", "local"] = "local"
    coiled: CoiledConfig = Field(default_factory=CoiledConfig)
    local: LocalConfig = Field(default_factory=LocalConfig)


class OutputConfig(BaseModel):
    partition_size: int = 100000
    write_geoparquet: bool = True
    also_write_csv: bool = False


class PVGeomConfig(BaseModel):
    crs: CRSConfig = Field(default_factory=CRSConfig)
    panel_plane: PanelPlaneConfig = Field(default_factory=PanelPlaneConfig)
    multi_plane: MultiPlaneConfig = Field(default_factory=MultiPlaneConfig)
    roof_plane: RoofPlaneConfig = Field(default_factory=RoofPlaneConfig)
    heights: HeightsConfig = Field(default_factory=HeightsConfig)
    mounting_rules: MountingRulesConfig = Field(default_factory=MountingRulesConfig)
    io: IOConfig = Field(default_factory=IOConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> PVGeomConfig:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)

    def hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
