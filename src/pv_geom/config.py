"""Pydantic settings models for pv_geom. Mirrors configs/default.yaml (PRD §9.2)."""

from __future__ import annotations

import hashlib
import json
from datetime import date
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
    # Consensus floor for the *ring* fit. Deliberately below the panel fit's
    # 0.6: a ring buffered around an array straddles roof facets, eaves and
    # parapets, so no single plane holds 60% of it on a gable roof. At 0.6 the
    # v0.1.0 Phoenix run rejected 163,716 rows (46.9%) as `roof_complex` whose
    # ring fits had a median RMSE of 5.8 cm — 99.999% of them would have passed
    # the rmse_max_m quality gate. Those rows lost their roof reference and with
    # it any chance of a panel-standoff (vintage) screen. Quality is enforced by
    # rmse_max_m; this only decides whether one plane describes enough of the
    # ring to be worth reporting.
    min_inlier_frac: float = 0.4
    # Collar guard. A ring wide enough to hold min_points can reach across a
    # ridge, and RANSAC then reports whichever facet is *larger* — not
    # necessarily the one the array sits on. Measured on real Phoenix polygons:
    # at min_inlier_frac=0.4 with no guard, 3 of 7 newly-recovered rows picked a
    # facet 20-45 deg away from the roof directly beside the array, which would
    # corrupt panel_roof_angle_deg and height_above_roof_m far worse than having
    # no roof fit at all. So the collar — ring points within `collar_m` of the
    # polygon, i.e. the roof the array is physically resting against — is the
    # authority: if the wide-ring plane does not explain at least
    # `collar_agreement_min` of it, the fit is redone on the collar alone.
    collar_m: float = 1.2
    collar_min_points: int = 40
    collar_agreement_min: float = 0.5
    # Minimum (footprint ∩ polygon area) / polygon area for on_building=True.
    # A sliver touch must not route ground mounts / adjacent carports down the
    # rooftop rules; 0.5 tolerates typical ML-footprint misregistration (1-3 m)
    # while rejecting edge clips. Calibrate against a labeled sample.
    min_overlap_frac: float = 0.5


class HeightsConfig(BaseModel):
    ground_search_radius_m: float = 10.0
    # Large buildings often have no class-2 return within the centroid disk
    # (ground stops at the wall). Fall back to the k nearest ground points
    # within the max radius so big commercial rooftops get a HAG instead of
    # NaN. Set ground_fallback_k=0 to disable.
    ground_fallback_k: int = 50
    ground_fallback_max_radius_m: float = 100.0
    # Minimum panel-above-roof separation that a plane fit can resolve. Panel
    # and roof fits each carry ~2 cm RMSE, so a smaller gap does not establish
    # that a panel is physically present above the roof surface; below it the
    # row gets the `no_panel_standoff` flag. See README (Quality flags).
    min_panel_standoff_m: float = 0.05
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
    # Real carports top out around 5-6 m of clearance. Above the cap, an
    # off-building elevated array is far more likely a rooftop whose building
    # is missing from the footprint layer — fall through to ambiguous +
    # possible_missing_footprint instead of a confident carport.
    height_above_ground_m_max: float = 6.0


class MountingRule4(BaseModel):
    aspect_ratio_min: float = 4.0
    tilt_deg_max: float = 35.0
    height_above_ground_m_max: float = 2.0


class MountingRule5(BaseModel):
    tilt_deg_min: float = 5.0
    height_above_ground_m_max: float = 2.0


class MountingRule7(BaseModel):
    """east_west_rack_rooftop — promotes the M5 ``east_west_rack`` flag to a
    label. EW racks sit at modest tilts; the cap rejects steep two-plane fits."""
    tilt_deg_max: float = 20.0


class MountingRule8(BaseModel):
    """pole_mount — small, near-square, elevated, off-building."""
    height_above_ground_m_min: float = 2.0
    area_m2_max: float = 20.0
    aspect_ratio_max: float = 2.0
    # Same missing-footprint guard as R3; pole mounts run taller than carports.
    height_above_ground_m_max: float = 8.0


class MountingRulesConfig(BaseModel):
    R1: MountingRule1 = Field(default_factory=MountingRule1)
    R2: MountingRule2 = Field(default_factory=MountingRule2)
    R3: MountingRule3 = Field(default_factory=MountingRule3)
    R4: MountingRule4 = Field(default_factory=MountingRule4)
    R5: MountingRule5 = Field(default_factory=MountingRule5)
    R7: MountingRule7 = Field(default_factory=MountingRule7)
    R8: MountingRule8 = Field(default_factory=MountingRule8)
    # Roof tilt at/below which a roof counts as "flat": splits R1's label
    # (flush_mount_flat_roof vs flush_mount_pitched_roof) and guards R7 against
    # the gable-facet artifact (two roof facets also face ~180 deg apart).
    # Flat commercial roofs drain at <= ~5 deg; the lowest common pitched roofs
    # (2:12) are ~9.5 deg. Calibrate against a labeled sample when one exists.
    flat_roof_tilt_deg_max: float = 7.0
    confidence_margin: float = 0.5
    # Canopy evidence: ground-class returns *inside* the polygon, several
    # metres below the panel plane, are the LiDAR signature of an open-sided
    # canopy (carport / pole mount). Footprint-independent, so it reroutes
    # canopies that footprint layers map as buildings, and blocks the
    # ground-mount rules when present.
    canopy_min_ground_points_under: int = 15
    canopy_gap_m_min: float = 2.0
    # Off-building + HAG above this + no canopy evidence => flag
    # possible_missing_footprint (the R3/R8 height caps then usually route the
    # polygon to ambiguous rather than a confident canopy label).
    missing_footprint_hag_m: float = 6.0
    # Confidence ceiling for a labelled row carrying `no_panel_standoff`. Such a
    # row satisfies R1 maximally (panel-roof angle ~0, height above roof ~0) —
    # on the v0.1.0 Phoenix run 34,873 of them scored >= 0.999 — even though the
    # evidence is equally consistent with there being no panel in the cloud at
    # all. Without the cap, filtering on high confidence selects *for* the
    # unmeasured rows. 0.5 keeps the label (it is still the best reading of what
    # was observed) while marking it as no better than borderline. Set to 1.0 to
    # disable. Does not apply to `ambiguous`, whose confidence means the
    # opposite thing.
    no_panel_standoff_confidence_max: float = 0.5


class S3Config(BaseModel):
    requester_pays: bool = False
    region: str = "us-east-2"


class ClassificationConfig(BaseModel):
    """ASPRS class assignments. USGS LPC tiles often lack class 6; we fall back
    to class 1 returns above ground when class 6 is absent."""
    panel_class_primary: int = 6        # building
    panel_class_fallback: int = 1       # unclassified
    ground_class: int = 2
    # Min height above the tile-group ground median to keep a class-1 return as
    # a panel candidate. Ground-mount panels live at ~0.5-2.5 m, so 1.5 m was
    # deleting their lower halves (biasing tilt + HAG and starving density);
    # 0.8 keeps them while still rejecting near-ground clutter.
    fallback_height_above_ground_m: float = 0.8


class IOConfig(BaseModel):
    lidar_reader: Literal["pdal", "laspy"] = "laspy"
    s3: S3Config = Field(default_factory=S3Config)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)


class CoiledConfig(BaseModel):
    name: str = "pv-geom"
    n_workers: int = 40
    worker_memory: str = "16GiB"
    worker_cpu: int = 4
    software: str = "pv-geom-2026-05"


class LocalConfig(BaseModel):
    n_workers: int = 8
    threads_per_worker: int = 2


class ComputeConfig(BaseModel):
    backend: Literal["coiled", "local"] = "local"
    coiled: CoiledConfig = Field(default_factory=CoiledConfig)
    local: LocalConfig = Field(default_factory=LocalConfig)


class VintageConfig(BaseModel):
    """Acquisition epochs of the two inputs, and how hard to check them.

    pv_geom measures panel geometry from LiDAR against an inventory detected
    some other way — usually orthoimagery. If the imagery postdates the LiDAR,
    arrays built in between are in the inventory but not in the point cloud, and
    the pipeline will happily fit and report the bare roof under the panel
    column names. This is a property of the *pairing*, not of any one row, so it
    cannot be inferred from the data; declare it and the run will check itself.
    """

    # When the detection inventory's source imagery was captured (ISO date, or
    # the later bound of a mosaic). Leave unset to skip the co-temporality
    # check; the LiDAR side is measured from the tiles either way.
    input_epoch: date | None = None
    # Tiles sampled for LiDAR flight dates. Only the header and first point
    # chunk are read per tile, streamed rather than downloaded. 0 disables.
    sample_tiles: int = 25


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
    vintage: VintageConfig = Field(default_factory=VintageConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> PVGeomConfig:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)

    def hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
