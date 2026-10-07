"""Pydantic settings models for pv_geom. Mirrors configs/default.yaml (PRD §9.2)."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, PrivateAttr


class CRSConfig(BaseModel):
    # CRS everything is computed in. Points are NOT reprojected, so this must be
    # the LiDAR's own horizontal CRS, and it must be metric (every threshold in
    # this file is in metres). "auto" takes it from the tile index, falling back
    # to a LAZ header; set an explicit "EPSG:xxxx" to assert it instead.
    target: str = "auto"


class PolygonsConfig(BaseModel):
    """Checks on the polygon layer. None of these drop a polygon: each input
    polygon keeps its row, and the checks decide its flags."""

    # Polygons smaller than this are flagged `below_min_area` (a single module
    # is ~1.7-2 m2; below ~3 m2 there are rarely enough returns for a fit).
    min_area_m2: float = 1.0
    # A polygon sharing at least this fraction of its own area with another
    # input polygon is flagged `overlaps_polygon`. Set to 0 to skip the check.
    overlap_flag_frac: float = 0.2


class PanelPlaneConfig(BaseModel):
    erosion_m: float = 0.15
    ransac_threshold_m: float = 0.05
    # Noise-adaptive tolerance. 5 cm suits clean, single-swath data (Phoenix:
    # ~2 cm scatter about the plane) but rejects most arrays where the returns
    # are noisier — on the Delaware 2023 collection the scatter is ~7.5 cm and
    # only 24% of polygons reached consensus at 5 cm, against 80% at 15 cm.
    # When the fit fails at ransac_threshold_m, the scatter of the returns
    # about the best plane is measured and the fit is repeated at twice that,
    # capped here. Such rows record the tolerance used and carry the
    # `wide_tolerance_fit` flag. Set the cap equal to ransac_threshold_m to
    # disable.
    ransac_threshold_max_m: float = 0.15
    min_inlier_frac: float = 0.6
    max_iter: int = 200
    min_density_pts_per_m2: float = 3.0
    min_points: int = 30                # flat floor; size-sweep showed ~5 is the precision
                                        # floor but RANSAC robustness needs ~30 inliers.
    tilt_floor_deg: float = 1.0
    uncertainty_method: Literal["bootstrap", "none"] = "bootstrap"   # "none" skips it
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
    # Fit a roof reference from an *open* ring (not clipped to a footprint) when
    # no footprint layer was supplied, or the polygon misses every footprint.
    # The ring is then just the elevated returns around the array; the consensus
    # floor, RMSE gate and collar guard decide whether that is a usable plane.
    # Rows record which kind they got in `roof_ref_source`.
    open_ring: bool = True
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
    # ARCHIVED in 0.2.0. Ground-truth validation (300 labelled polygons,
    # 2026-07-30) put carport precision at 5% and pole-mount at 0%, so mounting
    # classification is off by default and its columns are not part of the core
    # output schema. Enable to get the experimental mounting_* columns back; the
    # labels need better heuristics and tested examples before they are trusted.
    enabled: bool = False
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


class ClassificationConfig(BaseModel):
    """ASPRS class assignments. USGS LPC tiles often lack class 6; we fall back
    to class 1 returns above ground when class 6 is absent."""
    panel_class_primary: int = 6        # building
    panel_class_fallback: int = 1       # unclassified
    ground_class: int = 2
    # Min height above *local* ground to keep a class-1 return as a panel
    # candidate. Ground-mount panels live at ~0.5-2.5 m, so 1.5 m was
    # deleting their lower halves (biasing tilt + HAG and starving density);
    # 0.8 keeps them while still rejecting near-ground clutter.
    fallback_height_above_ground_m: float = 0.8
    # Cell size of the ground-elevation grid that "local ground" is read from.
    # (Before 0.2.0 this was one median per tile group, which only holds on
    # flat terrain.)
    ground_grid_cell_m: float = 5.0


class IOConfig(BaseModel):
    lidar_reader: Literal["pdal", "laspy"] = "laspy"   # pdal needs the [pdal] extra
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)


class CoiledConfig(BaseModel):
    name: str = "pv-geom"
    n_workers: int = 40
    worker_memory: str = "16GiB"
    worker_cpu: int = 4
    software: str = "pv-geom-2026-05"
    # Cloud region for the cluster: put it where the LiDAR bucket is.
    region: str = "us-east-2"
    # What workers ``pip install`` to get pv_geom (Coiled drops git+ URLs from
    # environment specs, so it is installed at cluster start). Must be the
    # version the client runs.
    package_source: str = "git+https://github.com/JobTaminiau/pv_geom.git@main"


class LocalConfig(BaseModel):
    n_workers: int = 8
    threads_per_worker: int = 2


class ComputeConfig(BaseModel):
    backend: Literal["coiled", "local"] = "local"
    coiled: CoiledConfig = Field(default_factory=CoiledConfig)
    local: LocalConfig = Field(default_factory=LocalConfig)


class VintageConfig(BaseModel):
    """Acquisition dates of the two inputs.

    pv_geom measures geometry from LiDAR at polygons detected some other way —
    usually aerial or satellite imagery. The two are rarely captured at the same
    time. If the imagery postdates the LiDAR, arrays built in between are in the
    polygon set but not in the point cloud, and the plane fitted there is the
    bare roof (or ground). Both dates are therefore run inputs: they are stamped
    on every row and decide each row's ``geometry_basis``.
    """

    model_config = ConfigDict(populate_by_name=True)

    # When the polygon set's source imagery was captured: "2024", "2024-04" or
    # "2024-04-01". A year or month is treated as a window and its *latest* day
    # is used, so the gap to the LiDAR is never understated. (`input_epoch` is
    # the pre-0.2.0 name and still accepted.)
    polygon_vintage: str | int | date | None = Field(
        default=None, validation_alias=AliasChoices("polygon_vintage", "input_epoch")
    )
    # Optional per-polygon date column in the polygon file (mosaics, permit
    # dates). Rows where it is null fall back to `polygon_vintage`.
    polygon_vintage_column: str | None = None
    # Declared LiDAR capture date (same formats). Leave unset to have it
    # *measured* per tile from per-point GPS time, which is the flight date; the
    # LAS header date is the delivery date and can lag by more than a year.
    lidar_date: str | int | date | None = None
    # Tiles sampled up front for the run-level flight window reported before
    # compute starts (header + first chunk only). 0 disables the preview; rows
    # still get their own tile's measured date.
    sample_tiles: int = 25


class InputsConfig(BaseModel):
    """Where the run's inputs are. With these set, a run is just
    ``pv-geom run --config area.yaml``; command-line options override them.
    Relative paths are resolved against the config file's folder."""

    polygons: str | None = None          # PV polygon layer (any vector format; path or s3://)
    polygon_id_col: str | None = None    # id column; None = auto-detect, else synthesize
    lidar_prefix: str | None = None      # folder or s3:// prefix holding the LAZ tiles
    tile_index: str | None = None        # tile index layer; None = build it from tile headers
    tile_id_col: str | None = None       # id column in the tile index; None = auto-detect
    name_template: str = "{name}.laz"    # tile filename from the tile id
    footprints: str | None = None        # optional building footprints
    # Optional scope limits, for trial runs. Polygons outside them are not part
    # of the run and get no row.
    bbox: tuple[float, float, float, float] | None = None    # xmin ymin xmax ymax, run CRS
    max_polygons: int | None = None


class StudyConfig(BaseModel):
    """What this run is of, and where its output goes."""

    name: str | None = None              # used in report titles and the methods text
    output: str | None = None            # output folder or s3:// prefix


class PVGeomConfig(BaseModel):
    # Unknown keys are ignored so configs written for older versions (which
    # carried a never-implemented `output:` block) still load.
    model_config = ConfigDict(extra="ignore")

    # Folder relative paths in the config are resolved against (the config
    # file's own folder; None for a config built in code).
    _base_dir: Path | None = PrivateAttr(default=None)

    study: StudyConfig = Field(default_factory=StudyConfig)
    inputs: InputsConfig = Field(default_factory=InputsConfig)

    crs: CRSConfig = Field(default_factory=CRSConfig)
    polygons: PolygonsConfig = Field(default_factory=PolygonsConfig)
    panel_plane: PanelPlaneConfig = Field(default_factory=PanelPlaneConfig)
    multi_plane: MultiPlaneConfig = Field(default_factory=MultiPlaneConfig)
    roof_plane: RoofPlaneConfig = Field(default_factory=RoofPlaneConfig)
    heights: HeightsConfig = Field(default_factory=HeightsConfig)
    mounting_rules: MountingRulesConfig = Field(default_factory=MountingRulesConfig)
    io: IOConfig = Field(default_factory=IOConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    vintage: VintageConfig = Field(default_factory=VintageConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> PVGeomConfig:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cfg = cls(**raw)
        cfg._base_dir = Path(path).resolve().parent
        return cfg

    def resolve(self, location: str | None) -> str | None:
        """A path from the config as an absolute location: remote URIs and
        absolute paths pass through, a relative path is taken against the
        config file's folder."""
        if location is None:
            return None
        s = str(location)
        if "://" in s or Path(s).is_absolute() or self._base_dir is None:
            return s
        return str((self._base_dir / s).resolve())

    def model_copy(self, *, update=None, deep: bool = False):
        copy = super().model_copy(update=update, deep=deep)
        copy._base_dir = self._base_dir
        return copy

    def result_hash(self) -> str:
        """Like :meth:`hash`, but ignoring how the run is executed.

        Worker counts and backends do not change a result, so an interrupted
        run may be resumed on a different cluster shape — but not with
        different measurement settings. ``--resume`` compares this.
        """
        dump = self.model_dump(mode="json")
        # Not what is computed, but how, on what, and where to: the inputs'
        # identity is checked separately, by the plan fingerprint.
        for key in ("compute", "inputs", "study"):
            dump.pop(key, None)
        if not self.mounting_rules.enabled:
            dump["mounting_rules"] = {"enabled": False}
        return hashlib.sha256(json.dumps(dump, sort_keys=True).encode("utf-8")).hexdigest()

    def hash(self) -> str:
        """sha256 of everything that can change a result or how it was run.

        The archived mounting classifier's thresholds count only when it is
        switched on: left off, they cannot affect the output, and should not
        make two otherwise identical runs look different.
        """
        dump = self.model_dump(mode="json")
        dump.pop("study", None)            # a label and a destination change nothing
        if not self.mounting_rules.enabled:
            dump["mounting_rules"] = {"enabled": False}
        canonical = json.dumps(dump, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
