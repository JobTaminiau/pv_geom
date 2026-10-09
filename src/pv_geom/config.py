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
    # A fit that needed the wider tolerance AND came out flatter than this is
    # flagged `envelope_fit`: scatter about a near-flat plane is what rows of
    # tilted modules on a flat roof look like, and the plane then describes the
    # envelope of the rows, not the modules. (Validation, 2026-10: a warehouse
    # roof reported at 10 degrees measured 1.4; 80 of its 82 fits matched this
    # signature against 0.2-0.3% of residential fits.)
    envelope_tilt_max_deg: float = 5.0
    # A fit is flagged `ambiguous_fit` when a different plane (2 degrees or more
    # away) holds at least this share of its inliers: the returns support two
    # planes about equally. The choice between them is reproducible, but a few
    # returns more or fewer could reverse it.
    ambiguous_rival_share: float = 0.9
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
    max_iter: int = 300                 # least number of plane hypotheses; small point sets
                                        # get many more, up to every triple (plane_fit.py)
    min_density_pts_per_m2: float = 3.0
    min_points: int = 30                # flat floor; size-sweep showed ~5 is the precision
                                        # floor but RANSAC robustness needs ~30 inliers.
    tilt_floor_deg: float = 1.0
    uncertainty_method: Literal["bootstrap", "none"] = "bootstrap"   # "none" skips it
    bootstrap_samples: int = 50


class MultiPlaneConfig(BaseModel):
    """Polygons that hold more than one plane are split into segments (facets)."""

    enabled: bool = True
    # A plane is a facet when it holds at least this share of the polygon's
    # returns (and at least panel_plane.min_points of them) ...
    secondary_min_frac: float = 0.20
    # ... and differs in orientation from every other facet by at least this.
    # Without it a single noisy surface is sliced into parallel "facets".
    segment_min_angle_deg: float = 10.0
    max_segments: int = 4
    # A further facet steeper than this is not an array: on the benchmarks the
    # planes found at 87-90 degrees are returns off walls, parapets and roof
    # edges inside a slightly oversized polygon. (The primary plane is not
    # subject to this; a genuinely steep array still gets its fit.)
    segment_max_tilt_deg: float = 70.0
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
    # corrupt angle_to_roof_deg and height_above_roof_m far worse than having
    # no roof fit at all. So the collar — ring points within `collar_m` of the
    # polygon, i.e. the roof the array is physically resting against — is the
    # authority: if the wide-ring plane does not explain at least
    # `collar_agreement_min` of it, the fit is redone on the collar alone.
    collar_m: float = 1.2
    collar_min_points: int = 40
    collar_agreement_min: float = 0.5
    # Facet search. On a hip or cross-gabled roof the ring covers several
    # facets and no single plane holds `min_inlier_frac` of it — on the Phoenix
    # test block that left 785 of 2,994 fitted polygons without a reference
    # (`roof_no_consensus`), the largest single cause of unscreened rows. When
    # the ring has no dominant plane, or its dominant plane is not the one beside
    # the array, the ring's planes are peeled off one at a time (up to
    # `max_facets`) and the one the collar agrees with is taken. Set
    # `facet_search: false` for the pre-0.5 behaviour.
    # Search the ring's planes as hard as the array's own (see
    # geometry.plane_fit). It costs about a third more time and, measured on
    # the benchmarks, does not make the roof reference any steadier.
    thorough_search: bool = False
    facet_search: bool = True
    max_facets: int = 4
    # A facet found by the search (or a fit to the collar alone) must explain a
    # clear majority of the collar. An array straddling a ridge has a collar
    # split about evenly between two facets; neither is its roof, and at a bare
    # majority one of them would be picked.
    facet_agreement_min: float = 0.65
    # Last resort, for arrays at a ridge or hip (or polygons covering two
    # facets), whose collar is genuinely split so that no facet holds a clear
    # majority of it: take the facet that is PARALLEL to the fitted array plane,
    # provided it also touches the array (holds `parallel_collar_min` of the
    # collar). A flush array is parallel to its own roof; so is a bare roof to
    # itself, which the standoff screen then correctly reports as unresolved. A
    # rack on a flat roof has no parallel facet and stays unscreened. Rows
    # referenced this way say so in `roof_ref_method`. 0 disables it.
    parallel_angle_max_deg: float = 10.0
    parallel_collar_min: float = 0.25
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
    # row gets the `no_standoff` flag. See README (Quality flags).
    min_panel_standoff_m: float = 0.05
    # The standoff is only meaningful against the plane the array sits on. On a
    # hip or cross-gable roof the reference can be a neighbouring facet: pitched,
    # and at an angle to the array. "Height above" such a plane is tens of
    # centimetres either way and says nothing (measured on New York City, 2026:
    # it confirmed 14% of real arrays and 16% of bare roofs). Where the
    # reference is pitched by at least reference_flat_max_tilt_deg and more than
    # reference_max_angle_deg from the array plane, the test is not applied and
    # the row is flagged roof_reference_not_parallel. A flat reference under a
    # tilted plane is a rack on a flat roof, and is tested as before.
    reference_max_angle_deg: float = 6.0
    reference_flat_max_tilt_deg: float = 7.0


class FreeStandingConfig(BaseModel):
    """Presence evidence for ground mounts and canopies, which have no roof
    beneath them for the standoff screen to use. A fitted plane counts as a
    free-standing structure when it is clear of the ground and the band of
    returns around it is mostly ground. (Measured 2026-10: 0.81-0.98 around
    ground-mounted rows, median 0.80 around parking canopies, 0.00-0.05 around
    arrays on large roofs, under 0.35 for nine in ten residential roofs.)"""

    enabled: bool = True
    band_m: float = 2.0               # width of the band around the polygon that is examined
    band_gap_m: float = 0.3           # skipped next to the polygon (its own edge returns)
    min_open_share: float = 0.6       # ground returns as a share of all returns in the band
    min_band_points: int = 20         # fewer than this and the band says nothing
    min_height_above_ground_m: float = 0.4


class MountingRule1(BaseModel):
    angle_to_roof_deg_max: float = 5.0
    height_above_roof_m_max: float = 0.5


class MountingRule2(BaseModel):
    angle_to_roof_deg_min: float = 5.0
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
    # Confidence ceiling for a labelled row carrying `no_standoff`. Such a
    # row satisfies R1 maximally (panel-roof angle ~0, height above roof ~0) —
    # on the v0.1.0 Phoenix run 34,873 of them scored >= 0.999 — even though the
    # evidence is equally consistent with there being no panel in the cloud at
    # all. Without the cap, filtering on high confidence selects *for* the
    # unmeasured rows. 0.5 keeps the label (it is still the best reading of what
    # was observed) while marking it as no better than borderline. Set to 1.0 to
    # disable. Does not apply to `ambiguous`, whose confidence means the
    # opposite thing.
    no_standoff_confidence_max: float = 0.5


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
    """The cloud cluster. Settings left on "auto" are worked out at run time
    (see ``pv_geom.coiled_env``), so nothing here is tied to one project."""

    name: str = "auto"                   # auto: pv-geom-<study.name>
    n_workers: int = 40
    worker_memory: str = "16GiB"
    worker_cpu: int = 4
    # Tasks per worker. Keep at 1 unless compute.memory_budget_gb times this
    # fits comfortably in worker_memory.
    worker_threads: int = 1
    # Coiled software environment. auto: one named after pv-geom's dependency
    # list, built if missing.
    software: str = "auto"
    # Cloud region. auto: where the LiDAR bucket is, so tiles are read in-region.
    region: str = "auto"
    # What workers ``pip install`` to get pv-geom. auto: the exact commit the
    # client is running (it must be pushed). Workers are checked after install
    # and the run stops if any has a different version.
    package_source: str = "auto"
    # Optional: what one worker costs per hour, so `--dry-run` can estimate the
    # bill. Leave unset to get a time estimate only.
    usd_per_worker_hour: float | None = None


class LocalConfig(BaseModel):
    n_workers: int = 8
    threads_per_worker: int = 2


class ComputeConfig(BaseModel):
    backend: Literal["coiled", "local"] = "local"
    # Memory one task may hold in LiDAR returns, in GB. Tiles are decoded in
    # chunks and only returns near a polygon are kept; if a tile group's kept
    # returns would exceed this, its polygons are split into spatial batches
    # that are measured one after another. Results do not depend on it. Set it
    # to about half a worker's memory; null removes the bound.
    memory_budget_gb: float | None = 6.0
    # Returns decoded at a time (about 150 bytes each while being filtered).
    lidar_chunk_points: int = 2_000_000
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
    # Optional per-polygon installation dates in the polygon file, from permits
    # or an interconnection record. Each stands in where the LiDAR cannot say
    # whether the array was there when it was flown:
    #   installed_by_column: a date by which the installation was complete (an
    #     inspection sign-off). On or before the LiDAR date, the row's basis is
    #     panel_by_install_date.
    #   not_installed_before_column: a date before which it did not exist (a
    #     first permit). After the LiDAR date, the row's basis is
    #     surface_before_install: the fitted plane is the roof without the array.
    # Height above the roof, where it can be measured, still wins over both.
    installed_by_column: str | None = None
    not_installed_before_column: str | None = None
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
    free_standing: FreeStandingConfig = Field(default_factory=FreeStandingConfig)
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
