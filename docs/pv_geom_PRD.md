# PRD: `pv_geom` — Geometric Attribute Extraction for Solar PV Polygons from LiDAR

**Status:** Draft v0.1
**Owner:** Job Taminiau, FREE
**Target implementer:** Claude Code
**Last updated:** 2026-05-01

---

## 1. Overview

`pv_geom` is a standalone Python package that enriches a vector inventory of solar PV installation polygons with geometric attributes derived from co-temporal classified LiDAR. Given hundreds of thousands of PV polygons and a multi-terabyte LAZ tile archive on S3, the package produces a polygon-keyed attribute table containing tilt, azimuth, mounting type, roof orientation, panel–roof angle, and quality/provenance metadata.

The package is designed to run on a Coiled-managed Dask cluster, with a single-machine fallback for development. It is metro-agnostic: the first deployment target is Metropolitan Phoenix (~100k+ polygons; ~15–18 pts/m² LiDAR), but the pipeline must accept any classified LAZ archive with a sidecar tile index and any polygon set with a unique ID.

### 1.1 Why this exists
Existing PV inventories (NREL OpenPV, Stanford DeepSolar, etc.) provide presence/absence and rough capacity estimates. None systematically resolve installation geometry at scale. Geometry unlocks (a) mounting-type classification, (b) tilt/azimuth-aware yield modeling, (c) shading and expansion-headroom analysis, and (d) much sharper equity and policy questions when fused with parcel and demographic data.

### 1.2 Relationship to other FREE codebases
- Depends on utilities already proven in `pv_mapper` (LAZ I/O, tile indexing, Coiled config patterns) where they exist. Reuse, do not re-implement.
- Outputs land in the FREE Research Data Commons under a versioned prefix.
- Designed to be releasable independently (PyPI + Zenodo DOI), supporting standalone methods publication.

---

## 2. Goals and Non-goals

### 2.1 Goals (v1)
1. Extract per-polygon **tilt** (deg from horizontal) and **azimuth** (deg, 0=N, clockwise) from LiDAR returns.
2. Extract per-polygon **roof plane orientation** (tilt, azimuth) from LiDAR returns in a building-constrained ring buffer.
3. Compute **panel–roof angle** (angle between panel and roof normals).
4. Classify **mounting type** via a configurable rules engine: `flush_mount_rooftop`, `tilted_rack_rooftop`, `ground_mount_fixed`, `ground_mount_tracker_suspected`, `carport`, `ambiguous`.
5. Carry rich quality and provenance fields per row.
6. Scale to several TB of LAZ and 10⁵–10⁶ polygons via Dask/Coiled.
7. Be metro-agnostic and reproducible (config-driven, version-pinned).

### 2.2 Non-goals (v1)
- Capacity (kW DC) and yield (kWh) estimation. (Separate downstream module behind a feature flag in a future PRD; will use pvlib + NSRDB.)
- Module count from imagery.
- Array geometry beyond what falls out of plane fitting (row spacing / GCR explicitly deferred).
- A learned mounting-type classifier. (v1 is rules-based; the interface must allow swap-in later.)
- Detecting and correcting tracker positions to a canonical state. (v1 flags trackers; does not normalize them.)
- Re-segmentation or correction of input PV polygons.

---

## 3. Inputs

### 3.1 PV polygons
- **Format:** GeoParquet.
- **Required fields:** unique `polygon_id` (string or int64), geometry (Polygon or MultiPolygon).
- **Optional fields preserved through pipeline:** any column not in the reserved output namespace is passed through unchanged.
- **CRS:** arbitrary; pipeline reprojects to LiDAR CRS once at ingest.

### 3.2 LiDAR archive
- **Location:** S3 prefix, e.g. `s3://asu-nsf-phoenix/data/lidar_data/`.
- **Format:** Classified LAZ tiles (ASPRS classes; minimally class 2 = ground, class 6 = building used by this package).
- **Sidecar tile index:** GeoParquet, GeoPackage, or shapefile of tile-bounding polygons with at minimum a `tile_path` (or `filename`) column resolvable to an S3 URI and a CRS that matches the LAZ data. The implementation must auto-detect index format.
- **Point density:** any; a configured minimum (default 4 pts/m² per polygon) is required to attempt a fit.

### 3.3 Building footprints
- **Format:** GeoParquet, GeoPackage, or shapefile.
- **Required fields:** `building_id` (or auto-generated), geometry. CRS arbitrary; reprojected to LiDAR CRS at ingest.
- **Use:** (a) determining whether a PV polygon is "on a building" for mounting classification; (b) constraining the ring buffer used to extract the underlying roof plane.

### 3.4 Configuration
A YAML file validated by Pydantic models. See §9.

---

## 4. Outputs

### 4.1 Primary output
A partitioned GeoParquet table written to a versioned output prefix, keyed on `polygon_id`. Stored separately from the input polygons (cleaner provenance; trivial to join). Schema in §8.

### 4.2 Run manifest
A JSON sidecar at the output prefix root recording: package version, config hash, full resolved config, input URIs, run timestamp (UTC), Coiled cluster spec, count of polygons attempted/succeeded/failed, aggregate quality stats, and a list of tiles touched.

### 4.3 Logs
Structured (JSON-lines) logs written to a `logs/` subdirectory of the output prefix. One file per Dask worker.

---

## 5. Architecture

### 5.1 Dataflow
```
[polygons.parquet]            [building_footprints]            [LiDAR S3 + tile index]
        │                              │                                  │
        ▼                              ▼                                  ▼
   reproject ────────────────► reproject ──────────────────► load tile index
        │                              │                                  │
        └──────────────┬───────────────┘                                  │
                       ▼                                                  │
              spatial join polygons ↔ tiles ◄─────────────────────────────┘
                       │
                       ▼
        partition: group polygons by tile (with edge handling)
                       │
                       ▼
       ┌───────────────────────────────────────────────┐
       │  Dask: per tile-group (parallel)              │
       │    1. Fetch LAZ from S3                       │
       │    2. For each polygon in group:              │
       │       a. Clip points (with erosion)           │
       │       b. Fit panel plane (RANSAC + LSQ)       │
       │       c. Detect multi-plane / tracker         │
       │       d. Extract roof plane (ring buffer)     │
       │       e. Compute height-above-ground          │
       │       f. Apply mounting rules                 │
       │       g. Build output row                     │
       │    3. Write partition parquet                 │
       └───────────────────────────────────────────────┘
                       │
                       ▼
       merge tile-edge duplicates (polygons spanning tiles)
                       │
                       ▼
       write final table + manifest + logs
```

### 5.2 Tile-centric vs polygon-centric
Tile-centric. LAZ I/O dominates cost; loading each tile once and processing all overlapping polygons amortizes that cost. Polygons whose geometry spans multiple tiles are processed once per relevant tile, and the per-tile point sets are merged before fitting. The merge step happens in-process during the per-polygon fit (load-then-clip-from-each-tile pattern), not as a separate Dask stage, to avoid shuffling raw point clouds.

### 5.3 Compute model
- Dask + Coiled. Cluster spec lives in config. Worker image must include `pdal`, `laspy[lazrs]`, `geopandas`, `shapely`, `pyproj`, `scikit-learn`, `numpy`, `pyarrow`, `fsspec`, `s3fs`, `whitebox`, `pyogrio`, `pydantic`.
- Single-machine mode: the same code runs against `dask.distributed.LocalCluster` when `--local` is set.
- Per-task memory budget: tiles can be hundreds of MB; assume worker memory ≥ 4 GB and stream points where possible (PDAL streaming pipelines preferred over loading entire tile arrays for very large tiles).

### 5.4 LiDAR I/O
- Primary path: PDAL pipelines (`readers.las` → `filters.crop` → `writers.numpy`-equivalent in-memory). PDAL handles LAZ natively and supports cloud-optimized reads.
- Fallback: `laspy` with `lazrs` backend.
- Implementation MUST cache tile reads within a single worker for the lifetime of a tile-group task.

### 5.5 WhiteboxTools usage
Optional and config-gated. Two use cases:
1. **Per-tile DEM/DSM rasterization** for height-above-ground, when a project wants raster-based heights instead of point-based. Default off; point-based heights from class-2 returns are sufficient and simpler.
2. **Time-in-daylight rasters** are out of scope for v1 (deferred to the capacity/yield module).

If the user enables WBT, the implementation reuses the patterns established in `pv_mapper`.

---

## 6. Module breakdown

Package layout (src layout, `pyproject.toml`):

```
pv_geom/
├── pyproject.toml
├── README.md
├── src/pv_geom/
│   ├── __init__.py
│   ├── cli.py                  # Typer CLI
│   ├── config.py               # Pydantic settings models
│   ├── io/
│   │   ├── polygons.py         # read/reproject PV polygons
│   │   ├── footprints.py       # read/reproject building footprints
│   │   ├── tile_index.py       # load + auto-detect tile index format
│   │   └── lidar.py            # PDAL/laspy wrappers, S3 fetch, point-cloud clipping
│   ├── geometry/
│   │   ├── plane_fit.py        # RANSAC + LSQ plane fitting; tilt/azimuth from normal
│   │   ├── multi_plane.py      # multi-plane detection + tracker heuristic
│   │   ├── roof_plane.py       # ring-buffer roof plane extraction
│   │   └── heights.py          # height-above-ground from class-2 returns
│   ├── classify/
│   │   ├── rules.py            # rules-based mounting classifier
│   │   └── interface.py        # ABC for swappable classifiers
│   ├── pipeline/
│   │   ├── partition.py        # spatial join polygons↔tiles, group, edge handling
│   │   ├── tile_task.py        # per-tile-group processing function
│   │   └── runner.py           # top-level orchestrator (Dask graph construction)
│   ├── schema.py               # output schema (pyarrow + Pydantic)
│   ├── quality.py              # quality flags + confidence scoring helpers
│   ├── provenance.py           # manifest writer, config hashing
│   └── utils/
│       ├── crs.py
│       └── logging.py
├── configs/
│   ├── default.yaml
│   ├── phoenix.yaml
│   └── coiled.yaml
├── tests/
│   ├── unit/
│   │   ├── test_plane_fit.py   # synthetic planes
│   │   ├── test_multi_plane.py
│   │   ├── test_roof_plane.py
│   │   ├── test_rules.py
│   │   └── test_partition.py
│   └── integration/
│       └── test_phoenix_subset.py
└── examples/
    └── phoenix_quickstart.ipynb
```

### 6.1 Key function signatures

```python
# geometry/plane_fit.py
@dataclass(frozen=True)
class PlaneFit:
    normal: np.ndarray          # unit vector, len 3
    centroid: np.ndarray        # len 3
    tilt_deg: float             # 0 = horizontal
    azimuth_deg: float          # 0 = N, clockwise; NaN if tilt < tilt_floor
    rmse: float                 # m
    n_inliers: int
    n_total: int
    inlier_mask: np.ndarray     # bool, len n_total

def fit_plane_ransac(
    points: np.ndarray,         # (N, 3)
    ransac_threshold: float = 0.05,   # m
    min_inlier_frac: float = 0.6,
    max_iter: int = 200,
    refine_with_lsq: bool = True,
    tilt_floor_deg: float = 1.0,      # below this, azimuth is NaN
) -> PlaneFit: ...

# geometry/multi_plane.py
def detect_multi_plane(
    points: np.ndarray,
    primary: PlaneFit,
    cfg: MultiPlaneConfig,
) -> MultiPlaneResult: ...
# returns primary + optional secondary plane(s) + flag

# geometry/roof_plane.py
def extract_roof_plane(
    pv_polygon: shapely.Polygon,
    building_footprints: gpd.GeoDataFrame,   # spatially indexed
    points_class6: np.ndarray,
    cfg: RoofPlaneConfig,
) -> Optional[PlaneFit]: ...
# Returns None if (a) polygon not over building, (b) ring buffer too small,
# (c) too few class-6 points, or (d) RMSE > threshold.

# classify/rules.py
def classify_mounting(features: MountingFeatures, cfg: MountingRulesConfig) -> MountingResult: ...
# MountingResult includes label, confidence (0-1), and triggered_rule (str)

# pipeline/tile_task.py
def process_tile_group(
    tile_ids: list[str],
    polygons: gpd.GeoDataFrame,    # already filtered to this group
    footprints: gpd.GeoDataFrame,  # spatially filtered to tile bbox
    cfg: PVGeomConfig,
) -> pa.Table: ...
```

### 6.2 Edge handling for tile-spanning polygons
`pipeline/partition.py` assigns each polygon to a *primary* tile (the tile containing the polygon centroid) and records all *overlapping* tile IDs. The per-tile-group worker fetches points from all overlapping tiles before clipping to the polygon. Output rows are emitted only by the primary-tile worker, eliminating duplicates without a separate dedup pass. This requires the partitioner to ensure all overlapping tiles for a polygon are co-scheduled in the same task — the simplest approach is to expand each task's tile list to the union of tiles overlapped by any polygon assigned to it.

---

## 7. Algorithms

### 7.1 Plane fitting (panel)
1. Clip class-6 (building) and unclassified returns to the polygon, eroded inward by `panel_erosion_m` (default 0.15 m) to suppress frame and racking edge returns.
2. If `n_points < min_points_panel` (default `max(50, ceil(0.5 * polygon_area_m2 * min_density))`), emit NaN with `flag = low_density`.
3. Run RANSAC plane fit with `ransac_threshold` (default 0.05 m) and `max_iter` (default 200).
4. If `n_inliers / n_total < min_inlier_frac` (default 0.6), emit NaN with `flag = poor_fit`.
5. Refine on inliers via least-squares (PCA → smallest eigenvector).
6. Compute tilt = `acos(|n_z|)`; azimuth = `atan2(n_x, n_y)` mapped to [0, 360); below `tilt_floor_deg` (default 1.0), azimuth is NaN with `flag = near_horizontal`.

### 7.2 Multi-plane detection
After the primary fit:
1. Examine residuals of *non-inliers*. If they form a coherent secondary cluster (DBSCAN in residual-vs-position space, or a second RANSAC pass on residuals exceeding `2 * ransac_threshold`) covering ≥ `secondary_min_frac` (default 0.2) of points and yielding RMSE < threshold, record secondary plane.
2. If two planes have azimuths roughly 180° apart and similar tilts, set `flag = east_west_rack`.
3. **Tracker heuristic** (utility-scale ground-mount): when a polygon is *not* on a building footprint, has an elongated aspect ratio (default `aspect > 4`), and neighboring polygons within `tracker_neighbor_radius_m` (default 50 m) show similar azimuths but variable tilts across the metro (computed in a post-pass), set `flag = tracker_suspected`. Implementation note: the post-pass requires a second Dask reduction; gate it behind `cfg.detect_trackers`. **For v1, a simpler per-polygon heuristic is sufficient: elongated, off-building, low height-above-ground, tilt < 35°. Mark as `tracker_suspected`. The robust spatial-clustering version is a v1.1 follow-up.**

### 7.3 Roof plane extraction
1. Locate the building footprint(s) that intersect the PV polygon. If none, return `None` and emit `on_building = False`.
2. Build a ring buffer: `pv_polygon.buffer(roof_buffer_m).difference(pv_polygon)` intersected with the building footprint, minus all *other* PV polygons within the same building.
3. Clip class-6 returns to the ring; require `n_points >= min_points_roof` (default 100). If insufficient (e.g., the PV occupies most of the roof), iteratively expand `roof_buffer_m` up to a configurable cap (default 5 m). If still insufficient, return `None` with `flag = roof_insufficient`.
4. Fit plane via RANSAC + LSQ as in §7.1.
5. Reject if RMSE > `roof_rmse_max` (default 0.10 m): roofs are sometimes complex (dormers, hips). Return `None` with `flag = roof_complex`.

### 7.4 Height-above-ground
Per polygon: median of (panel inlier z) − median of (class-2 z within `ground_search_radius_m`, default 10 m, around polygon centroid). Carried as `height_above_ground_m`.

Height-above-roof: median of (panel inlier z) − value of fitted roof plane at panel centroid (x, y). Carried as `height_above_roof_m`. Defined only when a roof plane was successfully fit.

### 7.5 Mounting rules (v1)
All thresholds in config. Rules evaluated in order; first match wins. Confidence = piecewise-linear function of margin from the deciding threshold(s).

```
features used:
  on_building: bool                  # PV polygon centroid in any footprint
  panel_tilt_deg: float
  panel_azimuth_deg: float
  panel_roof_angle_deg: float | NaN  # angle between panel and roof normals
  height_above_roof_m: float | NaN
  height_above_ground_m: float
  area_m2: float
  aspect_ratio: float                # long/short axis of min-rotated-rect
  roof_plane_available: bool

rules:
  R1 flush_mount_rooftop:
      on_building AND roof_plane_available
      AND panel_roof_angle_deg <= 5.0
      AND height_above_roof_m <= 0.5

  R2 tilted_rack_rooftop:
      on_building
      AND ((roof_plane_available AND panel_roof_angle_deg > 5.0 AND height_above_roof_m <= 1.5)
           OR (NOT roof_plane_available AND panel_tilt_deg > 5.0 AND height_above_ground_m > 2.5))

  R3 carport:
      NOT on_building
      AND height_above_ground_m >= 2.0
      AND aspect_ratio >= 2.0

  R4 ground_mount_tracker_suspected:
      NOT on_building
      AND height_above_ground_m < 2.0
      AND aspect_ratio >= 4.0
      AND panel_tilt_deg < 35.0

  R5 ground_mount_fixed:
      NOT on_building
      AND height_above_ground_m < 2.0
      AND panel_tilt_deg >= 5.0

  R6 (default): ambiguous
```

Confidence rules: each rule returns 1.0 when the input is ≥ `margin` past the threshold and 0.5 at the threshold itself, linearly. The `ambiguous` class returns confidence = 1.0 minus the best near-miss confidence.

The interface in `classify/interface.py` defines an ABC that takes `MountingFeatures` and returns `MountingResult`, so a learned classifier can be substituted later without touching the pipeline.

---

## 8. Output schema

Output: GeoParquet, partitioned by `partition_id` (a hashed bucket of `polygon_id`, ~100k rows per partition by default).

| field | type | nullable | description |
|---|---|---|---|
| `polygon_id` | string | no | passthrough from input |
| `geometry` | geometry | no | passthrough, in LiDAR CRS |
| `n_points_panel` | int32 | no | total returns clipped to polygon |
| `n_inliers_panel` | int32 | no | RANSAC + refinement inliers |
| `panel_tilt_deg` | float32 | yes | NaN on failure |
| `panel_azimuth_deg` | float32 | yes | NaN if near-horizontal or failure |
| `panel_rmse_m` | float32 | yes | inlier RMSE |
| `panel_tilt_unc_deg` | float32 | yes | bootstrap or covariance-derived 1σ |
| `panel_azimuth_unc_deg` | float32 | yes | bootstrap or covariance-derived 1σ |
| `n_planes_detected` | int8 | no | 1 or 2; 0 only if fit failed |
| `secondary_tilt_deg` | float32 | yes | when n_planes_detected = 2 |
| `secondary_azimuth_deg` | float32 | yes | when n_planes_detected = 2 |
| `roof_tilt_deg` | float32 | yes | NaN if no roof plane |
| `roof_azimuth_deg` | float32 | yes | NaN if no roof plane |
| `roof_rmse_m` | float32 | yes | |
| `panel_roof_angle_deg` | float32 | yes | NaN if no roof plane |
| `height_above_roof_m` | float32 | yes | NaN if no roof plane |
| `height_above_ground_m` | float32 | no | from class-2 returns |
| `on_building` | bool | no | |
| `building_id` | string | yes | from footprint dataset |
| `area_m2` | float32 | no | computed in projected CRS |
| `aspect_ratio` | float32 | no | min-rotated-rect long/short |
| `mounting_type` | string | no | one of the labels in §7.5 |
| `mounting_confidence` | float32 | no | [0, 1] |
| `mounting_rule` | string | no | which rule fired (R1..R6) |
| `flags` | list<string> | no | e.g. `[low_density, east_west_rack]` |
| `lidar_tile_ids` | list<string> | no | tiles contributing returns |
| `pkg_version` | string | no | e.g. `0.1.0` |
| `config_hash` | string | no | sha256 of resolved config |
| `run_id` | string | no | UUID per pipeline invocation |
| `partition_id` | int32 | no | partition assignment |

Pyarrow schema is the source of truth (`schema.py`); a Pydantic model mirrors it for in-memory validation.

---

## 9. CLI and configuration

### 9.1 CLI (Typer)
```
pv_geom run \
  --polygons s3://.../pv_polygons.parquet \
  --lidar-prefix s3://asu-nsf-phoenix/data/lidar_data/ \
  --tile-index s3://asu-nsf-phoenix/data/lidar_data/tile_index.parquet \
  --footprints s3://.../buildings.parquet \
  --output s3://free-data-commons/pv_geom/phoenix/v0.1/ \
  --config configs/phoenix.yaml \
  [--local]                  # use LocalCluster instead of Coiled
  [--max-polygons N]         # for dev/smoke runs
  [--bbox xmin ymin xmax ymax] # restrict to a region
  [--dry-run]                # plan only, write manifest, no compute
```

Additional commands: `pv_geom validate-config`, `pv_geom inspect-tile <tile_id>`, `pv_geom describe-output <output_uri>`.

### 9.2 Config (YAML, validated by Pydantic)
```yaml
crs:
  target: EPSG:6404           # NAD83(2011) / Arizona Central (ftUS) — confirm at runtime against tile index

panel_plane:
  erosion_m: 0.15
  ransac_threshold_m: 0.05
  min_inlier_frac: 0.6
  max_iter: 200
  min_density_pts_per_m2: 4
  tilt_floor_deg: 1.0
  uncertainty_method: bootstrap   # or covariance
  bootstrap_samples: 50

multi_plane:
  enabled: true
  secondary_min_frac: 0.20
  ew_rack_azimuth_tol_deg: 25
  ew_rack_tilt_tol_deg: 5

roof_plane:
  enabled: true
  buffer_m: 3.0
  buffer_max_m: 5.0
  min_points: 100
  rmse_max_m: 0.10

heights:
  ground_search_radius_m: 10.0
  use_whitebox_dem: false

mounting_rules:
  R1: { panel_roof_angle_deg_max: 5.0, height_above_roof_m_max: 0.5 }
  R2: { panel_roof_angle_deg_min: 5.0, height_above_roof_m_max: 1.5,
        fallback_tilt_deg_min: 5.0, fallback_height_above_ground_m_min: 2.5 }
  R3: { height_above_ground_m_min: 2.0, aspect_ratio_min: 2.0 }
  R4: { aspect_ratio_min: 4.0, tilt_deg_max: 35.0, height_above_ground_m_max: 2.0 }
  R5: { tilt_deg_min: 5.0, height_above_ground_m_max: 2.0 }
  confidence_margin: 0.5     # fractional margin for full confidence

io:
  lidar_reader: pdal          # or laspy
  s3:
    requester_pays: false
    region: us-west-2

compute:
  backend: coiled            # or local
  coiled:
    name: pv-geom
    n_workers: 40
    worker_memory: "8GiB"
    worker_cpu: 4
    software: pv-geom-runtime-2026-05
  local:
    n_workers: 8
    threads_per_worker: 2

output:
  partition_size: 100000
  write_geoparquet: true
  also_write_csv: false
```

---

## 10. Quality, provenance, reproducibility

- **Determinism:** RANSAC seed taken from `polygon_id` hash so re-runs on the same input/config produce identical outputs.
- **Config hash:** sha256 of the canonicalized resolved config; written to every row and the manifest.
- **Versioning:** semantic version in `pyproject.toml`; written to every row. Output prefix conventionally includes the version (`.../phoenix/v0.1/`).
- **Idempotency:** the runner accepts a `--resume` flag that detects already-written partitions and skips them.
- **Aggregate stats** in the manifest: counts by mounting type, distribution of confidences, tile coverage, fraction of polygons with each flag.

---

## 11. Testing

### 11.1 Unit
- `test_plane_fit.py`: synthesize planar point clouds at known tilt/azimuth with controlled noise; assert recovered values within tolerance. Include degenerate cases (collinear points, near-horizontal, sparse).
- `test_multi_plane.py`: synthesize east-west racks and nested planes; assert detection and primary/secondary assignment.
- `test_roof_plane.py`: synthesize a hip roof + a panel; assert ring-buffer extraction recovers the surrounding facet, not the panel.
- `test_rules.py`: parametrize with feature dicts spanning all rule branches; assert correct labels and monotone confidence.
- `test_partition.py`: synthetic polygons and tile index; assert correct primary-tile assignment and overlap union; assert no duplicate output rows for tile-spanning polygons.

### 11.2 Integration
- `test_phoenix_subset.py`: small fixed bbox in Phoenix (~50 polygons, 2–4 tiles), runs against real S3 (gated by env var so CI can skip), compares outputs to a checked-in reference parquet. Tolerances on float fields documented.

### 11.3 Performance smoke
- A `pv_geom benchmark` command that runs against a fixed 1k-polygon sample and reports wall time, peak memory, and per-polygon cost. Tracked in CI history.

---

## 12. Milestones

Designed for incremental Claude Code implementation. Each milestone is independently mergeable.

**M1 — Scaffolding (½ day)**
Package layout, `pyproject.toml`, config models with validation, CLI skeleton with `validate-config`, logging, CI.

**M2 — I/O layer (1 day)**
Polygon reader/reprojector, footprint reader, tile-index loader (auto-detect), LAZ reader (PDAL primary, laspy fallback), S3 fetch with caching. Unit tests on small synthetic inputs.

**M3 — Plane fitting (1 day)**
`fit_plane_ransac`, tilt/azimuth derivation, uncertainty (bootstrap), unit tests on synthetic planes.

**M4 — Roof plane + heights (½ day)**
Ring-buffer construction, roof plane extraction, height-above-ground/roof helpers, unit tests.

**M5 — Multi-plane + mounting rules (1 day)**
Multi-plane detection (east-west rack), tracker-suspected heuristic (per-polygon v1), rules engine with confidence, unit tests.

**M6 — Pipeline orchestration (1 day)**
Spatial join polygons↔tiles, partitioning with edge handling, `process_tile_group`, Dask graph construction, manifest writer, partition-level parquet output. Unit tests for partitioner; smoke test on synthetic LAZ.

**M7 — Coiled integration + Phoenix integration test (½–1 day)**
Coiled cluster spec, software environment definition, integration test against a small Phoenix bbox with reference output.

**M8 — Hardening (ongoing)**
Resume flag, benchmark command, README + quickstart notebook, Zenodo metadata for first tagged release.

Total greenfield estimate: ~5–7 working days for Claude Code with focused supervision.

---

## 13. Open questions to resolve before M7
- Confirm exact LiDAR CRS from the tile index (assume EPSG:6404 ftUS for Phoenix; verify).
- Confirm Coiled software environment name and whether to pin to `pv_mapper`'s environment or build a fresh one.
- Decide whether to require `building_id` in the footprint input or auto-generate (default: auto-generate if absent).
- Confirm output prefix convention in the FREE Data Commons (e.g. `s3://free-data-commons/pv_geom/<metro>/<version>/`).

---

## 14. Out-of-scope reminders for the implementer
Do not add capacity (kW) or yield (kWh) estimation. Do not add module-count estimation. Do not modify input PV polygons. Do not implement a learned mounting classifier. These are explicitly v2+.
