# Changelog

## Unreleased — 0.3.0

Two parts. A refactoring pass under a no-change-in-results rule, checked
against regression benchmarks (a synthetic study area in CI; 200 real Phoenix
and Delaware polygons locally). Then the accounting, safety and usability
stories of milestone 0.3. Measured values are identical to 0.2.0 throughout:
on the Phoenix test block all 3,390 rows match in every measured column.

### Added

- **Every input polygon has exactly one row.** New `status` column: `measured`,
  `no_fit`, `no_lidar_tile`, `outside_tile_index`, `tile_unreadable`,
  `invalid_geometry`. Polygons that reach no tile-group task are written to
  `part-unmeasured.parquet` with their identity, geometry, area and vintage.
  `geometry_basis` gains `not_measured`.
- **`fit_failure`** says why a covered polygon has no fit: `too_few_points`,
  `ground_level_only` or `no_consensus`.
- **Input screening.** Unusable features (null, empty, non-polygonal, zero
  area) are kept as `invalid_geometry` rows instead of being dropped. Invalid
  polygons are repaired and flagged `geometry_repaired`. New flags
  `below_min_area`, `overlaps_polygon`, `duplicate_geometry`
  (`polygons.min_area_m2`, `polygons.overlap_flag_frac`).
- **Schema version** (`0.3`) in every partition and the manifest, with a
  compatibility rule; older outputs are upgraded when read.
- **Safe resume.** Each partition records the schema version, the measurement
  configuration and a fingerprint of the plan; `--resume` refuses to continue
  an output made from something else (`--force-resume` overrides). Changing
  only the cluster shape is allowed.
- **Progress** per finished tile group: groups done, rows, failures, elapsed,
  estimated time left.
- **Typed errors with remedies** (`pv_geom.errors`): missing CRS, non-metric
  CRS, id column problems, unparseable vintages, tile-index mismatches, missing
  optional packages, storage credentials. The CLI prints the problem and the
  fix, not a traceback.
- **Optional extras.** `pip install pv-geom` no longer pulls cloud packages;
  `[cloud]` adds S3 access and `[coiled]` the Coiled backend.
- Report: status and fit-failure tables in the coverage section;
  `status_definitions.csv` in the release dataset.

### Fixed

- `configs/phoenix.yaml` declared the polygon vintage as 2024-04-01. The
  detections come from imagery acquired September-October 2024; the config now
  says `2024-10`, which puts the gap to the LiDAR at about 3.9 years, not 3.3.
- An interrupted S3 download could leave a truncated file that looked like a
  cached tile; downloads now land under a temporary name first.
- A tile missing from a *local* LiDAR folder is treated as missing
  (`no_lidar_tile`) instead of failing its group.

### Changed

- **Columns that were non-null are now nullable** (`geometry`, `area_m2`,
  `aspect_ratio`, `n_points_panel`, `n_inliers_panel`, `point_density`,
  `n_planes_detected`), because unmeasured rows have no value for them.
- The coverage funnel counts from `status`; a MultiPolygon is exploded before
  validity is judged, so parts sharing an edge stay separate arrays.

- **Pipeline split.** `run_pipeline` (449 lines) is now `plan` / `executor` /
  `sink` / `vintage_check` behind a ~100-line orchestrator. `_build_row` (224
  lines, 17 parameters, untyped dict) is now `measure_polygon` over typed
  `PolygonTask` / `LocalPoints` / `Measurement` records, with `fit_panel`,
  `roof_reference` and `screen_standoff` as separate steps and one module that
  converts to schema rows. `extract_roof_plane` is split into footprint
  selection, ring gathering and classification.
- **Module moves**: `pipeline.tile_task` -> `pipeline.worker`
  (`_build_row` -> `build_row`); `io._localize` -> `io.storage`; `classify` ->
  `experimental.mounting`; the report builder is split into `data`, `text`,
  `tables`, `render`, `export`.
- **One vector reader** (`io.vector.read_vector`) for polygons, footprints and
  the tile index; **one summary** (`summary.summarise_table`) for the manifest
  and `describe-output`.
- **Logging instead of `print`.** Readable console progress, `--quiet`,
  `--verbose`, and a JSON-lines `logs/run.jsonl` beside local outputs.
- **Config**: Coiled `region` and `package_source` are settings, not constants;
  `uncertainty_method` is `bootstrap | none`; mounting thresholds no longer
  affect the config hash while the classifier is off. **The config hash
  changes.**
- `version` and `inspect-tile --json` print plain text.

### Removed

- Unused `io.s3` config block; unused `scikit-learn` dependency and
  `[whitebox]` extra; dead `quality.py`.
- One-off spikes and benchmarks moved to `scripts/archive/`; `HANDOFF.md` to
  `docs/history/`.

### Quality

- Type errors 88 -> 0; lint findings 26 -> 0; tests 254 -> 277; line coverage
  83% -> 91%. CI now runs lint, types and tests with a coverage gate on Linux
  and Windows (it previously failed its own lint step).

## 0.2.0 — 2026-10-07

The package's job is restated: take a polygon set and a LiDAR point cloud, each
with a capture date, and produce datasets, figures and a report characterising
the geometry of the installations. Mounting-type classification is archived.

### Added

- **Vintage as a run input.** `--polygon-vintage` (or a per-polygon
  `--polygon-vintage-col`) and `--lidar-date`. When the LiDAR date is not
  declared it is measured for every tile from per-point GPS time.
- **Per-row vintage columns**: `polygon_vintage`, `lidar_date`,
  `lidar_date_source`, `vintage_gap_days`, and **`geometry_basis`**
  (`panel_confirmed` / `panel_by_vintage` / `surface_unresolved` / `unscreened`
  / `no_fit`), which states what each row's tilt and azimuth describe.
- **`pv-geom report`**: summary tables (CSV), publication figures (PNG 300 dpi +
  PDF + SVG), a self-contained HTML report, a Markdown report, a filled-in
  methods paragraph, `summary.json`, and a consolidated release dataset with a
  data dictionary. Statistics are given per polygon and weighted by array
  surface, for all fitted rows and per geometry basis. Works on pre-0.2.0
  outputs.
- **`pv-geom inspect-tile`** (was a stub): classes, density, CRS, units and true
  flight dates of a LAZ tile. **`pv-geom describe-output`** (was a stub).
- **Footprints are optional.** Without them the roof reference comes from an
  open ring; `roof_ref_source` records which kind each row used. With them, a
  polygon that misses every footprint now also gets an open-ring reference.
- New columns `input_row`, `surface_area_m2`, `point_density`,
  `roof_ref_source`.
- `configs/delaware.yaml`.

### Changed

- **Mounting classification is off by default** (`mounting_rules.enabled:
  false`). `mounting_type`, `mounting_confidence` and `mounting_rule` are no
  longer in the output schema, and the `tracker_suspected` and
  `possible_missing_footprint` flags are no longer emitted, unless it is
  enabled.
- **Output is valid GeoParquet** (it was plain Parquet with a WKB column).
- **Polygons** may be any vector format GeoPandas reads and need no id column.
- **`crs.target` defaults to `auto`** (taken from the tile index) instead of
  Phoenix's EPSG:6341. Non-metric CRSs are refused rather than silently
  mis-measured.
- **Panel candidates without a building class** are returns above *local*
  ground (a 5 m elevation grid), not above one tile-wide median, so the cutoff
  holds on sloping terrain.
- **Speed and memory.** Points are binned into a spatial grid once per tile
  group, so per-polygon cost no longer scales with the tile; tiles are
  class-filtered as they are read.
- The tile-index id column is auto-detected (`Name`, `NAME`, …).
- `on_building` is nullable (null = no footprint layer).
- `vintage.input_epoch` is renamed `vintage.polygon_vintage`; the old name still
  loads.
- Lands the 2026-08-03 roof-fit work: ring-specific consensus floor, collar
  guard, `roof_no_consensus` / `roof_complex` split, `standoff_unscreenable`.

### Removed

- Config keys that were validated and hashed but never read: the `output:`
  block and `heights.use_whitebox_dem`. Configs that still carry them load.

### Compatibility

- The config hash changes. Outputs from 0.1.0 cannot be resumed into.
- Rows written by 0.1.0 lack the vintage columns; `pv-geom report` re-derives
  `geometry_basis` for them from `height_above_roof_m` and the dates you pass.

## 0.1.0 — 2026-07-29

First full Phoenix atlas run (349,233 rows). See `STATUS.md`.
