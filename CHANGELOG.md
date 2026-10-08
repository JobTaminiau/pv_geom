# Changelog

## Unreleased — 0.6.0

### Added — documentation site (I5)

- `docs/guide/`: a tutorial on the bundled demo, how-to guides (a new study
  area, running in the cloud, reports, validation, reproducing a run), concept
  pages (vintage and basis, how it measures, limits), a FAQ, and reference
  pages for columns, status/basis/flags, configuration, the command line and
  the Python API.
- The reference pages are **generated from the code** by
  `scripts/build_docs.py`; a test fails if they are out of date. Other tests
  check that every command and option the pages show exists, that every column
  they name exists, that links resolve, and that the tutorial's stated results
  are what the demo produces.
- `mkdocs build --strict` runs in CI. The site is not published anywhere yet.
- CI now really runs on both Python versions (the version was being passed to
  the wrong setting).

### Added — reproducibility (G5)

- Every manifest now has a `reproducibility` block: a **content hash** of the
  rows (independent of run id, partitioning, row order and package version),
  the **inputs** read (size and SHA-256 of the polygon and footprint layers;
  names and sizes of the LiDAR tiles), and the **environment** (Python,
  platform, library and PROJ versions).
- `pv-geom verify OUTPUT` checks an output against its manifest;
  `--against OTHER` compares two outputs and says which columns differ, in how
  many rows and by how much.
- `pv-geom reproduce OUTPUT --out NEW` reruns a finished run from its manifest
  and compares, noting environment differences and inputs that have changed.
- **What is promised.** On one machine and environment: bit for bit (equal
  content hashes). Across platforms: every row, label and count identical and
  every number within 1e-4; measured in CI between Linux and Windows the
  largest differences are 1.5e-5 degrees and 1e-8 metres. Bit-for-bit across
  platforms, which the specification asked for, is not attainable:
  linear-algebra libraries differ in their last bits by platform and processor
  (two Windows CI machines differed from each other).
- CI enforces both on the synthetic benchmark.

### Changed — column names (schema 0.6)

The measured columns were called `panel_*`, but what is measured is the surface
inside the polygon, which `geometry_basis` may or may not establish as a panel.
They now say what they are, and match the names the facet table already used.

| Was | Is |
| --- | --- |
| `panel_tilt_deg`, `panel_azimuth_deg` | `tilt_deg`, `azimuth_deg` |
| `panel_tilt_unc_deg`, `panel_azimuth_unc_deg` | `tilt_unc_deg`, `azimuth_unc_deg` |
| `panel_rmse_m`, `panel_fit_tolerance_m` | `fit_rmse_m`, `fit_tolerance_m` |
| `panel_rival_share`, `panel_rival_angle_deg` | `fit_rival_share`, `fit_rival_angle_deg` |
| `n_points_panel`, `n_inliers_panel` | `n_points`, `n_inliers` |
| `panel_roof_angle_deg` | `angle_to_roof_deg` |
| `n_planes_detected` | `n_facets` |
| `segments` (and `segment_index`, `rmse_m` within it) | `facets` (`facet_index`, `fit_rmse_m`) |
| `pv_geom_segments.parquet` / `.csv`, `segment_id` | `pv_geom_facets.parquet` / `.csv`, `facet_id` |
| flag `no_panel_standoff` | flag `no_standoff` |

- **Old outputs still work.** Anything written under the old names is renamed
  as it is read, so `pv-geom report` and `pv_geom.load` need nothing from you.
- **Old code can ask for the old names**: `pv_geom.load(output,
  legacy_names=True)` adds them back as copies. This is for one release and
  goes in 1.0. The mapping is `pv_geom.schema.RENAMED_COLUMNS`.
- Unchanged on purpose: the `geometry_basis` values `panel_confirmed` and
  `panel_by_vintage` (they do assert a panel), the `roof_*` and `secondary_*`
  columns, and configuration keys such as `panel_plane:`.

## 0.5.0

Measurement depth, and a first outside check. Schema version 0.5.

### Changed — results

- **Phoenix runs on the dissolved detection layer** (435,295 overlap clusters,
  id `cluster_id`) instead of the raw one, which counted an array twice wherever
  two image composites both detected it.
- **Many more arrays get a roof reference.** Where the ring around an array has
  no dominant plane (hip and cross-gable roofs), its facets are separated and the
  one beside the array is used; where even that band is split, the facet parallel
  to the array plane. `unscreened` falls from 39% to 7% of fitted polygons on the
  Phoenix benchmark and from 42% to 8% on Delaware. `roof_ref_method` says how
  each reference was chosen.
- **Polygons covering more than one roof face are measured as segments.** A
  polygon with two faces and no dominant one used to be a `no_fit`. The
  `panel_*` columns describe the largest face; `segments` holds every face;
  `n_planes_detected` counts them; flag `multi_facet`.
- **`envelope_fit` flag.** A wide-tolerance fit on a near-flat plane is the
  signature of rows of tilted modules on a flat roof: the plane is the envelope
  of the rows, not the modules. Found by external validation (below).

### Added

- **`recommended` column**: measured, on a panel basis, and not flagged sparse,
  undersized, double-counted or an envelope fit. The rule is provisional and is
  written into the dataset metadata.
- **External validation**: `pv-geom compare-reference` / `pv_geom.compare_reference`
  compare a run with reported geometry, mount against facet, honouring the
  reference's resolution and whether the array existed at the LiDAR date.
  `scripts/pvdaq_references.py` and `scripts/uspvdb_references.py` build the
  reference tables. See `docs/validation.md`.
- **A first accuracy set**, `validation/pvdaq_documented/`: PVDAQ's documented
  systems measured on public 3DEP LiDAR with hand-drawn outlines, built by
  `scripts/accuracy_set.py`. On the three clean sites tilt is within 0.1 degrees
  on roofs and 0.1 to 1.0 on ground rows; azimuth within 0.04 and 0.2 degrees.
- **Web Mercator is refused as the working CRS** (public point-cloud services
  deliver it; it reads as metric but is not true to scale).
- **Compare runs**: `pv-geom report A B --compare --out dir`.
- **Results by region**: `pv-geom report --regions layer --region-col name`.
- **LiDAR in feet or another CRS** is converted to the run CRS on read, and the
  conversion recorded in the manifest. It used to be refused.
- Release dataset: `pv_geom_segments.parquet` / `.csv`, one row per facet.
- `cluster_id` is recognised as a polygon id column; geometry collections keep
  their polygons.

- **Bounded worker memory (H5).** Tiles are decoded in chunks
  (`compute.lidar_chunk_points`) and a tile group whose kept returns exceed
  `compute.memory_budget_gb` is measured in spatial batches. Peak memory on a
  44-million-return Delaware tile falls from 3.9 GB to 0.34 GB. Rows are
  identical with and without batching.
- **Cloud settings are configuration (H4).** `compute.coiled.package_source`,
  `region`, `software` and `name` default to `auto`: the client's exact commit,
  the LiDAR bucket's region, an environment named after the dependency list,
  and the study name. A cloud run stops before starting a cluster if the
  checkout has uncommitted or unpushed changes, and after install if any worker
  runs a different version. New `compute.coiled.worker_threads`.

### Changed — results: the fit no longer depends on the random seed (E9)

Rerunning a benchmark with different seeds used to move tilt by a median of
0.1 to 0.5 degrees, by more than 0.5 degrees for 17 of 100 Phoenix and 32 of 65
Delaware polygons, and to change `geometry_basis` for about one in five. Over
five seeds it now moves nothing: tilt, azimuth, status and facet count are
identical for every polygon in all three benchmarks. The plane fit:

- tries **every point triple** when a polygon has up to about 120 returns, and
  many more random ones than before for mid-sized sets, so a second plane with
  one inlier fewer is no longer missed by chance;
- refits each of the best *distinct* candidates to its inliers **until the
  inlier set stops changing**, and picks the winner by inlier count, then
  residual, then orientation, never by draw order;
- finishes with a **smooth robust fit** (Tukey biweight), which has one optimum
  where the hard-threshold refit had several a fraction of a degree apart.

Consequences to know about:

- Every result changes slightly, and a few polygons lose their fit (Delaware
  benchmark 65 -> 63): the settled plane can hold fewer returns than the
  count-maximising one and fall under the consensus floor.
- Against the accuracy set the new fit is as close or closer (NREL roofs
  10.02, 10.02, 10.02, 10.00 degrees against a reported 10).
- Fitting takes about three times as long per polygon (about 0.25 s).
- `geometry_basis` can still differ for a polygon sitting on the standoff
  threshold (0 to 3 of 100 between seeds): the roof reference is fitted to much
  larger point sets by random search.
- New columns `panel_rival_share` and `panel_rival_angle_deg`, and flag
  `ambiguous_fit` (about 10% of fits): a different plane, 2 degrees or more
  away, fits at least 90% as many returns. The choice is reproducible but
  fragile. It is not excluded from `recommended`.
- `scripts/seed_sensitivity.py` measures this; `tests/benchmark` enforces it.

### Added — ground mounts and canopies can be confirmed (C7)

- New `geometry_basis` value **`free_standing`**: the fitted plane is an
  elevated structure standing in open ground. The standoff screen needs a roof
  under the array, so correctly measured canopies and ground rows could never
  be confirmed; in the USPVDB school case 0 of 51 canopy polygons were usable,
  and 50 now are.
- New column `open_ground_share`: ground returns as a share of all returns in a
  2 m band around the polygon (neighbouring PV polygons excluded). Measured:
  0.81 to 0.98 around ground-mounted rows, median 0.96 around parking canopies,
  0.00 to 0.05 around arrays on large roofs, under 0.35 for nine in ten
  residential roofs. The threshold is 0.6 (`free_standing.min_open_share`).
- `free_standing` counts toward the panel stratum in reports and toward
  `recommended`. It ranks below `panel_confirmed` and `panel_by_vintage`.
- It establishes the structure, not the modules. A carport that was roofed
  with panels after the LiDAR was flown also qualifies; its geometry is still
  the array's, since modules lie flush on such structures.
- The building-footprint layer is not used for this: canopies are in it as
  buildings.

### Changed — results (small)

- Spatial grids are anchored to absolute coordinates instead of the extent of
  the loaded points, so a polygon's result no longer depends on which other
  polygons share its tile group. Random sampling sees points in a different
  order than before, which moves results exactly as a change of random seed
  does: on the Phoenix benchmark a median of 0.03 degrees in tilt, but more
  than 0.5 degrees for about one polygon in eight, and a different
  `geometry_basis` for 13 of 100. That sensitivity was always there; it is now
  measured, and is story E9.

### Not in this release

Row tilt on flat commercial roofs and presence evidence for canopies are new
stories (E7, C7).

## 0.4.0

Usable from a config, by someone who is not us. Schema version 0.4.

### Changed — results

- **Azimuth is now measured from true north.** `panel_azimuth_deg`,
  `secondary_azimuth_deg` and `roof_azimuth_deg` were relative to the grid north
  of the LiDAR's projected CRS; they now have the meridian convergence at each
  polygon added (from PROJ), and a new `grid_convergence_deg` column records it.
  The shift is about −0.8° in the Phoenix test block (−0.3° to −1.0° across the
  atlas) and −0.1° to −0.4° in Delaware. The manifest states
  `azimuth_reference`; outputs from before this change are labelled grid north
  when reported. Tilt and every non-azimuth value are unchanged.

### Added

- **Config-driven runs.** `study:` and `inputs:` blocks name the output, the
  polygon layer, the LiDAR and optional footprints, with paths relative to the
  config file; `pv-geom run --config area.yaml` is a complete run and
  command-line options override the config. The Phoenix and Delaware configs
  carry their inputs.
- **Python API**: `pv_geom.run()`, `load()`, `report()`, `describe()`.
- **`pv-geom demo`** and `pv_geom.sample`: a generated study area with known
  geometry, measured and reported offline.
- **The tile index is optional**: built from the tiles' headers when absent.
- **Run-start class check**: reports which point class supplies array
  candidates; stops when the LiDAR has no ground class or no candidate class.
- **Estimate before a run**: tiles, gigabytes, rough time, and cost on Coiled
  when `compute.coiled.usd_per_worker_hour` is set; shown by `--dry-run` and
  stored in the manifest.
- **Report**: `--headline` and `--weight`; 95% bootstrap intervals on the median
  tilt and the facing shares; the study name comes from the config.
- **Self-describing release dataset**: `README.md`, `metadata.json`
  (repository-deposit fields; authors, licence and funding left empty) and
  `SHA256SUMS.txt`.

### Fixed

- `SHA256SUMS.txt` is written with LF line endings on every platform.

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
