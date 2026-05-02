# pv-geom — status

PRD: `docs/pv_geom_PRD.md` (v0.1, 2026-05-01).

## Milestones

- [x] **M1 — Scaffolding** (this commit): package layout, `pyproject.toml`, Pydantic config models with validation, CLI skeleton with `validate-config`, JSON-line logging, CI stub.
- [x] **M2 — I/O layer** (2026-05-01):
  - `io/polygons.py` — `read_polygons` (GeoParquet, accepts `polygon_id` or `detection_id`, **explodes MultiPolygons** by default into one row per part with `parent_polygon_id` link + `__p<i>` id suffix; bbox + max_polygons filters).
  - `io/footprints.py` — `read_footprints` (GeoParquet/GPKG/SHP auto-detect, normalizes FEMA's `build_id` → canonical `building_id`, auto-synthesizes `auto_<i>` ids when both absent).
  - `io/tile_index.py` — `load_tile_index` (auto-detects parquet/GPKG/SHP/zipped-SHP via `zip://` virtual paths) + `build_tile_uris` helper (composes per-row URIs from a base + name template; default fits the Phoenix flat USGS prefix).
  - `io/lidar.py` — `read_tile_points` (laspy primary, PDAL via `[pdal]` extra) + `clip_points_to_polygon` with optional inward erosion. S3 caching via `_localize.py`.
  - 26 new unit tests on synthetic inputs (parquet/GPKG/SHP round-trips, MultiPolygon explosion, id-normalization paths, bbox/max-polygons filters, zipped-SHP read, LAZ class filter, polygon clip with erosion).
  - Real-data smoke test confirms readers handle the actual atlas, FEMA AZ, USGS tile-index zip, and a 25.4M-point LAZ tile end-to-end.
- [x] **M3 — Plane fitting** (2026-05-01): `fit_plane_ransac` + `tilt_azimuth_from_normal` + `bootstrap_uncertainty` in `geometry/plane_fit.py`. 16 synthetic-plane unit tests pass (clean / noisy / 30%-outlier / collinear / sparse / determinism / bootstrap monotonicity). Real-data verification on the spike's high-RMSE outlier (polygon `21_841798_397654__1`): RANSAC dropped RMSE from 0.377 m (SVD-only) to 0.018 m by rejecting 14% overshoot points — visual aerial overlay confirmed they were polygon-overshoot into the yard.
- [x] **M4 — Roof plane + heights** (2026-05-01): `extract_roof_plane` (with iterative buffer expansion + ring construction subtracting other PVs on the same building), `height_above_ground`, `height_above_roof`, `panel_roof_angle_deg` in `geometry/roof_plane.py` and `geometry/heights.py`. 18 new unit tests (8 roof, 10 heights) covering off-building, ring-excludes-PV, ring-subtracts-other-PVs, buffer expansion, insufficient-points flag, complex-roof flag, multi-footprint disambiguation, plus height-helper edge cases. Real-data spike against FEMA AZ + cached LiDAR confirms behaviour end-to-end (tilted-rack panel fit at 22.8° + correct `roof_complex` flag on a multi-facet roof). New `RoofPlaneConfig` fields: `buffer_step_m`, `ransac_threshold_m`. AZ FEMA footprints sourced from `s3://free-research-data/national/fema_footprints/az.geoparquet` (FEMA's `build_id` maps to canonical `building_id`).
- [x] **M5 — Multi-plane + mounting rules** (2026-05-01):
  - `geometry/multi_plane.py` — `detect_multi_plane` (second-RANSAC on non-inliers + east-west azimuth check), `is_tracker_suspected` per-polygon heuristic, `polygon_aspect_ratio` helper (long/short axis of min-rotated-rect).
  - `classify/rules.py` — `RulesMountingClassifier` implementing PRD §7.5's R1–R6, with fractional `_conf_le`/`_conf_ge` confidence smoothing. `ambiguous` (R6) returns `1 - best_near_miss_confidence`.
  - 35 new unit tests (16 multi_plane + 19 rules) covering: aspect-ratio invariance under rotation, EW-rack detection (2:1 mix to clear primary inlier-frac), tracker heuristic happy/no paths, every rule's happy path, R3-beats-R4 ordering, threshold-half-confidence semantics, NaN inputs.
  - **End-to-end real-data chain**: polygon `21_840435_397729__1` flows through M3 (panel tilt 22.95°/az 234.7°/RMSE 2.1cm) → M4 (`roof_complex` flag) → M5 (R2 fallback fires `tilted_rack_rooftop` with confidence 1.000).
- [x] **M6 — Pipeline orchestration** (2026-05-01):
  - `pipeline/partition.py` — `assign_polygons_to_tiles` (primary tile by centroid + intersecting tiles list) + `build_tile_groups` (group by primary, expand fetch list to union of overlapping tiles).
  - `pipeline/tile_task.py` — `process_tile_group` integrates M2 I/O + M3 panel fit + M4 roof + heights + M5 multi-plane + mounting rules per polygon, applies the class-1/2 fallback when class 6 is absent, returns one row per polygon matching `OUTPUT_SCHEMA`.
  - `pipeline/runner.py` — `run_pipeline` reads inputs, partitions, dispatches via `dask.delayed` (or serial when `--no-dask`), writes per-partition GeoParquet + a JSON manifest with mounting-type counts, RMSE percentiles, flag counts, tiles touched.
  - CLI `pv-geom run` now functional with `--bbox`, `--max-polygons`, `--dry-run`, `--name-template`, `--tile-id-col`, `--no-dask`, `--local`.
  - 8 new tests (6 partition + 2 runner smoke on synthetic LAZ/polygon/footprint/tile-index). 140/140 in the suite pass.
  - **Real-data verification**: ran end-to-end against a 900x900 m Phoenix bbox, 5 polygons → clean panel fits (RMSE ~16 mm, azimuths all SSW). All 5 ended up `on_building=False` due to FEMA footprint gaps in this neighborhood (correctly surfacing a data issue, not an algorithm one).
- [x] **M7 — Coiled integration + Phoenix integration test** (2026-05-01):
  - `src/pv_geom/coiled_env.py` — `SOFTWARE_ENV = "pv-geom-2026-05"` constant + `CONDA_SPEC` (conda-forge: numpy/pandas/geopandas/shapely/pyproj/pyogrio/pyarrow/scikit-learn/dask/distributed/coiled/pdal/laspy/lazrs-python/pydantic/typer/rich); `ensure_software_env`, `make_cluster`, `install_pv_geom_on_workers` helpers (last one works around Coiled silently dropping `git+` pip URLs).
  - `pipeline/runner.py::_cluster_for(cfg)` context manager — yields a `Dask Client` from `LocalCluster` (`backend=local`) or `coiled.Cluster` (`backend=coiled`), or `None` for the default scheduler.
  - `tests/integration/test_phoenix_subset.py` — real-data end-to-end gated on both `RUN_INTEGRATION=1` and the presence of cached prerequisite files (atlas parquet, FEMA AZ, USGS tile index, cached LAZ). Asserts schema, succeeded count, southern-azimuth concentration, RMSE ≤ 10 cm, manifest sanity.
  - **Verified**: integration test passes in 88 s on a 2-worker `LocalCluster` against a 900x900 m east-valley bbox + 20 polygons.
  - Software-env name aligned across `default.yaml`, `phoenix.yaml`, `coiled.yaml`, and `CoiledConfig` default → `pv-geom-2026-05`.
- [ ] **M8 — Hardening** (in progress):
  - [x] `--resume` (2026-05-01): skips groups whose `part-<id>.parquet` is already on disk; aggregates stats from existing + new partitions in the manifest; treats it as crash-recovery (same inputs/config). 2 new unit tests verify mtime + bytes unchanged when resuming, and that resume is a no-op when no partitions exist. Real-data verification: re-running the M6 smoke command with `--resume` left the partition mtime untouched.
  - [x] **1000-polygon benchmark unblocked** (2026-05-02): `pipeline/tile_task.py` refactor hoisted the class-equality scan + class-1 fallback HAG filter out of the per-polygon loop into a single tile-group pre-filter (`_split_classes_for_tile_group`), then `del`s the raw 4-column array (~1.5 GB) before iterating polygons. `_build_row` now takes pre-filtered `ground_xyz` + `roof_input_pts` instead of the full `all_points`. Fallback HAG uses a single tile-wide ground median (Phoenix terrain is flat at 1 km scale; behaviour validated against the M7 integration test). Bench result on `bbox=(442000, 3681000, 443000, 3682000)`, 1000 polygons, 2 LocalCluster workers × 2 threads: **322 s wall (5.4 min), 1000/1000 succeeded, panel RMSE p50/p90 = 1.3/1.6 cm**, mounting mix correctly identified the `w0442n368x` cluster as utility-scale tracker arrays (617 tracker-suspected, 363 fixed, 20 ambiguous). Previous attempt (pre-refactor) OOM-killed all 4 Dask workers after ~50 min — see HANDOFF.md for the failure mode and post-mortem. Side fix: renamed 5 spike-cached LAZ files in `~/AppData/Local/Temp/pv_geom_cache/` to add the `data__lidar_data__` prefix that `_localize.localize()` expects (the unification copy in 2026-05-01 missed the rename).
  - [x] **Coiled validated end-to-end** (2026-05-02): Coiled software env `pv-geom-2026-05` provisioned via `coiled.create_software_environment` (1.5 GB conda env, 169 s build). Repo published at `github.com/JobTaminiau/pv_geom` so `install_pv_geom_on_workers` can `pip install git+https://...@main` on every worker. Side fixes uncovered while bringing this up: (a) `coiled.list_software_environments()` returns `dict[name → metadata]` in coiled>=1.x, was being iterated as `[{name: ...}]`; (b) `install_pv_geom_on_workers` only installed on workers, but the Coiled scheduler also imports `pv_geom.*` during `pickle.loads` in `scheduler.update_graph`, so we now `client.run_on_scheduler` first; (c) S3 region corrected to `us-east-2` everywhere (`coiled_env.REGION`, `S3Config.region` default, `configs/default.yaml`, `configs/phoenix.yaml`) — verified via the `x-amz-bucket-region` header on a public 403; (d) pre-warm step gated on `cfg.compute.backend != "coiled"` (the original SSO-race motivation doesn't apply on Coiled, and pre-warming to the laptop is wasteful when workers are already in-region); (e) `process_tile_group` now catches `RemoteFileMissing` per tile and returns an empty schema-conformant table when the primary tile is missing, since Coiled doesn't have the runner-side missing-tile filter; new test covers this. **Bucket access**: AWS-side, the Coiled BYOC role `coiled-jobtaminiau` (account `972466985839`) needed S3 read on `asu-nsf-phoenix` — unblocked via a bucket policy granting `s3:GetObject + s3:ListBucket + s3:GetBucketLocation` (note: there is no `s3:HeadObject` IAM action; HEAD on objects is authorized under `GetObject`). Diagnostic at `scripts/_coiled_aws_probe.py` runs `sts.get_caller_identity` + `s3.head_bucket/object/list_objects_v2` on a worker — useful template for future cross-account access debugging. **Smoke run**: 20 polygons on the M7 east-valley bbox in 117.8 s wall (cluster spinup ~85 s + worker install ~25 s + ~7 s compute), 19/20 panel fits, RMSE p50/p90 = 1.7/1.8 cm, 100% south-facing. **1000-polygon benchmark on the same bbox as the local run**: 414 s wall (vs 322 s local; Coiled overhead is ~90 s spinup + ~25 s install + ~10 s cold S3 reads), bit-for-bit identical algorithm output (980/1000 fits, RMSE p50/p90 = 1.3/1.6 cm, mounting mix 617 tracker / 363 fixed / 20 ambiguous — matches local).
  - [ ] `pv-geom benchmark` CLI subcommand (timing + peak RSS + per-polygon cost; `scripts/_bench_1k.py` and `scripts/_coiled_bench_1k.py` are the prototypes)
  - [ ] README + quickstart notebook
  - [ ] Zenodo metadata for first tagged release
  - [ ] Byte-for-byte reference parquet for the M7 integration test

## Open questions (PRD §13)

- ~~LiDAR CRS~~ — confirmed **EPSG:6341 (NAD83/UTM 12N, metres)** via spike (2026-05-01). Configs updated.
- Coiled software env — reuse `pv_mapper` (`pv-detect`) env or build fresh `pv-geom-runtime-2026-05`.
- Footprint `building_id` — FEMA AZ ships `build_id`; M2 reader will normalize to canonical `building_id`. Synthesizes when both absent.
- Output prefix convention — proposed: `s3://free-data-commons/pv_geom/<metro>/<version>/`.

## FEMA AZ footprint findings (M4 spike, 2026-05-01)

- ~92% of atlas polygons intersect at least one FEMA AZ footprint; ~8% sit in gaps (FEMA appears to miss some residential structures).
- Among polygons that match, only a fraction are *cleanly* contained — many SAM3 polygons overshoot or sit at the edge of the FEMA footprint, producing tiny intersection fractions.
- FEMA's `shape_area` attribute column is **all zeros** in the dataset; always compute `geometry.area` instead.
- Implication for M2/M6: roughly 1 in 10 polygons will get `on_building=False` (correctly) due to footprint gaps; M5 mounting rules must handle this cleanly. Optional supplement: Microsoft US Building Footprints would close most of the gap if needed.

## Spike (pre-M2)

`scripts/spike.py` — LiDAR data-quality spike, runs as a uv inline script. Two phases:

1. `uv run scripts/spike.py --discover` — lists `s3://asu-nsf-phoenix/data/lidar_data/`, looks for a `tile_index.parquet`, sniffs one LAZ header (CRS, point count, bbox).
2. `uv run scripts/spike.py [--polygon-id ... --tile-uri ...]` — clips class-6 points to one polygon, fits a plane via SVD, writes a 3-panel PNG (plan view, side view, residual histogram) to `scripts/eda_outputs/`.

Default target: detection `21_840480_397711__0` (sam3 score 0.98, ~67 m², east valley). Polygons sourced from `C:\Users\job_t\code\free\pv_sam3\artifacts\atlas\latest.parquet`.

`scripts/_aerial.py` overlays the Maricopa County 2024 ortho (`gis.maricopa.gov/.../Aerial2024Sep2024OctOrtho/MapServer/tile/{lod}/{row}/{col}`, LOD 13 = 7.5 cm/px) under the polygon + LiDAR points so panel boundaries can be visually verified. Disk cache at `~/.cache/pv_geom_aerial/`. Note: ESRI MapServer tiles are built on demand; first request can race the cache build, so `_fetch_tile` retries 4× with linear backoff.

Success criterion: a clear, flat tilted surface in the side view; RMSE ≲ 0.10 m; tilt distribution across ~50 random polygons looks plausible (residential AZ → mode near roof pitch ~20°, flush-mount → near 0°). If RMSE blows up or points are visibly noisy, reconsider the panel-fit threshold defaults in PRD §7.1 before M3.

### Spike results (2026-05-01)

**Phase 1** — single polygon `21_840480_397711__0` (67 m², east valley): tilt=5.9°, az=199°, RMSE=0.022 m. Plane visually clean.
**Phase 2** — 5 more polygons across 5 tiles: 5/6 RMSE < 0.04 m; one outlier at 0.377 m (clear polygon overshoot, RANSAC will handle).
**Phase 3 — batch (`scripts/spike_validate.py batch`, n=29):**
  - density centred on 9–11 pts/m² (PRD assumed 15–18; floor of 4 still safe)
  - tilt distribution bimodal: peak at 5–10° (flush-mount) + secondary at 18–22° (tilted-rack)
  - azimuth concentrated 150–230° (south to SSW), ~zero north-facing — physically correct for AZ
  - 27/29 RMSEs < 0.05 m (PRD threshold)
**Phase 4 — size sweep (`scripts/spike_validate.py size-sweep`, 50 bootstraps × 12 N):**
  - on a clean polygon, tilt is *unbiased* down to N=5 (mean stays at truth ±0.24°)
  - statistical spread falls as 1/√N: tilt std 0.24°→0.01°, azimuth std 1.6°→0.03° from N=5 to N=1000
  - both stay under the 2°/10° threshold at every N tested
  - **Practical floor is RANSAC robustness, not statistical noise**: ~30 inlier points needed to reject overshoot/multi-plane outliers reliably. At ~10 pts/m² density that's ~3 m² minimum — about the bottom 10–15% of polygons (median ~11 m²).

### Three PRD assumptions updated (configs aligned 2026-05-01)

1. **CRS** EPSG:6404 → **EPSG:6341** (NAD83/UTM 12N, metres) in `default.yaml`, `phoenix.yaml`, and `CRSConfig` default.
2. **Classification fallback**: new `io.classification` block (`panel_class_primary=6`, `panel_class_fallback=1`, `ground_class=2`, `fallback_height_above_ground_m=1.5`). USGS LPC AZ tiles have no class 6, so the M2 LiDAR reader will fall back to class-1 returns above the configurable HAG when class 6 is missing.
3. **Density floor** `min_density_pts_per_m2` 4 → **3**.
4. **`panel_plane.min_points`** new field, default **30** (replaces PRD §7.1's area-based formula). Size-sweep showed precision is fine below 30 but RANSAC robustness needs ~30 inliers to reliably reject overshoot.

Default-config sha256: `9c8773a836a3...` Phoenix-config sha256: `ad03924fec48...`

Tile index lives at `s3://asu-nsf-phoenix/data/lidar_data/USGS_AZ_MaricopaPinal_1_2020_TileIndex.zip` (zipped shapefile, 13,373 tiles, naming `w<easting_km>n<northing_km>`). 3,381 tiles uploaded.

## Deviations from PRD made during scaffolding

- Project dir is `pv-geom` (dash); package import name remains `pv_geom` (matches recent FREE projects e.g. `delaware-pv-inventory`).
- `pdal` and `whitebox` are optional extras (`[pdal]`, `[whitebox]`) rather than core deps — both have nontrivial Windows install paths. `laspy[lazrs]` stays in core so dev installs work everywhere.
