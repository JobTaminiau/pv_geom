# pv-geom — status

PRD: `docs/pv_geom_PRD.md` (v0.1, 2026-05-01).

## Milestones

- [x] **M1 — Scaffolding** (this commit): package layout, `pyproject.toml`, Pydantic config models with validation, CLI skeleton with `validate-config`, JSON-line logging, CI stub.
- [ ] **M2 — I/O layer**: polygon reader/reprojector, footprint reader, tile-index loader (auto-detect format), LAZ reader (laspy primary; PDAL extra), S3 fetch with caching. Unit tests on synthetic inputs.
- [ ] **M3 — Plane fitting**: `fit_plane_ransac`, tilt/azimuth, uncertainty (bootstrap). Synthetic plane unit tests.
- [ ] **M4 — Roof plane + heights**: ring-buffer construction, roof plane extraction, height-above-ground/roof.
- [ ] **M5 — Multi-plane + mounting rules**: east-west detection, per-polygon tracker heuristic, rules engine with confidence.
- [ ] **M6 — Pipeline orchestration**: spatial join polygons↔tiles, partitioner with edge handling, `process_tile_group`, Dask graph, manifest writer, partition parquet output.
- [ ] **M7 — Coiled integration + Phoenix integration test**: cluster spec, software env definition, integration test on small Phoenix bbox with reference output.
- [ ] **M8 — Hardening**: `--resume`, `pv-geom benchmark`, quickstart notebook, Zenodo metadata.

## Open questions (PRD §13)

- LiDAR CRS — verify EPSG:6404 (NAD83(2011) / Arizona Central, ftUS) at runtime against the tile index.
- Coiled software env — reuse `pv_mapper` (`pv-detect`) env or build fresh `pv-geom-runtime-2026-05`.
- Footprint `building_id` — required vs auto-generate (default in scaffold: auto-generate when absent).
- Output prefix convention — proposed: `s3://free-data-commons/pv_geom/<metro>/<version>/`.

## Spike (pre-M2)

`scripts/spike.py` — LiDAR data-quality spike, runs as a uv inline script. Two phases:

1. `uv run scripts/spike.py --discover` — lists `s3://asu-nsf-phoenix/data/lidar_data/`, looks for a `tile_index.parquet`, sniffs one LAZ header (CRS, point count, bbox).
2. `uv run scripts/spike.py [--polygon-id ... --tile-uri ...]` — clips class-6 points to one polygon, fits a plane via SVD, writes a 3-panel PNG (plan view, side view, residual histogram) to `scripts/eda_outputs/`.

Default target: detection `21_840480_397711__0` (sam3 score 0.98, ~67 m², east valley). Polygons sourced from `C:\Users\job_t\code\free\pv_sam3\artifacts\atlas\latest.parquet`.

Success criterion: a clear, flat tilted surface in the side view; RMSE ≲ 0.10 m; tilt distribution across ~50 random polygons looks plausible (residential AZ → mode near roof pitch ~20°, flush-mount → near 0°). If RMSE blows up or points are visibly noisy, reconsider the panel-fit threshold defaults in PRD §7.1 before M3.

## Deviations from PRD made during scaffolding

- Project dir is `pv-geom` (dash); package import name remains `pv_geom` (matches recent FREE projects e.g. `delaware-pv-inventory`).
- `pdal` and `whitebox` are optional extras (`[pdal]`, `[whitebox]`) rather than core deps — both have nontrivial Windows install paths. `laspy[lazrs]` stays in core so dev installs work everywhere.
