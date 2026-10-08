# Changelog

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
