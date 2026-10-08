# pv-geom

Tilt, orientation and height of solar PV installations, measured from LiDAR.

`pv-geom` takes two things:

1. **a set of PV polygons** (from aerial or satellite imagery) and the date that imagery was captured, and
2. **a classified LiDAR point cloud** and the date it was flown,

and produces a per-polygon dataset, summary tables, publication figures and a
report characterising the geometry of the installations: tilt profile,
orientation profile, joint distributions, heights, and how far each measurement
can be trusted.

```
polygons (+ vintage)  ─┐
                       ├─►  pv-geom run  ─►  dataset (GeoParquet)  ─►  pv-geom report  ─►  tables · figures · report
LiDAR tiles (+ date)  ─┘
```

Per-milestone history is in `STATUS.md`; changes by version in `CHANGELOG.md`.

## Why the two dates matter

The polygons and the LiDAR are almost never captured at the same time. If the
imagery is the newer of the two, some polygons mark installations that **did not
exist yet when the LiDAR was flown**. There are then no panels in the point
cloud; the plane fitted inside the polygon is the bare roof, and nothing about
that row's fit quality gives it away.

So both dates are run inputs, and every row carries them together with a
verdict on what its tilt and azimuth actually describe:

| `geometry_basis` | The fitted plane is… | Use for panel geometry? |
| --- | --- | --- |
| `panel_confirmed` | resolvably above the surrounding roof plane — panels were physically in the point cloud | yes |
| `panel_by_vintage` | not separable from the roof by height, but the polygon vintage is on or before the LiDAR date, so the installation existed | yes |
| `surface_unresolved` | coincident with the roof, and the polygons postdate the LiDAR: a flush-mounted array **or** the roof before installation | only if you can assume flush mounting |
| `unscreened` | of unknown standing: polygons postdate the LiDAR and there is no roof reference to test against | with caution |
| `no_fit` | absent — too few points or no consensus | no |

The report leads with the *panel basis* rows (the first two) and shows every
statistic for all fitted rows alongside, so the effect of the vintage gap is
visible rather than buried.

## Quickstart

```bash
uv sync --extra dev

# 1. Vet the LiDAR: classes present, density, CRS, units, true flight dates
uv run pv-geom inspect-tile path/or/s3/to/one_tile.laz

# 2. Measure
uv run pv-geom run \
  --polygons   polygons.parquet \
  --polygon-vintage 2024-04 \
  --lidar-prefix s3://bucket/lidar/tiles \
  --tile-index   s3://bucket/lidar/tile_index.zip \
  --output ./out \
  --local

# 3. Report (run does this automatically for local outputs)
uv run pv-geom report ./out --area-name "Metropolitan Phoenix"
```

`--local` runs on a `LocalCluster`; add `--no-dask` to run serially in-process.
`--dry-run` plans the run and performs the vintage check without computing
anything — worth doing before a long run. `--bbox` and `--max-polygons` bound a
test run.

## Inputs

| Input | Required | Notes |
| --- | --- | --- |
| PV polygons | yes | GeoParquet, GeoPackage, GeoJSON, Shapefile or FlatGeobuf, in any CRS. An id column is optional: `polygon_id`, `detection_id`, `id`, `fid` or `objectid` is used if present (or name one with `--polygon-id-col`), otherwise ids are synthesized. MultiPolygons are exploded into one row per part. |
| Polygon vintage | recommended | `--polygon-vintage 2024`, `2024-04` or `2024-04-01`. A year or month counts as its last day, so the gap to the LiDAR is never understated. For mosaics or permit-dated layers use `--polygon-vintage-col` to read a per-polygon date. |
| LiDAR tiles | yes | Classified LAZ, one file per tile, local or on S3, in a **projected, metric CRS** (points are not reprojected; foot-based CRSs are refused). Needs ground (ASPRS class 2). Building (class 6) is used when present; otherwise unclassified returns above local ground are used, which is the common case for public collections. |
| LiDAR tile index | yes | GeoParquet / GPKG / SHP / zipped SHP with one polygon per tile. The id column is auto-detected (`Name`, `NAME`, `tile_id`, …); `--name-template` turns it into a filename (default `{name}.laz`). |
| LiDAR date | measured | Left out, it is **measured per tile from per-point GPS time**, which is the flight date. Declare it with `--lidar-date` only if the tiles carry no usable GPS time. The LAS header date is the *delivery* date: Phoenix's tiles were flown 2020-11-26/28 and stamped 2021-06-30; Delaware's were flown 2023-03 and stamped 2024-10. |
| Building footprints | no | `--footprints`. With them, `on_building`/`building_id` are filled and the roof reference is clipped to the building. Without them `on_building` is null and the roof reference comes from an open ring around each polygon. |

Without a polygon vintage the run still works, but no row can be credited as
present-by-date: `geometry_basis` then rests on the height screen alone.

## Outputs

### Dataset — `pv-geom run`

One GeoParquet partition per LiDAR tile group plus `manifest.json`. The
manifest records inputs, resolved CRS, both vintages and the gap, the
configuration and its hash, counts, and summary statistics.

Per-row columns (source of truth and descriptions: `src/pv_geom/schema.py`; a
data dictionary CSV is written with every report):

- **Identity** — `polygon_id`, `parent_polygon_id` (the input feature), `input_row` (its row position in the input file — a join key even when the input has no ids), `geometry`, `area_m2`, `surface_area_m2` (area along the plane), `aspect_ratio`
- **Vintage** — `polygon_vintage`, `lidar_date`, `lidar_date_source` (`gps_time` / `declared` / `header_date`), `vintage_gap_days` (positive = polygon newer than LiDAR), `geometry_basis`
- **Plane fit** — `panel_tilt_deg`, `panel_azimuth_deg` (0 = N, 180 = S; null below 1° tilt), `panel_rmse_m`, `panel_tilt_unc_deg`, `panel_azimuth_unc_deg`, `n_points_panel`, `n_inliers_panel`, `point_density`, `n_planes_detected`, `secondary_tilt_deg`, `secondary_azimuth_deg`
- **Roof reference** — `roof_ref_source` (`footprint_ring` / `open_ring` / `none`), `roof_tilt_deg`, `roof_azimuth_deg`, `roof_rmse_m`, `panel_roof_angle_deg`, `height_above_roof_m`, `height_above_ground_m`, `on_building`, `building_id`
- **Quality and provenance** — `flags`, `lidar_tile_ids`, `pkg_version`, `config_hash`, `run_id`, `partition_id`

`flags` is a list drawn from `low_density`, `poor_fit`, `near_horizontal`,
`east_west_rack`, `roof_insufficient`, `roof_no_consensus`, `roof_complex`,
`no_panel_standoff`, `standoff_unscreenable`.

### Report — `pv-geom report <output>`

Written to `<output>/report/` (or `--out`):

| Path | Contents |
| --- | --- |
| `report.html` | Self-contained report: key findings, vintage statement, every figure and table, methods, data dictionary |
| `report.md` | The same as Markdown with linked figures, for pasting into a manuscript or Quarto document |
| `methods.md` | A methods paragraph filled in with this run's parameters and dates |
| `summary.json` | The headline numbers, machine-readable |
| `tables/*.csv` | Coverage funnel, geometry-basis composition, summary statistics, tilt profile (5° bins), azimuth profile (8 and 16 sectors), joint tilt × azimuth, roof relation, fit quality, flags |
| `figures/*.png\|pdf\|svg` | Overview (tilt + orientation rose + joint heatmap), each of those separately, geometry basis, vintage timeline, array-vs-roof, measurement quality, spatial distribution |
| `dataset/` | Consolidated `pv_geom.parquet` (GeoParquet), `pv_geom.csv` (no geometry; centroid lon/lat), `data_dictionary.csv`, flag and basis definitions, manifest |

Conventions:

- Every statistic is given **per polygon and weighted by array surface**, for
  **all fitted polygons and per geometry basis**.
- Azimuth is summarised with circular statistics (mean direction, resultant
  length R, circular SD).
- Figures are sized to journal column widths (89 mm / 183 mm) in 7 pt type;
  PNGs are 300 dpi and the PDF/SVG keep text editable.

Reporting also works on outputs written before 0.2.0; give their dates with
`--polygon-vintage` / `--lidar-date`.

## How it measures

1. **Panel plane.** LiDAR returns inside the polygon (eroded 15 cm) are fitted
   with RANSAC, refined by least squares on the inliers. Tilt and azimuth come
   from the plane normal; uncertainty from a bootstrap over the inliers.
2. **Roof reference.** A second plane is fitted to returns in a 3–5 m ring
   around the polygon (other PV polygons removed). A *collar guard* refits on
   the band nearest the array when the ring plane does not describe it, which
   stops the fit landing on the facet across a ridge.
3. **Standoff screen.** If the panel plane sits at least 5 cm above the roof
   plane the row is `panel_confirmed`. Two fits at ~2 cm RMSE cannot resolve
   less, so below that the row is flagged `no_panel_standoff`; with no usable
   roof reference it is `standoff_unscreenable`.
4. **Geometry basis.** The screen and the two dates combine as in the table
   above.

Calibrated against dated permits for Phoenix (34,837 single-permit parcels),
the 5 cm screen flags 90% of arrays known to postdate the LiDAR and 32% of
those known to predate it. It is a filter for building a trustworthy stratum,
not a way to date an individual array.

## Configuration

`configs/default.yaml` documents every key with its default; a study-area
config only lists what it changes (`configs/phoenix.yaml`,
`configs/delaware.yaml`). `--config` is optional. The resolved configuration is
hashed into every row and the manifest.

Keys you are most likely to touch:

- `vintage.{polygon_vintage, polygon_vintage_column, lidar_date}` — also settable from the CLI.
- `crs.target` — `auto` (from the tile index) or an explicit EPSG code.
- `io.classification` — ASPRS classes for panel candidates and ground, and the height-above-local-ground cutoff used when there is no building class.
- `panel_plane.{ransac_threshold_m, min_inlier_frac, min_points, erosion_m}`.
- `roof_plane.{open_ring, min_inlier_frac, collar_m}` and `heights.min_panel_standoff_m`.
- `compute.backend` — `local` or `coiled`.

### Mounting classification (archived)

Earlier versions labelled each array's mounting type (flush, tilted rack,
carport, …). A 300-polygon ground-truth sample put carport precision at 5% and
pole-mount at 0%, so in 0.2.0 the classifier is **off by default and its
columns are not in the output**. `mounting_rules.enabled: true` restores the
experimental `mounting_type` / `mounting_confidence` / `mounting_rule` columns;
treat them as unvalidated. The code and its tests remain in `classify/`.

## Compute

**Local** (default): `--local` for a `LocalCluster`, `--no-dask` for serial.
Each task holds one tile group's points in memory — roughly 1–3 GB for 10
pts/m² 1 km tiles, more for denser or larger tiles.

**Coiled**: set `compute.backend: coiled`. One-time setup:

```bash
coiled login
uv run python -c "from pv_geom.coiled_env import ensure_software_env; ensure_software_env()"
```

Workers install `pv_geom` from the GitHub repo, so the version you want must be
pushed. Outputs can go straight to `s3://`; partitions are written as each tile
group finishes, and `--resume` retries only what is missing. Workers need read
access to the LiDAR bucket; for a bucket in another account grant the Coiled
role `s3:GetObject`, `s3:ListBucket` and `s3:GetBucketLocation` in the bucket
policy (`scripts/_coiled_aws_probe.py` checks access from a real worker).

## Limitations

- **Metric LiDAR only**, read in its native CRS; tiles in feet must be reprojected first.
- **The standoff screen needs a roof reference**, so ground mounts and canopies are never `panel_confirmed`; they are `panel_by_vintage` when the dates allow and `unscreened` otherwise.
- **`surface_unresolved` rows are not wrong rows.** A flush array is parallel to its roof facet, so their tilt and azimuth are right for flush-mounted arrays and wrong for racks on flat roofs.
- **Small polygons** (under ~3 m² at 10 pts/m²) rarely gather the 30 returns a robust fit needs and mostly end as `no_fit`.
- **Removal is not detected**: a polygon older than the LiDAR is assumed still present when the LiDAR was flown.

## Tests

```bash
uv run pytest tests/unit/                    # synthetic data, ~1 min
RUN_INTEGRATION=1 uv run pytest tests/integration/ -v   # real Phoenix data; needs cached inputs
```

## Layout

```
src/pv_geom/
  cli.py              run · report · inspect-tile · describe-output · validate-config
  config.py           Pydantic config models + hash
  schema.py           Output schema, flag definitions, data dictionary
  vintage.py          Vintage parsing and the geometry_basis rule
  io/                 Polygons, footprints, tile index, LAZ, GeoParquet output
  geometry/           Plane fit, multi-plane, roof reference, heights, point index
  pipeline/           Partitioner, per-tile-group task, Dask runner
  report/             Statistics, figures, report + dataset builder
  classify/           Archived mounting classifier (experimental)
configs/              default.yaml, phoenix.yaml, delaware.yaml, coiled.yaml
tests/                unit/, integration/
scripts/              spikes, benchmarks, validation tooling
docs/                 PRD, working paper
```

## License

Not yet selected. The `pyproject.toml` classifier reads
`License :: Other/Proprietary License` as a placeholder; until a LICENSE file is
added, treat the code as all-rights-reserved.

## Acknowledgments

Built at FREE. LiDAR: USGS 3DEP (Arizona) and the Delaware/Maryland 2023
statewide collection. Building footprints: FEMA USA Structures.
