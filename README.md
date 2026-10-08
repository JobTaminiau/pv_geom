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

**Documentation:** the guide is in [`docs/guide/`](docs/guide/index.md) (tutorial, how-to guides, concepts, reference). Build the site with `uv sync --extra docs && uv run mkdocs serve`.

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
| `free_standing` | an elevated structure standing in open ground (a ground mount or a canopy), present when the LiDAR was flown; whether it carried modules then is not tested, but modules lie flush on such structures | yes |
| `surface_unresolved` | coincident with the roof, and the polygons postdate the LiDAR: a flush-mounted array **or** the roof before installation | only if you can assume flush mounting |
| `unscreened` | of unknown standing: polygons postdate the LiDAR and there is no roof reference to test against | with caution |
| `no_fit` | absent — too few points or no consensus | no |

The report leads with the *panel basis* rows (the first two) and shows every
statistic for all fitted rows alongside, so the effect of the vintage gap is
visible rather than buried.

## Quickstart

```bash
uv sync                      # core: local files, local compute
# uv sync --extra cloud      # + reading from / writing to S3
# uv sync --extra coiled     # + running on a Coiled cluster

uv run pv-geom demo          # a synthetic study area, measured and reported, offline
```

`demo` writes a small sample area with known geometry, a config that runs it,
the dataset and the report — the quickest way to see every output.

A real study area is described by **one config file** and run from it:

```yaml
# configs/my_area.yaml          (paths are relative to this file, or s3://)
study:
  name: My study area
  output: ../out/my_area

inputs:
  polygons: ../data/pv_polygons.gpkg       # any vector format; ids optional
  lidar_prefix: s3://bucket/lidar/tiles    # folder or prefix of LAZ tiles
  # tile_index: ...                        # optional: read from tile headers if absent
  # footprints: ...                        # optional building footprints

vintage:
  polygon_vintage: 2024-10                 # when the polygons' imagery was captured
  # lidar_date: ...                        # optional: measured from the tiles otherwise
```

```bash
uv run pv-geom inspect-tile s3://bucket/lidar/tiles/one.laz   # vet the LiDAR first
uv run pv-geom validate-config configs/my_area.yaml           # what would this run use?
uv run pv-geom run --config configs/my_area.yaml --dry-run    # plan, checks, time estimate
uv run pv-geom run --config configs/my_area.yaml              # measure, then report
```

Every input can also be given on the command line (`--polygons`,
`--lidar-prefix`, `--tile-index`, `--output`, `--polygon-vintage`, …), and an
option given there overrides the config. `configs/phoenix.yaml` and
`configs/delaware.yaml` are complete examples.

From Python:

```python
import pv_geom

result = pv_geom.run("configs/my_area.yaml")      # or keyword arguments, or both
gdf = result.load()                               # GeoDataFrame, one row per polygon
result.report(headline="panel", weight="area")    # tables, figures, HTML
```

Before any work is done, a run checks the things that would silently spoil it:

- **Dates.** It measures the LiDAR's flight dates and warns if the polygons are newer.
- **Point classes.** It says which class will supply array candidates, and stops
  if the LiDAR has no ground class.
- **Size.** `--dry-run` reports tiles, gigabytes and a rough time estimate (and
  cost, on Coiled, if `compute.coiled.usd_per_worker_hour` is set).

`--local` runs on a `LocalCluster`; `--no-dask` runs serially. `--resume` keeps
the partitions already written and runs the rest; it refuses to continue an
output made from different polygons, measurement settings or schema
(`--force-resume` overrides). `--bbox` and `--max-polygons` bound a trial run.
Problems you can fix — a layer without a CRS, LiDAR in feet, a missing optional
package, expired credentials — are reported as one line saying what is wrong and
one saying what to do.

## Inputs

| Input | Required | Notes |
| --- | --- | --- |
| PV polygons | yes | GeoParquet, GeoPackage, GeoJSON, Shapefile or FlatGeobuf, in any CRS. An id column is optional: `polygon_id`, `detection_id`, `id`, `fid` or `objectid` is used if present (or name one with `--polygon-id-col`), otherwise ids are synthesized. MultiPolygons are exploded into one row per part. |
| Polygon vintage | recommended | `--polygon-vintage 2024`, `2024-04` or `2024-04-01`. A year or month counts as its last day, so the gap to the LiDAR is never understated. For mosaics or permit-dated layers use `--polygon-vintage-col` to read a per-polygon date. |
| LiDAR tiles | yes | Classified LAZ, one file per tile, local or on S3, in a **projected CRS**. Tiles in feet, or in a different CRS from the working one, are converted as they are read. Needs ground (ASPRS class 2). Building (class 6) is used when present; otherwise unclassified returns above local ground are used, which is the common case for public collections. |
| LiDAR tile index | no | Without one, the index is read from the tiles' own headers (they must declare a CRS). Otherwise: GeoParquet / GPKG / SHP / zipped SHP with one polygon per tile. The id column is auto-detected (`Name`, `NAME`, `tile_id`, …); `--name-template` turns it into a filename (default `{name}.laz`). |
| LiDAR date | measured | Left out, it is **measured per tile from per-point GPS time**, which is the flight date. Declare it with `--lidar-date` only if the tiles carry no usable GPS time. The LAS header date is the *delivery* date: Phoenix's tiles were flown 2020-11-26/28 and stamped 2021-06-30; Delaware's were flown 2023-03 and stamped 2024-10. |
| Building footprints | no | `--footprints`. With them, `on_building`/`building_id` are filled and the roof reference is clipped to the building. Without them `on_building` is null and the roof reference comes from an open ring around each polygon. |

Without a polygon vintage the run still works, but no row can be credited as
present-by-date: `geometry_basis` then rests on the height screen alone.

## Outputs

### Dataset — `pv-geom run`

One GeoParquet partition per LiDAR tile group, `part-unmeasured.parquet`, and
`manifest.json`. The manifest records inputs, resolved CRS, both vintages and
the gap, the configuration and its hash, the schema version, counts, and summary
statistics.

**Every input polygon has exactly one row**, so the output joins one-to-one to
the inventory. `status` says what happened to it:

| `status` | Meaning |
| --- | --- |
| `measured` | A plane was fitted to the LiDAR returns inside the polygon |
| `no_fit` | LiDAR covered it but no plane could be fitted; `fit_failure` says why (`too_few_points`, `ground_level_only`, `no_consensus`) |
| `no_lidar_tile` | Its tile is in the index but not in storage |
| `outside_tile_index` | It lies outside every tile of the LiDAR index |
| `tile_unreadable` | Its tile group failed; a resumed run retries it |
| `invalid_geometry` | The input feature has no usable polygon geometry |

Rows other than `measured` and `no_fit` carry the input's identity, geometry,
area and vintage, with every measured column null. (`--bbox` and
`--max-polygons` narrow the scope: polygons they exclude are not part of the
run and have no row.)

Per-row columns (source of truth and descriptions: `src/pv_geom/schema.py`; a
data dictionary CSV is written with every report):

- **Identity** — `polygon_id`, `parent_polygon_id` (the input feature), `input_row` (its row position in the input file — a join key even when the input has no ids), `status`, `geometry`, `area_m2`, `surface_area_m2` (area along the plane), `aspect_ratio`
- **Vintage** — `polygon_vintage`, `lidar_date`, `lidar_date_source` (`gps_time` / `declared` / `header_date`), `vintage_gap_days` (positive = polygon newer than LiDAR), `geometry_basis`
- **Plane fit** — `tilt_deg`, `azimuth_deg` (clockwise from **true north**: 0 = N, 180 = S; null below 1° tilt), `grid_convergence_deg`, `fit_rmse_m`, `tilt_unc_deg`, `azimuth_unc_deg`, `fit_tolerance_m`, `fit_failure`, `n_points`, `n_inliers`, `point_density`, `n_facets`, `secondary_tilt_deg`, `secondary_azimuth_deg`
- **Roof reference** — `roof_ref_source` (`footprint_ring` / `open_ring` / `none`), `roof_tilt_deg`, `roof_azimuth_deg`, `roof_rmse_m`, `angle_to_roof_deg`, `height_above_roof_m`, `height_above_ground_m`, `on_building`, `building_id`
- **Quality and provenance** — `flags`, `lidar_tile_ids`, `pkg_version`, `config_hash`, `run_id`, `partition_id`

`flags` is a list. Measurement flags: `low_density`, `poor_fit`,
`near_horizontal`, `wide_tolerance_fit`, `east_west_rack`, `roof_insufficient`,
`roof_no_consensus`, `roof_complex`, `no_standoff`,
`standoff_unscreenable`. Input-quality flags, set when the polygon layer is
read: `below_min_area`, `overlaps_polygon` (shares at least 20% of its area
with another input polygon), `duplicate_geometry`, `geometry_repaired`. Flagged
polygons are still measured.

**Azimuth is true north.** Planes are fitted in the LiDAR's projected
coordinates, where "north" is the grid's y axis. Grid north differs from true
north by the meridian convergence, which changes with position: about −0.8° in
west Phoenix, −0.1° in southern Delaware, up to ±1.7° at a UTM zone edge at this
latitude, and more in some State Plane zones. pv-geom adds the convergence at
each polygon (from PROJ) so that every azimuth column is measured from true
north and means the same thing in every study area; `grid_convergence_deg`
holds the value added, and the manifest says `azimuth_reference: true_north`.
Outputs written before schema 0.4 are relative to grid north, and reports on
them say so. Tilt needs no such correction.

**Schema version.** Every partition and the manifest carry `schema_version`
(currently `0.4`). Within a major version columns are only added, and added
columns are nullable; outputs from older versions are upgraded when read.

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
| `dataset/` | The release dataset, self-describing: `pv_geom.parquet` (GeoParquet), `pv_geom.csv` (no geometry; centroid lon/lat), `README.md`, `data_dictionary.csv`, status / basis / flag definitions, `metadata.json` (repository-deposit fields), `manifest.json`, `SHA256SUMS.txt` |

Conventions:

- Every statistic is given **per polygon and weighted by array surface**, for
  **all fitted polygons and per geometry basis**. `--headline` and `--weight`
  choose which of those the report leads with.
- The median tilt and the facing shares carry **95% bootstrap intervals** over
  polygons. They express sampling variability, not measurement error.
- Azimuth is summarised with circular statistics (mean direction, resultant
  length R, circular SD).
- Figures are sized to journal column widths (89 mm / 183 mm) in 7 pt type;
  PNGs are 300 dpi and the PDF/SVG keep text editable.

Reporting also works on outputs written before 0.2.0; give their dates with
`--polygon-vintage` / `--lidar-date`.

Two further modes:

- `pv-geom report A B --compare --out dir` sets two to four runs side by side:
  overlaid tilt and orientation profiles and a comparison table.
- `pv-geom report <output> --regions districts.gpkg --region-col name` adds a
  per-region table and map, and a `region` column in the release dataset.

### External validation — `pv-geom compare-reference`

Compares a run with geometry reported by someone else (PVDAQ, USPVDB, permits),
mount against measured facet. See [`docs/validation.md`](docs/validation.md).

### Reproducing a run — `pv-geom verify`, `pv-geom reproduce`

Every manifest records a content hash of the rows, the inputs read and the
environment. `pv-geom verify <output>` checks an output against its manifest
(or `--against` another output); `pv-geom reproduce <output> --out <new>` reruns
it and compares. The result is bit for bit on one machine and within 0.0001
across platforms.

### Choosing rows

Start from `recommended == True`. Polygons covering more than one roof face
have one row describing the largest face and a `facets` entry per face
(`pv_geom_facets` in the release dataset).

## How it measures

1. **Fitted plane.** LiDAR returns inside the polygon (eroded 15 cm) are fitted
   with a consensus search: planes through point triples are scored by how many
   returns lie within tolerance, the best distinct candidates are refitted to
   their inliers until nothing changes, and the winner is settled with a smooth
   robust fit. Small polygons try every triple, so the result does not depend
   on a random seed. Tilt and azimuth come from the plane normal; uncertainty
   from a bootstrap over the inliers. If a different plane fits nearly as well
   the row is flagged `ambiguous_fit`.
2. **Roof reference.** A second plane is fitted to returns in a 3–5 m ring
   around the polygon (other PV polygons removed). A *collar guard* refits on
   the band nearest the array when the ring plane does not describe it, which
   stops the fit landing on the facet across a ridge.
3. **Standoff screen.** If the fitted plane sits at least 5 cm above the roof
   plane the row is `panel_confirmed`. Two fits at ~2 cm RMSE cannot resolve
   less, so below that the row is flagged `no_standoff`; with no usable
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
`configs/delaware.yaml`). The resolved configuration is hashed into every row
and the manifest.

Keys you are most likely to touch:

- `study.{name, output}` and `inputs.{polygons, lidar_prefix, tile_index, footprints, polygon_id_col, name_template}` — what to measure and where to write it.
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

**Memory is bounded.** Tiles are decoded a chunk at a time and only returns
near a polygon are kept, so a 44-million-return tile peaks at about 0.3 GB
instead of 3.9 GB. If a tile group's kept returns would still exceed
`compute.memory_budget_gb` (default 6), its polygons are measured in spatial
batches. The rows are identical either way. Set the budget to about half a
worker's memory.

**Coiled**: `pip install "pv-geom[coiled]"`, `coiled login`, and set
`compute.backend: coiled`. The rest is worked out unless you set it under
`compute.coiled`:

| Setting | `auto` means |
| --- | --- |
| `package_source` | The exact commit this machine is running. It must be committed and pushed; a run stops before starting a cluster if it is not. After install every worker is asked what it has, and the run stops on a mismatch. |
| `region` | Where the LiDAR bucket is, so tiles are read in-region. |
| `software` | An environment named after pv-geom's dependency list, built on first use. |
| `name` | `pv-geom-<study.name>`. |

Worker count, memory and CPU are plain settings. Outputs can go straight to `s3://`; partitions are written as each tile
group finishes, and `--resume` retries only what is missing. Workers need read
access to the LiDAR bucket; for a bucket in another account grant the Coiled
role `s3:GetObject`, `s3:ListBucket` and `s3:GetBucketLocation` in the bucket
policy (`scripts/coiled_aws_probe.py` checks access from a real worker).

## Limitations

- **Rows of tilted modules on flat roofs are not resolved.** A polygon over many short rows is fitted by the envelope of the rows, which is nearly flat whatever the module tilt. Such fits are flagged `envelope_fit` and left out of `recommended`.
- **The run CRS is metric.** LiDAR in feet or another CRS is converted on read.
- **Ground mounts and canopies have no roof to stand off from**, so they are rarely `panel_confirmed`. They are recognised instead by the open ground around them (`free_standing`; see `open_ground_share`). That establishes the structure, not the modules: a carport roofed with panels after the LiDAR was flown also qualifies.
- **`surface_unresolved` rows are not wrong rows.** A flush array is parallel to its roof facet, so their tilt and azimuth are right for flush-mounted arrays and wrong for racks on flat roofs.
- **Small polygons** (under ~3 m² at 10 pts/m²) rarely gather the 30 returns a robust fit needs and mostly end as `no_fit`.
- **Removal is not detected**: a polygon older than the LiDAR is assumed still present when the LiDAR was flown.

## Development

```bash
uv sync --extra dev
uv run python scripts/check.py     # every gate: lint, types, tests, regression benchmarks
```

The gates, which CI runs on Linux and Windows:

| Gate | Command | Standard |
| --- | --- | --- |
| Lint | `ruff check src tests scripts` | no findings |
| Types | `mypy src` | no errors |
| Tests | `pytest -m "not integration"` | pass, line coverage >= 88% |
| Regression | `pytest tests/benchmark` | results match the golden output |

**Regression benchmarks.** `tests/benchmark` runs the whole pipeline over a fixed
study area and compares every measured value with a stored golden output, so a
refactor cannot quietly move the numbers. A synthetic area with known geometry is
committed and always runs. Two real slices (120 Phoenix and 80 Delaware polygons
with their LiDAR) run when present under `data/benchmark/`; they are not in the
repository because the polygon layers are unpublished. Build them with
`scripts/build_benchmark.py`. When a change in results is intended, regenerate the
golden files in the same commit and say why.

`RUN_INTEGRATION=1 uv run pytest tests/integration/` runs against real Phoenix
data on S3; point `PV_GEOM_IT_POLYGONS` at the polygon layer.

## Layout

```
src/pv_geom/
  api.py              run() · load() · report() · describe(): the Python API
  cli.py              run · demo · report · inspect-tile · describe-output · validate-config
  sample.py           The synthetic study area behind `pv-geom demo` and the benchmark
  errors.py           Typed errors, each with a remedy
  config.py           Pydantic config models + hash
  schema.py           Output schema, flag definitions, data dictionary
  vintage.py          Vintage parsing and the geometry_basis rule
  summary.py          At-a-glance statistics (manifest, describe-output)
  testing.py          Regression helpers: golden frames and their comparison
  io/
    vector.py         One reader for polygons, footprints and the tile index
    storage.py        Remote listing and local caching
    lidar.py          LAZ reading, flight dates, tile inspection
    tile_scan.py      Tile index built from LAZ headers
    output.py         GeoParquet partitions; reading a run back
  geometry/           Plane fit, multi-plane, roof reference, heights, point index
  pipeline/
    plan.py           Inputs -> tile groups -> work plan (no point data read)
    pointpool.py      One tile group's LiDAR -> indexed ground and panel pools
    measure.py        The measurement steps and the Measurement record
    rows.py           Measurements -> schema rows
    worker.py         One tile group -> one table
    executor.py       Serial / local Dask / Coiled
    sink.py           Local or S3 output
    vintage_check.py  Run-start checks: input dates, LiDAR point classes
    estimate.py       Rough time / cost estimate before a run
    runner.py         Plan, execute, write, record
  report/
    data.py  stats.py  figures.py  text.py  tables.py  render.py  export.py  build.py
  experimental/
    mounting/         Archived mounting classifier (off unless enabled)
configs/              default.yaml, phoenix.yaml, delaware.yaml, coiled.yaml
tests/                unit/, benchmark/, integration/
scripts/              check.py, build_benchmark.py, coiled_aws_probe.py, archive/
docs/                 1.0 specification, PRD (historical), working paper, history/
```

## License

Not yet selected. The `pyproject.toml` classifier reads
`License :: Other/Proprietary License` as a placeholder; until a LICENSE file is
added, treat the code as all-rights-reserved.

## Acknowledgments

Built at FREE. LiDAR: USGS 3DEP (Arizona) and the Delaware/Maryland 2023
statewide collection. Building footprints: FEMA USA Structures.
