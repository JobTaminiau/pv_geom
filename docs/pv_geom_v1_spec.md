# pv-geom 1.0 — specification

Status: draft for review, 2026-10-07. Baseline: v0.2.0 (branch `v0.2-geometry-report`).
Supersedes `docs/pv_geom_PRD.md` (v0.1, 2026-05-01) as the forward-looking document; the
PRD remains the record of how 0.1 was designed.

## 1. What 1.0 means

**One sentence.** Someone who is not us can point pv-geom at a PV polygon layer and a LiDAR
collection, each with a date, and get a dataset, figures and a report they can put in a
publication, with every number traceable and every limitation stated.

0.2.0 does this for the two study areas we have run it on, operated by its authors. 1.0 is
the version where that sentence holds for a third party and a third study area. Three things
separate the two:

1. **Evidence.** 0.2.0 reports precision (fit residuals, bootstrap spread). It has never been
   scored for *accuracy* against independent truth, and `geometry_basis` has been checked
   against permits only in its 0.1 form.
2. **Completeness.** The output has a row only for polygons the LiDAR covered. Polygons that
   dropped out are a count in the manifest, not rows with a reason. A published dataset needs
   the full accounting.
3. **Maintainability.** The code works but carries the shape of its history: one 449-line
   orchestration function, a 224-line row builder, 88 type errors, no tests on the CLI, and a
   CI that fails its own lint step. It needs a refactoring pass before more is built on it.

### Definition of done for 1.0

- [ ] Every Must story below is accepted.
- [ ] Accuracy and basis-validity numbers are published in the README and the report (Epic A).
- [ ] A person who has not seen the repo completes the tutorial on the bundled sample in
      under 15 minutes on Windows, macOS and Linux (Epic I).
- [ ] A third study area runs without a code change (Epic D).
- [ ] Quality gates in §6 are green in CI.
- [ ] Schema is versioned and frozen; a 1.x release will not break a 1.0 reader (Epic G).
- [ ] License chosen, DOI minted (user decisions, §8).

## 2. Who it is for

| Persona | Wants | Reads |
| --- | --- | --- |
| **Analyst** (Ana) | Tilt and orientation profile of a region's PV fleet for a report | The HTML report, the summary tables |
| **Researcher** (Ravi) | Per-array geometry as a model input; figures and a methods paragraph for a paper | The dataset, data dictionary, `methods.md`, vector figures |
| **Operator** (Omar) | To run a new study area end to end, cheaply, and know it finished correctly | CLI output, manifest, logs |
| **Data consumer** (Dana) | To use the published dataset without installing pv-geom | The release dataset, dictionary, README |
| **Maintainer** (Mia) | To change one stage without breaking another | The code, tests, CI |

## 3. Where 0.2.0 stands

| Area | In 0.2.0 | Gap to 1.0 |
| --- | --- | --- |
| Inputs | Polygons (any vector format) + LAZ tiles + tile index; dates declared or measured; footprints optional | Tile index required; metric LAZ only; no COPC/EPT; class handling assumes ASPRS 1/2/6 |
| Measurement | RANSAC plane, bootstrap uncertainty, adaptive tolerance, roof reference with collar guard | Accuracy never scored; one plane per polygon, although 58% of the fits in the first Delaware run found a second; uncertainty is precision only |
| Vintage | Per-row dates, gap, `geometry_basis` | 36–46% of fitted rows are `unscreened`; no screen for ground mounts; open-ring references unvalidated |
| Accounting | Rows for covered polygons; counts in manifest | No row or reason for uncovered polygons |
| Report | Tables, figures, HTML/Markdown, methods, release dataset | One run at a time; no regional breakdown; no installation-level view; no Word/Quarto output |
| Operation | Local and Coiled; resume; incremental writes | Resume does not check the config; no cost/time estimate; Coiled settings hard-coded to this project |
| Code | 6,245 lines, 254 unit tests, 83% line coverage | See §5 |

## 4. User stories

Priority: **M**ust for 1.0, **S**hould, **C**ould. Size: S ≤ 1 day, M 2–4 days, L 1–2 weeks
(my estimates, uncalibrated).

### Epic A — Evidence that the numbers are right

The single most important gap. Everything else assumes it.

| ID | Story | Acceptance criteria | P | Size |
| --- | --- | --- | --- | --- |
| A1 | As Ravi, I want measured tilt and azimuth scored against independent truth, so that I can cite an accuracy, not just a residual. | A validation set of ≥ 150 arrays with independently known tilt/azimuth (candidates: permit or interconnection records with design tilt; surveyed sites; arrays measured in high-resolution oblique imagery). Report bias, MAE and 90th-percentile error for tilt and azimuth, by `geometry_basis` and by tolerance class. **Azimuth is compared in true north** (E6): truth sources state true or magnetic bearings, never grid bearings, and a grid-north comparison would show the meridian convergence as a spurious bias (−0.3° to −1.0° across the Phoenix atlas). The truth source's own reference (true / magnetic, and the declination applied) is recorded. Numbers appear in README and report. | M | L |
| A2 | As Ravi, I want the stated uncertainty to mean something, so that I can propagate it. | Reliability check on the A1 set: the share of arrays whose true value falls within ±1σ and ±2σ of the estimate. If coverage is below nominal, uncertainties are rescaled or relabelled as precision. | M | M |
| A3 | As Ana, I want `geometry_basis` validated in its 0.2 form, so that "panel basis" can be trusted as a filter. | Using the Phoenix permit join (`../pv-cooling`): for each basis, the share of arrays installed before vs after the LiDAR. Separately for `footprint_ring` and `open_ring` references. Publish the confusion table; revise the 5 cm threshold if warranted. | M | M |
| A4 | As Mia, I want a regression benchmark, so that a refactor cannot silently move the numbers. | A frozen set of ~200 real polygons with their points (a few MB, in the repo or a release asset) and a golden output; CI fails if tilt changes by more than 0.05° on any row without an explicit golden update. | M | M |
| A5 | As Ravi, I want wide-tolerance fits characterised, so that I know whether to keep them. | A1 metrics split by `wide_tolerance_fit`; recommendation in the README (keep / down-weight / exclude). | S | S |
| A6 | As Mia, I want synthetic end-to-end truth tests across the parameter space. | Property tests: for tilt 0–60°, all azimuths, noise 1–10 cm, density 2–30 pts/m², recovered tilt within a stated bound. | S | M |

### Epic B — Complete accounting

| ID | Story | Acceptance criteria | P | Size |
| --- | --- | --- | --- | --- |
| B1 | As Dana, I want one row per input polygon whatever happened to it, so that the dataset joins one-to-one to the inventory. | Output has exactly one row per input polygon part. A `status` column: `measured`, `no_fit`, `no_lidar_tile`, `outside_tile_index`, `tile_unreadable`, `invalid_geometry`. Geometry columns null where not measured. Report's coverage funnel is derived from it. | M | M |
| B2 | As Omar, I want polygon input problems reported, not silently dropped. | Empty, invalid and duplicate geometries, and polygons below a minimum area, are counted, listed in a sidecar file, and (B1) carried as rows. Overlapping polygons are flagged. | M | S |
| B3 | As Ana, I want to know why a polygon has no fit. | `no_fit` is subdivided by cause in a `fit_failure` column: `too_few_points`, `no_consensus`, `ground_level_only`. Report tabulates it. | S | S |
| B4 | As Omar, I want the run to say what it did not cover and what it would take. | End-of-run summary and manifest list tiles missing from storage with polygon counts per tile, as a ready-to-use fetch list. | S | S |

### Epic C — Vintage and presence

| ID | Story | Acceptance criteria | P | Size |
| --- | --- | --- | --- | --- |
| C1 | As Ravi, I want more of the fleet on a verified basis, so that the trustworthy stratum is not a minority. | The `unscreened` share of fitted rows falls from 36–46% to below 20% on both test blocks, by improving the roof reference (candidates: multi-facet ring segmentation; using the fitted building roof from all building points rather than a ring). Validated under A3. | M | L |
| C2 | As Ana, I want ground mounts and canopies screened too. | An off-roof presence test (panel plane measurably above ground, with a planar tilted surface rather than terrain) yields a `panel_confirmed` equivalent; validated on the 28 confirmed ground mounts in the 0.1 label set. | S | M |
| C3 | As Ravi, I want the LiDAR date per polygon, not per tile. | `lidar_date` comes from the GPS time of the returns inside the polygon. Matters where a tile mixes flight days or overlapping collections. | S | M |
| C4 | As Ana, I want an imagery date range, not a single day. | `--polygon-vintage` accepts a start and end; rows carry both; `geometry_basis` uses the end and the report states the window. | S | S |
| C5 | As Ana, I want the report to estimate how many arrays postdate the LiDAR. | Mixture estimate of the post-LiDAR share (as done by hand for Phoenix: ~43%) with an interval, shown in the vintage section. | S | M |
| C6 | As Ravi, I want removals considered. | Documented limitation at minimum; Could: flag polygons older than the LiDAR whose surface shows no standoff as `possibly_removed_or_flush`. | C | S |

### Epic D — Inputs and portability

| ID | Story | Acceptance criteria | P | Size |
| --- | --- | --- | --- | --- |
| D1 | As Omar, I want to run a third study area with no code change. | A new US state with public 3DEP LiDAR and any available PV polygon layer runs from a config file and CLI flags alone. Whatever breaks is fixed in the engine, as Delaware's findings were. | M | L |
| D2 | As Omar, I want to point at a folder of tiles without a tile index. | `--tile-index` optional: built from LAZ headers (local or S3, parallel, cached to a GeoParquet beside the output). | M | M |
| D3 | As Omar, I want LiDAR in feet or another CRS handled. | Tiles in a foot-based or non-matching CRS are reprojected and unit-converted on read (horizontal and vertical), with the transformation recorded in the manifest. Today these are refused. **Azimuth must not depend on which CRS the work is done in** (E6): convergence is taken from the CRS the plane is actually fitted in, and the cross-CRS test in E6 covers every CRS family this story admits (UTM, State Plane Transverse Mercator and Lambert, in metres and feet). | M | M |
| D4 | As Omar, I want the class scheme checked up front. | Run start inspects sampled tiles and prints which classes will serve as ground and panel candidates; fails with a clear message if there is no ground class. Noise classes (7, 18) are excluded explicitly. | M | S |
| D5 | As Omar, I want COPC / EPT sources. | Read Cloud-Optimized Point Cloud and Entwine sources by spatial query, so only returns near polygons are fetched. Large cost reduction for sparse inventories (Delaware fetches 475 MB per tile for ~15 polygons). | S | L |
| D6 | As Omar, I want other object stores. | Any `fsspec` URL (S3, GCS, Azure, HTTPS) for every input and the output; requester-pays honoured. The `io.s3` config block is currently unused. | S | M |
| D7 | As Omar, I want vegetation kept out of the fit. | Where high-vegetation classes exist they are excluded; where they do not, a first/last-return or planarity filter removes overhanging canopy. Measured effect on fit rate under tree cover. | S | M |

### Epic E — Geometry depth

| ID | Story | Acceptance criteria | P | Size |
| --- | --- | --- | --- | --- |
| E1 | As Ravi, I want a polygon spanning two roof facets measured as two arrays, so that its tilt is not an average or a failure. | When a polygon holds more than one plane, the output carries one **segment** row per plane (own tilt, azimuth, area share, uncertainty) linked to the polygon, plus the polygon-level row. Schema decision in §8. Property-tested on synthetic gables. | M | L |
| E2 | As Ana, I want results per installation, not per detection fragment. | A post-processing step groups polygons into installations (same building, or spatial clustering with tilt/azimuth coherence) and writes an installation-level table: total area, area-weighted tilt, dominant azimuth, facet count. Report offers both levels. | S | L |
| E3 | As Ana, I want an indicative capacity. | Optional `capacity_kw_est` from surface area and a configurable power density, clearly labelled as an estimate, used as an alternative weight in the report. | S | S |
| E4 | As Ravi, I want small arrays measured where possible. | Erosion scales with polygon size; the minimum-points floor is re-derived from A6. Fit rate for polygons under 5 m² reported before and after. | S | M |
| E6 | As Ravi, I want azimuth referenced to **true north**, explicitly, so that it means the same thing in every study area and can be compared with other sources. | Today azimuth is the direction of the plane normal in projected x/y, i.e. relative to **grid north**. Grid north differs from true north by the meridian convergence, which varies with position: −0.8° in the Phoenix test block, −0.3° to −1.0° across the Phoenix atlas, −0.1° to −0.4° in Delaware, up to ±1.7° at a UTM zone edge at 33° N, and several degrees in some State Plane zones. Required: (1) `panel_azimuth_deg`, `secondary_azimuth_deg` and `roof_azimuth_deg` are **true-north** azimuths: grid azimuth plus the meridian convergence at the polygon centroid, from PROJ (`Proj.get_factors(lon, lat).meridian_convergence`); (2) a `grid_convergence_deg` column carries the value applied, so the grid azimuth is recoverable; (3) the manifest, data dictionary, report captions and methods text state `azimuth_reference: true_north`; an output written before this change is labelled `grid_north` when read, and reports on it say so; (4) tilt is unaffected (it does not depend on the horizontal axes) and angles between two planes are unaffected; (5) **test**: one physical plane of known true azimuth, expressed in at least three supported CRSs with different convergence (two adjacent UTM zones and a State Plane zone), yields the same true azimuth within 0.05° and the convergence PROJ reports matches an independent geodesic computation; (6) the regression goldens are regenerated once, deliberately, for this change. | M | S |
| E5 | As Ravi, I want orientation relative to the sun, not just compass. | Derived columns: annual plane-of-array irradiance factor relative to optimal for the site latitude (simple transposition model), enabling an "orientation loss" profile in the report. | C | M |

### Epic F — Report and figures

| ID | Story | Acceptance criteria | P | Size |
| --- | --- | --- | --- | --- |
| F1 | As Ana, I want two runs compared side by side. | `pv-geom report A B --compare` produces overlaid tilt and orientation profiles and a comparison table (e.g. Phoenix vs Delaware; or one area at two versions). | M | M |
| F2 | As Ana, I want results by district. | `--regions <polygon layer> --region-col <name>` adds per-region summary tables and a choropleth of median tilt and dominant orientation. | M | M |
| F3 | As Ravi, I want the report in my manuscript toolchain. | Outputs include a Quarto (`.qmd`) file and a Word (`.docx`) file besides HTML and Markdown; tables are also emitted as LaTeX. | S | M |
| F4 | As Ravi, I want figures to match a journal's style. | `--figure-preset {nature,elsevier,science,cell,report}` sets widths and type sizes; a `--style` file overrides colours and fonts. Compatible with the house `journal_style` conventions. | S | S |
| F5 | As Ana, I want to choose what the report leads with. | `--headline {panel,all_fitted,...}` and `--weight {area,count,capacity}`; the choice is printed on every figure caption. | M | S |
| F6 | As Ana, I want a figure that shows what the vintage gap does to the answer. | A dedicated figure: tilt and orientation profile per `geometry_basis`, small multiples. | S | S |
| F7 | As Ravi, I want uncertainty on the summary statistics. | Bootstrap intervals over polygons for medians and shares in `summary_statistics.csv`, shown in the report. | M | M |
| F8 | As Dana, I want an accessible report. | Figures carry alt text; every figure has its table; colour is never the only encoding; HTML passes an automated accessibility check. | S | S |
| F9 | As Ana, I want to explore the map. | An optional interactive map (single HTML file) of polygons coloured by tilt/azimuth/basis. | C | M |

### Epic G — The dataset as a product

| ID | Story | Acceptance criteria | P | Size |
| --- | --- | --- | --- | --- |
| G1 | As Dana, I want a stable schema. | `schema_version` in every file and the manifest; documented compatibility rule (1.x only adds nullable columns); a reader that upgrades older outputs. | M | S |
| G2 | As Dana, I want the release self-describing. | `dataset/` includes a README (what it is, how to cite, limitations), the data dictionary, value definitions, a checksum file, and machine-readable metadata (DataCite / Zenodo JSON generated from the manifest). | M | M |
| G3 | As Dana, I want formats I already use. | GeoParquet (primary), GeoPackage, CSV; optionally PMTiles for web maps. | S | S |
| G4 | As Dana, I want a recommended-use filter. | A boolean `recommended` column and a documented rule (fitted, panel basis, not flagged for quality), so the default subset is one filter away. Rule fixed after Epic A. | M | S |
| G5 | As Ravi, I want to reproduce a published dataset. | The manifest plus the pinned environment reproduce the output bit-for-bit on the same inputs; verified in CI on the A4 benchmark across two platforms. | M | M |

### Epic H — Operating at scale

| ID | Story | Acceptance criteria | P | Size |
| --- | --- | --- | --- | --- |
| H1 | As Omar, I want resume to be safe. | `--resume` refuses to continue when the config hash, schema version or input fingerprints differ from the existing partitions, unless forced. Today it is "crash recovery only" by convention. | M | S |
| H2 | As Omar, I want an estimate before I spend. | `--dry-run` reports tiles to read, bytes to fetch, estimated wall time and, on Coiled, estimated cost, from a small calibration table. | M | M |
| H3 | As Omar, I want to see progress. | A progress line (groups done / total, rows, failures, ETA) and structured logs; no bare `print`. | M | S |
| H4 | As Omar, I want the cloud backend not tied to this project. | Repository URL, region, software environment and worker shape are configuration with sensible detection, not constants in `coiled_env.py`. Workers install the same version the client runs. | M | M |
| H5 | As Omar, I want bounded memory whatever the tile. | Peak worker memory is bounded by a configured budget (chunked LAZ decoding; group splitting for dense tiles); verified on the densest Phoenix and Delaware groups. | M | M |
| H6 | As Omar, I want the full atlases rerun on 0.2+. | Phoenix (≈349k rows) and Delaware (13,982 polygons) full runs to versioned prefixes, with reports. Needs go-ahead (§8). | M | M |
| H7 | As Omar, I want faster fits. | Vectorised or compiled RANSAC and bootstrap; target ≥ 3× on per-polygon time with identical results on A4. | S | M |

### Epic I — Using it

| ID | Story | Acceptance criteria | P | Size |
| --- | --- | --- | --- | --- |
| I1 | As Ana, I want to try it in ten minutes. | A bundled sample (a few hundred polygons, one small LAZ clip, permissively licensed) and `pv-geom demo`, producing a full report offline. | M | M |
| I2 | As Ravi, I want a Python API. | `pv_geom.run(...)`, `pv_geom.load(...)`, `pv_geom.report(...)` documented and stable; the CLI is a thin layer over them. | M | M |
| I3 | As Omar, I want installation to be light. | Core install needs no cloud packages. Extras: `[cloud]` (s3fs, boto3), `[coiled]`, `[pdal]`. `scikit-learn` (declared, unused) removed. `pip install pv-geom` works. | M | S |
| I4 | As anyone, I want errors that say what to do. | Every anticipated failure (no CRS, non-metric CRS, missing id, no ground class, no credentials, tile index mismatch) raises a typed error with a remedy; covered by tests. | M | M |
| I5 | As anyone, I want documentation. | A docs site: tutorial, how-to guides (new study area, cloud run, report customisation), concept pages (vintage and basis; how the fit works), CLI and API reference, FAQ. | M | L |
| I6 | As Omar, I want a setup check. | `pv-geom doctor` verifies environment, credentials and read access to the given inputs. | S | S |
| I7 | As Omar, I want one config file per study area including the input paths. | Inputs may live in the config (`inputs:` block) so a run is `pv-geom run --config area.yaml`. | S | S |

## 5. Refactoring pass

Do this first (milestone 0.3). Rule for the pass: **no behaviour change**. The A4 regression
benchmark is built before anything is moved, and stays green throughout.

### 5.1 Findings (audit of v0.2.0, 2026-10-07)

| # | Finding | Evidence |
| --- | --- | --- |
| 1 | Orchestration is one function | `run_pipeline`: 449 lines, 14 parameters, with nested closures for output paths, resume, pre-warm, task building, dispatch and callbacks |
| 2 | The row builder does everything | `_build_row`: 224 lines, 17 parameters; fit, retry, roof, heights, screen, basis and serialisation in one body, returning an untyped dict |
| 3 | Statistics are computed twice | `runner._aggregate` (93 lines) and `report/stats.py` both summarise an output; they can disagree |
| 4 | Three vector readers | `footprints._read_any`, `polygons._read_vector`, and inline logic in `tile_index.load_tile_index` each re-implement format and S3 handling |
| 5 | `print` instead of logging | 23 bare `print` calls in the pipeline; `utils/logging.py` exists and has 0% test coverage |
| 6 | Project constants in library code | `coiled_env.py` hard-codes the GitHub URL, `us-east-2` and the environment name; `S3Config.region` duplicates it and is never read |
| 7 | Dead or half-wired configuration | `io.s3.*` unused; `panel_plane.uncertainty_method: covariance` accepted but not implemented; `scikit-learn` declared, never imported; `[whitebox]` extra unused |
| 8 | Archived code in the hot path | `classify/` and 7 `MountingRule*` config classes ship in the core package and the config hash although disabled |
| 9 | Types | `mypy` reports 88 errors in 20 files; rows and manifests are bare dicts |
| 10 | CI fails its own lint step | `ruff check .` reports 26 findings (29 before 0.2.0); CI runs on Linux only although development is on Windows |
| 11 | Untested surfaces | `cli.py` 0%, `coiled_env.py` 0%, `io/_localize.py` 23%, `io/lidar.py` 61%, `runner.py` 69% line coverage |
| 12 | Tests share helpers by importing another test module | `test_vintage_and_inputs.py` imports from `test_runner_smoke.py`; no `conftest.py` |
| 13 | Report module mixes concerns | `report/build.py` (651 lines): loading, headline numbers, prose, HTML, Markdown and dataset export together; HTML built by string concatenation |
| 14 | Private names used across modules | `tile_task` imports `plane_fit._failed_fit`; `cli` imports `runner._aggregate` |
| 15 | Stale documents and scripts | `HANDOFF.md` (238 lines, May); the PRD and working paper describe the SAM3 atlas and mounting labels; `scripts/` holds spikes with machine-specific absolute paths; `_review_bench.py` is superseded by `pv-geom report` |
| 16 | Line endings | Mixed CRLF/LF with no `.gitattributes`; every commit warns |
| 17 | Long functions elsewhere | `extract_roof_plane` 154 lines; `process_tile_group` 141; `cli.run` 94 lines / 20 parameters; `fit_plane_ransac` 90 with a Python loop over iterations |

### 5.2 Target shape

```
pv_geom/
  api.py              run(), load(), report(): the public surface (I2)
  cli/                one module per command; argument parsing only
  config/             models split by concern; study-area inputs block
  errors.py           typed exceptions with remedies (I4)
  schema.py           schema, versions, dictionary
  io/
    vector.py         one reader for polygons, footprints, tile index
    pointcloud/       laz.py, (copc.py, ept.py), classes.py, units.py
    storage.py        fsspec access, cache, listing (replaces _localize)
    output.py
  measure/            was geometry/: plane.py, roof.py, heights.py, segments.py, index.py
  presence.py         standoff screen + vintage -> geometry_basis (was split across tile_task and vintage)
  pipeline/
    plan.py           inputs -> tile groups -> work plan (pure, testable)
    worker.py         one tile group -> rows; a pipeline of small steps over a PolygonContext
    executor.py       serial / local / coiled behind one interface
    run.py            plan + execute + write + manifest, ~100 lines
    summary.py        the one implementation of run statistics (used by manifest and report)
  report/
    data.py  stats.py  figures/  text.py  render/{html,markdown,quarto,docx}.py  export.py
  experimental/
    mounting/         archived classifier and its config, outside the core config hash
```

### 5.3 Refactoring stories

**Progress (2026-10-07, branch `v0.3-refactor`).** Done: R1–R13, with these notes.
R1: the real-data benchmarks are kept out of the repository (it is public and the
polygon layers are unpublished), so CI runs a synthetic golden benchmark and the
Phoenix/Delaware ones run locally. R4: one implementation of manifest statistics,
sharing definitions with the report; the report's weighted per-stratum tables remain
their own module. R7: `default.yaml` is pinned equal to the code defaults by a test
rather than generated. R9: `mypy` passes in standard mode with `check_untyped_defs`;
full `--strict` is not yet enforced. The package was not renamed `geometry/` ->
`measure/` as sketched in §5.2 — the churn bought nothing. Open: R14 (column naming,
an owner decision).

**Milestone 0.3 stories (2026-10-07, branch `v0.3-accounting`).** Done: B1, B2, B3, G1,
H1, H3, I3, I4. Notes. B1: polygons that reach no tile-group task are written to a
separate `part-unmeasured.parquet`, rewritten on every run, so that a failed group
can still be retried by `--resume`; `--bbox` / `--max-polygons` define the scope, and
polygons outside it have no row. B2: tiny and overlapping polygons are flagged and
still measured, not given a status of their own. I4: the "no ground class" check
belongs to D4 (run-start class inspection) and is not done. H3: progress is one log
line per finished group, not a live bar. Milestone 0.3 is complete apart from R14.

| ID | Story | Acceptance criteria | Size |
| --- | --- | --- | --- |
| R1 | Build the safety net first. | A4 benchmark and golden output exist; a `make check` (or `just check`) runs lint, types, tests and the benchmark. | M |
| R2 | Split `run_pipeline` into plan / execute / write. | No function over 80 lines in `pipeline/`; the plan is a pure function of inputs and is unit-tested without LiDAR; output targets (local, S3) and executors (serial, local, coiled) are classes behind small interfaces. | L |
| R3 | Turn `_build_row` into steps over a typed record. | A `PolygonMeasurement` dataclass replaces the dict; steps (`fit_panel`, `fit_roof`, `heights`, `screen`, `basis`) are separately testable functions; schema conversion happens in one place. | L |
| R4 | One summary implementation. | `pipeline/summary.py` feeds both the manifest and the report; `_aggregate` removed. | S |
| R5 | One vector reader, one storage layer. | Polygons, footprints and the tile index go through `io/vector.py`; all remote access through `io/storage.py` on `fsspec`. | M |
| R6 | Logging. | No `print` in `src/`; a run logger with a human console handler and a JSON-lines file handler; `--verbose`/`--quiet`. | S |
| R7 | Configuration cleanup. | Unused keys removed or implemented; project constants moved to config with detection; mounting config out of the core hash; `default.yaml` generated from the models (a test already pins them equal). | M |
| R8 | Move archived code to `experimental/`. | Core imports nothing from it unless enabled; its tests move with it. | S |
| R9 | Types. | `mypy --strict` on `pv_geom` passes (or a documented allow-list for third-party stubs); public functions fully annotated. | M |
| R10 | Report split. | Data, statistics, prose, rendering and export in separate modules; HTML and Markdown from templates; adding a renderer (F3) touches one module. | M |
| R11 | Test structure. | `conftest.py` with shared fixtures and synthetic-scene builders; CLI tested through Typer's runner; storage tested against a local fake; coverage gate in CI. | M |
| R12 | Repository hygiene. | `.gitattributes` normalises line endings; `HANDOFF.md` and superseded scripts archived under `docs/history/` and `scripts/archive/`; machine-specific paths removed; integration test reads its inputs from environment variables; PRD marked historical. | S |
| R13 | Public/private boundaries. | No cross-module import of an underscore name; `__all__` on public modules. | S |
| R14 | Naming pass. | `panel_*` columns describe "the fitted surface", which is not always a panel; decide (§8) whether to rename for 1.0 while the schema can still change. | S |

## 6. Quality gates (CI, all platforms)

| Gate | Threshold |
| --- | --- |
| Lint (`ruff check`, `ruff format --check`) | 0 findings |
| Types (`mypy`) | 0 errors |
| Unit tests | pass on Linux, Windows, macOS; Python 3.11–3.13 |
| Line coverage | ≥ 90% overall; no module below 75% |
| Regression benchmark (A4) | no row moves beyond tolerance |
| Reproducibility (G5) | identical output hash across two platforms |
| Docs build | no warnings; tutorial executed |
| Performance budget | per-polygon time and peak memory on the benchmark within 10% of the recorded baseline |

## 7. Sequence

| Milestone | Theme | Contents | Exit |
| --- | --- | --- | --- |
| **0.3** | Solid ground | R1–R13; B1–B3; H1, H3; I3, I4; G1 | Same numbers as 0.2.0 on the benchmark; gates green; one row per input polygon |
| **0.4** | Usable by config | I7 (inputs in config), E6 (true-north azimuth), I2 (Python API), I1 (sample + demo), D2, D4, H2, F5, F7, G2 | A run is `pv-geom run --config area.yaml`; demo works offline |
| **0.5** | Measurement depth | C1 (`unscreened` < 20%), E1 (segments), D3 (feet), H4, H5, F1, F2, G4 | Multi-facet polygons measured as segments |
| **0.6** | Anyone, anywhere | D1 (third study area), I5 docs, G5 | Third study area runs unmodified; tutorial works |
| **1.0-rc** | Evidence and release runs | A1–A5 (needs truth data), H6 full Phoenix + Delaware runs from the final configs, schema freeze, paper rebuilt from report output | Definition of done in §1 |
| 1.x | Depth | Should/Could stories: D5–D7, E2–E5, C2–C6, F3, F4, F6, F8, F9, G3, H7, I6, I7 | — |

Rough total for the Must set: 14–18 working weeks for one person (my estimate; the two
largest uncertainties are sourcing truth data for A1 and how hard C1 turns out to be).

## 8. Decisions needed from the project owner

**Decided 2026-10-07 (owner):**

- *Truth data (1):* none is available at present; to be thought through. Stories A1, A2
  and A5 are therefore **deferred**, and the accuracy claims in §1's definition of done
  stay open until a source exists. A3 (basis validity against the permit join) needs a
  full Phoenix output, so it waits with H6.
- *Full reruns (4):* **not now** — to be done once the package is finalised. H6 moves to
  the end of the sequence.
- *Inputs:* runs are to be driven **from a config** that names the polygon and LiDAR
  inputs; the final Phoenix and Delaware configs will point at the layers chosen for the
  final analysis. Story I7 is promoted from Should to **Must** and done first.
- *Direction:* continue improving the package.
- *Azimuth reference (added 2026-10-07, owner):* azimuth must be defined explicitly as
  true north, distinguishing grid north, with a cross-CRS test. New story **E6** (Must);
  A1 and D3 amended to depend on it.

Revised sequence in §7: the "evidence" work that needs neither truth data nor full
runs (C1, E1) stays; A1/A2/A5/A3/H6 move to a final "evidence and release runs" step
before 1.0.

1. **Truth data for A1.** Which source of independently known tilt/azimuth is obtainable?
   This gates the whole "evidence" milestone.
2. **Segments (E1).** Add segment rows as a second table (clean, two files) or widen the
   polygon table (one file, awkward beyond two planes)? Recommendation: second table.
3. **Column naming (R14).** Rename `panel_*` to something neutral (`surface_*` / `fit_*`)
   before the schema freezes? Recommendation: yes, with aliases for one minor version.
4. **Full reruns (H6).** Go-ahead and budget for Phoenix and Delaware on Coiled, writing to
   `v0.2.0/`-style versioned prefixes.
5. **License and DOI.** MIT was drafted; Zenodo metadata needs authors, ORCID, funding.
6. **Mounting classification.** Stays archived through 1.0 (assumed here). Confirm.
7. **Third study area (D1).** Which one? Ideally unlike both existing areas: foot-based
   LiDAR, or with class 6 present, or satellite-derived polygons.
8. **Sample data licence (I1).** A clip of public 3DEP LiDAR is unencumbered; the polygon
   sample must be one we may redistribute.

## 9. Not in 1.0

- Detecting PV (pv-geom consumes polygons; it does not make them).
- Mounting-type classification (archived; `experimental/`).
- Energy yield modelling beyond the optional orientation factor (E5).
- Change detection across multiple LiDAR epochs.
- A hosted service or web application.
