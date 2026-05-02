# pv-geom

Geometric attribute extraction for solar PV polygons from co-temporal classified LiDAR.

Given a vector inventory of PV polygons and a multi-terabyte LAZ archive on
S3, `pv_geom` writes a polygon-keyed GeoParquet table containing tilt,
azimuth, mounting type, roof orientation, panel–roof angle,
height-above-ground / height-above-roof, and quality/provenance metadata.
Designed to run on a Coiled-managed Dask cluster with a single-machine
`LocalCluster` fallback for development. First deployment: metropolitan
Phoenix (~100k+ polygons, ~10 pts/m² LiDAR).

Full PRD in `docs/pv_geom_PRD.md`. Per-milestone log in `STATUS.md`.

## Status

M1 (scaffolding) → M7 (Coiled integration) complete. M8 hardening in
progress; see `STATUS.md` for the open list. Validated end-to-end on real
Phoenix data on both `LocalCluster` and Coiled, including a 1000-polygon
benchmark that produces bit-for-bit identical output across the two
backends.

143 unit tests pass; one integration test runs against real Phoenix data
when `RUN_INTEGRATION=1` is set and the prerequisite cache files are
present.

## Quickstart

```bash
# install (uv recommended; pip works too)
uv sync --extra dev

# sanity checks
uv run pv-geom version
uv run pv-geom validate-config configs/phoenix.yaml

# run on real data (paths illustrative — see "Inputs" below)
uv run pv-geom run \
  --config configs/phoenix.yaml \
  --polygons s3://your-bucket/sam3/atlas/latest.parquet \
  --tile-index s3://your-bucket/lidar/tile_index.zip \
  --lidar-prefix s3://your-bucket/lidar_data \
  --footprints s3://your-bucket/fema/az.geoparquet \
  --output ./out_phoenix \
  --bbox 432000 3719000 432900 3719900 \
  --max-polygons 20 \
  --local
```

`--local` forces the LocalCluster backend regardless of what the config
says. Drop it to use the backend selected in the config (`coiled` for
`configs/phoenix.yaml`).

For dev iteration without a cluster, add `--no-dask` to run the same
pipeline serially in-process.

## Inputs

| Input | Format | Notes |
|---|---|---|
| PV polygons | GeoParquet | Must have a `polygon_id` (or `detection_id`) column. MultiPolygons are exploded into one row per part with `parent_polygon_id` linking back. |
| Building footprints | GeoParquet / GPKG / SHP | FEMA's `build_id` is normalized to canonical `building_id`. Synthesizes `auto_<i>` ids when neither column is present. |
| LiDAR tile index | GeoParquet / GPKG / SHP / zipped SHP | Auto-detected from extension. The default tile-id column is `Name`; override with `--tile-id-col`. |
| LAZ tiles | classified ASPRS LAZ on S3 (or local) | One file per tile, named per a configurable template — Phoenix uses `USGS_LPC_AZ_MaricopaPinal_2020_B20_{name}.laz`. Class assignments configurable in `io.classification`. |

S3 reads are cached locally (`$TMP/pv_geom_cache/`). Missing tiles
(404s) are tolerated: each group's fetch list is filtered before
dispatch; if the *primary* tile is missing, the group emits zero rows.

## Outputs

A GeoParquet partition file per tile group plus a JSON `manifest.json`
under the output prefix. Per-row schema (canonical source:
`src/pv_geom/schema.py`):

- **Geometry**: `polygon_id`, `geometry` (WKB), `area_m2`, `aspect_ratio`
- **Panel fit (M3)**: `panel_tilt_deg`, `panel_azimuth_deg`, `panel_rmse_m`,
  `n_points_panel`, `n_inliers_panel`, `panel_tilt_unc_deg`,
  `panel_azimuth_unc_deg`
- **Multi-plane (M5)**: `n_planes_detected`, `secondary_tilt_deg`,
  `secondary_azimuth_deg`
- **Roof plane (M4)**: `roof_tilt_deg`, `roof_azimuth_deg`, `roof_rmse_m`,
  `panel_roof_angle_deg`, `on_building`, `building_id`
- **Heights (M4)**: `height_above_ground_m`, `height_above_roof_m`
- **Mounting (M5)**: `mounting_type`, `mounting_confidence`, `mounting_rule`
- **Quality + provenance**: `flags`, `lidar_tile_ids`, `pkg_version`,
  `config_hash`, `run_id`, `partition_id`

`mounting_type` is one of `flush_mount_rooftop`, `tilted_rack_rooftop`,
`ground_mount_fixed`, `ground_mount_tracker_suspected`, `carport`,
`ambiguous`. `mounting_rule` records which rule (R1–R6) fired.

`flags` is a list drawn from `low_density`, `poor_fit`, `near_horizontal`,
`east_west_rack`, `tracker_suspected`, `roof_insufficient`, `roof_complex`.

The manifest captures aggregate stats (mounting-type counts, RMSE
percentiles, flag counts), the config hash, the input URIs, the cluster
spec, the run id, and a UTC timestamp.

## Configuration

`configs/default.yaml` is the canonical schema; `configs/phoenix.yaml` and
`configs/coiled.yaml` are sparse overlays. Pydantic models in
`src/pv_geom/config.py` validate everything at load time. Every config is
hashed (sha256 of the model dump) and stamped into both the per-row
`config_hash` and the manifest, so any drift between runs is detectable.

Key knobs you'll likely touch:

- `crs.target` — coordinate system everything is reprojected into.
  Phoenix is `EPSG:6341` (NAD83(2011) / UTM 12N, metres).
- `panel_plane.{ransac_threshold_m, min_inlier_frac, min_points}` — RANSAC
  shape and "is the fit any good" floor.
- `io.classification` — ASPRS class assignments and the class-1 fallback
  height (USGS LPC tiles often lack class 6; `pv_geom` falls back to
  class-1 returns above local ground).
- `mounting_rules` — thresholds for each of R1–R6 in `classify/rules.py`.
- `compute.backend` — `local` or `coiled`.

## Compute backends

### Local

Default in `configs/default.yaml`. Spins up a `distributed.LocalCluster`
or runs serially with `--no-dask`. No setup beyond `uv sync`.

### Coiled

Phoenix config defaults to `compute.backend: coiled`. One-time setup:

```bash
# 1) authenticate
coiled login

# 2) build the software environment (idempotent; ~3 min)
uv run python -c "from pv_geom.coiled_env import ensure_software_env; ensure_software_env()"

# 3) make sure the repo is published so workers can pip-install pv_geom
#    git+https://github.com/JobTaminiau/pv_geom.git@main is what
#    install_pv_geom_on_workers expects.
```

The cluster is created in the region defined by `coiled_env.REGION` (this
project: `us-east-2`, matching the LiDAR bucket). `pv_geom` is installed
on **both the scheduler and the workers** at cluster start — the
scheduler needs it because Dask deserializes the task graph there before
dispatch.

#### Cross-account S3 access

Coiled BYOC workers run under an IAM role in your AWS account
(`coiled-<your-coiled-username>`). To read from a bucket owned by
another account (or a same-account bucket without an IAM identity policy
allowing the role), grant the role bucket-side access. Minimum bucket
policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"AWS": "arn:aws:iam::<your-account-id>:role/coiled-<username>"},
    "Action": ["s3:GetObject", "s3:ListBucket", "s3:GetBucketLocation"],
    "Resource": ["arn:aws:s3:::<bucket>", "arn:aws:s3:::<bucket>/*"]
  }]
}
```

Note there's no `s3:HeadObject` action in IAM; HEAD on objects is
authorized under `s3:GetObject`.

`scripts/_coiled_aws_probe.py` is a single-task diagnostic that runs
`sts.get_caller_identity` plus a battery of S3 calls on a real Coiled
worker — useful for confirming role and bucket-access state before
debugging the full pipeline.

## Phoenix-specific assumptions

- **CRS**: EPSG:6341 (NAD83(2011) / UTM 12N, metres). Verified in spike
  against USGS LPC AZ MaricopaPinal 2020 metadata.
- **ASPRS class fallback**: USGS LPC tiles in this dataset have classes
  1, 2, 7 only — no class 6 ("building"). Configured fallback uses
  class-1 returns above ground (`fallback_height_above_ground_m: 1.5`).
- **Density**: tiles deliver ~9–11 pts/m². The `min_density_pts_per_m2`
  floor of 3 leaves comfortable margin; the `min_points: 30` floor
  protects RANSAC robustness on small polygons (~3 m² and below).
- **FEMA AZ footprints**: ~92% of atlas polygons intersect at least one
  FEMA AZ footprint; the remaining ~8% sit in genuine FEMA gaps and
  surface as `on_building=False`. FEMA's `shape_area` column is all
  zeros — always compute `geometry.area` instead. Source bucket:
  `s3://free-research-data/national/fema_footprints/az.geoparquet`.

## Tests

```bash
# unit tests (synthetic data; ~3 s)
uv run pytest tests/unit/

# integration test (real Phoenix data; ~30–45 s; requires cached prereqs)
RUN_INTEGRATION=1 uv run pytest tests/integration/test_phoenix_subset.py -v
```

The integration test is gated on both `RUN_INTEGRATION=1` and the
presence of cached prerequisite files (atlas parquet, FEMA AZ
geoparquet, USGS tile index, the relevant LAZ tile). It runs the
pipeline on a 900x900 m east-valley bbox + 20 polygons through a
2-worker `LocalCluster` and asserts schema correctness, southern-azimuth
concentration, RMSE ≤ 10 cm, and manifest sanity.

## Layout

```
src/pv_geom/
  cli.py              Typer CLI (`pv-geom run`, `validate-config`, ...)
  config.py           Pydantic config models + hash
  schema.py           Output schema (pyarrow source of truth)
  quality.py          Quality flags + confidence helpers
  provenance.py       Run manifest + config hashing
  coiled_env.py       Coiled software-env spec, cluster + worker bootstrap
  io/                 Polygons, footprints, tile index, LAZ + S3 cache
  geometry/           Plane fitting, multi-plane, roof plane, heights
  classify/           Rules-based mounting classifier (+ ABC for ML swap-in)
  pipeline/           Partitioner, per-tile-group task, Dask runner
  utils/              CRS helpers, JSON-line logging
configs/              default.yaml, phoenix.yaml, coiled.yaml
tests/                unit/, integration/
scripts/              spike, eda, benchmark prototypes
docs/                 PRD
```

## License

Not yet selected. The `pyproject.toml` classifier reads
`License :: Other/Proprietary License` as a placeholder; until a real
LICENSE file is added, treat the code as all-rights-reserved.

## Citation

Zenodo DOI to be minted on the first tagged release. Until then, cite
this repo URL.

## Acknowledgments

Built at FREE. LiDAR data: USGS 3DEP / LPC. Building footprints: FEMA
USA Structures. Phoenix LiDAR is hosted under the ASU NSF Phoenix bucket.
