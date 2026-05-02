# pv-geom — handoff (2026-05-01 evening)

Pick this up tomorrow; everything below the **PICK UP HERE** marker is the
shortest path to a successful 1000-polygon benchmark run.

## Where we are

- M1–M7 + `--resume` complete (see `STATUS.md` for the per-milestone log).
- **142 unit tests pass**, 1 integration test passes when `RUN_INTEGRATION=1`.
- Pipeline runs end-to-end on real Phoenix data (verified on 5 polygons in M6,
  20 polygons in the M7 integration test).
- CLI `pv-geom run` is functional with `--bbox`, `--max-polygons`, `--resume`,
  `--dry-run`, `--no-dask`, `--local`.
- Recent quality-of-life changes shipped today:
  - **Cache dir unified to `pv_geom_cache`** (was split between
    `pv_geom_io_cache` and `pv_geom_spike_cache`). Existing spike-cached LAZ
    files were copied into the new dir; future runs share one cache.
  - **`RemoteFileMissing` exception** in `_localize` distinguishes 404 from
    transport errors; the runner gracefully drops missing tiles from each
    group's fetch list and skips groups whose primary tile is missing.
  - **Serial pre-warm before Dask dispatch** (`runner.py`): every unique S3
    URI is `localize`-d once in the main process so workers don't race on
    the AWS SSO token cache file (Windows-specific issue with parallel boto3
    + SSO; not a concern on Coiled, which uses IAM roles).
  - **`botocore[crt]` added to dev extras** so `aws login`-style SSO creds
    work in the package venv.

## What didn't work yet

A 1000-polygon benchmark on tile `w0442n3681` (~150 MB LAZ, ~3,000 polygons in
1 km², ~50M class-1 returns). Two attempted runs:

1. **Run 1** (`out_bench_1k`, bbox spanning multiple uncached tiles): crashed
   with **AWS SSO `PermissionError`** when 4 Dask workers raced on the
   `~/.aws/login/cache/<token>.json` file. Fixed by adding the serial
   pre-warm step.
2. **Run 2** (after pre-warm fix, on `w0442n3681` + `w0442n3680`): pre-warm
   succeeded, downloaded both LAZ tiles (118 MB + 100 MB). Workers then ran
   for ~50 minutes before all four were OOM-killed by Dask's nanny — Dask
   logged `Worker failed to heartbeat for 2839s; attempting restart` four
   times across two hours, then gave up with a `KilledWorker`. Final state:
   no partition files, exit code 1.

### Root cause

`process_tile_group` (in `src/pv_geom/pipeline/tile_task.py`) re-filters the
50M-point class-1 array **inside the per-polygon loop**:

```python
for _, row in polygons.iterrows():
    ...
    panel_class_pts, _ = _select_panel_class_points(all_pts, cfg, (cx, cy))
    in_panel = clip_points_to_polygon(panel_class_pts[:, :3], poly, ...)
```

`_select_panel_class_points` does:

1. `cls = pts[:, 3].astype(int)` — copies the whole class column (50M ints).
2. `pts[cls == fallback]` — copies all class-1 returns (~33M rows × 4 cols).
3. Local-disk neighbourhood ground-z lookup — another scan of the ground
   subset.
4. HAG threshold — filters again to a new array.

For 1000 polygons that's 1000 round-trips through 1.5+ GB of point data. GC
churn looks indistinguishable from a hang to Dask's worker heartbeat, then
the OS starts paging and workers get OOM-killed.

A second contributing factor: `process_tile_group` is a **single Dask task
per tile group**. With this bbox we got 1000 polygons → 2 tile groups, so
two workers held all 50M+ points in memory each while three other workers
sat idle. No intra-group parallelism.

## PICK UP HERE — shortest path to a working 1000-polygon benchmark

### Step 1 — refactor `process_tile_group` to hoist the class filter

In `src/pv_geom/pipeline/tile_task.py`, replace the per-polygon class-filter
call with a tile-group-level pre-filter, then iterate polygons against the
already-subset arrays. Sketch:

```python
def process_tile_group(...):
    # Load tiles
    chunks = []
    for tid in fetch_tile_ids:
        uri = tile_uri_map.get(tid)
        if uri is None:
            continue
        pts, _ = read_tile_points(uri)
        chunks.append(pts)
    all_pts = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 4))

    # ---- NEW: filter ONCE, not per polygon -----------------------------
    cls_col = all_pts[:, 3].astype(np.int16)
    cfg_cls = cfg.io.classification
    ground_xyz = all_pts[cls_col == cfg_cls.ground_class][:, :3]

    if (cls_col == cfg_cls.panel_class_primary).any():
        panel_pts_all = all_pts[cls_col == cfg_cls.panel_class_primary][:, :3]
    else:
        candidate = all_pts[cls_col == cfg_cls.panel_class_fallback][:, :3]
        if len(ground_xyz) and len(candidate):
            gz_global = float(np.median(ground_xyz[:, 2]))
            panel_pts_all = candidate[
                candidate[:, 2] > gz_global + cfg_cls.fallback_height_above_ground_m
            ]
        else:
            panel_pts_all = candidate

    del all_pts, cls_col   # ~1.5 GB freed

    # Per-polygon: just clip the pre-filtered panel set.
    rows = []
    for _, row in polygons.iterrows():
        poly = row.geometry
        pid = str(row[polygon_id_col])
        in_panel = clip_points_to_polygon(
            panel_pts_all, poly, erosion_m=cfg.panel_plane.erosion_m
        )
        # `_build_row` needs to take ground_xyz + roof_input_pts directly
        # instead of all_points + cls; refactor its signature.
        rows.append(_build_row(
            polygon=poly, polygon_id=pid, cfg=cfg,
            config_hash=config_hash, run_id=run_id, partition_id=partition_id,
            panel_pts=in_panel,
            ground_xyz=ground_xyz,
            roof_input_pts=panel_pts_all,    # same pool, ring-clipped inside
            footprints=footprints,
            other_pv_polygons=...,
            contributing_tile_ids=...,
        ))
```

`_build_row` currently takes `all_points` and re-derives `ground_xyz`,
`roof_input_pts`, and the panel-class set inside. Change its signature to
take pre-filtered arrays. Then drop the now-redundant filtering blocks.

The `near = ground[abs(ground.x - cx) < 50 ...]` per-polygon spatial pre-filter
inside the panel-class fallback is **not** the bottleneck — it's the global
class-equality scans on 50M elements that thrash. Keeping a per-polygon
`height_above_ground` call against the full ground_xyz is fine; the function
already uses a 10 m search disk via numpy which is fast.

### Step 2 — verify with the M7 integration test first

Before re-running the 1000-polygon benchmark, make sure the refactor doesn't
break the M7 integration test (still gated on `RUN_INTEGRATION=1`):

```powershell
RUN_INTEGRATION=1 PYTHONUTF8=1 uv run pytest tests/integration/test_phoenix_subset.py -v
```

This runs against 20 polygons end-to-end and checks azimuth/RMSE plausibility.
If it passes, the refactor is safe.

### Step 3 — the benchmark itself

```powershell
PYTHONUTF8=1 uv run python -c "
import time
from pv_geom.config import PVGeomConfig
from pv_geom.pipeline.runner import run_pipeline
cfg = PVGeomConfig.from_yaml('configs/phoenix.yaml')
cfg.compute.backend = 'local'
cfg.compute.local.n_workers = 2          # lower than 4 to leave headroom
cfg.compute.local.threads_per_worker = 2

t0 = time.perf_counter()
manifest = run_pipeline(
    polygons_uri=r'C:\Users\job_t\code\free\pv_sam3\artifacts\atlas\latest.parquet',
    tile_index_uri=r'C:\Users\job_t\AppData\Local\Temp\tileindex\USGS_AZ_MaricopaPinal_1_2020_TileIndex.shp',
    lidar_prefix='s3://asu-nsf-phoenix/data/lidar_data',
    footprints_uri=r'C:\Users\job_t\AppData\Local\Temp\pv_geom_spike_cache\az.geoparquet',
    output_uri='./out_bench_1k',
    cfg=cfg,
    name_template='USGS_LPC_AZ_MaricopaPinal_2020_B20_{name}.laz',
    bbox=(442000, 3681000, 443000, 3682000),
    max_polygons=1000,
    use_dask=True,
)
print(f'TOTAL: {time.perf_counter() - t0:.1f} s')
print(f'manifest: {manifest}')
"
```

The two LAZ tiles (`w0442n3680.laz` 118 MB, `w0442n3681.laz` 100 MB) are
**already cached** at
`C:\Users\job_t\AppData\Local\Temp\pv_geom_cache\`, so pre-warm should
report two cache hits and skip downloads.

After the refactor, expected wall time is ~3–6 minutes (RANSAC dominates;
50 bootstraps × 1000 polygons × ~1 ms ≈ 50 s, plus roof fits + IO).

### Step 4 — if the refactor still isn't fast enough

The deeper fix is to **chunk polygons within a tile group** so one Dask task
handles a bounded number of polygons (say 100). That preserves intra-group
parallelism: 1000 polygons / 100 per task = 10 tasks across N workers.
`process_tile_group` would still load the points for the chunk's tile set,
but each task processes ≤100 polygons → bounded memory + quick worker
heartbeat. Implementation lives in `pipeline/partition.py` (split each
`TileGroup.polygon_ids` into N-sized chunks and emit a sub-task per chunk).

Defer this until you've measured the post-hoist refactor; it may be enough.

## Useful background tabs

- **Densest tiles _with LAZ_** (from today's S3 listing):
  - `w0442n3681` — 2,993 polygons (1 km²), LAZ available, **already cached**
  - `w0442n3680` — 2,703, available, **cached**
  - `w0443n3681` — 1,647, available
  - `w0413n3697` — 917, available
  - **Avoid** `w0371n3710` (4,929 polygons — 1 km² but **no LAZ** in the
    bucket), `w0375n3735` (969 polygons, **no LAZ**).
- Total LAZ coverage: 3,380 of 13,373 indexed tiles (~25%).
- Cached LAZ tiles in `~/AppData/Local/Temp/pv_geom_cache/`:
  `w0417n3684`, `w0418n3678`, `w0419n3696`, `w0431n3698`, `w0432n3719`,
  `w0442n3680`, `w0442n3681`. Total ~875 MB on disk.
- FEMA AZ footprints cached at
  `~/AppData/Local/Temp/pv_geom_spike_cache/az.geoparquet` (491 MB).
- Tile-index SHP extracted at
  `~/AppData/Local/Temp/tileindex/USGS_AZ_MaricopaPinal_1_2020_TileIndex.shp`.

## After the benchmark works

Remaining M8 items, in priority order:

1. **`pv-geom benchmark`** — formal CLI subcommand wrapping the script
   above; reports wall-time, peak RSS (psutil), per-polygon cost. Tracked
   in CI history.
2. **README + quickstart notebook** — the project is now functional but
   the only docs are STATUS/HANDOFF/PRD. A short README pointing at
   `pv-geom run --help` plus the FEMA + LAZ assumptions would help anyone
   else picking it up.
3. **Reference parquet for the M7 integration test** — committing a
   "blessed" 20-row parquet from a known-good run so the test does
   byte-equality (within a tolerance) instead of just plausibility.
4. **Zenodo metadata** for the first tagged release.
