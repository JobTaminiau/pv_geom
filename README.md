# pv-geom

Geometric attribute extraction for solar PV polygons from co-temporal classified LiDAR.

Given a vector inventory of PV polygons and a multi-terabyte LAZ archive on S3, `pv_geom` writes a polygon-keyed GeoParquet table containing tilt, azimuth, mounting type, roof orientation, panel–roof angle, height-above-ground/roof, and quality/provenance metadata. Designed to run on a Coiled-managed Dask cluster with a single-machine `LocalCluster` fallback.

First deployment: Metropolitan Phoenix (~100k+ polygons, ~15–18 pts/m² LiDAR).

## Status

M1 (scaffolding) — see `STATUS.md`. Full PRD in `docs/pv_geom_PRD.md`.

## Quickstart (dev)

```bash
uv sync --extra dev
uv run pv-geom version
uv run pv-geom validate-config configs/phoenix.yaml
```

The `pv-geom run` command exists but raises until M6 lands.

## Layout

```
src/pv_geom/
  cli.py              Typer CLI
  config.py           Pydantic settings models
  schema.py           output schema (pyarrow)
  quality.py          quality flags + confidence helpers
  provenance.py       run manifest + config hashing
  io/                 polygons, footprints, tile index, LAZ + S3
  geometry/           plane fitting, multi-plane, roof plane, heights
  classify/           rules-based mounting classifier (+ ABC for swap-in ML)
  pipeline/           partitioner, per-tile-group task, Dask runner
  utils/              CRS helpers, JSON-line logging
configs/              default.yaml, phoenix.yaml, coiled.yaml
tests/                unit/, integration/
docs/                 PRD
```
