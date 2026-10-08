# A new study area

A study area is one config file. Nothing in the code is specific to a place.

## What you need

| Input | Needed | Notes |
| --- | --- | --- |
| PV polygons | yes | GeoParquet, GeoPackage, GeoJSON, Shapefile or FlatGeobuf, in any CRS. An id column is optional. |
| The date of their imagery | strongly recommended | A year, a month or a day. Without it no row can be credited as present by date. |
| LiDAR tiles | yes | Classified LAZ, one file per tile, in a folder or under an `s3://` prefix. They need a ground class (ASPRS 2) and a projected CRS. |
| A tile index | no | Read from the tiles' own headers if you do not have one. |
| Building footprints | no | Improve the roof reference on dense blocks. |

LiDAR in feet, or in a different CRS from the one you want to work in, is
converted as it is read. The working CRS is always metric.

## 1. Vet one tile

```bash
pv-geom inspect-tile path/to/one_tile.laz
```

This prints the point classes present, the density, the CRS and units, and the
**flight dates measured from the points**. The date in a LAS header is the
delivery date, often a year or more later; pv-geom never uses it when GPS time
is available.

## 2. Write the config

```yaml
# my_area.yaml  (paths are relative to this file, or s3://)
study:
  name: My study area
  output: out/my_area

inputs:
  polygons: data/pv_polygons.gpkg
  lidar_prefix: data/lidar            # or s3://bucket/prefix
  # tile_index: data/tile_index.gpkg  # optional
  # footprints: data/buildings.gpkg   # optional
  # polygon_id_col: my_id             # optional

vintage:
  polygon_vintage: 2024-10            # when the polygons' imagery was captured
```

Only list what differs from the defaults. Every setting and its default is in
the [configuration reference](../reference/configuration.md).

If different polygons have different dates (a mosaic, or permit-dated
records), name the column instead:

```yaml
vintage:
  polygon_vintage_col: capture_date
```

## 3. Check before you run

```bash
pv-geom validate-config my_area.yaml
pv-geom run --config my_area.yaml --dry-run
```

The dry run plans the work, measures the LiDAR dates on a sample of tiles,
checks the point classes, and estimates the size and time. It warns if the
polygons are newer than the LiDAR.

## 4. Try a small piece

```bash
pv-geom run --config my_area.yaml --max-polygons 500 --output out/trial
```

`--bbox xmin ymin xmax ymax` (in the working CRS) limits it by area instead.

## 5. Run it

```bash
pv-geom run --config my_area.yaml
```

Partitions are written as each tile group finishes. If the run stops, the same
command with `--resume` keeps what is there and does the rest. It refuses to
continue an output made from different inputs or settings.

## Things that commonly need a setting

| Symptom | Setting |
| --- | --- |
| Many `no_fit` rows with `fit_failure = no_consensus` on noisy LiDAR | `panel_plane.ransac_threshold_max_m` (already 0.15; the report shows how many fits needed it) |
| The LiDAR has no building class | Nothing: unclassified returns above local ground are used automatically |
| The LiDAR's unclassified class is not 1 | `io.classification.panel_class_fallback` |
| Workers run out of memory | `compute.memory_budget_gb` (about half a worker's memory) |
| Polygons overlap one another | Dissolve them first; overlapping rows are flagged `overlaps_polygon` and left out of `recommended` |
