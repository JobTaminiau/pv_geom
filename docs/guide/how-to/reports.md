# Reports

```bash
pv-geom report out/my_area
```

writes to `out/my_area/report/` (or `--out`). A local `pv-geom run` builds the
report by itself unless you pass `--no-report`.

## What is in it

| Path | Contents |
| --- | --- |
| `report.html` | Self-contained: findings, vintage statement, every figure and table, methods, data dictionary |
| `report.md` | The same as Markdown with linked figures |
| `methods.md` | A methods paragraph filled in with this run's parameters and dates |
| `summary.json` | Headline numbers, machine-readable |
| `tables/*.csv` | Every statistic, for every group of polygons and both weightings |
| `figures/` | PNG at 300 dpi, plus PDF and SVG with editable text |
| `dataset/` | The release dataset: GeoParquet, CSV, facet table, data dictionary, definitions, checksums |

## Which polygons, and weighted how

Every table file carries every combination. Two options choose what the report
leads with:

```bash
pv-geom report out/my_area --headline panel --weight area
```

- `--headline` is the group of polygons: `panel` (rows that rest on evidence
  the array was there; the default when there are enough), `all_fitted`,
  `surface_unresolved` or `unscreened`.
- `--weight` is `area` (each polygon counts by its array surface) or `count`.

The median tilt and the shares facing each direction come with 95% bootstrap
intervals. They express sampling variability between polygons, not measurement
error.

## By district

```bash
pv-geom report out/my_area --regions districts.gpkg --region-col name
```

adds a table and a map by region, and a `region` column in the release
dataset. A polygon belongs to the region its centroid falls in. Regions with
fewer than ten polygons in the headline group are counted but not described.

## Two runs side by side

```bash
pv-geom report out/phoenix out/delaware --compare --out out/comparison \
    --label Phoenix --label Delaware
```

writes overlaid tilt and orientation profiles and a comparison table. Use it
for two study areas, or for one area at two versions of the package.

## From Python

```python
import pv_geom

report = pv_geom.report("out/my_area", headline="panel", weight="area",
                        regions="districts.gpkg", region_col="name")
print(report.html, report.summary["n_fitted"])

pv_geom.compare_runs(["out/phoenix", "out/delaware"], "out/comparison")
```

## Older outputs

Reporting works on outputs written by earlier versions. Their column names are
brought up to date as they are read. For outputs from before dates were
recorded, give them: `--polygon-vintage 2024-10 --lidar-date 2020-11`.
