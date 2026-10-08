# Tutorial

This takes about ten minutes and needs no data and no network connection. You
will install pv-geom, measure a small generated study area, and read the
results.

## 1. Install

pv-geom needs Python 3.11 or 3.12. With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/JobTaminiau/pv_geom.git
cd pv_geom
uv sync
```

Or with pip, into an environment of your own:

```bash
pip install git+https://github.com/JobTaminiau/pv_geom.git
```

Check it:

```bash
uv run pv-geom version
```

(With pip, leave out `uv run` here and below.)

## 2. Run the demo

```bash
uv run pv-geom demo --out my_demo
```

This writes a synthetic study area, fifteen arrays whose true tilt and
orientation are known, as a LiDAR tile and a polygon layer. It then measures
and reports on them. You get:

```
my_demo/
  demo.yaml              the config that describes the run
  inputs/                the generated LiDAR tile and polygon layer
  output/
    manifest.json        what was run, on what, with which versions
    part-00000.parquet   the dataset
    report/
      report.html        open this
      report.md          the same, for pasting into a manuscript
      methods.md         a methods paragraph with this run's numbers
      tables/            every statistic as CSV
      figures/           PNG, PDF and SVG
      dataset/           the release dataset with its data dictionary
```

Open `my_demo/output/report/report.html` in a browser.

## 3. Read the config

`my_demo/demo.yaml` is the whole description of the run:

```yaml
study:
  name: pv-geom demo (synthetic)
  output: output                 # relative to this file

inputs:
  polygons: inputs/polygons.parquet
  lidar_prefix: inputs           # no tile index: it is read from the tile headers

vintage:
  polygon_vintage: "2023"        # the LiDAR date is measured from the tile
```

Run it again yourself:

```bash
uv run pv-geom run --config my_demo/demo.yaml --no-dask
```

Notice the warning about the **vintage gap**: the polygons are dated 2023 and
the LiDAR was flown in 2022, so any array built in between is not in the point
cloud. This is the normal situation with real data, and the reason for the
`geometry_basis` column.

## 4. Look at the rows

```bash
uv run pv-geom describe-output my_demo/output
```

Or in Python:

```python
import pv_geom

rows = pv_geom.load("my_demo/output")
print(rows[["polygon_id", "status", "geometry_basis", "tilt_deg", "azimuth_deg",
            "tilt_unc_deg", "recommended"]])
```

Three columns tell you how to read each row:

- `status` says whether the polygon was measured at all.
- `geometry_basis` says what the tilt and azimuth rest on.
- `recommended` is true for rows that are measured, rest on evidence that the
  array was there, and carry no flag marking them as unreliable.

For array geometry, start from `recommended == True`.

## 5. Check an answer

The demo's arrays have known geometry, so you can see the measurement work.
The first array was generated at a tilt of 20 degrees facing 180 degrees:

```python
print(rows.loc[0, ["tilt_deg", "azimuth_deg", "tilt_unc_deg"]])
```

You should see about 20.00 and 180.03.

## What next

- [Run your own area](how-to/new-study-area.md)
- [What the basis values mean](concepts/vintage-and-basis.md)
- [Every column](reference/columns.md)
