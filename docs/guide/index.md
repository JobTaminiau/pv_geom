# pv-geom

pv-geom measures the geometry of photovoltaic arrays from airborne LiDAR. You
give it **polygons** that outline arrays and the **LiDAR tiles** that cover
them; it gives back, for every polygon, the tilt and the direction it faces,
how sure it is, and whether what it measured was really the array.

```
polygons (with the date of their imagery)  ─┐
                                            ├─►  pv-geom run  ─►  dataset + report
LiDAR tiles (date measured from the data)  ─┘
```

## What you get

- **A dataset**, one row per polygon: tilt, azimuth from true north,
  uncertainty, height, the roof beneath, and what happened to polygons that
  could not be measured.
- **A report**: tilt and orientation profiles, summary statistics with
  confidence intervals, figures sized for a journal, a methods paragraph.
- **An honest basis for every number.** Polygons usually come from imagery
  that is newer than the LiDAR. An array built in between is in the polygon
  layer but not in the point cloud, and its row measures the bare roof.
  pv-geom tells those apart: see [vintage and basis](concepts/vintage-and-basis.md).

## Where to start

| If you want to | Read |
| --- | --- |
| See it work in five minutes | [Tutorial](tutorial.md) |
| Run your own area | [A new study area](how-to/new-study-area.md) |
| Run a large area on a cluster | [Running in the cloud](how-to/cloud.md) |
| Shape the report | [Reports](how-to/reports.md) |
| Check results against outside data | [Validation](how-to/validate.md) |
| Prove a dataset reproduces | [Reproducing a run](how-to/reproduce.md) |
| Know what a column means | [Output columns](reference/columns.md) |
| Know what it cannot do | [Limits](concepts/limits.md) |
