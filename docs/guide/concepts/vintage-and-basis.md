# Vintage and basis

## The problem

Array polygons usually come from recent imagery. LiDAR is flown rarely, and
the latest collection for an area is often years older. So the two inputs have
different dates:

```
2020 ──── LiDAR flown ───────────────────── 2024 ──── imagery ────►
                    │                              │
                    └── arrays built in here ──────┘
                        are in the polygons
                        but not in the point cloud
```

For an array built in that window, the returns inside its polygon come from
whatever was there before: the bare roof. A plane fitted to them is a real
measurement of the wrong thing.

## What pv-geom does about it

1. **It measures the LiDAR date**, per tile, from the GPS time of the points.
   The date in a LAS header is when the file was delivered.
2. **You declare the polygon date** (`vintage.polygon_vintage`). A year or a
   month counts as its last day, so the gap is never understated.
3. **It looks for physical evidence** that the array was in the point cloud.
4. **It records, per row, what the numbers rest on**: the `geometry_basis`
   column.

## The evidence

**A standoff from the roof.** Modules sit a few centimetres above a roof. A
second plane is fitted to the roof around the polygon; if the array's plane is
at least 5 cm above it, modules were there. Two planes each known to about
2 cm cannot resolve less.

The test is only meaningful against the plane the array sits on. On a hip or
cross-gable roof the reference can come out as a neighbouring facet: pitched,
and at an angle to the array. Height "above" that plane is tens of centimetres
either way and says nothing, so where the reference is pitched and more than
6 degrees from the array's plane the test is not applied
(`roof_reference_not_parallel`). A flat reference under a tilted plane is a
rack on a flat roof and is tested as usual.

Expect about half of flush-mounted arrays to pass even when everything is
right: a flush array stands roughly 6 cm above its roof, and the threshold is
5 cm.

**Open ground around it.** Ground mounts and canopies have no roof beneath
them. They are recognised by what surrounds them: a band around the polygon
that is mostly ground returns means an elevated structure standing in the
open.

**The dates themselves.** If the polygons are no newer than the LiDAR, the
array existed when it was flown.

**A record of the installation.** If the polygon layer carries per-polygon
dates from permits or interconnection, name them in the config:

```yaml
vintage:
  installed_by_column: signed_off              # complete by this date
  not_installed_before_column: first_permit    # did not exist before this date
```

An array complete on or before the LiDAR date was there. One that did not
exist until after it was not, and its row measured the roof. Between the two
the record says nothing. Height above the roof, where it can be measured,
still decides first; a row confirmed by height although its record says it
came later is flagged `standoff_before_install_date`.

## The basis values

| `geometry_basis` | The fitted plane is | Use for array geometry |
| --- | --- | --- |
| `panel_confirmed` | resolvably above the roof plane: modules were in the point cloud | yes |
| `panel_by_install_date` | not separable by height, but the layer records the installation as complete by the LiDAR date | yes |
| `panel_by_vintage` | not separable by height, but the polygons are no newer than the LiDAR | yes |
| `free_standing` | an elevated structure in open ground: a ground mount or canopy, present at the LiDAR date | yes |
| `surface_unresolved` | coincident with the roof, and the polygons are newer: a flush-mounted array or the bare roof before it | only if you can assume flush mounting |
| `surface_before_install` | the roof before the array: the layer records the installation as later than the LiDAR, and nothing stands above the roof | only if you can assume flush mounting |
| `unscreened` | untested: no usable roof reference and the polygons are newer | with caution |
| `no_fit` | not fitted | no |
| `not_measured` | the polygon never reached measurement; see `status` | no |

`recommended` is true for the first four, minus rows flagged as unreliable.

## Why unresolved rows are not wrong rows

A flush-mounted array is parallel to its roof. If its row measured the bare
roof, the tilt and azimuth are still the array's. That is why
`surface_unresolved` rows are kept and reported separately rather than
discarded: they are right for flush mounting and wrong for tilted racks on
flat roofs, and the data cannot say which a given row is. The same holds for
`surface_before_install`, where the row is known to be the roof.

## Two things the evidence does not show

- `free_standing` establishes the structure, not the modules. A carport roofed
  with modules after the flight also qualifies. Its geometry is still the
  array's, because modules lie flush on such structures.
- Removal is not detected. An array older than the LiDAR is assumed to have
  still been there when it was flown.
