# Accuracy set: PVDAQ documented systems on public 3DEP LiDAR

Built 2026-10-07 with `scripts/accuracy_set.py`. Everything here derives from
public sources: the PVDAQ inventory (NREL, via OEDI), 3DEP point clouds (USGS
Entwine point tiles) and NAIP imagery (USDA). The outlines are hand-drawn on
the LiDAR itself, so they carry no detection model and no private data.

It is small. PVDAQ lists 406 documented (non-PVOutput) systems; 108 have
usable fixed angles; they sit at 45 coordinates, of which 30 are real sites
rather than district centroids. Eight sites were outlined and measured, and
three of those have a surface and a reference clean enough to call an accuracy
test. Read it as evidence about what the method does, not as a population
statistic.

## Result

Sites where one plane was outlined and the reference is credible:

| Site | Surface | Reported tilt / azimuth | Measured tilt / azimuth | Tilt error | Azimuth error |
| --- | --- | --- | --- | ---: | ---: |
| NREL Research Support Facility (two records, three roof wings) | PV-covered roofs | 10° / 165° and 10° / 180° | 10.08°, 10.03°, 10.04° / 165.0° (two wings); 10.01° / 180.0° (third) | +0.01° to +0.08° | within 0.04° |
| Simon Solar Farm, GA (33 MW) | ground rows, 5 outlined | 20° / 180° | 20.1° / 174.2° | +0.12° | −5.8° raw; **+0.2°** with the ground's slope along the rows removed |
| Farm Solar Array, CA (0.9 MW) | ground rows, 5 outlined | 25° / 180° | 26.0° / 177.6° | +0.95° | −2.4° raw; **+0.2°** with the ground's slope removed |

Row-to-row repeatability within a site is 0.1° to 0.2° in both angles.

Which RSF record describes which wing is not established independently (the
two records and the two measured orientations match as a set), so the tilt
agreement holds under either assignment and the azimuth agreement is stated
for the set.

### Rows on sloping ground face where the ground sends them

Both farms' tables follow terrain that falls to the east (2.2° in Georgia,
1.25° in California, from the ground returns). A table tilted 20° to the south
whose long axis also drops 2° to the east faces 174°, not 180°: the reported
azimuth is the design value on level ground, and the measured one is the
plane as built. Removing the ground's along-row slope from the measured plane
gives 180.2° at both sites.

### Sites that did not give an accuracy test, and why

| Site | Outcome | Reading |
| --- | --- | --- |
| RTC-FSEC baseline, FL | 5 rows at 29.6° / 180.05°, reported 35° / 180° | Azimuth agrees to 0.1°. The same 35° is recorded for the four "RTC baseline" systems at latitudes from 28° to 44°, so the reported tilt is doubtful. Not scored as tool error or as agreement. |
| NREL parking garage, CO | 4 canopy planes at 5.54° to 5.60° / 180.0°, reported 16.77° and 60° | The planes are flat to 2 cm and agree with each other to 0.06°. They are not at either reported angle. Unresolved: the monitored arrays may be other surfaces of the structure. |
| Henderson aquatic complex, NV | 2.8° against a reported 12° | The array could not be told from the roof in the LiDAR and the outline is approximate. Unresolved. |
| Clark County Desert Breeze, NV | 1.3° against a reported 10°; flagged `envelope_fit` | Rows of tilted modules on a flat roof: the known limitation, correctly flagged. |
| Distributed Sun EJ DeSeta, DE | no fit | Tilted rows on a flat roof at 1.9 returns/m². |

Not outlined:

- **Same flat-roof-rows class, or not reviewed in detail:** Clark County Hollywood
  (array cut by the clip) and Government Center, Andre Agassi Academy (3 records),
  Distributed Sun 5 and 6 Executive Campus, Mercury Solar (5 records), NREL S&TF.
- **LiDAR too sparse for the array:** Univ. of Maine (0.5 returns/m²), Distributed
  Sun The Wharf (1.8), RTC-SNL (3.6, rows not resolved).
- **Array not identifiable at the coordinate:** RTC-NV, NREL x-Si 6 and 7, and a New
  Orleans commercial roof whose array is gone by the 2021 flight.
- **No point cloud in the Entwine service, or none flown during or after operation:**
  the three NIST arrays and two Distributed Sun sites in Maryland, GSA Raleigh,
  RTC-VT. The Maryland sites are the most valuable ones missing (a canopy, a ground
  array and a roof with separate records); they would need whole-tile downloads.

## Files

| File | Contents |
| --- | --- |
| `sites.csv` | The 38 candidate mounts at 30 locations |
| `lidar_log.json` | The 3DEP collection chosen for each site |
| `outlines/<site>/polygons.json` | Drawn outlines: corners in metres east/north of the frame centre |
| `outlines/<site>/frame.json` | The frame: CRS, centre, extent, return density |
| `references.csv` | The reference table given to `pv-geom compare-reference` |
| `measurements.csv` | pv-geom output for the outlines |
| `validation_references.csv`, `validation_facets.csv`, `validation_summary.json` | The comparison |

## Rebuild

```
python scripts/accuracy_set.py sites    --out work
python scripts/accuracy_set.py lidar    --out work        # ~110 MB of clips
python scripts/accuracy_set.py render   --out work --only 9069 --half-width 20
# copy validation/pvdaq_documented/outlines/<site>/* into work/<site>/
python scripts/accuracy_set.py polygons --out work
python scripts/accuracy_set.py measure  --out work
```

`render` writes the three-panel image outlines are drawn on (NAIP, LiDAR
height, LiDAR intensity). Draw on the LiDAR panels: NAIP is displaced by
several metres on tall buildings.

## Caveats

- The outlines were drawn by looking at the LiDAR, which is also what is
  fitted. They decide *where* the plane is taken, not its angle, and the
  reference is independent of both.
- Polygon vintage is set to the LiDAR date, so rows are `panel_by_vintage`
  unless the standoff test confirms them. The roofs and ground tables here
  have no separate roof plane beneath the modules, so `no_panel_standoff` is
  expected.
- Summary statistics in `validation_summary.json` pool every scored facet,
  including the doubtful references above. Use the table in this file.
