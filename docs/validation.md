# External validation against reported geometry

`pv-geom compare-reference` lines a run's measurements up with geometry that
someone else reported for the same arrays: a monitoring database such as PVDAQ,
a permit set, an installer's as-built, a survey.

```
pv-geom compare-reference <run output> --reference refs.csv --out validation/
```

```python
result = pv_geom.compare_reference(output, "refs.csv")
result.references    # one row per reference mount: eligibility, outcome, area share agreeing
result.facets        # every measured facet against its reference, signed errors
result.summary       # headline numbers
```

## The reference table

One row per **reported mount**. Only the first four columns are required.

| Column | Meaning |
| --- | --- |
| `reference_id` | Unique id of this reference mount. |
| `polygon_ids` | The polygons that are this system, separated by `;`. |
| `tilt_deg`, `azimuth_deg` | Reported tilt, and azimuth clockwise from true north. |
| `scope` | `mount` if the record describes one identified subarray; `system` (default) if one record stands for a whole system. |
| `grade` | `surveyed`, `documented` or `self_reported` (default). |
| `tilt_resolution_deg`, `azimuth_resolution_deg` | The step in which the source reports (default 1). Use 45 for compass-point sources. |
| `suspect` | True if the values look like placeholders. Never scored. |
| `present_by` | Date by which the array is evidenced to exist in its current form. |
| `absent_at` | Latest date at which the array is evidenced *not* to exist. |
| `system_id`, `source`, `notes` | Provenance. |

## What the comparison does, and why

**Mounts against facets, not systems against polygons.** A monitoring record
often gives one tilt and azimuth for a system that has several subarrays. Every
facet (the `segments` column) of every matched polygon is compared, and the
result for a reference is the **share of measured area it describes**:
`consistent` (at least 90%), `partly_consistent`, or `inconsistent`. The
comparison never picks the facet that agrees best; that would manufacture
accuracy. Error statistics (bias, MAE) are computed only over references with
`scope = mount`, where the pairing is known.

**The reference's resolution is not charged to the measurement.** A facet agrees
when its tilt is within 3 degrees and its azimuth within 10 degrees of the
reference, each widened by half the reference's resolution. The raw signed
difference is kept alongside.

**Near-flat planes have no azimuth.** Azimuth is not compared when either tilt
is below 5 degrees.

**The array must have existed when the LiDAR was flown.** A reference is
`eligible` only with a `present_by` on or before the LiDAR date of its
polygons. Without one it is compared but reported as `presence_unknown` and
kept out of the headline numbers.

**Arrays built after the flight are a negative control.** With an `absent_at`
on or after the LiDAR date, the polygons measure the roof that was there before.
They are not scored for accuracy; the summary instead reports what share of
their area pv-geom labelled `panel_confirmed`, which should be none. This is a
direct test of the vintage screen.

## Matching polygons to a system

This is the part no tool can do for you, and where a validation is most easily
spoiled.

- Establish the match from location, imagery and module counts.
- **Never** choose polygons by whether their measured angles agree with the
  reference.
- Establish presence from dated imagery, not from the reported start date.
- If the polygons overlap one another, they are not independent examples;
  prefer a dissolved layer.

## PVDAQ as a reference source

[PVDAQ](https://data.openei.org/submissions/4568) lists about 1,900 systems.
`scripts/pvdaq_references.py --bbox ... --out refs.csv` writes the reference
rows for an area, leaving `polygon_ids` and the presence dates for you to fill.
Two populations behave very differently:

**PVOutput-derived systems (about 1,450).** Self-reported by owners.

- Azimuth is always one of eight compass points: the reference cannot say more
  than "roughly south". Treat agreement as a check for gross errors (a wrong
  facet, a flipped axis), not as a measure of accuracy.
- About one in five reports a tilt of 1 degree, a placeholder. The script flags
  these as `suspect`.
- One mount per system, whatever the roof looks like (`scope = system`).
- Coordinates are approximate and can fall on a neighbouring property, a road
  or open land. In a pilot of 15 Phoenix systems, imagery supported a property
  match for three.

**Documented systems (about 400).** NREL test arrays, university and commercial
rooftops, monitored residential programmes. Angles are reported to a degree or
better and large sites list each array separately (`scope = mount`). These are
the ones that can support an accuracy figure. They are spread across the
country, so using them means running pv-geom there: public 3DEP LiDAR plus
hand-drawn polygons for a few dozen sites. That is a modest job and independent
of any detection layer.

So: PVDAQ can supply a **consistency check and a vintage negative control**
from its self-reported systems in a study area, and a **small accuracy set**
from its documented systems elsewhere. It does not by itself reach the 150
arrays story A1 asks for.
