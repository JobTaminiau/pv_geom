# Validation against outside data

`pv-geom compare-reference` lines a run up with geometry that someone else
reported for the same arrays: a monitoring database, a permit set, an
as-built, a survey.

```bash
pv-geom compare-reference out/my_area --reference refs.csv --out validation/
```

The reference table has one row per **reported mount**. Four columns are
required: `reference_id`, `polygon_ids` (separated by `;`), `tilt_deg` and
`azimuth_deg`. Optional columns say how far the reference can be trusted and
whether the array existed when the LiDAR was flown.

What the comparison does:

- **Mounts against facets.** A record often gives one angle pair for a system
  with several subarrays. Every measured facet of the matched polygons is
  compared, and the result is the share of measured area the reference
  describes. It never picks the facet that agrees best.
- **The reference's resolution is not held against the measurement.** A source
  that reports azimuth as a compass point cannot say more than "roughly south".
- **Presence matters.** A reference is scored only when the array is evidenced
  to have existed by the LiDAR date. One evidenced to be absent becomes a
  negative control: none of its area should be labelled `panel_confirmed`.

Two scripts build reference tables from public sources:

```bash
python scripts/pvdaq_references.py  --bbox -113 32.5 -111 34.5 --out refs.csv
python scripts/uspvdb_references.py --bbox -113 32.5 -111 34.5 \
    --polygons detections.parquet --id-col cluster_id --out refs.csv
```

The full protocol, what each source can and cannot support, and what the first
comparisons showed are in `docs/validation.md` in the repository. A small
accuracy set built entirely from public data is in
`validation/pvdaq_documented/`.

!!! warning "Matching is the part that spoils a validation"
    Decide which polygons belong to a system from location, imagery and module
    counts. Never choose them by whether their measured angles agree with the
    reference.
