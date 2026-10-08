# Questions

**Which rows should I use?**
Start from `recommended == True`. Then decide whether flush-mounted
`surface_unresolved` rows belong in your analysis; for pitched residential
roofs they usually do.

**Why do so many rows say `surface_unresolved`?**
Because the polygons are newer than the LiDAR and the fitted plane coincides
with the roof. For a flush-mounted array that is expected, whether or not it
had been built yet. The row is not wrong; it is unconfirmed.

**The report says the polygons postdate the LiDAR by years. Is the run useless?**
No. It means arrays built in that window measure the roof they were later put
on. `geometry_basis` separates rows with evidence the array was there from
rows without.

**Why is azimuth empty on some rows?**
The plane is within a degree of horizontal, so it faces nowhere in particular.

**Is azimuth relative to true north or grid north?**
True north. `grid_convergence_deg` holds the correction that was added, if you
need the grid value back.

**My LiDAR is in feet. Do I need to reproject it?**
No. It is converted as it is read, and the manifest records the conversion.

**My LiDAR has no building class.**
Most public collections do not. Unclassified returns above local ground are
used instead, automatically. It does need a ground class.

**Do I need a tile index?**
No. It is read from the tiles' headers if you do not supply one.

**Do I need building footprints?**
No. They improve the roof reference on dense blocks. Canopies are recognised
without them.

**Two detections cover the same array. What happens?**
Both are measured and both are flagged `overlaps_polygon` or
`duplicate_geometry`, and left out of `recommended`. Dissolving overlapping
polygons before the run is better.

**A polygon covers two roof faces.**
It gets one row describing the larger face and an entry per face in `facets`.
It is flagged `multi_facet`.

**Will I get the same numbers if I run it again?**
On the same machine, exactly. On another platform, to within 0.0001 degrees.
`pv-geom reproduce` checks.

**How long does it take?**
About a quarter of a second per polygon plus the time to read the tiles.
`--dry-run` gives an estimate for your inputs.

**An old script uses `panel_tilt_deg`.**
The measured columns were renamed in schema 0.6. Old outputs are renamed as
they are read, and `pv_geom.load(output, legacy_names=True)` gives the old
names back for one release.
