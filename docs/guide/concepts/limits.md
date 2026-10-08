# Limits

What pv-geom does not do, or does only partly.

**Rows of tilted modules on a flat roof are not resolved.** A polygon drawn
around many short rows is fitted by the envelope of the rows, which is nearly
flat whatever the module tilt. On a warehouse reported at 10 degrees the fit
gave 1.4. These fits are flagged `envelope_fit` and kept out of
`recommended`. Where the rows are outlined one by one and the LiDAR is dense
enough, each row measures correctly.

**Ground rows face where the ground sends them.** A table tilted 20 degrees
south whose long axis follows ground falling 2 degrees to the east faces 174
degrees, not 180. pv-geom reports the plane as built. Design documents report
the design.

**Presence is evidenced, not proven.** See
[vintage and basis](vintage-and-basis.md). Removal of an array is not detected.

**Small polygons often cannot be fitted.** Below about 3 square metres at ten
returns per square metre there are too few returns for a robust fit.

**Trackers are measured where they stood.** The tilt of a tracking array is
whatever it was at the moment of the flight.

**The accuracy evidence is thin.** A handful of sites with independent
reported values. A proper accuracy study needs surveyed or documented
references for many more arrays.

**Uncertainty is precision.** It says how well the plane is determined by its
returns, not whether the returns were the array.

**Results are reproducible to a tolerance across platforms**, and bit for bit
only on one machine. See [reproducing a run](../how-to/reproduce.md).
