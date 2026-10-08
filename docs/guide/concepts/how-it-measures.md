# How it measures

For each polygon, in this order.

## 1. The returns

LiDAR returns inside the polygon, shrunk by 15 cm to keep clear of its edge,
are the candidates. Where the LiDAR has a building class it is used; most
public collections do not, and unclassified returns more than a metre above
local ground are used instead.

## 2. The plane

A plane is fitted by consensus:

1. Planes through triples of returns are scored by how many returns lie within
   a tolerance (5 cm). For a polygon with up to about 120 returns **every**
   triple is tried; larger ones get a large random sample.
2. The best distinct candidates are each refitted to their inliers until the
   inlier set stops changing.
3. The winner is the one with the most inliers, then the smaller residual.
4. It is settled with a smooth robust fit, which has one optimum where the
   hard cut-off has several.

The result does not depend on a random seed. If a different plane, two degrees
or more away, fits nearly as many returns, the row is flagged `ambiguous_fit`
and `fit_rival_share` says how close it was.

If no plane holds 60% of the returns at 5 cm, the scatter of the returns is
measured and the fit repeated at twice that, up to 15 cm. Such rows are flagged
`wide_tolerance_fit`. This is what makes noisier collections usable.

## 3. More than one plane

A polygon can cover two roof faces. Planes that are each well supported and
differ in orientation are separate **facets**. The row describes the largest;
the `facets` column, and `pv_geom_facets` in the release dataset, hold all of
them with their own tilt, azimuth and share of the area.

## 4. Tilt and azimuth

Tilt is the angle of the plane from horizontal. Azimuth is the direction the
plane faces, clockwise from **true north**: the fit is made in the map
projection's grid, and the meridian convergence at the polygon is added
(`grid_convergence_deg` records it). Below a tilt of one degree a plane faces
nowhere in particular, and azimuth is left empty.

## 5. Uncertainty

`tilt_unc_deg` and `azimuth_unc_deg` come from refitting to resampled inliers.
They are the precision of the fit given its inliers. They do not include the
possibility that the wrong surface was fitted; `geometry_basis` and the flags
speak to that.

## 6. The roof beneath

A plane is fitted to returns in a ring around the polygon, with other array
polygons removed. Roofs with hips and ridges have several faces, so the face
is chosen by the band nearest the array, or failing that the face parallel to
the array's own plane. `roof_ref_method` records which. The array's height
above that plane is the standoff used as evidence that modules were present.

## 7. Heights

`height_above_ground_m` is the median height of the fitted returns above local
ground. `height_above_roof_m` is the standoff.

## How well it works

On the few sites where independent reported values and a clean surface were
both available, tilt agreed to within 0.1 degrees on module-covered roofs and
0.1 to 1.0 degrees on ground-mounted rows, and azimuth to within 0.2 degrees.
That is a handful of sites, not a population statistic. See
`validation/pvdaq_documented/` in the repository.
