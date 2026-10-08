"""Output schema (pyarrow source of truth) and the data dictionary built from it.

Every column carries a unit and a description in its field metadata, so the
dictionary that ships beside a published dataset is generated, not maintained
by hand.
"""

from __future__ import annotations

import pyarrow as pa

# Version of the output schema, stamped into every partition and the manifest.
# Compatibility rule: within a major version, columns are only ever ADDED and
# added columns are nullable, so a reader written for x.0 reads every x.y.
# Before 1.0 the major version is 0 and minor versions may still change types.
SCHEMA_VERSION = "0.5"


# One facet of a polygon (an entry of the `segments` column).
SEGMENT_FIELDS: list[tuple[str, pa.DataType, str, str]] = [
    ("segment_index", pa.int8(), "", "0 for the primary (largest) facet, then by size."),
    ("area_share", pa.float32(), "", "Share of the polygon's fitted returns on this facet."),
    ("area_m2", pa.float32(), "m2", "Plan area apportioned to the facet: polygon area x share."),
    ("surface_area_m2", pa.float32(), "m2", "Area along the facet's plane: area_m2 / cos(tilt)."),
    ("tilt_deg", pa.float32(), "deg", "Tilt of the facet from horizontal."),
    ("azimuth_deg", pa.float32(), "deg",
     "Direction the facet faces, clockwise from true north; null below the tilt floor."),
    ("rmse_m", pa.float32(), "m", "RMSE of the facet's returns about its plane."),
    ("tilt_unc_deg", pa.float32(), "deg", "Bootstrap 1-sigma uncertainty of the tilt."),
    ("azimuth_unc_deg", pa.float32(), "deg", "Bootstrap 1-sigma uncertainty of the azimuth."),
    ("n_points", pa.int32(), "", "LiDAR returns on the facet."),
    ("height_above_ground_m", pa.float32(), "m", "Median height of the facet above ground."),
    ("geometry", pa.binary(), "", "Where in the polygon the facet lies (WKB, run CRS)."),
]
SEGMENT_TYPE = pa.list_(pa.struct([pa.field(n, t) for n, t, _, _ in SEGMENT_FIELDS]))


# The `recommended` column. Provisional until the accuracy work (spec Epic A)
# says whether wide-tolerance fits and unresolved surfaces belong in or out.
RECOMMENDED_BASES = frozenset({"panel_confirmed", "panel_by_vintage"})
RECOMMENDED_EXCLUDING_FLAGS = frozenset({
    "low_density", "below_min_area", "overlaps_polygon", "duplicate_geometry",
    "envelope_fit",
})
RECOMMENDED_RULE = (
    "status == 'measured' and geometry_basis in "
    f"{sorted(RECOMMENDED_BASES)} and none of the flags {sorted(RECOMMENDED_EXCLUDING_FLAGS)}. "
    "Provisional: wide_tolerance_fit rows are included and surface_unresolved rows excluded "
    "pending validation."
)


def is_recommended(status: str, basis: str, flags: list[str] | tuple[str, ...]) -> bool:
    """Whether a row is in the suggested default subset (see RECOMMENDED_RULE)."""
    return (status == "measured" and basis in RECOMMENDED_BASES
            and not RECOMMENDED_EXCLUDING_FLAGS.intersection(flags))


def _f(name: str, typ: pa.DataType, *, nullable: bool = True, unit: str = "",
       desc: str = "") -> pa.Field:
    return pa.field(name, typ, nullable=nullable,
                    metadata={"unit": unit, "description": desc})


_CORE_FIELDS: list[pa.Field] = [
    # --- identity -----------------------------------------------------------
    _f("polygon_id", pa.string(), nullable=False,
       desc="Unique row id. Exploded MultiPolygon parts get a __p<i> suffix."),
    _f("parent_polygon_id", pa.string(), nullable=False,
       desc="Id of the input feature this row came from; equals polygon_id "
            "unless the row is an exploded MultiPolygon part."),
    _f("input_row", pa.int32(), nullable=False,
       desc="0-based row position of the parent feature in the input polygon "
            "file. A join key that works even when the input has no id column."),
    _f("status", pa.string(), nullable=False,
       desc="What happened to this polygon: measured, no_fit, no_lidar_tile, "
            "outside_tile_index, tile_unreadable or invalid_geometry. Every input "
            "polygon has exactly one row."),
    _f("geometry", pa.binary(),
       desc="Polygon footprint (WKB) in the run CRS. Null only when the input "
            "feature had no geometry."),
    _f("area_m2", pa.float32(), unit="m2",
       desc="Plan-view (horizontal) area of the polygon."),
    _f("surface_area_m2", pa.float32(), unit="m2",
       desc="Area along the fitted plane: area_m2 / cos(tilt). Null without a fit."),
    _f("aspect_ratio", pa.float32(),
       desc="Long/short side of the minimum rotated rectangle."),
    # --- vintage ------------------------------------------------------------
    _f("polygon_vintage", pa.date32(), unit="date",
       desc="Capture date of the imagery the polygon was derived from (latest "
            "day of the declared window)."),
    _f("lidar_date", pa.date32(), unit="date",
       desc="LiDAR capture date for this row's tile: declared, or measured "
            "from per-point GPS time."),
    _f("lidar_date_source", pa.string(),
       desc="Where lidar_date came from: declared, gps_time (measured flight "
            "date) or header_date (LAS header; a delivery date, can lag the flight)."),
    _f("vintage_gap_days", pa.int32(), unit="days",
       desc="polygon_vintage minus lidar_date. Positive means the polygon is "
            "newer than the LiDAR, so the installation may be absent from it."),
    _f("recommended", pa.bool_(), nullable=False,
       desc="True for rows suggested for analysis of array geometry: measured, on a panel "
            "basis (panel_confirmed or panel_by_vintage) and free of the flags that mark an "
            "unreliable or double-counted row. PROVISIONAL rule, to be fixed once accuracy "
            "is validated; see RECOMMENDED_RULE in the dataset metadata."),
    _f("geometry_basis", pa.string(), nullable=False,
       desc="What the fitted plane represents: panel_confirmed, "
            "panel_by_vintage, surface_unresolved, unscreened, no_fit, or "
            "not_measured when the polygon never reached the LiDAR."),
    # --- panel plane --------------------------------------------------------
    _f("n_points_panel", pa.int32(),
       desc="LiDAR returns inside the (eroded) polygon. Null when not measured."),
    _f("n_inliers_panel", pa.int32(),
       desc="Returns within the RANSAC threshold of the fitted plane."),
    _f("fit_failure", pa.string(),
       desc="Why there is no fit, when status is no_fit: too_few_points, "
            "ground_level_only or no_consensus."),
    _f("point_density", pa.float32(), unit="pts/m2",
       desc="n_points_panel / area_m2."),
    _f("panel_tilt_deg", pa.float32(), unit="deg",
       desc="Tilt of the fitted plane from horizontal (0 = flat)."),
    _f("panel_azimuth_deg", pa.float32(), unit="deg",
       desc="Direction the plane faces, clockwise from TRUE north (0 = N, 90 = E, "
            "180 = S). Null below the tilt floor, where it is undefined."),
    _f("grid_convergence_deg", pa.float32(), unit="deg",
       desc="Meridian convergence at the polygon: the true-north bearing of the "
            "run CRS's grid north. Already added to every azimuth column; "
            "subtract it to get the azimuth relative to grid north."),
    _f("panel_rmse_m", pa.float32(), unit="m",
       desc="Perpendicular RMSE of the inliers about the fitted plane."),
    _f("panel_fit_tolerance_m", pa.float32(), unit="m",
       desc="RANSAC inlier distance the accepted fit used. Larger than the "
            "configured base when the returns were too noisy for it."),
    _f("panel_rival_share", pa.float32(),
       desc="Support for the strongest different plane found in the same returns, as a "
            "share of the fitted plane's inliers. Near 1 means two planes fit about "
            "equally well and the reported one is a fragile choice. Null if none was found."),
    _f("panel_rival_angle_deg", pa.float32(), unit="deg",
       desc="Angle between the fitted plane and that rival plane."),
    _f("panel_tilt_unc_deg", pa.float32(), unit="deg",
       desc="Bootstrap 1-sigma uncertainty of the tilt."),
    _f("panel_azimuth_unc_deg", pa.float32(), unit="deg",
       desc="Bootstrap circular 1-sigma uncertainty of the azimuth."),
    _f("n_planes_detected", pa.int8(),
       desc="Number of distinct facets (planes) in the polygon; 0 without a fit. The "
            "panel_* columns describe the primary (largest) facet; `segments` holds all."),
    _f("secondary_tilt_deg", pa.float32(), unit="deg",
       desc="Tilt of the second plane, when one was found."),
    _f("secondary_azimuth_deg", pa.float32(), unit="deg",
       desc="Azimuth of the second plane (true north), when one was found."),
    _f("segments", SEGMENT_TYPE,
       desc="Every facet of the polygon, primary first: one entry for an ordinary "
            "array, several when the polygon covers more than one roof face. Each "
            "has its own tilt, true-north azimuth, area share and uncertainty. Null "
            "without a fit. Exported flat as pv_geom_segments in the release dataset."),
    # --- roof reference -----------------------------------------------------
    _f("roof_ref_source", pa.string(), nullable=False,
       desc="How the roof reference ring was built: footprint_ring (clipped to "
            "a building footprint), open_ring (no footprint) or none."),
    _f("roof_ref_method", pa.string(),
       desc="How the roof plane was chosen within the ring: dominant_plane (the "
            "ring's main plane), collar_facet (the facet the band beside the array "
            "lies on), panel_parallel_facet (the facet parallel to the array plane, "
            "where that band is split between facets) or collar_only. Null without "
            "a usable roof reference."),
    _f("roof_tilt_deg", pa.float32(), unit="deg",
       desc="Tilt of the plane fitted to the ring around the polygon."),
    _f("roof_azimuth_deg", pa.float32(), unit="deg",
       desc="Azimuth of the plane fitted to the ring around the polygon (true north)."),
    _f("roof_rmse_m", pa.float32(), unit="m",
       desc="RMSE of the roof-ring fit."),
    _f("panel_roof_angle_deg", pa.float32(), unit="deg",
       desc="Angle between the panel and roof planes. Null without a usable roof fit."),
    _f("height_above_roof_m", pa.float32(), unit="m",
       desc="Vertical offset of the panel plane above the roof plane. Null "
            "without a usable roof fit."),
    _f("height_above_ground_m", pa.float32(), unit="m",
       desc="Median panel height above nearby ground returns. Null when no "
            "ground reference exists nearby."),
    _f("on_building", pa.bool_(),
       desc="Polygon overlaps a building footprint. Null when the run had no "
            "footprint layer."),
    _f("building_id", pa.string(),
       desc="Id of the overlapped footprint, when there is one."),
    # --- quality + provenance ----------------------------------------------
    _f("flags", pa.list_(pa.string()), nullable=False,
       desc="Quality flags; see QUALITY_FLAGS."),
    _f("lidar_tile_ids", pa.list_(pa.string()), nullable=False,
       desc="LiDAR tiles read for this row's tile group."),
    _f("pkg_version", pa.string(), nullable=False, desc="pv-geom version."),
    _f("config_hash", pa.string(), nullable=False,
       desc="sha256 of the resolved run configuration."),
    _f("run_id", pa.string(), nullable=False, desc="Run identifier."),
    _f("partition_id", pa.int32(), nullable=False, desc="Output partition."),
]

# Experimental, archived in 0.2.0: only present when mounting_rules.enabled.
MOUNTING_FIELDS: list[pa.Field] = [
    _f("mounting_type", pa.string(),
       desc="EXPERIMENTAL rule-based mounting label; see MOUNTING_LABELS."),
    _f("mounting_confidence", pa.float32(),
       desc="EXPERIMENTAL confidence of mounting_type, 0-1."),
    _f("mounting_rule", pa.string(),
       desc="EXPERIMENTAL id of the rule that fired."),
]

OUTPUT_SCHEMA: pa.Schema = pa.schema(_CORE_FIELDS)
OUTPUT_SCHEMA_WITH_MOUNTING: pa.Schema = pa.schema(_CORE_FIELDS + MOUNTING_FIELDS)


def output_schema(include_mounting: bool = False) -> pa.Schema:
    """The run's output schema; mounting columns only when the experimental
    classifier is enabled."""
    return OUTPUT_SCHEMA_WITH_MOUNTING if include_mounting else OUTPUT_SCHEMA


MOUNTING_LABELS: frozenset[str] = frozenset(
    {
        "flush_mount_pitched_roof",     # R1; was flush_mount_rooftop pre-0.2
        "flush_mount_flat_roof",        # R1; was flush_mount_rooftop pre-0.2
        "tilted_rack_rooftop",
        "east_west_rack_rooftop",       # R7
        "ground_mount_fixed",
        "ground_mount_tracker_suspected",
        "carport",
        "pole_mount",                   # R8
        "ambiguous",
    }
)

FLAG_DESCRIPTIONS: dict[str, str] = {
    "low_density": "Too few returns in the polygon for a robust fit.",
    "poor_fit": "No plane reached consensus; tilt and azimuth are null.",
    "near_horizontal": "Tilt below the floor; azimuth is undefined and null.",
    "multi_facet": "The polygon holds more than one distinct plane; see `segments`.",
    "below_min_area": "Input polygon is smaller than polygons.min_area_m2.",
    "overlaps_polygon": "Input polygon substantially overlaps another input polygon.",
    "duplicate_geometry": "Input polygon has the same geometry as an earlier one.",
    "geometry_repaired": "Input geometry was invalid and was repaired before measuring.",
    "wide_tolerance_fit": "Fit accepted only at a wider inlier tolerance than the base "
                          "(noisy returns); see panel_fit_tolerance_m.",
    "ambiguous_fit": "A different plane (2 degrees or more away) fits nearly as many "
                     "returns; see panel_rival_share and panel_rival_angle_deg.",
    "envelope_fit": "Wide-tolerance fit on a near-flat plane: the signature of rows of "
                    "tilted modules on a flat roof (or rooftop clutter). Tilt and azimuth "
                    "describe the envelope of the rows, not the modules.",
    "east_west_rack": "Two planes facing ~180 deg apart at similar tilt.",
    "roof_insufficient": "The roof ring never gathered enough returns.",
    "roof_no_consensus": "No single plane described the roof ring.",
    "roof_complex": "A roof plane was found but was too rough to measure against.",
    "no_panel_standoff": "Panel plane is not resolvably above the roof plane.",
    "standoff_unscreenable": "No usable roof reference, so no standoff test was possible.",
    # Only emitted when the experimental mounting classifier is enabled.
    "tracker_suspected": "EXPERIMENTAL single-axis tracker heuristic fired.",
    "possible_missing_footprint": "EXPERIMENTAL elevated, off-footprint, no canopy evidence.",
}

QUALITY_FLAGS: frozenset[str] = frozenset(FLAG_DESCRIPTIONS)

# --- status ------------------------------------------------------------------
MEASURED = "measured"
NO_FIT = "no_fit"
NO_LIDAR_TILE = "no_lidar_tile"
OUTSIDE_TILE_INDEX = "outside_tile_index"
TILE_UNREADABLE = "tile_unreadable"
INVALID_GEOMETRY = "invalid_geometry"

STATUS_DESCRIPTIONS: dict[str, str] = {
    MEASURED: "A plane was fitted to the LiDAR returns inside the polygon.",
    NO_FIT: "LiDAR covered the polygon but no plane could be fitted; see fit_failure.",
    NO_LIDAR_TILE: "The tile that holds the polygon is in the index but not in storage.",
    OUTSIDE_TILE_INDEX: "The polygon lies outside every tile of the LiDAR index.",
    TILE_UNREADABLE: "The polygon's tile group failed to process; a resumed run retries it.",
    INVALID_GEOMETRY: "The input feature has no usable polygon geometry.",
}
STATUSES: tuple[str, ...] = tuple(STATUS_DESCRIPTIONS)

FIT_FAILURE_DESCRIPTIONS: dict[str, str] = {
    "too_few_points": "Fewer returns inside the polygon than panel_plane.min_points.",
    "ground_level_only": "Only ground-level returns inside the polygon: nothing stands "
                         "above the ground there.",
    "no_consensus": "Enough returns, but no single plane holds enough of them.",
}


def data_dictionary(schema: pa.Schema | None = None) -> list[dict[str, str]]:
    """One record per output column: name, type, unit, nullable, description."""
    schema = schema or OUTPUT_SCHEMA
    out = []
    for f in schema:
        md = {k.decode(): v.decode() for k, v in (f.metadata or {}).items()}
        out.append(
            {
                "column": f.name,
                "type": str(f.type),
                "unit": md.get("unit", ""),
                "nullable": "yes" if f.nullable else "no",
                "description": md.get("description", ""),
            }
        )
    return out
