"""Output schema (pyarrow source of truth) and the data dictionary built from it.

Every column carries a unit and a description in its field metadata, so the
dictionary that ships beside a published dataset is generated, not maintained
by hand.
"""

from __future__ import annotations

import pyarrow as pa


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
    _f("geometry", pa.binary(), nullable=False,
       desc="Polygon footprint (WKB) in the run CRS."),
    _f("area_m2", pa.float32(), nullable=False, unit="m2",
       desc="Plan-view (horizontal) area of the polygon."),
    _f("surface_area_m2", pa.float32(), unit="m2",
       desc="Area along the fitted plane: area_m2 / cos(tilt). Null without a fit."),
    _f("aspect_ratio", pa.float32(), nullable=False,
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
    _f("geometry_basis", pa.string(), nullable=False,
       desc="What the fitted plane represents: panel_confirmed, "
            "panel_by_vintage, surface_unresolved, unscreened or no_fit."),
    # --- panel plane --------------------------------------------------------
    _f("n_points_panel", pa.int32(), nullable=False,
       desc="LiDAR returns inside the (eroded) polygon."),
    _f("n_inliers_panel", pa.int32(), nullable=False,
       desc="Returns within the RANSAC threshold of the fitted plane."),
    _f("point_density", pa.float32(), nullable=False, unit="pts/m2",
       desc="n_points_panel / area_m2."),
    _f("panel_tilt_deg", pa.float32(), unit="deg",
       desc="Tilt of the fitted plane from horizontal (0 = flat)."),
    _f("panel_azimuth_deg", pa.float32(), unit="deg",
       desc="Compass direction the plane faces (0 = N, 90 = E, 180 = S). Null "
            "below the tilt floor, where it is undefined."),
    _f("panel_rmse_m", pa.float32(), unit="m",
       desc="Perpendicular RMSE of the inliers about the fitted plane."),
    _f("panel_tilt_unc_deg", pa.float32(), unit="deg",
       desc="Bootstrap 1-sigma uncertainty of the tilt."),
    _f("panel_azimuth_unc_deg", pa.float32(), unit="deg",
       desc="Bootstrap circular 1-sigma uncertainty of the azimuth."),
    _f("n_planes_detected", pa.int8(), nullable=False,
       desc="0 = no fit, 1 = one plane, 2 = a second plane found in the outliers."),
    _f("secondary_tilt_deg", pa.float32(), unit="deg",
       desc="Tilt of the second plane, when one was found."),
    _f("secondary_azimuth_deg", pa.float32(), unit="deg",
       desc="Azimuth of the second plane, when one was found."),
    # --- roof reference -----------------------------------------------------
    _f("roof_ref_source", pa.string(), nullable=False,
       desc="How the roof reference ring was built: footprint_ring (clipped to "
            "a building footprint), open_ring (no footprint) or none."),
    _f("roof_tilt_deg", pa.float32(), unit="deg",
       desc="Tilt of the plane fitted to the ring around the polygon."),
    _f("roof_azimuth_deg", pa.float32(), unit="deg",
       desc="Azimuth of the plane fitted to the ring around the polygon."),
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
    _f("mounting_type", pa.string(), nullable=False,
       desc="EXPERIMENTAL rule-based mounting label; see MOUNTING_LABELS."),
    _f("mounting_confidence", pa.float32(), nullable=False,
       desc="EXPERIMENTAL confidence of mounting_type, 0-1."),
    _f("mounting_rule", pa.string(), nullable=False,
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
