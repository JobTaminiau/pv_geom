"""Output schema (pyarrow source of truth). Mirrors PRD §8."""

from __future__ import annotations

import pyarrow as pa

OUTPUT_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("polygon_id", pa.string(), nullable=False),
        # Join key back to the input inventory: the original input id, equal to
        # polygon_id unless the row came from an exploded MultiPolygon part
        # (whose polygon_id gets a __p<i> suffix). Group on this to aggregate
        # part-level rows back to input detections.
        pa.field("parent_polygon_id", pa.string(), nullable=False),
        pa.field("geometry", pa.binary(), nullable=False),  # WKB; geoparquet writer wraps
        pa.field("n_points_panel", pa.int32(), nullable=False),
        pa.field("n_inliers_panel", pa.int32(), nullable=False),
        pa.field("panel_tilt_deg", pa.float32()),
        pa.field("panel_azimuth_deg", pa.float32()),
        pa.field("panel_rmse_m", pa.float32()),
        pa.field("panel_tilt_unc_deg", pa.float32()),
        pa.field("panel_azimuth_unc_deg", pa.float32()),
        pa.field("n_planes_detected", pa.int8(), nullable=False),
        pa.field("secondary_tilt_deg", pa.float32()),
        pa.field("secondary_azimuth_deg", pa.float32()),
        pa.field("roof_tilt_deg", pa.float32()),
        pa.field("roof_azimuth_deg", pa.float32()),
        pa.field("roof_rmse_m", pa.float32()),
        pa.field("panel_roof_angle_deg", pa.float32()),
        pa.field("height_above_roof_m", pa.float32()),
        pa.field("height_above_ground_m", pa.float32()),  # null = no ground reference found
        pa.field("on_building", pa.bool_(), nullable=False),
        pa.field("building_id", pa.string()),
        pa.field("area_m2", pa.float32(), nullable=False),
        pa.field("aspect_ratio", pa.float32(), nullable=False),
        pa.field("mounting_type", pa.string(), nullable=False),
        pa.field("mounting_confidence", pa.float32(), nullable=False),
        pa.field("mounting_rule", pa.string(), nullable=False),
        pa.field("flags", pa.list_(pa.string()), nullable=False),
        pa.field("lidar_tile_ids", pa.list_(pa.string()), nullable=False),
        pa.field("pkg_version", pa.string(), nullable=False),
        pa.field("config_hash", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("partition_id", pa.int32(), nullable=False),
    ]
)

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

QUALITY_FLAGS: frozenset[str] = frozenset(
    {
        "low_density",
        "poor_fit",
        "near_horizontal",
        "east_west_rack",
        "tracker_suspected",
        "roof_insufficient",
        "roof_no_consensus",
        "roof_complex",
        "possible_missing_footprint",
        # Panel-standoff (vintage) screen: failed / not applicable. Neither
        # flag present means the row was screened and passed. See tile_task.
        "no_panel_standoff",
        "standoff_unscreenable",
    }
)
