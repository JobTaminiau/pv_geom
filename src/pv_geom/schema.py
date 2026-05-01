"""Output schema (pyarrow source of truth). Mirrors PRD §8."""

from __future__ import annotations

import pyarrow as pa

OUTPUT_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("polygon_id", pa.string(), nullable=False),
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
        pa.field("height_above_ground_m", pa.float32(), nullable=False),
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
        "flush_mount_rooftop",
        "tilted_rack_rooftop",
        "ground_mount_fixed",
        "ground_mount_tracker_suspected",
        "carport",
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
        "roof_complex",
    }
)
