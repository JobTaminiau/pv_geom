"""Output schema sanity: shape, required mounting labels, expected non-null fields."""

import pyarrow as pa

from pv_geom.schema import MOUNTING_LABELS, OUTPUT_SCHEMA


def test_required_fields_present() -> None:
    names = set(OUTPUT_SCHEMA.names)
    required = {
        "polygon_id",
        "geometry",
        "panel_tilt_deg",
        "panel_azimuth_deg",
        "mounting_type",
        "mounting_confidence",
        "mounting_rule",
        "flags",
        "lidar_tile_ids",
        "pkg_version",
        "config_hash",
        "run_id",
        "partition_id",
    }
    assert required <= names


def test_polygon_id_not_nullable() -> None:
    assert not OUTPUT_SCHEMA.field("polygon_id").nullable


def test_parent_polygon_id_join_key() -> None:
    """Every output row carries the original input id so part-level rows can
    be aggregated back to input detections without parsing __p<i> suffixes."""
    field = OUTPUT_SCHEMA.field("parent_polygon_id")
    assert not field.nullable
    assert pa.types.is_string(field.type)


def test_height_above_ground_is_nullable() -> None:
    """Null means "no ground reference found" — pre-0.3 this was coerced to
    0.0, which made unknown heights indistinguishable from ground level."""
    assert OUTPUT_SCHEMA.field("height_above_ground_m").nullable


def test_quality_flags_include_missing_footprint() -> None:
    from pv_geom.schema import QUALITY_FLAGS

    assert "possible_missing_footprint" in QUALITY_FLAGS


def test_panel_tilt_is_nullable() -> None:
    assert OUTPUT_SCHEMA.field("panel_tilt_deg").nullable


def test_flags_is_list_of_string() -> None:
    flags_type = OUTPUT_SCHEMA.field("flags").type
    assert pa.types.is_list(flags_type)
    assert pa.types.is_string(flags_type.value_type)


def test_mounting_labels_complete() -> None:
    expected = {
        "flush_mount_pitched_roof",
        "flush_mount_flat_roof",
        "tilted_rack_rooftop",
        "east_west_rack_rooftop",
        "ground_mount_fixed",
        "ground_mount_tracker_suspected",
        "carport",
        "pole_mount",
        "ambiguous",
    }
    assert MOUNTING_LABELS == expected
