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


def test_height_above_ground_not_nullable() -> None:
    assert not OUTPUT_SCHEMA.field("height_above_ground_m").nullable


def test_panel_tilt_is_nullable() -> None:
    assert OUTPUT_SCHEMA.field("panel_tilt_deg").nullable


def test_flags_is_list_of_string() -> None:
    flags_type = OUTPUT_SCHEMA.field("flags").type
    assert pa.types.is_list(flags_type)
    assert pa.types.is_string(flags_type.value_type)


def test_mounting_labels_complete() -> None:
    expected = {
        "flush_mount_rooftop",
        "tilted_rack_rooftop",
        "ground_mount_fixed",
        "ground_mount_tracker_suspected",
        "carport",
        "ambiguous",
    }
    assert MOUNTING_LABELS == expected
