"""Output schema sanity: shape, required mounting labels, expected non-null fields."""

import pyarrow as pa

from pv_geom.schema import MOUNTING_LABELS, OUTPUT_SCHEMA, output_schema


def test_required_fields_present() -> None:
    names = set(OUTPUT_SCHEMA.names)
    required = {
        "polygon_id",
        "geometry",
        "tilt_deg",
        "azimuth_deg",
        "geometry_basis",
        "polygon_vintage",
        "lidar_date",
        "vintage_gap_days",
        "flags",
        "lidar_tile_ids",
        "pkg_version",
        "config_hash",
        "run_id",
        "partition_id",
    }
    assert required <= names


def test_mounting_columns_are_experimental_only() -> None:
    """Mounting classification is archived: its columns exist only when the
    classifier is switched on."""
    mounting = {"mounting_type", "mounting_confidence", "mounting_rule"}
    assert not mounting & set(OUTPUT_SCHEMA.names)
    assert mounting <= set(output_schema(include_mounting=True).names)


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


def test_quality_flags_include_no_panel_standoff() -> None:
    """Panel plane not resolvably above the roof — a genuinely flush mount, or
    an input polygon whose panels postdate the LiDAR (see README)."""
    from pv_geom.schema import QUALITY_FLAGS

    assert "no_standoff" in QUALITY_FLAGS


def test_panel_tilt_is_nullable() -> None:
    assert OUTPUT_SCHEMA.field("tilt_deg").nullable


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
    assert expected == MOUNTING_LABELS


def test_quality_flags_cover_the_standoff_tri_state() -> None:
    """Failed / not-applicable are distinct flags; "passed" is the absence of
    both. Without `standoff_unscreenable`, an unscreenable row is
    indistinguishable from a screened one."""
    from pv_geom.schema import QUALITY_FLAGS

    assert {"no_standoff", "standoff_unscreenable"} <= QUALITY_FLAGS


def test_roof_failure_flags_are_split() -> None:
    """A roof fit can fail on consensus (no single plane in the ring) or on
    quality (RMSE above the gate). Reporting both as `roof_complex` hid that
    nearly every rejection was well inside the RMSE gate."""
    from pv_geom.schema import QUALITY_FLAGS

    assert {"roof_no_consensus", "roof_complex", "roof_insufficient"} <= QUALITY_FLAGS


def test_recommended_rule() -> None:
    from pv_geom.schema import is_recommended

    assert is_recommended("measured", "panel_confirmed", [])
    assert is_recommended("measured", "panel_by_vintage", ["wide_tolerance_fit", "multi_facet"])
    assert not is_recommended("measured", "surface_unresolved", [])
    assert not is_recommended("measured", "unscreened", [])
    assert not is_recommended("no_fit", "no_fit", ["poor_fit"])
    assert not is_recommended("measured", "panel_confirmed", ["overlaps_polygon"])
    assert not is_recommended("measured", "panel_confirmed", ["below_min_area"])


def test_old_outputs_are_renamed_as_they_are_read(tmp_path) -> None:
    """Schema 0.6 renamed the measured columns. An output written under the old
    names must read back under the new ones, facets and flags included."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    import pv_geom
    from pv_geom.io.output import read_output_table, relabel_struct_list
    from pv_geom.sample import write_demo
    from pv_geom.schema import RENAMED_COLUMNS, RENAMED_FLAGS

    run = pv_geom.run(write_demo(tmp_path), use_dask=False)
    new = read_output_table(run.output)
    back = {v: k for k, v in RENAMED_COLUMNS.items()}
    old_facets = pa.list_(pa.struct([
        pa.field({"facet_index": "segment_index", "fit_rmse_m": "rmse_m"}.get(f.name, f.name),
                 f.type) for f in new.schema.field("facets").type.value_type]))
    flag_back = {v: k for k, v in RENAMED_FLAGS.items()}
    from pathlib import Path

    for part in sorted(Path(run.output).glob("part-*.parquet")):
        t = pq.read_table(part)
        t = t.set_column(t.column_names.index("facets"), "facets",
                         relabel_struct_list(t.column("facets"), old_facets))
        flags = [[flag_back.get(f, f) for f in fl] for fl in t.column("flags").to_pylist()]
        t = t.set_column(t.column_names.index("flags"), "flags",
                         pa.array(flags, pa.list_(pa.string())))
        t = t.rename_columns([back.get(n, n) for n in t.column_names])
        assert "panel_tilt_deg" in t.column_names and "segments" in t.column_names
        pq.write_table(t, part)

    old = read_output_table(run.output)
    assert old.column_names == new.column_names
    assert old.column("tilt_deg").to_pylist() == new.column("tilt_deg").to_pylist()
    assert old.column("facets").to_pylist() == new.column("facets").to_pylist()
    assert old.column("flags").to_pylist() == new.column("flags").to_pylist()
    assert not set(RENAMED_COLUMNS) & set(old.column_names)


def test_legacy_names_can_be_asked_for(tmp_path) -> None:
    import pv_geom
    from pv_geom.sample import write_demo
    from pv_geom.schema import RENAMED_COLUMNS

    run = pv_geom.run(write_demo(tmp_path), use_dask=False)
    plain = pv_geom.load(run.output)
    assert "tilt_deg" in plain.columns and "panel_tilt_deg" not in plain.columns
    legacy = pv_geom.load(run.output, legacy_names=True)
    for old, new in RENAMED_COLUMNS.items():
        assert legacy[old].equals(legacy[new]) or legacy[old].isna().all()


def test_no_measured_column_claims_to_be_a_panel() -> None:
    """The fitted surface is a panel only where geometry_basis says so."""
    from pv_geom.schema import output_schema

    names = output_schema(False).names
    assert not [n for n in names if n.startswith("panel_") or n.endswith("_panel")]
