"""The roof-reference gate on the standoff test, the per-polygon installation
dates, and the frozen 1.0 schema."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import geopandas as gpd
import pytest

import pv_geom
from pv_geom.config import HeightsConfig
from pv_geom.pipeline.measure import reference_not_parallel, screen_standoff
from pv_geom.schema import (
    FLAG_DESCRIPTIONS,
    OUTPUT_SCHEMA,
    RECOMMENDED_BASES,
    SCHEMA_VERSION,
    is_recommended,
)
from pv_geom.vintage import (
    BASIS_DESCRIPTIONS,
    GEOMETRY_BASIS,
    PANEL_BY_INSTALL_DATE,
    PANEL_BY_VINTAGE,
    PANEL_CONFIRMED,
    SURFACE_BEFORE_INSTALL,
    SURFACE_UNRESOLVED,
    UNSCREENED,
    geometry_basis,
    installed_at_lidar,
)

LIDAR = date(2021, 5, 15)


# --- the roof-reference gate ---------------------------------------------------

@pytest.mark.parametrize(("angle", "roof_tilt", "expected"), [
    (1.0, 25.0, False),        # the array's own facet
    (6.0, 25.0, False),        # at the limit
    (12.0, 25.0, True),        # another facet of a pitched roof
    (20.0, 2.0, False),        # a rack on a flat roof: the reference is the roof under it
    (float("nan"), 25.0, False),
    (12.0, float("nan"), False),
])
def test_a_pitched_reference_at_an_angle_is_not_the_arrays_roof(
        angle: float, roof_tilt: float, expected: bool) -> None:
    assert reference_not_parallel(angle, roof_tilt, HeightsConfig()) is expected


def test_the_standoff_is_not_tested_against_another_facet() -> None:
    # Well "above" the wrong plane: without the gate this would be confirmed.
    wrong = screen_standoff(0.40, True, 0.05, not_parallel=True)
    assert (wrong.screened, wrong.passed, wrong.not_parallel) == (False, False, True)
    assert wrong.flag == "standoff_unscreenable"
    right = screen_standoff(0.40, True, 0.05, not_parallel=False)
    assert (right.screened, right.passed, right.not_parallel) == (True, True, False)
    # No usable roof at all is a different reason, and is not reported as this one.
    assert screen_standoff(0.40, False, 0.05, not_parallel=True).not_parallel is False


# --- the installation record -----------------------------------------------------

@pytest.mark.parametrize(("by", "not_before", "expected"), [
    (date(2019, 6, 1), None, True),                  # signed off before the flight
    (LIDAR, None, True),                              # on the day counts
    (date(2022, 1, 1), None, None),                   # completed later: says nothing by itself
    (None, date(2022, 3, 1), False),                  # first permit after the flight
    (None, date(2020, 1, 1), None),                   # permitted before, completion unknown
    (date(2022, 6, 1), date(2022, 1, 1), False),
    (date(2020, 6, 1), date(2019, 1, 1), True),
    (None, None, None),
])
def test_what_an_installation_record_says_about_the_lidar_day(by, not_before, expected) -> None:
    assert installed_at_lidar(by, not_before, LIDAR) is expected
    assert installed_at_lidar(by, not_before, None) is None       # no LiDAR date, no answer


def test_basis_order_lidar_first_then_the_record_then_the_imagery() -> None:
    def basis(**kw):
        base = dict(fit_ok=True, standoff_passed=False, standoff_screened=True, gap_days=900)
        return geometry_basis(**{**base, **kw})

    assert basis(standoff_passed=True, installed_at_lidar=False) == PANEL_CONFIRMED
    assert basis(installed_at_lidar=True) == PANEL_BY_INSTALL_DATE
    assert basis(installed_at_lidar=False) == SURFACE_BEFORE_INSTALL
    assert basis(installed_at_lidar=False, standoff_screened=False) == SURFACE_BEFORE_INSTALL
    assert basis(installed_at_lidar=None) == SURFACE_UNRESOLVED
    assert basis(installed_at_lidar=None, standoff_screened=False) == UNSCREENED
    # A per-polygon record outranks the layer-wide imagery date.
    assert basis(installed_at_lidar=False, gap_days=-10) == SURFACE_BEFORE_INSTALL
    assert basis(installed_at_lidar=None, gap_days=-10) == PANEL_BY_VINTAGE


def test_new_values_are_documented_and_recommended_where_they_should_be() -> None:
    for value in (PANEL_BY_INSTALL_DATE, SURFACE_BEFORE_INSTALL):
        assert value in GEOMETRY_BASIS and value in BASIS_DESCRIPTIONS
    assert PANEL_BY_INSTALL_DATE in RECOMMENDED_BASES
    assert SURFACE_BEFORE_INSTALL not in RECOMMENDED_BASES
    assert is_recommended("measured", PANEL_BY_INSTALL_DATE, [])
    assert not is_recommended("measured", SURFACE_BEFORE_INSTALL, [])
    assert {"roof_reference_not_parallel", "standoff_before_install_date"} <= set(
        FLAG_DESCRIPTIONS)


def _demo_with_install_dates(tmp_path: Path) -> Path:
    """The demo, with half its polygons recorded as complete long before the
    LiDAR and the other half as not existing until long after."""
    from pv_geom.sample import write_demo

    config = write_demo(tmp_path / "demo")
    layer = tmp_path / "demo" / "inputs" / "polygons.parquet"
    polygons = gpd.read_parquet(layer)
    early = [i % 2 == 0 for i in range(len(polygons))]
    polygons["signed_off"] = [date(2001, 1, 1) if e else None for e in early]
    polygons["first_permit"] = [None if e else "2099" for e in early]
    polygons.to_parquet(layer)
    config.write_text(config.read_text(encoding="utf-8").replace(
        'polygon_vintage: "2023"',
        'polygon_vintage: "2100"\n  installed_by_column: signed_off\n'
        '  not_installed_before_column: first_permit'), encoding="utf-8")
    return config


def test_install_dates_flow_from_the_layer_to_the_rows(tmp_path: Path) -> None:
    run = pv_geom.run(_demo_with_install_dates(tmp_path), use_dask=False)
    rows = run.load().sort_values("input_row")
    assert {"installed_by", "not_installed_before"} <= set(rows.columns)
    early = rows[rows["input_row"] % 2 == 0]
    late = rows[rows["input_row"] % 2 == 1]
    assert (early["installed_by"].astype(str) == "2001-01-01").all()
    assert early["not_installed_before"].isna().all()
    # A bare year for "not before" is its first day, not its last.
    assert (late["not_installed_before"].astype(str) == "2099-01-01").all()

    measured_early = early[early["status"] == "measured"]
    measured_late = late[late["status"] == "measured"]
    assert len(measured_early) >= 3 and len(measured_late) >= 3
    # The record decides only where the LiDAR could not.
    assert set(measured_early["geometry_basis"]) <= {PANEL_CONFIRMED, PANEL_BY_INSTALL_DATE}
    assert set(measured_late["geometry_basis"]) <= {PANEL_CONFIRMED, SURFACE_BEFORE_INSTALL}
    # The demo's arrays are really there, so those the LiDAR confirms on a lot
    # the record says was empty are flagged as a disagreement.
    confirmed_late = measured_late[measured_late["geometry_basis"] == PANEL_CONFIRMED]
    assert all("standoff_before_install_date" in set(f) for f in confirmed_late["flags"])
    assert not any("standoff_before_install_date" in set(f) for f in measured_early["flags"])


def test_a_missing_install_date_column_is_an_input_error(tmp_path: Path) -> None:
    from pv_geom.errors import InputError
    from pv_geom.sample import write_demo

    config = write_demo(tmp_path / "demo")
    config.write_text(config.read_text(encoding="utf-8").replace(
        'polygon_vintage: "2023"', 'polygon_vintage: "2023"\n  installed_by_column: nope'),
        encoding="utf-8")
    with pytest.raises(InputError, match="nope"):
        pv_geom.run(config, use_dask=False)


# --- the frozen schema --------------------------------------------------------------

# Schema 1.0. Within major version 1 a column is never removed, renamed or
# retyped, so this list may only ever GROW, and only at the end of a group with
# a nullable column. If this test fails because a line here had to change, that
# is a breaking change and needs schema 2.0.
SCHEMA_1_0 = [
    ("polygon_id", "string"), ("parent_polygon_id", "string"), ("input_row", "int32"),
    ("status", "string"), ("geometry", "binary"), ("area_m2", "float"),
    ("surface_area_m2", "float"), ("aspect_ratio", "float"),
    ("polygon_vintage", "date32[day]"), ("installed_by", "date32[day]"),
    ("not_installed_before", "date32[day]"), ("lidar_date", "date32[day]"),
    ("lidar_date_source", "string"), ("vintage_gap_days", "int32"),
    ("recommended", "bool"), ("geometry_basis", "string"), ("n_points", "int32"),
    ("n_inliers", "int32"), ("fit_failure", "string"), ("point_density", "float"),
    ("tilt_deg", "float"), ("azimuth_deg", "float"), ("grid_convergence_deg", "float"),
    ("fit_rmse_m", "float"), ("fit_tolerance_m", "float"), ("fit_rival_share", "float"),
    ("fit_rival_angle_deg", "float"), ("tilt_unc_deg", "float"),
    ("azimuth_unc_deg", "float"), ("n_facets", "int8"), ("secondary_tilt_deg", "float"),
    ("secondary_azimuth_deg", "float"),
    ("facets", "list<item: struct<facet_index: int8, area_share: float, area_m2: float, "
               "surface_area_m2: float, tilt_deg: float, azimuth_deg: float, "
               "fit_rmse_m: float, tilt_unc_deg: float, azimuth_unc_deg: float, "
               "n_points: int32, height_above_ground_m: float, geometry: binary>>"),
    ("roof_ref_source", "string"), ("open_ground_share", "float"),
    ("roof_ref_method", "string"), ("roof_tilt_deg", "float"),
    ("roof_azimuth_deg", "float"), ("roof_rmse_m", "float"),
    ("angle_to_roof_deg", "float"), ("height_above_roof_m", "float"),
    ("height_above_ground_m", "float"), ("on_building", "bool"),
    ("building_id", "string"), ("flags", "list<item: string>"),
    ("lidar_tile_ids", "list<item: string>"), ("pkg_version", "string"),
    ("config_hash", "string"), ("run_id", "string"), ("partition_id", "int32"),
]
BASES_1_0 = {"panel_confirmed", "panel_by_install_date", "panel_by_vintage", "free_standing",
             "surface_unresolved", "surface_before_install", "unscreened", "no_fit",
             "not_measured"}


def test_schema_1_0_is_frozen() -> None:
    assert SCHEMA_VERSION.split(".")[0] == "1"
    current = {f.name: str(f.type) for f in OUTPUT_SCHEMA}
    for name, kind in SCHEMA_1_0:
        assert name in current, f"column {name} was removed: breaking, needs schema 2.0"
        assert current[name] == kind, f"column {name} changed type: breaking, needs schema 2.0"
    added = [f for f in OUTPUT_SCHEMA if f.name not in dict(SCHEMA_1_0)]
    assert all(f.nullable for f in added), "columns added after 1.0 must be nullable"
    assert set(GEOMETRY_BASIS) >= BASES_1_0, "a geometry_basis value was removed"
