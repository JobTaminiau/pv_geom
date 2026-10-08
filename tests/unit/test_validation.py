"""External reference comparison (PVDAQ-style): mounts against measured facets."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import pv_geom
from pv_geom.cli import app
from pv_geom.sample import write_demo
from pv_geom.validation import (
    ELIGIBLE,
    NO_MEASUREMENT,
    NOT_PRESENT,
    PRESENCE_UNKNOWN,
    SUSPECT,
    ReferenceTableError,
    circular_difference_deg,
    compare_to_reference,
    normalise_references,
)


def _poly(pid: str, tilt: float | None, az: float | None, area: float = 10.0,
          basis: str = "panel_confirmed", segments: list | None = None) -> dict:
    return {"polygon_id": pid, "panel_tilt_deg": tilt, "panel_azimuth_deg": az,
            "area_m2": area, "geometry_basis": basis, "lidar_date": date(2020, 11, 26),
            "segments": segments}


def _seg(i: int, tilt: float, az: float, area: float) -> dict:
    return {"segment_index": i, "tilt_deg": tilt, "azimuth_deg": az, "area_m2": area}


def _ref(**kw) -> dict:
    return {"reference_id": "r1", "polygon_ids": "a", "tilt_deg": 18, "azimuth_deg": 180,
            "present_by": "2017-11-30", **kw}


def _compare(polys: list[dict], refs: list[dict], **kw):
    return compare_to_reference(pd.DataFrame(polys), pd.DataFrame(refs), **kw)


def test_circular_difference_wraps() -> None:
    assert circular_difference_deg(350, 10) == pytest.approx(-20)
    assert circular_difference_deg(10, 350) == pytest.approx(20)
    assert circular_difference_deg(173, 180) == pytest.approx(-7)


def test_a_matching_array_is_consistent() -> None:
    res = _compare([_poly("a", 18.4, 176.0)], [_ref()])
    (r,) = res.references.itertuples()
    assert r.eligibility == ELIGIBLE and r.outcome == "consistent"
    assert r.area_share_agreeing == pytest.approx(1.0)
    assert r.tilt_error_deg == pytest.approx(0.4) and r.azimuth_error_deg == pytest.approx(-4.0)


def test_compass_point_azimuth_is_not_held_against_the_measurement() -> None:
    """PVOutput reports azimuth as one of eight compass points. A measured 160
    against a reported 'south' is within what the reference can express."""
    polys = [_poly("a", 18.0, 160.0)]
    fine = _compare(polys, [_ref()])
    coarse = _compare(polys, [_ref(azimuth_resolution_deg=45)])
    assert fine.references.outcome.iloc[0] == "inconsistent"
    assert coarse.references.outcome.iloc[0] == "consistent"
    f = coarse.facets.iloc[0]
    assert f.azimuth_error_deg == pytest.approx(-20.0)           # the raw difference is kept
    assert f.azimuth_error_beyond_resolution_deg == pytest.approx(0.0)


def test_one_mount_record_for_a_system_with_two_tilts() -> None:
    """The subarray problem: one record, several blocks. The result is the
    share of area the record describes — not the best block's error."""
    polys = [_poly("a", 17.8, 173.5, area=20.0), _poly("b", 18.1, 173.0, area=10.0),
             _poly("c", 11.1, 173.2, area=13.0), _poly("d", 11.6, 173.3, area=12.0)]
    res = _compare(polys, [_ref(polygon_ids="a;b;c;d", azimuth_resolution_deg=45)])
    (r,) = res.references.itertuples()
    assert r.outcome == "partly_consistent" and r.n_facets == 4
    assert r.area_share_agreeing == pytest.approx(30.0 / 55.0)
    # A whole-system record yields no accuracy statistic.
    assert res.summary["mount_scope_accuracy"] is None


def test_mount_scope_references_are_scored() -> None:
    polys = [_poly("a", 19.0, 181.0), _poly("b", 9.0, 268.0)]
    refs = [_ref(reference_id="m1", scope="mount", grade="documented"),
            _ref(reference_id="m2", polygon_ids="b", tilt_deg=10, azimuth_deg=270,
                 scope="mount", grade="documented")]
    acc = _compare(polys, refs).summary["mount_scope_accuracy"]
    assert acc["n_references"] == 2 and acc["n_facets"] == 2
    assert acc["tilt_mae_deg"] == pytest.approx(1.0) and acc["tilt_bias_deg"] == pytest.approx(0.0)
    assert acc["azimuth_mae_deg"] == pytest.approx(1.5)


def test_every_facet_of_a_multi_facet_polygon_is_compared() -> None:
    segs = [_seg(0, 25.0, 270.0, 6.0), _seg(1, 25.0, 90.0, 4.0)]
    res = _compare([_poly("a", 25.0, 270.0, segments=segs)],
                   [_ref(tilt_deg=25, azimuth_deg=90)])
    assert len(res.facets) == 2
    assert res.references.area_share_agreeing.iloc[0] == pytest.approx(0.4)
    assert list(res.facets.agrees) == [False, True]


def test_azimuth_is_not_compared_for_near_flat_planes() -> None:
    res = _compare([_poly("a", 2.0, 295.0)], [_ref(tilt_deg=1, azimuth_deg=225)])
    f = res.facets.iloc[0]
    assert pd.isna(f.azimuth_error_deg) and f.azimuth_agrees is None and f.agrees


def test_placeholder_references_are_not_scored() -> None:
    res = _compare([_poly("a", 2.0, 295.0)], [_ref(tilt_deg=1, suspect=True)])
    (r,) = res.references.itertuples()
    assert r.eligibility == SUSPECT and r.outcome == "not_scored"
    assert res.summary["eligible"]["n"] == 0


def test_array_absent_at_the_lidar_date_is_a_negative_control() -> None:
    """Installed after the flight: agreement would say nothing about panels.
    What matters is that the tool did not call the surface a confirmed panel."""
    polys = [_poly("a", 18.0, 270.0, basis="surface_unresolved"),
             _poly("b", 18.0, 180.0, basis="panel_confirmed", area=30.0)]
    refs = [_ref(polygon_ids="a;b", present_by=None, absent_at="2022-10-31")]
    res = _compare(polys, refs)
    (r,) = res.references.itertuples()
    assert r.eligibility == NOT_PRESENT and r.outcome == "not_scored"
    control = res.summary["negative_control"]
    assert control["n_references"] == 1
    assert control["area_share_panel_confirmed"] == pytest.approx(0.75)


def test_presence_must_be_evidenced_to_count_as_eligible() -> None:
    unknown = _compare([_poly("a", 18.0, 180.0)], [_ref(present_by=None)])
    late = _compare([_poly("a", 18.0, 180.0)], [_ref(present_by="2021-06-01")])
    assert unknown.references.eligibility.iloc[0] == PRESENCE_UNKNOWN
    assert late.references.eligibility.iloc[0] == PRESENCE_UNKNOWN
    assert unknown.references.outcome.iloc[0] == "consistent"     # compared, not counted
    assert unknown.summary["eligible"]["n"] == 0


def test_unmeasured_and_missing_polygons() -> None:
    res = _compare([_poly("a", None, None, basis="no_fit")], [_ref(polygon_ids="a;zzz")])
    (r,) = res.references.itertuples()
    assert r.eligibility == NO_MEASUREMENT
    assert r.n_polygons_listed == 2 and r.n_polygons_found == 1 and r.n_facets == 0


def test_reference_table_is_checked() -> None:
    with pytest.raises(ReferenceTableError, match="missing column"):
        normalise_references(pd.DataFrame([{"reference_id": "r"}]))
    with pytest.raises(ReferenceTableError, match="not unique"):
        normalise_references(pd.DataFrame([_ref(), _ref()]))
    with pytest.raises(ReferenceTableError, match="scope"):
        normalise_references(pd.DataFrame([_ref(scope="array")]))
    refs = normalise_references(pd.DataFrame([_ref(suspect="true")]))
    assert bool(refs.suspect.iloc[0]) and refs.scope.iloc[0] == "system"
    assert refs.azimuth_resolution_deg.iloc[0] == 1.0


def test_end_to_end_on_the_demo_scene(tmp_path: Path) -> None:
    """The demo's arrays against references taken from their own measurement
    (rounded as a reporting source would): the CLI scores them as consistent."""
    result = pv_geom.run(write_demo(tmp_path / "demo"), use_dask=False)
    gdf = result.load()
    measured = gdf[(gdf["status"] == "measured") & (gdf["panel_tilt_deg"] > 5)].head(3)
    refs = pd.DataFrame([{
        "reference_id": f"ref-{i}", "polygon_ids": r.polygon_id,
        "tilt_deg": round(float(r.panel_tilt_deg)),
        "azimuth_deg": round(float(r.panel_azimuth_deg) / 45) * 45 % 360,
        "azimuth_resolution_deg": 45, "scope": "mount", "present_by": "2020-01-01",
    } for i, r in enumerate(measured.itertuples())])
    ref_path = tmp_path / "refs.csv"
    refs.to_csv(ref_path, index=False)

    out = tmp_path / "validation"
    res = CliRunner().invoke(app, ["compare-reference", str(result.output),
                                   "--reference", str(ref_path), "--out", str(out)])
    assert res.exit_code == 0, res.output
    per_ref = pd.read_csv(out / "validation_references.csv")
    assert list(per_ref.outcome) == ["consistent"] * 3
    assert (out / "validation_facets.csv").exists() and (out / "validation_summary.json").exists()

    api_result = pv_geom.compare_reference(result.output, ref_path)
    assert api_result.summary["mount_scope_accuracy"]["tilt_mae_deg"] < 0.6
