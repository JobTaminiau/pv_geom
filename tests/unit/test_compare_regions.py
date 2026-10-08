"""Milestone 0.5: runs side by side (F1) and results by region (F2)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box
from typer.testing import CliRunner

import pv_geom
from pv_geom.cli import app
from pv_geom.errors import InputError
from pv_geom.sample import write_demo


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory) -> pv_geom.RunResult:
    return pv_geom.run(write_demo(tmp_path_factory.mktemp("demo")), use_dask=False)


def _halves(run: pv_geom.RunResult, path: Path) -> Path:
    """West and east halves of the demo's extent, with a nameless sliver."""
    gdf = run.load()
    x0, y0, x1, y1 = gdf.total_bounds
    mid = (x0 + x1) / 2
    layer = gpd.GeoDataFrame(
        {"district": ["West", "East", None]},
        geometry=[box(x0 - 5, y0 - 5, mid, y1 + 5), box(mid, y0 - 5, x1 + 5, y1 + 5),
                  box(x1 + 50, y0, x1 + 60, y1)], crs=gdf.crs).to_crs("EPSG:4326")
    layer.to_file(path, driver="GPKG")
    return path


def test_report_by_region(demo_run, tmp_path: Path) -> None:
    regions = _halves(demo_run, tmp_path / "districts.gpkg")
    report = pv_geom.report(demo_run.output, tmp_path / "report", regions=regions,
                            region_col="district", headline="all_fitted")
    table = pd.read_csv(report.tables_dir / "regions.csv")
    assert set(table["region"]) == {"West", "East"}
    gdf = demo_run.load()
    assert table["n_polygons"].sum() == len(gdf)          # every polygon in one region
    assert table["n_measured"].sum() == int((gdf["status"] == "measured").sum())
    assert {"tilt_p50_deg", "share_facing_S", "dominant_orientation"} <= set(table.columns)

    html = report.html.read_text(encoding="utf-8")
    assert "By region" in html and "West" in html
    released = gpd.read_parquet(report.dataset_dir / "pv_geom.parquet")
    assert set(released["region"]) == {"West", "East"}


def test_small_regions_are_counted_but_not_described(demo_run, tmp_path: Path) -> None:
    from pv_geom.report import regions as rg
    from pv_geom.report.data import load_run

    gdf, df, _ = load_run(demo_run.output)
    names, layer = rg.assign_regions(gdf, _halves(demo_run, tmp_path / "d.gpkg"), "district")
    table = rg.regions_table(df, names, "all_fitted")
    small = table[~table["sufficient"]]
    assert small["tilt_p50_deg"].isna().all()
    assert (table["n_polygons"] > 0).all()
    # Every region is drawn or greyed; the figure needs at least one described.
    spec = rg.fig_regions(layer, table, "all_fitted")
    assert (spec is None) == (not table["sufficient"].any())


def test_region_arguments_are_checked(demo_run, tmp_path: Path) -> None:
    regions = _halves(demo_run, tmp_path / "districts.gpkg")
    with pytest.raises(InputError, match="go together"):
        pv_geom.report(demo_run.output, tmp_path / "r1", regions=regions)
    with pytest.raises(InputError, match="no column"):
        pv_geom.report(demo_run.output, tmp_path / "r2", regions=regions, region_col="nope")


def test_report_without_regions_has_no_region_section(demo_run, tmp_path: Path) -> None:
    report = pv_geom.report(demo_run.output, tmp_path / "plain", export_dataset=False)
    assert "By region" not in report.html.read_text(encoding="utf-8")
    assert not (report.tables_dir / "regions.csv").exists()


def test_compare_two_runs(demo_run, tmp_path: Path) -> None:
    other = pv_geom.run(write_demo(tmp_path / "other"), use_dask=False)
    out = tmp_path / "cmp"
    res = CliRunner().invoke(app, ["report", str(demo_run.output), str(other.output),
                                   "--compare", "--out", str(out), "--label", "First",
                                   "--label", "Second"])
    assert res.exit_code == 0, res.output
    table = pd.read_csv(out / "tables" / "comparison.csv")
    assert list(table["run"]) == ["First", "Second"]
    # Same inputs, same code: the two columns must agree exactly.
    assert table["tilt_p50_deg"].nunique() == 1 and table["n_polygons"].nunique() == 1
    for name in ("tilt_profile_comparison.csv", "azimuth_profile_comparison.csv"):
        prof = pd.read_csv(out / "tables" / name)
        assert set(prof["run"]) == {"First", "Second"}
        assert prof.groupby("run")["share"].sum().round(6).eq(1.0).all()
    assert (out / "figures" / "profile_comparison.png").stat().st_size > 5000
    md = (out / "comparison.md").read_text(encoding="utf-8")
    assert "First vs Second" in md and "Median tilt" in md


def test_compare_labels_default_to_distinct_names(demo_run, tmp_path: Path) -> None:
    res = pv_geom.compare_runs([demo_run.output, demo_run.output], tmp_path / "cmp")
    assert res.table["run"].is_unique


def test_compare_needs_two_runs_and_an_out_dir(demo_run, tmp_path: Path) -> None:
    with pytest.raises(InputError, match="2 to 4"):
        pv_geom.compare_runs([demo_run.output], tmp_path / "x")
    runner = CliRunner()
    res = runner.invoke(app, ["report", str(demo_run.output), str(demo_run.output)])
    assert res.exit_code != 0
    res = runner.invoke(app, ["report", str(demo_run.output), str(demo_run.output), "--compare"])
    assert res.exit_code != 0
