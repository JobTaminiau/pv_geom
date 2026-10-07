"""Milestone 0.4: config-driven runs, the Python API, true-north azimuth, the
header-built tile index, the demo, and report options."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import laspy
import numpy as np
import pytest
from pyproj import CRS, Geod, Transformer
from shapely.geometry import box
from typer.testing import CliRunner

import pv_geom
from pv_geom.cli import app
from pv_geom.config import PVGeomConfig
from pv_geom.errors import InputError, TileIndexError
from pv_geom.io.tile_scan import scan_tile_index
from pv_geom.pipeline.worker import build_row
from pv_geom.sample import ARRAYS, grid_convergence, write_demo
from pv_geom.utils.north import meridian_convergence_deg, to_true_azimuth

from .scenes import write_synthetic_laz

runner = CliRunner()


# --------------------------------------------------------------------------- #
# E6: azimuth is measured from true north
# --------------------------------------------------------------------------- #

# Three metric CRSs covering one place (near Phoenix), with different grid
# north: two adjacent UTM zones and Arizona Central State Plane.
PLACE = (-112.40, 33.60)
CRS_SET = ["EPSG:6341", "EPSG:6340", "EPSG:26949"]      # UTM 12N, UTM 11N, AZ Central (m)


def _xy(crs: str) -> tuple[float, float]:
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(*PLACE)


@pytest.mark.parametrize("crs", CRS_SET)
def test_convergence_matches_an_independent_geodesic_computation(crs: str) -> None:
    """PROJ's meridian convergence equals the true bearing of a short step
    along grid north, computed on the ellipsoid."""
    x, y = _xy(crs)
    to_ll = Transformer.from_crs(crs, CRS.from_user_input(crs).geodetic_crs, always_xy=True)
    lon0, lat0 = to_ll.transform(x, y)
    lon1, lat1 = to_ll.transform(x, y + 200.0)
    bearing, _, _ = Geod(ellps="GRS80").inv(lon0, lat0, lon1, lat1)
    assert meridian_convergence_deg(crs, x, y)[0] == pytest.approx(bearing, abs=1e-3)


def test_the_three_grids_really_disagree_about_north() -> None:
    conv = [float(meridian_convergence_deg(c, *_xy(c))[0]) for c in CRS_SET]
    assert max(conv) - min(conv) > 2.0            # UTM 11N vs 12N here: ~3 degrees apart
    assert all(abs(c) > 0.1 for c in conv)


@pytest.mark.parametrize("true_azimuth", [0.0, 135.0, 180.0, 270.0, 359.5])
def test_one_physical_plane_measures_the_same_in_every_crs(true_azimuth: float) -> None:
    """The acceptance test for E6: a plane of known TRUE azimuth, laid out in
    each CRS's own grid coordinates, must come back with that true azimuth
    whichever CRS the work is done in."""
    tilt = 25.0
    rng = np.random.default_rng(11)
    measured = []
    for crs in CRS_SET:
        x0, y0 = _xy(crs)
        conv = float(meridian_convergence_deg(crs, x0, y0)[0])
        # In this grid the plane faces (true - convergence).
        a, t = np.radians(true_azimuth - conv), np.radians(tilt)
        nx, ny, nz = np.sin(t) * np.sin(a), np.sin(t) * np.cos(a), np.cos(t)
        xy = rng.uniform([x0, y0], [x0 + 8, y0 + 5], size=(600, 2))
        z = 300.0 - (nx * (xy[:, 0] - x0) + ny * (xy[:, 1] - y0)) / nz + rng.normal(0, 0.005, 600)
        pts = np.column_stack([xy, z])
        row = build_row(
            polygon=box(x0, y0, x0 + 8, y0 + 5), polygon_id="p", cfg=PVGeomConfig(),
            config_hash="x", run_id="r", partition_id=0,
            panel_pts=pts, ground_xyz=np.zeros((0, 3)), roof_input_pts=pts, footprints=None,
            other_pv_polygons=gpd.GeoDataFrame(geometry=[], crs=crs),
            contributing_tile_ids=("t",), grid_convergence_deg=conv,
        )
        assert row["grid_convergence_deg"] == pytest.approx(conv, abs=1e-4)
        assert row["panel_tilt_deg"] == pytest.approx(tilt, abs=0.05)     # tilt needs no correction
        measured.append(row["panel_azimuth_deg"])

    for az in measured:
        d = abs(az - true_azimuth) % 360
        assert min(d, 360 - d) < 0.05, measured


def test_to_true_azimuth_wraps_and_passes_missing_through() -> None:
    assert to_true_azimuth(359.5, 1.0) == pytest.approx(0.5)
    assert to_true_azimuth(0.3, -0.8) == pytest.approx(359.5)
    assert to_true_azimuth(None, 1.0) is None
    assert np.isnan(to_true_azimuth(float("nan"), 1.0))


def test_pipeline_reports_true_north_and_says_so(tmp_path: Path) -> None:
    """End to end on the sample area, whose truth is stated in true north and
    whose grid is 0.8 degrees off it: a grid-north answer would fail this."""
    assert abs(grid_convergence()) > 0.5
    result = pv_geom.run(write_demo(tmp_path), use_dask=False)
    assert result.manifest["azimuth_reference"] == "true_north"
    rows = {r["input_row"]: r for r in result.load().to_dict("records")}
    checked = 0
    for i, a in enumerate(ARRAYS):
        if a.kind != "roof" or a.noise > 0.02 or a.panel_tilt < 5:
            continue
        d = abs(rows[i]["panel_azimuth_deg"] - a.panel_az) % 360
        assert min(d, 360 - d) < 0.3, (a.name, rows[i]["panel_azimuth_deg"])
        assert rows[i]["grid_convergence_deg"] == pytest.approx(grid_convergence(), abs=0.01)
        checked += 1
    assert checked >= 5


# --------------------------------------------------------------------------- #
# I7 / I2: runs driven by a config, through the API and the CLI
# --------------------------------------------------------------------------- #


@pytest.fixture
def study_config(synth_inputs: dict[str, Path], tmp_path: Path) -> Path:
    """A config that names everything, with paths relative to the config file."""
    root = synth_inputs["laz_dir"]
    path = root / "study.yaml"
    path.write_text(
        "study:\n  name: Testville\n  output: out_from_config\n"
        "inputs:\n"
        f"  polygons: {synth_inputs['polygons'].name}\n"
        f"  tile_index: {synth_inputs['tindex'].name}\n"
        "  lidar_prefix: .\n"
        "  name_template: tile.laz\n"
        f"  footprints: {synth_inputs['footprints'].name}\n"
        "vintage:\n  polygon_vintage: 2024-10\n",
        encoding="utf-8")
    return path


def test_run_needs_nothing_but_the_config(study_config: Path) -> None:
    result = runner.invoke(app, ["run", "--config", str(study_config), "--no-dask"])
    assert result.exit_code == 0, result.output
    out = study_config.parent / "out_from_config"          # relative to the config file
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["counts"]["rows"] == 1
    assert manifest["config"]["study"]["name"] == "Testville"
    assert manifest["vintage"]["polygon_vintage"] == "2024-10-31"
    # The study name reaches the report without being passed again.
    assert "Testville" in (out / "report" / "report.md").read_text(encoding="utf-8")


def test_command_line_overrides_the_config(study_config: Path, tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "--config", str(study_config), "--no-dask",
                                 "--output", str(tmp_path / "elsewhere"),
                                 "--polygon-vintage", "2019", "--no-report"])
    assert result.exit_code == 0, result.output
    manifest = json.loads((tmp_path / "elsewhere" / "manifest.json").read_text())
    assert manifest["vintage"]["polygon_vintage"] == "2019-12-31"
    assert not (study_config.parent / "out_from_config").exists()


def test_validate_config_shows_resolved_inputs(study_config: Path) -> None:
    result = runner.invoke(app, ["validate-config", str(study_config)])
    assert result.exit_code == 0
    flat = result.output.replace("\n", "")
    assert "Testville" in flat and "out_from_config" in flat


def test_missing_inputs_say_which_and_where(tmp_path: Path) -> None:
    with pytest.raises(InputError) as err:
        pv_geom.run(polygons="p.parquet")
    assert "lidar-prefix" in err.value.message and "output" in err.value.message
    assert "inputs.lidar_prefix" in err.value.remedy and "study.output" in err.value.remedy


def test_python_api_round_trip(study_config: Path) -> None:
    result = pv_geom.run(study_config, use_dask=False)
    gdf = result.load()
    assert len(gdf) == 1 and gdf.crs.to_epsg() == 6341
    assert gdf["status"].iloc[0] == "measured"
    assert pv_geom.describe(result.output)["fitted"] == 1
    report = result.report(export_dataset=False)
    assert report.html.exists() and report.summary["n_rows"] == 1
    assert pv_geom.load(result.output).equals(gdf)


def test_config_object_is_not_mutated_by_a_run(study_config: Path) -> None:
    cfg = PVGeomConfig.from_yaml(study_config)
    pv_geom.run(cfg, use_dask=False, polygon_vintage="2001")
    assert str(cfg.vintage.polygon_vintage) == "2024-10"
    assert cfg.crs.target == "auto"


def test_inputs_and_study_do_not_change_what_resume_compares(study_config: Path) -> None:
    """Moving the inputs or the output is not a change of measurement settings."""
    a = PVGeomConfig.from_yaml(study_config)
    b = PVGeomConfig()
    b.vintage.polygon_vintage = "2024-10"
    assert a.result_hash() == b.result_hash()
    assert a.hash() != b.hash()                    # but the provenance hash records the inputs


def test_shipped_study_configs_name_their_inputs() -> None:
    configs = Path(__file__).resolve().parents[2] / "configs"
    for name in ("phoenix.yaml", "delaware.yaml"):
        cfg = PVGeomConfig.from_yaml(configs / name)
        assert cfg.study.name and cfg.inputs.polygons and cfg.inputs.lidar_prefix, name
        assert cfg.vintage.polygon_vintage is not None, name


# --------------------------------------------------------------------------- #
# D2: tile index from the tiles' own headers
# --------------------------------------------------------------------------- #


def _tile_with_crs(path: Path, x0: float, crs: str | None = "EPSG:6341") -> None:
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    if crs:
        header.add_crs(CRS.from_user_input(crs))
    las = laspy.LasData(header)
    las.x = np.array([x0, x0 + 100.0])
    las.y = np.array([0.0, 50.0])
    las.z = np.array([0.0, 1.0])
    las.write(str(path))


def test_scan_builds_an_index_from_headers(tmp_path: Path) -> None:
    _tile_with_crs(tmp_path / "a.laz", 0.0)
    _tile_with_crs(tmp_path / "b.laz", 100.0)
    (tmp_path / "notes.txt").write_text("not a tile")
    index = scan_tile_index(tmp_path)
    assert index["Name"].tolist() == ["a", "b"] and index.crs.to_epsg() == 6341
    assert index.geometry.iloc[1].bounds == pytest.approx((100.0, 0.0, 200.0, 50.0))
    assert all(Path(p).exists() for p in index["tile_path"])


def test_scan_errors_are_actionable(tmp_path: Path) -> None:
    with pytest.raises(TileIndexError, match=r"no \.laz or \.las tiles") as err:
        scan_tile_index(tmp_path)
    assert "--tile-index" in err.value.remedy

    _tile_with_crs(tmp_path / "a.laz", 0.0)
    _tile_with_crs(tmp_path / "b.laz", 100.0, crs="EPSG:6340")
    with pytest.raises(TileIndexError, match="more than one CRS"):
        scan_tile_index(tmp_path)

    other = tmp_path / "nocrs"
    other.mkdir()
    write_synthetic_laz(other / "t.laz")                  # written without a CRS
    with pytest.raises(TileIndexError, match="declare no CRS") as err:
        scan_tile_index(other)
    assert "--crs" in err.value.remedy


# --------------------------------------------------------------------------- #
# I1: the demo;  F5: report options
# --------------------------------------------------------------------------- #


def test_demo_command_runs_offline_and_writes_everything(tmp_path: Path) -> None:
    result = runner.invoke(app, ["demo", "--out", str(tmp_path / "demo")])
    assert result.exit_code == 0, result.output
    demo = tmp_path / "demo"
    assert (demo / "demo.yaml").exists() and (demo / "inputs" / "bench.laz").exists()
    assert not (demo / "inputs" / "tile_index.parquet").exists()      # indexed from the header
    assert (demo / "output" / "report" / "report.html").exists()
    manifest = json.loads((demo / "output" / "manifest.json").read_text())
    assert manifest["counts"]["rows"] == len(ARRAYS)
    assert manifest["inputs"]["tile_index"] is None


def test_report_headline_and_weight_can_be_chosen(tmp_path: Path) -> None:
    result = pv_geom.run(write_demo(tmp_path), use_dask=False)
    default = result.report(tmp_path / "r1", export_dataset=False).summary
    chosen = result.report(tmp_path / "r2", export_dataset=False,
                           headline="all_fitted", weight="count").summary
    assert default["weight"] == "area" and chosen["weight"] == "count"
    assert chosen["headline_stratum"] == "all_fitted"
    text = (tmp_path / "r2" / "report.md").read_text(encoding="utf-8")
    assert "per polygon" in text and "Share of polygons" in text
    assert "true north" in text

    with pytest.raises(InputError, match="unknown weight"):
        result.report(tmp_path / "r3", weight="capacity")
    with pytest.raises(InputError, match="unknown headline"):
        result.report(tmp_path / "r3", headline="everything")


def test_report_on_an_older_output_says_grid_north(tmp_path: Path) -> None:
    """An output written before azimuths were converted must not be presented
    as true north."""
    result = pv_geom.run(write_demo(tmp_path), use_dask=False)
    manifest_path = Path(result.output) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["azimuth_reference"]
    manifest_path.write_text(json.dumps(manifest))
    report = result.report(tmp_path / "old", export_dataset=False)
    assert report.summary["azimuth_reference"] == "grid_north"
    assert "GRID north" in report.methods.read_text(encoding="utf-8")
    assert "not true north" in report.markdown.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# D4: the LiDAR's classes are checked before any work is done
# --------------------------------------------------------------------------- #


def _classes_tile(path: Path, classes: list[int]) -> str:
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    las = laspy.LasData(header)
    n = 60
    las.x, las.y, las.z = np.linspace(0, 10, n), np.linspace(0, 10, n), np.zeros(n)
    las.classification = np.array([classes[i % len(classes)] for i in range(n)], dtype=np.uint8)
    las.write(str(path))
    return str(path)


def test_class_check_names_the_candidate_class(tmp_path: Path, caplog) -> None:
    import logging

    from pv_geom.pipeline.vintage_check import probe_lidar_vintage

    with caplog.at_level(logging.INFO, logger="pv_geom"):
        with_buildings = probe_lidar_vintage(
            PVGeomConfig(), [_classes_tile(tmp_path / "a.laz", [1, 2, 6])])
        without = probe_lidar_vintage(
            PVGeomConfig(), [_classes_tile(tmp_path / "b.laz", [1, 2, 7])])
    assert with_buildings["panel_candidate_class"] == 6
    assert without["panel_candidate_class"] == 1 and without["lidar_classes_sampled"] == [1, 2, 7]
    assert any("above local ground" in r.getMessage() for r in caplog.records)


def test_no_ground_class_stops_the_run_with_a_remedy(tmp_path: Path) -> None:
    from pv_geom.errors import LidarClassError
    from pv_geom.pipeline.vintage_check import probe_lidar_vintage

    with pytest.raises(LidarClassError, match="no ground returns") as err:
        probe_lidar_vintage(PVGeomConfig(), [_classes_tile(tmp_path / "a.laz", [1, 5])])
    assert "io.classification.ground_class" in err.value.remedy
    assert "[1, 5]" in err.value.message

    # ...and the remedy works: tell it which class is ground.
    cfg = PVGeomConfig()
    cfg.io.classification.ground_class = 5
    assert probe_lidar_vintage(cfg, [_classes_tile(tmp_path / "b.laz", [1, 5])])


def test_no_candidate_class_stops_the_run(tmp_path: Path) -> None:
    from pv_geom.errors import LidarClassError
    from pv_geom.pipeline.vintage_check import probe_lidar_vintage

    with pytest.raises(LidarClassError, match="neither class 6 nor class 1"):
        probe_lidar_vintage(PVGeomConfig(), [_classes_tile(tmp_path / "a.laz", [2, 9])])


# --------------------------------------------------------------------------- #
# H2: an estimate before the run
# --------------------------------------------------------------------------- #


def test_dry_run_estimates_size_and_time(study_config: Path) -> None:
    result = pv_geom.run(study_config, dry_run=True)
    est = result.manifest["estimate"]
    assert est["tile_groups"] == 1 and est["polygons"] == 1 and est["tiles"] == 1
    assert est["tiles_sized"] == 1 and est["gigabytes"] >= 0
    assert est["wall_seconds"] >= 0 and "factor of two" in est["accuracy"]
    assert "usd" not in est                              # no price configured, local backend


def test_estimate_scales_with_data_and_prices_coiled_runs() -> None:
    from pv_geom.pipeline import estimate as est_mod
    from pv_geom.pipeline.partition import TileGroup

    plan = SimpleNamespace(tile_uri_map={"t1": "a.laz", "t2": "b.laz"})
    groups = [(0, TileGroup("t1", tuple(f"p{i}" for i in range(1000)), ("t1",))),
              (1, TileGroup("t2", tuple(f"q{i}" for i in range(1000)), ("t2",)))]
    sizes = {"a.laz": 2_000_000_000, "b.laz": 2_000_000_000}
    original = est_mod.object_sizes
    est_mod.object_sizes = lambda uris: {u: sizes[u] for u in uris}
    try:
        cfg = PVGeomConfig()
        cfg.compute.local.n_workers = 2
        local = est_mod.estimate_run(plan, groups, cfg)
        serial = est_mod.estimate_run(plan, groups, cfg, use_dask=False)
        cfg.compute.backend = "coiled"
        cfg.compute.coiled.n_workers = 2
        cfg.compute.coiled.usd_per_worker_hour = 0.10
        coiled = est_mod.estimate_run(plan, groups, cfg)
    finally:
        est_mod.object_sizes = original
    assert local["gigabytes"] == 4.0 and local["polygons"] == 2000
    work = 4 * est_mod.WORKER_SECONDS_PER_GB + 2000 * est_mod.WORKER_SECONDS_PER_POLYGON
    assert local["wall_seconds"] == pytest.approx(work / 2, abs=1)
    assert serial["wall_seconds"] == pytest.approx(work, abs=1)       # one worker
    assert coiled["wall_seconds"] > local["wall_seconds"]             # cluster start-up
    assert coiled["usd"] == pytest.approx(2 * coiled["wall_seconds"] / 3600 * 0.10, abs=0.01)


# --------------------------------------------------------------------------- #
# D3: LiDAR in feet, or in another CRS than the run's
# --------------------------------------------------------------------------- #

def _demo_in(crs: str, src: Path, dst: Path) -> Path:
    """The demo study area with its LiDAR tile rewritten into ``crs``,
    horizontal and vertical both in that CRS's unit."""
    import shutil

    import laspy

    from pv_geom.utils.crs import horizontal

    shutil.copytree(src, dst)
    tile = dst / "inputs" / "bench.laz"
    las = laspy.read(str(tile))
    target = CRS.from_user_input(crs)
    to_m = horizontal(target).axis_info[0].unit_conversion_factor
    x, y = Transformer.from_crs(las.header.parse_crs(), target, always_xy=True).transform(
        np.asarray(las.x), np.asarray(las.y))
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([float(x.min()), float(y.min()), 0.0])
    header.add_crs(target)
    out = laspy.LasData(header)
    out.x, out.y, out.z = x, y, np.asarray(las.z) / to_m
    out.classification = np.asarray(las.classification)
    out.gps_time = np.asarray(las.gps_time)
    tile.unlink()
    out.write(str(tile))
    return dst / "demo.yaml"


@pytest.mark.parametrize("crs", ["EPSG:2223", "EPSG:2230", "EPSG:6340"],
                         ids=["state-plane-TM-intl-ft", "state-plane-Lambert-US-ft",
                              "other-UTM-zone-m"])
def test_lidar_in_feet_or_another_crs_measures_the_same(crs: str, tmp_path: Path) -> None:
    """The acceptance test for D3: the same scene delivered in a foot-based
    State Plane CRS (Transverse Mercator and Lambert) or a different metric
    zone gives the same tilt and the same true azimuth."""
    from pv_geom.sample import write_demo

    base = pv_geom.run(write_demo(tmp_path / "metric"), use_dask=False)
    other = pv_geom.run(_demo_in(crs, tmp_path / "metric", tmp_path / "other"), use_dask=False)

    a = base.load().set_index("polygon_id")
    b = other.load().set_index("polygon_id").loc[a.index]
    assert list(a["status"]) == list(b["status"])
    assert CRS.from_user_input(b.crs).axis_info[0].unit_name == "metre"
    ok = a["panel_tilt_deg"].notna().to_numpy()
    assert ok.sum() >= 5
    # The points land on a different grid (rotated by the convergence, rounded
    # to the tile's own millimetre), so the returns inside each polygon and the
    # RANSAC draws differ slightly: agreement is close, not bit-exact. The two
    # non-native cases also work in a UTM zone the scene lies outside of, whose
    # grid scale (about 1.001 there) shifts tilt by a few hundredths of a degree;
    # the in-zone foot case agrees to a thousandth.
    d_tilt = np.abs(a["panel_tilt_deg"] - b["panel_tilt_deg"])[ok]
    assert d_tilt.median() < 0.08 and d_tilt.max() < 0.6
    if crs == "EPSG:2223":
        assert d_tilt.max() < 0.01
    both = ok & a["panel_azimuth_deg"].notna().to_numpy() & b["panel_azimuth_deg"].notna().to_numpy()
    d_az = np.abs((a["panel_azimuth_deg"] - b["panel_azimuth_deg"] + 180.0) % 360.0 - 180.0)[both]
    assert d_az.median() < 0.2 and d_az.max() < 1.5
    for col in ("height_above_ground_m", "area_m2"):
        assert np.allclose(a[col][ok], b[col][ok], rtol=0.02, atol=0.02), col
    # An array sitting at the standoff threshold can fall either side of it.
    changed = int((a["geometry_basis"] != b["geometry_basis"]).sum())
    assert changed <= (0 if crs == "EPSG:2223" else 1)


def test_foot_based_lidar_run_records_the_conversion(tmp_path: Path) -> None:
    from pv_geom.io.output import read_manifest
    from pv_geom.sample import write_demo

    write_demo(tmp_path / "metric")
    run = pv_geom.run(_demo_in("EPSG:2223", tmp_path / "metric", tmp_path / "ft"),
                      use_dask=False)
    manifest = read_manifest(run.output)
    converted = manifest["vintage"]["lidar_converted_on_read"]
    (what,) = converted.values()
    assert "horizontal" in what and "heights foot -> metre" in what
    assert manifest["crs"] != "EPSG:2223"


def test_metric_equivalent_and_point_conversion() -> None:
    from pv_geom.utils.crs import metric_equivalent, to_run_crs

    assert CRS.from_user_input(metric_equivalent("EPSG:2223")).axis_info[0].unit_name == "metre"
    pts = np.array([[100.0, 200.0, 30.0, 1.0]])
    same, what = to_run_crs(pts, "EPSG:6341", "EPSG:6341")
    assert what is None and same is pts
    untouched, what = to_run_crs(pts, None, "EPSG:6341")
    assert what is None and untouched is pts
    x, y = Transformer.from_crs("EPSG:6341", "EPSG:2223", always_xy=True).transform(*_xy("EPSG:6341"))
    ft, what = to_run_crs(np.array([[x, y, 1000.0, 6.0]]), "EPSG:2223", "EPSG:6341")
    assert ft[0, :2] == pytest.approx(_xy("EPSG:6341"), abs=1e-3)
    assert ft[0, 2] == pytest.approx(304.8) and ft[0, 3] == 6.0
