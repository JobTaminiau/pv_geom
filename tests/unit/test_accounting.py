"""Milestone 0.3: every polygon accounted for, safe resume, progress, typed errors."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq
import pytest
from shapely.geometry import LineString, MultiPolygon, Polygon, box

from pv_geom.config import PVGeomConfig
from pv_geom.errors import (
    CRSResolutionError,
    MissingCRSError,
    MissingDependencyError,
    NonMetricCRSError,
    PolygonIdError,
    PVGeomError,
    ResumeMismatchError,
    TileIndexError,
    VintageFormatError,
    require,
)
from pv_geom.io.output import META_PLAN_FINGERPRINT, META_SCHEMA_VERSION, read_output_table
from pv_geom.io.polygons import read_polygons
from pv_geom.pipeline.runner import run_pipeline
from pv_geom.schema import OUTPUT_SCHEMA, SCHEMA_VERSION, STATUSES
from pv_geom.vintage import NOT_MEASURED

from .scenes import write_synthetic_laz

CRS = "EPSG:6341"
BOWTIE = Polygon([(60, 60), (70, 70), (70, 60), (60, 70)])     # self-intersecting


# --------------------------------------------------------------------------- #
# B2: the polygon layer is screened, never silently filtered
# --------------------------------------------------------------------------- #


def _layer(tmp_path: Path, geoms: list, **cols) -> Path:
    path = tmp_path / "layer.parquet"
    gpd.GeoDataFrame(cols, geometry=geoms, crs=CRS).to_parquet(path)
    return path


def test_unusable_features_are_kept_with_a_reason(tmp_path: Path) -> None:
    path = _layer(tmp_path, [box(0, 0, 4, 4), None, Polygon(), LineString([(0, 0), (1, 1)])])
    out = read_polygons(path, target_crs=CRS)
    assert len(out) == 4
    assert out["input_issue"].tolist() == [None, "null_geometry", "empty_geometry",
                                           "not_polygonal"]
    assert out["input_row"].tolist() == [0, 1, 2, 3]


def test_invalid_polygon_is_repaired_and_flagged(tmp_path: Path) -> None:
    out = read_polygons(_layer(tmp_path, [BOWTIE]), target_crs=CRS)
    assert out["input_issue"].isna().all()
    assert out.geometry.is_valid.all() and out.geometry.area.sum() == pytest.approx(50.0)
    assert all("geometry_repaired" in f for f in out["input_flags"])


def test_multipolygon_parts_sharing_an_edge_stay_separate(tmp_path: Path) -> None:
    """Invalid as a whole, but each part is a good array: explode, don't merge."""
    multi = MultiPolygon([box(0, 0, 5, 10), box(5, 0, 10, 10)])
    out = read_polygons(_layer(tmp_path, [multi], polygon_id=["m"]), target_crs=CRS)
    assert out["polygon_id"].tolist() == ["m__p0", "m__p1"]
    assert out["input_flags"].tolist() == [(), ()]


def test_geometry_collection_keeps_its_polygons_and_sheds_debris(tmp_path: Path) -> None:
    """A dissolve can return polygons with a stray line attached: the polygons
    are the arrays, the line is nothing, and it must not become a row."""
    from shapely.geometry import GeometryCollection

    mixed = GeometryCollection([box(0, 0, 4, 4), LineString([(4, 4), (9, 9)]),
                                box(10, 10, 14, 14)])
    only_line = GeometryCollection([LineString([(0, 0), (1, 1)])])
    out = read_polygons(_layer(tmp_path, [mixed, only_line], polygon_id=["c", "junk"]),
                        target_crs=CRS)
    assert out["polygon_id"].tolist() == ["c__p0", "c__p1", "junk"]
    assert out["input_issue"].tolist() == [None, None, "not_polygonal"]
    assert out.geometry.iloc[:2].area.tolist() == [16.0, 16.0]


def test_input_quality_flags(tmp_path: Path) -> None:
    geoms = [
        box(0, 0, 4, 4),                 # 0: fine
        box(0, 0, 4, 4),                 # 1: duplicate of 0 (and so overlaps it)
        box(20, 20, 20.5, 20.5),         # 2: 0.25 m2
        box(40, 40, 50, 50),             # 3: overlapped by 4 over 25% of its area
        box(45, 40, 60, 50),             # 4: 5x10 of its 150 m2 shared = 33%
        box(80, 80, 90, 90),             # 5: touches nothing
    ]
    out = read_polygons(_layer(tmp_path, geoms), target_crs=CRS,
                        min_area_m2=1.0, overlap_flag_frac=0.2)
    flags = [set(f) for f in out["input_flags"]]
    assert flags[0] == {"overlaps_polygon"}
    assert flags[1] == {"duplicate_geometry", "overlaps_polygon"}
    assert flags[2] == {"below_min_area"}
    assert flags[3] == {"overlaps_polygon"} and flags[4] == {"overlaps_polygon"}
    assert flags[5] == set()


def test_overlap_check_can_be_switched_off(tmp_path: Path) -> None:
    out = read_polygons(_layer(tmp_path, [box(0, 0, 4, 4), box(1, 1, 5, 5)]), target_crs=CRS)
    assert out["input_flags"].tolist() == [(), ()]


# --------------------------------------------------------------------------- #
# B1 / B3 / G1: one row per input polygon, with a status and a reason
# --------------------------------------------------------------------------- #


@pytest.fixture
def mixed_area(tmp_path: Path) -> dict:
    """Six input features that each meet a different fate:

    measured, no fit (nothing but ground under it), on a tile that is in the
    index but not in storage, on an unreadable tile, off the index entirely,
    and with no geometry at all.
    """
    write_synthetic_laz(tmp_path / "good.laz")
    (tmp_path / "corrupt.laz").write_bytes(b"not a laz file")
    gpd.GeoDataFrame(
        {"site": ["roof", "bare_ground", "tile_gone", "tile_corrupt", "far_away", "blank"]},
        geometry=[box(40, 40, 50, 50), box(5, 5, 15, 15), box(140, 40, 150, 50),
                  box(240, 40, 250, 50), box(900, 900, 910, 910), None],
        crs=CRS,
    ).to_parquet(tmp_path / "polys.parquet")
    gpd.GeoDataFrame(
        {"Name": ["good", "absent", "corrupt"]},
        geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100), box(200, 0, 300, 100)], crs=CRS,
    ).to_parquet(tmp_path / "tindex.parquet")
    return dict(
        polygons_uri=str(tmp_path / "polys.parquet"),
        tile_index_uri=str(tmp_path / "tindex.parquet"),
        lidar_prefix=str(tmp_path), output_uri=str(tmp_path / "out"),
        polygon_id_col="site", use_dask=False,
    )


def test_every_input_polygon_gets_exactly_one_row(mixed_area: dict) -> None:
    manifest_path = run_pipeline(**mixed_area, cfg=PVGeomConfig())
    table = read_output_table(mixed_area["output_uri"])
    rows = {r["polygon_id"]: r for r in table.to_pylist()}
    assert len(table) == len(rows) == 6
    assert {k: v["status"] for k, v in rows.items()} == {
        "roof": "measured",
        "bare_ground": "no_fit",
        "tile_gone": "no_lidar_tile",
        "tile_corrupt": "tile_unreadable",
        "far_away": "outside_tile_index",
        "blank": "invalid_geometry",
    }
    assert rows["bare_ground"]["fit_failure"] == "ground_level_only"
    assert rows["roof"]["fit_failure"] is None

    # Unmeasured rows carry what the input knew, and nothing measured.
    for name in ("tile_gone", "tile_corrupt", "far_away"):
        r = rows[name]
        assert r["geometry"] is not None and r["area_m2"] == pytest.approx(100.0)
        assert r["geometry_basis"] == NOT_MEASURED and r["tilt_deg"] is None
        assert r["n_points"] is None and r["partition_id"] == -1
    assert rows["blank"]["geometry"] is None and rows["blank"]["area_m2"] is None
    assert [r["input_row"] for r in rows.values()].count(5) == 1     # still joins to the input

    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["counts"]["polygons"] == 6 and manifest["counts"]["rows"] == 6
    assert manifest["counts"]["not_measured"] == 4
    counts = manifest["aggregate_stats"]["status_counts"]
    assert set(counts) == set(STATUSES) and all(v == 1 for v in counts.values())
    assert manifest["aggregate_stats"]["fit_failure_counts"] == {"ground_level_only": 1}


def test_output_reads_back_as_one_layer_despite_null_geometry(mixed_area: dict) -> None:
    from pv_geom.io.output import read_output

    run_pipeline(**mixed_area, cfg=PVGeomConfig())
    gdf = read_output(mixed_area["output_uri"])
    assert len(gdf) == 6 and int(gdf.geometry.isna().sum()) == 1
    assert set(gdf.columns) >= set(OUTPUT_SCHEMA.names) - {"geometry"}


def test_report_funnel_counts_every_status(mixed_area: dict, tmp_path: Path) -> None:
    from pv_geom.report import build_report

    run_pipeline(**mixed_area, cfg=PVGeomConfig())
    r = build_report(mixed_area["output_uri"], tmp_path / "rep", export_dataset=False)
    import pandas as pd

    cov = pd.read_csv(r.tables_dir / "coverage.csv").set_index("step")["n"]
    assert cov["Input polygons"] == 6
    assert cov["Covered by LiDAR"] == 2
    assert cov["Plane fitted"] == 1
    status = pd.read_csv(r.tables_dir / "status.csv").set_index("status")["n"]
    assert status["no_lidar_tile"] == 1 and status["invalid_geometry"] == 1


def test_fit_failure_reasons() -> None:
    from pv_geom.pipeline.measure import LocalPoints, PolygonTask, why_no_fit

    cfg = PVGeomConfig().panel_plane
    task = PolygonTask(box(0, 0, 10, 10), "p")
    empty = gpd.GeoDataFrame(geometry=[], crs=CRS)
    rng = np.random.default_rng(0)
    ground = np.column_stack([rng.uniform(0, 10, (50, 2)), np.zeros(50)])

    def pts(n_panel: int, ground_xyz: np.ndarray) -> LocalPoints:
        panel = np.column_stack([rng.uniform(0, 10, (n_panel, 2)), np.full(n_panel, 3.0)])
        return LocalPoints(panel, ground_xyz, panel, None, empty)

    assert why_no_fit(task, pts(0, ground), cfg) == "ground_level_only"
    assert why_no_fit(task, pts(0, np.zeros((0, 3))), cfg) == "too_few_points"
    assert why_no_fit(task, pts(12, ground), cfg) == "too_few_points"
    assert why_no_fit(task, pts(200, ground), cfg) == "no_consensus"


def test_partitions_record_what_made_them(mixed_area: dict) -> None:
    run_pipeline(**mixed_area, cfg=PVGeomConfig())
    out = Path(mixed_area["output_uri"])
    for part in out.glob("part-*.parquet"):
        md = pq.read_schema(part).metadata
        assert md[META_SCHEMA_VERSION].decode() == SCHEMA_VERSION
        assert len(md[META_PLAN_FINGERPRINT]) == 64


def test_pre_0_3_output_is_upgraded_on_read(tmp_path: Path) -> None:
    """An output without `status` (schema < 0.3) gains it from whether it has a tilt."""
    import pyarrow as pa

    from pv_geom.io.output import upgrade_table

    old = pa.table({"polygon_id": ["a", "b"], "tilt_deg": [12.0, None]})
    assert upgrade_table(old).column("status").to_pylist() == ["measured", "no_fit"]


# --------------------------------------------------------------------------- #
# H1: resume only continues what it started
# --------------------------------------------------------------------------- #


@pytest.fixture
def finished_run(synth_inputs: dict[str, Path]) -> dict:
    kwargs = dict(
        polygons_uri=str(synth_inputs["polygons"]),
        tile_index_uri=str(synth_inputs["tindex"]),
        lidar_prefix=str(synth_inputs["laz_dir"]),
        output_uri=str(synth_inputs["out"]),
        name_template="tile.laz", use_dask=False,
    )
    run_pipeline(**kwargs, cfg=PVGeomConfig())
    return kwargs


def test_resume_with_the_same_setup_is_allowed(finished_run: dict) -> None:
    run_pipeline(**finished_run, cfg=PVGeomConfig(), resume=True)


def test_resume_tolerates_a_different_cluster_shape(finished_run: dict) -> None:
    cfg = PVGeomConfig()
    cfg.compute.local.n_workers = 3            # how it runs, not what it computes
    run_pipeline(**finished_run, cfg=cfg, resume=True)


def test_resume_refuses_different_measurement_settings(finished_run: dict) -> None:
    cfg = PVGeomConfig()
    cfg.panel_plane.ransac_threshold_m = 0.08
    with pytest.raises(ResumeMismatchError, match="measurement configuration") as err:
        run_pipeline(**finished_run, cfg=cfg, resume=True)
    assert "--force-resume" in err.value.remedy


def test_resume_refuses_different_polygons(finished_run: dict, tmp_path: Path) -> None:
    other = tmp_path / "other.parquet"
    gpd.GeoDataFrame({"polygon_id": ["different"]}, geometry=[box(40, 40, 50, 50)],
                     crs=CRS).to_parquet(other)
    with pytest.raises(ResumeMismatchError, match="inputs"):
        run_pipeline(**{**finished_run, "polygons_uri": str(other)},
                     cfg=PVGeomConfig(), resume=True)


def test_force_resume_overrides_the_check(finished_run: dict) -> None:
    cfg = PVGeomConfig()
    cfg.panel_plane.ransac_threshold_m = 0.08
    run_pipeline(**finished_run, cfg=cfg, resume=True, force_resume=True)


def test_resume_refuses_partitions_from_before_the_check_existed(finished_run: dict) -> None:
    part = next(Path(finished_run["output_uri"]).glob("part-0*.parquet"))
    pq.write_table(pq.read_table(part).replace_schema_metadata(None), part)   # strip metadata
    with pytest.raises(ResumeMismatchError, match="predates schema"):
        run_pipeline(**finished_run, cfg=PVGeomConfig(), resume=True)


# --------------------------------------------------------------------------- #
# H3: progress
# --------------------------------------------------------------------------- #


def test_progress_is_reported_per_group(mixed_area: dict, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="pv_geom"):
        run_pipeline(**mixed_area, cfg=PVGeomConfig())
    progress = [r.getMessage() for r in caplog.records if "groups]" in r.getMessage()]
    assert progress[0].startswith("[1/2 groups]")
    assert progress[-1].startswith("[2/2 groups] 2 rows, 1 failed")
    assert "left" in progress[-1]


# --------------------------------------------------------------------------- #
# I4: anticipated failures say what to do
# --------------------------------------------------------------------------- #


def test_errors_carry_a_remedy_and_keep_their_builtin_type() -> None:
    err = NonMetricCRSError("LiDAR is in feet", "reproject the tiles")
    assert isinstance(err, PVGeomError) and isinstance(err, NotImplementedError)
    assert err.remedy == "reproject the tiles" and "reproject" in str(err)
    assert issubclass(PolygonIdError, ValueError) and issubclass(ResumeMismatchError, RuntimeError)


def test_layer_without_crs(tmp_path: Path) -> None:
    path = tmp_path / "nocrs.parquet"
    gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)]).to_parquet(path)
    with pytest.raises(MissingCRSError) as err:
        read_polygons(path, target_crs=CRS)
    assert "set_crs" in err.value.remedy


def test_duplicate_ids(tmp_path: Path) -> None:
    path = _layer(tmp_path, [box(0, 0, 1, 1), box(2, 2, 3, 3)], polygon_id=["a", "a"])
    with pytest.raises(PolygonIdError, match="duplicate") as err:
        read_polygons(path, target_crs=CRS)
    assert "--polygon-id-col" in err.value.remedy


def test_other_anticipated_errors() -> None:
    from pv_geom.io.tile_index import resolve_tile_id_col
    from pv_geom.utils.crs import assert_metric_projected, resolve_target_crs
    from pv_geom.vintage import parse_vintage

    with pytest.raises(VintageFormatError):
        parse_vintage("last spring")
    with pytest.raises(CRSResolutionError):
        resolve_target_crs("auto", "EPSG:4326")
    with pytest.raises(NonMetricCRSError):
        assert_metric_projected("EPSG:2223")
    with pytest.raises(TileIndexError) as err:
        resolve_tile_id_col(gpd.GeoDataFrame({"zzz": [1]}, geometry=[box(0, 0, 1, 1)], crs=CRS))
    assert "--tile-id-col" in err.value.remedy


def test_missing_optional_package_names_the_extra() -> None:
    with pytest.raises(MissingDependencyError) as err:
        require("a_package_that_does_not_exist", "cloud", "reading from S3")
    assert 'pip install "pv-geom[cloud]"' in err.value.remedy
    assert isinstance(err.value, ImportError)


def test_cli_prints_the_remedy_instead_of_a_traceback(tmp_path: Path, capsys, monkeypatch) -> None:
    from pv_geom import cli

    path = tmp_path / "nocrs.parquet"
    gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)]).to_parquet(path)
    tindex = tmp_path / "t.parquet"
    gpd.GeoDataFrame({"Name": ["t"]}, geometry=[box(0, 0, 9, 9)], crs=CRS).to_parquet(tindex)
    monkeypatch.setattr("sys.argv", [
        "pv-geom", "run", "--polygons", str(path), "--tile-index", str(tindex),
        "--lidar-prefix", str(tmp_path), "--output", str(tmp_path / "out"), "--no-dask"])
    saved = logging.getLogger("pv_geom").handlers[:]
    try:
        with pytest.raises(SystemExit) as exit_info:
            cli.main()
    finally:
        logging.getLogger("pv_geom").handlers[:] = saved
    assert exit_info.value.code == 1
    printed = capsys.readouterr().out
    assert "error:" in printed and "fix:" in printed and "Traceback" not in printed
