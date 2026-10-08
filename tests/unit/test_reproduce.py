"""Story G5: a published dataset can be reproduced, and shown to have been."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

import pv_geom
from pv_geom import reproduce as rp
from pv_geom.cli import app
from pv_geom.io.output import read_manifest, read_output_table
from pv_geom.sample import write_demo


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> pv_geom.RunResult:
    return pv_geom.run(write_demo(tmp_path_factory.mktemp("demo")), use_dask=False)


def test_manifest_records_content_inputs_and_environment(demo) -> None:
    record = read_manifest(demo.output)["reproducibility"]
    assert len(record["content_hash"]) == 64
    assert len(record["inputs"]["polygons"]["sha256"]) == 64
    assert record["inputs"]["lidar"]["tiles"] == record["inputs"]["lidar"]["tiles_sized"] == 1
    env = record["environment"]
    assert env["python"] and env["libraries"]["numpy"] and env["libraries"]["PROJ"]


def test_an_untouched_output_verifies(demo) -> None:
    result = rp.verify(demo.output)
    assert result["intact"] and result["recorded"] == result["actual"]
    res = CliRunner().invoke(app, ["verify", str(demo.output)])
    assert res.exit_code == 0 and "intact" in res.output


def test_a_second_run_is_identical_and_the_run_id_does_not_matter(demo, tmp_path: Path) -> None:
    again = pv_geom.run(write_demo(tmp_path), use_dask=False)
    a, b = read_output_table(demo.output), read_output_table(again.output)
    assert a.column("run_id")[0] != b.column("run_id")[0]
    comparison = rp.compare_outputs(demo.output, again.output)
    assert comparison.identical, comparison.summary()


def test_row_order_and_partitioning_do_not_change_the_hash(demo) -> None:
    table = read_output_table(demo.output)
    shuffled = table.take(list(range(len(table) - 1, -1, -1)))
    assert rp.content_digest(shuffled) == rp.content_digest(table)


def test_a_changed_value_is_found_and_located(demo, tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(demo.output, copy)
    part = sorted(copy.glob("part-[0-9]*.parquet"))[0]
    t = pq.read_table(part)
    tilt = t.column("tilt_deg").to_pylist()
    row = next(i for i, v in enumerate(tilt) if v is not None)
    tilt[row] += 0.001
    import pyarrow as pa

    t = t.set_column(t.column_names.index("tilt_deg"), "tilt_deg", pa.array(tilt, pa.float32()))
    pq.write_table(t, part)

    assert not rp.verify(copy)["intact"]
    res = CliRunner().invoke(app, ["verify", str(copy)])
    assert res.exit_code == 1 and "modified" in res.output
    comparison = rp.compare_outputs(demo.output, copy)
    assert not comparison.identical
    assert list(comparison.columns) == ["tilt_deg"]
    assert comparison.columns["tilt_deg"]["rows"] == 1
    assert comparison.columns["tilt_deg"]["max_abs_diff"] == pytest.approx(0.001, rel=0.05)
    res = CliRunner().invoke(app, ["verify", str(demo.output), "--against", str(copy)])
    assert res.exit_code == 1 and "tilt_deg" in res.output


def test_reproduce_reruns_from_the_manifest(demo, tmp_path: Path) -> None:
    comparison, notes = rp.reproduce(demo.output, tmp_path / "rerun")
    assert comparison.identical, comparison.summary()
    assert notes == []
    res = CliRunner().invoke(app, ["reproduce", str(demo.output), "--out",
                                   str(tmp_path / "rerun2")])
    assert res.exit_code == 0 and "identical" in res.output


def test_reproduce_says_when_the_inputs_are_not_the_same(tmp_path: Path) -> None:
    import geopandas as gpd

    run = pv_geom.run(write_demo(tmp_path / "demo"), use_dask=False)
    polygons = Path(read_manifest(run.output)["inputs"]["polygons"])
    layer = gpd.read_parquet(polygons)
    layer.iloc[:-1].to_parquet(polygons)                     # one polygon fewer
    comparison, notes = rp.reproduce(run.output, tmp_path / "rerun")
    assert not comparison.identical and comparison.only_in_a == 1
    assert any("polygons" in n for n in notes)


def test_environment_differences_are_listed() -> None:
    now = rp.environment()
    then = json.loads(json.dumps(now))
    then["python"] = "3.10.0"
    then["libraries"]["numpy"] = "1.0"
    lines = rp.environment_differences(then, now)
    assert any(line.startswith("python") for line in lines)
    assert any(line.startswith("numpy") for line in lines)
    assert rp.environment_differences(now, now) == []
