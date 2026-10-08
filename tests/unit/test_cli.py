"""The command line, end to end, through Typer's test runner."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pv_geom import __version__
from pv_geom.cli import app

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
runner = CliRunner()


@pytest.fixture(autouse=True)
def _restore_logging():
    """The CLI configures logging; put it back so other tests are unaffected."""
    pkg = logging.getLogger("pv_geom")
    saved = (pkg.handlers[:], pkg.level, pkg.propagate)
    yield
    pkg.handlers[:], pkg.level, pkg.propagate = saved[0], saved[1], saved[2]


def _run_args(inputs: dict[str, Path], *extra: str) -> list[str]:
    return [
        "run",
        "--polygons", str(inputs["polygons"]),
        "--tile-index", str(inputs["tindex"]),
        "--lidar-prefix", str(inputs["laz_dir"]),
        "--name-template", "tile.laz",
        "--output", str(inputs["out"]),
        "--no-dask",
        *extra,
    ]


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0 and __version__ in result.output


@pytest.mark.parametrize("name", ["default.yaml", "phoenix.yaml", "delaware.yaml"])
def test_validate_config(name: str) -> None:
    result = runner.invoke(app, ["validate-config", str(CONFIGS / name)])
    assert result.exit_code == 0 and "config_hash" in result.output


def test_run_writes_dataset_log_and_report(synth_inputs: dict[str, Path]) -> None:
    result = runner.invoke(app, _run_args(
        synth_inputs, "--footprints", str(synth_inputs["footprints"]),
        "--polygon-vintage", "2024-04", "--lidar-date", "2023-03-31"))
    assert result.exit_code == 0, result.output
    out = synth_inputs["out"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["vintage"]["polygon_vintage"] == "2024-04-30"     # month -> its last day
    assert manifest["vintage"]["lidar_date_declared"] == "2023-03-31"
    assert manifest["counts"]["succeeded"] == 1
    assert (out / "report" / "report.html").exists()
    records = [json.loads(line) for line in (out / "logs" / "run.jsonl").read_text().splitlines()]
    assert any("tile groups" in r["msg"] for r in records)
    assert "VINTAGE GAP" in result.output                              # shown on the console


def test_dry_run_checks_vintage_without_computing(synth_inputs: dict[str, Path]) -> None:
    result = runner.invoke(app, _run_args(synth_inputs, "--polygon-vintage", "2024", "--dry-run"))
    assert result.exit_code == 0, result.output
    out = synth_inputs["out"]
    assert not list(out.glob("part-*.parquet"))
    assert json.loads((out / "manifest.json").read_text())["aggregate_stats"] == {"dry_run": True}


def test_quiet_suppresses_progress(synth_inputs: dict[str, Path]) -> None:
    result = runner.invoke(app, ["--quiet", *_run_args(synth_inputs, "--no-report")])
    assert result.exit_code == 0, result.output
    assert "tile groups" not in result.output


def test_describe_and_report_on_a_finished_run(synth_inputs: dict[str, Path],
                                               tmp_path: Path) -> None:
    assert runner.invoke(app, _run_args(synth_inputs, "--no-report")).exit_code == 0
    described = runner.invoke(app, ["describe-output", str(synth_inputs["out"])])
    assert described.exit_code == 0 and "fitted" in described.output
    reported = runner.invoke(app, ["report", str(synth_inputs["out"]),
                                   "--out", str(tmp_path / "r"),
                                   "--area-name", "Testville", "--no-export"])
    assert reported.exit_code == 0, reported.output
    assert "Testville" in (tmp_path / "r" / "report.md").read_text(encoding="utf-8")
    assert not (tmp_path / "r" / "dataset").exists()


def test_inspect_tile(synth_inputs: dict[str, Path]) -> None:
    result = runner.invoke(app, ["inspect-tile", str(synth_inputs["laz_dir"] / "tile.laz"),
                                 "--json"])
    assert result.exit_code == 0, result.output
    info = json.loads(result.output)
    assert info["has_building_class_6"] and info["has_ground_class_2"]
    assert info["n_points"] > 0 and info["classes"]["6"] > 0


def test_missing_inputs_are_reported_with_where_to_give_them() -> None:
    """Inputs may come from options or a config, so none is a required option;
    what is missing after both is an InputError naming option and config key."""
    from pv_geom.errors import InputError

    result = runner.invoke(app, ["run", "--polygons", "p.parquet"])
    assert result.exit_code == 1
    assert isinstance(result.exception, InputError)
    assert "inputs.lidar_prefix" in result.exception.remedy
