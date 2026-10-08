"""Story E9: a polygon's result must not depend on the random seed.

Runs each benchmark with the per-polygon seeds shifted and requires the fitted
plane to come out the same. The roof reference is allowed a small residual
(it is fitted to far larger point sets by random search), which shows up only
as a change of `geometry_basis` for a polygon sitting on the standoff threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pv_geom.sample import write_scene
from pv_geom.testing import benchmark_root

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from seed_sensitivity import sensitivity


def _check(result: dict, basis_allowance: float) -> None:
    assert result["tilt_spread_p95"] < 0.01, result
    assert result["tilt_over_0.1"] == 0, result
    assert result["azimuth_over_1"] == 0, result
    assert result["status_differs"] == 0 and result["facets_differ"] == 0, result
    assert result["basis_differs"] <= basis_allowance * result["polygons"], result


def test_synthetic_results_do_not_depend_on_the_seed(tmp_path: Path) -> None:
    _check(sensitivity(write_scene(tmp_path), n_seeds=3), basis_allowance=0.0)


@pytest.mark.parametrize("name", ["phoenix", "delaware"])
def test_real_results_do_not_depend_on_the_seed(name: str) -> None:
    bench = benchmark_root() / name
    if not (bench / "polygons.parquet").exists():
        pytest.skip(f"benchmark {name} not present (it holds unpublished polygons)")
    _check(sensitivity(bench, n_seeds=3), basis_allowance=0.04)
