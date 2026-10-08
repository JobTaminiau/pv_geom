"""Regression benchmarks: the pipeline must keep producing the same numbers.

Two tiers:

* **Synthetic** (always runs, including in CI): a generated study area with
  known geometry. Its golden output is committed next to this file.
* **Real** (runs when the data is present): frozen slices of Phoenix and
  Delaware under ``data/benchmark/`` — not committed, because the polygon layers
  are unpublished; rebuild with ``scripts/build_benchmark.py``.

When a change in results is intended, regenerate the golden files
(``PV_GEOM_UPDATE_GOLDEN=1 pytest tests/benchmark`` for the synthetic tier,
``scripts/build_benchmark.py <name> --golden-only`` for the real one) and say
why in the commit message.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from pv_geom.testing import benchmark_root, compare_golden, golden_frame, run_benchmark

from .synthetic import ARRAYS, write_scene

SYNTHETIC_GOLDEN = Path(__file__).with_name("synthetic_golden.json")


def _report(problems: list[str]) -> str:
    head = "\n".join(problems[:25])
    more = f"\n... and {len(problems) - 25} more" if len(problems) > 25 else ""
    return f"{len(problems)} differences from golden output:\n{head}{more}"


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    root = tmp_path_factory.mktemp("synthetic")
    return golden_frame(run_benchmark(write_scene(root / "bench"), root / "out"))


def test_synthetic_matches_golden(synthetic_run: pd.DataFrame) -> None:
    if os.environ.get("PV_GEOM_UPDATE_GOLDEN"):
        SYNTHETIC_GOLDEN.write_text(
            synthetic_run.to_json(orient="records", indent=1), encoding="utf-8")
    golden = pd.DataFrame(json.loads(SYNTHETIC_GOLDEN.read_text(encoding="utf-8")))
    problems = compare_golden(synthetic_run, golden)
    assert not problems, _report(problems)


def test_synthetic_recovers_the_truth(synthetic_run: pd.DataFrame) -> None:
    """Independent of the golden file: fitted tilt and azimuth match what the
    scene was built with."""
    by_row = dict(zip(synthetic_run["polygon_id"], synthetic_run.to_dict("records"), strict=True))
    for i, a in enumerate(ARRAYS):
        row = by_row[f"poly_{i:07d}"]
        if a.kind == "empty":
            assert row["geometry_basis"] == "no_fit", a.name
            continue
        if a.kind == "sparse":
            # Twelve returns still define a plane; the row is fitted but flagged.
            assert "low_density" in row["flags"], a.name
            continue
        tol = 1.5 if a.noise > 0.05 else 0.5
        assert row["panel_tilt_deg"] == pytest.approx(a.panel_tilt, abs=tol), a.name
        if a.panel_tilt >= 5:
            d = abs(row["panel_azimuth_deg"] - a.panel_az) % 360
            assert min(d, 360 - d) < (6.0 if a.noise > 0.05 else 2.0), a.name
        if a.kind == "bare_roof":
            assert row["geometry_basis"] == "surface_unresolved", a.name
        elif a.kind == "roof":
            assert row["geometry_basis"] == "panel_confirmed", a.name


@pytest.mark.parametrize("name", ["phoenix", "delaware"])
def test_real_benchmark_matches_golden(name: str, tmp_path: Path) -> None:
    bench = benchmark_root() / name
    if not (bench / "golden.parquet").exists():
        pytest.skip(f"no {name} benchmark under {benchmark_root()}; "
                    f"build it with scripts/build_benchmark.py")
    actual = golden_frame(run_benchmark(bench, tmp_path / "out"))
    problems = compare_golden(actual, pd.read_parquet(bench / "golden.parquet"))
    assert not problems, _report(problems)
