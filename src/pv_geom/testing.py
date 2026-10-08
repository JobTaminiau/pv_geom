"""Regression helpers: reduce a run to its comparable numbers and diff two of them.

Used by the benchmark tests and by ``scripts/build_benchmark.py``. A refactor
must leave these numbers where they are; a deliberate change in results updates
the golden file in the same commit and says why.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Measured columns and the absolute tolerance within which two runs agree.
# Tolerances absorb floating-point differences between platforms and BLAS
# builds, not algorithm changes: a real change moves many rows by far more.
GOLDEN_TOLERANCES: dict[str, float] = {
    "tilt_deg": 0.05,
    "azimuth_deg": 0.25,
    "fit_rmse_m": 0.002,
    "fit_tolerance_m": 0.002,
    "tilt_unc_deg": 0.05,
    "roof_tilt_deg": 0.05,
    "height_above_roof_m": 0.005,
    "height_above_ground_m": 0.01,
    "angle_to_roof_deg": 0.1,
    "surface_area_m2": 0.01,
    # A return sitting exactly on the inlier boundary can fall either side
    # depending on the last bit of a dot product.
    "n_inliers": 2,
}

# Columns that must match exactly.
GOLDEN_EXACT: tuple[str, ...] = (
    "geometry_basis", "n_points", "n_facets",
    "roof_ref_source", "on_building", "lidar_date", "vintage_gap_days", "flags",
)


def golden_frame(output_uri: str | Path) -> pd.DataFrame:
    """The comparable slice of a run's output, one row per polygon, sorted by id."""
    from pv_geom.io.output import read_output_table

    cols = ["polygon_id", *GOLDEN_EXACT, *GOLDEN_TOLERANCES]
    df = read_output_table(output_uri).select(cols).to_pandas()
    df["flags"] = df["flags"].map(lambda f: ";".join(sorted(f)))
    df["lidar_date"] = df["lidar_date"].map(lambda d: None if d is None else str(d))
    df["on_building"] = df["on_building"].map(lambda v: None if v is None else bool(v))
    return df.sort_values("polygon_id").reset_index(drop=True)


def _same(a, b) -> bool:
    a_missing = a is None or (isinstance(a, float) and np.isnan(a)) or a is pd.NA or a is pd.NaT
    b_missing = b is None or (isinstance(b, float) and np.isnan(b)) or b is pd.NA or b is pd.NaT
    if a_missing or b_missing:
        return a_missing and b_missing
    return a == b


def compare_golden(actual: pd.DataFrame, golden: pd.DataFrame) -> list[str]:
    """Differences between a run and its golden frame, as readable lines.
    An empty list means the run reproduces the golden output."""
    problems: list[str] = []
    if list(actual["polygon_id"]) != list(golden["polygon_id"]):
        missing = sorted(set(golden["polygon_id"]) - set(actual["polygon_id"]))
        extra = sorted(set(actual["polygon_id"]) - set(golden["polygon_id"]))
        return [f"polygon ids differ: {len(missing)} missing, {len(extra)} unexpected"]

    for col in GOLDEN_EXACT:
        for pid, a, g in zip(actual["polygon_id"], actual[col], golden[col], strict=True):
            if not _same(a, g):
                problems.append(f"{pid} {col}: {a!r} != golden {g!r}")

    for col, tol in GOLDEN_TOLERANCES.items():
        a = pd.to_numeric(actual[col], errors="coerce").to_numpy(dtype=float)
        g = pd.to_numeric(golden[col], errors="coerce").to_numpy(dtype=float)
        null_mismatch = np.isnan(a) != np.isnan(g)
        diff = np.abs(a - g)
        if col == "azimuth_deg":              # 359.9 and 0.1 are 0.2 apart
            diff = np.minimum(diff, 360.0 - diff)
        bad = null_mismatch | (np.nan_to_num(diff, nan=0.0) > tol)
        for i in np.flatnonzero(bad):
            problems.append(
                f"{actual['polygon_id'][i]} {col}: {a[i]:.4f} vs golden {g[i]:.4f} (tol {tol})"
            )
    return problems


def benchmark_root() -> Path:
    """Where the real-data benchmarks live: ``$PV_GEOM_BENCHMARK_DIR`` or
    ``<repo>/data/benchmark`` (gitignored — see scripts/build_benchmark.py)."""
    default = Path(__file__).resolve().parents[2] / "data" / "benchmark"
    return Path(os.environ.get("PV_GEOM_BENCHMARK_DIR", default))


def run_benchmark(bench_dir: Path, out_dir: Path) -> Path:
    """Run the pipeline serially over one benchmark directory (``polygons.parquet``,
    ``tile_index.parquet``, the LAZ it names, optional ``footprints.parquet`` and a
    ``meta.json`` carrying the polygon vintage). Returns ``out_dir``."""
    from pv_geom.config import PVGeomConfig
    from pv_geom.pipeline.runner import run_pipeline

    meta = json.loads((bench_dir / "meta.json").read_text(encoding="utf-8"))
    cfg = PVGeomConfig()
    cfg.vintage.polygon_vintage = meta["polygon_vintage"]
    cfg.vintage.sample_tiles = 0
    footprints = bench_dir / "footprints.parquet"
    run_pipeline(
        polygons_uri=str(bench_dir / "polygons.parquet"),
        tile_index_uri=str(bench_dir / "tile_index.parquet"),
        lidar_prefix=str(bench_dir),
        footprints_uri=str(footprints) if footprints.exists() else None,
        output_uri=str(out_dir),
        cfg=cfg,
        name_template="{name}.laz",
        use_dask=False,
    )
    return out_dir
