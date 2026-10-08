"""How much do results depend on the random seed?

Runs a benchmark several times with the per-polygon seeds shifted and reports
how far each polygon's result moves between runs. A measurement that changes
with the seed is not yet a measurement; story E9 sets the target at under 0.1
degrees of tilt for 95% of polygons and no change of status or geometry basis.

    python scripts/seed_sensitivity.py                 # every benchmark present
    python scripts/seed_sensitivity.py phoenix --seeds 5
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import pv_geom.pipeline.measure as measure
from pv_geom.io.output import read_output_table
from pv_geom.sample import write_scene
from pv_geom.testing import benchmark_root, run_benchmark


def run_with_seed_offset(bench: Path, offset: int) -> pd.DataFrame:
    original = measure.seed_for_polygon
    measure.seed_for_polygon = lambda pid: original(pid) + offset
    try:
        with tempfile.TemporaryDirectory() as tmp:
            table = read_output_table(run_benchmark(bench, Path(tmp) / "out"))
            return table.drop(["geometry"]).to_pandas().set_index("polygon_id")
    finally:
        measure.seed_for_polygon = original


def sensitivity(bench: Path, n_seeds: int = 3) -> dict:
    """Spread of each polygon's result over ``n_seeds`` runs."""
    start = time.time()
    runs = [run_with_seed_offset(bench, k) for k in range(n_seeds)]
    seconds = (time.time() - start) / n_seeds
    base = runs[0]
    tilt = np.column_stack([r["tilt_deg"].to_numpy(dtype=float) for r in runs])
    az = np.column_stack([r["azimuth_deg"].to_numpy(dtype=float) for r in runs])
    fitted = ~np.isnan(tilt).any(axis=1)
    d_tilt = (tilt.max(axis=1) - tilt.min(axis=1))[fitted]
    # Azimuth spread on the circle, where every run reports one.
    has_az = fitted & ~np.isnan(az).any(axis=1)
    rel = (az[has_az] - az[has_az][:, :1] + 180.0) % 360.0 - 180.0
    d_az = rel.max(axis=1) - rel.min(axis=1)

    def differs(col: str) -> int:
        vals = np.column_stack([r[col].astype(str).to_numpy() for r in runs])
        return int((vals != vals[:, :1]).any(axis=1).sum())

    def q(x: np.ndarray, p: float) -> float:
        return float(np.quantile(x, p)) if len(x) else float("nan")

    return {
        "polygons": len(base), "fitted_in_all": int(fitted.sum()),
        "tilt_spread_median": q(d_tilt, 0.5), "tilt_spread_p95": q(d_tilt, 0.95),
        "tilt_spread_max": float(d_tilt.max()) if len(d_tilt) else float("nan"),
        "tilt_over_0.1": int((d_tilt > 0.1).sum()), "tilt_over_0.5": int((d_tilt > 0.5).sum()),
        "azimuth_spread_p95": q(d_az, 0.95),
        "azimuth_over_1": int((d_az > 1.0).sum()),
        "status_differs": differs("status"), "basis_differs": differs("geometry_basis"),
        "facets_differ": differs("n_facets"),
        "seconds_per_run": round(seconds, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", help="benchmarks (default: all present + synthetic)")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    benches: dict[str, Path] = {}
    with tempfile.TemporaryDirectory() as tmp:
        if not args.names or "synthetic" in args.names:
            benches["synthetic"] = write_scene(Path(tmp) / "synthetic")
        root = benchmark_root()
        for name in (args.names or sorted(p.name for p in root.iterdir() if p.is_dir())
                     if root.exists() else []):
            if name != "synthetic" and (root / name / "polygons.parquet").exists():
                benches[name] = root / name
        rows = {name: sensitivity(path, args.seeds) for name, path in benches.items()}
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()
