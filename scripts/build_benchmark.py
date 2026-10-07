"""Build the real-data regression benchmark used by ``tests/benchmark``.

The benchmark is a small frozen slice of a real study area: a sample of PV
polygons, the LiDAR returns around them (one clipped LAZ), a one-tile index and,
optionally, the building footprints they touch. Running the pipeline over it
must keep producing the same numbers; ``tests/benchmark/test_regression.py``
compares against ``golden.parquet`` written here.

The data is NOT committed (the repository is public and the polygon layers are
unpublished research products). It lives under ``data/benchmark/<name>/`` —
gitignored — or wherever ``PV_GEOM_BENCHMARK_DIR`` points, and is rebuilt from
the source tile with this script::

    uv run python scripts/build_benchmark.py phoenix \\
        --polygons pv_attributes.parquet --tile w0366n3721.laz \\
        --polygon-vintage 2024-04-01 --footprints az.geoparquet -n 100

``--golden-only`` re-runs the pipeline over an existing benchmark and rewrites
the golden file: do that only when a change in results is intended, and say so
in the commit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
from shapely.geometry import box

from pv_geom.testing import benchmark_root


def build(name: str, polygons: str, tile: str, n: int, pad_m: float, seed: int,
          polygon_vintage: str, footprints: str | None) -> Path:
    out = benchmark_root() / name
    out.mkdir(parents=True, exist_ok=True)

    with laspy.open(tile) as src:
        header = src.header
        crs = header.parse_crs()
        las = src.read()
    horiz = crs.sub_crs_list[0] if crs.is_compound else crs
    tile_box = box(header.mins[0], header.mins[1], header.maxs[0], header.maxs[1])

    polys = gpd.read_file(polygons) if not polygons.endswith(".parquet") \
        else gpd.read_parquet(polygons)
    polys = polys.to_crs(horiz)
    # Keep polygons comfortably inside the tile so no neighbour tile is needed.
    polys = polys[polys.geometry.within(tile_box.buffer(-pad_m - 1.0))]
    polys = polys.sample(n=min(n, len(polys)), random_state=seed).sort_index()
    id_col = next((c for c in ("polygon_id", "detection_id") if c in polys.columns), None)
    keep = polys[[id_col, "geometry"]] if id_col else polys[["geometry"]]
    keep = keep.reset_index(drop=True)
    keep.to_parquet(out / "polygons.parquet")

    # Returns within pad_m of any sampled polygon's bounds.
    from pv_geom.pipeline.pointpool import NearMask

    near = NearMask(list(keep.geometry), pad_m=pad_m)
    mask = near(np.asarray(las.x), np.asarray(las.y))
    clipped = laspy.LasData(header)
    clipped.points = las.points[mask]
    clipped.write(str(out / "bench.laz"))

    gpd.GeoDataFrame({"Name": ["bench"]}, geometry=[tile_box], crs=horiz) \
        .to_parquet(out / "tile_index.parquet")

    if footprints:
        from pv_geom.io.footprints import read_footprints

        x0, y0, x1, y1 = tile_box.bounds
        fp = read_footprints(footprints, target_crs=horiz).cx[x0:x1, y0:y1]
        hit = fp.sindex.query(keep.geometry.buffer(pad_m), predicate="intersects")[1]
        cols = [c for c in ("build_id", "building_id") if c in fp.columns] + ["geometry"]
        fp.iloc[np.unique(hit)][cols].reset_index(drop=True).to_parquet(out / "footprints.parquet")

    (out / "meta.json").write_text(json.dumps({
        "name": name, "source_tile": Path(tile).name, "n_polygons": len(keep),
        "n_points": int(mask.sum()), "pad_m": pad_m, "seed": seed,
        "polygon_vintage": polygon_vintage,
    }, indent=2), encoding="utf-8")
    return out


def write_golden(bench_dir: Path) -> None:
    import tempfile

    from pv_geom.testing import golden_frame, run_benchmark

    with tempfile.TemporaryDirectory() as tmp:
        df = golden_frame(run_benchmark(bench_dir, Path(tmp) / "out"))
    df.to_parquet(bench_dir / "golden.parquet", index=False)
    print(f"{bench_dir.name}: golden written, {len(df)} rows, "
          f"{int(df['panel_tilt_deg'].notna().sum())} fitted")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("name")
    ap.add_argument("--polygons")
    ap.add_argument("--tile")
    ap.add_argument("--footprints")
    ap.add_argument("--polygon-vintage", default="2024")
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--pad-m", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--golden-only", action="store_true")
    args = ap.parse_args()

    bench_dir = benchmark_root() / args.name
    if not args.golden_only:
        if not (args.polygons and args.tile):
            ap.error("--polygons and --tile are required unless --golden-only")
        bench_dir = build(args.name, args.polygons, args.tile, args.n, args.pad_m,
                          args.seed, args.polygon_vintage, args.footprints)
    write_golden(bench_dir)


if __name__ == "__main__":
    main()
