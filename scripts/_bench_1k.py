"""One-shot 1000-polygon benchmark per HANDOFF.md Step 3.

Standalone script (not a uv inline header) — run with `uv run`. Writes nothing
permanent; partition output goes under ./out_bench_1k/.
"""

from __future__ import annotations

import time
from pathlib import Path

from pv_geom.config import PVGeomConfig
from pv_geom.pipeline.runner import run_pipeline


def main() -> None:
    cfg = PVGeomConfig.from_yaml("configs/phoenix.yaml")
    cfg.compute.backend = "local"
    cfg.compute.local.n_workers = 2
    cfg.compute.local.threads_per_worker = 2

    t0 = time.perf_counter()
    manifest = run_pipeline(
        polygons_uri=r"C:\Users\job_t\code\free\pv_sam3\artifacts\atlas\latest.parquet",
        tile_index_uri=r"C:\Users\job_t\AppData\Local\Temp\tileindex\USGS_AZ_MaricopaPinal_1_2020_TileIndex.shp",
        lidar_prefix="s3://asu-nsf-phoenix/data/lidar_data",
        footprints_uri=r"C:\Users\job_t\AppData\Local\Temp\pv_geom_spike_cache\az.geoparquet",
        output_uri="./out_bench_1k",
        cfg=cfg,
        name_template="USGS_LPC_AZ_MaricopaPinal_2020_B20_{name}.laz",
        bbox=(442000, 3681000, 443000, 3682000),
        max_polygons=1000,
        use_dask=True,
    )
    dt = time.perf_counter() - t0
    print(f"\nTOTAL: {dt:.1f} s ({dt/60:.1f} min)")
    print(f"manifest: {manifest}")

    parts = sorted(Path("out_bench_1k").glob("part-*.parquet"))
    print(f"partitions: {len(parts)}")


if __name__ == "__main__":
    main()
