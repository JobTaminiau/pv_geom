"""10k-polygon Coiled stress test — intermediate between the 1k benchmark
and a full atlas run. 3 km x 3 km bbox around the `w0442n368x` utility
cluster, up to 9 indexed tiles. Larger cluster (16 workers x 2 CPU) to
exercise multi-worker parallelism for the first time.
"""

from __future__ import annotations

import time
from pathlib import Path

from pv_geom.config import PVGeomConfig
from pv_geom.pipeline.runner import run_pipeline


def main() -> None:
    cfg = PVGeomConfig.from_yaml("configs/phoenix.yaml")
    # 8 tile groups in this bbox; right-size workers to that. Memory bumped
    # to 16 GiB because the densest tile (w0442n3681, ~3000 polygons / 50M+
    # points) blew through 8 GiB on the first attempt.
    cfg.compute.coiled.n_workers = 8
    cfg.compute.coiled.worker_cpu = 2
    cfg.compute.coiled.worker_memory = "16GiB"
    cfg.compute.coiled.name = "pv-geom-bench-10k"

    t0 = time.perf_counter()
    manifest = run_pipeline(
        polygons_uri=r"C:\Users\job_t\code\free\pv_sam3\artifacts\atlas\latest.parquet",
        tile_index_uri=r"C:\Users\job_t\AppData\Local\Temp\tileindex\USGS_AZ_MaricopaPinal_1_2020_TileIndex.shp",
        lidar_prefix="s3://free-research-data-raw/US/arizona/top-level/lidar/lidar_data",
        footprints_uri=r"C:\Users\job_t\AppData\Local\Temp\pv_geom_spike_cache\az.geoparquet",
        output_uri="./out_coiled_bench_10k",
        cfg=cfg,
        name_template="USGS_LPC_AZ_MaricopaPinal_2020_B20_{name}.laz",
        bbox=(441000, 3680000, 444000, 3683000),
        max_polygons=10000,
        use_dask=True,
        resume=True,                  # crash-recovery: re-run picks up where it left off
    )
    dt = time.perf_counter() - t0
    print(f"\nTOTAL: {dt:.1f} s ({dt/60:.1f} min)")
    print(f"manifest: {manifest}")

    parts = sorted(Path("out_coiled_bench_10k").glob("part-*.parquet"))
    print(f"partitions: {len(parts)}")


if __name__ == "__main__":
    main()
