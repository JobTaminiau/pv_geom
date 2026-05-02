"""20-polygon Coiled smoke test.

Mirrors the M7 integration-test bbox (east-valley `w0432n3719`). Uses a
small ad-hoc cluster shape that's enough for one tile group; the
`phoenix.yaml` 40-worker default would be wasteful for 20 polygons.
"""

from __future__ import annotations

import time
from pathlib import Path

from pv_geom.config import PVGeomConfig
from pv_geom.pipeline.runner import run_pipeline


def main() -> None:
    cfg = PVGeomConfig.from_yaml("configs/phoenix.yaml")
    # Trim the cluster: 20 polygons -> ~1 tile group, no need for 40 workers.
    cfg.compute.coiled.n_workers = 2
    cfg.compute.coiled.worker_cpu = 2
    cfg.compute.coiled.worker_memory = "8GiB"
    cfg.compute.coiled.name = "pv-geom-smoke"

    t0 = time.perf_counter()
    manifest = run_pipeline(
        polygons_uri=r"C:\Users\job_t\code\free\pv_sam3\artifacts\atlas\latest.parquet",
        tile_index_uri=r"C:\Users\job_t\AppData\Local\Temp\tileindex\USGS_AZ_MaricopaPinal_1_2020_TileIndex.shp",
        lidar_prefix="s3://asu-nsf-phoenix/data/lidar_data",
        footprints_uri=r"C:\Users\job_t\AppData\Local\Temp\pv_geom_spike_cache\az.geoparquet",
        output_uri="./out_coiled_smoke",
        cfg=cfg,
        name_template="USGS_LPC_AZ_MaricopaPinal_2020_B20_{name}.laz",
        bbox=(432000, 3719000, 432900, 3719900),  # same as M7 integration test
        max_polygons=20,
        use_dask=True,
    )
    dt = time.perf_counter() - t0
    print(f"\nTOTAL: {dt:.1f} s")
    print(f"manifest: {manifest}")

    parts = sorted(Path("out_coiled_smoke").glob("part-*.parquet"))
    print(f"partitions: {len(parts)}")


if __name__ == "__main__":
    main()
