"""Phoenix end-to-end integration test. M7.

Runs ``pv-geom run`` against a small bbox of real Phoenix data on the local
machine using a ``LocalCluster``. Gated on:

  1. ``RUN_INTEGRATION=1`` env var — opt-in, never fires in CI by default.
  2. Presence of cached prerequisite files on disk (atlas parquet, FEMA AZ
     geoparquet, USGS LiDAR tile, USGS tile-index shapefile).

This is NOT a byte-for-byte reference comparison (deferred to M8); it asserts
schema correctness, sensible aggregate stats, and physical plausibility of
recovered tilts/azimuths/RMSEs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from pv_geom.config import PVGeomConfig
from pv_geom.pipeline.runner import run_pipeline
from pv_geom.schema import OUTPUT_SCHEMA

pytestmark = pytest.mark.integration

if not os.environ.get("RUN_INTEGRATION"):
    pytest.skip("set RUN_INTEGRATION=1 to run the Phoenix smoke test",
                allow_module_level=True)

# ---- Prerequisite file paths (on the dev machine) -------------------------
ATLAS_PARQUET = Path(r"C:\Users\job_t\code\free\pv_sam3\artifacts\atlas\latest.parquet")
FEMA_AZ_PARQUET = Path(r"C:\Users\job_t\AppData\Local\Temp\pv_geom_spike_cache\az.geoparquet")
TILE_INDEX_SHP = Path(r"C:\Users\job_t\AppData\Local\Temp\tileindex\USGS_AZ_MaricopaPinal_1_2020_TileIndex.shp")
LAZ_DIR = Path(r"C:\Users\job_t\AppData\Local\Temp\pv_geom_spike_cache")
LIDAR_PREFIX = "s3://asu-nsf-phoenix/data/lidar_data"
LAZ_NAME_TEMPLATE = "USGS_LPC_AZ_MaricopaPinal_2020_B20_{name}.laz"

PREREQS = {
    "atlas": ATLAS_PARQUET,
    "fema": FEMA_AZ_PARQUET,
    "tile_index": TILE_INDEX_SHP,
}

missing = [k for k, p in PREREQS.items() if not p.exists()]
if missing:
    pytest.skip(
        f"missing prerequisite files: {missing}; run scripts/spike_roof.py first "
        f"to populate the cache",
        allow_module_level=True,
    )


def test_phoenix_bbox_end_to_end(tmp_path: Path) -> None:
    """Run the pipeline on a 900x900 m east-valley bbox and verify outputs."""
    cfg = PVGeomConfig.from_yaml(Path(__file__).resolve().parents[2] / "configs" / "phoenix.yaml")
    cfg.compute.backend = "local"
    cfg.compute.local.n_workers = 2
    cfg.compute.local.threads_per_worker = 2

    out = tmp_path / "out"
    manifest_path = run_pipeline(
        polygons_uri=str(ATLAS_PARQUET),
        tile_index_uri=str(TILE_INDEX_SHP),
        lidar_prefix=LIDAR_PREFIX,
        footprints_uri=str(FEMA_AZ_PARQUET),
        output_uri=str(out),
        cfg=cfg,
        name_template=LAZ_NAME_TEMPLATE,
        tile_id_col="Name",
        max_polygons=20,
        bbox=(432000, 3719000, 432900, 3719900),  # subset of tile w0432n3719
        use_dask=True,
    )
    assert manifest_path.exists()

    # Read all partitions
    parts = sorted(out.glob("part-*.parquet"))
    assert len(parts) >= 1
    table = pq.read_table(parts[0])
    for p in parts[1:]:
        table = pq.concat_tables([table, pq.read_table(p)])

    # Schema matches the canonical OUTPUT_SCHEMA
    assert set(table.schema.names) == set(OUTPUT_SCHEMA.names)

    df = table.to_pandas()
    assert len(df) > 0
    assert len(df) <= 20

    # Physical-plausibility checks on successfully fit panels.
    fit_ok = df.dropna(subset=["panel_tilt_deg"])
    assert len(fit_ok) > 0, "expected at least one successful panel fit"
    # AZ residential PV: tilts <= ~35 deg, azimuth in southern half (90-270 deg)
    assert (fit_ok["panel_tilt_deg"] <= 45.0).mean() > 0.9
    south = (fit_ok["panel_azimuth_deg"].between(90, 270)).mean()
    assert south > 0.8, f"expected most azimuths in southern half, got {south:.1%}"
    # RMSE should mostly be sub-10 cm
    assert (fit_ok["panel_rmse_m"] <= 0.10).mean() > 0.9

    # Manifest sanity
    manifest = json.loads(manifest_path.read_text())
    assert manifest["counts"]["succeeded"] == len(df)
    assert manifest["pkg_version"]
    assert manifest["config_hash"] == cfg.hash()
    assert "w0432n3719" in manifest["tiles_touched"]
    assert manifest["aggregate_stats"]["mounting_type_counts"]
