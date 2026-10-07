"""The distributed execution path, on a real one-worker local cluster."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pyarrow.parquet as pq
import pytest
from shapely.geometry import box

from pv_geom.config import PVGeomConfig
from pv_geom.pipeline.executor import cluster_for, execute
from pv_geom.pipeline.runner import run_pipeline

from .scenes import write_synthetic_laz


def _local_cfg() -> PVGeomConfig:
    cfg = PVGeomConfig()
    cfg.compute.backend = "local"
    cfg.compute.local.n_workers = 1
    cfg.compute.local.threads_per_worker = 1
    return cfg


@pytest.mark.slow
def test_distributed_run_writes_partitions_and_survives_a_bad_tile(tmp_path: Path) -> None:
    """Two tile groups on a cluster: one good, one whose tile is corrupt. The
    good partition is written, the failure is recorded, and the run completes."""
    write_synthetic_laz(tmp_path / "good.laz")
    (tmp_path / "bad.laz").write_bytes(b"not a laz file")
    gpd.GeoDataFrame(
        {"polygon_id": ["ok", "broken"]},
        geometry=[box(40.0, 40.0, 50.0, 50.0), box(140.0, 40.0, 150.0, 50.0)],
        crs="EPSG:6341",
    ).to_parquet(tmp_path / "polys.parquet")
    gpd.GeoDataFrame(
        {"Name": ["good", "bad"]},
        geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)], crs="EPSG:6341",
    ).to_parquet(tmp_path / "tindex.parquet")

    out = tmp_path / "out"
    manifest_path = run_pipeline(
        polygons_uri=str(tmp_path / "polys.parquet"),
        tile_index_uri=str(tmp_path / "tindex.parquet"),
        lidar_prefix=str(tmp_path), output_uri=str(out),
        cfg=_local_cfg(), use_dask=True,
    )
    parts = sorted(out.glob("part-[0-9]*.parquet"))
    assert len(parts) == 1
    assert pq.read_table(parts[0]).column("polygon_id").to_pylist() == ["ok"]
    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest["counts"]["failed_groups"] == 1
    assert manifest["counts"]["succeeded"] == 1
    assert manifest["cluster_spec"] == {"backend": "local", "use_dask": True}
    assert len(manifest["aggregate_stats"]["task_errors"]) == 1


def test_execute_with_nothing_pending_starts_no_cluster() -> None:
    def _fail(*_args):  # pragma: no cover - must never be called
        raise AssertionError("no task should run")

    execute([], _fail, _fail, _fail, _local_cfg(), use_dask=True)


def test_unknown_backend_is_rejected() -> None:
    cfg = _local_cfg()
    object.__setattr__(cfg.compute, "backend", "slurm")
    with pytest.raises(ValueError, match="unknown compute backend"), cluster_for(cfg):
        pass
