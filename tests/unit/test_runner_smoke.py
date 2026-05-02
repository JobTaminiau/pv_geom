"""End-to-end smoke test for the M6 runner on fully synthetic inputs.

Synthesizes a tile (with a planar surface representing a PV array on a roof
plus surrounding roof returns + ground), a single PV polygon over the panel,
a building footprint covering both, and a tile index. Runs ``run_pipeline``
and verifies the output schema + the recovered tilt/azimuth + mounting label.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pyarrow.parquet as pq
import pytest
from shapely.geometry import box

from pv_geom.config import PVGeomConfig
from pv_geom.pipeline.runner import run_pipeline
from pv_geom.schema import OUTPUT_SCHEMA


def _write_synthetic_laz(
    path: Path,
    *,
    tile_extent: tuple[float, float, float, float] = (0.0, 0.0, 100.0, 100.0),
    panel_extent: tuple[float, float, float, float] = (40.0, 40.0, 50.0, 50.0),
    panel_tilt_deg: float = 20.0,
    panel_az_deg: float = 180.0,
    roof_z0: float = 5.0,
    panel_z0: float = 5.5,
    seed: int = 0,
) -> None:
    """Synthesize a LAZ tile with class-2 ground + class-6 roof + class-6 panel.
    The panel is a planar surface tilted as specified, parked above the roof."""
    rng = np.random.default_rng(seed)

    def _plane_z(xy, tilt_deg, az_deg, z_at_centroid, centroid_xy):
        """Plane z(x,y) such that z(centroid_xy) == z_at_centroid."""
        t = np.radians(tilt_deg)
        a = np.radians(az_deg)
        nx, ny, nz = np.sin(t) * np.sin(a), np.sin(t) * np.cos(a), np.cos(t)
        cx, cy = centroid_xy
        return z_at_centroid - (nx * (xy[:, 0] - cx) + ny * (xy[:, 1] - cy)) / nz

    # Ground (class 2): scattered points across the tile, all at z=0
    n_ground = 5_000
    g_xy = rng.uniform(
        [tile_extent[0], tile_extent[1]],
        [tile_extent[2], tile_extent[3]],
        size=(n_ground, 2),
    )
    g_z = np.zeros(n_ground)
    ground = np.column_stack([g_xy, g_z, np.full(n_ground, 2)])

    # Roof (class 6) inside a 30x30m building; planar, low-tilt south
    bx0, by0, bx1, by1 = 35.0, 35.0, 65.0, 65.0
    bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
    n_roof = 4_000
    r_xy = rng.uniform([bx0, by0], [bx1, by1], size=(n_roof, 2))
    r_z = _plane_z(r_xy, 5.0, 180.0, roof_z0, (bcx, bcy)) + rng.normal(0, 0.01, n_roof)
    roof = np.column_stack([r_xy, r_z, np.full(n_roof, 6)])

    # Panel (class 6) above the roof; tilted at panel_tilt_deg facing panel_az_deg
    n_panel = 800
    pcx, pcy = (panel_extent[0] + panel_extent[2]) / 2, (panel_extent[1] + panel_extent[3]) / 2
    p_xy = rng.uniform(
        [panel_extent[0], panel_extent[1]],
        [panel_extent[2], panel_extent[3]],
        size=(n_panel, 2),
    )
    p_z = _plane_z(p_xy, panel_tilt_deg, panel_az_deg, panel_z0, (pcx, pcy)) + rng.normal(0, 0.01, n_panel)
    panel = np.column_stack([p_xy, p_z, np.full(n_panel, 6)])

    pts = np.concatenate([ground, roof, panel])

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x = pts[:, 0]
    las.y = pts[:, 1]
    las.z = pts[:, 2]
    las.classification = pts[:, 3].astype(np.uint8)
    las.write(str(path))


@pytest.fixture
def synth_inputs(tmp_path: Path) -> dict[str, Path]:
    """Build an end-to-end synthetic dataset under tmp_path."""
    laz = tmp_path / "tile.laz"
    _write_synthetic_laz(laz)

    polygons = gpd.GeoDataFrame(
        {"polygon_id": ["panel_1"]},
        geometry=[box(40.0, 40.0, 50.0, 50.0)],
        crs="EPSG:6341",
    )
    polygons_path = tmp_path / "polys.parquet"
    polygons.to_parquet(polygons_path)

    footprints = gpd.GeoDataFrame(
        {"building_id": ["b1"]},
        geometry=[box(35.0, 35.0, 65.0, 65.0)],
        crs="EPSG:6341",
    )
    footprints_path = tmp_path / "fp.parquet"
    footprints.to_parquet(footprints_path)

    tindex = gpd.GeoDataFrame(
        {"Name": ["t1"]},
        geometry=[box(0.0, 0.0, 100.0, 100.0)],
        crs="EPSG:6341",
    )
    tindex_path = tmp_path / "tindex.parquet"
    tindex.to_parquet(tindex_path)

    out_dir = tmp_path / "out"
    return {
        "polygons": polygons_path,
        "footprints": footprints_path,
        "tindex": tindex_path,
        "laz_dir": tmp_path,
        "out": out_dir,
    }


def test_runner_end_to_end(synth_inputs: dict[str, Path]) -> None:
    cfg = PVGeomConfig()
    # Synthetic LAZ has class 6, so primary class is used; HAG threshold relaxed
    cfg.compute.backend = "local"

    manifest = run_pipeline(
        polygons_uri=str(synth_inputs["polygons"]),
        tile_index_uri=str(synth_inputs["tindex"]),
        lidar_prefix=str(synth_inputs["laz_dir"]),
        footprints_uri=str(synth_inputs["footprints"]),
        output_uri=str(synth_inputs["out"]),
        cfg=cfg,
        name_template="tile.laz",   # single tile in the dir
        tile_id_col="Name",
        use_dask=False,             # serial for deterministic test runtime
    )
    assert manifest.exists()
    out_files = sorted(synth_inputs["out"].glob("part-*.parquet"))
    assert len(out_files) == 1

    table = pq.read_table(out_files[0])
    assert set(table.schema.names) == set(OUTPUT_SCHEMA.names)
    assert len(table) == 1
    row = table.to_pylist()[0]
    assert row["polygon_id"] == "panel_1"
    assert row["panel_tilt_deg"] == pytest.approx(20.0, abs=1.0)
    diff = ((row["panel_azimuth_deg"] - 180.0 + 180) % 360) - 180
    assert abs(diff) < 1.5
    assert row["panel_rmse_m"] < 0.05
    assert row["on_building"] is True
    assert row["building_id"] == "b1"
    # roof was at 5 deg tilt; panel-roof angle ~15 deg → tilted_rack
    assert row["mounting_type"] == "tilted_rack_rooftop"
    assert row["mounting_rule"] == "R2"
    assert row["mounting_confidence"] >= 0.5


def test_resume_preserves_existing_partition(synth_inputs: dict[str, Path]) -> None:
    """First run writes the partition; second run with resume=True must NOT
    overwrite it. We mark the file with mtime + a sentinel-byte change to
    detect any rewrite.
    """
    cfg = PVGeomConfig()
    cfg.compute.backend = "local"

    kwargs = dict(
        polygons_uri=str(synth_inputs["polygons"]),
        tile_index_uri=str(synth_inputs["tindex"]),
        lidar_prefix=str(synth_inputs["laz_dir"]),
        footprints_uri=str(synth_inputs["footprints"]),
        output_uri=str(synth_inputs["out"]),
        cfg=cfg,
        name_template="tile.laz",
        use_dask=False,
    )
    run_pipeline(**kwargs)
    parts = list(synth_inputs["out"].glob("part-*.parquet"))
    assert len(parts) == 1
    part = parts[0]
    pre_mtime = part.stat().st_mtime_ns
    pre_bytes = part.read_bytes()

    # Touch the manifest to a no-op; second invocation with resume should skip.
    run_pipeline(**kwargs, resume=True)
    post_mtime = part.stat().st_mtime_ns
    post_bytes = part.read_bytes()
    assert pre_mtime == post_mtime, "resume=True must not rewrite an existing partition"
    assert pre_bytes == post_bytes
    # Manifest still reflects the same row count (read-back from disk)
    import json
    manifest = json.loads((synth_inputs["out"] / "manifest.json").read_text())
    assert manifest["counts"]["succeeded"] == 1
    assert manifest["counts"]["failed"] == 0


def test_resume_without_existing_runs_normally(synth_inputs: dict[str, Path]) -> None:
    """If no partitions on disk, --resume just runs everything (no-op flag)."""
    cfg = PVGeomConfig()
    manifest = run_pipeline(
        polygons_uri=str(synth_inputs["polygons"]),
        tile_index_uri=str(synth_inputs["tindex"]),
        lidar_prefix=str(synth_inputs["laz_dir"]),
        footprints_uri=str(synth_inputs["footprints"]),
        output_uri=str(synth_inputs["out"]),
        cfg=cfg,
        name_template="tile.laz",
        resume=True,
        use_dask=False,
    )
    assert manifest.exists()
    parts = list(synth_inputs["out"].glob("part-*.parquet"))
    assert len(parts) == 1


def test_process_tile_group_missing_primary_tile(synth_inputs: dict[str, Path]) -> None:
    """If the primary LAZ tile is missing on S3, the worker emits an empty
    OUTPUT_SCHEMA-conformant table instead of crashing the whole pipeline.
    Coiled relies on this — pre-warm is skipped there, so 404s surface inside
    workers rather than being filtered up-front.
    """
    import geopandas as gpd
    from shapely.geometry import box

    from pv_geom.io._localize import RemoteFileMissing
    from pv_geom.pipeline.tile_task import process_tile_group

    cfg = PVGeomConfig()
    polys = gpd.GeoDataFrame(
        {"polygon_id": ["p1"]},
        geometry=[box(40.0, 40.0, 50.0, 50.0)],
        crs="EPSG:6341",
    )
    footprints = gpd.GeoDataFrame(
        {"building_id": ["b1"]},
        geometry=[box(35.0, 35.0, 65.0, 65.0)],
        crs="EPSG:6341",
    )

    def _raise_missing(*args, **kwargs):
        raise RemoteFileMissing("s3://fake/missing.laz not found")

    import pv_geom.pipeline.tile_task as tile_task_mod

    orig = tile_task_mod.read_tile_points
    tile_task_mod.read_tile_points = _raise_missing
    try:
        table = process_tile_group(
            tile_uri_map={"t1": "s3://fake/missing.laz"},
            primary_tile_id="t1",
            polygons=polys,
            fetch_tile_ids=("t1",),
            footprints=footprints,
            cfg=cfg,
            config_hash="deadbeef",
            run_id="run1",
            partition_id=0,
        )
    finally:
        tile_task_mod.read_tile_points = orig

    # Empty table, but with the canonical schema (so pyarrow.concat_tables works).
    assert len(table) == 0
    assert set(table.schema.names) == set(OUTPUT_SCHEMA.names)


def test_runner_dry_run(synth_inputs: dict[str, Path]) -> None:
    cfg = PVGeomConfig()
    manifest = run_pipeline(
        polygons_uri=str(synth_inputs["polygons"]),
        tile_index_uri=str(synth_inputs["tindex"]),
        lidar_prefix=str(synth_inputs["laz_dir"]),
        footprints_uri=str(synth_inputs["footprints"]),
        output_uri=str(synth_inputs["out"]),
        cfg=cfg,
        name_template="tile.laz",
        dry_run=True,
        use_dask=False,
    )
    assert manifest.exists()
    # No partition files written in dry-run mode
    assert not list(synth_inputs["out"].glob("part-*.parquet"))
