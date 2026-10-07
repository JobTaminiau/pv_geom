"""End-to-end smoke test for the M6 runner on fully synthetic inputs.

Synthesizes a tile (with a planar surface representing a PV array on a roof
plus surrounding roof returns + ground), a single PV polygon over the panel,
a building footprint covering both, and a tile index. Runs ``run_pipeline``
and verifies the output schema + the recovered tilt/azimuth + mounting label.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pyarrow.parquet as pq
import pytest
from shapely.geometry import box

from pv_geom.config import PVGeomConfig
from pv_geom.pipeline.runner import run_pipeline
from pv_geom.schema import OUTPUT_SCHEMA, OUTPUT_SCHEMA_WITH_MOUNTING


def _mounting_cfg() -> PVGeomConfig:
    """Config with the archived (experimental) mounting classifier switched on."""
    cfg = PVGeomConfig()
    cfg.mounting_rules.enabled = True
    return cfg


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

    # Ground (class 2): scattered points across the tile at z=0, but NONE
    # inside the building footprint — LiDAR cannot see the ground through a
    # roof, and unphysical under-roof ground returns would (correctly) trip
    # the canopy-evidence detector and reroute the rooftop label.
    bx0, by0, bx1, by1 = 35.0, 35.0, 65.0, 65.0
    n_ground = 5_000
    g_xy = rng.uniform(
        [tile_extent[0], tile_extent[1]],
        [tile_extent[2], tile_extent[3]],
        size=(n_ground, 2),
    )
    under_building = (
        (g_xy[:, 0] >= bx0) & (g_xy[:, 0] <= bx1)
        & (g_xy[:, 1] >= by0) & (g_xy[:, 1] <= by1)
    )
    g_xy = g_xy[~under_building]
    ground = np.column_stack([g_xy, np.zeros(len(g_xy)), np.full(len(g_xy), 2)])

    # Roof (class 6) inside a 30x30m building; planar, low-tilt south
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
    cfg = _mounting_cfg()
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
    assert set(table.schema.names) == set(OUTPUT_SCHEMA_WITH_MOUNTING.names)
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
    # Single-part input: the join key degenerates to the row id.
    assert row["parent_polygon_id"] == "panel_1"


def test_runner_multipolygon_explode_join_back(synth_inputs: dict[str, Path]) -> None:
    """A MultiPolygon detection (under the `detection_id` alias, as in the S8
    atlas parquet) explodes into per-part rows whose parent_polygon_id joins
    back to the input detection id — no `__p<i>` suffix parsing needed."""
    from shapely.geometry import MultiPolygon

    multi = gpd.GeoDataFrame(
        {"detection_id": ["det_1"]},
        geometry=[MultiPolygon([box(40.0, 40.0, 45.0, 50.0), box(45.0, 40.0, 50.0, 50.0)])],
        crs="EPSG:6341",
    )
    polygons_path = synth_inputs["laz_dir"] / "polys_multi.parquet"
    multi.to_parquet(polygons_path)

    cfg = PVGeomConfig()
    cfg.compute.backend = "local"
    run_pipeline(
        polygons_uri=str(polygons_path),
        tile_index_uri=str(synth_inputs["tindex"]),
        lidar_prefix=str(synth_inputs["laz_dir"]),
        footprints_uri=str(synth_inputs["footprints"]),
        output_uri=str(synth_inputs["out"]),
        cfg=cfg,
        name_template="tile.laz",
        tile_id_col="Name",
        use_dask=False,
    )
    table = pq.read_table(sorted(synth_inputs["out"].glob("part-*.parquet"))[0])
    rows = table.to_pylist()
    assert {r["polygon_id"] for r in rows} == {"det_1__p0", "det_1__p1"}
    assert {r["parent_polygon_id"] for r in rows} == {"det_1"}


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
    from pv_geom.pipeline.worker import process_tile_group

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

    import pv_geom.pipeline.pointpool as worker_mod

    orig = worker_mod.read_tile
    worker_mod.read_tile = _raise_missing
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
        worker_mod.read_tile = orig

    # Empty table, but with the canonical schema (so pyarrow.concat_tables works).
    assert len(table) == 0
    assert set(table.schema.names) == set(OUTPUT_SCHEMA.names)


def test_build_row_carport_from_under_panel_returns() -> None:
    """LiDAR-only canopy classification: an elevated panel with ground returns
    underneath and NO footprint involvement must come out as carport."""
    from pv_geom.pipeline.worker import build_row

    rng = np.random.default_rng(0)
    poly = box(0.0, 0.0, 12.0, 3.0)                  # aspect ratio 4
    n = 300
    p_xy = rng.uniform([0.0, 0.0], [12.0, 3.0], size=(n, 2))
    t = np.radians(8.0)                              # ~8 deg south-facing at ~3 m
    p_z = 3.0 - np.tan(t) * (p_xy[:, 1] - 1.5) + rng.normal(0, 0.01, n)
    panel_pts = np.column_stack([p_xy, p_z])
    g_xy = rng.uniform([-20.0, -20.0], [30.0, 20.0], size=(4000, 2))
    ground = np.column_stack([g_xy, np.zeros(4000)])  # z=0, incl. under the canopy
    footprints = gpd.GeoDataFrame({"building_id": []}, geometry=[], crs="EPSG:6341")
    others = gpd.GeoDataFrame(geometry=[], crs="EPSG:6341")

    row = build_row(
        polygon=poly, polygon_id="c1", cfg=_mounting_cfg(), config_hash="x",
        run_id="r", partition_id=0,
        panel_pts=panel_pts, ground_xyz=ground, roof_input_pts=panel_pts,
        footprints=footprints, other_pv_polygons=others,
        contributing_tile_ids=("t1",),
    )
    assert row["on_building"] is False
    assert row["mounting_type"] == "carport"
    assert row["mounting_confidence"] >= 0.5
    assert "possible_missing_footprint" not in row["flags"]


def test_build_row_nan_hag_stays_unknown() -> None:
    """No ground reference anywhere near the polygon: HAG must be null in the
    output and the mounting ambiguous — pre-0.3 the NaN was coerced to 0.0 and
    the polygon fired R4/R5 as a confident ground mount."""
    from pv_geom.pipeline.worker import build_row

    rng = np.random.default_rng(1)
    poly = box(0.0, 0.0, 12.0, 3.0)
    n = 300
    p_xy = rng.uniform([0.0, 0.0], [12.0, 3.0], size=(n, 2))
    t = np.radians(8.0)
    p_z = 3.0 - np.tan(t) * (p_xy[:, 1] - 1.5) + rng.normal(0, 0.01, n)
    panel_pts = np.column_stack([p_xy, p_z])
    # Ground exists but only ~150 m away — beyond the fallback max radius.
    g_xy = rng.uniform([150.0, 0.0], [160.0, 10.0], size=(200, 2))
    ground = np.column_stack([g_xy, np.zeros(200)])
    footprints = gpd.GeoDataFrame({"building_id": []}, geometry=[], crs="EPSG:6341")
    others = gpd.GeoDataFrame(geometry=[], crs="EPSG:6341")

    row = build_row(
        polygon=poly, polygon_id="u1", cfg=_mounting_cfg(), config_hash="x",
        run_id="r", partition_id=0,
        panel_pts=panel_pts, ground_xyz=ground, roof_input_pts=panel_pts,
        footprints=footprints, other_pv_polygons=others,
        contributing_tile_ids=("t1",),
    )
    assert row["height_above_ground_m"] is None
    assert row["mounting_type"] == "ambiguous"


def _rooftop_scene(standoff_m: float) -> dict:
    """A pitched roof with a polygon on it, panels ``standoff_m`` above the
    roof surface. standoff 0 reproduces the case where the detection's panels
    were absent from the point cloud and RANSAC can only fit the bare roof.
    """
    rng = np.random.default_rng(7)
    footprint = box(-10.0, -10.0, 20.0, 20.0)
    panel_poly = box(0.0, 0.0, 6.0, 4.0)
    slope = np.tan(np.radians(20.0))          # 20 deg, facing -y (south)

    def roof_z(xy: np.ndarray) -> np.ndarray:
        return 6.0 - slope * (xy[:, 1] - 5.0)

    r_xy = rng.uniform([-10.0, -10.0], [20.0, 20.0], size=(6000, 2))
    roof_pts = np.column_stack([r_xy, roof_z(r_xy) + rng.normal(0, 0.01, len(r_xy))])
    p_xy = rng.uniform([0.0, 0.0], [6.0, 4.0], size=(600, 2))
    panel_pts = np.column_stack(
        [p_xy, roof_z(p_xy) + standoff_m + rng.normal(0, 0.01, len(p_xy))]
    )
    # Ground only outside the building — LiDAR cannot see through a roof.
    g_xy = rng.uniform([-60.0, -60.0], [60.0, 60.0], size=(4000, 2))
    keep = (g_xy[:, 0] < -12) | (g_xy[:, 0] > 22) | (g_xy[:, 1] < -12) | (g_xy[:, 1] > 22)
    ground = np.column_stack([g_xy[keep], np.zeros(keep.sum())])
    return {
        "polygon": panel_poly,
        "panel_pts": panel_pts,
        "roof_input_pts": np.vstack([roof_pts, panel_pts]),
        "ground_xyz": ground,
        "footprints": gpd.GeoDataFrame(
            {"building_id": ["b1"]}, geometry=[footprint], crs="EPSG:6341"
        ),
        "other_pv_polygons": gpd.GeoDataFrame(geometry=[], crs="EPSG:6341"),
    }


def test_build_row_flags_no_panel_standoff_when_panels_absent() -> None:
    """Imagery postdating the LiDAR: the polygon is real but its panels were
    not in the cloud, so the "panel" plane is the bare roof. The row still
    looks like a clean flush mount, so it must carry `no_panel_standoff`."""
    from pv_geom.pipeline.worker import build_row

    scene = _rooftop_scene(standoff_m=0.0)
    row = build_row(
        polygon_id="v0", cfg=PVGeomConfig(), config_hash="x", run_id="r",
        partition_id=0, contributing_tile_ids=("t1",), **scene,
    )
    assert row["on_building"] is True
    assert "no_panel_standoff" in row["flags"]


def test_build_row_no_standoff_flag_for_real_flush_array() -> None:
    """A genuine flush mount sits ~10 cm above the roof (racking + frame) —
    resolvably above it, so the flag must stay off."""
    from pv_geom.pipeline.worker import build_row

    scene = _rooftop_scene(standoff_m=0.10)
    row = build_row(
        polygon_id="v1", cfg=PVGeomConfig(), config_hash="x", run_id="r",
        partition_id=0, contributing_tile_ids=("t1",), **scene,
    )
    assert row["on_building"] is True
    assert "no_panel_standoff" not in row["flags"]


def test_seed_for_polygon_is_process_stable() -> None:
    """RNG seeds must be identical across processes/runs (PRD §10). The builtin
    ``hash()`` is salted per process, so the seed is sha256-derived; pin the
    exact value so a regression to ``hash()`` fails on any interpreter."""
    from pv_geom.pipeline.measure import seed_for_polygon

    assert seed_for_polygon("21_840480_397711__0") == 1798131859
    assert 0 <= seed_for_polygon("anything") < 2**32


def test_failed_tile_group_does_not_abort_run(tmp_path: Path) -> None:
    """A tile group whose processing raises (here: corrupt LAZ) must not abort
    the run: the other group's partition is still written, the failure lands in
    the manifest, and a follow-up --resume retries only the failed group."""
    import json

    good_laz = tmp_path / "good.laz"
    _write_synthetic_laz(good_laz)
    bad_laz = tmp_path / "bad.laz"
    bad_laz.write_bytes(b"this is not a LAZ file")

    polygons = gpd.GeoDataFrame(
        {"polygon_id": ["p_good", "p_bad"]},
        geometry=[box(40.0, 40.0, 50.0, 50.0), box(140.0, 40.0, 150.0, 50.0)],
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
        {"Name": ["good", "bad"]},
        geometry=[box(0.0, 0.0, 100.0, 100.0), box(100.0, 0.0, 200.0, 100.0)],
        crs="EPSG:6341",
    )
    tindex_path = tmp_path / "tindex.parquet"
    tindex.to_parquet(tindex_path)

    kwargs = dict(
        polygons_uri=str(polygons_path),
        tile_index_uri=str(tindex_path),
        lidar_prefix=str(tmp_path),
        footprints_uri=str(footprints_path),
        output_uri=str(tmp_path / "out"),
        cfg=PVGeomConfig(),
        name_template="{name}.laz",
        use_dask=False,
    )
    run_pipeline(**kwargs)

    parts = sorted((tmp_path / "out").glob("part-*.parquet"))
    assert len(parts) == 1, "the good group's partition must survive the bad group's failure"
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["counts"]["succeeded"] == 1
    assert manifest["counts"]["failed"] == 1
    assert manifest["counts"]["failed_groups"] == 1
    assert len(manifest["aggregate_stats"]["task_errors"]) == 1

    # Fix the bad tile; --resume retries only the failed group.
    _write_synthetic_laz(
        bad_laz,
        tile_extent=(100.0, 0.0, 200.0, 100.0),
        panel_extent=(140.0, 40.0, 150.0, 50.0),
    )
    run_pipeline(**kwargs, resume=True)
    parts = sorted((tmp_path / "out").glob("part-*.parquet"))
    assert len(parts) == 2
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["counts"]["succeeded"] == 2
    assert manifest["counts"]["failed"] == 0
    assert manifest["counts"]["failed_groups"] == 0


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


def test_build_row_marks_unscreenable_when_no_roof_reference() -> None:
    """Off-building: there is no roof to measure the panel plane against, so the
    vintage screen could not run. The row must say so rather than look like it
    passed — the distinction is the whole point of the tri-state."""
    from pv_geom.pipeline.worker import build_row

    rng = np.random.default_rng(3)
    poly = box(0.0, 0.0, 6.0, 4.0)
    p_xy = rng.uniform([0.0, 0.0], [6.0, 4.0], size=(400, 2))
    panel_pts = np.column_stack([p_xy, np.full(400, 1.5) + rng.normal(0, 0.01, 400)])
    g_xy = rng.uniform([-20.0, -20.0], [20.0, 20.0], size=(600, 2))
    ground = np.column_stack([g_xy, np.zeros(len(g_xy))])

    row = build_row(
        polygon=poly, polygon_id="w0", cfg=PVGeomConfig(), config_hash="x",
        run_id="r", partition_id=0,
        panel_pts=panel_pts, ground_xyz=ground, roof_input_pts=panel_pts,
        footprints=gpd.GeoDataFrame({"building_id": []}, geometry=[], crs="EPSG:6341"),
        other_pv_polygons=gpd.GeoDataFrame(geometry=[], crs="EPSG:6341"),
        contributing_tile_ids=("t1",),
    )
    assert row["on_building"] is False
    assert row["height_above_roof_m"] is None
    assert "standoff_unscreenable" in row["flags"]
    assert "no_panel_standoff" not in row["flags"]


def test_standoff_screen_states_are_mutually_exclusive() -> None:
    """The three outcomes must partition the rows: failed, unscreenable, or
    (neither flag) screened and passed."""
    from pv_geom.pipeline.worker import build_row

    for standoff, expect in ((0.0, "no_panel_standoff"), (0.10, None)):
        scene = _rooftop_scene(standoff_m=standoff)
        row = build_row(
            polygon_id=f"s{standoff}", cfg=PVGeomConfig(), config_hash="x",
            run_id="r", partition_id=0, contributing_tile_ids=("t1",), **scene,
        )
        flags = set(row["flags"])
        assert "standoff_unscreenable" not in flags       # a roof fit existed
        assert ("no_panel_standoff" in flags) is (expect == "no_panel_standoff")


def test_absent_panels_do_not_earn_full_confidence() -> None:
    """End-to-end version of the rules-level cap: the bare-roof row is still
    labelled a flush mount, but must not outrank a measured one."""
    from pv_geom.pipeline.worker import build_row

    cfg = _mounting_cfg()
    absent = build_row(polygon_id="c0", cfg=cfg, config_hash="x", run_id="r",
                        partition_id=0, contributing_tile_ids=("t1",),
                        **_rooftop_scene(standoff_m=0.0))
    real = build_row(polygon_id="c1", cfg=cfg, config_hash="x", run_id="r",
                      partition_id=0, contributing_tile_ids=("t1",),
                      **_rooftop_scene(standoff_m=0.10))

    assert absent["mounting_type"].startswith("flush_mount")
    assert real["mounting_type"].startswith("flush_mount")
    assert absent["mounting_confidence"] <= cfg.mounting_rules.no_panel_standoff_confidence_max
    assert real["mounting_confidence"] > absent["mounting_confidence"]


# --------------------------------------------------------------------------- #
# Vintage probe + standoff aggregate (manifest surface)
# --------------------------------------------------------------------------- #


def _laz_flown_on(path, flight, n: int = 200) -> None:
    import laspy

    from pv_geom.io.lidar import (
        _GPS_EPOCH,
        _GPS_UTC_LEAP_SECONDS,
        _LAS_GPS_TIME_OFFSET,
    )

    rng = np.random.default_rng(0)
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([0.0, 0.0, 0.0])
    las = laspy.LasData(header)
    las.x = rng.uniform(0, 10, size=n)
    las.y = rng.uniform(0, 10, size=n)
    las.z = rng.uniform(0, 5, size=n)
    las.classification = np.full(n, 6, dtype=np.uint8)
    secs = (
        (datetime.combine(flight, datetime.min.time()) - _GPS_EPOCH).total_seconds()
        + _GPS_UTC_LEAP_SECONDS - _LAS_GPS_TIME_OFFSET
    )
    las.gps_time = np.full(n, secs)
    las.write(str(path))


def test_probe_lidar_vintage_reports_gap_when_input_postdates_lidar(tmp_path) -> None:
    """The Phoenix pairing: 2024 imagery against a 2020 flight. The run must
    measure the LiDAR side itself and record the gap in the manifest."""
    from pv_geom.pipeline.vintage_check import probe_lidar_vintage as _probe_lidar_vintage

    p = tmp_path / "t.laz"
    _laz_flown_on(p, date(2020, 11, 26))
    cfg = PVGeomConfig()
    cfg.vintage.polygon_vintage = date(2024, 4, 1)

    v = _probe_lidar_vintage(cfg, [str(p)])
    assert v["lidar_flight_end"] == "2020-11-26"
    assert v["lidar_date_source"] == "gps_time"
    assert v["polygon_vintage"] == "2024-04-01"
    assert v["vintage_gap_days"] == (date(2024, 4, 1) - date(2020, 11, 26)).days
    assert "vintage_gap_warning" in v


def test_probe_lidar_vintage_no_warning_when_cotemporal(tmp_path) -> None:
    from pv_geom.pipeline.vintage_check import probe_lidar_vintage as _probe_lidar_vintage

    p = tmp_path / "t.laz"
    _laz_flown_on(p, date(2024, 6, 1))
    cfg = PVGeomConfig()
    cfg.vintage.polygon_vintage = date(2024, 4, 1)

    v = _probe_lidar_vintage(cfg, [str(p)])
    assert v["vintage_gap_days"] < 0
    assert "vintage_gap_warning" not in v


def test_probe_lidar_vintage_survives_unreadable_tiles(tmp_path) -> None:
    """A bad tile costs a sample, never the run."""
    from pv_geom.pipeline.vintage_check import probe_lidar_vintage as _probe_lidar_vintage

    good = tmp_path / "good.laz"
    _laz_flown_on(good, date(2020, 11, 26))
    bad = tmp_path / "bad.laz"
    bad.write_bytes(b"not a laz file")

    v = _probe_lidar_vintage(PVGeomConfig(), [str(good), str(bad)])
    assert v["lidar_tiles_sampled"] == 1
    assert v["lidar_tiles_unreadable"] == 1
    assert v["lidar_flight_end"] == "2020-11-26"


def test_aggregate_reports_the_standoff_screen() -> None:
    """Manifest must carry the screen as a partition plus the cross-tab against
    mounting_type — that is what tells a reader which labels rest on unmeasured
    panel geometry."""
    import pyarrow as pa

    from pv_geom.summary import summarise_table as _aggregate

    tbl = pa.table({
        "mounting_type": ["flush_mount_flat_roof", "flush_mount_flat_roof",
                          "tilted_rack_rooftop", "carport"],
        "mounting_rule": ["R1", "R1", "R2", "R3"],
        "mounting_confidence": [0.5, 1.0, 1.0, 1.0],
        "on_building": [True, True, True, False],
        "panel_tilt_deg": [5.0, 5.0, 20.0, 8.0],
        "panel_azimuth_deg": [180.0, 180.0, 180.0, 90.0],
        "panel_rmse_m": [0.02, 0.02, 0.03, 0.04],
        "height_above_roof_m": [0.0, 0.12, None, None],
        "flags": [["no_panel_standoff"], [], ["standoff_unscreenable"],
                  ["standoff_unscreenable"]],
    })
    out = _aggregate(tbl)["standoff_screen"]
    assert out["no_panel_standoff"] == 1
    assert out["passed"] == 1
    assert out["unscreenable"] == 2
    assert out["screened_frac"] == pytest.approx(0.5)
    assert out["failed_frac_of_screened"] == pytest.approx(0.5)
    assert out["by_mounting_type"]["flush_mount_flat_roof"]["no_panel_standoff"] == 1
    assert out["by_mounting_type"]["flush_mount_flat_roof"]["passed"] == 1
