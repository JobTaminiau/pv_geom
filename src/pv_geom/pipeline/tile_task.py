"""Per-tile-group worker. M6.

Given a set of tiles to fetch and the polygons assigned to one primary tile,
load all relevant LiDAR returns, apply the M3 panel fit + M4 roof plane and
heights + M5 multi-plane and mounting classification, and emit one output row
per polygon. Output is a pyarrow Table matching ``schema.OUTPUT_SCHEMA``.
"""

from __future__ import annotations

import hashlib
from typing import Any

import geopandas as gpd
import numpy as np
import pyarrow as pa
from shapely import wkb

from pv_geom import __version__
from pv_geom.classify.interface import MountingFeatures
from pv_geom.classify.rules import classify_mounting
from pv_geom.config import PVGeomConfig
from pv_geom.geometry.heights import (
    ground_under_polygon,
    height_above_ground,
    height_above_roof,
    panel_roof_angle_deg,
)
from pv_geom.geometry.multi_plane import (
    detect_multi_plane,
    is_tracker_suspected,
    polygon_aspect_ratio,
)
from pv_geom.geometry.plane_fit import bootstrap_uncertainty, fit_plane_ransac
from pv_geom.geometry.roof_plane import extract_roof_plane
from pv_geom.io._localize import RemoteFileMissing
from pv_geom.io.lidar import clip_points_to_polygon, read_tile_points
from pv_geom.schema import OUTPUT_SCHEMA


def _seed_for_polygon(polygon_id: str) -> int:
    """Stable RNG seed derived from polygon_id (PRD §10 determinism).

    Must not use the builtin ``hash()``: string hashing is salted per process
    (PYTHONHASHSEED), so every Dask worker would draw different RANSAC and
    bootstrap seeds and runs would not be reproducible.
    """
    digest = hashlib.sha256(polygon_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _split_classes_for_tile_group(
    pts: np.ndarray, cfg: PVGeomConfig
) -> tuple[np.ndarray, np.ndarray, int]:
    """Split a tile-group point cloud into ground_xyz + panel-class points once.

    Returns ``(ground_xyz, panel_pts_xyz, class_used)``. When class 6 returns are
    present they win; otherwise class-1 returns above the global ground median
    plus ``fallback_height_above_ground_m`` are used. The fallback uses a single
    tile-group-wide ground median (rather than a per-polygon neighborhood
    median) because Phoenix terrain is flat at the 1 km tile scale and a
    per-polygon class-equality scan over 50M+ points was the bottleneck for
    1000-polygon runs (see HANDOFF.md).
    """
    if pts.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), cfg.io.classification.panel_class_primary

    cls = pts[:, 3].astype(np.int16)
    primary = cfg.io.classification.panel_class_primary
    fallback = cfg.io.classification.panel_class_fallback
    ground_cls = cfg.io.classification.ground_class

    ground_xyz = pts[cls == ground_cls][:, :3]

    if (cls == primary).any():
        return ground_xyz, pts[cls == primary][:, :3], primary

    panel = pts[cls == fallback][:, :3]
    if len(panel) == 0:
        return ground_xyz, panel, fallback

    if len(ground_xyz):
        gz = float(np.median(ground_xyz[:, 2]))
        panel = panel[panel[:, 2] > gz + cfg.io.classification.fallback_height_above_ground_m]
    return ground_xyz, panel, fallback


def _build_row(
    polygon: Any,                           # shapely geometry
    polygon_id: str,
    cfg: PVGeomConfig,
    config_hash: str,
    run_id: str,
    partition_id: int,
    *,
    panel_pts: np.ndarray,                  # (N, 3) panel-class points clipped to polygon
    ground_xyz: np.ndarray,                 # (G, 3) class-2 returns for the tile group
    roof_input_pts: np.ndarray,             # (P, 3) panel-class returns for the tile group (ring-clipped inside extract_roof_plane)
    footprints: gpd.GeoDataFrame,
    other_pv_polygons: gpd.GeoDataFrame,
    contributing_tile_ids: tuple[str, ...],
    parent_polygon_id: str | None = None,   # defaults to polygon_id (single-part input)
) -> dict[str, Any]:
    """Compute all per-row fields. Returns a dict keyed on schema names."""
    flags: list[str] = []
    centroid = polygon.centroid
    cx, cy = float(centroid.x), float(centroid.y)
    area_m2 = float(polygon.area)
    aspect = polygon_aspect_ratio(polygon)

    # Density check
    density = len(panel_pts) / area_m2 if area_m2 > 0 else 0.0
    if density < cfg.panel_plane.min_density_pts_per_m2 or len(panel_pts) < cfg.panel_plane.min_points:
        flags.append("low_density")

    # M3 panel fit (always attempted; flags low_density does NOT prevent fit)
    seed = _seed_for_polygon(polygon_id)
    if len(panel_pts) >= 3:
        panel_fit = fit_plane_ransac(
            panel_pts,
            ransac_threshold=cfg.panel_plane.ransac_threshold_m,
            min_inlier_frac=cfg.panel_plane.min_inlier_frac,
            max_iter=cfg.panel_plane.max_iter,
            tilt_floor_deg=cfg.panel_plane.tilt_floor_deg,
            seed=seed,
        )
    else:
        from pv_geom.geometry.plane_fit import _failed_fit
        panel_fit = _failed_fit(len(panel_pts))

    if np.isnan(panel_fit.tilt_deg):
        flags.append("poor_fit")
    elif np.isnan(panel_fit.azimuth_deg):
        flags.append("near_horizontal")

    # Bootstrap uncertainty
    if cfg.panel_plane.uncertainty_method == "bootstrap" and panel_fit.n_inliers >= 3:
        tilt_unc, az_unc = bootstrap_uncertainty(
            panel_pts, panel_fit,
            n_samples=cfg.panel_plane.bootstrap_samples, seed=seed,
        )
    else:
        tilt_unc, az_unc = float("nan"), float("nan")

    # M5 multi-plane detection
    if cfg.multi_plane.enabled and panel_fit.n_inliers >= 10 and not np.isnan(panel_fit.tilt_deg):
        mp = detect_multi_plane(panel_pts, panel_fit, cfg.multi_plane, seed=seed)
        flags.extend(mp.flags)
        secondary = mp.secondary
    else:
        secondary = None

    # M4 roof plane (roof_input_pts is the tile-group panel-class set, pre-filtered upstream)
    if cfg.roof_plane.enabled:
        roof_res = extract_roof_plane(
            polygon, footprints, other_pv_polygons,
            roof_input_pts, cfg.roof_plane, seed=seed,
        )
    else:
        from pv_geom.geometry.roof_plane import RoofPlaneResult
        roof_res = RoofPlaneResult(fit=None, on_building=False, building_id=None,
                                   flag=None, used_buffer_m=None)

    if roof_res.flag:
        flags.append(roof_res.flag)

    # A flagged roof result still reports its fit in the roof_* columns for QC,
    # but only an unflagged one is precise enough to measure a panel against.
    roof_plane_available = roof_res.usable

    # M4 heights
    panel_inliers = panel_pts[panel_fit.inlier_mask] if panel_fit.n_inliers > 0 else panel_pts[:0]
    panel_z = panel_inliers[:, 2] if len(panel_inliers) else panel_pts[:, 2]
    hag = height_above_ground(
        panel_z,
        ground_xyz, (cx, cy),
        cfg.heights.ground_search_radius_m,
        fallback_k=cfg.heights.ground_fallback_k,
        fallback_max_radius_m=cfg.heights.ground_fallback_max_radius_m,
    )
    if roof_plane_available:
        har = height_above_roof(panel_inliers, roof_res.fit) if len(panel_inliers) else float("nan")
        pra = panel_roof_angle_deg(panel_fit, roof_res.fit)
    else:
        har = float("nan")
        pra = float("nan")

    # Panel-standoff screen. Three outcomes, and the row must say which:
    #
    #   no_panel_standoff     — screened, FAILED. The panel plane is not
    #       resolvably above the roof. Expected on a genuinely low-profile
    #       flush mount, but also the signature of an input polygon whose
    #       panels were absent from the point cloud: when the imagery the
    #       detection came from postdates the LiDAR there are no panel returns
    #       to fit, so RANSAC fits the bare roof and the row reports *roof*
    #       geometry under the panel column names.
    #   standoff_unscreenable — could not be screened. No trustworthy roof
    #       reference (off-building, or the ring fit failed), so there is
    #       nothing to measure the panel plane against. Says nothing about
    #       whether the panels were there; it is the absence of a test, and on
    #       the v0.1.0 Phoenix run this was the majority of rows.
    #   neither flag          — screened, PASSED.
    #
    # Downstream work that needs measured *panel* geometry (rack tilt/azimuth,
    # panel-roof angle) should keep only the third case.
    standoff_screenable = roof_plane_available and not np.isnan(har)
    if not standoff_screenable:
        flags.append("standoff_unscreenable")
    elif har < cfg.heights.min_panel_standoff_m:
        flags.append("no_panel_standoff")

    # Canopy evidence: ground-class returns under the panel polygon (NaN HAG
    # or footprint errors must not silently become "ground level"; the gap
    # under the panels is the direct LiDAR signal).
    n_ground_under, ground_under_gap = ground_under_polygon(panel_z, ground_xyz, polygon)
    canopy_evidence = (
        n_ground_under >= cfg.mounting_rules.canopy_min_ground_points_under
        and not np.isnan(ground_under_gap)
        and ground_under_gap >= cfg.mounting_rules.canopy_gap_m_min
    )

    # Off-building + high HAG + no canopy evidence: most likely a rooftop on a
    # building the footprint layer is missing. The R3/R8 height caps route it
    # toward ambiguous; the flag makes the population auditable.
    if (
        not roof_res.on_building
        and not canopy_evidence
        and not np.isnan(hag)
        and hag > cfg.mounting_rules.missing_footprint_hag_m
    ):
        flags.append("possible_missing_footprint")

    # Tracker-suspected flag (per-polygon heuristic; PRD §7.2 v1).
    # NaN HAG/tilt propagate — is_tracker_suspected returns False on NaN,
    # which is the correct "unknown" behavior (was coerced to 0.0 pre-0.3).
    if is_tracker_suspected(
        on_building=roof_res.on_building,
        aspect_ratio=aspect,
        height_above_ground_m=hag,
        panel_tilt_deg=panel_fit.tilt_deg,
    ):
        flags.append("tracker_suspected")

    # M5 mounting classification. NaN HAG propagates: the rules treat unknown
    # heights as non-evidence and fall through to ambiguous rather than
    # confidently classifying at "ground level".
    feats = MountingFeatures(
        on_building=roof_res.on_building,
        panel_tilt_deg=panel_fit.tilt_deg,
        panel_azimuth_deg=panel_fit.azimuth_deg,
        panel_roof_angle_deg=pra,
        height_above_roof_m=har,
        height_above_ground_m=hag,
        area_m2=area_m2,
        aspect_ratio=aspect,
        roof_plane_available=roof_plane_available,
        roof_tilt_deg=float(roof_res.fit.tilt_deg) if roof_plane_available else None,
        east_west_rack="east_west_rack" in flags,
        n_ground_under=n_ground_under,
        ground_under_gap_m=ground_under_gap,
        no_panel_standoff="no_panel_standoff" in flags,
    )
    mr = classify_mounting(feats, cfg.mounting_rules)

    return {
        "polygon_id": str(polygon_id),
        "parent_polygon_id": str(parent_polygon_id if parent_polygon_id is not None else polygon_id),
        "geometry": wkb.dumps(polygon),
        "n_points_panel": int(panel_fit.n_total),
        "n_inliers_panel": int(panel_fit.n_inliers),
        "panel_tilt_deg": _f32(panel_fit.tilt_deg),
        "panel_azimuth_deg": _f32(panel_fit.azimuth_deg),
        "panel_rmse_m": _f32(panel_fit.rmse),
        "panel_tilt_unc_deg": _f32(tilt_unc),
        "panel_azimuth_unc_deg": _f32(az_unc),
        "n_planes_detected": np.int8(2 if secondary is not None else (1 if not np.isnan(panel_fit.tilt_deg) else 0)),
        "secondary_tilt_deg": _f32(secondary.tilt_deg) if secondary is not None else None,
        "secondary_azimuth_deg": _f32(secondary.azimuth_deg) if secondary is not None else None,
        "roof_tilt_deg": _f32(roof_res.fit.tilt_deg) if roof_res.fit is not None else None,
        "roof_azimuth_deg": _f32(roof_res.fit.azimuth_deg) if roof_res.fit is not None else None,
        "roof_rmse_m": _f32(roof_res.fit.rmse) if roof_res.fit is not None else None,
        "panel_roof_angle_deg": _f32(pra),
        "height_above_roof_m": _f32(har),
        "height_above_ground_m": _f32(hag),   # null when unknown (schema nullable)
        "on_building": bool(roof_res.on_building),
        "building_id": roof_res.building_id,
        "area_m2": np.float32(area_m2),
        "aspect_ratio": _f32(aspect),
        "mounting_type": str(mr.label),
        "mounting_confidence": np.float32(mr.confidence),
        "mounting_rule": str(mr.triggered_rule),
        "flags": list(flags),
        "lidar_tile_ids": list(contributing_tile_ids),
        "pkg_version": __version__,
        "config_hash": config_hash,
        "run_id": run_id,
        "partition_id": np.int32(partition_id),
    }


def _f32(v) -> Any:
    """np.float32 with NaN passthrough as None (so pa null is honored)."""
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    return np.float32(v)


def process_tile_group(
    tile_uri_map: dict[str, str],
    primary_tile_id: str,
    polygons: gpd.GeoDataFrame,
    fetch_tile_ids: tuple[str, ...],
    footprints: gpd.GeoDataFrame,
    cfg: PVGeomConfig,
    *,
    config_hash: str,
    run_id: str,
    partition_id: int,
    polygon_id_col: str = "polygon_id",
) -> pa.Table:
    """Fetch all tiles in ``fetch_tile_ids``, run M3-M5 per polygon, return a table."""
    # 1) Load all required tiles' points (caller-side caching via io.lidar).
    # Missing tiles are tolerated so the runner doesn't have to pre-screen
    # the whole bucket; if the *primary* tile is missing we emit no rows
    # because a tile group's polygons live on its primary tile by
    # construction.
    primary_loaded = False
    chunks: list[np.ndarray] = []
    for tid in fetch_tile_ids:
        uri = tile_uri_map.get(tid)
        if uri is None:
            continue
        try:
            pts, _ = read_tile_points(uri)
        except RemoteFileMissing:
            print(f"[tile_task] {uri} missing; skipping")
            continue
        chunks.append(pts)
        if tid == primary_tile_id:
            primary_loaded = True

    if not primary_loaded:
        print(f"[tile_task] primary tile {primary_tile_id} unavailable; "
              f"emitting empty table for partition {partition_id}")
        return pa.table(
            {f.name: [] for f in OUTPUT_SCHEMA},
            schema=OUTPUT_SCHEMA,
        )

    all_pts = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 4))

    # 2) Filter classes ONCE for the whole tile group, not per polygon. The
    # 50M-point class-equality scan was the per-polygon-loop bottleneck that
    # OOM-killed Dask workers on the 1000-polygon benchmark (HANDOFF.md).
    ground_xyz, panel_pts_all, _ = _split_classes_for_tile_group(all_pts, cfg)
    del all_pts  # ~1.5 GB freed before the per-polygon loop

    contributing_tile_ids = tuple(t for t in fetch_tile_ids if tile_uri_map.get(t))

    # 3) For each polygon, just clip the pre-filtered panel pool and build a row.
    rows: list[dict[str, Any]] = []
    has_parent_col = "parent_polygon_id" in polygons.columns
    for _, row in polygons.iterrows():
        poly = row.geometry
        pid = str(row[polygon_id_col])
        parent_id = str(row["parent_polygon_id"]) if has_parent_col else pid

        in_panel = clip_points_to_polygon(
            panel_pts_all,
            poly,
            erosion_m=cfg.panel_plane.erosion_m,
        )

        # Other PV polygons sharing this group
        other_pvs = polygons[polygons[polygon_id_col] != pid][["geometry"]]
        other_pvs = gpd.GeoDataFrame(other_pvs.reset_index(drop=True), crs=polygons.crs)

        rows.append(
            _build_row(
                polygon=poly,
                polygon_id=pid,
                parent_polygon_id=parent_id,
                cfg=cfg,
                config_hash=config_hash,
                run_id=run_id,
                partition_id=partition_id,
                panel_pts=in_panel,
                ground_xyz=ground_xyz,
                roof_input_pts=panel_pts_all,
                footprints=footprints,
                other_pv_polygons=other_pvs,
                contributing_tile_ids=contributing_tile_ids,
            )
        )

    # Build pyarrow table aligned to OUTPUT_SCHEMA
    cols = {f.name: [r.get(f.name) for r in rows] for f in OUTPUT_SCHEMA}
    return pa.table(cols, schema=OUTPUT_SCHEMA)
