"""Per-tile-group worker.

Given a set of tiles to fetch and the polygons assigned to one primary tile,
load the relevant LiDAR returns, fit the panel plane, the roof reference and the
heights for each polygon, decide what each fit rests on (``geometry_basis``),
and emit one output row per polygon as a pyarrow Table matching
``schema.output_schema``.
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Any

import geopandas as gpd
import numpy as np
import pyarrow as pa
import shapely
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
from pv_geom.geometry.point_index import GroundModel, PointGrid
from pv_geom.geometry.roof_plane import extract_roof_plane
from pv_geom.io._localize import RemoteFileMissing
from pv_geom.io.lidar import clip_points_to_polygon, read_tile
from pv_geom.schema import output_schema
from pv_geom.vintage import geometry_basis, vintage_gap_days


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
    """Split a tile-group point cloud into ground_xyz + panel-candidate points.

    Returns ``(ground_xyz, panel_pts_xyz, class_used)``. When the primary panel
    class (ASPRS 6, building) is present it wins. Most public LiDAR does not
    carry it — neither the Phoenix USGS tiles nor the Delaware state collection
    do — so the usual path is the fallback: unclassified returns more than
    ``fallback_height_above_ground_m`` above *local* ground, read from a grid of
    ground elevations so the cut holds on sloping terrain.
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
    return ground_xyz, _above_ground(panel, ground_xyz, cfg), fallback


def _above_ground(candidates: np.ndarray, ground_xyz: np.ndarray, cfg: PVGeomConfig) -> np.ndarray:
    """Keep candidate returns above the local-ground cutoff (all of them when
    there is no ground reference at all)."""
    if len(candidates) == 0 or len(ground_xyz) == 0:
        return candidates
    model = GroundModel(ground_xyz, cell_m=cfg.io.classification.ground_grid_cell_m)
    gz = model.ground_z(candidates[:, 0], candidates[:, 1])
    return candidates[candidates[:, 2] > gz + cfg.io.classification.fallback_height_above_ground_m]


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
    roof_input_pts: np.ndarray,             # (P, 3) panel-class returns around the polygon (ring-clipped inside extract_roof_plane)
    footprints: gpd.GeoDataFrame | None,    # None = run has no footprint layer
    other_pv_polygons: gpd.GeoDataFrame,
    contributing_tile_ids: tuple[str, ...],
    parent_polygon_id: str | None = None,   # defaults to polygon_id (single-part input)
    input_row: int = 0,
    polygon_vintage: date | None = None,    # imagery capture date for this polygon
    lidar_date: date | None = None,         # LiDAR capture date for this polygon's tile
    lidar_date_source: str | None = None,   # declared | gps_time | header_date
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
        roof_res = RoofPlaneResult(fit=None,
                                   on_building=None if footprints is None else False,
                                   building_id=None, flag=None, used_buffer_m=None)

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

    # What this row's tilt and azimuth rest on: the standoff screen is direct
    # evidence that panels were in the cloud; failing that, the declared dates
    # of the two inputs decide (see pv_geom.vintage).
    fit_ok = not np.isnan(panel_fit.tilt_deg)
    gap_days = vintage_gap_days(polygon_vintage, lidar_date)
    basis = geometry_basis(
        fit_ok=fit_ok,
        standoff_passed=standoff_screenable and har >= cfg.heights.min_panel_standoff_m,
        standoff_screened=standoff_screenable,
        gap_days=gap_days,
    )

    row: dict[str, Any] = {
        "polygon_id": str(polygon_id),
        "parent_polygon_id": str(parent_polygon_id if parent_polygon_id is not None else polygon_id),
        "input_row": np.int32(input_row),
        "geometry": wkb.dumps(polygon),
        "area_m2": np.float32(area_m2),
        "surface_area_m2": (
            np.float32(area_m2 / np.cos(np.radians(panel_fit.tilt_deg)))
            if fit_ok and panel_fit.tilt_deg < 89.0 else None
        ),
        "aspect_ratio": _f32(aspect),
        "polygon_vintage": polygon_vintage,
        "lidar_date": lidar_date,
        "lidar_date_source": lidar_date_source,
        "vintage_gap_days": None if gap_days is None else np.int32(gap_days),
        "geometry_basis": basis,
        "n_points_panel": int(panel_fit.n_total),
        "n_inliers_panel": int(panel_fit.n_inliers),
        "point_density": np.float32(density),
        "panel_tilt_deg": _f32(panel_fit.tilt_deg),
        "panel_azimuth_deg": _f32(panel_fit.azimuth_deg),
        "panel_rmse_m": _f32(panel_fit.rmse),
        "panel_tilt_unc_deg": _f32(tilt_unc),
        "panel_azimuth_unc_deg": _f32(az_unc),
        "n_planes_detected": np.int8(2 if secondary is not None else (1 if fit_ok else 0)),
        "secondary_tilt_deg": _f32(secondary.tilt_deg) if secondary is not None else None,
        "secondary_azimuth_deg": _f32(secondary.azimuth_deg) if secondary is not None else None,
        "roof_ref_source": roof_res.source,
        "roof_tilt_deg": _f32(roof_res.fit.tilt_deg) if roof_res.fit is not None else None,
        "roof_azimuth_deg": _f32(roof_res.fit.azimuth_deg) if roof_res.fit is not None else None,
        "roof_rmse_m": _f32(roof_res.fit.rmse) if roof_res.fit is not None else None,
        "panel_roof_angle_deg": _f32(pra),
        "height_above_roof_m": _f32(har),
        "height_above_ground_m": _f32(hag),   # null when unknown (schema nullable)
        "on_building": None if roof_res.on_building is None else bool(roof_res.on_building),
        "building_id": roof_res.building_id,
        "flags": flags,
        "lidar_tile_ids": list(contributing_tile_ids),
        "pkg_version": __version__,
        "config_hash": config_hash,
        "run_id": run_id,
        "partition_id": np.int32(partition_id),
    }

    if cfg.mounting_rules.enabled:
        row.update(
            _mounting_fields(
                cfg, flags, polygon=polygon, panel_fit=panel_fit, panel_z=panel_z,
                ground_xyz=ground_xyz, roof_res=roof_res,
                roof_plane_available=roof_plane_available,
                pra=pra, har=har, hag=hag, area_m2=area_m2, aspect=aspect,
            )
        )
    return row


def _mounting_fields(
    cfg: PVGeomConfig,
    flags: list[str],
    *,
    polygon: Any,
    panel_fit: Any,
    panel_z: np.ndarray,
    ground_xyz: np.ndarray,
    roof_res: Any,
    roof_plane_available: bool,
    pra: float,
    har: float,
    hag: float,
    area_m2: float,
    aspect: float,
) -> dict[str, Any]:
    """EXPERIMENTAL mounting classification (archived in 0.2.0; only runs when
    ``mounting_rules.enabled``). Appends its own flags to ``flags`` in place."""
    on_building = bool(roof_res.on_building)

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
        not on_building
        and not canopy_evidence
        and not np.isnan(hag)
        and hag > cfg.mounting_rules.missing_footprint_hag_m
    ):
        flags.append("possible_missing_footprint")

    # Tracker-suspected flag (per-polygon heuristic; PRD §7.2 v1).
    # NaN HAG/tilt propagate — is_tracker_suspected returns False on NaN,
    # which is the correct "unknown" behavior (was coerced to 0.0 pre-0.3).
    if is_tracker_suspected(
        on_building=on_building,
        aspect_ratio=aspect,
        height_above_ground_m=hag,
        panel_tilt_deg=panel_fit.tilt_deg,
    ):
        flags.append("tracker_suspected")

    # NaN HAG propagates: the rules treat unknown heights as non-evidence and
    # fall through to ambiguous rather than classifying at "ground level". The
    # rules only use roof features from a footprint ring — an open-ring
    # reference exists for the standoff screen, not as evidence of a building.
    use_roof = roof_plane_available and on_building
    feats = MountingFeatures(
        on_building=on_building,
        panel_tilt_deg=panel_fit.tilt_deg,
        panel_azimuth_deg=panel_fit.azimuth_deg,
        panel_roof_angle_deg=pra if use_roof else float("nan"),
        height_above_roof_m=har if use_roof else float("nan"),
        height_above_ground_m=hag,
        area_m2=area_m2,
        aspect_ratio=aspect,
        roof_plane_available=use_roof,
        roof_tilt_deg=float(roof_res.fit.tilt_deg) if use_roof else None,
        east_west_rack="east_west_rack" in flags,
        n_ground_under=n_ground_under,
        ground_under_gap_m=ground_under_gap,
        no_panel_standoff=use_roof and "no_panel_standoff" in flags,
    )
    mr = classify_mounting(feats, cfg.mounting_rules)
    return {
        "mounting_type": str(mr.label),
        "mounting_confidence": np.float32(mr.confidence),
        "mounting_rule": str(mr.triggered_rule),
    }


def _f32(v) -> Any:
    """np.float32 with NaN passthrough as None (so pa null is honored)."""
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    return np.float32(v)


def _lidar_date_for(vintage, declared: date | None) -> tuple[date | None, str | None]:
    """The capture date to stamp on a tile's rows, and where it came from."""
    if declared is not None:
        return declared, "declared"
    if vintage is None:
        return None, None
    if vintage.flight_end is not None:
        return vintage.flight_end, "gps_time"
    if vintage.creation_date is not None:
        return vintage.creation_date, "header_date"
    return None, None


def process_tile_group(
    tile_uri_map: dict[str, str],
    primary_tile_id: str,
    polygons: gpd.GeoDataFrame,
    fetch_tile_ids: tuple[str, ...],
    footprints: gpd.GeoDataFrame | None,
    cfg: PVGeomConfig,
    *,
    config_hash: str,
    run_id: str,
    partition_id: int,
    polygon_id_col: str = "polygon_id",
    polygon_vintage: date | None = None,
    lidar_date: date | None = None,
) -> pa.Table:
    """Fetch all tiles in ``fetch_tile_ids``, fit every polygon, return a table.

    ``polygon_vintage`` is the run-level imagery date (a per-row
    ``polygon_vintage`` column on ``polygons`` overrides it). ``lidar_date`` is
    a *declared* capture date; when None each row gets the flight date measured
    from its primary tile's GPS time.
    """
    schema = output_schema(cfg.mounting_rules.enabled)
    cls_cfg = cfg.io.classification

    # 1) Load the tiles one at a time, keeping only the returns that can matter
    # (ground, and the two panel-candidate classes) so peak memory tracks those
    # rather than every column of every tile. Missing tiles are tolerated so
    # the runner doesn't have to pre-screen the whole bucket; if the *primary*
    # tile is missing we emit no rows because a tile group's polygons live on
    # its primary tile by construction.
    primary_vintage = None
    primary_loaded = False
    ground_chunks: list[np.ndarray] = []
    primary_cls_chunks: list[np.ndarray] = []
    fallback_cls_chunks: list[np.ndarray] = []
    for tid in fetch_tile_ids:
        uri = tile_uri_map.get(tid)
        if uri is None:
            continue
        try:
            data = read_tile(uri, reader=cfg.io.lidar_reader)
        except RemoteFileMissing:
            print(f"[tile_task] {uri} missing; skipping")
            continue
        pts = data.points
        cls = pts[:, 3].astype(np.int16)
        ground_chunks.append(pts[cls == cls_cfg.ground_class][:, :3])
        primary_cls_chunks.append(pts[cls == cls_cfg.panel_class_primary][:, :3])
        fallback_cls_chunks.append(pts[cls == cls_cfg.panel_class_fallback][:, :3])
        if tid == primary_tile_id:
            primary_loaded = True
            primary_vintage = data.vintage
        del pts, cls, data              # release the raw tile before the next read

    if not primary_loaded:
        print(f"[tile_task] primary tile {primary_tile_id} unavailable; "
              f"emitting empty table for partition {partition_id}")
        return pa.table({f.name: [] for f in schema}, schema=schema)

    def _cat(chunks: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3))

    # 2) Choose the panel-candidate pool ONCE for the whole group: the primary
    # class when the tiles carry it, else the fallback class above local ground.
    ground_xyz = _cat(ground_chunks)
    panel_pts_all = _cat(primary_cls_chunks)
    if len(panel_pts_all) == 0:
        panel_pts_all = _above_ground(_cat(fallback_cls_chunks), ground_xyz, cfg)
    del ground_chunks, primary_cls_chunks, fallback_cls_chunks

    # 3) Index both pools so each polygon only touches the points around it.
    panel_grid = PointGrid(panel_pts_all)
    ground_grid = PointGrid(ground_xyz)
    del panel_pts_all, ground_xyz

    contributing_tile_ids = tuple(t for t in fetch_tile_ids if tile_uri_map.get(t))
    row_lidar_date, row_lidar_source = _lidar_date_for(primary_vintage, lidar_date)

    roof_pad = cfg.roof_plane.buffer_max_m + 1.0
    ground_pad = max(cfg.heights.ground_search_radius_m,
                     cfg.heights.ground_fallback_max_radius_m)
    geoms = polygons.geometry.to_numpy()
    neighbours = shapely.STRtree(geoms)
    has_parent_col = "parent_polygon_id" in polygons.columns
    has_row_col = "input_row" in polygons.columns
    has_vintage_col = "polygon_vintage" in polygons.columns

    rows: list[dict[str, Any]] = []
    for i, (_, row) in enumerate(polygons.iterrows()):
        poly = row.geometry
        pid = str(row[polygon_id_col])

        in_panel = clip_points_to_polygon(
            panel_grid.query_polygon_bounds(poly),
            poly,
            erosion_m=cfg.panel_plane.erosion_m,
        )

        # Other PV polygons close enough to intrude on this one's roof ring.
        near = neighbours.query(poly.buffer(roof_pad))
        other_pvs = gpd.GeoDataFrame(
            geometry=[geoms[j] for j in near if j != i], crs=polygons.crs
        )

        row_vintage = polygon_vintage
        if has_vintage_col and row["polygon_vintage"] is not None and not _is_nat(row["polygon_vintage"]):
            row_vintage = row["polygon_vintage"]

        rows.append(
            _build_row(
                polygon=poly,
                polygon_id=pid,
                parent_polygon_id=str(row["parent_polygon_id"]) if has_parent_col else pid,
                input_row=int(row["input_row"]) if has_row_col else i,
                cfg=cfg,
                config_hash=config_hash,
                run_id=run_id,
                partition_id=partition_id,
                panel_pts=in_panel,
                ground_xyz=ground_grid.query_polygon_bounds(poly, pad_m=ground_pad),
                roof_input_pts=panel_grid.query_polygon_bounds(poly, pad_m=roof_pad),
                footprints=footprints,
                other_pv_polygons=other_pvs,
                contributing_tile_ids=contributing_tile_ids,
                polygon_vintage=row_vintage,
                lidar_date=row_lidar_date,
                lidar_date_source=row_lidar_source,
            )
        )

    cols = {f.name: [r.get(f.name) for r in rows] for f in schema}
    return pa.table(cols, schema=schema)


def _is_nat(v: Any) -> bool:
    try:
        return bool(v != v)             # NaN / NaT are the only values unequal to themselves
    except Exception:
        return False
