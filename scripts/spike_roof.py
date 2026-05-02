# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "geopandas>=0.14",
#   "shapely>=2.0",
#   "pyproj>=3.6",
#   "laspy[lazrs]>=2.5",
#   "numpy>=1.26",
#   "matplotlib>=3.9",
#   "boto3>=1.35",
#   "botocore[crt]>=1.35",
#   "pyarrow>=15",
#   "pillow>=10",
#   "pv-geom",
# ]
#
# [tool.uv.sources]
# pv-geom = { path = "..", editable = true }
# ///
"""End-to-end M4 validation: FEMA AZ footprints + atlas polygon + cached LiDAR.

Loads `s3://free-research-data/national/fema_footprints/az.geoparquet`
(downloaded once, cached locally), bbox-filters around the target polygon,
and runs ``extract_roof_plane`` and the height helpers.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import boto3
import geopandas as gpd
import laspy
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
from shapely import contains_xy
from shapely.geometry import box

from pv_geom.classify.interface import MountingFeatures
from pv_geom.classify.rules import classify_mounting
from pv_geom.config import (
    HeightsConfig,
    MountingRulesConfig,
    MultiPlaneConfig,
    PanelPlaneConfig,
    RoofPlaneConfig,
)
from pv_geom.geometry.heights import (
    height_above_ground,
    height_above_roof,
    panel_roof_angle_deg,
)
from pv_geom.geometry.multi_plane import (
    detect_multi_plane,
    is_tracker_suspected,
    polygon_aspect_ratio,
)
from pv_geom.geometry.plane_fit import fit_plane_ransac
from pv_geom.geometry.roof_plane import extract_roof_plane

# --------------------------------------------------------------------------- #
# Constants — same conventions as scripts/spike.py
# --------------------------------------------------------------------------- #

POLYGONS_PARQUET = Path(r"C:\Users\job_t\code\free\pv_sam3\artifacts\atlas\latest.parquet")
FEMA_AZ_S3 = ("free-research-data", "national/fema_footprints/az.geoparquet")
CACHE = Path(tempfile.gettempdir()) / "pv_geom_spike_cache"
CACHE.mkdir(exist_ok=True)
OUT = Path(__file__).parent / "eda_outputs"
OUT.mkdir(exist_ok=True)
LAZ_CRS = 6341
DEFAULT_POLYGON = "21_840435_397729__1"  # 26 m^2, 99.95% inside FEMA footprint
DEFAULT_TILE = ("asu-nsf-phoenix",
                "data/lidar_data/USGS_LPC_AZ_MaricopaPinal_2020_B20_w0432n3719.laz")


def _s3_download(bucket: str, key: str) -> Path:
    local = CACHE / Path(key).name
    if not local.exists():
        print(f"# downloading s3://{bucket}/{key} ...")
        boto3.client("s3").download_file(bucket, key, str(local))
        print(f"  {local.stat().st_size / 1e6:.1f} MB cached at {local}")
    return local


def main() -> int:
    polygon_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_POLYGON

    # 1) Polygon -----------------------------------------------------------
    print(f"# polygon: {polygon_id}")
    gdf = gpd.read_parquet(
        POLYGONS_PARQUET,
        columns=["detection_id", "geometry"],
    )
    poly_wgs = gdf[gdf["detection_id"] == polygon_id].iloc[0].geometry
    poly_proj = gpd.GeoSeries([poly_wgs], crs=4326).to_crs(LAZ_CRS).iloc[0]
    print(f"  area: {poly_proj.area:.1f} m^2  centroid (UTM): "
          f"({poly_proj.centroid.x:.1f}, {poly_proj.centroid.y:.1f})")

    # 2) Footprints (bbox-filtered, in LAZ CRS) ----------------------------
    fema_local = _s3_download(*FEMA_AZ_S3)
    cx, cy = poly_proj.centroid.x, poly_proj.centroid.y
    bbox_proj = (cx - 200, cy - 200, cx + 200, cy + 200)
    # FEMA file is in WGS84; convert our local-CRS bbox back to lon/lat
    tr_to_wgs = Transformer.from_crs(LAZ_CRS, 4326, always_xy=True)
    lons, lats = tr_to_wgs.transform([bbox_proj[0], bbox_proj[2]],
                                     [bbox_proj[1], bbox_proj[3]])
    bbox_wgs = (min(lons), min(lats), max(lons), max(lats))
    # FEMA AZ geoparquet has no bbox covering column, so we read once and filter
    # in-memory. Cached locally; ~5 s on subsequent runs.
    print("# loading FEMA AZ footprints (no bbox column; reading full file)")
    fema_full = gpd.read_parquet(fema_local)
    print(f"  total footprints: {len(fema_full):,}; cols: {list(fema_full.columns)}")
    bbox_geom = box(*bbox_wgs)
    footprints_wgs = fema_full[fema_full.geometry.intersects(bbox_geom)].copy()
    print(f"  in target bbox:   {len(footprints_wgs)}")
    # Normalize FEMA's `build_id` to our canonical `building_id` column.
    if "building_id" not in footprints_wgs.columns:
        if "build_id" in footprints_wgs.columns:
            footprints_wgs = footprints_wgs.assign(
                building_id=footprints_wgs["build_id"].astype(str)
            )
        else:
            footprints_wgs = footprints_wgs.assign(
                building_id=[f"auto_{i}" for i in range(len(footprints_wgs))]
            )
    footprints = footprints_wgs.to_crs(LAZ_CRS)

    # 3) Other PV polygons in the same area --------------------------------
    poly_buf = poly_proj.buffer(50)
    poly_buf_wgs = gpd.GeoSeries([poly_buf], crs=LAZ_CRS).to_crs(4326).iloc[0]
    nearby_pv = gdf[gdf.geometry.intersects(poly_buf_wgs)]
    nearby_pv = nearby_pv[nearby_pv["detection_id"] != polygon_id]
    print(f"  {len(nearby_pv)} other PV polygons within 50 m of the target")
    others = nearby_pv.to_crs(LAZ_CRS)[["geometry"]].reset_index(drop=True)

    # 4) LiDAR -------------------------------------------------------------
    laz_local = _s3_download(*DEFAULT_TILE)
    print(f"# reading LAZ {laz_local.name}")
    with laspy.open(str(laz_local)) as src:
        las = src.read()
    pts_all = np.column_stack([las.x, las.y, las.z, las.classification])
    cls = pts_all[:, 3].astype(int)
    print(f"  {len(pts_all):,} points, classes: {dict(zip(*np.unique(cls, return_counts=True)))}")

    # Panel-class fallback (PRD §7.3 + spike findings)
    primary = 6 if (cls == 6).any() else 1
    panel_pts = pts_all[cls == primary]
    print(f"  using class {primary} for panel + roof candidates ({len(panel_pts):,} pts)")
    if primary == 1:
        ground = pts_all[cls == 2]
        near = ground[(np.abs(ground[:, 0] - cx) < 50) & (np.abs(ground[:, 1] - cy) < 50)]
        ground_z = float(np.median(near[:, 2]))
        panel_pts = panel_pts[panel_pts[:, 2] > ground_z + 1.5]
        print(f"  ground_z (median): {ground_z:.2f}; HAG>1.5m filter -> "
              f"{len(panel_pts):,} pts")

    # 5) Panel fit (M3) ----------------------------------------------------
    in_panel = panel_pts[contains_xy(poly_proj, panel_pts[:, 0], panel_pts[:, 1])]
    print(f"\n# panel fit  (n={len(in_panel)})")
    cfg_panel = PanelPlaneConfig()
    panel_fit = fit_plane_ransac(
        in_panel[:, :3],
        ransac_threshold=cfg_panel.ransac_threshold_m,
        min_inlier_frac=cfg_panel.min_inlier_frac,
        max_iter=cfg_panel.max_iter,
        seed=hash(polygon_id) & 0xFFFFFFFF,
    )
    print(f"  tilt={panel_fit.tilt_deg:.2f} az={panel_fit.azimuth_deg:.2f} "
          f"rmse={panel_fit.rmse:.3f} inliers={panel_fit.n_inliers}/{panel_fit.n_total}")

    # 6) Roof fit (M4) -----------------------------------------------------
    cfg_roof = RoofPlaneConfig()
    print(f"\n# roof fit  (cfg buffer={cfg_roof.buffer_m}-{cfg_roof.buffer_max_m} m, "
          f"min_points={cfg_roof.min_points})")
    roof_res = extract_roof_plane(
        poly_proj, footprints, others, panel_pts[:, :3], cfg_roof,
        seed=hash(polygon_id) & 0xFFFFFFFF,
    )
    print(f"  on_building={roof_res.on_building} building_id={roof_res.building_id} "
          f"flag={roof_res.flag} used_buffer={roof_res.used_buffer_m}")
    if roof_res.fit is not None:
        rf = roof_res.fit
        print(f"  roof tilt={rf.tilt_deg:.2f} az={rf.azimuth_deg:.2f} "
              f"rmse={rf.rmse:.3f} inliers={rf.n_inliers}/{rf.n_total}")

    # 6.5) Multi-plane detection on panel returns (M5) --------------------
    mp = detect_multi_plane(in_panel[:, :3], panel_fit, MultiPlaneConfig(),
                            seed=hash(polygon_id) & 0xFFFFFFFF)
    print(f"\n# multi-plane: secondary={'yes' if mp.secondary else 'no'} flags={mp.flags}")
    if mp.secondary is not None:
        print(f"  secondary tilt={mp.secondary.tilt_deg:.2f} "
              f"az={mp.secondary.azimuth_deg:.2f}")

    # 7) Heights + panel-roof angle ----------------------------------------
    cfg_heights = HeightsConfig()
    panel_inliers = in_panel[panel_fit.inlier_mask] if panel_fit.n_inliers > 0 else in_panel
    hag = height_above_ground(
        panel_inliers[:, 2],
        pts_all[cls == 2],
        (cx, cy),
        cfg_heights.ground_search_radius_m,
    )
    print(f"\n# heights")
    print(f"  height_above_ground: {hag:.2f} m")
    if roof_res.fit is not None:
        har = height_above_roof(panel_inliers, roof_res.fit)
        pra = panel_roof_angle_deg(panel_fit, roof_res.fit)
        print(f"  height_above_roof:   {har:.2f} m")
        print(f"  panel_roof_angle:    {pra:.2f} deg "
              f"(<5 -> flush_mount; >5 -> tilted_rack)")
    else:
        har = float("nan")
        pra = float("nan")

    # 8) Mounting classification (M5) -------------------------------------
    aspect = polygon_aspect_ratio(poly_proj)
    feats = MountingFeatures(
        on_building=roof_res.on_building,
        panel_tilt_deg=panel_fit.tilt_deg,
        panel_azimuth_deg=panel_fit.azimuth_deg,
        panel_roof_angle_deg=pra,
        height_above_roof_m=har,
        height_above_ground_m=hag,
        area_m2=poly_proj.area,
        aspect_ratio=aspect,
        roof_plane_available=(roof_res.fit is not None and not np.isnan(roof_res.fit.tilt_deg)),
    )
    mr = classify_mounting(feats, MountingRulesConfig())
    print()
    print("# classify (M5)")
    print(f"  features: aspect_ratio={aspect:.2f}  roof_plane_available="
          f"{feats.roof_plane_available}")
    print(f"  -> mounting_type={mr.label}  rule={mr.triggered_rule}  "
          f"confidence={mr.confidence:.3f}")
    if is_tracker_suspected(
        on_building=feats.on_building, aspect_ratio=aspect,
        height_above_ground_m=hag, panel_tilt_deg=panel_fit.tilt_deg,
    ):
        print("  flag: tracker_suspected")

    # 9) Plot -------------------------------------------------------------
    plot_path = OUT / f"spike_roof_{polygon_id.replace('/','_')}.png"
    _plot(
        poly_wgs, footprints, others, panel_inliers, roof_res, panel_fit,
        plot_path, polygon_id,
    )
    print(f"\n# wrote {plot_path}")
    return 0


def _plot(
    poly_wgs, footprints, others, panel_inliers, roof_res, panel_fit, out, polygon_id
):
    sys.path.insert(0, str(Path(__file__).parent))
    from _aerial import aerial_basemap

    lon_min, lat_min, lon_max, lat_max = poly_wgs.bounds
    margin = 0.0004    # ~35 m to show the building + neighborhood
    bbox = (lon_min - margin, lat_min - margin, lon_max + margin, lat_max + margin)
    img, extent = aerial_basemap(bbox, target_lod=13)

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 8.5), dpi=140)
    ax.imshow(img, extent=extent, origin="upper", interpolation="bilinear")

    def _outline(geom, ax, **kw):
        """Plot the outer boundary of a Polygon or MultiPolygon."""
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for p in polys:
            xs, ys = p.exterior.xy
            ax.plot(xs, ys, **kw)

    # Footprints in WGS84 — outlines only
    fp_wgs = footprints.to_crs(4326)
    chosen_id = roof_res.building_id
    for _, r in fp_wgs.iterrows():
        is_chosen = r.get("building_id") == chosen_id
        _outline(
            r.geometry, ax,
            color="#3DBDFF" if is_chosen else "#88AACC",
            lw=2.0 if is_chosen else 0.9,
            alpha=0.95 if is_chosen else 0.5,
        )

    # Other PVs
    for g in others.to_crs(4326).geometry:
        _outline(g, ax, color="#FFD23A", lw=0.9, alpha=0.7)

    # Target polygon
    x, y = poly_wgs.exterior.xy
    ax.plot(x, y, color="#FF2D2D", lw=2.4,
            path_effects=[pe.withStroke(linewidth=4.5, foreground="white")])

    # LiDAR panel inliers
    if len(panel_inliers):
        tr = Transformer.from_crs(LAZ_CRS, 4326, always_xy=True)
        plon, plat = tr.transform(panel_inliers[:, 0], panel_inliers[:, 1])
        ax.scatter(plon, plat, c=panel_inliers[:, 2], s=6, cmap="viridis",
                   alpha=0.65, edgecolors="none")

    title_lines = [f"polygon {polygon_id}"]
    if not np.isnan(panel_fit.tilt_deg):
        title_lines.append(
            f"panel: tilt={panel_fit.tilt_deg:.1f} deg, az={panel_fit.azimuth_deg:.1f} deg, "
            f"rmse={panel_fit.rmse*100:.1f} cm"
        )
    if roof_res.fit is not None:
        title_lines.append(
            f"roof:  tilt={roof_res.fit.tilt_deg:.1f} deg, az={roof_res.fit.azimuth_deg:.1f} deg, "
            f"rmse={roof_res.fit.rmse*100:.1f} cm  (buf={roof_res.used_buffer_m} m)"
        )
    elif roof_res.on_building:
        title_lines.append(f"roof:  not extracted ({roof_res.flag})")
    else:
        title_lines.append("roof:  off-building")
    ax.set_title("\n".join(title_lines), fontsize=11)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.tick_params(labelsize=8)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color="#FF2D2D", lw=2.4, label="target PV polygon"),
        Line2D([], [], color="#3DBDFF", lw=2.0, label="chosen building footprint"),
        Line2D([], [], color="#88AACC", lw=0.9, label="other footprints"),
        Line2D([], [], color="#FFD23A", lw=0.9, label="other PV polygons (subtracted)"),
        Line2D([], [], marker="o", color="w", markerfacecolor="green",
               markersize=5, label="panel inliers (z-coloured)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
