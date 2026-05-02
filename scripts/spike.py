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
#   "fsspec>=2024.10",
#   "s3fs>=2024.10",
#   "pyarrow>=15",
#   "pillow>=10",
# ]
# ///
"""LiDAR data-quality spike for pv_geom.

Two phases driven by CLI flags:

  --discover     List the LiDAR S3 prefix, look for a tile_index, summarize.
                 Run this FIRST; it tells us the bucket layout and CRS.

  (default)      Spike: pick one polygon, locate the covering tile, fetch it,
                 clip class-6 points, fit a quick plane, plot.

The spike is intentionally throwaway. Goal: answer "is the LiDAR resolution
high enough to recover panel tilt/azimuth on residential roofs?" in one
session, before committing to M2/M3.

Usage:
  uv run scripts/spike.py --discover
  uv run scripts/spike.py
  uv run scripts/spike.py --polygon-id 21_840480_397711__0 --tile-uri s3://...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import boto3
import geopandas as gpd
import laspy
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from botocore.exceptions import ClientError, NoCredentialsError
from shapely.geometry import Point, Polygon

# --------------------------------------------------------------------------- #
# Defaults — change here, not via CLI, when iterating
# --------------------------------------------------------------------------- #

POLYGONS_PARQUET = Path(r"C:\Users\job_t\code\free\pv_sam3\artifacts\atlas\latest.parquet")
LIDAR_BUCKET = "asu-nsf-phoenix"
LIDAR_PREFIX = "data/lidar_data/"
DEFAULT_POLYGON_ID = "21_840480_397711__0"   # score 0.98, ~67 m^2, east valley
OUT = Path(__file__).parent / "eda_outputs"
OUT.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# Phase 1 — discover bucket layout
# --------------------------------------------------------------------------- #

def discover() -> int:
    """List the LiDAR prefix and print structure. Looks for tile_index.parquet."""
    print(f"# Listing s3://{LIDAR_BUCKET}/{LIDAR_PREFIX}")
    s3 = boto3.client("s3")
    try:
        # Top-level keys + common prefixes (1 level deep)
        resp = s3.list_objects_v2(
            Bucket=LIDAR_BUCKET, Prefix=LIDAR_PREFIX, Delimiter="/", MaxKeys=200
        )
    except (NoCredentialsError, ClientError) as exc:
        print(f"# S3 access failed: {exc}", file=sys.stderr)
        print("# Refresh your AWS session and retry.", file=sys.stderr)
        return 2

    common = [p["Prefix"] for p in resp.get("CommonPrefixes", [])]
    files = [(c["Key"], c["Size"]) for c in resp.get("Contents", [])]
    print(f"  subdirs: {len(common)}")
    for p in common[:20]:
        print(f"    {p}")
    print(f"  files at this level: {len(files)}")
    for k, sz in files[:20]:
        print(f"    {k}\t{sz:>12,}")

    # Sniff for a tile index
    candidates = [k for k, _ in files if k.endswith(("tile_index.parquet", "index.parquet", "tiles.parquet"))]
    print(f"\n# tile_index candidates at top-level: {len(candidates)}")
    for k in candidates:
        print(f"  s3://{LIDAR_BUCKET}/{k}")

    # If nothing at top level, peek into the first subdir to see file naming
    if common and not files:
        sub = common[0]
        print(f"\n# Peeking into first subdir: s3://{LIDAR_BUCKET}/{sub}")
        sub_resp = s3.list_objects_v2(Bucket=LIDAR_BUCKET, Prefix=sub, MaxKeys=10)
        for c in sub_resp.get("Contents", []):
            print(f"  {c['Key']}\t{c['Size']:>12,}")

    # Sample one LAZ header to learn its CRS
    laz_keys = [k for k, _ in files if k.lower().endswith((".laz", ".las"))]
    if not laz_keys:
        # search subdirs lightly
        for sub in common[:3]:
            r = s3.list_objects_v2(Bucket=LIDAR_BUCKET, Prefix=sub, MaxKeys=5)
            laz_keys.extend(c["Key"] for c in r.get("Contents", []) if c["Key"].lower().endswith((".laz", ".las")))
            if laz_keys:
                break

    if laz_keys:
        sample = laz_keys[0]
        uri = f"s3://{LIDAR_BUCKET}/{sample}"
        print(f"\n# Sniffing header of one LAZ: {uri}")
        sniff_laz_header(uri)
    else:
        print("\n# No LAZ files found in the prefix sampled.")
    return 0


def sniff_laz_header(uri: str) -> None:
    """Read just the LAZ header (no point load) to learn CRS, bbox, point count."""
    import s3fs

    fs = s3fs.S3FileSystem(anon=False)
    with fs.open(uri, "rb") as f:
        with laspy.open(f) as src:
            h = src.header
            print(f"  point_count: {h.point_count:,}")
            print(f"  point_format: {h.point_format.id}")
            print(f"  scales: {h.scales}")
            print(f"  offsets: {h.offsets}")
            print(f"  mins: {h.mins}")
            print(f"  maxs: {h.maxs}")
            try:
                crs = h.parse_crs()
                print(f"  CRS: {crs}")
            except Exception as exc:
                print(f"  CRS: <unparseable: {exc}>")


# --------------------------------------------------------------------------- #
# Phase 2 — spike on one polygon
# --------------------------------------------------------------------------- #

def find_polygon(polygon_id: str) -> gpd.GeoDataFrame:
    print(f"# Loading polygons from {POLYGONS_PARQUET.name}")
    cols = ["detection_id", "geometry", "sam3_score", "bbox_wgs84_w", "bbox_wgs84_s",
            "bbox_wgs84_e", "bbox_wgs84_n"]
    gdf = gpd.read_parquet(POLYGONS_PARQUET, columns=cols)
    sub = gdf[gdf["detection_id"] == polygon_id]
    if sub.empty:
        sys.exit(f"polygon_id {polygon_id} not found")
    return sub


def find_tile_for_polygon(
    polygon_wgs84: Polygon, tile_index_uri: str | None
) -> str:
    """Return the s3:// URI of the LAZ tile covering the polygon centroid."""
    if tile_index_uri is None:
        sys.exit(
            "No tile-index URI given. Run with --discover first to locate it, "
            "or pass --tile-uri to skip the lookup and target a known tile."
        )
    print(f"# Reading tile index: {tile_index_uri}")
    tindex = gpd.read_parquet(tile_index_uri)
    if tindex.crs and tindex.crs.to_epsg() != 4326:
        polygon_wgs84_gs = gpd.GeoSeries([polygon_wgs84], crs=4326).to_crs(tindex.crs)
        probe = polygon_wgs84_gs.iloc[0].centroid
    else:
        probe = polygon_wgs84.centroid
    hit = tindex[tindex.geometry.contains(probe)]
    if hit.empty:
        sys.exit(f"No tile covers polygon centroid {probe}")
    # Try common path columns
    for col in ("tile_path", "filename", "path", "s3_uri", "uri"):
        if col in hit.columns:
            return str(hit.iloc[0][col])
    sys.exit(f"Tile index has no recognized path column. Cols: {list(hit.columns)}")


def fetch_tile_points(tile_uri: str) -> tuple[np.ndarray, str | None]:
    """Download a LAZ tile via boto3, read all points. Returns (xyzc, crs_str)."""
    import tempfile

    assert tile_uri.startswith("s3://"), f"expected s3:// URI, got {tile_uri}"
    bucket, _, key = tile_uri[len("s3://"):].partition("/")
    print(f"# Downloading tile: {tile_uri}")

    s3 = boto3.client("s3")
    cache_dir = Path(tempfile.gettempdir()) / "pv_geom_spike_cache"
    cache_dir.mkdir(exist_ok=True)
    local = cache_dir / Path(key).name
    if not local.exists():
        s3.download_file(bucket, key, str(local))
        print(f"  downloaded {local.stat().st_size / 1e6:.1f} MB to {local}")
    else:
        print(f"  using cached {local}")

    with laspy.open(str(local)) as src:
        print(f"  point_count: {src.header.point_count:,}")
        try:
            crs = src.header.parse_crs()
            crs_str = crs.to_string() if crs else None
        except Exception:
            crs_str = None
        las = src.read()
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    cls = np.asarray(las.classification, dtype=np.int16)
    pts = np.column_stack([xyz, cls])
    print(f"  loaded shape: {pts.shape}, CRS: {crs_str}")
    return pts, crs_str


def clip_to_polygon(pts: np.ndarray, polygon: Polygon) -> np.ndarray:
    """Clip (N,4) [x,y,z,class] to polygon. Uses vectorized contains via shapely 2."""
    from shapely import contains_xy

    mask = contains_xy(polygon, pts[:, 0], pts[:, 1])
    return pts[mask]


def fit_plane_svd(xyz: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Fit a plane via PCA. Returns (normal, tilt_deg, azimuth_deg, rmse)."""
    centroid = xyz.mean(axis=0)
    centered = xyz - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]                                 # smallest-variance direction
    if normal[2] < 0:
        normal = -normal                            # canonical: nz >= 0
    nx, ny, nz = normal
    tilt_deg = float(np.degrees(np.arccos(abs(nz))))
    # 0 = north, clockwise. Project normal onto horizontal, atan2(east, north).
    azimuth_deg = float(np.degrees(np.arctan2(nx, ny))) % 360.0
    residuals = centered @ normal
    rmse = float(np.sqrt((residuals ** 2).mean()))
    return normal, tilt_deg, azimuth_deg, rmse


def plot_spike(
    pts_in: np.ndarray,           # (N,4) clipped points x,y,z,class — in LAZ CRS
    polygon_proj: Polygon,
    polygon_wgs84: Polygon,
    crs_str: str | None,
    polygon_id: str,
    tilt_deg: float,
    azimuth_deg: float,
    rmse: float,
    out_path: Path,
) -> None:
    """4-panel figure: aerial+overlay (large, top-left), plan view, side view, residuals."""
    import matplotlib.patheffects as pe
    sys.path.insert(0, str(Path(__file__).parent))
    from _aerial import aerial_basemap
    from pyproj import Transformer

    fig = plt.figure(figsize=(13, 9), dpi=140)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.6, 1, 1], hspace=0.32, wspace=0.28)

    # ---------------- aerial + polygon + LiDAR (top-left, double width) ----
    lon_min, lat_min, lon_max, lat_max = polygon_wgs84.bounds
    margin_deg = 0.00018   # ~15 m at lat 33
    bbox = (lon_min - margin_deg, lat_min - margin_deg,
            lon_max + margin_deg, lat_max + margin_deg)
    print(f"  fetching aerial for bbox {bbox}")
    img, extent = aerial_basemap(bbox, target_lod=13)

    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(img, extent=extent, origin="upper", interpolation="bilinear")
    poly_lon, poly_lat = polygon_wgs84.exterior.xy
    ax0.plot(poly_lon, poly_lat, color="#FF2D2D", lw=2.2,
             path_effects=[pe.withStroke(linewidth=4.0, foreground="white")])
    if crs_str:
        tr = Transformer.from_crs(crs_str, 4326, always_xy=True)
    else:
        tr = Transformer.from_crs(6341, 4326, always_xy=True)
    plon, plat = tr.transform(pts_in[:, 0], pts_in[:, 1])
    sc0 = ax0.scatter(plon, plat, c=pts_in[:, 2], cmap="viridis", s=8,
                      alpha=0.75, edgecolors="none")
    ax0.set_xlim(extent[0], extent[1])
    ax0.set_ylim(extent[2], extent[3])
    ax0.set_xlabel("lon")
    ax0.set_ylabel("lat")
    ax0.set_title("aerial (Maricopa 2024 ortho) + polygon + LiDAR returns")
    ax0.tick_params(labelsize=8)
    fig.colorbar(sc0, ax=ax0, label="z (m)", shrink=0.7)

    # ---------------- plan view (top-middle) -------------------------------
    x0, y0 = polygon_proj.centroid.x, polygon_proj.centroid.y
    px = pts_in[:, 0] - x0
    py = pts_in[:, 1] - y0
    pz = pts_in[:, 2]
    poly_x, poly_y = polygon_proj.exterior.xy
    poly_x = np.array(poly_x) - x0
    poly_y = np.array(poly_y) - y0

    ax1 = fig.add_subplot(gs[0, 1])
    sc = ax1.scatter(px, py, c=pz, s=6, cmap="viridis")
    ax1.plot(poly_x, poly_y, "r-", lw=1.5)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x (centred, m)")
    ax1.set_ylabel("y (centred, m)")
    ax1.set_title(f"plan view — {len(pts_in)} pts")
    fig.colorbar(sc, ax=ax1, label="z (m)")

    # ---------------- side view (top-right) --------------------------------
    az_rad = np.radians(azimuth_deg)
    along = px * np.sin(az_rad) + py * np.cos(az_rad)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(along, pz, s=6, c=pz, cmap="viridis")
    ax2.set_xlabel("along-azimuth (m)")
    ax2.set_ylabel("z (m)")
    ax2.set_title(f"side view  |  tilt={tilt_deg:.1f} deg  az={azimuth_deg:.1f} deg")
    ax2.grid(alpha=0.3)

    # ---------------- residuals (bottom-middle) ----------------------------
    centroid = pts_in[:, :3].mean(axis=0)
    centered = pts_in[:, :3] - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1] * (1 if vt[-1, 2] >= 0 else -1)
    res = centered @ normal
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(res, bins=40, color="#1F3A5F")
    ax3.axvline(0, color="k", lw=0.7)
    ax3.set_xlabel("residual along normal (m)")
    ax3.set_ylabel("count")
    ax3.set_title(f"residuals  |  RMSE={rmse:.3f} m")
    ax3.grid(alpha=0.3)

    # ---------------- z histogram (bottom-right) ---------------------------
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(pts_in[:, 2], bins=40, color="#C0502C")
    ax4.set_xlabel("z (m)")
    ax4.set_ylabel("count")
    ax4.set_title("z distribution")
    ax4.grid(alpha=0.3)

    fig.suptitle(f"pv_geom spike - polygon {polygon_id}", fontsize=13)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_spike(polygon_id: str, tile_uri: str | None, tile_index: str | None) -> int:
    poly_gdf = find_polygon(polygon_id)
    polygon_wgs84 = poly_gdf.iloc[0].geometry

    if tile_uri is None:
        tile_uri = find_tile_for_polygon(polygon_wgs84, tile_index)

    pts, crs_str = fetch_tile_points(tile_uri)

    # Reproject polygon into the LAZ CRS so we can clip in native units
    if crs_str:
        poly_proj = poly_gdf.to_crs(crs_str).iloc[0].geometry
    else:
        # Last-resort fallback: assume LAZ matches PRD default (EPSG:6404, ftUS)
        print("  WARNING: LAZ has no CRS; assuming EPSG:6404 (Arizona Central, ftUS)")
        poly_proj = poly_gdf.to_crs("EPSG:6404").iloc[0].geometry

    print(f"  polygon (proj) bounds: {poly_proj.bounds}")
    print(f"  polygon (proj) area:   {poly_proj.area:.2f} (sq units of LAZ CRS)")

    # Class histogram for visibility — many USGS LPC tiles are ground-only classified.
    import collections
    counts = collections.Counter(pts[:, 3].astype(int).tolist())
    print(f"  class histogram: {dict(sorted(counts.items()))}")

    # Per-class candidates for panel surface, in order of preference:
    #   class 6 (building) when present; otherwise class 1 (unclassified, includes
    #   buildings + veg + PV) clipped above local ground.
    primary_class = 6 if counts.get(6, 0) > 0 else 1
    print(f"  using class {primary_class} for panel fit")
    panel_pts = pts[pts[:, 3] == primary_class]

    # Clip
    in_poly = clip_to_polygon(panel_pts, poly_proj)
    print(f"  in-polygon points (class {primary_class}): {len(in_poly):,}")

    # If we used class 1, drop points below local ground + 1.5 m to remove
    # vegetation / driveway returns that happen to fall inside the polygon.
    if primary_class == 1 and len(in_poly):
        ground = pts[pts[:, 3] == 2]
        if len(ground):
            cx, cy = poly_proj.centroid.x, poly_proj.centroid.y
            near_ground = ground[
                (np.abs(ground[:, 0] - cx) < 25) & (np.abs(ground[:, 1] - cy) < 25)
            ]
            if len(near_ground) >= 5:
                ground_z = float(np.median(near_ground[:, 2]))
                before = len(in_poly)
                in_poly = in_poly[in_poly[:, 2] > ground_z + 1.5]
                print(f"  ground-z (median, 25m square): {ground_z:.2f}")
                print(f"  after height-above-ground>1.5m filter: {len(in_poly):,} (was {before})")
    if len(in_poly) < 30:
        print("  WARNING: very few points — plane fit will be unreliable")

    # Density (need polygon area in m^2; convert if CRS is in feet)
    crs_lower = (crs_str or "EPSG:6404").lower()
    is_feet = "ftus" in crs_lower or "us-ft" in crs_lower or "_ftus" in crs_lower or "6404" in crs_lower
    poly_area_m2 = poly_proj.area * (0.3048 ** 2) if is_feet else poly_proj.area
    print(f"  polygon area: {poly_area_m2:.2f} m^2")
    print(f"  density:      {len(in_poly) / poly_area_m2:.2f} pts/m^2")

    if len(in_poly) < 4:
        sys.exit("Not enough points to fit a plane.")

    normal, tilt_deg, azimuth_deg, rmse = fit_plane_svd(in_poly[:, :3])
    print()
    print(f"  fitted normal: {normal}")
    print(f"  tilt_deg:      {tilt_deg:.2f}")
    print(f"  azimuth_deg:   {azimuth_deg:.2f}  (0=N, 90=E, 180=S, 270=W)")
    print(f"  rmse:          {rmse:.3f} (units of LAZ CRS)")
    if is_feet:
        print(f"  rmse (m):      {rmse * 0.3048:.3f}")

    out_path = OUT / f"spike_{polygon_id.replace('/','_')}.png"
    plot_spike(in_poly, poly_proj, polygon_wgs84, crs_str,
               polygon_id, tilt_deg, azimuth_deg, rmse, out_path)
    print(f"\n# wrote {out_path}")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--discover", action="store_true",
                   help="List the LiDAR S3 prefix and inspect a sample header.")
    p.add_argument("--polygon-id", default=DEFAULT_POLYGON_ID,
                   help="detection_id from the polygon parquet.")
    p.add_argument("--tile-uri", default=None,
                   help="Direct s3:// URI of the LAZ tile (skips tile-index lookup).")
    p.add_argument("--tile-index", default=None,
                   help="s3:// URI of a tile-index GeoParquet (used to locate tile).")
    args = p.parse_args()

    if args.discover:
        return discover()
    return run_spike(args.polygon_id, args.tile_uri, args.tile_index)


if __name__ == "__main__":
    raise SystemExit(main())
