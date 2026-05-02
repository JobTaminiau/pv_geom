# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "geopandas>=0.14",
#   "shapely>=2.0",
#   "pyproj>=3.6",
#   "laspy[lazrs]>=2.5",
#   "numpy>=1.26",
#   "matplotlib>=3.9",
#   "pandas>=2.2",
#   "boto3>=1.35",
#   "botocore[crt]>=1.35",
#   "pyarrow>=15",
# ]
# ///
"""Extended spike validation.

Two subcommands:

  batch        Fit N polygons sampled across the already-cached tiles,
               stratified by area, write a parquet + 4-panel summary plot.

  size-sweep   For one well-fit polygon, repeatedly subsample its inliers
               at decreasing N and plot how the tilt/azimuth estimate
               spreads. Answers: "what's the smallest polygon (in points,
               and at ~10 pts/m^2 thus in m^2) where the fit is reliable?"

Run from the project root:
  uv run scripts/spike_validate.py batch
  uv run scripts/spike_validate.py size-sweep
"""

from __future__ import annotations

import argparse
import collections
import sys
import tempfile
from pathlib import Path

import boto3
import geopandas as gpd
import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import contains_xy
from shapely.geometry import Polygon

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

POLYGONS_PARQUET = Path(r"C:\Users\job_t\code\free\pv_sam3\artifacts\atlas\latest.parquet")
LIDAR_BUCKET = "asu-nsf-phoenix"
LIDAR_PREFIX = "data/lidar_data/"
LAZ_NAME_FMT = "USGS_LPC_AZ_MaricopaPinal_2020_B20_{tile}.laz"
LAZ_CRS = 6341  # NAD83(2011) / UTM 12N, meters
CACHE = Path(tempfile.gettempdir()) / "pv_geom_spike_cache"
CACHE.mkdir(exist_ok=True)
OUT = Path(__file__).parent / "eda_outputs"
OUT.mkdir(exist_ok=True)

# Already-cached tiles from prior spike runs:
CACHED_TILES = [
    "w0432n3719",  # east valley
    "w0431n3698",  # Tempe
    "w0419n3696",  # Glendale
    "w0417n3684",  # Chandler
    "w0418n3678",  # Maricopa city / SE
]

# --------------------------------------------------------------------------- #
# Tile + point I/O
# --------------------------------------------------------------------------- #

def fetch_tile(tile_name: str) -> Path:
    """Download a tile to local cache if not present; return local path."""
    fname = LAZ_NAME_FMT.format(tile=tile_name)
    local = CACHE / fname
    if local.exists():
        return local
    s3 = boto3.client("s3")
    key = f"{LIDAR_PREFIX}{fname}"
    print(f"  downloading {tile_name}...")
    s3.download_file(LIDAR_BUCKET, key, str(local))
    print(f"    {local.stat().st_size / 1e6:.0f} MB")
    return local


def load_tile_points(local_path: Path) -> np.ndarray:
    """Read all points from a local LAZ as (N, 4) [x, y, z, class]."""
    with laspy.open(str(local_path)) as src:
        las = src.read()
    return np.column_stack(
        [np.asarray(las.x, np.float64),
         np.asarray(las.y, np.float64),
         np.asarray(las.z, np.float64),
         np.asarray(las.classification, np.int16)]
    )


# --------------------------------------------------------------------------- #
# Plane fitting
# --------------------------------------------------------------------------- #

def fit_plane_svd(xyz: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Plane via PCA. Returns (normal, tilt_deg, azimuth_deg, rmse)."""
    centroid = xyz.mean(axis=0)
    centered = xyz - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    nx, ny, nz = normal
    tilt_deg = float(np.degrees(np.arccos(abs(nz))))
    azimuth_deg = float(np.degrees(np.arctan2(nx, ny))) % 360.0
    rmse = float(np.sqrt(((centered @ normal) ** 2).mean()))
    return normal, tilt_deg, azimuth_deg, rmse


def fit_plane_ransac(
    xyz: np.ndarray,
    threshold: float = 0.10,
    iters: int = 200,
    min_inlier_frac: float = 0.6,
    seed: int = 0,
) -> tuple[np.ndarray, float, float, float, int, int]:
    """Tiny RANSAC; refines on inliers via SVD. Returns (normal, tilt, az, rmse, n_in, n_total)."""
    rng = np.random.default_rng(seed)
    n = len(xyz)
    if n < 3:
        raise ValueError("need at least 3 points")
    best_inliers = None
    best_count = 0
    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        p = xyz[idx]
        v1 = p[1] - p[0]
        v2 = p[2] - p[0]
        nrm = np.cross(v1, v2)
        s = np.linalg.norm(nrm)
        if s < 1e-9:
            continue
        nrm = nrm / s
        d = np.abs((xyz - p[0]) @ nrm)
        in_mask = d < threshold
        cnt = int(in_mask.sum())
        if cnt > best_count:
            best_count = cnt
            best_inliers = in_mask
    if best_inliers is None or best_count < max(3, int(min_inlier_frac * n)):
        # Fallback to plain SVD on all points
        normal, tilt, az, rmse = fit_plane_svd(xyz)
        return normal, tilt, az, rmse, n, n
    inliers = xyz[best_inliers]
    normal, tilt, az, rmse = fit_plane_svd(inliers)
    return normal, tilt, az, rmse, best_count, n


# --------------------------------------------------------------------------- #
# Per-polygon clip + fit (reused by both subcommands)
# --------------------------------------------------------------------------- #

def clip_and_fit(pts_tile: np.ndarray, polygon_proj: Polygon) -> dict | None:
    """Clip class-1 returns above local ground, RANSAC plane, return summary."""
    counts = collections.Counter(pts_tile[:, 3].astype(int).tolist())
    primary_class = 6 if counts.get(6, 0) > 0 else 1
    panel_pts = pts_tile[pts_tile[:, 3] == primary_class]
    in_poly = panel_pts[contains_xy(polygon_proj, panel_pts[:, 0], panel_pts[:, 1])]

    # height-above-ground filter when using class 1
    if primary_class == 1 and len(in_poly):
        ground = pts_tile[pts_tile[:, 3] == 2]
        cx, cy = polygon_proj.centroid.x, polygon_proj.centroid.y
        near = ground[(np.abs(ground[:, 0] - cx) < 25) & (np.abs(ground[:, 1] - cy) < 25)]
        if len(near) >= 5:
            ground_z = float(np.median(near[:, 2]))
            in_poly = in_poly[in_poly[:, 2] > ground_z + 1.5]

    if len(in_poly) < 10:
        return None

    normal, tilt, az, rmse, n_in, n_total = fit_plane_ransac(in_poly[:, :3])
    return {
        "n_points": int(n_total),
        "n_inliers": int(n_in),
        "tilt_deg": float(tilt),
        "azimuth_deg": float(az),
        "rmse_m": float(rmse),
        "area_m2": float(polygon_proj.area),
        "density_pts_per_m2": float(n_total / polygon_proj.area) if polygon_proj.area > 0 else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Subcommand: batch
# --------------------------------------------------------------------------- #

def sample_polygons(n_total: int = 30) -> gpd.GeoDataFrame:
    """Stratified sample across area bins, restricted to cached tiles."""
    print(f"# Loading polygons from {POLYGONS_PARQUET.name}")
    cols = ["detection_id", "geometry", "sam3_score"]
    gdf = gpd.read_parquet(POLYGONS_PARQUET, columns=cols)
    proj = gdf.to_crs(LAZ_CRS)
    gdf = gdf.copy()
    gdf["area_m2"] = proj.geometry.area
    gdf["x_km"] = (proj.geometry.centroid.x // 1000).astype(int)
    gdf["y_km"] = (proj.geometry.centroid.y // 1000).astype(int)
    gdf["tile"] = "w" + gdf["x_km"].astype(str).str.zfill(4) + "n" + gdf["y_km"].astype(str).str.zfill(4)
    gdf["geom_proj"] = proj.geometry.values

    # Restrict to cached tiles, sam3>=0.85, and bin by area
    in_cache = gdf[gdf["tile"].isin(CACHED_TILES) & (gdf["sam3_score"] >= 0.85)]
    bins = [0, 5, 15, 40, 80, 250]
    labels = ["<5", "5-15", "15-40", "40-80", "80-250"]
    in_cache = in_cache.copy()
    in_cache["abin"] = pd.cut(in_cache["area_m2"], bins=bins, labels=labels)

    per_bin = max(1, n_total // len(labels))
    rng = np.random.default_rng(42)
    parts = []
    for lab in labels:
        sub = in_cache[in_cache["abin"] == lab]
        if len(sub) == 0:
            continue
        idx = rng.choice(len(sub), size=min(per_bin, len(sub)), replace=False)
        parts.append(sub.iloc[idx])
    sample = pd.concat(parts).reset_index(drop=True)
    print(f"  sampled {len(sample)} polygons across {sample['tile'].nunique()} tiles "
          f"and {sample['abin'].nunique()} area bins")
    print(f"  area bin counts: {sample['abin'].value_counts().sort_index().to_dict()}")
    return sample


def cmd_batch(n: int) -> int:
    sample = sample_polygons(n)

    # Group by tile, load each once
    rows = []
    for tile, grp in sample.groupby("tile"):
        print(f"# Tile {tile}: {len(grp)} polygons")
        path = fetch_tile(tile)
        pts = load_tile_points(path)
        for _, r in grp.iterrows():
            res = clip_and_fit(pts, r["geom_proj"])
            row = {"detection_id": r["detection_id"], "tile": tile, "abin": r["abin"]}
            if res is None:
                row.update({"n_points": 0, "n_inliers": 0,
                            "tilt_deg": np.nan, "azimuth_deg": np.nan,
                            "rmse_m": np.nan, "area_m2": float(r["area_m2"]),
                            "density_pts_per_m2": np.nan, "ok": False})
            else:
                row.update(res)
                row["ok"] = True
            rows.append(row)
    df = pd.DataFrame(rows)
    out_csv = OUT / "spike_batch.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n# wrote {out_csv}  (n={len(df)}, fit_ok={df['ok'].sum()})")

    # Summary
    okdf = df[df["ok"]].copy()
    print()
    print(okdf[["area_m2", "n_points", "density_pts_per_m2", "tilt_deg", "azimuth_deg", "rmse_m"]]
          .describe(percentiles=[0.1, 0.5, 0.9]).round(2).to_string())

    plot_batch(okdf, OUT / "spike_batch.png")
    return 0


def plot_batch(df: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(12, 8), dpi=130)
    fig.suptitle(f"pv_geom batch validation — {len(df)} polygons across "
                 f"{df['tile'].nunique()} tiles", fontsize=12)

    # Tilt histogram
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.hist(df["tilt_deg"], bins=20, color="#1F3A5F")
    ax1.set_xlabel("tilt (deg)")
    ax1.set_ylabel("count")
    ax1.set_title(f"Tilt distribution  (median={df.tilt_deg.median():.1f}°)")
    ax1.grid(alpha=0.3)

    # Azimuth rose
    ax2 = fig.add_subplot(2, 2, 2, projection="polar")
    az_rad = np.radians(df["azimuth_deg"].dropna().values)
    bins = np.linspace(0, 2 * np.pi, 25)
    h, _ = np.histogram(az_rad, bins=bins)
    centers = (bins[:-1] + bins[1:]) / 2
    ax2.bar(centers, h, width=2 * np.pi / 24, color="#C0502C", alpha=0.85)
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)  # clockwise
    ax2.set_title("Azimuth rose (0=N, clockwise)")

    # Area vs RMSE
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(df["area_m2"], df["rmse_m"], c=df["density_pts_per_m2"],
                cmap="viridis", s=22, alpha=0.85)
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("polygon area (m^2)")
    ax3.set_ylabel("RMSE (m)")
    ax3.axhline(0.05, color="r", ls="--", lw=1, label="PRD threshold (0.05 m)")
    ax3.set_title("Area vs RMSE")
    ax3.grid(alpha=0.3, which="both")
    ax3.legend()

    # Density histogram
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.hist(df["density_pts_per_m2"], bins=20, color="#1F3A5F")
    ax4.axvline(4.0, color="r", ls="--", lw=1, label="PRD floor (4 pts/m^2)")
    ax4.set_xlabel("density (pts/m^2)")
    ax4.set_ylabel("count")
    ax4.set_title("Point density")
    ax4.grid(alpha=0.3)
    ax4.legend()

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"# wrote {out_path}")


# --------------------------------------------------------------------------- #
# Subcommand: size-sweep
# --------------------------------------------------------------------------- #

def get_polygon_inliers(polygon_id: str) -> tuple[np.ndarray, float, float, float, float]:
    """Load the inlier set for one polygon. Returns (xyz_inliers, tilt0, az0, rmse0, area)."""
    cols = ["detection_id", "geometry"]
    gdf = gpd.read_parquet(POLYGONS_PARQUET, columns=cols)
    sub = gdf[gdf["detection_id"] == polygon_id]
    if sub.empty:
        sys.exit(f"polygon {polygon_id} not found")
    poly_wgs = sub.iloc[0].geometry
    poly_proj = gpd.GeoSeries([poly_wgs], crs=4326).to_crs(LAZ_CRS).iloc[0]

    # Compute tile from centroid
    cx_km = int(poly_proj.centroid.x // 1000)
    cy_km = int(poly_proj.centroid.y // 1000)
    tile = f"w{cx_km:04d}n{cy_km:04d}"
    print(f"# polygon {polygon_id} -> tile {tile}, area {poly_proj.area:.1f} m^2")
    path = fetch_tile(tile)
    pts = load_tile_points(path)

    # Class-1 above ground+1.5 m, clipped
    cls1 = pts[pts[:, 3] == 1]
    in_poly = cls1[contains_xy(poly_proj, cls1[:, 0], cls1[:, 1])]
    ground = pts[pts[:, 3] == 2]
    near = ground[(np.abs(ground[:, 0] - poly_proj.centroid.x) < 25) &
                  (np.abs(ground[:, 1] - poly_proj.centroid.y) < 25)]
    ground_z = float(np.median(near[:, 2]))
    in_poly = in_poly[in_poly[:, 2] > ground_z + 1.5]

    normal, tilt0, az0, rmse0, _, _ = fit_plane_ransac(in_poly[:, :3])
    # Use only RANSAC inliers for the sweep, so the underlying truth is clean.
    d = np.abs((in_poly[:, :3] - in_poly[:, :3].mean(0)) @ normal)
    inliers = in_poly[d < 0.10]
    print(f"  truth fit (RANSAC, n={len(inliers)}): tilt={tilt0:.2f}°, az={az0:.2f}°, "
          f"rmse={rmse0:.3f} m")
    return inliers[:, :3], tilt0, az0, rmse0, float(poly_proj.area)


def cmd_size_sweep(polygon_id: str, n_bootstrap: int) -> int:
    inliers, tilt0, az0, rmse0, area = get_polygon_inliers(polygon_id)
    n_total = len(inliers)
    density = n_total / area

    # Sweep over N points from 5 up to n_total
    Ns = [5, 8, 12, 20, 30, 50, 75, 100, 150, 250, 500, 1000]
    Ns = [n for n in Ns if n <= n_total]

    rng = np.random.default_rng(42)
    rows = []
    for N in Ns:
        for k in range(n_bootstrap):
            idx = rng.choice(n_total, size=N, replace=False)
            sub = inliers[idx]
            _, tilt, az, _ = fit_plane_svd(sub)
            # Wrap azimuth deviation onto [-180, 180] for std calculation
            d_az = ((az - az0 + 180) % 360) - 180
            rows.append({"N": N, "tilt": tilt, "az": az, "tilt_err": tilt - tilt0, "az_err": d_az})
    df = pd.DataFrame(rows)
    summary = df.groupby("N").agg(
        tilt_mean=("tilt", "mean"),
        tilt_std=("tilt", "std"),
        az_std=("az_err", "std"),
        tilt_p10=("tilt", lambda s: np.percentile(s, 10)),
        tilt_p90=("tilt", lambda s: np.percentile(s, 90)),
        az_p10=("az_err", lambda s: np.percentile(s, 10)),
        az_p90=("az_err", lambda s: np.percentile(s, 90)),
    ).reset_index()
    print()
    print("Subsample sweep summary:")
    print(summary.round(3).to_string(index=False))

    # Save raw + summary
    df.to_csv(OUT / f"spike_size_sweep_{polygon_id}.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    fig.suptitle(
        f"Subsample sweep — {polygon_id}  "
        f"(area={area:.0f} m^2, density={density:.1f} pts/m^2, "
        f"truth tilt={tilt0:.2f}°, az={az0:.2f}°)",
        fontsize=11,
    )
    axes[0].errorbar(summary["N"], summary["tilt_mean"],
                     yerr=summary["tilt_std"], fmt="o-", color="#1F3A5F", capsize=3)
    axes[0].axhline(tilt0, color="r", ls="--", lw=1, label=f"truth ({tilt0:.2f}°)")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("N points subsample")
    axes[0].set_ylabel("tilt (deg)")
    axes[0].set_title("Tilt vs sample size  (mean ± std)")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend()

    axes[1].plot(summary["N"], summary["tilt_std"], "o-", color="#1F3A5F", label="tilt std (deg)")
    axes[1].plot(summary["N"], summary["az_std"], "s-", color="#C0502C", label="azimuth std (deg)")
    axes[1].axhline(2.0, color="grey", ls=":", lw=1, label="2° threshold")
    axes[1].axhline(10.0, color="grey", ls=":", lw=1)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("N points subsample")
    axes[1].set_ylabel("estimate std (deg)")
    axes[1].set_title("Spread vs sample size")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend()

    fig.tight_layout()
    out_path = OUT / f"spike_size_sweep_{polygon_id}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"# wrote {out_path}")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    pa = sp.add_parser("batch")
    pa.add_argument("--n", type=int, default=30)

    pb = sp.add_parser("size-sweep")
    pb.add_argument("--polygon-id", default="21_840480_397710__0")
    pb.add_argument("--n-bootstrap", type=int, default=50)

    args = ap.parse_args()
    if args.cmd == "batch":
        return cmd_batch(args.n)
    if args.cmd == "size-sweep":
        return cmd_size_sweep(args.polygon_id, args.n_bootstrap)
    raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
