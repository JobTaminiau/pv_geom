"""Before/after harness for the 2026-08-03 roof-fit + vintage-screen changes.

Reruns `_build_row` over real polygons from a locally cached LAZ tile under two
configs and diffs them, so the effect of the roof-fit consensus floor, the
collar guard, and the confidence cap can be re-measured without a cloud run.
"before" is reproduced purely by config, so this keeps working after the code
moves on.

Reports:
  * usable roof references (the ceiling on vintage screening coverage)
  * the tri-state panel-standoff screen
  * mounting label/rule shifts
  * wrong-facet rate: angle between the ring plane and a collar-only fit,
    which is what the collar guard exists to bound

Usage
-----
    python scripts/_vintage_impact.py [--tile PATH] [--out CSV]

Defaults to the Phoenix tile in the local pv_geom cache. Needs `out_full_v0.1.0/`
(for the polygon geometries on that tile) and `data/fema/az.geoparquet`.
See STATUS.md, 2026-08-03 entry, for the numbers this produced when written.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from shapely import contains_xy, wkb

from pv_geom.config import PVGeomConfig, RoofPlaneConfig
from pv_geom.geometry.plane_fit import fit_plane_ransac
from pv_geom.geometry.roof_plane import extract_roof_plane
from pv_geom.io.lidar import clip_points_to_polygon, read_tile_points
from pv_geom.pipeline.tile_task import _build_row

DEFAULT_TILE = (
    Path.home() / "AppData/Local/Temp/pv_geom_cache"
    / "US__arizona__top-level__lidar__lidar_data__"
      "USGS_LPC_AZ_MaricopaPinal_2020_B20_w0432n3719.laz"
)
DEFAULT_OUTPUT = "out_full_v0.1.0"
DEFAULT_FOOTPRINTS = "data/fema/az.geoparquet"


def _pre_0_4_config() -> PVGeomConfig:
    """Reproduce pre-0.4.0 behaviour by config alone.

    The ring inherited the panel fit's 0.6 consensus floor, there was no collar
    guard, and `no_panel_standoff` did not cap confidence.
    """
    cfg = PVGeomConfig()
    cfg.roof_plane.min_inlier_frac = 0.6
    cfg.roof_plane.collar_agreement_min = 0.0        # guard can never trigger
    cfg.mounting_rules.no_panel_standoff_confidence_max = 1.0
    return cfg


def load_scene(tile: Path, output_dir: str, footprints_uri: str):
    pts, _ = read_tile_points(tile)
    cls = pts[:, 3].astype(np.int16)
    ground = pts[cls == 2][:, :3]
    if (cls == 6).any():
        pool = pts[cls == 6][:, :3]
    else:
        c1 = pts[cls == 1][:, :3]
        pool = c1[c1[:, 2] > float(np.median(ground[:, 2])) + 0.8]

    xmin, ymin = pool[:, 0].min(), pool[:, 1].min()
    xmax, ymax = pool[:, 0].max(), pool[:, 1].max()
    tile_id = tile.stem.split("_")[-1]

    files = sorted(glob.glob(f"{output_dir}/part-*.parquet"))
    if not files:
        raise SystemExit(f"no partitions under {output_dir}/ — need the v0.1.0 mirror")
    df = ds.dataset(files, format="parquet").to_table(
        columns=["polygon_id", "geometry", "lidar_tile_ids"]).to_pandas()
    df = df[df["lidar_tile_ids"].apply(lambda a: a is not None and tile_id in list(a))]
    df["geom"] = df["geometry"].apply(wkb.loads)
    df = df[df["geom"].apply(
        lambda g: xmin <= g.centroid.x <= xmax and ymin <= g.centroid.y <= ymax)]

    fp = gpd.read_parquet(footprints_uri).to_crs("EPSG:6341")
    fp = fp.cx[xmin - 50:xmax + 50, ymin - 50:ymax + 50].reset_index(drop=True)
    if "building_id" not in fp.columns:
        fp["building_id"] = [f"b{i}" for i in range(len(fp))]
    return df, fp, pool, ground, tile_id


def run(cfg: PVGeomConfig, df, fp, pool, ground, tile_id) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        poly = r["geom"]
        others = gpd.GeoDataFrame(
            geometry=[g for pid, g in zip(df["polygon_id"], df["geom"])
                      if pid != r["polygon_id"]],
            crs="EPSG:6341")
        row = _build_row(
            polygon=poly, polygon_id=str(r["polygon_id"]), cfg=cfg,
            config_hash="h", run_id="r", partition_id=0,
            panel_pts=clip_points_to_polygon(pool, poly, erosion_m=cfg.panel_plane.erosion_m),
            ground_xyz=ground, roof_input_pts=pool, footprints=fp,
            other_pv_polygons=others, contributing_tile_ids=(tile_id,))
        flags = set(row["flags"])
        rows.append({
            "pid": row["polygon_id"], "type": row["mounting_type"],
            "rule": row["mounting_rule"], "conf": float(row["mounting_confidence"]),
            "har": row["height_above_roof_m"], "flags": flags,
            "state": ("unscreenable" if "standoff_unscreenable" in flags
                      else "flagged" if "no_panel_standoff" in flags else "passed"),
        })
    out = pd.DataFrame(rows)
    out["has_roof"] = out["har"].notna()
    return out


def wrong_facet_rate(df, fp, pool, cfg: PVGeomConfig) -> pd.Series:
    """Angle between the delivered roof plane and a collar-only fit.

    Large values mean the ring fit landed on the facet across the ridge — the
    failure the collar guard bounds. Also Open improvement #9.
    """
    angles = []
    for _, r in df.iterrows():
        poly = r["geom"]
        others = gpd.GeoDataFrame(geometry=[], crs="EPSG:6341")
        res = extract_roof_plane(poly, fp, others, pool, cfg.roof_plane, seed=0)
        if not res.usable:
            continue
        zone = poly.buffer(1.2).difference(poly)
        if zone.is_empty:
            continue
        m = contains_xy(zone, pool[:, 0], pool[:, 1])
        collar = pool[m]
        if len(collar) < 40:
            continue
        cf = fit_plane_ransac(collar, ransac_threshold=0.15, min_inlier_frac=0.35,
                              max_iter=200, seed=0)
        if np.isnan(cf.tilt_deg):
            continue
        angles.append(float(np.degrees(np.arccos(
            np.clip(abs(float(np.dot(res.fit.normal, cf.normal))), -1, 1)))))
    return pd.Series(angles, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=Path, default=DEFAULT_TILE)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    ap.add_argument("--footprints", default=DEFAULT_FOOTPRINTS)
    ap.add_argument("--out", default=None, help="write the per-row diff to this CSV")
    args = ap.parse_args()

    if not args.tile.exists():
        raise SystemExit(f"tile not found: {args.tile}\nPre-warm one via a --max-polygons run.")

    df, fp, pool, ground, tile_id = load_scene(args.tile, args.output_dir, args.footprints)
    print(f"tile {tile_id}: {len(pool):,} panel-class pts, {len(df)} polygons, "
          f"{len(fp)} footprints\n")

    old_cfg, new_cfg = _pre_0_4_config(), PVGeomConfig()
    o = run(old_cfg, df, fp, pool, ground, tile_id)
    n = run(new_cfg, df, fp, pool, ground, tile_id)

    print(f"{'':30s} {'BEFORE':>8s} {'AFTER':>8s}")
    print(f"{'usable roof reference':30s} {o['has_roof'].sum():>8d} {n['has_roof'].sum():>8d}")
    for st in ("passed", "flagged", "unscreenable"):
        print(f"{'  standoff screen: ' + st:30s} "
              f"{(o['state'] == st).sum():>8d} {(n['state'] == st).sum():>8d}")
    print(f"{'mean confidence':30s} {o['conf'].mean():>8.3f} {n['conf'].mean():>8.3f}")
    print(f"{'rows at confidence >= .999':30s} "
          f"{(o['conf'] >= .999).sum():>8d} {(n['conf'] >= .999).sum():>8d}")

    comp = pd.DataFrame({"before": o.set_index("pid")["type"],
                         "after": n.set_index("pid")["type"],
                         "rule_before": o.set_index("pid")["rule"],
                         "rule_after": n.set_index("pid")["rule"]})
    ch = comp[comp.before != comp.after]
    print(f"\nmounting_type: {len(ch)} of {len(comp)} rows changed label")
    if len(ch):
        print(ch[["before", "after"]].value_counts().to_string())

    # Why the collar guard exists: the middle config is the naive fix — a looser
    # consensus floor with no guard — and it is the one that fits the far facet.
    unguarded = PVGeomConfig()
    unguarded.roof_plane.collar_agreement_min = 0.0
    print("\nring-plane vs collar-plane angle (large = fitted the facet ACROSS the ridge):")
    for tag, cfg in (("0.6 floor, no guard  ", old_cfg),
                     ("0.4 floor, NO GUARD  ", unguarded),
                     ("0.4 floor + guard    ", new_cfg)):
        a = wrong_facet_rate(df, fp, pool, cfg)
        if len(a):
            print(f"  {tag} n={len(a):>3d}  median {a.median():>6.2f} deg  "
                  f"max {a.max():>6.2f}  >20 deg: {(a > 20).sum()}")

    if args.out:
        comp.to_csv(args.out)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
