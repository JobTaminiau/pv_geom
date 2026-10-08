# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib>=3.9",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "pyarrow>=17",
#   "shapely>=2.0",
#   "pyproj>=3.6",
#   "pillow>=10",
# ]
# ///
"""Visual + textual review of the 1000-polygon benchmark partitions.

Reads ``out_bench_1k/part-*.parquet``, prints a stdout summary, and writes
``scripts/eda_outputs/bench_1k_review.png`` — a 2x3 panel:

  (a) tilt vs azimuth scatter, coloured by mounting_type
  (b) RMSE histogram, log-scale, 5 cm threshold marked
  (c) spatial map: polygon centroids coloured by mounting_type
  (d) mounting-type bar with confidence p10/p50/p90 whiskers
  (e) confidence histogram per mounting_type
  (f) flag-count bar
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyproj
from shapely import wkb
from shapely.ops import transform as shapely_transform

sys.path.insert(0, str(Path(__file__).parent))
from _aerial import aerial_basemap  # noqa: E402

_DEFAULT = "out_bench_1k"
OUT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else _DEFAULT)
FIG_PATH = Path(f"scripts/eda_outputs/{OUT_DIR.name}_review.png")

MOUNT_COLOURS = {
    "ground_mount_tracker_suspected": "#1f77b4",
    "ground_mount_fixed": "#ff7f0e",
    "ambiguous": "#7f7f7f",
    "carport": "#2ca02c",
    "flush_mount_rooftop": "#d62728",       # pre-0.2 outputs
    "flush_mount_pitched_roof": "#d62728",
    "flush_mount_flat_roof": "#e377c2",
    "tilted_rack_rooftop": "#9467bd",
    "east_west_rack_rooftop": "#17becf",
    "pole_mount": "#bcbd22",
}


def _load() -> pd.DataFrame:
    parts = sorted(OUT_DIR.glob("part-*.parquet"))
    table = pa.concat_tables([pq.read_table(p) for p in parts])
    df = table.to_pandas()
    geoms = [wkb.loads(b) for b in df["geometry"]]
    df["geom_utm"] = geoms
    df["cx"] = [g.centroid.x for g in geoms]
    df["cy"] = [g.centroid.y for g in geoms]
    return df


def _to_wgs84(geoms_utm: list, src_crs: str = "EPSG:6341") -> list:
    """Reproject a list of shapely geoms from src_crs to WGS84 lon/lat."""
    tr = pyproj.Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    return [shapely_transform(tr.transform, g) for g in geoms_utm]


def _summary(df: pd.DataFrame) -> None:
    print(f"rows: {len(df)}")
    print(f"successful tilt fits: {df['panel_tilt_deg'].notna().sum()} / {len(df)}")

    fit = df.dropna(subset=["panel_tilt_deg"])
    print()
    print("--- panel fit quality ---")
    print(f"  tilt    p10/p50/p90 = "
          f"{fit['panel_tilt_deg'].quantile(.1):5.2f} / "
          f"{fit['panel_tilt_deg'].median():5.2f} / "
          f"{fit['panel_tilt_deg'].quantile(.9):5.2f}  deg")
    print(f"  rmse    p50/p90    = "
          f"{fit['panel_rmse_m'].median():.4f} / {fit['panel_rmse_m'].quantile(.9):.4f}  m")
    n_under_5cm = (fit["panel_rmse_m"] <= 0.05).sum()
    print(f"  rmse <= 5 cm: {n_under_5cm}/{len(fit)} ({n_under_5cm/len(fit):.1%})")

    print()
    print("--- azimuth distribution ---")
    az_bin = pd.cut(
        fit["panel_azimuth_deg"],
        bins=[0, 45, 135, 225, 315, 360],
        labels=["N1", "E", "S", "W", "N2"],
        include_lowest=True,
    )
    az_counts = az_bin.value_counts().sort_index()
    az_counts["N"] = az_counts.get("N1", 0) + az_counts.get("N2", 0)
    az_counts = az_counts.drop(["N1", "N2"], errors="ignore")
    for k, v in az_counts.items():
        print(f"  {k}: {v}")

    print()
    print("--- mounting classification ---")
    for label, n in df["mounting_type"].value_counts().items():
        slice_ = df[df["mounting_type"] == label]["mounting_confidence"]
        print(f"  {label:35s}  n={n:4d}  "
              f"conf p10/p50 = {slice_.quantile(.1):.2f} / {slice_.median():.2f}")

    print()
    print("--- rules fired ---")
    for rule, n in df["mounting_rule"].value_counts().items():
        print(f"  {rule}: {n}")

    print()
    print("--- flags ---")
    flag_counts: dict[str, int] = {}
    for flags in df["flags"]:
        if flags is None:
            continue
        for f in flags:
            flag_counts[f] = flag_counts.get(f, 0) + 1
    for k, v in sorted(flag_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    print()
    print("--- spatial extent (EPSG:6341 metres) ---")
    print(f"  easting  : {df['cx'].min():.0f} .. {df['cx'].max():.0f}  "
          f"(span {df['cx'].max() - df['cx'].min():.0f} m)")
    print(f"  northing : {df['cy'].min():.0f} .. {df['cy'].max():.0f}  "
          f"(span {df['cy'].max() - df['cy'].min():.0f} m)")
    print(f"  area p50/p90: {df['area_m2'].median():.1f} / "
          f"{df['area_m2'].quantile(.9):.1f}  m^2")


def _figure(df: pd.DataFrame) -> None:
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fit = df.dropna(subset=["panel_tilt_deg"])

    # (a) tilt vs azimuth scatter, coloured by mounting type
    ax = axes[0, 0]
    for label, sub in fit.groupby("mounting_type"):
        ax.scatter(sub["panel_azimuth_deg"], sub["panel_tilt_deg"],
                   s=8, alpha=0.6, c=MOUNT_COLOURS.get(label, "k"), label=label)
    ax.set_xlabel("panel azimuth (deg)")
    ax.set_ylabel("panel tilt (deg)")
    ax.set_title("(a) tilt vs azimuth")
    ax.set_xlim(0, 360)
    ax.axvline(180, color="k", lw=0.4, ls=":")
    ax.legend(fontsize=7, loc="upper right")

    # (b) RMSE histogram (log-scale)
    ax = axes[0, 1]
    ax.hist(fit["panel_rmse_m"], bins=np.logspace(-3.5, 0, 50),
            color="#444", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.axvline(0.05, color="r", lw=1, ls="--", label="5 cm spec floor")
    ax.set_xlabel("panel RMSE (m, log)")
    ax.set_ylabel("polygons")
    ax.set_title("(b) panel-fit RMSE")
    ax.legend(fontsize=8)

    # (c) spatial map on Maricopa County 2024 ortho
    ax = axes[0, 2]
    geoms_wgs = _to_wgs84(df["geom_utm"].tolist())
    df_wgs_minx = min(g.bounds[0] for g in geoms_wgs)
    df_wgs_miny = min(g.bounds[1] for g in geoms_wgs)
    df_wgs_maxx = max(g.bounds[2] for g in geoms_wgs)
    df_wgs_maxy = max(g.bounds[3] for g in geoms_wgs)
    pad_lon = (df_wgs_maxx - df_wgs_minx) * 0.05
    pad_lat = (df_wgs_maxy - df_wgs_miny) * 0.05
    fetch_bbox = (
        df_wgs_minx - pad_lon, df_wgs_miny - pad_lat,
        df_wgs_maxx + pad_lon, df_wgs_maxy + pad_lat,
    )
    # An ortho basemap only makes sense at bench scale; a metro-wide bbox
    # (full-atlas review) would need a multi-GB z21 canvas (MemoryError).
    metro_scale = max(df_wgs_maxx - df_wgs_minx, df_wgs_maxy - df_wgs_miny) > 0.1
    if metro_scale:
        print("[aerial] bbox too large for ortho basemap; plotting centroids instead")
        for label, sub_idx in df.groupby("mounting_type").groups.items():
            xs = [geoms_wgs[i].centroid.x for i in sub_idx]
            ys = [geoms_wgs[i].centroid.y for i in sub_idx]
            ax.scatter(xs, ys, s=0.2, color=MOUNT_COLOURS.get(label, "k"),
                       alpha=0.4, linewidths=0, rasterized=True)
            ax.plot([], [], color=MOUNT_COLOURS.get(label, "k"), lw=1.2, label=label)
        ax.set_xlim(fetch_bbox[0], fetch_bbox[2])
        ax.set_ylim(fetch_bbox[1], fetch_bbox[3])
        ax.set_title("(c) detection centroids by mounting type")
    else:
        print(f"[aerial] fetching basemap for {fetch_bbox}")
        img, extent = aerial_basemap(fetch_bbox)
        ax.imshow(img, extent=extent, zorder=0)

        # Overlay polygon outlines; group by mounting_type for the legend.
        for label, sub_idx in df.groupby("mounting_type").groups.items():
            for i in sub_idx:
                g = geoms_wgs[i]
                if g.geom_type == "Polygon":
                    xs, ys = g.exterior.xy
                    ax.plot(xs, ys, color=MOUNT_COLOURS.get(label, "k"),
                            lw=0.8, alpha=0.9)
            # Add a sentinel for the legend
            ax.plot([], [], color=MOUNT_COLOURS.get(label, "k"), lw=1.2, label=label)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title("(c) polygons over Maricopa 2024 ortho")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect("equal")
    ax.legend(fontsize=6, loc="lower left")

    # (d) mounting-type bar with confidence whiskers
    ax = axes[1, 0]
    cats = df["mounting_type"].value_counts().index.tolist()
    counts = [int((df["mounting_type"] == c).sum()) for c in cats]
    p10 = [df[df["mounting_type"] == c]["mounting_confidence"].quantile(.1) for c in cats]
    p50 = [df[df["mounting_type"] == c]["mounting_confidence"].median() for c in cats]
    p90 = [df[df["mounting_type"] == c]["mounting_confidence"].quantile(.9) for c in cats]
    bars = ax.bar(cats, counts, color=[MOUNT_COLOURS.get(c, "k") for c in cats])
    ax.set_ylabel("polygons")
    ax.set_title("(d) mounting type counts")
    for bar, n, p_50 in zip(bars, counts, p50, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, n + 5,
                f"{n}\nconf p50 {p_50:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=7)
    ax.set_ylim(0, max(counts) * 1.25)

    # (e) confidence histogram per mounting type
    ax = axes[1, 1]
    for label, sub in df.groupby("mounting_type"):
        ax.hist(sub["mounting_confidence"], bins=20, alpha=0.55,
                color=MOUNT_COLOURS.get(label, "k"), label=label, edgecolor="white",
                linewidth=0.3)
    ax.set_xlabel("mounting_confidence")
    ax.set_ylabel("polygons")
    ax.set_title("(e) confidence by mounting type")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=7)

    # (f) flag counts
    ax = axes[1, 2]
    flag_counts: dict[str, int] = {}
    for flags in df["flags"]:
        if flags is None:
            continue
        for f in flags:
            flag_counts[f] = flag_counts.get(f, 0) + 1
    if flag_counts:
        items = sorted(flag_counts.items(), key=lambda x: -x[1])
        labels, vals = zip(*items, strict=True)
        ax.barh(labels, vals, color="#7f3f00")
        for i, v in enumerate(vals):
            ax.text(v + 5, i, str(v), va="center", fontsize=8)
    ax.set_xlabel("polygons")
    ax.set_title("(f) quality flags")
    ax.invert_yaxis()

    fig.suptitle(f"pv-geom benchmark review — {OUT_DIR.name} ({len(df)} polygons)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=110, bbox_inches="tight")
    print(f"\nwrote {FIG_PATH}")


def main() -> None:
    df = _load()
    _summary(df)
    _figure(df)


if __name__ == "__main__":
    main()
