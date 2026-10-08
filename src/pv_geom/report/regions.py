"""Results by region: districts, municipalities, utility territories.

A polygon belongs to the region its centroid falls in. Statistics are for the
report's headline stratum, so a region's figures are comparable with the
study-wide ones.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

from pv_geom.errors import InputError
from pv_geom.io.vector import read_vector
from pv_geom.report import figures as figs
from pv_geom.report import stats

# Below this many polygons a region's distribution is not described.
MIN_REGION_ROWS = 10
OUTSIDE = "(outside all regions)"


def assign_regions(gdf: gpd.GeoDataFrame, regions_uri: str | Path, region_col: str
                   ) -> tuple[np.ndarray, gpd.GeoDataFrame]:
    """The region of every row of ``gdf`` (by centroid), and the region layer
    dissolved by name in the run's CRS."""
    layer = read_vector(regions_uri)
    if region_col not in layer.columns:
        raise InputError(
            f"region layer has no column {region_col!r}",
            f"available columns: {', '.join(c for c in layer.columns if c != 'geometry')}")
    if layer.crs is None:
        raise InputError("region layer has no CRS", "write it with a CRS and try again")
    layer = layer[[region_col, "geometry"]].dropna(subset=[region_col])
    layer[region_col] = layer[region_col].astype(str)
    regions = layer.to_crs(gdf.crs).dissolve(by=region_col, as_index=False)

    cent = gpd.GeoDataFrame({"row": np.arange(len(gdf))}, geometry=gdf.geometry.centroid.values,
                            crs=gdf.crs)
    hit = gpd.sjoin(cent, regions, how="left", predicate="within")
    hit = hit[~hit["row"].duplicated()]          # overlapping regions: first wins
    names = hit.set_index("row")[region_col].reindex(np.arange(len(gdf)))
    return names.fillna(OUTSIDE).to_numpy(dtype=object), regions.rename(
        columns={region_col: "region"})


def regions_table(df: pd.DataFrame, region: np.ndarray, headline: str,
                  weight: str = "area") -> pd.DataFrame:
    """One row per region: coverage, and tilt and facing of the headline stratum."""
    in_stratum = stats.stratum_mask(df, headline).to_numpy()
    rows = []
    for name in sorted(set(region), key=lambda r: (r == OUTSIDE, r)):
        here = region == name
        sub = df[here & in_stratum]
        w = (np.ones(len(sub)) if weight == "count"
             else sub["w_area"].to_numpy(dtype=float))
        row: dict = {
            "region": name, "n_polygons": int(here.sum()),
            "n_measured": int(df.loc[here, "fitted"].sum()),
            "n_recommended": (int(df.loc[here, "recommended"].sum())
                              if "recommended" in df.columns else None),
            "stratum": headline, "weight": weight, "n": len(sub),
            "area_m2": float(sub["w_area"].sum()),
            "sufficient": len(sub) >= MIN_REGION_ROWS,
        }
        if len(sub) >= MIN_REGION_ROWS and w.sum() > 0:
            tilt = sub["tilt_deg"].to_numpy(dtype=float)
            q = stats.weighted_quantile(tilt, w, [0.25, 0.5, 0.75])
            row.update({"tilt_p25_deg": q[0], "tilt_p50_deg": q[1], "tilt_p75_deg": q[2]})
            has_az = sub["azimuth_deg"].notna().to_numpy()
            az, w_az = sub["azimuth_deg"].to_numpy(dtype=float)[has_az], w[has_az]
            quad = stats.sector_index(az, 4)
            shares = {q_: (float(w_az[quad == i].sum() / w_az.sum()) if w_az.sum() > 0
                           else np.nan) for i, q_ in enumerate(stats.QUADRANTS)}
            row.update({f"share_facing_{k}": v for k, v in shares.items()})
            if w_az.sum() > 0:
                top = max(shares, key=lambda k: shares[k])
                row.update({"dominant_orientation": top, "dominant_share": shares[top]})
        rows.append(row)
    cols = ["region", "n_polygons", "n_measured", "n_recommended", "stratum", "weight", "n",
            "area_m2", "sufficient", "tilt_p25_deg", "tilt_p50_deg", "tilt_p75_deg",
            *[f"share_facing_{q}" for q in stats.QUADRANTS], "dominant_orientation",
            "dominant_share"]
    return pd.DataFrame(rows).reindex(columns=cols)


def display_table(table: pd.DataFrame) -> pd.DataFrame:
    def _num(v: float, fmt: str) -> str:
        return "–" if pd.isna(v) else format(v, fmt)

    return pd.DataFrame({
        "Region": table["region"],
        "Polygons": table["n_polygons"].map("{:,}".format),
        "Measured": table["n_measured"].map("{:,}".format),
        "In headline group": table["n"].map("{:,}".format),
        "Median tilt (°)": table["tilt_p50_deg"].map(lambda v: _num(v, ".1f")),
        "Facing south": table["share_facing_S"].map(lambda v: _num(v, ".0%")),
        "Dominant facing": [
            "–" if pd.isna(o) else f"{o} ({s:.0%})"
            for o, s in zip(table["dominant_orientation"], table["dominant_share"],
                            strict=True)],
    })


def fig_regions(regions: gpd.GeoDataFrame, table: pd.DataFrame, headline: str
                ) -> figs.FigureSpec | None:
    """Choropleths of median tilt and of the share facing south; each region is
    lettered with its dominant facing when there are few enough to read."""
    geo = regions.merge(table, on="region", how="left")
    ok = geo[geo["sufficient"].fillna(False).astype(bool)]
    if not len(ok):
        return None
    x0, y0, x1, y1 = geo.total_bounds
    aspect = (y1 - y0) / (x1 - x0) if x1 > x0 else 1.0
    height = float(np.clip(figs.FULL_WIDTH_MM * 0.42 * aspect + 14, 45, 125))
    panels = [("tilt_p50_deg", "a  Median tilt", "Median tilt (°)", 0.0, 40.0, 1.0),
              ("share_facing_S", "b  Facing south", "Share facing south (%)", 0.0, 100.0, 100.0)]
    with matplotlib.rc_context(figs.RC):
        fig = figs._figure(figs.FULL_WIDTH_MM, height)
        axes = fig.subplots(1, 2)
        for ax, (col, title, label, vmin, vmax, scale) in zip(axes, panels, strict=True):
            geo.plot(ax=ax, color=figs.CONTEXT_FILL, edgecolor=figs.SURFACE, linewidth=0.6)
            values = ok[col].to_numpy(dtype=float) * scale
            norm = matplotlib.colors.Normalize(vmin, vmax)
            ok.plot(ax=ax, color=[figs.SEQUENTIAL(norm(v)) for v in values],
                    edgecolor=figs.SURFACE, linewidth=0.6)
            cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm, figs.SEQUENTIAL),
                              cax=ax.inset_axes([1.03, 0.0, 0.035, 1.0]))
            cb.set_label(label)
            cb.outline.set_visible(False)
            cb.ax.tick_params(length=0)
            if col == "share_facing_S" and len(ok) <= 30:
                for r in ok.itertuples():
                    if isinstance(r.dominant_orientation, str):
                        p = r.geometry.representative_point()
                        ax.annotate(r.dominant_orientation, (p.x, p.y), ha="center",
                                    va="center", fontsize=6.5, color=figs.INK,
                                    bbox={"boxstyle": "round,pad=0.15", "fc": figs.SURFACE,
                                          "ec": "none", "alpha": 0.85})
            ax.set_aspect("equal")
            ax.set_axis_off()
            ax.set_title(title)
    label = stats.STRATA[headline][0].lower()
    return figs.FigureSpec(
        "regions", "Geometry by region",
        f"Array geometry by region ({label}). (a) Median tilt. (b) Share facing south; "
        "the letter is the region's dominant facing where regions are few enough to label. "
        f"Grey regions have fewer than {MIN_REGION_ROWS} polygons in the group.", fig)
