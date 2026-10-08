"""Load a finished run for reporting, and reduce it to its headline numbers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pv_geom import __version__
from pv_geom.io.output import read_manifest, read_output
from pv_geom.report import stats
from pv_geom.utils.north import GRID_NORTH
from pv_geom.vintage import parse_vintage


def load_run(output_uri: str | Path, *, polygon_vintage=None, lidar_date=None):
    """``(GeoDataFrame, prepared DataFrame, manifest)`` for a run."""
    manifest = read_manifest(output_uri)
    gdf = read_output(output_uri)
    df = stats.prepare(pd.DataFrame(gdf.drop(columns="geometry")), manifest,
                       polygon_vintage=parse_vintage(polygon_vintage),
                       lidar_date=parse_vintage(lidar_date))
    cent = gdf.geometry.centroid
    df["cx"] = cent.x.to_numpy()
    df["cy"] = cent.y.to_numpy()
    if gdf.crs is not None:
        ll = cent.to_crs("EPSG:4326")
        df["centroid_lon"] = ll.x.to_numpy()
        df["centroid_lat"] = ll.y.to_numpy()
    return gdf, df, manifest


def _compass(deg: float) -> str:
    if not np.isfinite(deg):
        return "n/a"
    return stats.COMPASS_16[int(((deg + 11.25) % 360) // 22.5)]


def _iso(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v.isoformat() if hasattr(v, "isoformat") else str(v)


def summarise(df: pd.DataFrame, tables: dict[str, pd.DataFrame], manifest: dict,
              headline: str, weight: str = "area") -> dict:
    """The numbers the report text is written from."""
    summ = tables["summary_statistics"]

    def _row(stratum: str, weight: str) -> dict:
        r = summ[(summ["stratum"] == stratum) & (summ["weight"] == weight)].iloc[0]
        return {
            "n": int(r["n"]),
            "area_m2": float(r["area_m2"]),
            "tilt_median_deg": float(r["tilt_p50_deg"]),
            "tilt_p25_deg": float(r["tilt_p25_deg"]),
            "tilt_p75_deg": float(r["tilt_p75_deg"]),
            "tilt_mean_deg": float(r["tilt_mean_deg"]),
            "azimuth_circular_mean_deg": float(r["azimuth_circular_mean_deg"]),
            "azimuth_circular_mean_compass": _compass(float(r["azimuth_circular_mean_deg"])),
            "azimuth_resultant_length": float(r["azimuth_resultant_length"]),
            "azimuth_defined_share": float(r["azimuth_defined_share"]),
            **{f"share_facing_{q}": float(r[f"share_facing_{q}"]) for q in stats.QUADRANTS},
        }

    basis = tables["geometry_basis"].set_index("geometry_basis")
    v = manifest.get("vintage", {})
    lidar = df["lidar_date"].dropna()
    poly = df["polygon_vintage"].dropna()
    gaps = pd.to_numeric(df["vintage_gap_days"], errors="coerce").dropna()
    n_fit = int(df["fitted"].sum())
    rmse = df.loc[df["fitted"], "fit_rmse_m"].dropna()
    return {
        "pkg_version": manifest.get("pkg_version") or __version__,
        "run_id": manifest.get("run_id"),
        "crs": manifest.get("crs") or manifest.get("config", {}).get("crs", {}).get("target"),
        "inputs": manifest.get("inputs", {}),
        "n_input_polygons": manifest.get("counts", {}).get("polygons"),
        "n_rows": len(df),
        "n_fitted": n_fit,
        "fit_rate": n_fit / len(df) if len(df) else float("nan"),
        "total_plan_area_m2": float(df["area_m2"].sum()),
        "polygon_vintage": _iso(poly.max()) if len(poly) else v.get("polygon_vintage"),
        # How the vintage was declared ("2024"), when it was a window rather
        # than a day and every row shares it.
        "polygon_vintage_declared_as": (
            v.get("polygon_vintage_declared_as")
            if not v.get("polygon_vintage_column") and len(set(poly)) <= 1 else None
        ),
        "polygon_vintage_min": _iso(poly.min()) if len(poly) else v.get("polygon_vintage"),
        "lidar_date_min": _iso(lidar.min()) if len(lidar) else v.get("lidar_flight_start"),
        "lidar_date_max": _iso(lidar.max()) if len(lidar) else v.get("lidar_flight_end"),
        "lidar_date_source": (
            df["lidar_date_source"].dropna().mode().iloc[0]
            if df["lidar_date_source"].notna().any() else v.get("lidar_date_source")
        ),
        "vintage_gap_days": float(gaps.median()) if len(gaps) else v.get("vintage_gap_days"),
        "geometry_basis": {
            b: {"n": int(basis.loc[b, "n"]), "share_count": float(basis.loc[b, "share_count"]),
                "share_area": float(basis.loc[b, "share_area"])}
            for b in basis.index
        },
        "headline_stratum": headline,
        "weight": weight,
        "headline_label": stats.STRATA[headline][0],
        "headline": {"area": _row(headline, "area"), "count": _row(headline, "count")},
        "all_fitted": {"area": _row("all_fitted", "area"), "count": _row("all_fitted", "count")},
        "fit_rmse_m_p50": float(rmse.median()) if len(rmse) else float("nan"),
        "fit_rmse_m_p90": float(rmse.quantile(0.9)) if len(rmse) else float("nan"),
        "point_density_p50": float(df["point_density"].median()),
        # Outputs from before schema 0.4 report azimuth relative to grid north.
        "azimuth_reference": manifest.get("azimuth_reference", GRID_NORTH),
        "wide_tolerance_share_of_fitted": (
            float(df.loc[df["fitted"], "flags"]
                  .map(lambda f: f is not None and "wide_tolerance_fit" in f).mean())
            if n_fit else float("nan")
        ),
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
