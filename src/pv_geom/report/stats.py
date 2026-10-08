"""Profile statistics for a pv-geom output.

Everything here is a pure function of the output table, returning tidy
DataFrames that are written to CSV as-is — the same numbers feed the figures,
the report text and whatever a reader wants to recompute.

Two conventions run through all of it:

* **Strata.** Statistics are reported for every fitted polygon and, separately,
  by ``geometry_basis``. When the polygons postdate the LiDAR, only the *panel
  basis* stratum is known to describe panels; the others describe a surface that
  may be the roof underneath.
* **Weights.** Each statistic is given per polygon (``count``) and per unit of
  array surface (``area``, using ``surface_area_m2``), since a fleet's
  orientation by capacity is dominated by its large arrays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pv_geom.schema import FIT_FAILURE_DESCRIPTIONS, MEASURED, STATUS_DESCRIPTIONS
from pv_geom.schema import NO_FIT as NO_FIT_STATUS
from pv_geom.vintage import (
    GEOMETRY_BASIS,
    NO_FIT,
    NOT_MEASURED,
    PANEL_BASES,
    PANEL_BY_VINTAGE,
    PANEL_CONFIRMED,
    SURFACE_UNRESOLVED,
    UNSCREENED,
    geometry_basis,
)

# Report strata: key -> (label, set of geometry_basis values).
STRATA: dict[str, tuple[str, frozenset[str]]] = {
    "all_fitted": ("All fitted polygons",
                   frozenset({PANEL_CONFIRMED, PANEL_BY_VINTAGE, SURFACE_UNRESOLVED, UNSCREENED})),
    "panel": ("Panel basis", PANEL_BASES),
    "surface_unresolved": ("Surface unresolved", frozenset({SURFACE_UNRESOLVED})),
    "unscreened": ("Unscreened", frozenset({UNSCREENED})),
}

BASIS_LABELS: dict[str, str] = {
    PANEL_CONFIRMED: "Panel confirmed",
    PANEL_BY_VINTAGE: "Panel by vintage",
    SURFACE_UNRESOLVED: "Surface unresolved",
    UNSCREENED: "Unscreened",
    NO_FIT: "No fit",
    NOT_MEASURED: "Not measured",
}

WEIGHTS = ("count", "area")

COMPASS_16 = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
              "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")
COMPASS_8 = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
QUADRANTS = ("N", "E", "S", "W")

TILT_BIN_DEG = 5.0
TILT_MAX_DEG = 60.0


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #

def prepare(df: pd.DataFrame, manifest: dict | None = None, *,
            polygon_vintage=None, lidar_date=None) -> pd.DataFrame:
    """Normalise an output table for reporting.

    ``polygon_vintage`` / ``lidar_date`` supply dates for an output that was
    written without them (anything before 0.2.0); they never override dates
    already stamped on the rows.

    Adds ``fitted`` and ``w_area`` and makes sure the 0.2.0 columns exist, so
    outputs written by earlier versions (no ``geometry_basis``, no
    ``surface_area_m2``) can be reported too: their basis is re-derived from
    ``height_above_roof_m`` and the vintage recorded in the manifest, if any.
    """
    df = df.copy()
    df["fitted"] = df["panel_tilt_deg"].notna()
    if "status" not in df.columns:
        df["status"] = np.where(df["fitted"], MEASURED, NO_FIT_STATUS)

    if "surface_area_m2" not in df.columns:
        with np.errstate(invalid="ignore"):
            df["surface_area_m2"] = df["area_m2"] / np.cos(np.radians(df["panel_tilt_deg"]))
    if "point_density" not in df.columns:
        df["point_density"] = df["n_points_panel"] / df["area_m2"].where(df["area_m2"] > 0)

    for col in ("polygon_vintage", "lidar_date", "vintage_gap_days", "lidar_date_source",
                "roof_ref_source", "panel_fit_tolerance_m"):
        if col not in df.columns:
            df[col] = None
    supplied_gap = None
    if polygon_vintage is not None and df["polygon_vintage"].isna().all():
        df["polygon_vintage"] = polygon_vintage
    if lidar_date is not None and df["lidar_date"].isna().all():
        df["lidar_date"] = lidar_date
        df["lidar_date_source"] = "declared"
    if polygon_vintage is not None and lidar_date is not None \
            and df["vintage_gap_days"].isna().all():
        supplied_gap = (polygon_vintage - lidar_date).days
        df["vintage_gap_days"] = supplied_gap

    if "geometry_basis" not in df.columns:
        gap = supplied_gap
        if gap is None:
            gap = (manifest or {}).get("vintage", {}).get("vintage_gap_days")
        standoff = float(
            (manifest or {}).get("config", {}).get("heights", {})
            .get("min_panel_standoff_m", 0.05)
        )
        har = df["height_above_roof_m"]
        df["geometry_basis"] = [
            geometry_basis(
                fit_ok=bool(f),
                standoff_passed=bool(pd.notna(h) and h >= standoff),
                standoff_screened=bool(pd.notna(h)),
                gap_days=gap,
            )
            for f, h in zip(df["fitted"], har, strict=True)
        ]
    df["w_area"] = df["surface_area_m2"].fillna(df["area_m2"]).astype(float)
    return df


def stratum_mask(df: pd.DataFrame, key: str) -> pd.Series:
    return df["fitted"] & df["geometry_basis"].isin(STRATA[key][1])


def headline_stratum(df: pd.DataFrame, min_rows: int = 30) -> str:
    """The stratum the report leads with: the panel basis when there is enough
    of it to describe, else every fitted polygon."""
    return "panel" if int(stratum_mask(df, "panel").sum()) >= min_rows else "all_fitted"


def _weights(sub: pd.DataFrame, weight: str) -> np.ndarray:
    return np.ones(len(sub)) if weight == "count" else sub["w_area"].to_numpy(dtype=float)


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #

def weighted_quantile(x, w, qs) -> np.ndarray:
    """Quantiles of ``x`` under weights ``w`` (midpoint CDF, linear interpolation)."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    qs = np.atleast_1d(np.asarray(qs, dtype=float))
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    if len(x) == 0:
        return np.full(len(qs), np.nan)
    order = np.argsort(x)
    x, w = x[order], w[order]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(qs, cdf, x)


def weighted_mean_sd(x, w) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    if len(x) == 0:
        return float("nan"), float("nan")
    mean = float(np.average(x, weights=w))
    return mean, float(np.sqrt(np.average((x - mean) ** 2, weights=w)))


def circular_stats(az_deg, w=None) -> dict[str, float]:
    """Circular mean direction, mean resultant length and circular SD.

    Azimuth wraps at 360, so an arithmetic mean is meaningless (350 and 10
    average to 180). ``resultant_length`` R runs from 0 (directions spread
    evenly) to 1 (all identical); ``circular_sd_deg`` is sqrt(-2 ln R).
    """
    az = np.asarray(az_deg, dtype=float)
    w = np.ones(len(az)) if w is None else np.asarray(w, dtype=float)
    ok = np.isfinite(az) & np.isfinite(w) & (w > 0)
    az, w = az[ok], w[ok]
    if len(az) == 0:
        return {"circular_mean_deg": np.nan, "resultant_length": np.nan,
                "circular_sd_deg": np.nan}
    rad = np.radians(az)
    c = float(np.average(np.cos(rad), weights=w))
    s = float(np.average(np.sin(rad), weights=w))
    r = float(np.hypot(c, s))
    return {
        "circular_mean_deg": float(np.degrees(np.arctan2(s, c)) % 360.0),
        "resultant_length": r,
        "circular_sd_deg": float(np.degrees(np.sqrt(-2.0 * np.log(r)))) if r > 0 else np.nan,
    }


def sector_index(az_deg, n_sectors: int) -> np.ndarray:
    """Index of the compass sector each azimuth falls in; sector 0 is centred
    on north."""
    width = 360.0 / n_sectors
    return (np.floor(((np.asarray(az_deg, dtype=float) + width / 2.0) % 360.0) / width)
            .astype(int))


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

def basis_composition(df: pd.DataFrame) -> pd.DataFrame:
    """Rows and array surface in each ``geometry_basis`` category."""
    n_total = len(df)
    area_total = float(df["w_area"].sum())
    rows = []
    for basis in GEOMETRY_BASIS:
        sub = df[df["geometry_basis"] == basis]
        rows.append({
            "geometry_basis": basis,
            "label": BASIS_LABELS[basis],
            "n": len(sub),
            "share_count": len(sub) / n_total if n_total else np.nan,
            "area_m2": float(sub["w_area"].sum()),
            "share_area": float(sub["w_area"].sum()) / area_total if area_total else np.nan,
        })
    return pd.DataFrame(rows)


BOOTSTRAP_REPLICATES = 200
BOOTSTRAP_MIN_ROWS = 20


def bootstrap_intervals(tilt: np.ndarray, az: np.ndarray, w: np.ndarray,
                        n_boot: int = BOOTSTRAP_REPLICATES, seed: int = 0) -> dict[str, float]:
    """95% intervals for the median tilt and the four facing shares.

    A Poisson bootstrap over polygons: each replicate reweights every polygon by
    an independent Poisson(1) draw, which for these sample sizes is the same as
    resampling polygons with replacement but needs no re-sorting. The interval
    reflects *sampling* variability across polygons — how much the statistic
    would move with a different draw of arrays from the same fleet — and not
    measurement error in any one polygon, nor the vintage question.

    ``az`` is NaN where a polygon has no defined azimuth. Returns NaN intervals
    below ``BOOTSTRAP_MIN_ROWS`` polygons.
    """
    keys = ["tilt_p50_deg", *[f"share_facing_{q}" for q in QUADRANTS]]
    out = {f"{k}_ci_{side}": np.nan for k in keys for side in ("lo", "hi")}
    n = len(tilt)
    if n < BOOTSTRAP_MIN_ROWS:
        return out
    order = np.argsort(tilt)
    x, w_sorted = tilt[order], w[order]
    has_az = np.isfinite(az)
    quad = np.full(n, -1)
    quad[has_az] = sector_index(az[has_az], 4)

    rng = np.random.default_rng(seed)
    medians = np.empty(n_boot)
    shares = np.empty((n_boot, 4))
    for b in range(n_boot):
        draw = rng.poisson(1.0, n)
        ww = w_sorted * draw[order]
        total = ww.sum()
        if total <= 0:
            medians[b] = np.nan
        else:
            cdf = (np.cumsum(ww) - 0.5 * ww) / total
            medians[b] = np.interp(0.5, cdf, x)
        wa = w * draw
        denom = wa[has_az].sum()
        for i in range(4):
            shares[b, i] = wa[quad == i].sum() / denom if denom > 0 else np.nan

    out["tilt_p50_deg_ci_lo"], out["tilt_p50_deg_ci_hi"] = (
        float(v) for v in np.nanpercentile(medians, [2.5, 97.5]))
    for i, q in enumerate(QUADRANTS):
        if np.isfinite(shares[:, i]).any():
            lo, hi = np.nanpercentile(shares[:, i], [2.5, 97.5])
            out[f"share_facing_{q}_ci_lo"], out[f"share_facing_{q}_ci_hi"] = float(lo), float(hi)
    return out


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Headline tilt and azimuth statistics per stratum and weighting, with 95%
    bootstrap intervals on the median tilt and the facing shares."""
    rows = []
    for key, (label, _) in STRATA.items():
        sub = df[stratum_mask(df, key)]
        for weight in WEIGHTS:
            w = _weights(sub, weight)
            row: dict = {"stratum": key, "stratum_label": label, "weight": weight,
                         "n": len(sub), "area_m2": float(sub["w_area"].sum())}
            tilt = sub["panel_tilt_deg"].to_numpy(dtype=float)
            mean, sd = weighted_mean_sd(tilt, w)
            q = weighted_quantile(tilt, w, [0.05, 0.25, 0.5, 0.75, 0.95])
            row.update({"tilt_mean_deg": mean, "tilt_sd_deg": sd, "tilt_p05_deg": q[0],
                        "tilt_p25_deg": q[1], "tilt_p50_deg": q[2], "tilt_p75_deg": q[3],
                        "tilt_p95_deg": q[4]})

            has_az = sub["panel_azimuth_deg"].notna().to_numpy()
            az = sub["panel_azimuth_deg"].to_numpy(dtype=float)[has_az]
            w_az = w[has_az]
            total = w.sum()
            row["azimuth_defined_share"] = float(w_az.sum() / total) if total > 0 else np.nan
            row.update({f"azimuth_{k}": v for k, v in circular_stats(az, w_az).items()})
            quad = sector_index(az, 4)
            for i, name in enumerate(QUADRANTS):
                row[f"share_facing_{name}"] = (
                    float(w_az[quad == i].sum() / w_az.sum()) if w_az.sum() > 0 else np.nan
                )
            row.update(bootstrap_intervals(
                tilt, sub["panel_azimuth_deg"].to_numpy(dtype=float), w))
            rows.append(row)
    return pd.DataFrame(rows)


def tilt_profile(df: pd.DataFrame, bin_deg: float = TILT_BIN_DEG,
                 max_deg: float = TILT_MAX_DEG) -> pd.DataFrame:
    """Share of polygons / array surface in each tilt bin, per stratum. The
    last bin is open-ended (``>= max_deg``)."""
    edges = np.arange(0.0, max_deg + bin_deg, bin_deg)
    rows = []
    for key, (label, _) in STRATA.items():
        sub = df[stratum_mask(df, key)]
        tilt = sub["panel_tilt_deg"].to_numpy(dtype=float)
        idx = np.minimum((tilt // bin_deg).astype(int), len(edges) - 1)
        for weight in WEIGHTS:
            w = _weights(sub, weight)
            total = w.sum()
            binned = np.bincount(idx, weights=w, minlength=len(edges)) if len(sub) else np.zeros(len(edges))
            counts = np.bincount(idx, minlength=len(edges)) if len(sub) else np.zeros(len(edges), int)
            for i, lo in enumerate(edges):
                rows.append({
                    "stratum": key, "stratum_label": label, "weight": weight,
                    "tilt_lo_deg": float(lo),
                    "tilt_hi_deg": float(lo + bin_deg) if i < len(edges) - 1 else np.inf,
                    "n": int(counts[i]),
                    "share": float(binned[i] / total) if total > 0 else np.nan,
                })
    return pd.DataFrame(rows)


def azimuth_profile(df: pd.DataFrame, n_sectors: int = 16) -> pd.DataFrame:
    """Share of polygons / array surface facing each compass sector, per
    stratum. Shares are of rows with a defined azimuth; near-horizontal arrays
    (no azimuth) are reported by ``summary_statistics.azimuth_defined_share``."""
    names = {16: COMPASS_16, 8: COMPASS_8, 4: QUADRANTS}.get(n_sectors)
    width = 360.0 / n_sectors
    rows = []
    for key, (label, _) in STRATA.items():
        sub = df[stratum_mask(df, key) & df["panel_azimuth_deg"].notna()]
        idx = sector_index(sub["panel_azimuth_deg"], n_sectors)
        for weight in WEIGHTS:
            w = _weights(sub, weight)
            total = w.sum()
            binned = np.bincount(idx, weights=w, minlength=n_sectors) if len(sub) else np.zeros(n_sectors)
            counts = np.bincount(idx, minlength=n_sectors) if len(sub) else np.zeros(n_sectors, int)
            for i in range(n_sectors):
                rows.append({
                    "stratum": key, "stratum_label": label, "weight": weight,
                    "sector": names[i] if names else str(i),
                    "azimuth_centre_deg": i * width,
                    "azimuth_lo_deg": (i * width - width / 2.0) % 360.0,
                    "azimuth_hi_deg": (i * width + width / 2.0) % 360.0,
                    "n": int(counts[i]),
                    "share": float(binned[i] / total) if total > 0 else np.nan,
                })
    return pd.DataFrame(rows)


JOINT_TILT_EDGES = (0.0, 10.0, 20.0, 30.0, 40.0, np.inf)


def tilt_azimuth_joint(df: pd.DataFrame) -> pd.DataFrame:
    """Joint share over 8 compass sectors x tilt classes, per stratum."""
    edges = np.asarray(JOINT_TILT_EDGES)
    rows = []
    for key, (label, _) in STRATA.items():
        sub = df[stratum_mask(df, key) & df["panel_azimuth_deg"].notna()]
        s_idx = sector_index(sub["panel_azimuth_deg"], 8)
        t_idx = np.clip(np.searchsorted(edges, sub["panel_tilt_deg"].to_numpy(dtype=float),
                                        side="right") - 1, 0, len(edges) - 2)
        for weight in WEIGHTS:
            w = _weights(sub, weight)
            total = w.sum()
            for si, sname in enumerate(COMPASS_8):
                for ti in range(len(edges) - 1):
                    m = (s_idx == si) & (t_idx == ti)
                    rows.append({
                        "stratum": key, "stratum_label": label, "weight": weight,
                        "sector": sname, "tilt_lo_deg": float(edges[ti]),
                        "tilt_hi_deg": float(edges[ti + 1]),
                        "n": int(m.sum()),
                        "share": float(w[m].sum() / total) if total > 0 else np.nan,
                    })
    return pd.DataFrame(rows)


def _pct(series: pd.Series, qs=(10, 50, 90)) -> dict[str, float]:
    v = series.dropna().to_numpy(dtype=float)
    if len(v) == 0:
        return {f"p{q}": np.nan for q in qs}
    return {f"p{q}": float(np.percentile(v, q)) for q in qs}


def fit_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of the per-polygon quality measures."""
    fitted = df[df["fitted"]]
    specs = [
        ("panel_rmse_m", "Plane-fit RMSE", "m", fitted),
        ("panel_fit_tolerance_m", "Fit tolerance used", "m", fitted),
        ("panel_tilt_unc_deg", "Tilt uncertainty (1 sigma)", "deg", fitted),
        ("panel_azimuth_unc_deg", "Azimuth uncertainty (1 sigma)", "deg", fitted),
        ("point_density", "Point density in polygon", "pts/m2", df),
        ("n_points_panel", "Points in polygon", "count", df),
        ("area_m2", "Polygon plan area", "m2", df),
    ]
    rows = []
    for col, label, unit, frame in specs:
        values = pd.to_numeric(frame[col], errors="coerce")
        rows.append({"measure": col, "label": label, "unit": unit,
                     "n": int(values.notna().sum()), **_pct(values)})
    return pd.DataFrame(rows)


def roof_relation(df: pd.DataFrame) -> pd.DataFrame:
    """Panel-versus-roof measures, for rows with a usable roof reference."""
    fitted = df[df["fitted"]]
    n_fit = len(fitted)
    rows = []
    for col, label, unit in [
        ("height_above_roof_m", "Height of panel plane above roof plane", "m"),
        ("panel_roof_angle_deg", "Angle between panel and roof planes", "deg"),
        ("roof_tilt_deg", "Roof tilt", "deg"),
        ("height_above_ground_m", "Height above ground", "m"),
    ]:
        rows.append({"measure": col, "label": label, "unit": unit,
                     "n": int(fitted[col].notna().sum()),
                     "share_of_fitted": fitted[col].notna().sum() / n_fit if n_fit else np.nan,
                     **_pct(fitted[col])})
    return pd.DataFrame(rows)


def flag_counts(df: pd.DataFrame) -> pd.DataFrame:
    from pv_geom.schema import FLAG_DESCRIPTIONS

    counts: dict[str, int] = {}
    for flags in df["flags"]:
        if flags is None:
            continue
        for f in set(flags):
            counts[f] = counts.get(f, 0) + 1
    n = len(df)
    rows = [{"flag": f, "n": c, "share": c / n if n else np.nan,
             "description": FLAG_DESCRIPTIONS.get(f, "")}
            for f, c in sorted(counts.items(), key=lambda kv: -kv[1])]
    return pd.DataFrame(rows, columns=["flag", "n", "share", "description"])


def coverage(df: pd.DataFrame, manifest: dict | None = None) -> pd.DataFrame:
    """The funnel from input polygons to rows whose geometry describes panels."""
    counts = (manifest or {}).get("counts", {})
    # Outputs before schema 0.3 held rows only for covered polygons; the input
    # count then comes from the manifest.
    n_input = max(len(df), int(counts.get("polygons") or 0))
    steps = [
        ("Input polygons", n_input),
        ("Covered by LiDAR", int(df["status"].isin([MEASURED, NO_FIT_STATUS]).sum())),
        ("Plane fitted", int(df["fitted"].sum())),
        ("Panel basis (confirmed or by vintage)", int(stratum_mask(df, "panel").sum())),
    ]
    first = steps[0][1] if steps else 0
    return pd.DataFrame([
        {"step": s, "n": n, "share_of_first": n / first if first else np.nan}
        for s, n in steps
    ])


def status_table(df: pd.DataFrame) -> pd.DataFrame:
    """Every input polygon by what happened to it."""
    n = len(df)
    counts = df["status"].value_counts()
    return pd.DataFrame([
        {"status": s, "n": int(counts.get(s, 0)),
         "share": counts.get(s, 0) / n if n else np.nan, "description": desc}
        for s, desc in STATUS_DESCRIPTIONS.items()
    ])


def fit_failure_table(df: pd.DataFrame) -> pd.DataFrame:
    """Why covered polygons have no fit."""
    failed = df[df["status"] == NO_FIT_STATUS]
    counts = failed["fit_failure"].value_counts() if "fit_failure" in df.columns else {}
    n = len(failed)
    return pd.DataFrame([
        {"fit_failure": f, "n": int(counts.get(f, 0)),
         "share_of_no_fit": counts.get(f, 0) / n if n else np.nan, "description": desc}
        for f, desc in FIT_FAILURE_DESCRIPTIONS.items()
    ])


def facets_table(df: pd.DataFrame) -> pd.DataFrame:
    """Measured polygons by the number of distinct facets they hold."""
    measured = df[df["fitted"]]
    n = len(measured)
    planes = measured["n_planes_detected"].fillna(1).astype(int)
    counts = planes.value_counts().sort_index()
    return pd.DataFrame([
        {"facets": int(k), "n": int(v), "share": v / n if n else np.nan,
         "area_m2": float(measured.loc[planes == k, "w_area"].sum())}
        for k, v in counts.items()
    ], columns=["facets", "n", "share", "area_m2"])


def all_tables(df: pd.DataFrame, manifest: dict | None = None) -> dict[str, pd.DataFrame]:
    """Every report table, keyed by the file stem it is written under."""
    return {
        "coverage": coverage(df, manifest),
        "status": status_table(df),
        "fit_failure": fit_failure_table(df),
        "facets": facets_table(df),
        "geometry_basis": basis_composition(df),
        "summary_statistics": summary_statistics(df),
        "tilt_profile": tilt_profile(df),
        "azimuth_profile_16": azimuth_profile(df, 16),
        "azimuth_profile_8": azimuth_profile(df, 8),
        "tilt_azimuth_joint": tilt_azimuth_joint(df),
        "roof_relation": roof_relation(df),
        "fit_quality": fit_quality(df),
        "flags": flag_counts(df),
    }
