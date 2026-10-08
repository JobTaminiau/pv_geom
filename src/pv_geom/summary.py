"""At-a-glance statistics of an output table.

The one implementation behind the manifest's ``aggregate_stats`` and
``pv-geom describe-output``. It shares its definitions (compass sectors,
geometry-basis categories) with :mod:`pv_geom.report.stats`, which produces the
full, weighted, per-stratum tables.
"""

from __future__ import annotations

import pyarrow as pa

from pv_geom.report.stats import QUADRANTS, sector_index
from pv_geom.schema import STATUSES
from pv_geom.vintage import ARRAY_BASES, GEOMETRY_BASIS


def summarise_table(table: pa.Table) -> dict:
    """Manifest-friendly summary stats. The full profile tables live in
    ``pv_geom.report``; this is the at-a-glance version that travels with the
    data."""
    import numpy as np
    import pandas as pd

    df = table.to_pandas()
    n = len(df)
    out: dict = {"rows": int(n)}

    if "status" in df.columns:
        counts = df["status"].value_counts()
        out["status_counts"] = {s: int(counts.get(s, 0)) for s in STATUSES}
    if "fit_failure" in df.columns:
        out["fit_failure_counts"] = {
            str(k): int(v) for k, v in df["fit_failure"].dropna().value_counts().items()
        }

    fitted = df["panel_tilt_deg"].notna()
    out["fitted"] = int(fitted.sum())
    out["fit_rate"] = float(fitted.mean()) if n else 0.0

    if "geometry_basis" in df.columns:
        counts = df["geometry_basis"].value_counts()
        out["geometry_basis_counts"] = {b: int(counts.get(b, 0)) for b in GEOMETRY_BASIS}
        panel = df["geometry_basis"].isin(ARRAY_BASES)
        out["panel_basis_frac_of_fitted"] = (
            float(panel.sum() / fitted.sum()) if fitted.sum() else None
        )

    tilt = df.loc[fitted, "panel_tilt_deg"]
    if len(tilt):
        out["panel_tilt_deg"] = {
            f"p{q}": float(np.percentile(tilt, q)) for q in (10, 25, 50, 75, 90)
        }
    az = df["panel_azimuth_deg"].dropna()
    if len(az):
        quadrant = np.bincount(sector_index(az, 4), minlength=4)
        out["azimuth_quadrant_counts"] = {
            name: int(quadrant[i]) for i, name in enumerate(QUADRANTS)
        }

    if "on_building" in df.columns:
        ob = df["on_building"]
        out["on_building_count"] = int((ob == True).sum())      # noqa: E712 (nullable)
        out["off_building_count"] = int((ob == False).sum())    # noqa: E712
        out["on_building_unknown_count"] = int(ob.isna().sum())

    if "mounting_type" in df.columns:
        out["mounting_type_counts"] = df["mounting_type"].value_counts().to_dict()
        out["mounting_rule_counts"] = df["mounting_rule"].value_counts().to_dict()
        confidences = df["mounting_confidence"].dropna()
        if len(confidences):
            out["mounting_confidence_p50"] = float(confidences.median())
            out["mounting_confidence_p10"] = float(confidences.quantile(0.10))

    rmses = df["panel_rmse_m"].dropna()
    if len(rmses):
        out["panel_rmse_p50"] = float(rmses.median())
        out["panel_rmse_p90"] = float(rmses.quantile(0.90))
    flag_counts: dict[str, int] = {}
    flag_sets: list[set] = []
    for flags in df["flags"]:
        # `flags` may be a numpy array, list, or None; iterate uniformly.
        s = set(flags) if flags is not None else set()
        flag_sets.append(s)
        for f in s:
            flag_counts[f] = flag_counts.get(f, 0) + 1
    out["flag_counts"] = flag_counts

    # Panel-standoff screen: how many rows could be tested for panels being
    # physically above the roof, and how many passed.
    failed = pd.Series([("no_panel_standoff" in s) for s in flag_sets], index=df.index)
    unscreenable = pd.Series([("standoff_unscreenable" in s) for s in flag_sets], index=df.index)
    state = pd.Series("passed", index=df.index)
    state[failed] = "no_panel_standoff"
    state[unscreenable] = "unscreenable"
    out["standoff_screen"] = {
        "passed": int((state == "passed").sum()),
        "no_panel_standoff": int(failed.sum()),
        "unscreenable": int(unscreenable.sum()),
        "screened_frac": float((~unscreenable).mean()) if n else 0.0,
        "failed_frac_of_screened": (
            float(failed.sum() / (~unscreenable).sum()) if (~unscreenable).sum() else None
        ),
    }
    if "mounting_type" in df.columns:
        out["standoff_screen"]["by_mounting_type"] = {
            str(k): {str(kk): int(vv) for kk, vv in v.items()}
            for k, v in pd.crosstab(df["mounting_type"], state).to_dict("index").items()
        }

    har = df["height_above_roof_m"].dropna()
    if len(har):
        out["height_above_roof_p10"] = float(har.quantile(0.10))
        out["height_above_roof_p50"] = float(har.median())
        out["height_above_roof_p90"] = float(har.quantile(0.90))
    return out
