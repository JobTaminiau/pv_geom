"""Reader-facing tables: subsets of the CSV statistics, formatted for a page."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pv_geom.report import stats
from pv_geom.report.fmt import integer as _i
from pv_geom.report.fmt import num as _n
from pv_geom.report.fmt import pct as _p
from pv_geom.vintage import BASIS_DESCRIPTIONS


def display_tables(tables: dict[str, pd.DataFrame], headline: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    cov = tables["coverage"]
    out["coverage"] = pd.DataFrame({
        "Step": cov["step"], "Polygons": cov["n"].map(_i),
        "Share": cov["share_of_first"].map(_p)})

    gb = tables["geometry_basis"]
    out["geometry_basis"] = pd.DataFrame({
        "Basis": gb["label"], "Polygons": gb["n"].map(_i),
        "Share of polygons": gb["share_count"].map(_p),
        "Share of array surface": gb["share_area"].map(_p),
        "Meaning": gb["geometry_basis"].map(BASIS_DESCRIPTIONS)})

    s = tables["summary_statistics"]
    s = s[s["weight"] == "area"]
    out["summary_statistics"] = pd.DataFrame({
        "Stratum": s["stratum_label"], "Polygons": s["n"].map(_i),
        "Tilt median (°)": s["tilt_p50_deg"].map(_n),
        "Tilt IQR (°)": [f"{_n(a)}–{_n(b)}" for a, b in
                         zip(s["tilt_p25_deg"], s["tilt_p75_deg"], strict=True)],
        "Tilt mean (°)": s["tilt_mean_deg"].map(_n),
        "Azimuth mean (°)": s["azimuth_circular_mean_deg"].map(lambda x: _n(x, 0)),
        "Concentration R": s["azimuth_resultant_length"].map(lambda x: _n(x, 2)),
        "Facing S": s["share_facing_S"].map(_p), "Facing E": s["share_facing_E"].map(_p),
        "Facing W": s["share_facing_W"].map(_p), "Facing N": s["share_facing_N"].map(_p)})

    def _two(table: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
        t = table[table["weight"] == "area"]
        a = t[t["stratum"] == headline].set_index(key_cols)["share"]
        b = t[t["stratum"] == "all_fitted"].set_index(key_cols)["share"]
        return pd.DataFrame({stats.STRATA[headline][0]: a, "All fitted polygons": b}).reset_index()

    tp = _two(tables["tilt_profile"], ["tilt_lo_deg", "tilt_hi_deg"])
    tilt = pd.DataFrame({"Tilt (°)": [
        f"{lo:.0f}–{hi:.0f}" if np.isfinite(hi) else f"≥ {lo:.0f}"
        for lo, hi in zip(tp["tilt_lo_deg"], tp["tilt_hi_deg"], strict=True)]})
    az = _two(tables["azimuth_profile_8"], ["sector", "azimuth_centre_deg"])
    az = az.sort_values("azimuth_centre_deg")
    azt = pd.DataFrame({"Facing": az["sector"].to_numpy(),
                        "Centre (°)": az["azimuth_centre_deg"].map(lambda x: f"{x:.0f}").to_numpy()})
    for frame, src in ((tilt, tp), (azt, az)):
        for col in dict.fromkeys([stats.STRATA[headline][0], "All fitted polygons"]):
            frame[col] = src[col].map(_p).to_numpy()
    out["tilt_profile"] = tilt
    out["azimuth_profile"] = azt

    rr = tables["roof_relation"]
    out["roof_relation"] = pd.DataFrame({
        "Measure": rr["label"], "Unit": rr["unit"], "Polygons": rr["n"].map(_i),
        "Share of fitted": rr["share_of_fitted"].map(_p),
        "p10": rr["p10"].map(lambda x: _n(x, 2)), "Median": rr["p50"].map(lambda x: _n(x, 2)),
        "p90": rr["p90"].map(lambda x: _n(x, 2))})

    fq = tables["fit_quality"]
    out["fit_quality"] = pd.DataFrame({
        "Measure": fq["label"], "Unit": fq["unit"], "Polygons": fq["n"].map(_i),
        "p10": fq["p10"].map(lambda x: _n(x, 3)), "Median": fq["p50"].map(lambda x: _n(x, 3)),
        "p90": fq["p90"].map(lambda x: _n(x, 3))})

    fl = tables["flags"]
    out["flags"] = pd.DataFrame({
        "Flag": fl["flag"], "Polygons": fl["n"].map(_i), "Share": fl["share"].map(_p),
        "Meaning": fl["description"]})
    return out
