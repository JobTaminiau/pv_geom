"""The report's prose: the vintage statement, key findings and methods paragraph.

Written from the headline numbers (:func:`pv_geom.report.data.summarise`) and
the run's configuration, so the words always match the tables.
"""

from __future__ import annotations

import numpy as np

from pv_geom.report import stats
from pv_geom.report.fmt import pct as _p


def _years(days) -> str:
    return f"{abs(days) / 365.25:.1f} years"


def vintage_statement(s: dict) -> str:
    lid = s["lidar_date_max"]
    pol = s["polygon_vintage"]
    declared = s.get("polygon_vintage_declared_as")
    if pol is not None and declared and str(declared) != str(pol):
        pol = f"{declared} (counted as {pol})"
    gap = s["vintage_gap_days"]
    src = {"gps_time": "measured from per-point GPS time",
           "declared": "as declared for the run",
           "header_date": "taken from the LAS header, which records delivery rather than "
                          "flight and may be late"}.get(s.get("lidar_date_source") or "", "")
    if pol is None and lid is None:
        return ("Neither input date is recorded for this run, so the panel-standoff screen "
                "is the only evidence that panels were present when the LiDAR was flown.")
    if pol is None:
        return (f"The LiDAR was captured by {lid} ({src}). No polygon vintage was declared, "
                f"so the panel-standoff screen is the only evidence that panels were present "
                f"when the LiDAR was flown.")
    if lid is None:
        return (f"The polygons derive from imagery dated {pol}; the LiDAR capture "
                f"date could not be established.")
    if gap is not None and gap > 0:
        return (f"The polygons derive from imagery dated {pol}; the LiDAR was captured "
                f"by {lid} ({src}). The polygons are therefore {_years(gap)} newer than the "
                f"LiDAR: an installation built in that interval has a polygon but was not "
                f"there to be measured, and its fitted plane is the surface that preceded it.")
    return (f"The polygons derive from imagery dated {pol}; the LiDAR was captured by "
            f"{lid} ({src}). The polygons are no newer than the LiDAR, so each installation "
            f"was present when it was flown.")


def key_findings(s: dict) -> list[str]:
    h, hc = s["headline"]["area"], s["headline"]["count"]
    label = ("panel-basis polygons" if s["headline_stratum"] == "panel"
             else "fitted polygons")
    gb = s["geometry_basis"]
    panel_share = gb["panel_confirmed"]["share_count"] + gb["panel_by_vintage"]["share_count"]
    quad = {q: h[f"share_facing_{q}"] for q in stats.QUADRANTS}
    names = {"N": "north", "E": "east", "S": "south", "W": "west"}
    ranked = sorted(quad, key=lambda q: -quad[q] if np.isfinite(quad[q]) else 0)
    out = [
        f"{s['n_rows']:,} polygons were covered by LiDAR and a plane was fitted to "
        f"{s['n_fitted']:,} of them ({_p(s['fit_rate'])}), with a median fit residual of "
        f"{100 * s['panel_rmse_m_p50']:.1f} cm.",
        f"{_p(panel_share)} of polygons are on a panel basis — the fitted plane is known to "
        f"be the array — and the statistics below lead with that group "
        f"({h['n']:,} polygons)." if s["headline_stratum"] == "panel" else
        f"Too few polygons ({_p(panel_share)}) are on a panel basis to describe separately, "
        f"so the statistics below cover every fitted polygon.",
        f"Median tilt of {label} is {h['tilt_median_deg']:.1f}° weighted by array surface "
        f"(interquartile range {h['tilt_p25_deg']:.1f}–{h['tilt_p75_deg']:.1f}°); per polygon "
        f"it is {hc['tilt_median_deg']:.1f}°.",
        f"{_p(quad[ranked[0]])} of that surface faces {names[ranked[0]]} and "
        f"{_p(quad[ranked[1]])} faces {names[ranked[1]]}; the mean direction is "
        f"{h['azimuth_circular_mean_deg']:.0f}° ({h['azimuth_circular_mean_compass']}) with "
        f"concentration R = {h['azimuth_resultant_length']:.2f}.",
    ]
    if s["headline_stratum"] == "panel":
        a = s["all_fitted"]["area"]
        out.append(
            f"Across all fitted polygons the median tilt is {a['tilt_median_deg']:.1f}° and "
            f"{_p(a['share_facing_S'])} face south; the difference from the panel-basis "
            f"figures is the effect of including surfaces that may not be panels.")
    return out


def methods_text(s: dict, manifest: dict, area_name: str | None) -> str:
    cfg = manifest.get("config", {})
    pp = cfg.get("panel_plane", {})
    rp = cfg.get("roof_plane", {})
    ht = cfg.get("heights", {})
    where = f" in {area_name}" if area_name else ""
    footprints = (manifest.get("inputs") or {}).get("footprints")
    ring = ("clipped to the building footprint the polygon overlaps where one exists and "
            "left unclipped otherwise" if footprints else
            "not clipped to building footprints, as none were supplied")
    share = s.get("wide_tolerance_share_of_fitted")
    cap = pp.get("ransac_threshold_max_m")
    wide = ""
    if cap and share is not None and np.isfinite(share) and share > 0:
        wide = (f" Where no plane reached that consensus, the scatter of the returns about "
                f"the best plane was measured and the fit repeated with an inlier distance "
                f"of twice that scatter, up to {cap:g} m; {100 * share:.0f}% of fitted "
                f"polygons were fitted this way and are flagged.")
    if s["lidar_date_max"] is None:
        lid = ""
    elif s["lidar_date_min"] != s["lidar_date_max"]:
        lid = f" (LiDAR flown between {s['lidar_date_min']} and {s['lidar_date_max']})"
    else:
        lid = f" (LiDAR flown on {s['lidar_date_max']})"
    return (
        f"Array geometry{where} was measured with pv-geom v{s['pkg_version']} from airborne "
        f"LiDAR at {s['n_rows']:,} photovoltaic polygons. "
        f"{vintage_statement(s)} "
        f"For each polygon, LiDAR returns inside the outline (eroded by "
        f"{pp.get('erosion_m', 0.15):g} m to avoid edge returns) were fitted with a plane by "
        f"RANSAC (inlier distance {pp.get('ransac_threshold_m', 0.05):g} m, "
        f"{pp.get('max_iter', 200)} iterations) and refined by least squares on the inliers; "
        f"a fit was accepted when at least {100 * pp.get('min_inlier_frac', 0.6):.0f}% of "
        f"returns were inliers.{wide} Tilt is the angle of the plane from horizontal and azimuth "
        f"the compass direction of its downslope normal; azimuth is not reported below "
        f"{pp.get('tilt_floor_deg', 1.0):g}° of tilt. Uncertainties are the standard "
        f"deviation over {pp.get('bootstrap_samples', 50)} bootstrap resamples of the "
        f"inliers. The median point density was {s['point_density_p50']:.1f} returns per m² "
        f"and the median fit residual {100 * s['panel_rmse_m_p50']:.1f} cm{lid}. "
        f"A reference plane for the surrounding roof was fitted to returns in a "
        f"{rp.get('buffer_m', 3.0):g}–{rp.get('buffer_max_m', 5.0):g} m ring around each "
        f"polygon, {ring}; a polygon whose fitted plane stood at least "
        f"{100 * ht.get('min_panel_standoff_m', 0.05):.0f} cm above that reference was "
        f"classed as panel-confirmed. Polygons that could not be confirmed this way were "
        f"credited as panels only if the polygon vintage was no later than the LiDAR date. "
        f"Statistics are reported per polygon and weighted by array surface (plan area "
        f"divided by the cosine of tilt); azimuth is summarised with circular statistics."
    )
