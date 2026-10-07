"""Build the report for a finished run.

One call turns an output prefix into everything a write-up needs::

    <out>/report.html          self-contained report (figures embedded)
    <out>/report.md            the same content as Markdown, figures linked
    <out>/methods.md           a methods paragraph filled in with this run's parameters
    <out>/summary.json         the headline numbers, machine-readable
    <out>/tables/*.csv         every statistic behind the report
    <out>/figures/*.{png,pdf,svg}
    <out>/dataset/             the consolidated dataset + data dictionary
"""

from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pv_geom import __version__
from pv_geom.io.output import read_manifest, read_output
from pv_geom.report import figures as figs
from pv_geom.report import stats
from pv_geom.schema import FLAG_DESCRIPTIONS, data_dictionary, output_schema
from pv_geom.vintage import BASIS_DESCRIPTIONS, parse_vintage


@dataclass
class ReportResult:
    out_dir: Path
    html: Path
    markdown: Path
    methods: Path
    summary: dict
    tables_dir: Path
    figures_dir: Path
    dataset_dir: Path | None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Headline numbers
# --------------------------------------------------------------------------- #

def _compass(deg: float) -> str:
    if not np.isfinite(deg):
        return "n/a"
    return stats.COMPASS_16[int(((deg + 11.25) % 360) // 22.5)]


def _iso(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v.isoformat() if hasattr(v, "isoformat") else str(v)


def summarise(df: pd.DataFrame, tables: dict[str, pd.DataFrame], manifest: dict,
              headline: str) -> dict:
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
    rmse = df.loc[df["fitted"], "panel_rmse_m"].dropna()
    return {
        "pkg_version": manifest.get("pkg_version") or __version__,
        "run_id": manifest.get("run_id"),
        "crs": manifest.get("crs") or manifest.get("config", {}).get("crs", {}).get("target"),
        "inputs": manifest.get("inputs", {}),
        "n_input_polygons": manifest.get("counts", {}).get("polygons"),
        "n_rows": int(len(df)),
        "n_fitted": n_fit,
        "fit_rate": n_fit / len(df) if len(df) else float("nan"),
        "total_plan_area_m2": float(df["area_m2"].sum()),
        "polygon_vintage": _iso(poly.max()) if len(poly) else v.get("polygon_vintage"),
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
        "headline_label": stats.STRATA[headline][0],
        "headline": {"area": _row(headline, "area"), "count": _row(headline, "count")},
        "all_fitted": {"area": _row("all_fitted", "area"), "count": _row("all_fitted", "count")},
        "panel_rmse_m_p50": float(rmse.median()) if len(rmse) else float("nan"),
        "panel_rmse_m_p90": float(rmse.quantile(0.9)) if len(rmse) else float("nan"),
        "point_density_p50": float(df["point_density"].median()),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------------------- #
# Display tables (subsets of the CSVs, formatted for reading)
# --------------------------------------------------------------------------- #

def _p(x, digits=1) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{100 * x:.{digits}f}%"


def _n(x, digits=1) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:,.{digits}f}"


def _i(x) -> str:
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{int(x):,}"


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


# --------------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------------- #

def _years(days) -> str:
    return f"{abs(days) / 365.25:.1f} years"


def vintage_statement(s: dict) -> str:
    lid = s["lidar_date_max"]
    pol = s["polygon_vintage"]
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
        return (f"The polygons derive from imagery captured by {pol}; the LiDAR capture "
                f"date could not be established.")
    if gap is not None and gap > 0:
        return (f"The polygons derive from imagery captured by {pol}; the LiDAR was captured "
                f"by {lid} ({src}). The polygons are therefore {_years(gap)} newer than the "
                f"LiDAR: an installation built in that interval has a polygon but was not "
                f"there to be measured, and its fitted plane is the surface that preceded it.")
    return (f"The polygons derive from imagery captured by {pol}; the LiDAR was captured by "
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
        f"returns were inliers. Tilt is the angle of the plane from horizontal and azimuth "
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


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

SECTIONS = [
    # (heading, figure names, display-table keys, intro)
    ("Coverage", [], ["coverage"],
     "How many input polygons reached each stage."),
    ("Input vintages and geometry basis", ["vintage_timeline", "geometry_basis"],
     ["geometry_basis"], None),
    ("Array geometry", ["geometry_overview"], ["summary_statistics"],
     "Statistics are weighted by array surface; per-polygon versions are in "
     "tables/summary_statistics.csv."),
    ("Tilt profile", ["tilt_distribution"], ["tilt_profile"],
     "Share of array surface in each tilt class."),
    ("Orientation profile", ["azimuth_rose", "tilt_azimuth_joint"], ["azimuth_profile"],
     "Share of array surface facing each compass sector (arrays with a defined azimuth)."),
    ("Array against roof", ["roof_relation"], ["roof_relation"], None),
    ("Measurement quality", ["fit_quality"], ["fit_quality", "flags"], None),
    ("Spatial distribution", ["spatial_distribution"], [], None),
]

CSS = """
:root{color-scheme:light;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;
--surface:#ffffff;--page:#f9f9f7;--accent:#2a78d6}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;line-height:1.2;margin:0 0 4px}
h2{font-size:19px;margin:44px 0 10px;padding-top:18px;border-top:1px solid var(--line)}
p{margin:0 0 12px}.sub{color:var(--ink2);margin-bottom:24px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 16px}
.tile .v{font-size:26px;font-weight:650;line-height:1.15}
.tile .l{color:var(--ink2);font-size:13px;margin-top:2px}
.note{background:var(--surface);border:1px solid var(--line);border-left:3px solid var(--accent);
border-radius:6px;padding:12px 16px;margin:16px 0}
figure{margin:18px 0;background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:16px}
figure img{display:block;max-width:100%;height:auto;margin:0 auto}
figcaption{color:var(--ink2);font-size:13px;margin-top:10px}
.scroll{overflow-x:auto;margin:12px 0}
table{border-collapse:collapse;width:100%;font-size:13px;background:var(--surface)}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--ink2);font-weight:600;white-space:nowrap}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
code{font-size:12.5px;background:#f0efec;padding:1px 5px;border-radius:4px}
footer{color:var(--muted);font-size:12.5px;margin-top:40px}
ul{padding-left:20px;margin:0 0 12px}li{margin-bottom:6px}
"""

_TEXT_COLS = {"Step", "Basis", "Meaning", "Stratum", "Measure", "Unit", "Flag", "Facing",
              "Tilt (°)", "column", "type", "unit", "nullable", "description", "File",
              "Contents"}


def _html_table(df: pd.DataFrame) -> str:
    head = "".join(
        f'<th{"" if c in _TEXT_COLS else " class=num"}>{html.escape(str(c))}</th>'
        for c in df.columns)
    body = "".join(
        "<tr>" + "".join(
            f'<td{"" if c in _TEXT_COLS else " class=num"}>{html.escape(str(v))}</td>'
            for c, v in zip(df.columns, row, strict=True)) + "</tr>"
        for row in df.itertuples(index=False))
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _md_table(df: pd.DataFrame) -> str:
    def cell(v) -> str:
        return str(v).replace("|", "\\|")
    lines = ["| " + " | ".join(cell(c) for c in df.columns) + " |",
             "|" + "|".join(" --- " if c in _TEXT_COLS else " ---: " for c in df.columns) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |"
              for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _tiles(s: dict) -> list[tuple[str, str]]:
    h = s["headline"]["area"]
    gap = s["vintage_gap_days"]
    tiles = [
        (f"{s['n_rows']:,}", "polygons measured"),
        (_p(s["fit_rate"]), "with a fitted plane"),
        (f"{h['tilt_median_deg']:.1f}°", f"median tilt · {s['headline_label'].lower()}"),
        (_p(h["share_facing_S"]), "of array surface faces south"),
    ]
    if gap is not None and np.isfinite(gap):
        tiles.append((f"{gap / 365.25:+.1f} yr", "polygon vintage minus LiDAR date"))
    return tiles


def render_html(title: str, subtitle: str, s: dict, specs: dict[str, figs.FigureSpec],
                fig_paths: dict[str, dict[str, Path]], disp: dict[str, pd.DataFrame],
                methods: str, files: pd.DataFrame, dictionary: pd.DataFrame) -> str:
    parts = [f"<!doctype html><html lang=en><head><meta charset=utf-8>"
             f"<meta name=viewport content='width=device-width,initial-scale=1'>"
             f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>",
             f"<h1>{html.escape(title)}</h1><p class=sub>{html.escape(subtitle)}</p>",
             '<div class="tiles">' + "".join(
                 f'<div class="tile"><div class="v">{html.escape(v)}</div>'
                 f'<div class="l">{html.escape(lab)}</div></div>' for v, lab in _tiles(s))
             + "</div>",
             "<h2>Key findings</h2><ul>" + "".join(
                 f"<li>{html.escape(k)}</li>" for k in key_findings(s)) + "</ul>",
             f'<div class="note">{html.escape(vintage_statement(s))}</div>']
    for heading, fig_names, table_keys, intro in SECTIONS:
        present = [n for n in fig_names if n in specs]
        if not present and not table_keys:
            continue
        parts.append(f"<h2>{html.escape(heading)}</h2>")
        if intro:
            parts.append(f"<p>{html.escape(intro)}</p>")
        for name in present:
            png = base64.b64encode(fig_paths[name]["png"].read_bytes()).decode("ascii")
            parts.append(
                f'<figure><img alt="{html.escape(specs[name].title)}" '
                f'src="data:image/png;base64,{png}">'
                f"<figcaption><b>{html.escape(specs[name].title)}.</b> "
                f"{html.escape(specs[name].caption)}</figcaption></figure>")
        for key in table_keys:
            if len(disp[key]):
                parts.append(_html_table(disp[key]))
    parts += ["<h2>Methods</h2>", f"<p>{html.escape(methods)}</p>",
              "<h2>Files</h2>", _html_table(files),
              "<h2>Data dictionary</h2>", _html_table(dictionary),
              f"<footer>Generated {html.escape(s['generated_utc'])} by pv-geom "
              f"v{html.escape(__version__)} · run {html.escape(str(s['run_id'])[:8])} · "
              f"CRS {html.escape(str(s['crs']))}</footer></main></body></html>"]
    return "\n".join(parts)


def render_markdown(title: str, subtitle: str, s: dict, specs: dict[str, figs.FigureSpec],
                    disp: dict[str, pd.DataFrame], methods: str, files: pd.DataFrame) -> str:
    parts = [f"# {title}", "", subtitle, "", "## Key findings", ""]
    parts += [f"- {k}" for k in key_findings(s)]
    parts += ["", f"> {vintage_statement(s)}", ""]
    for heading, fig_names, table_keys, intro in SECTIONS:
        present = [n for n in fig_names if n in specs]
        if not present and not table_keys:
            continue
        parts += [f"## {heading}", ""]
        if intro:
            parts += [intro, ""]
        for name in present:
            parts += [f"![{specs[name].title}](figures/{name}.png)", "",
                      f"**{specs[name].title}.** {specs[name].caption}", ""]
        for key in table_keys:
            if len(disp[key]):
                parts += [_md_table(disp[key]), ""]
    parts += ["## Methods", "", methods, "", "## Files", "", _md_table(files), "",
              f"_Generated {s['generated_utc']} by pv-geom v{__version__}; run "
              f"{str(s['run_id'])[:8]}; CRS {s['crs']}._", ""]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Dataset export
# --------------------------------------------------------------------------- #

def export_dataset(gdf, df: pd.DataFrame, manifest: dict, dataset_dir: Path) -> pd.DataFrame:
    """Write the consolidated release dataset and its dictionary."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    out = gdf.copy()
    for col in ("geometry_basis", "surface_area_m2", "point_density"):
        if col not in out.columns:
            out[col] = df[col].to_numpy()
    for col in ("centroid_lon", "centroid_lat"):
        if col in df.columns:
            out[col] = df[col].to_numpy()
    out.to_parquet(dataset_dir / "pv_geom.parquet", index=False)

    flat = pd.DataFrame(out.drop(columns="geometry"))
    for col in ("flags", "lidar_tile_ids"):
        if col in flat.columns:
            flat[col] = flat[col].map(lambda v: ";".join(v) if v is not None else "")
    flat.to_csv(dataset_dir / "pv_geom.csv", index=False)

    schema = output_schema("mounting_type" in out.columns)
    known = {f.name for f in schema}
    dictionary = pd.DataFrame(data_dictionary(schema))
    dictionary = dictionary[dictionary["column"].isin(out.columns)]
    extra = [
        {"column": "centroid_lon", "type": "double", "unit": "deg", "nullable": "no",
         "description": "Polygon centroid longitude (WGS 84)."},
        {"column": "centroid_lat", "type": "double", "unit": "deg", "nullable": "no",
         "description": "Polygon centroid latitude (WGS 84)."},
    ]
    dictionary = pd.concat(
        [dictionary, pd.DataFrame([e for e in extra if e["column"] in out.columns])],
        ignore_index=True)
    legacy = [c for c in out.columns if c not in known and c not in {"centroid_lon", "centroid_lat"}]
    if legacy:
        dictionary = pd.concat([dictionary, pd.DataFrame(
            [{"column": c, "type": str(out[c].dtype), "unit": "", "nullable": "",
              "description": "Column from an earlier pv-geom version."} for c in legacy])],
            ignore_index=True)
    dictionary.to_csv(dataset_dir / "data_dictionary.csv", index=False)

    flag_rows = pd.DataFrame(
        [{"flag": k, "description": v} for k, v in FLAG_DESCRIPTIONS.items()])
    basis_rows = pd.DataFrame(
        [{"geometry_basis": k, "description": v} for k, v in BASIS_DESCRIPTIONS.items()])
    flag_rows.to_csv(dataset_dir / "flag_definitions.csv", index=False)
    basis_rows.to_csv(dataset_dir / "geometry_basis_definitions.csv", index=False)
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return dictionary


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def build_report(
    output_uri: str | Path,
    out_dir: str | Path | None = None,
    *,
    title: str | None = None,
    area_name: str | None = None,
    export_dataset: bool = True,
    weight: str = "area",
    polygon_vintage=None,
    lidar_date=None,
) -> ReportResult:
    """Build tables, figures, reports and the release dataset for a run."""
    if out_dir is None:
        if str(output_uri).startswith("s3://"):
            raise ValueError("pass out_dir when reporting on an s3:// output")
        out_dir = Path(output_uri) / "report"
    out = Path(out_dir)
    tables_dir, figures_dir = out / "tables", out / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)

    gdf, df, manifest = load_run(output_uri, polygon_vintage=polygon_vintage,
                                 lidar_date=lidar_date)
    headline = stats.headline_stratum(df)
    tables = stats.all_tables(df, manifest)
    for name, table in tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)

    s = summarise(df, tables, manifest, headline)
    specs = {spec.name: spec for spec in figs.all_figures(df, headline, manifest, weight)}
    fig_paths = {name: figs.save(spec, figures_dir) for name, spec in specs.items()}

    dataset_dir: Path | None = None
    if export_dataset:
        dataset_dir = out / "dataset"
        dictionary = _export(gdf, df, manifest, dataset_dir)
    else:
        dictionary = pd.DataFrame(data_dictionary(
            output_schema("mounting_type" in gdf.columns)))
        dictionary = dictionary[dictionary["column"].isin(gdf.columns)]

    title = title or (f"PV array geometry: {area_name}" if area_name else "PV array geometry")
    subtitle = (f"{s['n_rows']:,} polygons · polygon vintage {s['polygon_vintage'] or 'not declared'}"
                f" · LiDAR {s['lidar_date_max'] or 'date unknown'} · pv-geom v{s['pkg_version']}")
    methods = methods_text(s, manifest, area_name)
    disp = display_tables(tables, headline)

    file_rows = [("report.html", "This report, self-contained"),
                 ("report.md", "The report as Markdown with linked figures"),
                 ("methods.md", "Methods paragraph with this run's parameters"),
                 ("summary.json", "Headline numbers, machine-readable"),
                 ("tables/*.csv", "Every statistic, by stratum and weighting"),
                 ("figures/*.png|pdf|svg", "Figures at 300 dpi and as editable vectors")]
    if dataset_dir is not None:
        file_rows += [("dataset/pv_geom.parquet", "Per-polygon dataset (GeoParquet)"),
                      ("dataset/pv_geom.csv", "The same without geometry, with centroid lon/lat"),
                      ("dataset/data_dictionary.csv", "Column definitions and units")]
    files = pd.DataFrame(file_rows, columns=["File", "Contents"])

    (out / "summary.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    (out / "methods.md").write_text(methods + "\n", encoding="utf-8")
    md = out / "report.md"
    md.write_text(render_markdown(title, subtitle, s, specs, disp, methods, files),
                  encoding="utf-8")
    page = out / "report.html"
    page.write_text(render_html(title, subtitle, s, specs, fig_paths, disp, methods, files,
                                dictionary), encoding="utf-8")
    return ReportResult(out_dir=out, html=page, markdown=md, methods=out / "methods.md",
                        summary=s, tables_dir=tables_dir, figures_dir=figures_dir,
                        dataset_dir=dataset_dir)


_export = export_dataset      # build_report's keyword of the same name shadows the function
