"""Render the report as a self-contained HTML page and as Markdown.

Both walk the same ``SECTIONS`` outline, so adding a section — or a renderer —
is one edit.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

import numpy as np
import pandas as pd

from pv_geom import __version__
from pv_geom.report import figures as figs
from pv_geom.report.fmt import pct as _p
from pv_geom.report.text import key_findings, vintage_statement

SECTIONS = [
    # (heading, figure names, display-table keys, intro)
    ("Coverage", [], ["coverage", "status", "fit_failure"],
     "How many input polygons reached each stage, what happened to each, and why "
     "covered polygons have no fit."),
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


_TEXT_COLS = {"Status", "No fit because", "Step", "Basis", "Meaning", "Stratum", "Measure", "Unit", "Flag", "Facing",
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
