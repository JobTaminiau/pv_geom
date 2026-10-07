"""Build the report for a finished run.

One call turns an output prefix into everything a write-up needs::

    <out>/report.html          self-contained report (figures embedded)
    <out>/report.md            the same content as Markdown, figures linked
    <out>/methods.md           a methods paragraph filled in with this run's parameters
    <out>/summary.json         the headline numbers, machine-readable
    <out>/tables/*.csv         every statistic behind the report
    <out>/figures/*.{png,pdf,svg}
    <out>/dataset/             the consolidated dataset + data dictionary

The pieces: :mod:`data` (load + headline numbers), :mod:`stats` (tables),
:mod:`figures`, :mod:`text` (prose), :mod:`tables` (display tables),
:mod:`render` (HTML / Markdown) and :mod:`export` (release dataset).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pv_geom.errors import InputError
from pv_geom.report import figures as figs
from pv_geom.report import stats
from pv_geom.report.data import load_run, summarise
from pv_geom.report.export import export_dataset as write_dataset
from pv_geom.report.render import render_html, render_markdown
from pv_geom.report.tables import display_tables
from pv_geom.report.text import methods_text
from pv_geom.schema import data_dictionary, output_schema


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


def build_report(
    output_uri: str | Path,
    out_dir: str | Path | None = None,
    *,
    title: str | None = None,
    area_name: str | None = None,
    export_dataset: bool = True,
    headline: str | None = None,
    weight: str = "area",
    polygon_vintage=None,
    lidar_date=None,
    regions: str | Path | None = None,
    region_col: str | None = None,
) -> ReportResult:
    """Build tables, figures, reports and the release dataset for a run.

    ``headline`` names the stratum the report leads with (``panel``,
    ``all_fitted``, ``surface_unresolved``, ``unscreened``); by default the
    panel basis when it has enough rows to describe. ``weight`` is ``area``
    (array surface) or ``count`` (per polygon). Every table file carries all
    strata and both weightings whatever is chosen here. ``area_name`` defaults
    to ``study.name`` from the run's config. ``regions`` (a polygon layer) with
    ``region_col`` (its name column) adds a per-region table and map.
    """
    if (regions is None) != (region_col is None):
        raise InputError("regions and region_col go together",
                         "pass both --regions <layer> and --region-col <name column>")
    if weight not in stats.WEIGHTS:
        raise InputError(f"unknown weight {weight!r}", f"use one of {list(stats.WEIGHTS)}")
    if headline is not None and headline not in stats.STRATA:
        raise InputError(f"unknown headline stratum {headline!r}",
                         f"use one of {list(stats.STRATA)}")
    if out_dir is None:
        if str(output_uri).startswith("s3://"):
            raise ValueError("pass out_dir when reporting on an s3:// output")
        out_dir = Path(output_uri) / "report"
    out = Path(out_dir)
    tables_dir, figures_dir = out / "tables", out / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)

    gdf, df, manifest = load_run(output_uri, polygon_vintage=polygon_vintage,
                                 lidar_date=lidar_date)
    headline = headline or stats.headline_stratum(df)
    area_name = area_name or (manifest.get("config", {}).get("study") or {}).get("name")
    tables = stats.all_tables(df, manifest)
    for name, table in tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)

    s = summarise(df, tables, manifest, headline, weight)
    specs = {spec.name: spec for spec in figs.all_figures(df, headline, manifest, weight)}
    region_display = None
    if regions is not None and region_col is not None:
        from pv_geom.report import regions as rg

        names, layer = rg.assign_regions(gdf, regions, region_col)
        by_region = rg.regions_table(df, names, headline, weight)
        by_region.to_csv(tables_dir / "regions.csv", index=False)
        region_display = rg.display_table(by_region)
        region_fig = rg.fig_regions(layer, by_region, headline)
        if region_fig is not None:
            specs[region_fig.name] = region_fig
        gdf = gdf.assign(region=names)           # carried into the release dataset
    fig_paths = {name: figs.save(spec, figures_dir) for name, spec in specs.items()}

    dataset_dir: Path | None = None
    if export_dataset:
        dataset_dir = out / "dataset"
        dictionary = write_dataset(gdf, df, manifest, dataset_dir)
    else:
        dictionary = pd.DataFrame(data_dictionary(
            output_schema("mounting_type" in gdf.columns)))
        dictionary = dictionary[dictionary["column"].isin(gdf.columns)]

    title = title or (f"PV array geometry: {area_name}" if area_name else "PV array geometry")
    subtitle = (f"{s['n_rows']:,} polygons · polygon vintage "
                f"{s.get('polygon_vintage_declared_as') or s['polygon_vintage'] or 'not declared'}"
                f" · LiDAR {s['lidar_date_max'] or 'date unknown'} · pv-geom v{s['pkg_version']}")
    methods = methods_text(s, manifest, area_name)
    disp = display_tables(tables, headline, weight)
    if region_display is not None:
        disp["regions"] = region_display

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
