"""pv_geom Typer CLI."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pv_geom import __version__
from pv_geom.config import PVGeomConfig
from pv_geom.errors import PVGeomError
from pv_geom.utils.logging import add_file_log, configure_logging

app = typer.Typer(
    name="pv-geom",
    help="Tilt, orientation and height of solar PV polygons, measured from LiDAR.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def _root(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug detail"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Warnings and errors only"),
) -> None:
    configure_logging(verbose=verbose, quiet=quiet)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(f"pv-geom v{__version__}")


@app.command("validate-config")
def validate_config(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
) -> None:
    """Validate a YAML config against the Pydantic schema."""
    cfg = PVGeomConfig.from_yaml(config)
    console.print(f"[green]OK[/green] {config}")
    console.print(f"config_hash: {cfg.hash()}")
    console.print(f"backend: {cfg.compute.backend}")
    console.print(f"target CRS: {cfg.crs.target}")
    console.print(f"polygon vintage: {cfg.vintage.polygon_vintage}")
    console.print(f"lidar date: {cfg.vintage.lidar_date or 'measured from GPS time'}")


@app.command()
def run(
    polygons: str = typer.Option(
        ..., help="PV polygon layer: GeoParquet / GPKG / GeoJSON / SHP (path or s3://)"
    ),
    lidar_prefix: str = typer.Option(..., help="Directory or S3 prefix holding the LAZ tiles"),
    tile_index: str = typer.Option(..., help="LiDAR tile index: GeoParquet/GPKG/SHP/.zip"),
    output: str = typer.Option(..., help="Output directory or s3:// prefix"),
    polygon_vintage: str | None = typer.Option(
        None,
        help="When the polygons' source imagery was captured: YYYY, YYYY-MM or YYYY-MM-DD",
    ),
    polygon_vintage_col: str | None = typer.Option(
        None, help="Per-polygon capture-date column in the polygon layer"
    ),
    lidar_date: str | None = typer.Option(
        None,
        help="LiDAR capture date (same formats). Omit to measure it from the tiles' GPS time",
    ),
    footprints: str | None = typer.Option(
        None, help="Optional building footprints (GeoParquet/GPKG/SHP)"
    ),
    config: Path | None = typer.Option(
        None, exists=True, dir_okay=False, readable=True,
        help="YAML config; defaults are used when omitted",
    ),
    crs: str | None = typer.Option(
        None, help="Run CRS (the LiDAR's metric CRS). Default: taken from the tile index"
    ),
    polygon_id_col: str | None = typer.Option(
        None, help="Id column in the polygon layer. Default: auto-detect, else synthesize"
    ),
    local: bool = typer.Option(False, help="Force compute.backend=local"),
    max_polygons: int | None = typer.Option(None, help="Limit polygon count (dev/smoke)"),
    bbox: tuple[float, float, float, float] | None = typer.Option(
        None, help="Restrict to bbox in run-CRS units: xmin ymin xmax ymax"
    ),
    dry_run: bool = typer.Option(False, help="Plan + vintage check only; no compute"),
    resume: bool = typer.Option(
        False,
        help="Keep partitions already at the output and run the rest. Refused if they "
             "were made from different inputs, settings or schema.",
    ),
    force_resume: bool = typer.Option(
        False, help="Resume even if the existing partitions were made from different "
                    "inputs, settings or schema"
    ),
    name_template: str = typer.Option(
        "{name}.laz",
        help="LAZ filename template using {name} from the tile index id column",
    ),
    tile_id_col: str | None = typer.Option(
        None, help="Tile-id column in the tile index. Default: auto-detect"
    ),
    no_dask: bool = typer.Option(False, help="Run serially instead of via Dask"),
    report: bool = typer.Option(
        True, help="Build the report (tables, figures, HTML) after a local run"
    ),
) -> None:
    """Measure every polygon against the LiDAR and write the dataset."""
    from pv_geom.pipeline.runner import run_pipeline

    cfg = PVGeomConfig.from_yaml(config) if config else PVGeomConfig()
    if local:
        cfg.compute.backend = "local"
    if polygon_vintage is not None:
        cfg.vintage.polygon_vintage = polygon_vintage
    if polygon_vintage_col is not None:
        cfg.vintage.polygon_vintage_column = polygon_vintage_col
    if lidar_date is not None:
        cfg.vintage.lidar_date = lidar_date
    if crs is not None:
        cfg.crs.target = crs

    # A local run keeps its own log beside the output, as JSON lines.
    log_handler = None
    if not dry_run and not str(output).startswith("s3://"):
        log_handler = add_file_log(Path(output) / "logs" / "run.jsonl")
    manifest = run_pipeline(
        polygons_uri=polygons,
        tile_index_uri=tile_index,
        lidar_prefix=lidar_prefix,
        footprints_uri=footprints,
        output_uri=output,
        cfg=cfg,
        name_template=name_template,
        tile_id_col=tile_id_col,
        polygon_id_col=polygon_id_col,
        max_polygons=max_polygons,
        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]) if bbox else None,
        dry_run=dry_run,
        resume=resume,
        force_resume=force_resume,
        use_dask=not no_dask,
    )
    if log_handler is not None:
        logging.getLogger("pv_geom").removeHandler(log_handler)
        log_handler.close()
    console.print(f"[green]wrote manifest:[/green] {manifest}")

    if report and not dry_run and not str(output).startswith("s3://"):
        from pv_geom.report import build_report

        try:
            result = build_report(output)
        except FileNotFoundError:
            console.print("[yellow]no rows written; skipping report[/yellow]")
        else:
            console.print(f"[green]wrote report:[/green] {result.html}")


@app.command("report")
def report_cmd(
    output_uri: str = typer.Argument(..., help="A pv-geom output directory or s3:// prefix"),
    out: Path | None = typer.Option(
        None, help="Where to write the report. Default: <output>/report (local outputs)"
    ),
    title: str | None = typer.Option(None, help="Report title"),
    area_name: str | None = typer.Option(
        None, help="Name of the study area, used in titles and the methods text"
    ),
    export: bool = typer.Option(
        True, help="Also write the consolidated dataset (GeoParquet + CSV + data dictionary)"
    ),
    polygon_vintage: str | None = typer.Option(
        None, help="Polygon vintage, for outputs written without one (pre-0.2.0)"
    ),
    lidar_date: str | None = typer.Option(
        None, help="LiDAR date, for outputs written without one (pre-0.2.0)"
    ),
) -> None:
    """Build tables, figures, an HTML/Markdown report and the release dataset
    from a finished run."""
    from pv_geom.report import build_report

    result = build_report(output_uri, out_dir=out, title=title, area_name=area_name,
                          export_dataset=export, polygon_vintage=polygon_vintage,
                          lidar_date=lidar_date)
    console.print(f"[green]report:[/green]  {result.html}")
    console.print(f"[green]tables:[/green]  {result.tables_dir}")
    console.print(f"[green]figures:[/green] {result.figures_dir}")
    if result.dataset_dir:
        console.print(f"[green]dataset:[/green] {result.dataset_dir}")


@app.command("inspect-tile")
def inspect_tile(
    tile: str = typer.Argument(..., help="One LAZ tile (path or s3://)"),
    as_json: bool = typer.Option(False, "--json", help="Print JSON instead of a table"),
) -> None:
    """Vet a LiDAR source: classes present, density, CRS, units, flight dates."""
    from pv_geom.io.lidar import inspect_tile as _inspect

    info = _inspect(tile)
    if as_json:
        typer.echo(json.dumps(info, indent=2))      # plain: this is for machines
        return
    t = Table(show_header=False)
    for k, v in info.items():
        t.add_row(k, json.dumps(v) if isinstance(v, (dict, list)) else str(v))
    console.print(t)
    if not info["has_building_class_6"]:
        console.print(
            "[yellow]no class 6 (building): panel candidates will be class-1 returns "
            "above local ground (io.classification fallback)[/yellow]"
        )
    if not info["has_ground_class_2"]:
        console.print("[red]no class 2 (ground): heights above ground cannot be computed[/red]")
    if info["horizontal_units"] and info["horizontal_units"].lower() not in {"metre", "meter"}:
        console.print(f"[red]non-metric CRS units ({info['horizontal_units']}); "
                      f"reproject the tiles before running[/red]")
    if info["flight_end"] is None:
        console.print("[yellow]no usable GPS time: declare --lidar-date; the header date "
                      "is a delivery date[/yellow]")


@app.command("describe-output")
def describe_output(output_uri: str = typer.Argument(...)) -> None:
    """Summarise a finished run: counts, vintage, geometry basis, fit quality."""
    from pv_geom.io.output import read_manifest, read_output_table
    from pv_geom.summary import summarise_table

    manifest = read_manifest(output_uri)
    stats = summarise_table(read_output_table(output_uri))
    console.print(f"[bold]{output_uri}[/bold]  (pv-geom {manifest.get('pkg_version', '?')}, "
                  f"run {str(manifest.get('run_id', '?'))[:8]})")
    v = manifest.get("vintage", {})
    console.print(f"polygon vintage: {v.get('polygon_vintage')}   "
                  f"LiDAR: {v.get('rows_lidar_date_min') or v.get('lidar_flight_start')} .. "
                  f"{v.get('rows_lidar_date_max') or v.get('lidar_flight_end')}   "
                  f"gap: {v.get('vintage_gap_days')} days")
    console.print(f"rows: {stats['rows']:,}   fitted: {stats['fitted']:,} "
                  f"({stats['fit_rate']:.1%})")
    for basis, n in stats.get("geometry_basis_counts", {}).items():
        console.print(f"  {basis:20s} {n:>9,}")
    if "panel_tilt_deg" in stats:
        t = stats["panel_tilt_deg"]
        console.print(f"tilt p10/p50/p90: {t['p10']:.1f} / {t['p50']:.1f} / {t['p90']:.1f} deg")
    if "azimuth_quadrant_counts" in stats:
        console.print(f"azimuth quadrants: {stats['azimuth_quadrant_counts']}")
    if "panel_rmse_p50" in stats:
        console.print(f"panel RMSE p50/p90: {stats['panel_rmse_p50'] * 100:.1f} / "
                      f"{stats['panel_rmse_p90'] * 100:.1f} cm")


def main() -> None:
    """Entry point: run the app, turning anticipated errors into a message and a
    remedy instead of a traceback."""
    try:
        app()
    except PVGeomError as exc:
        console.print(f"[red]error:[/red] {exc.message}")
        if exc.remedy:
            console.print(f"  [bold]fix:[/bold] {exc.remedy}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
