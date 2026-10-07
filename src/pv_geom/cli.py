"""pv_geom command line: a thin layer over :mod:`pv_geom.api`."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pv_geom import __version__, api
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
    """Validate a YAML config and show what a run from it would use."""
    cfg = PVGeomConfig.from_yaml(config)
    console.print(f"[green]OK[/green] {config}")
    console.print(f"config_hash: {cfg.hash()}")
    console.print(f"study: {cfg.study.name or '(unnamed)'}")
    for label, value in (
        ("polygons", cfg.resolve(cfg.inputs.polygons)),
        ("lidar tiles", cfg.resolve(cfg.inputs.lidar_prefix)),
        ("tile index", cfg.resolve(cfg.inputs.tile_index) or "built from tile headers"),
        ("footprints", cfg.resolve(cfg.inputs.footprints) or "none"),
        ("output", cfg.resolve(cfg.study.output)),
    ):
        console.print(f"{label}: {value or '[yellow]not set[/yellow]'}")
    console.print(f"backend: {cfg.compute.backend}")
    console.print(f"target CRS: {cfg.crs.target}")
    console.print(f"polygon vintage: {cfg.vintage.polygon_vintage}")
    console.print(f"lidar date: {cfg.vintage.lidar_date or 'measured from GPS time'}")


def _with_run_log(output: str | None, dry_run: bool) -> logging.Handler | None:
    """A JSON-lines log beside a local output, for the duration of a run."""
    if dry_run or not output or str(output).startswith("s3://"):
        return None
    return add_file_log(Path(output) / "logs" / "run.jsonl")


def _close_run_log(handler: logging.Handler | None) -> None:
    if handler is not None:
        logging.getLogger("pv_geom").removeHandler(handler)
        handler.close()


@app.command()
def run(
    config: Path | None = typer.Option(
        None, exists=True, dir_okay=False, readable=True,
        help="YAML config. Its `inputs` and `study` blocks can name everything below; "
             "options given here override it",
    ),
    polygons: str | None = typer.Option(
        None, help="PV polygon layer: GeoParquet / GPKG / GeoJSON / SHP (path or s3://)"
    ),
    lidar_prefix: str | None = typer.Option(
        None, help="Directory or S3 prefix holding the LAZ tiles"
    ),
    tile_index: str | None = typer.Option(
        None, help="LiDAR tile index: GeoParquet/GPKG/SHP/.zip. Default: read from tile headers"
    ),
    output: str | None = typer.Option(None, help="Output directory or s3:// prefix"),
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
    name_template: str | None = typer.Option(
        None, help="LAZ filename template using {name} from the tile index id column "
                   "(default {name}.laz)",
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
    cfg = api.load_config(config)
    destination = cfg.resolve(output or cfg.study.output)
    log_handler = _with_run_log(destination, dry_run)
    try:
        result = api.run(
            cfg,
            polygons=polygons, lidar_prefix=lidar_prefix, tile_index=tile_index,
            output=output, footprints=footprints,
            polygon_vintage=polygon_vintage, polygon_vintage_col=polygon_vintage_col,
            lidar_date=lidar_date, polygon_id_col=polygon_id_col, tile_id_col=tile_id_col,
            name_template=name_template, crs=crs,
            bbox=(bbox[0], bbox[1], bbox[2], bbox[3]) if bbox else None,
            max_polygons=max_polygons, backend="local" if local else None,
            use_dask=not no_dask, dry_run=dry_run, resume=resume, force_resume=force_resume,
        )
    finally:
        _close_run_log(log_handler)
    console.print(f"[green]wrote manifest:[/green] {result.manifest_path}")

    if report and not dry_run and not result.output.startswith("s3://"):
        try:
            built = result.report()
        except FileNotFoundError:
            console.print("[yellow]no rows written; skipping report[/yellow]")
        else:
            console.print(f"[green]wrote report:[/green] {built.html}")


@app.command()
def demo(
    out: Path = typer.Option(Path("pv_geom_demo"), help="Where to write the demo"),
) -> None:
    """Generate a small synthetic study area and run the whole pipeline on it.

    Needs no data and no network: fifteen arrays with known tilt and
    orientation are written as a LAZ tile and a polygon layer, measured, and
    reported. A quick way to see every output, and to check an installation.
    """
    from pv_geom.sample import write_demo

    cfg_path = write_demo(out)
    console.print(f"[green]wrote sample study area:[/green] {out / 'inputs'}")
    console.print(f"[green]wrote config:[/green] {cfg_path}")
    result = api.run(cfg_path, use_dask=False)
    built = result.report()
    console.print(f"[green]dataset:[/green] {result.output}")
    console.print(f"[green]report:[/green]  {built.html}")
    console.print(f"\nRun it again yourself with:  pv-geom run --config {cfg_path} --no-dask")


@app.command("report")
def report_cmd(
    output_uri: str = typer.Argument(..., help="A pv-geom output directory or s3:// prefix"),
    out: Path | None = typer.Option(
        None, help="Where to write the report. Default: <output>/report (local outputs)"
    ),
    title: str | None = typer.Option(None, help="Report title"),
    area_name: str | None = typer.Option(
        None, help="Name of the study area, used in titles and the methods text "
                   "(default: study.name from the run's config)"
    ),
    headline: str | None = typer.Option(
        None, help="Stratum the report leads with: panel, all_fitted, surface_unresolved "
                   "or unscreened (default: panel when there is enough of it)"
    ),
    weight: str = typer.Option(
        "area", help="Weighting of the headline figures: area (array surface) or count"
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
    result = api.report(
        output_uri, out, title=title, area_name=area_name, headline=headline, weight=weight,
        export_dataset=export, polygon_vintage=polygon_vintage, lidar_date=lidar_date,
    )
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
    from pv_geom.io.output import read_manifest

    manifest = read_manifest(output_uri)
    stats = api.describe(output_uri)
    console.print(f"[bold]{output_uri}[/bold]  (pv-geom {manifest.get('pkg_version', '?')}, "
                  f"run {str(manifest.get('run_id', '?'))[:8]})")
    v = manifest.get("vintage", {})
    console.print(f"polygon vintage: {v.get('polygon_vintage')}   "
                  f"LiDAR: {v.get('rows_lidar_date_min') or v.get('lidar_flight_start')} .. "
                  f"{v.get('rows_lidar_date_max') or v.get('lidar_flight_end')}   "
                  f"gap: {v.get('vintage_gap_days')} days")
    console.print(f"rows: {stats['rows']:,}   fitted: {stats['fitted']:,} "
                  f"({stats['fit_rate']:.1%})")
    for status, n in stats.get("status_counts", {}).items():
        if n:
            console.print(f"  status {status:20s} {n:>9,}")
    for basis, n in stats.get("geometry_basis_counts", {}).items():
        console.print(f"  basis  {basis:20s} {n:>9,}")
    if "panel_tilt_deg" in stats:
        t = stats["panel_tilt_deg"]
        console.print(f"tilt p10/p50/p90: {t['p10']:.1f} / {t['p50']:.1f} / {t['p90']:.1f} deg")
    if "azimuth_quadrant_counts" in stats:
        console.print(f"azimuth quadrants: {stats['azimuth_quadrant_counts']}")
    if "panel_rmse_p50" in stats:
        console.print(f"panel RMSE p50/p90: {stats['panel_rmse_p50'] * 100:.1f} / "
                      f"{stats['panel_rmse_p90'] * 100:.1f} cm")


@app.command("compare-reference")
def compare_reference_cmd(
    output_uri: str = typer.Argument(..., help="A finished run's output."),
    reference: Path = typer.Option(..., "--reference", "-r",
                                   help="Table of reference mounts (CSV or Parquet)."),
    out: Path | None = typer.Option(None, "--out", help="Directory for the result tables."),
    tilt_tolerance: float = typer.Option(3.0, help="Tilt agreement tolerance, degrees."),
    azimuth_tolerance: float = typer.Option(10.0, help="Azimuth agreement tolerance, degrees."),
) -> None:
    """Compare measured geometry with externally reported geometry (e.g. PVDAQ)."""
    result = api.compare_reference(output_uri, reference, out,
                                   tilt_tolerance_deg=tilt_tolerance,
                                   azimuth_tolerance_deg=azimuth_tolerance)
    table = Table("reference", "scope", "eligibility", "outcome", "facets", "area agreeing",
                  "tilt err", "azimuth err")

    def _n(v: float, fmt: str) -> str:
        return "-" if v is None or v != v else format(v, fmt)

    for r in result.references.itertuples(index=False):
        table.add_row(str(r.reference_id), r.scope, r.eligibility, r.outcome, str(r.n_facets),
                      _n(r.area_share_agreeing, ".0%"), _n(r.tilt_error_deg, "+.1f"),
                      _n(r.azimuth_error_deg, "+.1f"))
    console.print(table)
    acc = result.summary["mount_scope_accuracy"]
    if acc:
        console.print(f"mount-scope references: {acc['n_references']}  facets: {acc['n_facets']}  "
                      f"tilt MAE {acc['tilt_mae_deg']:.2f} deg (bias {acc['tilt_bias_deg']:+.2f})")
    control = result.summary["negative_control"]
    if control["n_references"]:
        console.print(f"negative control: {control['n_references']} reference(s) not present at "
                      f"the LiDAR date; {_n(control['area_share_panel_confirmed'], '.0%')} of "
                      "their area was labelled panel_confirmed")
    if out is not None:
        console.print(f"tables written to {out}")


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
