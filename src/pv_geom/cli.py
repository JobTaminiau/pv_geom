"""pv_geom Typer CLI. Surface from PRD §9.1."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from pv_geom import __version__
from pv_geom.config import PVGeomConfig
from pv_geom.utils.logging import configure_logging

app = typer.Typer(
    name="pv-geom",
    help="Geometric attribute extraction for solar PV polygons from LiDAR.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def _root(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    configure_logging(verbose=verbose)


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"pv-geom v{__version__}")


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


@app.command()
def run(
    polygons: str = typer.Option(..., help="GeoParquet of PV polygons (path or s3://)"),
    lidar_prefix: str = typer.Option(..., help="S3 prefix containing LAZ tiles"),
    tile_index: str = typer.Option(..., help="GeoParquet/GPKG/SHP tile index"),
    footprints: str = typer.Option(..., help="GeoParquet/GPKG/SHP building footprints"),
    output: str = typer.Option(..., help="Output prefix (s3:// or local path)"),
    config: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    local: bool = typer.Option(False, help="Use LocalCluster instead of Coiled"),
    max_polygons: int | None = typer.Option(None, help="Limit polygon count (dev/smoke)"),
    bbox: tuple[float, float, float, float] | None = typer.Option(
        None, help="Restrict to bbox: xmin ymin xmax ymax"
    ),
    dry_run: bool = typer.Option(False, help="Plan only; write manifest, no compute"),
    resume: bool = typer.Option(False, help="Skip partitions already written"),
) -> None:
    """Run the pv_geom pipeline."""
    cfg = PVGeomConfig.from_yaml(config)
    if local:
        cfg.compute.backend = "local"
    console.print(
        f"[yellow]TODO[/yellow] runner not implemented yet (M6). "
        f"Config OK ({cfg.hash()[:12]}); inputs: {polygons=} {lidar_prefix=} "
        f"{tile_index=} {footprints=} {output=} {max_polygons=} {bbox=} "
        f"{dry_run=} {resume=}"
    )
    raise typer.Exit(code=2)


@app.command("inspect-tile")
def inspect_tile(
    tile_id: str = typer.Argument(...),
    lidar_prefix: str = typer.Option(...),
    tile_index: str = typer.Option(...),
) -> None:
    """Print summary stats for a single LiDAR tile."""
    raise NotImplementedError("M2: implement after lidar IO is in place")


@app.command("describe-output")
def describe_output(output_uri: str = typer.Argument(...)) -> None:
    """Describe a previously written output prefix (manifest + counts)."""
    raise NotImplementedError("M8: implement once output writer is in place")


if __name__ == "__main__":
    app()
