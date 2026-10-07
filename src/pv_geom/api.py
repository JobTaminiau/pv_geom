"""The Python API: run a study area, load its output, build its report.

    import pv_geom

    result = pv_geom.run("configs/phoenix.yaml")           # everything from the config
    result = pv_geom.run(polygons="arrays.gpkg",            # or spelled out
                         lidar_prefix="tiles/", tile_index="tiles/index.gpkg",
                         output="out/", polygon_vintage="2024-10")
    gdf = result.load()                                     # GeoDataFrame, one row per polygon
    report = result.report()                                # tables, figures, HTML

The command line is a thin layer over these three functions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pv_geom.config import PVGeomConfig
from pv_geom.errors import InputError

if TYPE_CHECKING:
    import geopandas as gpd

    from pv_geom.report import ReportResult

_REQUIRED = {
    "polygons": "--polygons / inputs.polygons",
    "lidar_prefix": "--lidar-prefix / inputs.lidar_prefix",
    "output": "--output / study.output",
}


@dataclass
class RunResult:
    """A finished (or dry) run."""

    output: str
    manifest_path: str

    @property
    def manifest(self) -> dict[str, Any]:
        from pv_geom.io.output import read_manifest

        return read_manifest(self.output)

    def load(self) -> gpd.GeoDataFrame:
        """The run's rows as a GeoDataFrame."""
        return load(self.output)

    def report(self, out_dir: str | Path | None = None, **kwargs: Any) -> ReportResult:
        """Build the report for this run (see :func:`report`)."""
        return report(self.output, out_dir, **kwargs)

    def __repr__(self) -> str:
        return f"RunResult(output={self.output!r})"


def load_config(config: str | Path | PVGeomConfig | None) -> PVGeomConfig:
    """A config object from a YAML path, an existing object (copied), or defaults."""
    if config is None:
        return PVGeomConfig()
    if isinstance(config, PVGeomConfig):
        return config.model_copy(deep=True)
    return PVGeomConfig.from_yaml(config)


def run(
    config: str | Path | PVGeomConfig | None = None,
    *,
    polygons: str | Path | None = None,
    lidar_prefix: str | Path | None = None,
    tile_index: str | Path | None = None,
    output: str | Path | None = None,
    footprints: str | Path | None = None,
    polygon_vintage: str | int | None = None,
    polygon_vintage_col: str | None = None,
    lidar_date: str | int | None = None,
    polygon_id_col: str | None = None,
    tile_id_col: str | None = None,
    name_template: str | None = None,
    crs: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    max_polygons: int | None = None,
    backend: str | None = None,
    use_dask: bool = True,
    dry_run: bool = False,
    resume: bool = False,
    force_resume: bool = False,
) -> RunResult:
    """Measure every polygon against the LiDAR and write the dataset.

    Everything can come from ``config`` (a YAML path or a config object) — its
    ``inputs`` block names the layers and ``study.output`` the destination — and
    any keyword given here overrides the config. The tile index is optional:
    without one it is built from the tiles' headers.
    """
    from pv_geom.pipeline.runner import run_pipeline

    cfg = load_config(config)
    inp = cfg.inputs
    for field, value in (("polygons", polygons), ("lidar_prefix", lidar_prefix),
                         ("tile_index", tile_index), ("footprints", footprints),
                         ("polygon_id_col", polygon_id_col), ("tile_id_col", tile_id_col),
                         ("name_template", name_template)):
        if value is not None:
            setattr(inp, field, str(value))
    if bbox is not None:
        inp.bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
    if max_polygons is not None:
        inp.max_polygons = max_polygons
    if output is not None:
        cfg.study.output = str(output)
    if polygon_vintage is not None:
        cfg.vintage.polygon_vintage = polygon_vintage
    if polygon_vintage_col is not None:
        cfg.vintage.polygon_vintage_column = polygon_vintage_col
    if lidar_date is not None:
        cfg.vintage.lidar_date = lidar_date
    if crs is not None:
        cfg.crs.target = crs
    if backend is not None:
        cfg.compute.backend = backend                      # type: ignore[assignment]

    have = {"polygons": inp.polygons, "lidar_prefix": inp.lidar_prefix,
            "output": cfg.study.output}
    missing = [where for key, where in _REQUIRED.items() if not have[key]]
    if missing:
        raise InputError(
            "the run is missing " + ", ".join(m.split(" / ")[0] for m in missing),
            "give " + "; ".join(missing) + " (command-line option / config key)",
        )

    out = cfg.resolve(cfg.study.output)
    polygons_uri = cfg.resolve(inp.polygons)
    lidar_uri = cfg.resolve(inp.lidar_prefix)
    assert out is not None and polygons_uri is not None and lidar_uri is not None
    manifest = run_pipeline(
        polygons_uri=polygons_uri,
        tile_index_uri=cfg.resolve(inp.tile_index),
        lidar_prefix=lidar_uri,
        footprints_uri=cfg.resolve(inp.footprints),
        output_uri=out,
        cfg=cfg,
        name_template=inp.name_template,
        tile_id_col=inp.tile_id_col,
        polygon_id_col=inp.polygon_id_col,
        max_polygons=inp.max_polygons,
        bbox=inp.bbox,
        dry_run=dry_run,
        resume=resume,
        force_resume=force_resume,
        use_dask=use_dask,
    )
    return RunResult(output=str(out), manifest_path=str(manifest))


def load(output: str | Path) -> gpd.GeoDataFrame:
    """A run's rows as a GeoDataFrame in the run CRS — one row per input
    polygon, with outputs from older schema versions upgraded on the way."""
    from pv_geom.io.output import read_output

    return read_output(output)


def report(output: str | Path, out_dir: str | Path | None = None, **kwargs: Any) -> ReportResult:
    """Build tables, figures, the HTML/Markdown report and the release dataset
    for a finished run. Keyword arguments are those of
    :func:`pv_geom.report.build_report` (``area_name``, ``title``, ``headline``,
    ``weight``, ``export_dataset``, ...)."""
    from pv_geom.report import build_report

    return build_report(output, out_dir, **kwargs)


def describe(output: str | Path) -> dict[str, Any]:
    """At-a-glance statistics of a finished run (what ``describe-output`` prints)."""
    from pv_geom.io.output import read_output_table
    from pv_geom.summary import summarise_table

    return json.loads(json.dumps(summarise_table(read_output_table(output)), default=str))
