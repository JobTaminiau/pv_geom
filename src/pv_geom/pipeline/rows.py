"""Measurements -> output rows. The one place the schema's types are applied."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pyarrow as pa
from shapely import wkb

from pv_geom import __version__
from pv_geom.pipeline.measure import Measurement, Segment
from pv_geom.schema import is_recommended
from pv_geom.utils.north import to_true_azimuth
from pv_geom.vintage import NOT_MEASURED


@dataclass(frozen=True)
class Provenance:
    """What every row of one partition records about the run that made it."""

    config_hash: str
    run_id: str
    partition_id: int
    lidar_tile_ids: tuple[str, ...] = ()


def _f32(v: Any) -> Any:
    """``np.float32``, with None / NaN passed through as None (an Arrow null)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return np.float32(v)


def to_row(m: Measurement, prov: Provenance) -> dict[str, Any]:
    """One measurement as a dict keyed on the core output schema's names."""
    fit = m.panel.fit
    roof_fit = m.roof.fit
    conv = m.task.grid_convergence_deg

    def _true(grid_azimuth: float) -> Any:
        """A fitted plane's azimuth, from grid north to true north."""
        return _f32(to_true_azimuth(grid_azimuth, conv))

    return {
        "polygon_id": str(m.task.polygon_id),
        "parent_polygon_id": str(m.task.parent_id),
        "input_row": np.int32(m.task.input_row),
        "status": m.status,
        "geometry": wkb.dumps(m.task.polygon),
        "area_m2": np.float32(m.area_m2),
        "surface_area_m2": _f32(m.surface_area_m2),
        "aspect_ratio": _f32(m.aspect_ratio),
        "polygon_vintage": m.task.polygon_vintage,
        "lidar_date": m.lidar_date,
        "lidar_date_source": m.lidar_date_source,
        "vintage_gap_days": None if m.gap_days is None else np.int32(m.gap_days),
        "recommended": is_recommended(m.status, m.basis, m.flags),
        "geometry_basis": m.basis,
        "n_points_panel": int(fit.n_total),
        "n_inliers_panel": int(fit.n_inliers),
        "fit_failure": m.fit_failure,
        "point_density": np.float32(m.point_density),
        "panel_tilt_deg": _f32(fit.tilt_deg),
        "panel_azimuth_deg": _true(fit.azimuth_deg),
        "grid_convergence_deg": np.float32(conv),
        "panel_rmse_m": _f32(fit.rmse),
        "panel_fit_tolerance_m": np.float32(m.panel.tolerance_m) if m.fit_ok else None,
        "panel_tilt_unc_deg": _f32(m.tilt_unc_deg),
        "panel_azimuth_unc_deg": _f32(m.azimuth_unc_deg),
        "n_planes_detected": np.int8(len(m.segments)),
        "secondary_tilt_deg": _f32(m.secondary.tilt_deg) if m.secondary is not None else None,
        "secondary_azimuth_deg": (
            _true(m.secondary.azimuth_deg) if m.secondary is not None else None
        ),
        "segments": [_segment(m, s, conv) for s in m.segments] if m.fit_ok else None,
        "roof_ref_source": m.roof.source,
        "roof_ref_method": m.roof.method,
        "roof_tilt_deg": _f32(roof_fit.tilt_deg) if roof_fit is not None else None,
        "roof_azimuth_deg": _true(roof_fit.azimuth_deg) if roof_fit is not None else None,
        "roof_rmse_m": _f32(roof_fit.rmse) if roof_fit is not None else None,
        "panel_roof_angle_deg": _f32(m.panel_roof_angle_deg),
        "height_above_roof_m": _f32(m.height_above_roof_m),
        "height_above_ground_m": _f32(m.height_above_ground_m),
        "on_building": None if m.roof.on_building is None else bool(m.roof.on_building),
        "building_id": m.roof.building_id,
        "flags": m.flags,
        "lidar_tile_ids": list(prov.lidar_tile_ids),
        "pkg_version": __version__,
        "config_hash": prov.config_hash,
        "run_id": prov.run_id,
        "partition_id": np.int32(prov.partition_id),
    }


def _segment(m: Measurement, s: Segment, conv: float) -> dict[str, Any]:
    """One facet as an entry of the `segments` column."""
    area = m.area_m2 * s.share
    tilt = s.fit.tilt_deg
    return {
        "segment_index": np.int8(s.index),
        "area_share": _f32(s.share),
        "area_m2": _f32(area),
        "surface_area_m2": _f32(area / float(np.cos(np.radians(tilt))) if tilt < 89.0 else None),
        "tilt_deg": _f32(tilt),
        "azimuth_deg": _f32(to_true_azimuth(s.fit.azimuth_deg, conv)),
        "rmse_m": _f32(s.fit.rmse),
        "tilt_unc_deg": _f32(s.tilt_unc_deg),
        "azimuth_unc_deg": _f32(s.azimuth_unc_deg),
        "n_points": int(s.fit.n_inliers),
        "height_above_ground_m": _f32(s.height_above_ground_m),
        "geometry": wkb.dumps(s.footprint),
    }


def unmeasured_row(
    *,
    polygon_id: str,
    parent_polygon_id: str,
    input_row: int,
    polygon: Any,
    status: str,
    polygon_vintage: date | None,
    flags: tuple[str, ...],
    prov: Provenance,
) -> dict[str, Any]:
    """The row for a polygon that never reached measurement. It carries what is
    known from the input — identity, geometry, area, vintage — and its status;
    every measured column is left null."""
    has_geom = polygon is not None and not polygon.is_empty
    polygonal = has_geom and polygon.geom_type in ("Polygon", "MultiPolygon")
    return {
        "polygon_id": str(polygon_id),
        "parent_polygon_id": str(parent_polygon_id),
        "input_row": np.int32(input_row),
        "status": status,
        "geometry": wkb.dumps(polygon) if has_geom else None,
        "area_m2": np.float32(polygon.area) if polygonal else None,
        "polygon_vintage": polygon_vintage,
        "recommended": False,
        "geometry_basis": NOT_MEASURED,
        "roof_ref_source": "none",
        "flags": list(flags),
        "lidar_tile_ids": list(prov.lidar_tile_ids),
        "pkg_version": __version__,
        "config_hash": prov.config_hash,
        "run_id": prov.run_id,
        "partition_id": np.int32(prov.partition_id),
    }


def rows_to_table(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    """Rows -> an Arrow table in ``schema`` (an empty list gives an empty table)."""
    return pa.table({f.name: [r.get(f.name) for r in rows] for f in schema}, schema=schema)
