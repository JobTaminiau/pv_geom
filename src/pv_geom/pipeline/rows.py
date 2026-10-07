"""Measurements -> output rows. The one place the schema's types are applied."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
from shapely import wkb

from pv_geom import __version__
from pv_geom.pipeline.measure import Measurement


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
    return {
        "polygon_id": str(m.task.polygon_id),
        "parent_polygon_id": str(m.task.parent_id),
        "input_row": np.int32(m.task.input_row),
        "geometry": wkb.dumps(m.task.polygon),
        "area_m2": np.float32(m.area_m2),
        "surface_area_m2": _f32(m.surface_area_m2),
        "aspect_ratio": _f32(m.aspect_ratio),
        "polygon_vintage": m.task.polygon_vintage,
        "lidar_date": m.lidar_date,
        "lidar_date_source": m.lidar_date_source,
        "vintage_gap_days": None if m.gap_days is None else np.int32(m.gap_days),
        "geometry_basis": m.basis,
        "n_points_panel": int(fit.n_total),
        "n_inliers_panel": int(fit.n_inliers),
        "point_density": np.float32(m.point_density),
        "panel_tilt_deg": _f32(fit.tilt_deg),
        "panel_azimuth_deg": _f32(fit.azimuth_deg),
        "panel_rmse_m": _f32(fit.rmse),
        "panel_fit_tolerance_m": np.float32(m.panel.tolerance_m) if m.fit_ok else None,
        "panel_tilt_unc_deg": _f32(m.tilt_unc_deg),
        "panel_azimuth_unc_deg": _f32(m.azimuth_unc_deg),
        "n_planes_detected": np.int8(2 if m.secondary is not None else int(m.fit_ok)),
        "secondary_tilt_deg": _f32(m.secondary.tilt_deg) if m.secondary is not None else None,
        "secondary_azimuth_deg": (
            _f32(m.secondary.azimuth_deg) if m.secondary is not None else None
        ),
        "roof_ref_source": m.roof.source,
        "roof_tilt_deg": _f32(roof_fit.tilt_deg) if roof_fit is not None else None,
        "roof_azimuth_deg": _f32(roof_fit.azimuth_deg) if roof_fit is not None else None,
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


def rows_to_table(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    """Rows -> an Arrow table in ``schema`` (an empty list gives an empty table)."""
    return pa.table({f.name: [r.get(f.name) for r in rows] for f in schema}, schema=schema)
