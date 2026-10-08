"""Run-level vintage facts: what the two inputs' dates are, and how they compare.

Rows carry their own dates; this is the whole-run view that goes in the manifest
and is printed before compute starts (and is all ``--dry-run`` reports).
"""

from __future__ import annotations

import logging
from datetime import date

import pyarrow as pa

from pv_geom.config import PVGeomConfig
from pv_geom.errors import LidarClassError
from pv_geom.io.lidar import read_tile_vintage
from pv_geom.vintage import parse_vintage

log = logging.getLogger(__name__)


def probe_lidar_vintage(cfg: PVGeomConfig, tile_uris: list[str]) -> dict:
    """Establish both input dates before compute starts, and compare them.

    Returns the manifest's ``vintage`` block. The polygon vintage is declared;
    the LiDAR date is declared too if given, otherwise measured from a sample of
    tiles (per-point GPS time). Prints a warning when the polygons postdate the
    LiDAR — the case where installations exist in the input but not in the data
    used to measure them. An unreadable tile costs a sample, not the run. Every
    row later carries its own tile's exact date; this is the up-front, whole-run
    view, and what ``--dry-run`` reports.

    The same sample is used to check that the LiDAR carries the point classes
    the run depends on (:func:`check_lidar_classes`), which does raise: without
    a ground class nothing downstream can be trusted.
    """
    polygon_vintage = parse_vintage(cfg.vintage.polygon_vintage)
    declared_lidar = parse_vintage(cfg.vintage.lidar_date)
    out: dict = {
        "polygon_vintage": polygon_vintage.isoformat() if polygon_vintage else None,
        "polygon_vintage_declared_as": (
            str(cfg.vintage.polygon_vintage) if cfg.vintage.polygon_vintage is not None else None
        ),
        "polygon_vintage_column": cfg.vintage.polygon_vintage_column,
        "lidar_date_declared": declared_lidar.isoformat() if declared_lidar else None,
        "lidar_tiles_sampled": 0,
    }

    latest: date | None = declared_lidar
    n = int(cfg.vintage.sample_tiles)
    if n > 0 and tile_uris:
        # Even stride over the sorted URIs so the sample spans the AOI. A
        # collection is flown in strips over weeks, so one corner is not
        # representative.
        ordered = sorted(set(tile_uris))
        step = max(1, len(ordered) // n)
        samples = []
        n_errors = 0
        for uri in ordered[::step][:n]:
            try:
                samples.append(read_tile_vintage(uri))
            except Exception:
                n_errors += 1
        out["lidar_tiles_sampled"] = len(samples)
        out["lidar_tiles_unreadable"] = n_errors
        out.update(check_lidar_classes(cfg, samples))

        starts = [s.flight_start for s in samples if s.flight_start]
        ends = [s.flight_end for s in samples if s.flight_end]
        creations = [s.creation_date for s in samples if s.creation_date]
        out["lidar_flight_start"] = min(starts).isoformat() if starts else None
        out["lidar_flight_end"] = max(ends).isoformat() if ends else None
        out["lidar_header_date_max"] = max(creations).isoformat() if creations else None
        observed = ends or creations
        if observed:
            out["lidar_date_source"] = "gps_time" if ends else "header_date"
            log.info(
                f"LiDAR flown "
                f"{min(starts).isoformat() if starts else '?'} -> {max(observed).isoformat()} "
                f"(source: {out['lidar_date_source']}, {len(samples)} tiles sampled)"
            )
            if declared_lidar is None:
                latest = max(observed)
            elif abs((declared_lidar - max(observed)).days) > 366:
                log.warning(
                    f"declared lidar_date {declared_lidar.isoformat()} "
                    f"differs from the measured {max(observed).isoformat()} by more than "
                    f"a year; rows will use the declared date"
                )
        elif not samples:
            log.warning("no tile could be read; LiDAR date not measured")
    if declared_lidar is not None:
        out["lidar_date_source"] = "declared"

    if polygon_vintage is None and cfg.vintage.polygon_vintage_column is None:
        log.warning(
            "no polygon vintage declared — rows cannot be "
            "credited as present-by-date; geometry_basis rests on the standoff "
            "screen alone. Pass --polygon-vintage."
        )
        return out
    if polygon_vintage is None or latest is None:
        return out

    gap_days = (polygon_vintage - latest).days
    out["vintage_gap_days"] = gap_days
    if gap_days > 0:
        out["vintage_gap_warning"] = f"polygons postdate LiDAR by {gap_days} days"
        log.warning(
            f"*** VINTAGE GAP: the polygons ({polygon_vintage.isoformat()}) "
            f"postdate the LiDAR ({latest.isoformat()}) by {gap_days} days "
            f"({gap_days / 365.25:.1f} years). Installations built in that window are "
            f"in the polygon set but not in the point cloud; their rows measure the "
            f"surface that was there before. See the geometry_basis column. ***"
        )
    else:
        log.info(
            f"polygons ({polygon_vintage.isoformat()}) are no newer "
            f"than the LiDAR ({latest.isoformat()}); installations were present at capture"
        )
    return out


def check_lidar_classes(cfg: PVGeomConfig, samples: list) -> dict:
    """Check the sampled tiles against the configured point classes.

    Says which class will supply panel candidates — the building class when the
    collection has one, otherwise unclassified returns above local ground — and
    refuses to go on when there is no ground class or no candidate class at all.
    Also flags a tile CRS that is not the run CRS, since points are used in
    their native coordinates.
    """
    classes = sorted({c for s in samples for c in s.classes_seen})
    if not classes:
        return {}
    cls = cfg.io.classification
    out: dict = {"lidar_classes_sampled": classes}
    seen = f"classes present in the sampled tiles: {classes}"

    if cls.ground_class not in classes:
        raise LidarClassError(
            f"the LiDAR has no ground returns of class {cls.ground_class} ({seen})",
            "set io.classification.ground_class to the class your collection uses for "
            "ground, or classify the tiles first; heights and the panel-candidate cut "
            "both depend on it",
        )
    if cls.panel_class_primary in classes:
        out["panel_candidate_class"] = cls.panel_class_primary
        log.info("LiDAR classes %s: panel candidates are class %d returns",
                 classes, cls.panel_class_primary)
    elif cls.panel_class_fallback in classes:
        out["panel_candidate_class"] = cls.panel_class_fallback
        log.info("LiDAR classes %s: no class %d, so panel candidates are class %d returns "
                 "more than %.1f m above local ground", classes, cls.panel_class_primary,
                 cls.panel_class_fallback, cls.fallback_height_above_ground_m)
    else:
        raise LidarClassError(
            f"the LiDAR has neither class {cls.panel_class_primary} nor class "
            f"{cls.panel_class_fallback} returns to fit arrays to ({seen})",
            "set io.classification.panel_class_primary / panel_class_fallback to the "
            "classes your collection uses for buildings / unclassified returns",
        )

    target = str(cfg.crs.target)
    others = sorted({s.crs for s in samples if s.crs and s.crs != target})
    if others and target.lower() != "auto":
        from pv_geom.utils.crs import _tile_transform

        out["lidar_converted_on_read"] = {
            c: _tile_transform(c, target)[2] or "same grid; no conversion needed"
            for c in others}
        log.info("tiles declare CRS %s, not the run CRS %s; converted on read: %s",
                 others, target, out["lidar_converted_on_read"])
    return out


def row_vintage_summary(table: pa.Table) -> dict:
    """Vintage facts as actually stamped on the rows (exact, not sampled)."""
    df = table.select(["lidar_date", "lidar_date_source", "vintage_gap_days"]).to_pandas()
    out: dict = {}
    dates = df["lidar_date"].dropna()
    if len(dates):
        out["rows_lidar_date_min"] = min(dates).isoformat()
        out["rows_lidar_date_max"] = max(dates).isoformat()
    out["rows_lidar_date_source_counts"] = {
        str(k): int(v) for k, v in df["lidar_date_source"].fillna("unknown").value_counts().items()
    }
    gaps = df["vintage_gap_days"].dropna()
    if len(gaps):
        out["rows_vintage_gap_days_median"] = float(gaps.median())
        out["rows_polygons_newer_than_lidar"] = int((gaps > 0).sum())
    return out
