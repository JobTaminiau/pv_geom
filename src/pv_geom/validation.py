"""Compare measured geometry with externally reported geometry.

An external reference (PVDAQ, a permit set, an installer's as-built, a survey)
says: *this system has a mount at tilt T facing azimuth A*. This module lines
such statements up with what pv-geom measured, without overstating what the
comparison shows. Four things make that harder than a subtraction:

**A reference describes a mount, a measurement describes a facet.** A system can
have several mounts and the record may list one. Every facet (see the
``facets`` column) of every polygon matched to the system is therefore
compared, and the result is the *share of measured area* the reference
describes — not the error of whichever facet happens to agree best. Choosing
the facet by agreement would manufacture accuracy.

**References have a resolution.** PVOutput-derived PVDAQ records give azimuth
as one of eight compass points; a measured 173 degrees agrees with a reported
"180" as well as it possibly can. Each reference carries its resolution, and a
difference within half of it is not counted as error.

**References can be placeholders.** A reported tilt of 1 degree on a pitched
residential roof is a default, not a measurement. Rows marked ``suspect`` are
reported but never scored.

**The array must have existed when the LiDAR was flown.** A reference is only
scored when there is evidence the array was present by the LiDAR date. Where
there is evidence it was *not* (the installation appears in later imagery), the
polygons are a negative control instead: pv-geom should not have called them
``panel_confirmed``.

The polygons belonging to a system must be established from location, imagery
and module counts — never from which polygons have agreeable angles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pv_geom.errors import InputError

# Reference table columns. Only the first four are required.
REFERENCE_COLUMNS: dict[str, str] = {
    "reference_id": "Unique id of this reference mount.",
    "polygon_ids": "Ids of the polygons that are this system, separated by ';'.",
    "tilt_deg": "Reported tilt from horizontal.",
    "azimuth_deg": "Reported azimuth, clockwise from true north (blank if not reported).",
    "system_id": "Id of the system in the source (several mounts can share one).",
    "source": "Where the reference comes from, e.g. 'PVDAQ 10733'.",
    "scope": "'mount' if the record describes one identified subarray; 'system' if it is "
             "a single record standing for a whole system (default).",
    "grade": "'surveyed', 'documented' or 'self_reported' (default).",
    "tilt_resolution_deg": "Step in which tilt is reported (default 1).",
    "azimuth_resolution_deg": "Step in which azimuth is reported (default 1; 45 for "
                              "compass-point sources such as PVOutput).",
    "suspect": "True if the reported values look like placeholders; never scored.",
    "present_by": "Date by which the array is evidenced to exist in its current form.",
    "absent_at": "Latest date at which the array is evidenced NOT to exist.",
    "notes": "Free text.",
}
_REQUIRED = ("reference_id", "polygon_ids", "tilt_deg", "azimuth_deg")
_DEFAULTS: dict[str, Any] = {
    "system_id": None, "source": None, "scope": "system", "grade": "self_reported",
    "tilt_resolution_deg": 1.0, "azimuth_resolution_deg": 1.0, "suspect": False,
    "present_by": None, "absent_at": None, "notes": None,
}

# Eligibility of a reference for scoring.
ELIGIBLE = "eligible"
NOT_PRESENT = "not_present_at_lidar"      # negative control
PRESENCE_UNKNOWN = "presence_unknown"
SUSPECT = "reference_suspect"
NO_MEASUREMENT = "no_measurement"

ELIGIBILITY_DESCRIPTIONS = {
    ELIGIBLE: "Evidenced present by the LiDAR date; scored.",
    NOT_PRESENT: "Evidenced absent at the LiDAR date; a negative control for the vintage "
                 "screen, not scored for accuracy.",
    PRESENCE_UNKNOWN: "No evidence either way about presence at the LiDAR date; compared "
                      "but kept out of accuracy statistics.",
    SUSPECT: "Reported values look like placeholders; not scored.",
    NO_MEASUREMENT: "None of the matched polygons has a fitted plane.",
}


class ReferenceTableError(InputError):
    """The reference table cannot be used as given."""


@dataclass(frozen=True)
class ValidationResult:
    """Facet-level comparisons, one row per reference, and headline numbers."""

    facets: pd.DataFrame
    references: pd.DataFrame
    summary: dict[str, Any]

    def write(self, out_dir: str | Path) -> Path:
        import json

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.facets.to_csv(out / "validation_facets.csv", index=False)
        self.references.to_csv(out / "validation_references.csv", index=False)
        (out / "validation_summary.json").write_text(
            json.dumps(self.summary, indent=2, default=str), encoding="utf-8", newline="\n")
        return out


def read_references(path: str | Path) -> pd.DataFrame:
    """Read a reference table (CSV or Parquet) and fill the optional columns."""
    p = Path(path)
    if not p.exists():
        raise ReferenceTableError(f"Reference table not found: {p}",
                             remedy="Check the path given to --reference.")
    table = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, dtype=str)
    return normalise_references(table)


def _as_bool(v: Any) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes", "y", "t")


def normalise_references(table: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in _REQUIRED if c not in table.columns]
    if missing:
        raise ReferenceTableError(
            f"Reference table is missing column(s): {', '.join(missing)}",
            remedy="Required columns: " + ", ".join(_REQUIRED)
                   + ". See pv_geom.validation.REFERENCE_COLUMNS.")
    refs = table.copy()
    for col, default in _DEFAULTS.items():
        if col not in refs.columns:
            refs[col] = default
        elif default is not None:
            refs[col] = refs[col].where(refs[col].notna() & (refs[col].astype(str) != ""),
                                        default)
    for col in ("tilt_deg", "azimuth_deg", "tilt_resolution_deg", "azimuth_resolution_deg"):
        refs[col] = pd.to_numeric(refs[col], errors="coerce")
    refs["suspect"] = refs["suspect"].map(_as_bool)
    for col in ("present_by", "absent_at"):
        refs[col] = pd.to_datetime(refs[col], errors="coerce").dt.date
    refs["reference_id"] = refs["reference_id"].astype(str)
    if refs["reference_id"].duplicated().any():
        dup = refs.loc[refs["reference_id"].duplicated(), "reference_id"].iloc[0]
        raise ReferenceTableError(f"reference_id is not unique: {dup!r}",
                             remedy="Give each reference mount its own id.")
    bad = sorted(set(refs["scope"]) - {"mount", "system"})
    if bad:
        raise ReferenceTableError(f"Unknown scope value(s): {bad}",
                             remedy="scope must be 'mount' or 'system'.")
    return refs[list(REFERENCE_COLUMNS)]


def circular_difference_deg(a: float, b: float) -> float:
    """Signed difference a - b between two azimuths, in (-180, 180]."""
    return float((a - b + 180.0) % 360.0 - 180.0)


def _facets_of(row: Any) -> list[dict[str, Any]]:
    """The facets of one measured polygon; the primary plane alone for output
    written before the ``segments`` column existed."""
    segs = getattr(row, "facets", None)
    if segs is not None and len(segs):
        return [dict(s) for s in segs]
    if pd.isna(row.tilt_deg):
        return []
    return [{"facet_index": 0, "area_m2": row.area_m2, "tilt_deg": row.tilt_deg,
             "azimuth_deg": row.azimuth_deg}]


def _beyond(error: float, resolution: float) -> float:
    """How much of an error is not explained by the reference's resolution."""
    return max(0.0, abs(error) - resolution / 2.0)


def compare_to_reference(
    measured: pd.DataFrame,
    references: pd.DataFrame,
    *,
    tilt_tolerance_deg: float = 3.0,
    azimuth_tolerance_deg: float = 10.0,
    min_tilt_for_azimuth_deg: float = 5.0,
) -> ValidationResult:
    """Line reported geometry up with measured geometry.

    ``measured`` is pv-geom output (``pv_geom.load``). A facet *agrees* with a
    reference when its tilt is within ``tilt_tolerance_deg`` and its azimuth
    within ``azimuth_tolerance_deg``, each widened by half the reference's
    resolution. Azimuth is not compared when either tilt is below
    ``min_tilt_for_azimuth_deg``: a near-flat plane has no meaningful facing.
    """
    refs = normalise_references(references)
    by_id = {str(r.polygon_id): r for r in measured.itertuples(index=False)}

    facet_rows: list[dict[str, Any]] = []
    ref_rows: list[dict[str, Any]] = []
    for ref in refs.itertuples(index=False):
        ids = [i.strip() for i in str(ref.polygon_ids).split(";") if i.strip()]
        rows = [by_id[i] for i in ids if i in by_id]
        lidar_dates = [pd.Timestamp(r.lidar_date).date() for r in rows
                       if getattr(r, "lidar_date", None) is not None
                       and not pd.isna(r.lidar_date)]
        lidar_date = max(lidar_dates) if lidar_dates else None

        own: list[dict[str, Any]] = []
        for r in rows:
            for seg in _facets_of(r):
                tilt = float(seg["tilt_deg"])
                az = seg.get("azimuth_deg")
                az = float(az) if az is not None and not pd.isna(az) else np.nan
                tilt_err = tilt - ref.tilt_deg
                az_comparable = (not np.isnan(az) and not pd.isna(ref.azimuth_deg)
                                 and min(tilt, ref.tilt_deg) >= min_tilt_for_azimuth_deg)
                az_err = circular_difference_deg(az, ref.azimuth_deg) if az_comparable else np.nan
                tilt_ok = _beyond(tilt_err, ref.tilt_resolution_deg) <= tilt_tolerance_deg
                az_ok = (_beyond(az_err, ref.azimuth_resolution_deg) <= azimuth_tolerance_deg
                         if az_comparable else None)
                own.append({
                    "reference_id": ref.reference_id, "polygon_id": str(r.polygon_id),
                    "facet_index": int(seg["facet_index"]),
                    "area_m2": float(seg["area_m2"]),
                    "geometry_basis": getattr(r, "geometry_basis", None),
                    "lidar_date": getattr(r, "lidar_date", None),
                    "tilt_deg": tilt, "azimuth_deg": az,
                    "reference_tilt_deg": ref.tilt_deg,
                    "reference_azimuth_deg": ref.azimuth_deg,
                    "tilt_error_deg": tilt_err, "azimuth_error_deg": az_err,
                    "tilt_error_beyond_resolution_deg": _beyond(tilt_err,
                                                                ref.tilt_resolution_deg),
                    "azimuth_error_beyond_resolution_deg": (
                        _beyond(az_err, ref.azimuth_resolution_deg) if az_comparable
                        else np.nan),
                    "tilt_agrees": tilt_ok, "azimuth_agrees": az_ok,
                    "agrees": bool(tilt_ok and az_ok is not False),
                })
        facet_rows.extend(own)

        if ref.suspect:
            eligibility = SUSPECT
        elif not own:
            eligibility = NO_MEASUREMENT
        elif ref.absent_at is not None and not pd.isna(ref.absent_at) \
                and lidar_date is not None and lidar_date <= ref.absent_at:
            eligibility = NOT_PRESENT
        elif ref.present_by is not None and not pd.isna(ref.present_by) \
                and lidar_date is not None and ref.present_by <= lidar_date:
            eligibility = ELIGIBLE
        else:
            eligibility = PRESENCE_UNKNOWN

        area = sum(f["area_m2"] for f in own)

        def _share(key: str, own: list[dict[str, Any]] = own, area: float = area) -> float:
            if not area:
                return np.nan
            return sum(f["area_m2"] for f in own if f[key]) / area

        def _mean(key: str, own: list[dict[str, Any]] = own) -> float:
            pairs = [(f[key], f["area_m2"]) for f in own if not np.isnan(f[key])]
            if not pairs:
                return np.nan
            v, w = np.array(pairs).T
            return float(np.sum(v * w) / np.sum(w))

        share = _share("agrees")
        comparable = eligibility in (ELIGIBLE, PRESENCE_UNKNOWN)
        if not comparable:
            outcome = "not_scored"
        elif share >= 0.9:
            outcome = "consistent"
        elif share > 0:
            outcome = "partly_consistent"
        else:
            outcome = "inconsistent"
        confirmed = sum(f["area_m2"] for f in own if f["geometry_basis"] == "panel_confirmed")
        ref_rows.append({
            "reference_id": ref.reference_id, "system_id": ref.system_id, "source": ref.source,
            "scope": ref.scope, "grade": ref.grade,
            "reference_tilt_deg": ref.tilt_deg, "reference_azimuth_deg": ref.azimuth_deg,
            "tilt_resolution_deg": ref.tilt_resolution_deg,
            "azimuth_resolution_deg": ref.azimuth_resolution_deg,
            "n_polygons_listed": len(ids), "n_polygons_found": len(rows),
            "n_facets": len(own), "measured_area_m2": area, "lidar_date": lidar_date,
            "eligibility": eligibility, "outcome": outcome,
            "area_share_agreeing": share,
            "area_share_tilt_agreeing": _share("tilt_agrees"),
            "area_share_panel_confirmed": confirmed / area if area else np.nan,
            "tilt_error_deg": _mean("tilt_error_deg"),
            "azimuth_error_deg": _mean("azimuth_error_deg"),
            "notes": ref.notes,
        })

    facets = pd.DataFrame(facet_rows)
    per_ref = pd.DataFrame(ref_rows)
    return ValidationResult(facets, per_ref, _summarise(
        facets, per_ref, tilt_tolerance_deg, azimuth_tolerance_deg, min_tilt_for_azimuth_deg))


def _summarise(facets: pd.DataFrame, refs: pd.DataFrame, tilt_tol: float, az_tol: float,
               min_tilt: float) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "settings": {"tilt_tolerance_deg": tilt_tol, "azimuth_tolerance_deg": az_tol,
                     "min_tilt_for_azimuth_deg": min_tilt},
        "n_references": len(refs),
        "eligibility": refs["eligibility"].value_counts().to_dict() if len(refs) else {},
        "outcome": refs["outcome"].value_counts().to_dict() if len(refs) else {},
    }
    eligible = refs[refs["eligibility"] == ELIGIBLE] if len(refs) else refs
    summary["eligible"] = {
        "n": len(eligible),
        "outcome": eligible["outcome"].value_counts().to_dict() if len(eligible) else {},
        "median_area_share_agreeing": (float(eligible["area_share_agreeing"].median())
                                       if len(eligible) else None),
    }
    # Error statistics only where a reference describes one identified subarray:
    # for a whole-system record the facet-to-mount pairing is unknown, and any
    # error figure would depend on how that was guessed.
    mounts = set(eligible.loc[eligible["scope"] == "mount", "reference_id"]) if len(eligible) \
        else set()
    scored = facets[facets["reference_id"].isin(mounts)] if len(facets) else facets
    if len(scored):
        tilt = scored["tilt_error_deg"].dropna()
        az = scored["azimuth_error_deg"].dropna()
        summary["mount_scope_accuracy"] = {
            "n_references": len(mounts), "n_facets": len(scored),
            "tilt_bias_deg": float(tilt.mean()), "tilt_mae_deg": float(tilt.abs().mean()),
            "n_azimuth": len(az),
            "azimuth_bias_deg": float(az.mean()) if len(az) else None,
            "azimuth_mae_deg": float(az.abs().mean()) if len(az) else None,
        }
    else:
        summary["mount_scope_accuracy"] = None
    control = refs[refs["eligibility"] == NOT_PRESENT] if len(refs) else refs
    summary["negative_control"] = {
        "n_references": len(control),
        "measured_area_m2": float(control["measured_area_m2"].sum()) if len(control) else 0.0,
        "area_share_panel_confirmed": (
            float((control["area_share_panel_confirmed"] * control["measured_area_m2"]).sum()
                  / control["measured_area_m2"].sum())
            if len(control) and control["measured_area_m2"].sum() else None),
    }
    return summary
