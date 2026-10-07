"""Write the consolidated release dataset and its dictionary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from pv_geom.schema import (
    FLAG_DESCRIPTIONS,
    STATUS_DESCRIPTIONS,
    data_dictionary,
    output_schema,
)
from pv_geom.vintage import BASIS_DESCRIPTIONS


def export_dataset(gdf, df: pd.DataFrame, manifest: dict, dataset_dir: Path) -> pd.DataFrame:
    """Write the consolidated release dataset and its dictionary."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    out = gdf.copy()
    for col in ("geometry_basis", "surface_area_m2", "point_density"):
        if col not in out.columns:
            out[col] = df[col].to_numpy()
    for col in ("centroid_lon", "centroid_lat"):
        if col in df.columns:
            out[col] = df[col].to_numpy()
    out.to_parquet(dataset_dir / "pv_geom.parquet", index=False)

    flat = pd.DataFrame(out.drop(columns="geometry"))
    for col in ("flags", "lidar_tile_ids"):
        if col in flat.columns:
            flat[col] = flat[col].map(lambda v: ";".join(v) if v is not None else "")
    flat.to_csv(dataset_dir / "pv_geom.csv", index=False)

    schema = output_schema("mounting_type" in out.columns)
    known = {f.name for f in schema}
    dictionary = pd.DataFrame(data_dictionary(schema))
    dictionary = dictionary[dictionary["column"].isin(out.columns)]
    extra = [
        {"column": "centroid_lon", "type": "double", "unit": "deg", "nullable": "no",
         "description": "Polygon centroid longitude (WGS 84)."},
        {"column": "centroid_lat", "type": "double", "unit": "deg", "nullable": "no",
         "description": "Polygon centroid latitude (WGS 84)."},
    ]
    dictionary = pd.concat(
        [dictionary, pd.DataFrame([e for e in extra if e["column"] in out.columns])],
        ignore_index=True)
    legacy = [c for c in out.columns if c not in known and c not in {"centroid_lon", "centroid_lat"}]
    if legacy:
        dictionary = pd.concat([dictionary, pd.DataFrame(
            [{"column": c, "type": str(out[c].dtype), "unit": "", "nullable": "",
              "description": "Column from an earlier pv-geom version."} for c in legacy])],
            ignore_index=True)
    dictionary.to_csv(dataset_dir / "data_dictionary.csv", index=False)

    flag_rows = pd.DataFrame(
        [{"flag": k, "description": v} for k, v in FLAG_DESCRIPTIONS.items()])
    basis_rows = pd.DataFrame(
        [{"geometry_basis": k, "description": v} for k, v in BASIS_DESCRIPTIONS.items()])
    pd.DataFrame(
        [{"status": k, "description": v} for k, v in STATUS_DESCRIPTIONS.items()]
    ).to_csv(dataset_dir / "status_definitions.csv", index=False)
    flag_rows.to_csv(dataset_dir / "flag_definitions.csv", index=False)
    basis_rows.to_csv(dataset_dir / "geometry_basis_definitions.csv", index=False)
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    metadata = dataset_metadata(out, df, manifest)
    (dataset_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    (dataset_dir / "README.md").write_text(dataset_readme(metadata, dictionary),
                                           encoding="utf-8")
    write_checksums(dataset_dir)
    return dictionary


CHECKSUM_FILE = "SHA256SUMS.txt"


def write_checksums(dataset_dir: Path) -> Path:
    """``SHA256SUMS.txt`` covering every other file in the folder, in the format
    ``sha256sum -c`` reads."""
    lines = []
    for path in sorted(dataset_dir.iterdir()):
        if path.is_file() and path.name != CHECKSUM_FILE:
            digest = hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    digest.update(block)
            lines.append(f"{digest.hexdigest()}  {path.name}")
    target = dataset_dir / CHECKSUM_FILE
    # Always LF: `sha256sum -c` takes a trailing carriage return as part of the name.
    with target.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return target


def dataset_metadata(out, df: pd.DataFrame, manifest: dict) -> dict:
    """Machine-readable description of the release dataset, shaped after the
    DataCite / Zenodo fields a repository deposit asks for. Fields only a person
    can supply (creators, licence, funding) are left empty, not guessed."""
    cfg = manifest.get("config", {})
    study = (cfg.get("study") or {}).get("name")
    vintage = manifest.get("vintage", {})
    status = df["status"].value_counts().to_dict() if "status" in df.columns else {}
    bbox = None
    if {"centroid_lon", "centroid_lat"} <= set(df.columns) and df["centroid_lon"].notna().any():
        bbox = [float(df["centroid_lon"].min()), float(df["centroid_lat"].min()),
                float(df["centroid_lon"].max()), float(df["centroid_lat"].max())]
    lidar = df["lidar_date"].dropna()
    return {
        "title": f"PV array geometry from LiDAR{': ' + study if study else ''}",
        "description": (
            "Per-polygon tilt, true-north azimuth and height of photovoltaic arrays, "
            "measured by fitting planes to airborne LiDAR returns inside PV polygons "
            "derived from imagery. Every input polygon has one row; `status` says what "
            "happened to it and `geometry_basis` what its fitted plane represents."
        ),
        "resource_type": "dataset",
        "version": manifest.get("pkg_version"),
        "schema_version": manifest.get("schema_version"),
        "run_id": manifest.get("run_id"),
        "created": manifest.get("run_timestamp_utc"),
        "software": {"name": "pv-geom", "version": manifest.get("pkg_version"),
                     "config_hash": manifest.get("config_hash")},
        "creators": [],                         # to be completed before deposit
        "license": None,                        # to be completed before deposit
        "funding": [],                          # to be completed before deposit
        "keywords": ["photovoltaics", "solar", "LiDAR", "tilt", "azimuth", "rooftop PV"],
        "spatial": {"crs": manifest.get("crs"), "bbox_wgs84": bbox,
                    "azimuth_reference": manifest.get("azimuth_reference", "grid_north")},
        "temporal": {
            "polygon_vintage": vintage.get("polygon_vintage"),
            "lidar_date_min": lidar.min().isoformat() if len(lidar) else None,
            "lidar_date_max": lidar.max().isoformat() if len(lidar) else None,
            "vintage_gap_days": vintage.get("vintage_gap_days"),
        },
        "inputs": manifest.get("inputs", {}),
        "rows": len(out),
        "rows_by_status": {str(k): int(v) for k, v in status.items()},
        "rows_by_geometry_basis": {
            str(k): int(v) for k, v in df["geometry_basis"].value_counts().items()},
        "files": {
            "pv_geom.parquet": "The dataset as GeoParquet (geometry in the run CRS).",
            "pv_geom.csv": "The same rows without geometry, with centroid lon/lat (WGS 84).",
            "data_dictionary.csv": "Every column: type, unit, nullability, description.",
            "status_definitions.csv": "What each `status` means.",
            "geometry_basis_definitions.csv": "What each `geometry_basis` means.",
            "flag_definitions.csv": "What each quality flag means.",
            "manifest.json": "The run's full provenance: inputs, configuration, counts.",
            "metadata.json": "This description.",
            CHECKSUM_FILE: "SHA-256 of every file here (`sha256sum -c`).",
        },
    }


def dataset_readme(md: dict, dictionary: pd.DataFrame) -> str:
    """A human-readable README for the release dataset."""
    t, sp = md["temporal"], md["spatial"]
    gap = t["vintage_gap_days"]
    if gap is None:
        vintage = ("The capture dates of the two inputs are not both recorded, so no row "
                   "is credited as present-by-date.")
    elif gap > 0:
        vintage = (f"The polygons come from imagery dated {t['polygon_vintage']}; the LiDAR "
                   f"was flown by {t['lidar_date_max']}. The polygons are {gap / 365.25:.1f} "
                   f"years newer, so an array built in between has a polygon but was not "
                   f"there to be measured. Use `geometry_basis` to select rows whose plane "
                   f"is known to be the array.")
    else:
        vintage = (f"The polygons come from imagery dated {t['polygon_vintage']} and the "
                   f"LiDAR was flown by {t['lidar_date_max']}: the polygons are no newer "
                   f"than the LiDAR.")
    north = ("Azimuths are clockwise from true north." if sp["azimuth_reference"] == "true_north"
             else "Azimuths are relative to the GRID north of the dataset's CRS, not true north.")

    def _counts(d: dict) -> str:
        return "\n".join(f"| `{k}` | {v:,} |" for k, v in d.items()) or "| (none) | 0 |"

    cols = "\n".join(
        f"| `{r.column}` | {r.unit or ''} | {r.description} |" for r in dictionary.itertuples())
    files = "\n".join(f"| `{k}` | {v} |" for k, v in md["files"].items())
    return f"""# {md['title']}

{md['description']}

- Rows: **{md['rows']:,}** (one per input polygon)
- Produced by pv-geom {md['version']} (schema {md['schema_version']}), run `{md['run_id']}`
- Coordinate reference system: {sp['crs']}
- {north}

## Dates of the inputs

{vintage}

## What happened to each polygon

| `status` | Rows |
| --- | ---: |
{_counts(md['rows_by_status'])}

| `geometry_basis` | Rows |
| --- | ---: |
{_counts(md['rows_by_geometry_basis'])}

Definitions are in `status_definitions.csv` and `geometry_basis_definitions.csv`.

## Using it

- For array geometry, keep `status == "measured"` and a `geometry_basis` of
  `panel_confirmed` or `panel_by_vintage`.
- `surface_unresolved` rows are right for flush-mounted arrays and wrong for racks
  on flat roofs; `unscreened` rows are unverified.
- `flags` lists quality notes per row (`flag_definitions.csv`). Polygons flagged
  `overlaps_polygon` or `duplicate_geometry` may describe the same array twice.
- Tilt is degrees from horizontal. `panel_tilt_unc_deg` and `panel_azimuth_unc_deg`
  are bootstrap spreads of the fit: precision, not accuracy.

## Files

| File | Contents |
| --- | --- |
{files}

## Columns

| Column | Unit | Description |
| --- | --- | --- |
{cols}

## Citation and licence

To be completed before deposit: authors, licence and funding are left empty in
`metadata.json`.
"""
