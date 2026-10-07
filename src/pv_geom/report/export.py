"""Write the consolidated release dataset and its dictionary."""

from __future__ import annotations

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
    return dictionary
