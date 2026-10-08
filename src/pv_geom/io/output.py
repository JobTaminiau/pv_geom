"""Output writer/reader: partitions as valid GeoParquet, and reading a run back.

Partitions are written with pyarrow (the schema is the contract) plus the
GeoParquet ``geo`` metadata key, so any GIS tool opens them as spatial layers.
Releases before 0.2.0 wrote the same WKB column without that key; the reader
handles both.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from pv_geom.errors import require
from pv_geom.schema import MEASURED, NO_FIT, SCHEMA_VERSION, is_recommended


def geo_metadata(table: pa.Table, crs: str | None) -> bytes:
    """GeoParquet 1.0 ``geo`` metadata for a table with a WKB ``geometry`` column."""
    col: dict[str, Any] = {"encoding": "WKB", "geometry_types": ["Polygon"]}
    if crs:
        from pyproj import CRS

        col["crs"] = CRS.from_user_input(crs).to_json_dict()
    if len(table):
        import shapely

        geoms = shapely.from_wkb(table.column("geometry").to_pylist())
        col["bbox"] = [float(v) for v in shapely.total_bounds(geoms)]
    return json.dumps(
        {"version": "1.0.0", "primary_column": "geometry", "columns": {"geometry": col}}
    ).encode("utf-8")


def with_geo_metadata(table: pa.Table, crs: str | None) -> pa.Table:
    md = dict(table.schema.metadata or {})
    md[b"geo"] = geo_metadata(table, crs)
    return table.replace_schema_metadata(md)


# Schema-metadata keys every partition carries, so a partition file is
# self-describing and ``--resume`` can check what it is continuing.
META_SCHEMA_VERSION = b"pv_geom.schema_version"
META_RESULT_HASH = b"pv_geom.result_hash"
META_PLAN_FINGERPRINT = b"pv_geom.plan_fingerprint"


def write_partition(table: pa.Table, target: str | Path, crs: str | None, fs=None,
                    metadata: dict[bytes, str] | None = None) -> None:
    """Write one partition as GeoParquet to a local path or (with ``fs``) S3."""
    table = with_geo_metadata(table, crs)
    md = dict(table.schema.metadata or {})
    md[META_SCHEMA_VERSION] = SCHEMA_VERSION.encode()
    for key, value in (metadata or {}).items():
        md[key] = str(value).encode()
    table = table.replace_schema_metadata(md)
    if fs is not None:
        with fs.open(str(target), "wb") as f:
            pq.write_table(table, f)
    else:
        pq.write_table(table, target)


def _list_parts(output_uri: str) -> tuple[list[str], Any]:
    s = str(output_uri).rstrip("/")
    if s.startswith("s3://"):
        import fsspec

        require("s3fs", "cloud", "reading output from S3")
        fs = fsspec.filesystem("s3")
        return sorted("s3://" + p for p in fs.glob(f"{s}/part-*.parquet")), fs
    return sorted(str(p) for p in Path(s).glob("part-*.parquet")), None


def read_manifest(output_uri: str | Path) -> dict:
    """The run manifest, or ``{}`` when the output has none."""
    s = str(output_uri).rstrip("/")
    try:
        if s.startswith("s3://"):
            import fsspec

            with fsspec.open(f"{s}/manifest.json", "r", encoding="utf-8") as f:
                return json.load(f)
        return json.loads((Path(s) / "manifest.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def read_output_table(output_uri: str | Path) -> pa.Table:
    """All partitions of a run as one Arrow table (schemas unified)."""
    parts, fs = _list_parts(str(output_uri))
    if not parts:
        raise FileNotFoundError(f"no part-*.parquet files under {output_uri}")
    tables = []
    for p in parts:
        if fs is not None:
            with fs.open(p, "rb") as f:
                tables.append(pq.read_table(f))
        else:
            tables.append(pq.read_table(p))
    # Partitions of one run share a schema; "default" promotion also lets a
    # prefix written across versions (missing columns -> null) be read.
    return upgrade_table(pa.concat_tables(
        [t.replace_schema_metadata(None) for t in tables], promote_options="default"
    ))


def upgrade_table(table: pa.Table) -> pa.Table:
    """Bring a table written by an older schema version up to the current one.

    Older outputs lack ``status`` (added in schema 0.3): they held rows only for
    polygons the LiDAR covered, so each is ``measured`` or ``no_fit`` according
    to whether it has a tilt.
    """
    if "status" not in table.column_names and "panel_tilt_deg" in table.column_names:
        fitted = table.column("panel_tilt_deg").is_valid()
        status = pa.array([MEASURED if ok else NO_FIT for ok in fitted.to_pylist()], pa.string())
        table = table.append_column("status", status)
    needed = {"status", "geometry_basis", "flags"}
    if "recommended" not in table.column_names and needed <= set(table.column_names):
        # Added in schema 0.5; derived from columns older outputs already have.
        rec = [is_recommended(st, basis, flags or ()) for st, basis, flags in zip(
            table.column("status").to_pylist(), table.column("geometry_basis").to_pylist(),
            table.column("flags").to_pylist(), strict=True)]
        table = table.append_column("recommended", pa.array(rec, pa.bool_()))
    return table


def output_crs(output_uri: str | Path, manifest: dict | None = None) -> str | None:
    """CRS of a run's geometry: from the manifest, else the GeoParquet metadata."""
    manifest = manifest if manifest is not None else read_manifest(output_uri)
    crs = (manifest.get("crs") or manifest.get("config", {}).get("crs", {}).get("target"))
    if crs and str(crs).lower() != "auto":
        return str(crs)
    parts, fs = _list_parts(str(output_uri))
    if parts:
        if fs is not None:
            with fs.open(parts[0], "rb") as f:
                md = pq.read_schema(f).metadata or {}
        else:
            md = pq.read_schema(parts[0]).metadata or {}
        if b"geo" in md:
            from pyproj import CRS

            geo = json.loads(md[b"geo"])
            c = geo["columns"]["geometry"].get("crs")
            if c:
                return CRS.from_json_dict(c).to_string()
    return None


def read_output(output_uri: str | Path):
    """A run's rows as a GeoDataFrame in the run CRS."""
    import geopandas as gpd
    import shapely

    table = read_output_table(output_uri)
    df = table.to_pandas()
    geom = shapely.from_wkb(df.pop("geometry").to_numpy())
    return gpd.GeoDataFrame(df, geometry=geom, crs=output_crs(output_uri))
