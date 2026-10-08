"""Reproducing a run, and proving that it reproduced.

Three things have to be pinned down for "same inputs, same output" to be
checkable:

**What the output contains.** ``content_hash`` is a fingerprint of every row's
measured content. It ignores what legitimately differs between two runs of the
same thing (the run id, how the work was partitioned, which version wrote it)
and the order rows come in, so a serial run on a laptop and a cluster run agree.

**What went in.** ``input_fingerprint`` records the size (and, for local files,
the SHA-256) of the polygon layer and footprints, and the names and sizes of
the LiDAR tiles read.

**What it ran on.** ``environment`` records the Python, platform and library
versions. The repository's ``uv.lock`` pins them; this says which were used.

All three are written to the manifest. ``verify`` checks an output against its
own manifest; ``compare_outputs`` says how two outputs differ; ``reproduce``
reruns a manifest and compares.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from pv_geom.errors import InputError
from pv_geom.io.output import read_manifest, read_output_table
from pv_geom.io.storage import is_remote, object_sizes

# Columns that describe the run rather than the measurement.
VOLATILE_COLUMNS = ("run_id", "config_hash", "pkg_version", "partition_id")

# Libraries whose versions can change a result.
_LIBRARIES = ("numpy", "pandas", "geopandas", "shapely", "pyproj", "pyarrow", "laspy",
              "lazrs", "laszip", "pdal", "dask", "distributed")

# Local files up to this size are hashed in full; larger ones by size only.
_HASH_LIMIT_BYTES = 2_000_000_000


# --------------------------------------------------------------------------- #
# content
# --------------------------------------------------------------------------- #

def _canonical(value: Any) -> str:
    """One value of a nested column as text that is the same wherever and
    whenever it is produced: floats by their exact bits, bytes by digest."""
    if value is None:
        return ""
    if isinstance(value, bytes | bytearray | memoryview):
        return hashlib.blake2b(bytes(value), digest_size=16).hexdigest()
    if isinstance(value, float | np.floating):
        return "nan" if np.isnan(value) else float(value).hex()
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}={_canonical(v)}" for k, v in sorted(value.items())) + "}"
    if isinstance(value, list | tuple | np.ndarray):
        return "[" + ",".join(_canonical(v) for v in value) + "]"
    return str(value)


def content_digest(table: pa.Table) -> str:
    """Fingerprint of the rows' measured content (see the module docstring)."""
    table = table.drop_columns([c for c in VOLATILE_COLUMNS if c in table.column_names])
    table = table.select(sorted(table.column_names))
    frame = table.to_pandas()
    for name in frame.columns:
        column = frame[name]
        if column.dtype == object:
            frame[name] = column.map(_canonical)
        elif column.dtype.kind == "M":
            frame[name] = column.astype("int64")
    rows = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
    h = hashlib.sha256()
    h.update(",".join(frame.columns).encode())
    h.update(np.sort(rows).tobytes())
    return h.hexdigest()


def column_digests(table: pa.Table) -> dict[str, str]:
    """A fingerprint per column (rows in polygon order), for finding out *what*
    differs when two content hashes do not match."""
    table = table.drop_columns([c for c in VOLATILE_COLUMNS if c in table.column_names])
    frame = table.to_pandas().sort_values("polygon_id", kind="stable")
    out = {}
    for name in sorted(frame.columns):
        text = chr(10).join(frame[name].map(_canonical))
        out[name] = hashlib.sha256(text.encode()).hexdigest()[:16]
    return out


def content_hash(output: str | Path) -> str:
    """``content_digest`` of a finished run, read back from its partitions."""
    return content_digest(read_output_table(output))


# --------------------------------------------------------------------------- #
# inputs and environment
# --------------------------------------------------------------------------- #

def _file_record(uri: str | None) -> dict[str, Any] | None:
    if not uri:
        return None
    record: dict[str, Any] = {"uri": str(uri)}
    sizes = object_sizes([str(uri)])
    if str(uri) in sizes:
        record["bytes"] = int(sizes[str(uri)])
    path = Path(str(uri))
    if not is_remote(str(uri)) and path.is_file() and path.stat().st_size <= _HASH_LIMIT_BYTES:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        record["sha256"] = h.hexdigest()
    return record


def input_fingerprint(inputs: dict[str, str | None], tile_uris: list[str]) -> dict[str, Any]:
    """What a run read: the vector layers by size and hash, the tiles by name
    and size (a LiDAR collection is too large to hash on every run)."""
    sizes = object_sizes(sorted(tile_uris))
    lines = sorted(f"{uri.replace(chr(92), '/').rsplit('/', 1)[-1]}:{size}"
                   for uri, size in sizes.items())
    return {
        "polygons": _file_record(inputs.get("polygons")),
        "tile_index": _file_record(inputs.get("tile_index")),
        "footprints": _file_record(inputs.get("footprints")),
        "lidar": {
            "tiles": len(tile_uris), "tiles_sized": len(sizes),
            "bytes": int(sum(sizes.values())),
            "names_and_sizes_sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
        },
    }


def environment() -> dict[str, Any]:
    """The interpreter, platform and library versions in use."""
    libraries = {}
    for name in _LIBRARIES:
        try:
            libraries[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    try:
        import pyproj

        libraries["PROJ"] = pyproj.proj_version_str
    except Exception:                                    # pragma: no cover
        pass
    return {
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "libraries": libraries,
    }


# --------------------------------------------------------------------------- #
# comparing and reproducing
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Comparison:
    """How two outputs differ."""

    identical: bool                 # bit for bit
    hash_a: str
    hash_b: str
    rows_a: int
    rows_b: int
    only_in_a: int = 0
    only_in_b: int = 0
    columns: dict[str, dict[str, Any]] = field(default_factory=dict)   # differing columns
    tolerance: float = 0.0
    # True when the outputs hold the same rows, every non-numeric value is the
    # same, and every number agrees to within ``tolerance``.
    equivalent: bool = False

    def summary(self) -> str:
        if self.identical:
            return f"identical: {self.rows_a:,} rows, content hash {self.hash_a[:16]}"
        head = (f"equivalent within {self.tolerance:g} (not bit for bit)" if self.equivalent
                else "DIFFERENT")
        lines = [f"{head}: {self.rows_a:,} rows vs {self.rows_b:,}"]
        if self.only_in_a or self.only_in_b:
            lines.append(f"  polygons only in the first: {self.only_in_a:,}; "
                         f"only in the second: {self.only_in_b:,}")
        for name, d in self.columns.items():
            extra = f", largest difference {d['max_abs_diff']:.6g}" if "max_abs_diff" in d else ""
            lines.append(f"  {name}: {d['rows']:,} rows differ{extra}")
        return "\n".join(lines)


def _facet_difference(x: pd.Series, y: pd.Series) -> dict[str, Any]:
    """Difference between two facet columns: numeric where the facets line up
    (same count per polygon, same footprints), structural otherwise."""
    rows, worst, structural = 0, 0.0, False
    for fa, fb in zip(x, y, strict=True):
        la = [] if fa is None else list(fa)
        lb = [] if fb is None else list(fb)
        if _canonical(la) == _canonical(lb):
            continue
        rows += 1
        if len(la) != len(lb):
            structural = True
            continue
        for sa, sb in zip(la, lb, strict=True):
            for key in sa:
                va, vb = sa[key], sb[key]
                if isinstance(va, float | np.floating) or isinstance(vb, float | np.floating):
                    if va is None or vb is None or np.isnan(va) != np.isnan(vb):
                        structural = True
                    elif not np.isnan(va):
                        worst = max(worst, abs(float(va) - float(vb)))
                elif key != "geometry" and va != vb:
                    structural = True
    if not rows:
        return {}
    return {"rows": rows, "note": "facet structure differs"} if structural else {
        "rows": rows, "max_abs_diff": worst}


def compare_tables(a: pa.Table, b: pa.Table, tolerance: float = 0.0) -> Comparison:
    """Compare two outputs. They are ``identical`` when their content hashes
    match, and ``equivalent`` when the only differences are numbers that agree
    to within ``tolerance`` (in each column's own unit)."""
    hash_a, hash_b = content_digest(a), content_digest(b)
    if hash_a == hash_b:
        return Comparison(True, hash_a, hash_b, len(a), len(b), tolerance=tolerance,
                          equivalent=True)

    def frame(t: pa.Table) -> pd.DataFrame:
        t = t.drop_columns([c for c in VOLATILE_COLUMNS if c in t.column_names])
        return t.to_pandas().set_index("polygon_id").sort_index()

    fa, fb = frame(a), frame(b)
    shared = fa.index.intersection(fb.index)
    columns: dict[str, dict[str, Any]] = {}
    for name in sorted(set(fa.columns) | set(fb.columns)):
        if name not in fa.columns or name not in fb.columns:
            columns[name] = {"rows": len(shared), "note": "column missing from one output"}
            continue
        x, y = fa.loc[shared, name], fb.loc[shared, name]
        if x.dtype.kind in "fiu" and y.dtype.kind in "fiu":
            xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            differ = ~((xv == yv) | (np.isnan(xv) & np.isnan(yv)))
            if differ.any():
                both = differ & ~np.isnan(xv) & ~np.isnan(yv)
                d: dict[str, Any] = {"rows": int(differ.sum())}
                if both.any():
                    d["max_abs_diff"] = float(np.abs(xv - yv)[both].max())
                columns[name] = d
        elif name == "facets":
            d = _facet_difference(x, y)
            if d:
                columns[name] = d
        else:
            differ = x.map(_canonical).to_numpy() != y.map(_canonical).to_numpy()
            if differ.any():
                columns[name] = {"rows": int(differ.sum())}
    only_a, only_b = len(fa.index.difference(fb.index)), len(fb.index.difference(fa.index))
    numeric_only = all(set(d) == {"rows", "max_abs_diff"} for d in columns.values())
    equivalent = (only_a == 0 and only_b == 0 and numeric_only
                  and all(d["max_abs_diff"] <= tolerance for d in columns.values()))
    return Comparison(False, hash_a, hash_b, len(a), len(b), only_in_a=only_a,
                      only_in_b=only_b, columns=columns, tolerance=tolerance,
                      equivalent=equivalent)


def compare_outputs(a: str | Path, b: str | Path, tolerance: float = 0.0) -> Comparison:
    """How two finished runs differ, row by row and column by column."""
    return compare_tables(read_output_table(a), read_output_table(b), tolerance)


def verify(output: str | Path) -> dict[str, Any]:
    """Check an output against the content hash in its own manifest."""
    manifest = read_manifest(output)
    recorded = (manifest.get("reproducibility") or {}).get("content_hash")
    actual = content_hash(output)
    return {"recorded": recorded, "actual": actual,
            "intact": recorded is not None and recorded == actual}


def environment_differences(then: dict[str, Any], now: dict[str, Any]) -> list[str]:
    """What differs between two environment records, as readable lines."""
    out = []
    for key in ("python", "platform", "machine"):
        if then.get(key) != now.get(key):
            out.append(f"{key}: {then.get(key)} -> {now.get(key)}")
    a, b = then.get("libraries", {}), now.get("libraries", {})
    for name in sorted(set(a) | set(b)):
        if a.get(name) != b.get(name):
            out.append(f"{name}: {a.get(name, 'absent')} -> {b.get(name, 'absent')}")
    return out


def reproduce(output: str | Path, out_dir: str | Path, *, use_dask: bool = False
              ) -> tuple[Comparison, list[str]]:
    """Rerun a finished run from its manifest into ``out_dir`` and compare.

    Returns the comparison and any notes: environment differences, and inputs
    that are not what the original run read (in which case a difference in the
    output proves nothing about the code).
    """
    from pv_geom import api
    from pv_geom.config import PVGeomConfig

    manifest = read_manifest(output)
    if not manifest.get("config"):
        raise InputError(f"{output} has no manifest to reproduce from",
                         "give the output directory of a finished run")
    record = manifest.get("reproducibility") or {}
    notes = [f"environment: {line}" for line in
             environment_differences(record.get("environment", {}), environment())]

    cfg = PVGeomConfig.model_validate(manifest["config"])
    cfg.study.output = None
    inputs = manifest.get("inputs") or {}
    result = api.run(
        cfg, polygons=inputs.get("polygons"), lidar_prefix=inputs.get("lidar_prefix"),
        tile_index=inputs.get("tile_index"), footprints=inputs.get("footprints"),
        crs=manifest.get("crs"), output=out_dir, use_dask=use_dask)

    then = record.get("inputs") or {}
    now = (read_manifest(result.output).get("reproducibility") or {}).get("inputs") or {}
    for key in ("polygons", "tile_index", "footprints"):
        a, b = then.get(key) or {}, now.get(key) or {}
        if a.get("sha256", a.get("bytes")) != b.get("sha256", b.get("bytes")):
            notes.append(f"input changed: {key} is not the file the original run read")
    if (then.get("lidar") or {}).get("names_and_sizes_sha256") != \
            (now.get("lidar") or {}).get("names_and_sizes_sha256"):
        notes.append("input changed: the LiDAR tiles differ in name or size")
    return compare_outputs(output, result.output), notes
