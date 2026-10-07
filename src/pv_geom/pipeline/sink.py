"""Where a run's partitions and manifest go: a local directory or an S3 prefix.

Writes happen on the client (per-group tables return to it either way), so
workers need no write permissions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from pv_geom.io.output import write_partition


class OutputSink:
    """Partition files ``part-<id>.parquet`` plus ``manifest.json`` under one prefix."""

    def __init__(self, output_uri: str | Path) -> None:
        self.uri = str(output_uri).rstrip("/")
        self.is_remote = self.uri.startswith("s3://")
        self._fs: Any = None
        self._root: Path | None = None
        if self.is_remote:
            import fsspec

            self._fs = fsspec.filesystem("s3")
        else:
            self._root = Path(output_uri)
            self._root.mkdir(parents=True, exist_ok=True)

    def _target(self, name: str) -> str | Path:
        if self._root is not None:
            return self._root / name
        return f"{self.uri}/{name}"

    def part_target(self, partition_id: int) -> str | Path:
        return self._target(f"part-{partition_id:05d}.parquet")

    @property
    def manifest_target(self) -> str | Path:
        return self._target("manifest.json")

    def existing_part_ids(self) -> set[int]:
        """Partition ids with a non-empty part file already at the output (one
        listing for S3 outputs rather than a HEAD per group)."""
        if self._root is not None:
            entries = [(p.name, p.stat().st_size) for p in self._root.glob("part-*.parquet")]
        else:
            try:
                infos = self._fs.find(self.uri, detail=True)
            except FileNotFoundError:
                return set()
            entries = [(i.get("name", ""), i.get("size", 0)) for i in infos.values()]
        found: set[int] = set()
        for name, size in entries:
            base = str(name).rsplit("/", 1)[-1]
            if base.startswith("part-") and base.endswith(".parquet") and size > 0:
                try:
                    found.add(int(base[len("part-"):-len(".parquet")]))
                except ValueError:
                    continue
        return found

    def write_part(self, table: pa.Table, partition_id: int, crs: str | None) -> str | Path:
        target = self.part_target(partition_id)
        write_partition(table, target, crs, fs=self._fs)
        return target

    def read_part(self, partition_id: int) -> pa.Table:
        target = self.part_target(partition_id)
        if self._fs is not None:
            with self._fs.open(str(target), "rb") as f:
                return pq.read_table(f)
        return pq.read_table(target)
