"""Story G5: the same inputs give the same output, bit for bit, on every
platform CI runs on.

The fingerprint of the synthetic benchmark's output is committed. Each CI job
(Linux and Windows, two Python versions, all from the same lock file)
recomputes it. Set PV_GEOM_UPDATE_GOLDEN=1 to accept a deliberate change.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pv_geom.io.output import read_output_table
from pv_geom.reproduce import column_digests, content_digest
from pv_geom.sample import write_scene
from pv_geom.testing import run_benchmark

EXPECTED = Path(__file__).with_name("synthetic_fingerprint.json")


def test_synthetic_benchmark_is_bit_for_bit_reproducible(tmp_path: Path) -> None:
    table = read_output_table(run_benchmark(write_scene(tmp_path / "scene"), tmp_path / "out"))
    actual = {"content_hash": content_digest(table), "columns": column_digests(table)}
    if os.environ.get("PV_GEOM_UPDATE_GOLDEN"):
        EXPECTED.write_text(json.dumps(actual, indent=2) + "\n", encoding="utf-8", newline="\n")
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    differing = sorted(c for c in set(actual["columns"]) | set(expected["columns"])
                       if actual["columns"].get(c) != expected["columns"].get(c))
    assert not differing, f"columns that differ from the committed fingerprint: {differing}"
    assert actual["content_hash"] == expected["content_hash"]
