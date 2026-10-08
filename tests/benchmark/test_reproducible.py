"""Story G5: the same inputs give the same output.

Two promises are tested:

* **On one machine, bit for bit.** Running twice gives the same content hash.
* **Across platforms, to a tolerance.** The synthetic benchmark's output is
  committed as a reference. Each CI job (Linux and Windows, two Python
  versions, one lock file) must reproduce every row, label and count exactly
  and every number to within ``TOLERANCE``. Bit-for-bit agreement across
  platforms is not attainable: linear-algebra libraries differ in the last
  bits by platform and processor.

Set PV_GEOM_UPDATE_GOLDEN=1 to accept a deliberate change to the reference.
"""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow.parquet as pq

from pv_geom.io.output import read_output_table
from pv_geom.reproduce import VOLATILE_COLUMNS, compare_tables, content_digest
from pv_geom.sample import write_scene
from pv_geom.testing import run_benchmark

REFERENCE = Path(__file__).with_name("synthetic_reference.parquet")
# Degrees for angles, metres for lengths, square metres for areas.
TOLERANCE = 1e-4          # measured across CI platforms: 1.5e-5 deg, 1e-8 m


def _run(tmp_path: Path, name: str):
    return read_output_table(run_benchmark(write_scene(tmp_path / f"scene_{name}"),
                                           tmp_path / f"out_{name}"))


def test_two_runs_on_this_machine_are_bit_for_bit_identical(tmp_path: Path) -> None:
    assert content_digest(_run(tmp_path, "a")) == content_digest(_run(tmp_path, "b"))


def test_synthetic_benchmark_matches_the_committed_reference(tmp_path: Path) -> None:
    actual = _run(tmp_path, "a")
    if os.environ.get("PV_GEOM_UPDATE_GOLDEN"):
        pq.write_table(actual.drop_columns(
            [c for c in VOLATILE_COLUMNS if c in actual.column_names]), REFERENCE)
    comparison = compare_tables(pq.read_table(REFERENCE), actual, tolerance=TOLERANCE)
    assert comparison.equivalent, comparison.summary()
