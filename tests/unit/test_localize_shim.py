"""The old ``pv_geom.io._localize`` path still works (deprecated alias)."""

from __future__ import annotations

import importlib
import sys

import pytest

from pv_geom.io import storage


def _fresh_import():
    sys.modules.pop("pv_geom.io._localize", None)
    return importlib.import_module("pv_geom.io._localize")


def test_old_path_reexports_storage_names():
    with pytest.warns(DeprecationWarning, match="pv_geom.io.storage"):
        mod = _fresh_import()
    assert mod.localize is storage.localize
    assert mod.is_remote is storage.is_remote
    assert mod.RemoteFileMissing is storage.RemoteFileMissing
    assert mod.DEFAULT_CACHE == storage.DEFAULT_CACHE


def test_old_import_statement_used_by_building_lod2(tmp_path):
    with pytest.warns(DeprecationWarning):
        sys.modules.pop("pv_geom.io._localize", None)
        from pv_geom.io._localize import RemoteFileMissing, is_remote, localize
    assert issubclass(RemoteFileMissing, FileNotFoundError)
    assert is_remote("s3://bucket/key.laz")
    assert not is_remote(tmp_path / "a.laz")
    assert localize(tmp_path / "a.laz") == tmp_path / "a.laz"
