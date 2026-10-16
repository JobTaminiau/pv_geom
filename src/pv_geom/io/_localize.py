"""Deprecated alias of :mod:`pv_geom.io.storage`.

``pv_geom.io._localize`` was renamed to ``pv_geom.io.storage`` in 0.3. This
module re-exports the old names so code written against the old path
(building-lod2 ``io/points.py`` and ``pipeline/tile_task.py``) keeps importing.
It will be removed in 0.6; import from ``pv_geom.io.storage`` instead.
"""

from __future__ import annotations

import warnings

from pv_geom.io.storage import DEFAULT_CACHE, RemoteFileMissing, is_remote, localize

__all__ = ["DEFAULT_CACHE", "RemoteFileMissing", "is_remote", "localize"]

warnings.warn(
    "pv_geom.io._localize is deprecated and will be removed in pv_geom 0.6; "
    "import from pv_geom.io.storage instead",
    DeprecationWarning,
    stacklevel=2,
)
