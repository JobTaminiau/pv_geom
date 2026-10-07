"""pv_geom: tilt, orientation and height of solar PV polygons, measured from LiDAR.

    import pv_geom

    result = pv_geom.run("configs/area.yaml")
    gdf = result.load()
    result.report()
"""

__version__ = "0.4.0.dev0"

__all__ = ["RunResult", "__version__", "describe", "load", "report", "run"]

_API = frozenset({"RunResult", "describe", "load", "report", "run"})


def __getattr__(name: str):
    # The API is imported on first use so that `import pv_geom` (and
    # `pv_geom.__version__`) stays instant and free of heavy imports.
    if name in _API:
        import importlib

        value = getattr(importlib.import_module("pv_geom.api"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'pv_geom' has no attribute {name!r}")
