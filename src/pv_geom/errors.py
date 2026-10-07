"""Errors pv-geom raises on purpose, each with a remedy.

Anything a user can cause and fix — a layer with no CRS, LiDAR in feet, a
missing optional package, a resume against different inputs — is raised as a
:class:`PVGeomError` carrying a one-line ``remedy``. The CLI prints the message
and the remedy without a traceback; library callers can catch the base class or
a specific one. Each also inherits from the builtin it replaced (``ValueError``,
``NotImplementedError``, ...) so existing ``except`` clauses keep working.
"""

from __future__ import annotations


class PVGeomError(Exception):
    """Base class. ``remedy`` says what to do about it."""

    def __init__(self, message: str, remedy: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.remedy = remedy

    def __str__(self) -> str:
        return f"{self.message} ({self.remedy})" if self.remedy else self.message


class InputError(PVGeomError, ValueError):
    """An input layer or option is unusable as given."""


class MissingCRSError(InputError):
    """A layer carries no coordinate reference system."""


class CRSResolutionError(InputError):
    """The run CRS could not be worked out from the inputs."""


class NonMetricCRSError(PVGeomError, NotImplementedError):
    """The LiDAR's CRS is not in metres."""


class PolygonIdError(InputError):
    """The polygon id column is missing or not unique."""


class TileIndexError(InputError):
    """The tile index cannot be matched to tiles."""


class VintageFormatError(InputError):
    """A vintage or date could not be parsed."""


class ResumeMismatchError(PVGeomError, RuntimeError):
    """``--resume`` was asked to continue an output made from different inputs,
    configuration or schema."""


class MissingDependencyError(PVGeomError, ImportError):
    """An optional package is needed for what was asked."""


class StorageError(PVGeomError, OSError):
    """Remote storage could not be read (credentials, permissions, network)."""


def require(module: str, extra: str, purpose: str):
    """Import an optional dependency, or explain how to get it."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependencyError(
            f"{purpose} needs the '{module}' package, which is not installed",
            f'install it with: pip install "pv-geom[{extra}]"',
        ) from exc
