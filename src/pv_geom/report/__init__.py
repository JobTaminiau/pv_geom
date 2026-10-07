"""Tables, figures and reports from a pv-geom output."""

from pv_geom.report.build import ReportResult, build_report
from pv_geom.report.data import load_run

__all__ = ["ReportResult", "build_report", "load_run"]


# `pv_geom.report` is both this package and the API function of the same name.
# Importing the package binds it over the function on `pv_geom`, so which one a
# caller got used to depend on import order. Making the package callable keeps
# `pv_geom.report(output)` working either way.
import sys as _sys
import types as _types


class _CallablePackage(_types.ModuleType):
    def __call__(self, *args, **kwargs):
        from pv_geom.api import report

        return report(*args, **kwargs)


_sys.modules[__name__].__class__ = _CallablePackage
