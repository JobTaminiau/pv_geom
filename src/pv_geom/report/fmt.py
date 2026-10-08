"""Number formatting shared by the report's tables and prose."""

from __future__ import annotations

import numpy as np


def pct(x, digits=1) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{100 * x:.{digits}f}%"


def num(x, digits=1) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:,.{digits}f}"


def integer(x) -> str:
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{int(x):,}"
