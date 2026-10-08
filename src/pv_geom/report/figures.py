"""Publication figures for a pv-geom output.

Figures are sized in millimetres to journal column widths, drawn with 7 pt
type, and saved as PNG (300 dpi) plus vector PDF and SVG with live text, so
they drop into a manuscript or a slide without being redrawn. Each ``fig_*``
function returns a :class:`FigureSpec` (or ``None`` when the data cannot
support the figure) and never touches global matplotlib state.

Colour carries one job per figure: a single blue for magnitude, grey for
context, and blue-versus-grey for "the stratum the report leads with" against
"everything fitted".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from pv_geom.report.stats import (
    BASIS_LABELS,
    STRATA,
    stratum_mask,
    weighted_quantile,
)
from pv_geom.vintage import ARRAY_BASES, GEOMETRY_BASIS

MM = 1.0 / 25.4
SINGLE_COL_MM = 89.0
FULL_WIDTH_MM = 183.0

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CONTEXT_FILL = "#e1e0d9"
CONTEXT_LINE = "#a9a79f"
SURFACE = "#ffffff"

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "pvgeom_blues", ["#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.5,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "text.color": INK,
    "legend.frameon": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "svg.fonttype": "none",       # keep text as text in SVG
    "pdf.fonttype": 42,           # embed TrueType so text stays editable in PDF
}


@dataclass
class FigureSpec:
    name: str
    title: str
    caption: str
    fig: Figure


def _figure(width_mm: float, height_mm: float) -> Figure:
    return Figure(figsize=(width_mm * MM, height_mm * MM), layout="constrained")


def save(spec: FigureSpec, out_dir: Path, formats=("png", "pdf", "svg"), dpi: int = 300) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    with matplotlib.rc_context(RC):
        for fmt in formats:
            path = out_dir / f"{spec.name}.{fmt}"
            spec.fig.savefig(path, dpi=dpi)
            paths[fmt] = path
    return paths


def _weight_label(weight: str) -> str:
    return "array surface" if weight == "area" else "polygons"


def _w(sub: pd.DataFrame, weight: str) -> np.ndarray:
    return sub["w_area"].to_numpy(dtype=float) if weight == "area" else np.ones(len(sub))


def _series(df: pd.DataFrame, headline: str) -> list[tuple[str, str, pd.DataFrame]]:
    """(key, label, rows) for the context series (all fitted) and, when it is a
    proper subset, the headline stratum."""
    out = [("all_fitted", STRATA["all_fitted"][0], df[stratum_mask(df, "all_fitted")])]
    if headline != "all_fitted":
        out.append((headline, STRATA[headline][0], df[stratum_mask(df, headline)]))
    return out


# --------------------------------------------------------------------------- #
# Panels (draw onto a given axis so they can be reused in the overview)
# --------------------------------------------------------------------------- #

def _draw_tilt(ax, df: pd.DataFrame, headline: str, weight: str, *, max_deg: float = 60.0,
               bin_deg: float = 2.0, legend: bool = True) -> None:
    edges = np.arange(0.0, max_deg + bin_deg, bin_deg)
    series = _series(df, headline)
    ymax = 0.0
    legend_loc = "upper right"
    for i, (_key, label, sub) in enumerate(series):
        tilt = np.clip(sub["tilt_deg"].to_numpy(dtype=float), 0, max_deg - 1e-6)
        w = _w(sub, weight)
        if w.sum() <= 0:
            continue
        hist, _ = np.histogram(tilt, bins=edges, weights=w)
        share = 100.0 * hist / w.sum()
        ymax = max(ymax, float(share.max()))
        is_context = len(series) > 1 and i == 0
        if is_context:
            ax.stairs(share, edges, fill=True, color=CONTEXT_FILL, label=label, zorder=1)
            ax.stairs(share, edges, color=CONTEXT_LINE, linewidth=0.6, zorder=2)
        else:
            if len(series) == 1:
                ax.stairs(share, edges, fill=True, color=BLUE, alpha=0.18, zorder=1)
            ax.stairs(share, edges, color=BLUE, linewidth=1.2, label=label, zorder=3)
            med = float(weighted_quantile(sub["tilt_deg"], w, 0.5)[0])
            ax.axvline(med, color=BLUE, linewidth=0.6, zorder=2)
            # Legend and median label take opposite sides of the median line.
            if med > 0.4 * max_deg:
                legend_loc, left = "upper left", False
            else:
                left = len(series) > 1 and med > 0.2 * max_deg
            ax.annotate(f"median {med:.1f}°", xy=(med, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(-3 if left else 3, -2), textcoords="offset points", va="top",
                        ha="right" if left else "left", fontsize=6.5, color=INK_2)
    ax.set_xlim(0, max_deg)
    ax.set_ylim(0, ymax * 1.3 if ymax > 0 else 1)
    ax.set_xlabel("Tilt from horizontal (°)")
    ax.set_ylabel(f"Share of {_weight_label(weight)} (%)")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    if legend and len(series) > 1:
        ax.legend(loc=legend_loc, handlelength=1.4, borderaxespad=0.2)


def _draw_rose(ax, df: pd.DataFrame, headline: str, weight: str, *, n_sectors: int = 24) -> None:
    width = 2 * np.pi / n_sectors
    centres = np.arange(n_sectors) * width
    series = _series(df, headline)
    rmax = 0.0
    for i, (_key, label, sub) in enumerate(series):
        sub = sub[sub["azimuth_deg"].notna()]
        w = _w(sub, weight)
        if w.sum() <= 0:
            continue
        idx = (np.floor(((sub["azimuth_deg"].to_numpy(dtype=float) + 180.0 / n_sectors)
                         % 360.0) / (360.0 / n_sectors))).astype(int)
        share = 100.0 * np.bincount(idx, weights=w, minlength=n_sectors) / w.sum()
        rmax = max(rmax, float(share.max()))
        is_context = len(series) > 1 and i == 0
        if is_context:
            ax.bar(centres, share, width=width, color=CONTEXT_FILL, edgecolor=SURFACE,
                   linewidth=0.4, label=label, zorder=1)
        else:
            ax.bar(centres, share, width=width * (0.62 if len(series) > 1 else 0.9),
                   color=BLUE, edgecolor=SURFACE, linewidth=0.3, label=label, zorder=3)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    ax.tick_params(axis="x", pad=-2)
    step = next((s for s in (1, 2, 5, 10, 20, 25, 50) if rmax / s <= 3.5), 50)
    ticks = np.arange(step, rmax + step, step)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}%" for t in ticks], fontsize=5.5, color=MUTED)
    ax.set_rlabel_position(22.5)
    ax.set_ylim(0, ticks[-1] if len(ticks) else 1)
    ax.grid(color=GRID, linewidth=0.5)
    ax.spines["polar"].set_color(AXIS)
    ax.spines["polar"].set_linewidth(0.5)
    ax.set_axisbelow(True)


def _draw_heatmap(ax, fig: Figure, df: pd.DataFrame, headline: str, weight: str, *,
                  max_deg: float = 60.0) -> None:
    sub = df[stratum_mask(df, headline) & df["azimuth_deg"].notna()]
    az_edges = np.arange(0.0, 360.0 + 15.0, 15.0)
    tilt_edges = np.arange(0.0, max_deg + 5.0, 5.0)
    w = _w(sub, weight)
    hist, _, _ = np.histogram2d(
        sub["azimuth_deg"].to_numpy(dtype=float),
        np.clip(sub["tilt_deg"].to_numpy(dtype=float), 0, max_deg - 1e-6),
        bins=[az_edges, tilt_edges], weights=w,
    )
    share = 100.0 * hist / w.sum() if w.sum() > 0 else hist
    mesh = ax.pcolormesh(az_edges, tilt_edges, share.T, cmap=SEQUENTIAL, vmin=0,
                         edgecolors=SURFACE, linewidth=0.3)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xticklabels(["N", "E", "S", "W", "N"])
    ax.set_xlabel("Azimuth (direction the array faces)")
    ax.set_ylabel("Tilt from horizontal (°)")
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(mesh, ax=ax, pad=0.02, aspect=18)
    cb.set_label(f"Share of {_weight_label(weight)} (%)")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def _headline_note(headline: str) -> str:
    if headline == "all_fitted":
        return "All polygons with a fitted plane."
    return (f"Blue: {STRATA[headline][0].lower()} polygons, where the fitted plane is known "
            f"to be the array. Grey: all polygons with a fitted plane.")


def fig_tilt_distribution(df: pd.DataFrame, headline: str, weight: str = "area") -> FigureSpec:
    with matplotlib.rc_context(RC):
        fig = _figure(SINGLE_COL_MM, 62)
        ax = fig.add_subplot()
        _draw_tilt(ax, df, headline, weight)
    return FigureSpec(
        "tilt_distribution", "Tilt profile",
        f"Distribution of array tilt, as share of {_weight_label(weight)} in 2° bins "
        f"(tilts above 60° are counted in the last bin). {_headline_note(headline)}", fig)


def fig_azimuth_rose(df: pd.DataFrame, headline: str, weight: str = "area") -> FigureSpec:
    with matplotlib.rc_context(RC):
        fig = _figure(SINGLE_COL_MM, 80)
        ax = fig.add_subplot(projection="polar")
        _draw_rose(ax, df, headline, weight)
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 1:
            fig.legend(handles, labels, loc="outside lower center", ncols=2, handlelength=1.2)
    return FigureSpec(
        "azimuth_rose", "Orientation profile",
        f"Direction arrays face, as share of {_weight_label(weight)} in 15° sectors. "
        f"Near-horizontal arrays have no defined azimuth and are excluded. "
        f"{_headline_note(headline)}", fig)


def fig_tilt_azimuth(df: pd.DataFrame, headline: str, weight: str = "area") -> FigureSpec:
    with matplotlib.rc_context(RC):
        fig = _figure(SINGLE_COL_MM, 58)
        ax = fig.add_subplot()
        _draw_heatmap(ax, fig, df, headline, weight)
    return FigureSpec(
        "tilt_azimuth_joint", "Joint tilt and orientation",
        f"Share of {_weight_label(weight)} by azimuth (15° bins) and tilt (5° bins) for "
        f"{STRATA[headline][0].lower()}.", fig)


def fig_overview(df: pd.DataFrame, headline: str, weight: str = "area") -> FigureSpec:
    with matplotlib.rc_context(RC):
        fig = _figure(FULL_WIDTH_MM, 62)
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.8, 1.15])
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1], projection="polar")
        ax_c = fig.add_subplot(gs[0, 2])
        _draw_tilt(ax_a, df, headline, weight)
        _draw_rose(ax_b, df, headline, weight)
        _draw_heatmap(ax_c, fig, df, headline, weight)
        ax_a.set_title("a  Tilt")
        ax_b.set_title("b  Orientation", pad=10)
        ax_c.set_title("c  Tilt by orientation")
    return FigureSpec(
        "geometry_overview", "Array geometry overview",
        f"Array geometry measured from LiDAR. (a) Tilt distribution. (b) Direction arrays "
        f"face, 15° sectors. (c) Joint distribution for {STRATA[headline][0].lower()}. "
        f"All as share of {_weight_label(weight)}. {_headline_note(headline)}", fig)


def fig_geometry_basis(df: pd.DataFrame) -> FigureSpec:
    n_total = len(df)
    counts = df["geometry_basis"].value_counts()
    cats = list(GEOMETRY_BASIS)
    n = np.array([int(counts.get(c, 0)) for c in cats])
    share = 100.0 * n / n_total if n_total else np.zeros(len(cats))
    with matplotlib.rc_context(RC):
        fig = _figure(SINGLE_COL_MM, 46)
        ax = fig.add_subplot()
        y = np.arange(len(cats))[::-1]
        colours = [BLUE if c in ARRAY_BASES else CONTEXT_LINE for c in cats]
        ax.barh(y, share, height=0.58, color=colours)
        for yi, s, k in zip(y, share, n, strict=True):
            ax.annotate(f"{s:.1f}%  ({k:,})", xy=(s, yi), xytext=(3, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=6.5, color=INK_2)
        ax.set_yticks(y)
        ax.set_yticklabels([BASIS_LABELS[c] for c in cats])
        ax.set_xlim(0, max(float(share.max()) * 1.45, 10))
        ax.set_xlabel("Share of polygons (%)")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
    return FigureSpec(
        "geometry_basis", "What each measurement rests on",
        "Polygons by geometry basis. Blue categories are those where the fitted plane is "
        "known to be the array: either it stands measurably above the roof, or the polygon "
        "vintage is no later than the LiDAR.", fig)


def fig_vintage_timeline(df: pd.DataFrame, manifest: dict | None = None) -> FigureSpec | None:
    lidar = pd.to_datetime(df["lidar_date"].dropna())
    poly = pd.to_datetime(df["polygon_vintage"].dropna())
    v = (manifest or {}).get("vintage", {})
    if lidar.empty and v.get("lidar_flight_end"):
        lidar = pd.to_datetime(pd.Series(
            [v.get("lidar_flight_start") or v["lidar_flight_end"], v["lidar_flight_end"]]))
    if poly.empty and v.get("polygon_vintage"):
        poly = pd.to_datetime(pd.Series([v["polygon_vintage"]]))
    if lidar.empty or poly.empty:
        return None

    def _yr(ts) -> float:
        return ts.year + (ts.dayofyear - 1) / 365.25

    l0, l1 = _yr(lidar.min()), _yr(lidar.max())
    p0, p1 = _yr(poly.min()), _yr(poly.max())
    lo, hi = min(l0, p0), max(l1, p1)
    pad = max(0.6, 0.18 * (hi - lo))
    gap_years = (poly.max() - lidar.max()).days / 365.25

    with matplotlib.rc_context(RC):
        fig = _figure(SINGLE_COL_MM, 34)
        ax = fig.add_subplot()
        ax.axhline(0, color=AXIS, linewidth=0.5, zorder=1)
        if gap_years > 0:
            ax.axvspan(l1, p1, ymin=0.4, ymax=0.6, color=CONTEXT_FILL, zorder=0)
            ax.annotate(f"{gap_years:.1f} yr gap: built here, not in the LiDAR",
                        xy=((l1 + p1) / 2, 0), xytext=(0, 12), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5, color=INK_2)
        for a, b, colour, label, dy in [
            (l0, l1, BLUE, f"LiDAR flown\n{lidar.max():%Y-%m-%d}", -1),
            (p0, p1, ORANGE, f"Polygon imagery\n{poly.max():%Y-%m-%d}", -1),
        ]:
            ax.plot([a, b], [0, 0], color=colour, linewidth=3, solid_capstyle="round", zorder=3)
            ax.plot([b], [0], marker="o", markersize=5, color=colour,
                    markeredgecolor=SURFACE, markeredgewidth=1, zorder=4)
            ax.annotate(label, xy=(b, 0), xytext=(0, 9 * dy), textcoords="offset points",
                        ha="center", va="top", fontsize=6.5, color=INK_2)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=7))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
    caption = ("Capture dates of the two inputs. " + (
        f"The polygon imagery is {gap_years:.1f} years newer than the LiDAR, so "
        f"installations built in between appear as polygons but are absent from the "
        f"point cloud." if gap_years > 0 else
        "The polygons are no newer than the LiDAR, so every installation was present "
        "when the LiDAR was flown."))
    return FigureSpec("vintage_timeline", "Input vintages", caption, fig)


def _hist_panel(ax, values: np.ndarray, edges: np.ndarray, xlabel: str, unit_fmt: str) -> None:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        ax.set_axis_off()
        return
    hist, _ = np.histogram(np.clip(values, edges[0], edges[-1] - 1e-9), bins=edges)
    share = 100.0 * hist / len(values)
    ax.stairs(share, edges, fill=True, color=BLUE, alpha=0.18)
    ax.stairs(share, edges, color=BLUE, linewidth=1.0)
    med = float(np.median(values))
    ax.axvline(med, color=BLUE, linewidth=0.6)
    ax.annotate(f"median {unit_fmt.format(med)}", xy=(med, 1.0),
                xycoords=("data", "axes fraction"), xytext=(3, -2),
                textcoords="offset points", va="top", ha="left", fontsize=6.5, color=INK_2)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, float(share.max()) * 1.18)
    ax.set_xlabel(xlabel)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)


def fig_fit_quality(df: pd.DataFrame) -> FigureSpec:
    fitted = df[df["fitted"]]
    with matplotlib.rc_context(RC):
        fig = _figure(FULL_WIDTH_MM, 52)
        axes = fig.subplots(1, 3)
        _hist_panel(axes[0], 100.0 * fitted["fit_rmse_m"].to_numpy(dtype=float),
                    np.linspace(0, 10, 41), "Plane-fit RMSE (cm)", "{:.1f} cm")
        _hist_panel(axes[1], fitted["tilt_unc_deg"].to_numpy(dtype=float),
                    np.linspace(0, 3, 31), "Tilt uncertainty, 1σ (°)", "{:.2f}°")
        _hist_panel(axes[2], df["point_density"].to_numpy(dtype=float),
                    np.linspace(0, 40, 41), "LiDAR returns per m² of polygon", "{:.1f}")
        axes[0].set_ylabel("Share of polygons (%)")
        for ax, t in zip(axes, ["a  Fit residual", "b  Tilt precision", "c  Point density"],
                         strict=True):
            ax.set_title(t)
    return FigureSpec(
        "fit_quality", "Measurement quality",
        "Per-polygon measurement quality. (a) Perpendicular RMSE of LiDAR returns about the "
        "fitted plane. (b) Bootstrap 1σ uncertainty of the tilt. (c) Returns per square "
        "metre of polygon. Values beyond the axis range are counted in the last bin.", fig)


def fig_roof_relation(df: pd.DataFrame, standoff_m: float = 0.05) -> FigureSpec | None:
    fitted = df[df["fitted"]]
    har = 100.0 * fitted["height_above_roof_m"].dropna().to_numpy(dtype=float)
    pra = fitted["angle_to_roof_deg"].dropna().to_numpy(dtype=float)
    if len(har) < 10:
        return None
    with matplotlib.rc_context(RC):
        fig = _figure(FULL_WIDTH_MM * 0.72, 52)
        axes = fig.subplots(1, 2)
        ax = axes[0]
        edges = np.arange(-20.0, 62.5, 2.5)
        hist, _ = np.histogram(np.clip(har, edges[0], edges[-1] - 1e-9), bins=edges)
        share = 100.0 * hist / len(har)
        below = edges[:-1] < standoff_m * 100.0
        ax.bar(edges[:-1][below], share[below], width=2.5, align="edge", color=CONTEXT_LINE,
               edgecolor=SURFACE, linewidth=0.3, label="Not resolved from roof")
        ax.bar(edges[:-1][~below], share[~below], width=2.5, align="edge", color=BLUE,
               edgecolor=SURFACE, linewidth=0.3, label="Panel above roof")
        ax.axvline(standoff_m * 100.0, color=INK_2, linewidth=0.6)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_xlabel("Height of fitted plane above roof plane (cm)")
        ax.set_ylabel("Share of polygons with a roof reference (%)")
        ax.legend(loc="upper right", handlelength=1.2)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_title("a  Standoff above roof")
        _hist_panel(axes[1], pra, np.linspace(0, 40, 41),
                    "Angle between fitted plane and roof plane (°)", "{:.1f}°")
        axes[1].set_title("b  Angle to roof")
    return FigureSpec(
        "roof_relation", "Array against roof",
        f"Fitted plane relative to the surrounding roof plane, for polygons with a usable "
        f"roof reference (n = {len(har):,}). (a) Vertical offset; below the "
        f"{standoff_m * 100:.0f} cm line the two fits cannot be told apart, which is what a "
        f"flush-mounted array and a roof with no array both look like. (b) Angle between "
        f"the planes; near zero means parallel to the roof.", fig)


def fig_spatial(df: pd.DataFrame) -> FigureSpec | None:
    if "cx" not in df.columns or len(df) < 20:
        return None
    x = (df["cx"].to_numpy(dtype=float) - df["cx"].min()) / 1000.0
    y = (df["cy"].to_numpy(dtype=float) - df["cy"].min()) / 1000.0
    span_x, span_y = float(x.max()), float(y.max())
    if span_x <= 0 or span_y <= 0:
        return None
    gridsize = int(np.clip(np.sqrt(len(df)) / 1.5, 12, 60))
    aspect = span_y / span_x
    height = float(np.clip(FULL_WIDTH_MM * 0.42 * aspect + 14, 45, 120))
    fitted = df["fitted"].to_numpy()
    with matplotlib.rc_context(RC):
        fig = _figure(FULL_WIDTH_MM, height)
        axes = fig.subplots(1, 2)
        hb = axes[0].hexbin(x, y, gridsize=gridsize, cmap=SEQUENTIAL, mincnt=1,
                            linewidths=0.1, edgecolors=SURFACE)
        cb = fig.colorbar(hb, cax=axes[0].inset_axes([1.03, 0.0, 0.035, 1.0]))
        cb.set_label("Polygons per cell")
        hb2 = axes[1].hexbin(x[fitted], y[fitted],
                             C=df["tilt_deg"].to_numpy(dtype=float)[fitted],
                             reduce_C_function=np.median, gridsize=gridsize, cmap=SEQUENTIAL,
                             mincnt=1, vmin=0, vmax=40, linewidths=0.1, edgecolors=SURFACE)
        cb2 = fig.colorbar(hb2, cax=axes[1].inset_axes([1.03, 0.0, 0.035, 1.0]),
                           extend="max")
        cb2.set_label("Median tilt (°)")
        for cbar in (cb, cb2):
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(length=0)
        for ax, t in zip(axes, ["a  Polygon density", "b  Median tilt"], strict=True):
            ax.set_aspect("equal")
            ax.set_xlabel("Easting (km)")
            ax.set_title(t)
        axes[0].set_ylabel("Northing (km)")
    return FigureSpec(
        "spatial_distribution", "Spatial distribution",
        "Measured polygons across the study area, in hexagonal cells. (a) Number of "
        "polygons. (b) Median fitted tilt. Axes are kilometres from the south-west corner "
        "of the data extent.", fig)


def all_figures(df: pd.DataFrame, headline: str, manifest: dict | None = None,
                weight: str = "area") -> list[FigureSpec]:
    standoff = float((manifest or {}).get("config", {}).get("heights", {})
                     .get("min_panel_standoff_m", 0.05))
    specs = [
        fig_overview(df, headline, weight),
        fig_tilt_distribution(df, headline, weight),
        fig_azimuth_rose(df, headline, weight),
        fig_tilt_azimuth(df, headline, weight),
        fig_geometry_basis(df),
        fig_vintage_timeline(df, manifest),
        fig_roof_relation(df, standoff),
        fig_fit_quality(df),
        fig_spatial(df),
    ]
    return [s for s in specs if s is not None]
