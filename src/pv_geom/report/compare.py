"""Two or more runs side by side: two study areas, or one area at two versions.

Each run is described by the same stratum, so the comparison is like with
like: the panel basis when every run has enough of it, otherwise all fitted
polygons.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from pv_geom.errors import InputError
from pv_geom.report import figures as figs
from pv_geom.report import stats
from pv_geom.report.data import load_run

MAX_RUNS = 4
# Series colours in fixed order; a third and fourth run are told apart by dash
# as well as colour.
_STYLES = [(figs.BLUE, "-"), (figs.ORANGE, "-"), (figs.INK, "--"), (figs.MUTED, ":")]


@dataclass(frozen=True)
class CompareResult:
    out_dir: Path
    markdown: Path
    table: pd.DataFrame
    tables_dir: Path
    figures_dir: Path
    stratum: str


def _label(output: str | Path, manifest: dict) -> str:
    name = (manifest.get("config", {}).get("study") or {}).get("name")
    return str(name) if name else Path(str(output).rstrip("/\\")).name


def comparison_table(runs: list[tuple[str, pd.DataFrame, dict]], stratum: str,
                     weight: str) -> pd.DataFrame:
    rows = []
    for label, df, manifest in runs:
        summary = stats.summary_statistics(df)
        s = summary[(summary["stratum"] == stratum) & (summary["weight"] == weight)].iloc[0]
        basis = df["geometry_basis"].value_counts(normalize=True)
        fitted = df[df["fitted"]]
        row = {
            "run": label, "pkg_version": manifest.get("pkg_version"),
            "polygon_vintage": (manifest.get("vintage") or {}).get("polygon_vintage"),
            "lidar_date": (manifest.get("vintage") or {}).get("rows_lidar_date_max"),
            "n_polygons": len(df), "share_measured": float(df["fitted"].mean()),
            "share_panel_basis": float(stats.stratum_mask(df, "panel").mean()),
            "share_unscreened": float(basis.get("unscreened", 0.0)),
            "share_multi_facet": (float((fitted["n_facets"] > 1).mean())
                                  if len(fitted) else np.nan),
            "stratum": stratum, "weight": weight, "n": int(s["n"]),
            "area_m2": float(s["area_m2"]),
        }
        for col in ("tilt_mean_deg", "tilt_p05_deg", "tilt_p25_deg", "tilt_p50_deg",
                    "tilt_p75_deg", "tilt_p95_deg", "azimuth_circular_mean_deg",
                    *[f"share_facing_{q}" for q in stats.QUADRANTS]):
            row[col] = float(s[col]) if col in s.index else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _profiles(runs: list[tuple[str, pd.DataFrame, dict]], stratum: str, weight: str
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    tilt: list[pd.DataFrame] = []
    az: list[pd.DataFrame] = []
    for label, df, _ in runs:
        for fn, sink in ((stats.tilt_profile, tilt), (stats.azimuth_profile, az)):
            t = fn(df)
            t = t[(t["stratum"] == stratum) & (t["weight"] == weight)].copy()
            t.insert(0, "run", label)
            sink.append(t)
    return pd.concat(tilt, ignore_index=True), pd.concat(az, ignore_index=True)


def fig_comparison(tilt: pd.DataFrame, az: pd.DataFrame, labels: list[str], stratum: str,
                   weight: str) -> figs.FigureSpec:
    with matplotlib.rc_context(figs.RC):
        fig = figs._figure(figs.FULL_WIDTH_MM, 70)
        ax_t, ax_a = fig.subplots(1, 2)
        for label, (colour, dash) in zip(labels, _STYLES, strict=False):
            t = tilt[tilt["run"] == label]
            edges = np.append(t["tilt_lo_deg"].to_numpy(), t["tilt_lo_deg"].iloc[-1]
                              + (t["tilt_lo_deg"].iloc[1] - t["tilt_lo_deg"].iloc[0]))
            ax_t.stairs(100.0 * t["share"].to_numpy(), edges, color=colour, linestyle=dash,
                        linewidth=1.2, label=label)
            a = az[az["run"] == label]
            x = np.arange(len(a) + 1)
            share = 100.0 * a["share"].to_numpy()
            ax_a.plot(x, np.append(share, share[0]), color=colour, linestyle=dash,
                      linewidth=1.2, marker="o", markersize=2.5, label=label)
        sectors = az[az["run"] == labels[0]]["sector"].tolist()
        ax_a.set_xticks(np.arange(len(sectors) + 1)[::2])
        ax_a.set_xticklabels([*sectors, sectors[0]][::2])
        ax_a.set_xlabel("Direction the array faces (true north)")
        ax_t.set_xlabel("Tilt from horizontal (°)")
        unit = "array area" if weight == "area" else "polygons"
        for ax, title in ((ax_t, "a  Tilt profile"), (ax_a, "b  Orientation profile")):
            ax.set_ylabel(f"Share of {unit} (%)")
            ax.set_ylim(bottom=0)
            ax.yaxis.grid(True)
            ax.set_axisbelow(True)
            ax.set_title(title)
        ax_t.legend(loc="upper right", handlelength=1.8, borderaxespad=0.2)
    group = stats.STRATA[stratum][0].lower()
    return figs.FigureSpec(
        "profile_comparison", "Tilt and orientation profiles compared",
        f"Tilt (a) and orientation (b) profiles of {' and '.join(labels)} ({group}), as "
        f"shares of {unit}. Orientation is in 16 compass sectors; the first is repeated at "
        "the right edge.", fig)


def compare_runs(outputs: list[str | Path], out_dir: str | Path, *,
                 labels: list[str] | None = None, headline: str | None = None,
                 weight: str = "area") -> CompareResult:
    """Write a comparison of several finished runs to ``out_dir``."""
    if not 2 <= len(outputs) <= MAX_RUNS:
        raise InputError(f"compare takes 2 to {MAX_RUNS} runs, got {len(outputs)}",
                         "pass the outputs to compare, e.g. `pv-geom report A B --compare`")
    if weight not in stats.WEIGHTS:
        raise InputError(f"unknown weight {weight!r}", f"use one of {list(stats.WEIGHTS)}")
    if labels is not None and len(labels) != len(outputs):
        raise InputError("one label per run is needed", "give as many --label as outputs")
    loaded = [load_run(o) for o in outputs]
    names = labels or [_label(o, m) for o, (_, _, m) in zip(outputs, loaded, strict=True)]
    if len(set(names)) != len(names):            # same study name twice: two versions
        names = [f"{n} ({m.get('pkg_version', i)})" if names.count(n) > 1 else n
                 for i, (n, (_, _, m)) in enumerate(zip(names, loaded, strict=True))]
    if len(set(names)) != len(names):
        names = [f"{n} [{i + 1}]" for i, n in enumerate(names)]
    runs = [(n, df, m) for n, (_, df, m) in zip(names, loaded, strict=True)]

    if headline is None:
        own = {stats.headline_stratum(df) for _, df, _ in runs}
        headline = "panel" if own == {"panel"} else "all_fitted"
    elif headline not in stats.STRATA:
        raise InputError(f"unknown headline stratum {headline!r}",
                         f"use one of {list(stats.STRATA)}")

    out = Path(out_dir)
    tables_dir, figures_dir = out / "tables", out / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    table = comparison_table(runs, headline, weight)
    tilt, az = _profiles(runs, headline, weight)
    table.to_csv(tables_dir / "comparison.csv", index=False)
    tilt.to_csv(tables_dir / "tilt_profile_comparison.csv", index=False)
    az.to_csv(tables_dir / "azimuth_profile_comparison.csv", index=False)
    spec = fig_comparison(tilt, az, names, headline, weight)
    figs.save(spec, figures_dir)

    def _p(v: float) -> str:
        return "–" if pd.isna(v) else f"{v:.0%}"

    def _d(v: float) -> str:
        return "–" if pd.isna(v) else f"{v:.1f}"

    shown = [
        ("Polygons", lambda r: f"{r.n_polygons:,}"),
        ("Measured", lambda r: _p(r.share_measured)),
        ("Panel basis", lambda r: _p(r.share_panel_basis)),
        ("Unscreened", lambda r: _p(r.share_unscreened)),
        ("More than one facet (of measured)", lambda r: _p(r.share_multi_facet)),
        ("Polygons in compared group", lambda r: f"{r.n:,}"),
        ("Median tilt (°)", lambda r: _d(r.tilt_p50_deg)),
        ("Tilt, middle half (°)", lambda r: f"{_d(r.tilt_p25_deg)} – {_d(r.tilt_p75_deg)}"),
        ("Mean tilt (°)", lambda r: _d(r.tilt_mean_deg)),
        *[(f"Facing {name}", lambda r, q=q: _p(getattr(r, f"share_facing_{q}")))
          for q, name in zip(stats.QUADRANTS, ("north", "east", "south", "west"), strict=True)],
        ("Polygon vintage", lambda r: str(r.polygon_vintage or "–")),
        ("LiDAR date (latest)", lambda r: str(r.lidar_date or "–")),
        ("pv-geom version", lambda r: str(r.pkg_version or "–")),
    ]
    recs = list(table.itertuples(index=False))
    group = stats.STRATA[headline][0].lower()
    unit = "array area" if weight == "area" else "polygon count"
    lines = [f"# Comparison: {' vs '.join(names)}", "",
             f"Tilt and facing describe the **{group}** of each run, weighted by {unit}.", "",
             "| | " + " | ".join(names) + " |", "| --- |" + " ---: |" * len(names)]
    lines += [f"| {name} | " + " | ".join(fn(r) for r in recs) + " |" for name, fn in shown]
    lines += ["", f"![{spec.title}](figures/{spec.name}.png)", "", f"*{spec.caption}*", "",
              "Full numbers: `tables/comparison.csv`, `tables/tilt_profile_comparison.csv`, "
              "`tables/azimuth_profile_comparison.csv`.", ""]
    md = out / "comparison.md"
    md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    (out / "comparison.json").write_text(
        json.dumps({"runs": names, "stratum": headline, "weight": weight,
                    "table": table.to_dict("records")}, indent=2, default=str),
        encoding="utf-8", newline="\n")
    return CompareResult(out, md, table, tables_dir, figures_dir, headline)
