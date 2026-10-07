"""Report layer: statistics, figures, and the end-to-end build."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from shapely import wkb
from shapely.geometry import box

from pv_geom.io.output import write_partition
from pv_geom.report import build_report, stats
from pv_geom.schema import OUTPUT_SCHEMA
from pv_geom.vintage import PANEL_CONFIRMED, SURFACE_UNRESOLVED, UNSCREENED

# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


def test_weighted_quantile_matches_unweighted_median() -> None:
    x = np.arange(1, 102, dtype=float)
    assert stats.weighted_quantile(x, np.ones_like(x), 0.5)[0] == pytest.approx(51.0)


def test_weighted_quantile_follows_the_weight() -> None:
    q = stats.weighted_quantile([10.0, 30.0], [1.0, 99.0], 0.5)[0]
    assert q > 25.0


def test_circular_mean_wraps_through_north() -> None:
    """350 and 10 average to north, not to south."""
    c = stats.circular_stats([350.0, 10.0])
    assert min(c["circular_mean_deg"], 360 - c["circular_mean_deg"]) < 1e-6
    assert c["resultant_length"] == pytest.approx(np.cos(np.radians(10)))


def test_circular_stats_of_opposed_directions_have_no_concentration() -> None:
    assert stats.circular_stats([90.0, 270.0])["resultant_length"] == pytest.approx(0, abs=1e-9)


def test_sector_index_centres_on_north() -> None:
    assert stats.sector_index([350.0, 10.0, 180.0, 269.0], 4).tolist() == [0, 0, 2, 3]
    assert stats.sector_index([11.0, 12.0], 16).tolist() == [0, 1]


# --------------------------------------------------------------------------- #
# A synthetic run on disk
# --------------------------------------------------------------------------- #


def _synthetic_output(out: Path, n: int = 400, *, with_dates: bool = True) -> None:
    """A fleet with a known shape: 60% south-facing at ~20 deg (panel
    confirmed), 30% west-facing at ~10 deg (surface unresolved), 10% no fit."""
    rng = np.random.default_rng(0)
    n_s, n_w = int(0.6 * n), int(0.3 * n)
    n_x = n - n_s - n_w
    tilt = np.concatenate([rng.normal(20, 1, n_s), rng.normal(10, 1, n_w), np.full(n_x, np.nan)])
    az = np.concatenate([rng.normal(180, 5, n_s), rng.normal(270, 5, n_w), np.full(n_x, np.nan)])
    basis = [PANEL_CONFIRMED] * n_s + [SURFACE_UNRESOLVED] * n_w + ["no_fit"] * n_x
    har = np.concatenate([np.full(n_s, 0.10), np.full(n_w, 0.0), np.full(n_x, np.nan)])
    xs = rng.uniform(0, 5000, n)
    ys = rng.uniform(0, 5000, n)
    cols: dict = {f.name: [None] * n for f in OUTPUT_SCHEMA}

    def _opt(a):
        return [None if np.isnan(v) else float(v) for v in a]

    cols.update({
        "polygon_id": [f"p{i}" for i in range(n)],
        "parent_polygon_id": [f"p{i}" for i in range(n)],
        "input_row": list(range(n)),
        "status": ["no_fit" if np.isnan(t) else "measured" for t in tilt],
        "geometry": [wkb.dumps(box(x, y, x + 5, y + 4)) for x, y in zip(xs, ys, strict=True)],
        "area_m2": [20.0] * n,
        "surface_area_m2": [None if np.isnan(t) else 20.0 / np.cos(np.radians(t)) for t in tilt],
        "aspect_ratio": [1.25] * n,
        "polygon_vintage": [date(2024, 4, 1) if with_dates else None] * n,
        "lidar_date": [date(2020, 11, 28) if with_dates else None] * n,
        "lidar_date_source": ["gps_time" if with_dates else None] * n,
        "vintage_gap_days": [1220 if with_dates else None] * n,
        "geometry_basis": basis,
        "n_points_panel": [200] * n,
        "n_inliers_panel": [190] * n,
        "point_density": [10.0] * n,
        "panel_tilt_deg": _opt(tilt),
        "panel_azimuth_deg": _opt(az),
        "panel_rmse_m": [None if np.isnan(t) else 0.02 for t in tilt],
        "panel_tilt_unc_deg": [None if np.isnan(t) else 0.1 for t in tilt],
        "panel_azimuth_unc_deg": [None if np.isnan(t) else 0.5 for t in tilt],
        "n_planes_detected": [0 if np.isnan(t) else 1 for t in tilt],
        "roof_ref_source": ["none" if np.isnan(h) else "footprint_ring" for h in har],
        "height_above_roof_m": _opt(har),
        "panel_roof_angle_deg": [None if np.isnan(h) else 1.0 for h in har],
        "on_building": [True] * n,
        "flags": [["poor_fit"] if np.isnan(t) else [] for t in tilt],
        "lidar_tile_ids": [["t1"]] * n,
        "pkg_version": ["0.2.0"] * n,
        "config_hash": ["h"] * n,
        "run_id": ["r"] * n,
        "partition_id": [0] * n,
    })
    out.mkdir(parents=True, exist_ok=True)
    write_partition(pa.table(cols, schema=OUTPUT_SCHEMA), out / "part-00000.parquet", "EPSG:6341")
    (out / "manifest.json").write_text(json.dumps({
        "pkg_version": "0.2.0", "run_id": "abcdef123456", "crs": "EPSG:6341",
        "config": {"heights": {"min_panel_standoff_m": 0.05}},
        "inputs": {"polygons": "p.parquet", "footprints": None},
        "counts": {"polygons": n + 50},
        "vintage": {},
    }), encoding="utf-8")


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    _synthetic_output(tmp_path / "out")
    return tmp_path / "out"


def test_report_writes_every_product(run_dir: Path) -> None:
    r = build_report(run_dir, area_name="Testville")
    assert r.html.exists() and r.markdown.exists() and r.methods.exists()
    for name in ("coverage", "geometry_basis", "summary_statistics", "tilt_profile",
                 "azimuth_profile_16", "azimuth_profile_8", "tilt_azimuth_joint",
                 "roof_relation", "fit_quality", "flags"):
        assert (r.tables_dir / f"{name}.csv").stat().st_size > 0
    for name in ("geometry_overview", "tilt_distribution", "azimuth_rose",
                 "tilt_azimuth_joint", "geometry_basis", "vintage_timeline",
                 "roof_relation", "fit_quality", "spatial_distribution"):
        for ext in ("png", "pdf", "svg"):
            assert (r.figures_dir / f"{name}.{ext}").stat().st_size > 0
    # SVG text stays text, so a figure can be relabelled without redrawing it.
    assert "<text" in (r.figures_dir / "tilt_distribution.svg").read_text(encoding="utf-8")
    html = r.html.read_text(encoding="utf-8")
    assert "Testville" in html and "data:image/png;base64," in html
    assert "3.3 years newer" in html


def test_report_leads_with_the_panel_basis(run_dir: Path) -> None:
    """The panel-basis stratum is south-facing at 20 deg; pooling in the
    unresolved west-facing 10 deg rows would pull both numbers."""
    s = build_report(run_dir, export_dataset=False).summary
    assert s["headline_stratum"] == "panel"
    assert s["headline"]["area"]["tilt_median_deg"] == pytest.approx(20.0, abs=0.5)
    assert s["headline"]["area"]["share_facing_S"] == pytest.approx(1.0)
    assert s["all_fitted"]["area"]["share_facing_S"] == pytest.approx(2 / 3, abs=0.02)
    assert s["all_fitted"]["area"]["tilt_median_deg"] < 20.0
    assert s["vintage_gap_days"] == 1220
    assert s["geometry_basis"][UNSCREENED]["n"] == 0


def test_profile_shares_sum_to_one(run_dir: Path) -> None:
    r = build_report(run_dir, export_dataset=False)
    for name in ("tilt_profile", "azimuth_profile_16", "tilt_azimuth_joint"):
        t = pd.read_csv(r.tables_dir / f"{name}.csv")
        sums = t[t["n"].groupby([t["stratum"], t["weight"]]).transform("sum") > 0] \
            .groupby(["stratum", "weight"])["share"].sum()
        assert np.allclose(sums, 1.0), name


def test_dataset_export_round_trips(run_dir: Path) -> None:
    r = build_report(run_dir)
    gdf = gpd.read_parquet(r.dataset_dir / "pv_geom.parquet")
    assert len(gdf) == 400 and gdf.crs.to_epsg() == 6341
    flat = pd.read_csv(r.dataset_dir / "pv_geom.csv")
    assert {"centroid_lon", "centroid_lat", "geometry_basis"} <= set(flat.columns)
    assert "geometry" not in flat.columns
    dictionary = pd.read_csv(r.dataset_dir / "data_dictionary.csv")
    assert set(flat.columns) <= set(dictionary["column"])


def test_report_on_an_output_without_dates_takes_them_as_arguments(tmp_path: Path) -> None:
    """Outputs written before 0.2.0 carry no vintage; the report can be given it."""
    out = tmp_path / "old"
    _synthetic_output(out, with_dates=False)
    bare = build_report(out, tmp_path / "r1", export_dataset=False).summary
    assert bare["vintage_gap_days"] is None
    dated = build_report(out, tmp_path / "r2", export_dataset=False,
                         polygon_vintage="2024-04-01", lidar_date="2020-11-28").summary
    assert dated["vintage_gap_days"] == 1220
    assert dated["lidar_date_max"] == "2020-11-28"


def test_report_falls_back_to_all_fitted_when_few_panel_rows(tmp_path: Path) -> None:
    out = tmp_path / "small"
    _synthetic_output(out, n=40)                       # 24 panel-basis rows < 30
    s = build_report(out, export_dataset=False).summary
    assert s["headline_stratum"] == "all_fitted"


# --------------------------------------------------------------------------- #
# F7: bootstrap intervals;  G2: the self-describing dataset
# --------------------------------------------------------------------------- #


def test_bootstrap_interval_brackets_the_estimate_and_narrows_with_n() -> None:
    rng = np.random.default_rng(3)

    def interval(n: int) -> tuple[float, float, float, float]:
        tilt = rng.normal(20.0, 5.0, n)
        az = np.where(rng.random(n) < 0.6, 180.0, 270.0)
        ci = stats.bootstrap_intervals(tilt, az, np.ones(n))
        return (ci["tilt_p50_deg_ci_lo"], ci["tilt_p50_deg_ci_hi"],
                ci["share_facing_S_ci_lo"], ci["share_facing_S_ci_hi"])

    lo, hi, s_lo, s_hi = interval(400)
    assert lo < 20.0 < hi and s_lo < 0.6 < s_hi
    lo2, hi2, _, _ = interval(40_000)
    assert (hi2 - lo2) < (hi - lo) / 5            # ~ 1/sqrt(100)


def test_bootstrap_interval_is_reproducible_and_absent_for_tiny_samples() -> None:
    tilt = np.linspace(5, 35, 200)
    az = np.full(200, 180.0)
    a = stats.bootstrap_intervals(tilt, az, np.ones(200))
    b = stats.bootstrap_intervals(tilt, az, np.ones(200))
    assert a == b
    assert a["share_facing_S_ci_lo"] == a["share_facing_S_ci_hi"] == 1.0
    tiny = stats.bootstrap_intervals(tilt[:5], az[:5], np.ones(5))
    assert all(np.isnan(v) for v in tiny.values())


def test_summary_table_carries_intervals(run_dir: Path) -> None:
    r = build_report(run_dir, export_dataset=False)
    summ = pd.read_csv(r.tables_dir / "summary_statistics.csv")
    row = summ[(summ["stratum"] == "all_fitted") & (summ["weight"] == "count")].iloc[0]
    assert row["tilt_p50_deg_ci_lo"] <= row["tilt_p50_deg"] <= row["tilt_p50_deg_ci_hi"]
    assert row["share_facing_S_ci_lo"] < row["share_facing_S"] < row["share_facing_S_ci_hi"]
    assert "95% bootstrap intervals" in r.markdown.read_text(encoding="utf-8")


def test_release_dataset_describes_and_verifies_itself(run_dir: Path) -> None:
    import hashlib

    r = build_report(run_dir)
    d = r.dataset_dir
    for name in ("README.md", "metadata.json", "SHA256SUMS.txt", "status_definitions.csv"):
        assert (d / name).exists(), name

    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    assert meta["rows"] == 400 and meta["rows_by_status"]["measured"] == 360
    assert meta["creators"] == [] and meta["license"] is None      # left for a person
    assert len(meta["spatial"]["bbox_wgs84"]) == 4

    readme = (d / "README.md").read_text(encoding="utf-8")
    assert "one per input polygon" in readme and "`panel_tilt_deg`" in readme

    raw = (d / "SHA256SUMS.txt").read_bytes()
    assert b"\r" not in raw                       # sha256sum -c needs LF line endings
    listed = dict(line.split("  ")[::-1] for line in raw.decode().splitlines())
    assert set(listed) == {p.name for p in d.iterdir()} - {"SHA256SUMS.txt"}
    for name, digest in listed.items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == digest, name
