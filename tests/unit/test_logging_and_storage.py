"""Logging setup, the storage helpers and the shared vector reader."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from pv_geom.io.storage import is_remote, list_s3_uris, localize
from pv_geom.io.vector import read_vector, reproject
from pv_geom.utils.logging import ConsoleFormatter, JSONFormatter, add_file_log, configure_logging


@pytest.fixture
def clean_logger():
    pkg = logging.getLogger("pv_geom")
    saved = (pkg.handlers[:], pkg.level, pkg.propagate)
    yield pkg
    pkg.handlers[:], pkg.level, pkg.propagate = saved[0], saved[1], saved[2]


def _record(level: int, msg: str) -> logging.LogRecord:
    return logging.LogRecord("pv_geom.x", level, __file__, 1, msg, None, None)


def test_console_format_shows_level_only_for_problems() -> None:
    fmt = ConsoleFormatter()
    assert fmt.format(_record(logging.INFO, "3 groups")) == "[pv-geom] 3 groups"
    assert fmt.format(_record(logging.WARNING, "gap")) == "[pv-geom] WARNING: gap"


def test_json_format_is_one_object_per_record() -> None:
    payload = json.loads(JSONFormatter().format(_record(logging.ERROR, "boom")))
    assert payload["level"] == "ERROR" and payload["msg"] == "boom"
    assert payload["logger"] == "pv_geom.x"


@pytest.mark.parametrize("kwargs, level", [
    ({}, logging.INFO), ({"verbose": True}, logging.DEBUG), ({"quiet": True}, logging.WARNING),
])
def test_configure_logging_levels(clean_logger, kwargs, level) -> None:
    configure_logging(**kwargs)
    assert clean_logger.level == level
    assert len(clean_logger.handlers) == 1 and clean_logger.propagate is False


def test_file_log_records_json_lines(clean_logger, tmp_path: Path) -> None:
    clean_logger.setLevel(logging.INFO)
    handler = add_file_log(tmp_path / "logs" / "run.jsonl")
    logging.getLogger("pv_geom.pipeline.plan").info("planned %d groups", 4)
    clean_logger.removeHandler(handler)
    handler.close()
    line = json.loads((tmp_path / "logs" / "run.jsonl").read_text().strip())
    assert line["msg"] == "planned 4 groups"


def test_is_remote() -> None:
    assert is_remote("s3://bucket/key.laz") and is_remote("https://host/x.laz")
    assert not is_remote("C:/data/x.laz") and not is_remote(Path("x.laz"))


def test_localize_passes_local_paths_through(tmp_path: Path) -> None:
    f = tmp_path / "t.laz"
    assert localize(f) == f


def test_localize_refuses_schemes_it_cannot_fetch() -> None:
    with pytest.raises(NotImplementedError, match="only s3://"):
        localize("https://example.org/tile.laz")


def test_list_s3_uris_needs_an_s3_prefix() -> None:
    with pytest.raises(ValueError, match="expected an s3://"):
        list_s3_uris("/local/dir")


def test_read_vector_reads_every_format_the_same(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame({"k": [1, 2]}, geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)],
                           crs="EPSG:6341")
    paths = {"parquet": tmp_path / "a.parquet", "gpkg": tmp_path / "a.gpkg",
             "geojson": tmp_path / "a.geojson", "shp": tmp_path / "a.shp"}
    gdf.to_parquet(paths["parquet"])
    gdf.to_file(paths["gpkg"], driver="GPKG")
    gdf.to_crs("EPSG:4326").to_file(paths["geojson"], driver="GeoJSON")
    gdf.to_file(paths["shp"])
    for path in paths.values():
        got = reproject(read_vector(path), "EPSG:6341")
        assert len(got) == 2 and got.crs.to_epsg() == 6341
        assert got.geometry.area.sum() == pytest.approx(2.0, rel=1e-6)


def test_reproject_is_a_no_op_without_a_target() -> None:
    gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:6341")
    assert reproject(gdf, None) is gdf and reproject(gdf, "EPSG:6341") is gdf
