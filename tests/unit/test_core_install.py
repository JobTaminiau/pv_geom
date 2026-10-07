"""The core install: measuring and reporting need no cloud packages."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pv_geom.config import PVGeomConfig
from pv_geom.errors import MissingDependencyError

CLOUD_PACKAGES = ("boto3", "botocore", "s3fs", "coiled")


@pytest.fixture
def no_cloud_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the cloud packages unimportable, as on a core-only install."""
    for name in list(sys.modules):
        if name.split(".")[0] in CLOUD_PACKAGES:
            monkeypatch.delitem(sys.modules, name)
    for name in CLOUD_PACKAGES:
        monkeypatch.setitem(sys.modules, name, None)       # import -> ImportError


def test_local_run_and_report_need_no_cloud_packages(
    no_cloud_packages: None, synth_inputs: dict[str, Path]
) -> None:
    from pv_geom.pipeline.runner import run_pipeline
    from pv_geom.report import build_report

    run_pipeline(
        polygons_uri=str(synth_inputs["polygons"]),
        tile_index_uri=str(synth_inputs["tindex"]),
        lidar_prefix=str(synth_inputs["laz_dir"]),
        output_uri=str(synth_inputs["out"]),
        cfg=PVGeomConfig(), name_template="tile.laz", use_dask=False,
    )
    assert build_report(synth_inputs["out"], export_dataset=False).html.exists()


@pytest.mark.parametrize("call", ["read", "list", "sink", "vector"])
def test_s3_without_the_cloud_extra_says_how_to_get_it(no_cloud_packages: None, call: str) -> None:
    from pv_geom.io.storage import list_s3_uris, localize
    from pv_geom.io.vector import read_vector
    from pv_geom.pipeline.sink import OutputSink

    actions = {
        "read": lambda: localize("s3://bucket/tile.laz"),
        "list": lambda: list_s3_uris("s3://bucket/prefix"),
        "sink": lambda: OutputSink("s3://bucket/out"),
        "vector": lambda: read_vector("s3://bucket/polygons.parquet"),
    }
    with pytest.raises(MissingDependencyError) as err:
        actions[call]()
    assert 'pv-geom[cloud]' in err.value.remedy


def test_coiled_backend_without_the_extra_says_how_to_get_it(no_cloud_packages: None) -> None:
    from pv_geom.pipeline.executor import cluster_for

    cfg = PVGeomConfig()
    cfg.compute.backend = "coiled"
    with pytest.raises(MissingDependencyError) as err, cluster_for(cfg):
        pass
    assert 'pv-geom[coiled]' in err.value.remedy
