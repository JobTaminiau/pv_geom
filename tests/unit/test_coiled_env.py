"""Milestone 0.5b (H4): the cloud backend is configured, not hard-wired."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import pv_geom
from pv_geom import coiled_env as ce
from pv_geom.config import PVGeomConfig


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch) -> Path:
    """A git checkout with a pushed commit, standing in for the package's."""
    monkeypatch.setattr(ce, "_installed_from", dict)
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", "-q", str(remote))
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.org")
    _git(repo, "config", "user.name", "t")
    (repo / "f.txt").write_text("one")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-q", "-m", "one")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "origin", "HEAD:refs/heads/main")
    _git(repo, "fetch", "-q", "origin")
    return repo


def test_defaults_are_not_tied_to_one_project() -> None:
    c = PVGeomConfig().compute.coiled
    assert (c.name, c.software, c.region, c.package_source) == ("auto",) * 4


def test_workers_get_the_commit_the_client_runs(checkout: Path) -> None:
    source = ce.resolve_package_source(PVGeomConfig(), package_dir=checkout)
    head = _git(checkout, "rev-parse", "HEAD")
    assert source.startswith("git+") and source.endswith(f"@{head}")
    assert ce.expected_commit(source) == head


def test_uncommitted_changes_stop_a_cloud_run(checkout: Path) -> None:
    (checkout / "f.txt").write_text("edited")
    with pytest.raises(ce.ClusterSetupError, match="uncommitted") as err:
        ce.resolve_package_source(PVGeomConfig(), package_dir=checkout)
    assert "commit and push" in err.value.remedy


def test_an_unpushed_commit_stops_a_cloud_run(checkout: Path) -> None:
    (checkout / "f.txt").write_text("two")
    _git(checkout, "commit", "-q", "-am", "two")
    with pytest.raises(ce.ClusterSetupError, match="not on the remote"):
        ce.resolve_package_source(PVGeomConfig(), package_dir=checkout)


def test_explicit_source_and_pip_git_install_are_respected(monkeypatch, tmp_path: Path) -> None:
    cfg = PVGeomConfig()
    cfg.compute.coiled.package_source = "pv-geom==9.9"
    assert ce.resolve_package_source(cfg) == "pv-geom==9.9"

    monkeypatch.setattr(ce, "_installed_from", lambda: {
        "url": "git@github.com:someone/fork.git",
        "vcs_info": {"vcs": "git", "commit_id": "a" * 40}})
    assert ce.resolve_package_source(PVGeomConfig(), package_dir=tmp_path) == (
        "git+https://github.com/someone/fork.git@" + "a" * 40)


def test_ssh_remotes_become_https() -> None:
    assert ce._https_remote("git@github.com:o/r.git") == "https://github.com/o/r.git"
    assert ce._https_remote("ssh://git@host.org/o/r.git") == "https://host.org/o/r.git"
    assert ce._https_remote("https://github.com/o/r.git") == "https://github.com/o/r.git"


def test_software_environment_is_named_after_its_contents(monkeypatch) -> None:
    cfg = PVGeomConfig()
    name = ce.software_env_name(cfg)
    assert name.startswith("pv-geom-") and len(name) == len("pv-geom-") + 10
    monkeypatch.setitem(ce.CONDA_SPEC, "dependencies", [*ce.CONDA_SPEC["dependencies"], "x"])
    assert ce.software_env_name(cfg) != name            # new contents, new environment
    cfg.compute.coiled.software = "my-env"
    assert ce.software_env_name(cfg) == "my-env"


def test_cluster_name_and_region(monkeypatch) -> None:
    cfg = PVGeomConfig()
    assert ce.cluster_name(cfg) == "pv-geom"
    cfg.study.name = "Greater Phoenix, AZ (2024)"
    assert ce.cluster_name(cfg) == "pv-geom-greater-phoenix-az-2024"

    cfg.inputs.lidar_prefix = "s3://some-bucket/lidar/"
    monkeypatch.setattr(ce, "bucket_region", lambda uri: "eu-west-1" if "some-bucket" in uri
                        else None)
    assert ce.cluster_region(cfg) == "eu-west-1"
    cfg.compute.coiled.region = "us-west-2"
    assert ce.cluster_region(cfg) == "us-west-2"
    assert ce.bucket_region.__name__                     # (patched)


def test_bucket_region_only_asks_about_s3() -> None:
    assert ce.bucket_region(None) is None
    assert ce.bucket_region("C:/data/lidar") is None


def test_workers_must_match_the_client() -> None:
    src = "git+https://github.com/o/r.git@" + "b" * 40
    good = {"version": pv_geom.__version__, "commit": "b" * 40}
    ce.check_identities({"w1": good, "w2": good}, src)
    with pytest.raises(ce.ClusterSetupError, match="commit"):
        ce.check_identities({"w1": good, "w2": {**good, "commit": "c" * 40}}, src)
    with pytest.raises(ce.ClusterSetupError, match="version"):
        ce.check_identities({"w1": {"version": "0.0.1", "commit": "b" * 40}}, src)
    # A source that pins no commit is checked by version alone.
    ce.check_identities({"w1": {"version": pv_geom.__version__, "commit": None}}, "pv-geom==1")


def test_cluster_is_built_from_the_config(monkeypatch) -> None:
    import sys
    import types

    made = {}
    fake = types.SimpleNamespace(Cluster=lambda **kw: made.update(kw) or "cluster")
    monkeypatch.setitem(sys.modules, "coiled", fake)
    monkeypatch.setattr(ce, "bucket_region", lambda uri: None)
    cfg = PVGeomConfig()
    cfg.study.name = "demo"
    cfg.compute.coiled.n_workers = 3
    cfg.compute.coiled.worker_memory = "8GiB"
    assert ce.make_cluster(cfg) == "cluster"
    assert made["name"] == "pv-geom-demo" and made["n_workers"] == 3
    assert made["worker_memory"] == "8GiB" and made["worker_options"] == {"nthreads": 1}
    assert made["software"].startswith("pv-geom-") and "region" not in made
