"""Running on Coiled: what the cluster looks like and what its workers run.

Everything here comes from ``compute.coiled`` in the config. Each setting can
be given outright or left on ``auto``, in which case it is worked out:

``package_source``
    What workers ``pip install`` to get pv-geom. ``auto`` resolves to the
    *exact* code the client is running: the commit of the checkout (or of the
    git install) it was imported from. After installing, every worker is asked
    what it has, and the run stops if any answers differently.
``region``
    Where the cluster runs. ``auto`` asks the LiDAR bucket where it lives, so
    tiles are read in-region.
``software``
    The Coiled software environment. ``auto`` names it after the dependency
    list below, so changing that list makes a new environment instead of
    silently reusing a stale one.
``name``
    The cluster's name. ``auto`` derives it from ``study.name``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pv_geom import __version__
from pv_geom.config import PVGeomConfig
from pv_geom.errors import PVGeomError

if TYPE_CHECKING:                                  # pragma: no cover
    from coiled import Cluster

log = logging.getLogger(__name__)

DIST_NAME = "pv-geom"

# Conda specification for the worker environment: pv-geom's dependencies, by
# major version so the environment survives minor upstream churn. pv-geom
# itself is installed at cluster start (see install_pv_geom_on_workers).
CONDA_SPEC: dict = {
    "channels": ["conda-forge"],
    "dependencies": [
        "python=3.11",
        "numpy>=1.26",
        "pandas>=2.2",
        "geopandas>=0.14",
        "shapely>=2.0",
        "pyproj>=3.6",
        "pyogrio>=0.9",
        "pyarrow>=17",
        "fsspec>=2024.10",
        "s3fs>=2024.10",
        "boto3>=1.35",
        "dask>=2025.10",
        "distributed>=2025.10",
        "coiled>=1.128",
        "pdal>=2.6",
        "python-pdal>=3.4",
        "laspy>=2.5",
        "lazrs-python",
        "pydantic>=2.6",
        "pyyaml>=6",
        "typer>=0.12",
        "rich>=13.7",
    ],
}


class ClusterSetupError(PVGeomError, RuntimeError):
    """The cluster cannot be set up to run the client's code."""


def _auto(value: str | None) -> bool:
    return value is None or str(value).strip().lower() == "auto"


# --------------------------------------------------------------------------- #
# software environment, name, region
# --------------------------------------------------------------------------- #

def software_env_name(cfg: PVGeomConfig) -> str:
    """The software environment to use: the configured one, or one named after
    the dependency list (``pv-geom-<hash>``)."""
    configured = cfg.compute.coiled.software
    if not _auto(configured):
        return str(configured)
    digest = hashlib.sha256(json.dumps(CONDA_SPEC, sort_keys=True).encode()).hexdigest()[:10]
    return f"pv-geom-{digest}"


def ensure_software_env(cfg: PVGeomConfig | None = None, rebuild: bool = False) -> str:
    """Create the Coiled software environment if it does not exist yet. An
    explicitly named environment that is missing is built from ``CONDA_SPEC``."""
    import coiled

    name = software_env_name(cfg or PVGeomConfig())
    if name in set(coiled.list_software_environments()) and not rebuild:
        return name
    log.info("building Coiled software environment %s", name)
    coiled.create_software_environment(name=name, conda=CONDA_SPEC)
    return name


def cluster_name(cfg: PVGeomConfig) -> str:
    configured = cfg.compute.coiled.name
    if not _auto(configured):
        return str(configured)
    slug = re.sub(r"[^a-z0-9]+", "-", (cfg.study.name or "").lower()).strip("-")[:40]
    return f"pv-geom-{slug}" if slug else "pv-geom"


def bucket_region(uri: str | None) -> str | None:
    """The AWS region of an ``s3://`` bucket, or None if it cannot be told."""
    if not uri or not str(uri).startswith("s3://"):
        return None
    bucket = str(uri)[5:].split("/", 1)[0]
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        return None
    try:
        head = boto3.client("s3").head_bucket(Bucket=bucket)
        return head["ResponseMetadata"]["HTTPHeaders"].get("x-amz-bucket-region")
    except ClientError as exc:
        # A refused HEAD still names the bucket's region in its headers.
        headers = exc.response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
        return headers.get("x-amz-bucket-region")
    except Exception:
        return None


def cluster_region(cfg: PVGeomConfig) -> str | None:
    """The region to run in: configured, else the LiDAR bucket's, else Coiled's
    own default (None)."""
    configured = cfg.compute.coiled.region
    if not _auto(configured):
        return str(configured)
    region = bucket_region(cfg.inputs.lidar_prefix)
    if region:
        log.info("cluster region %s (where the LiDAR bucket is)", region)
    else:
        log.info("cluster region left to Coiled's default: the LiDAR is not in an S3 bucket "
                 "whose region could be read; set compute.coiled.region to choose")
    return region


# --------------------------------------------------------------------------- #
# what the workers install
# --------------------------------------------------------------------------- #

def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                             timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _https_remote(url: str) -> str:
    """A remote URL in the form pip can install from without SSH keys."""
    url = url.strip()
    m = re.match(r"^(?:ssh://)?git@([^:/]+)[:/](.+)$", url)
    if m:
        url = f"https://{m.group(1)}/{m.group(2)}"
    return url


def _installed_from() -> dict[str, Any]:
    """PEP 610 record of how the running pv-geom was installed ({} if none)."""
    try:
        text = metadata.distribution(DIST_NAME).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return {}
    return json.loads(text) if text else {}


def installed_commit() -> str | None:
    """The git commit the running pv-geom was pip-installed from, if any."""
    return (_installed_from().get("vcs_info") or {}).get("commit_id")


def resolve_package_source(cfg: PVGeomConfig, package_dir: Path | None = None) -> str:
    """What workers install so that they run the client's code.

    An explicit ``compute.coiled.package_source`` is used as given. ``auto``
    resolves, in order: the commit pv-geom was pip-installed from; the commit
    of the git checkout it is imported from (which must be pushed, because
    workers fetch it from the remote); the released version on the package
    index.
    """
    configured = cfg.compute.coiled.package_source
    if not _auto(configured):
        return str(configured)

    record = _installed_from()
    vcs = record.get("vcs_info") or {}
    if vcs.get("vcs") == "git" and vcs.get("commit_id") and record.get("url"):
        return f"git+{_https_remote(record['url'])}@{vcs['commit_id']}"

    here = package_dir or Path(__file__).resolve().parent
    top = _git(["rev-parse", "--show-toplevel"], here)
    if top:
        root = Path(top)
        commit = _git(["rev-parse", "HEAD"], root)
        remote = _git(["remote", "get-url", "origin"], root)
        if not commit or not remote:
            raise ClusterSetupError(
                "pv-geom is running from a git checkout with no commit or no 'origin' remote, "
                "so there is nothing for cloud workers to install",
                "commit and push, or set compute.coiled.package_source to something pip "
                "can install")
        if _git(["status", "--porcelain", "--untracked-files=no"], root):
            raise ClusterSetupError(
                "the pv-geom checkout has uncommitted changes; cloud workers would run the "
                f"last commit ({commit[:8]}), not the code on this machine",
                "commit and push the changes (or stash them), then run again")
        if not _git(["branch", "-r", "--contains", commit], root):
            raise ClusterSetupError(
                f"commit {commit[:8]} of the pv-geom checkout is not on the remote, so cloud "
                "workers cannot fetch it",
                "push the branch, then run again")
        return f"git+{_https_remote(remote)}@{commit}"

    if "dev" not in __version__ and "+" not in __version__:
        return f"{DIST_NAME}=={__version__}"
    raise ClusterSetupError(
        f"cannot tell where pv-geom {__version__} came from, so cloud workers cannot be "
        "given the same code",
        "set compute.coiled.package_source (e.g. git+https://...@<commit>)")


def expected_commit(package_source: str) -> str | None:
    """The commit a ``git+...@<sha>`` source pins, if it pins one."""
    m = re.search(r"@([0-9a-f]{7,40})$", package_source)
    return m.group(1) if m else None


def _worker_identity() -> dict[str, Any]:
    """Runs on a worker: which pv-geom it has."""
    import pv_geom
    from pv_geom.coiled_env import installed_commit

    return {"version": pv_geom.__version__, "commit": installed_commit()}


def check_identities(identities: dict[str, dict[str, Any]], package_source: str) -> None:
    """Stop unless every worker runs the client's version (and commit, when the
    source pins one)."""
    commit = expected_commit(package_source)
    wrong = []
    for worker, ident in identities.items():
        if ident.get("version") != __version__:
            wrong.append(f"{worker}: version {ident.get('version')} (client {__version__})")
        elif commit and not str(ident.get("commit") or "").startswith(commit[:7]):
            wrong.append(f"{worker}: commit {str(ident.get('commit'))[:8]} (expected "
                         f"{commit[:8]})")
    if wrong:
        raise ClusterSetupError(
            "cloud workers are not running the client's pv-geom: " + "; ".join(wrong[:3])
            + (f" (+{len(wrong) - 3} more)" if len(wrong) > 3 else ""),
            "rebuild the software environment or fix compute.coiled.package_source")


# --------------------------------------------------------------------------- #
# cluster
# --------------------------------------------------------------------------- #

def make_cluster(cfg: PVGeomConfig) -> Cluster:
    """Start a Coiled cluster shaped by ``cfg.compute.coiled``.

    The caller closes it (``cluster_for`` does). pv-geom is installed on the
    workers once the client connects: Coiled drops ``git+`` requirements from
    environment specifications.
    """
    import coiled

    c = cfg.compute.coiled
    kwargs: dict[str, Any] = {
        "name": cluster_name(cfg),
        "n_workers": c.n_workers,
        "worker_memory": c.worker_memory,
        "worker_cpu": c.worker_cpu,
        "software": software_env_name(cfg),
        # One tile-group task per worker. A task's memory is what binds, not
        # its CPU: two concurrent dense-metro tasks on a 16 GiB worker were
        # killed in pairs on the 2026-07-29 run. compute.memory_budget_gb now
        # bounds each task; one thread keeps the bound per worker.
        "worker_options": {"nthreads": c.worker_threads},
    }
    region = cluster_region(cfg)
    if region:
        kwargs["region"] = region
    return coiled.Cluster(**kwargs)


def install_pv_geom_on_workers(client: Any, package_source: str) -> None:
    """Install pv-geom on the scheduler and on every worker, present and
    future, then confirm that each has the client's version.

    Workers get a ``PipInstall`` plugin rather than a one-off ``client.run``:
    the plugin also runs on workers that join late or are restarted, and a
    worker without pv-geom dies at task deserialisation. The scheduler needs it
    too, because it unpickles the task graph before dispatching.
    """
    from distributed import PipInstall

    def _install(source: str = package_source) -> str:
        import subprocess
        import sys
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "install", "--quiet", "--force-reinstall",
             "--no-deps", source],
            stderr=subprocess.STDOUT,
        )
        return out.decode("utf-8", errors="replace")[-200:]

    log.info("workers install %s", package_source)
    client.run_on_scheduler(_install)
    client.register_plugin(
        PipInstall(packages=[package_source],
                   pip_options=["--quiet", "--force-reinstall", "--no-deps"]))
    check_identities(client.run(_worker_identity), package_source)
