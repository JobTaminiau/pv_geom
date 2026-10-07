"""Coiled software environment + cluster shape constants for pv_geom.

Kept as module-level constants so the environment a given code version was
developed against travels with the code (mirrors the pattern used in
``pv_prediction_pipeline.coiled_env``). Promotion to a ``coiled:`` config
section can happen once the env stabilises across runs.

Provisioning a fresh env is a one-shot:

    python -c "from pv_geom.coiled_env import ensure_software_env; ensure_software_env()"

After that, ``make_cluster`` will spin up workers using the named env.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pv_geom.config import PVGeomConfig

if TYPE_CHECKING:                                  # pragma: no cover
    from coiled import Cluster

# --- Software environment + region ---------------------------------------- #

SOFTWARE_ENV = "pv-geom-2026-05"                   # default; cfg.compute.coiled.software wins

# Conda specification (built via coiled.create_software_environment).
# We keep it pinned-light: major versions only, so the env survives minor
# upstream churn between rebuilds.
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


def ensure_software_env(rebuild: bool = False) -> str:
    """Create the Coiled software environment if missing. Returns the env name."""
    import coiled

    # ``list_software_environments`` returns a ``dict[name -> metadata]`` in
    # coiled>=1.x; iterating it yields env names directly.
    existing = set(coiled.list_software_environments())
    if SOFTWARE_ENV in existing and not rebuild:
        return SOFTWARE_ENV
    coiled.create_software_environment(name=SOFTWARE_ENV, conda=CONDA_SPEC)
    return SOFTWARE_ENV


def make_cluster(cfg: PVGeomConfig) -> "Cluster":
    """Spin up a Coiled cluster from ``cfg.compute.coiled``.

    The caller is responsible for using the cluster as a context manager (or
    closing it explicitly). On scheduler connect, ``pv_geom`` is installed on
    every worker via ``client.run`` because Coiled silently drops
    ``git+`` URLs from pip requirements (see ``coiled_git_url_requirements``
    note in user-memory).
    """
    import coiled

    c = cfg.compute.coiled
    cluster = coiled.Cluster(
        name=c.name,
        n_workers=c.n_workers,
        worker_memory=c.worker_memory,
        worker_cpu=c.worker_cpu,
        software=c.software or SOFTWARE_ENV,
        region=c.region,
        # One tile-group task per worker: a task's peak is the concatenated
        # multi-tile point cloud (up to ~13 GB for dense-metro fetch
        # neighborhoods), so even two concurrent tasks breach a 16 GiB
        # worker — observed as paired KilledWorker failures on the
        # 2026-07-29 full run once it reached the metro core. Memory, not
        # CPU, is the binding constraint.
        worker_options={"nthreads": 1},
    )
    return cluster


def install_pv_geom_on_workers(client, package_source: str) -> None:
    """Install pv_geom on the scheduler AND every worker (works around
    Coiled's silent dropping of ``git+`` pip requirements).

    Workers get a ``PipInstall`` worker plugin rather than a one-shot
    ``client.run``: the plugin runs on every CURRENT AND FUTURE worker,
    which matters on large clusters — with 40 staggered-boot workers, a
    one-shot install missed the late joiners, and a worker without
    ``pv_geom`` dies at task *deserialization* (protocol-layer
    ``ModuleNotFoundError`` → ``KilledWorker``, took down the 2026-07-29
    full-run attempt). The plugin also covers nanny-restarted workers
    mid-run.

    The scheduler also needs ``pv_geom`` because Dask deserializes the
    task graph on it before dispatching: any class/function reference
    in a delayed task imports ``pv_geom.*`` during ``pickle.loads``.
    Without this, ``client.compute`` blows up with
    ``ModuleNotFoundError: No module named 'pv_geom'`` from
    ``scheduler.update_graph``.
    """
    from distributed import PipInstall

    def _install(source: str = package_source) -> str:
        import subprocess
        import sys
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "install", "--quiet", source],
            stderr=subprocess.STDOUT,
        )
        return out.decode("utf-8", errors="replace")[-200:]

    client.run_on_scheduler(_install)
    client.register_plugin(
        PipInstall(
            packages=[package_source],
            pip_options=["--quiet"],
        )
    )
