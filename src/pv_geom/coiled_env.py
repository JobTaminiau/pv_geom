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

SOFTWARE_ENV = "pv-geom-2026-05"
REGION = "us-east-2"                               # matches asu-nsf-phoenix bucket

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
        "scikit-learn>=1.5",
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
        region=REGION,
    )
    return cluster


def install_pv_geom_on_workers(client, ref: str = "main") -> None:
    """Install pv_geom on every worker via ``client.run`` (works around
    Coiled's silent dropping of ``git+`` pip requirements)."""

    def _install(ref: str = ref) -> str:
        import subprocess
        import sys
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "install", "--quiet",
             f"git+https://github.com/JobTaminiau/pv_geom.git@{ref}"],
            stderr=subprocess.STDOUT,
        )
        return out.decode("utf-8", errors="replace")[-200:]

    client.run(_install)
