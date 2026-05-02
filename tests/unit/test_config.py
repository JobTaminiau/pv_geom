"""Config validation: round-trip the YAML files we ship and check hash stability."""

from pathlib import Path

import pytest

from pv_geom.config import PVGeomConfig

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@pytest.mark.parametrize("name", ["default.yaml", "phoenix.yaml", "coiled.yaml"])
def test_shipped_config_loads(name: str) -> None:
    PVGeomConfig.from_yaml(CONFIGS / name)


def test_default_values() -> None:
    cfg = PVGeomConfig.from_yaml(CONFIGS / "default.yaml")
    assert cfg.panel_plane.ransac_threshold_m == 0.05
    assert cfg.panel_plane.tilt_floor_deg == 1.0
    assert cfg.crs.target == "EPSG:6341"
    assert cfg.mounting_rules.R1.panel_roof_angle_deg_max == 5.0
    # Spike-derived overrides
    assert cfg.panel_plane.min_density_pts_per_m2 == 3
    assert cfg.panel_plane.min_points == 30
    assert cfg.io.classification.panel_class_primary == 6
    assert cfg.io.classification.panel_class_fallback == 1
    assert cfg.io.classification.ground_class == 2
    assert cfg.io.classification.fallback_height_above_ground_m == 1.5


def test_phoenix_overrides() -> None:
    cfg = PVGeomConfig.from_yaml(CONFIGS / "phoenix.yaml")
    assert cfg.crs.target == "EPSG:6341"
    assert cfg.compute.backend == "coiled"
    assert cfg.compute.coiled.name == "pv-geom-phoenix"


def test_config_hash_stable() -> None:
    cfg1 = PVGeomConfig.from_yaml(CONFIGS / "default.yaml")
    cfg2 = PVGeomConfig.from_yaml(CONFIGS / "default.yaml")
    h = cfg1.hash()
    assert h == cfg2.hash()
    assert len(h) == 64


def test_config_hash_changes_with_value() -> None:
    cfg = PVGeomConfig.from_yaml(CONFIGS / "default.yaml")
    h1 = cfg.hash()
    cfg.panel_plane.ransac_threshold_m = 0.10
    assert cfg.hash() != h1
