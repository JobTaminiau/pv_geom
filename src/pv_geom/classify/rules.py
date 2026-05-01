"""Rules-based mounting classifier (PRD §7.5). M5."""

from __future__ import annotations

from pv_geom.classify.interface import MountingClassifier, MountingFeatures, MountingResult
from pv_geom.config import MountingRulesConfig


class RulesMountingClassifier(MountingClassifier):
    def __init__(self, cfg: MountingRulesConfig) -> None:
        self.cfg = cfg

    def classify(self, features: MountingFeatures) -> MountingResult:
        raise NotImplementedError("M5")


def classify_mounting(features: MountingFeatures, cfg: MountingRulesConfig) -> MountingResult:
    return RulesMountingClassifier(cfg).classify(features)
