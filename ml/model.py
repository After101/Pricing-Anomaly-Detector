"""
Per-mode model bundle: StandardScaler + IsolationForest + ScoreCalibrator.

MODE 1 and MODE 2 use structurally different feature sets (product-history
statistics vs. category/brand statistics), so they are two separate
Isolation Forest models rather than one model with padded/fabricated
features for whichever mode doesn't apply to a given row. This avoids
ever having to invent placeholder historical values for a brand-new
product, which the project spec explicitly forbids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from . import config
from .scoring import ScoreCalibrator


@dataclass
class ModeModel:
    mode_name: str
    feature_names: list
    scaler: StandardScaler
    forest: IsolationForest
    calibrator: ScoreCalibrator

    def raw_score(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.forest.score_samples(X_scaled)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        raw = self.raw_score(X)
        return self.calibrator.transform_many(raw)


def fit_mode_model(mode_name: str, feature_names: list, X_train: np.ndarray) -> ModeModel:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    forest = IsolationForest(**config.ISOLATION_FOREST_PARAMS)
    forest.fit(X_scaled)

    train_raw_scores = forest.score_samples(X_scaled)
    calibrator = ScoreCalibrator.fit(train_raw_scores)

    return ModeModel(
        mode_name=mode_name,
        feature_names=feature_names,
        scaler=scaler,
        forest=forest,
        calibrator=calibrator,
    )
