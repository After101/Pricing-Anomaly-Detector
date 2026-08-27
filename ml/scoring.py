"""
Score calibration: turn Isolation Forest's raw output into a stable,
reproducible 0-1 "Pricing Anomaly Score".

WHY NOT MIN-MAX ON EACH REQUEST:
Isolation Forest's `score_samples` is not a probability, and its raw scale
has no fixed, meaningful bounds. Min-max scaling a single incoming request
against itself is meaningless (a single point has no range). The spec also
explicitly forbids per-request normalization, since it would make the same
raw score map to different anomaly scores on different days depending on
what else got submitted.

WHY NOT A PERCENTILE-RANK TRANSFORM EITHER (tried first, rejected):
An earlier version of this calibrator mapped a raw score to
"1 - (fraction of TRAINING scores <= this score)". That is deterministic
and reproducible, but it silently forces the calibrated-score distribution
over the training population to be perfectly UNIFORM on [0, 1] -- by
construction, the single MEDIAN training point always ends up with
anomaly_score exactly 0.5, and fully half of all (mostly ordinary) training
prices score above 0.5. Verified empirically: a price sitting exactly at
its subcategory's median (about as unremarkable as a price can be) was
calibrated to ~0.5 ("somewhat unusual"). That is a mis-calibration, not a
model problem -- Isolation Forest's raw scores for this data are NOT
uniformly spread; most points cluster tightly together as "normal" with a
comparatively small tail toward "abnormal", and the calibration should
preserve that shape rather than flattening it.

WHAT WE DO INSTEAD:
We fit a FIXED min-max scaler on the TRAINING rows' raw scores, once, at
training time, and save (train_min, train_max) as the calibration
artifact:

  1. raw_score = model.score_samples(x)              # lower = more anomalous
  2. anomaly_score = (train_max - raw_score) / (train_max - train_min)
  3. clip to [0, 1]

train_max corresponds to the LEAST anomalous point seen in training
(anomaly_score -> 0) and train_min to the MOST anomalous point seen in
training (anomaly_score -> 1). Because these two constants are fixed at
training time and reused unchanged for every subsequent prediction, this
is NOT a per-request normalization -- the same raw score always maps to
the same anomaly score, regardless of what else is being scored that day.
Scores for inference points more extreme than anything seen in training
are clipped to [0, 1] rather than extrapolated.
"""

from __future__ import annotations

import numpy as np


class ScoreCalibrator:
    def __init__(self, train_min: float, train_max: float):
        self.train_min = float(train_min)
        self.train_max = float(train_max)

    @classmethod
    def fit(cls, raw_scores: np.ndarray) -> "ScoreCalibrator":
        raw_scores = np.asarray(raw_scores, dtype=float)
        return cls(train_min=float(raw_scores.min()), train_max=float(raw_scores.max()))

    def transform(self, raw_score: float) -> float:
        span = self.train_max - self.train_min
        if span <= 0:
            return 0.0
        anomaly_score = (self.train_max - raw_score) / span
        return float(np.clip(anomaly_score, 0.0, 1.0))

    def transform_many(self, raw_scores: np.ndarray) -> np.ndarray:
        raw_scores = np.asarray(raw_scores, dtype=float)
        span = self.train_max - self.train_min
        if span <= 0:
            return np.zeros_like(raw_scores)
        anomaly = (self.train_max - raw_scores) / span
        return np.clip(anomaly, 0.0, 1.0)


def classify(anomaly_score: float) -> str:
    """Human-readable bucket label. Bucket edges are a documented,
    fixed convention -- not derived from any statistical test."""
    if anomaly_score < 0.50:
        return "Typical pricing"
    elif anomaly_score < 0.75:
        return "Somewhat unusual pricing"
    elif anomaly_score < 0.90:
        return "Unusual pricing"
    else:
        return "Highly unusual pricing"
