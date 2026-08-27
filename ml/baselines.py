"""
Simple statistical baselines, used to check whether Isolation Forest
actually adds value over a one-line formula.

BASELINE 1 (applies in both modes): standardized distance of current price
from the relevant subcategory median, expressed on a 0-1 scale by treating
a 3-standard-deviation gap as "maximally anomalous" (a conventional,
disclosed choice, not a fitted parameter).

BASELINE 2 (Mode 1 only): standardized distance of current price from the
PRODUCT'S OWN historical mean, on the same 0-1 scale.
"""

from __future__ import annotations

import numpy as np


def baseline_subcategory_zscore(current_price: float, subcat_median: float, subcat_std: float) -> float:
    if subcat_std is None or subcat_std <= 0:
        return 0.0
    z = abs(current_price - subcat_median) / subcat_std
    return float(np.clip(z / 3.0, 0.0, 1.0))


def baseline_own_history_zscore(current_price: float, hist_mean: float, hist_std: float) -> float:
    if hist_std is None or hist_std <= 0:
        return 0.0
    z = abs(current_price - hist_mean) / hist_std
    return float(np.clip(z / 3.0, 0.0, 1.0))
