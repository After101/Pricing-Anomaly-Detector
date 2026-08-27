"""
Feature engineering for the two inference modes.

LEAKAGE RULE (enforced here):
For any row being scored at timestamp t for product P, MODE-1 historical
features may only be computed from P's OWN observations strictly BEFORE t.
Never from P's observations at or after t. This module builds training
rows by walking each product's price history in chronological order and,
at each point, computing features only from what came before.

Subcategory/brand statistics (category_stats.py) are fit once on the
training split as a whole (like a preprocessing scaler) and are constant
context features -- they are not re-derived per timestamp. This is
documented as a deliberate, disclosed modeling choice, not a leakage bug:
no product's Mode-1 score ever depends on that SAME product's own future
prices.

DESIGN NOTE ON FEATURE REDUNDANCY (important, found during validation):
An earlier version of this module fed the Isolation Forest a mix of RAW
absolute prices (current_price, hist_mean, hist_min, hist_max, subcat_
median/mean/std/p25/p75/p95) alongside RELATIVE features derived from the
same signal (percentile, z-score, pct-difference). This caused two
problems, confirmed by inspection of scored training rows:
  1. Absolute prices differ in scale by ~50x across subcategories
     (Earphones ~Rs.450 vs TWS up to ~Rs.27,000), so mixing raw price into
     a single model trained across all subcategories let scale
     differences dominate over genuine relative-pricing signal.
  2. Several features were near-perfectly collinear (current_price,
     pct-vs-median, and percentile are all monotonic functions of the same
     underlying number), which let Isolation Forest's random splits
     over-weight that one signal 2-3x while under-using the others.
  Net effect: a price sitting EXACTLY at the subcategory median (a very
  common, non-anomalous price point in this dataset) was sometimes scored
  as "unusual" purely as an artifact of that redundancy.

The feature sets below are RELATIVE ONLY (log price ratios, z-scores,
percentiles, coefficient of variation) so that: (a) every feature is on a
comparable, roughly symmetric scale regardless of which subcategory a
product belongs to, and (b) each feature captures a genuinely different
aspect of "how does this price compare" rather than restating the same
number three ways. Raw prices and category statistics are still computed
and returned for the EXPLANATION layer (human-readable reference values);
they are simply excluded from the numeric vector handed to the model.

MODE 1 model features (7):
    log_price_vs_hist_median      log(current_price / historical median)
    hist_zscore                   (current_price - hist_mean) / hist_std, clipped
    hist_percentile_frac          product's own-history percentile, 0-1
    hist_cv                       historical coefficient of variation (std/mean)
    hist_range_ratio              (hist_max - hist_min) / hist_mean
    n_hist_obs_log                log1p(number of prior observations)
    log_price_vs_subcat_median    log(current_price / subcategory median) -- category context

MODE 2 model features (6):
    log_price_vs_subcat_median    log(current_price / subcategory median)
    subcat_zscore                 (current_price - subcat_mean) / subcat_std, clipped
    subcat_percentile_frac        percentile within subcategory, 0-1
    subcat_iqr_position           (current_price - p25) / (p75 - p25)
    brand_available                0/1
    log_price_vs_brand_median     log(current_price / brand median); falls back to
                                   log_price_vs_subcat_median when brand stats unavailable
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .category_stats import percentile_of_score

MODE1_FEATURE_NAMES = [
    "log_price_vs_hist_median",
    "hist_zscore",
    "hist_percentile_frac",
    "hist_cv",
    "hist_range_ratio",
    "n_hist_obs_log",
    "log_price_vs_subcat_median",
]

MODE2_FEATURE_NAMES = [
    "log_price_vs_subcat_median",
    "subcat_zscore",
    "subcat_percentile_frac",
    "subcat_iqr_position",
    "brand_available",
    "log_price_vs_brand_median",
]

_Z_CLIP = 6.0


def _safe_div(numerator, denom):
    if denom is None or denom == 0 or (isinstance(denom, float) and np.isnan(denom)):
        return 0.0
    return float(numerator / denom)


def _log_ratio(price, reference):
    if reference is None or reference <= 0 or price <= 0:
        return 0.0
    return float(np.log(price / reference))


def mode1_features_from_prior(current_price: float, prior_prices: np.ndarray, subcat_stats: dict) -> dict:
    """Build MODE 1 features (+ human-readable extras). `prior_prices` must
    contain ONLY observations strictly before the point being scored
    (leakage-safe by construction -- see build_training_rows).
    """
    prior_prices = np.asarray(prior_prices, dtype=float)
    prior_prices = prior_prices[~np.isnan(prior_prices)]
    n = len(prior_prices)

    hist_mean = float(np.mean(prior_prices))
    hist_median = float(np.median(prior_prices))
    hist_std = float(np.std(prior_prices, ddof=1)) if n > 1 else 0.0
    hist_min = float(np.min(prior_prices))
    hist_max = float(np.max(prior_prices))
    hist_range = hist_max - hist_min
    hist_cv = _safe_div(hist_std, hist_mean)
    hist_range_ratio = _safe_div(hist_range, hist_mean)

    log_price_vs_hist_median = _log_ratio(current_price, hist_median)
    hist_zscore = float(np.clip(_safe_div(current_price - hist_mean, hist_std), -_Z_CLIP, _Z_CLIP)) if hist_std > 0 else 0.0
    hist_percentile = percentile_of_score(sorted(prior_prices.tolist()), current_price)
    n_hist_obs_log = float(np.log1p(n))

    subcat_median = subcat_stats["median"] if subcat_stats else None
    log_price_vs_subcat_median = _log_ratio(current_price, subcat_median) if subcat_median else 0.0

    return {
        # --- model features ---
        "log_price_vs_hist_median": log_price_vs_hist_median,
        "hist_zscore": hist_zscore,
        "hist_percentile_frac": hist_percentile / 100.0,
        "hist_cv": hist_cv,
        "hist_range_ratio": hist_range_ratio,
        "n_hist_obs_log": n_hist_obs_log,
        "log_price_vs_subcat_median": log_price_vs_subcat_median,
        # --- explanation / reference-only extras (not fed to the model) ---
        "current_price": float(current_price),
        "n_hist_obs": float(n),
        "hist_mean": hist_mean,
        "hist_median": hist_median,
        "hist_std": hist_std,
        "hist_min": hist_min,
        "hist_max": hist_max,
        "current_vs_hist_median_pct": _safe_div(current_price - hist_median, hist_median),
        "current_hist_percentile": hist_percentile,
    }


def mode2_features(current_price: float, subcat_stats: dict, brand_stats: dict | None) -> dict:
    """Build MODE 2 features (+ human-readable extras) from subcategory
    (+ optional brand) statistics only. No product-specific historical
    information is used or referenced.
    """
    subcat_median = subcat_stats["median"]
    subcat_mean = subcat_stats["mean"]
    subcat_std = subcat_stats["std"]
    p25, p75 = subcat_stats["p25"], subcat_stats["p75"]

    log_price_vs_subcat_median = _log_ratio(current_price, subcat_median)
    subcat_zscore = float(np.clip(_safe_div(current_price - subcat_mean, subcat_std), -_Z_CLIP, _Z_CLIP)) if subcat_std > 0 else 0.0
    subcat_percentile = percentile_of_score(subcat_stats["sorted_prices"], current_price)
    iqr = p75 - p25
    subcat_iqr_position = _safe_div(current_price - p25, iqr) if iqr > 0 else 0.0

    brand_available = brand_stats is not None
    if brand_available:
        brand_median = brand_stats["median"]
        log_price_vs_brand_median = _log_ratio(current_price, brand_median)
    else:
        # No fabricated brand number: fall back to the (real) subcategory
        # log-ratio so the feature vector has a consistent, non-invented
        # value, and the brand_available flag tells the model (and the
        # explanation layer) that this came from category level, not brand.
        brand_median = None
        log_price_vs_brand_median = log_price_vs_subcat_median

    return {
        # --- model features ---
        "log_price_vs_subcat_median": log_price_vs_subcat_median,
        "subcat_zscore": subcat_zscore,
        "subcat_percentile_frac": subcat_percentile / 100.0,
        "subcat_iqr_position": subcat_iqr_position,
        "brand_available": 1.0 if brand_available else 0.0,
        "log_price_vs_brand_median": log_price_vs_brand_median,
        # --- explanation / reference-only extras (not fed to the model) ---
        "current_price": float(current_price),
        "subcat_median": float(subcat_median),
        "subcat_mean": float(subcat_mean),
        "subcat_std": float(subcat_std),
        "subcat_p25": float(p25),
        "subcat_p75": float(p75),
        "subcat_p95": float(subcat_stats["p95"]),
        "current_vs_subcat_median_pct": _safe_div(current_price - subcat_median, subcat_median),
        "current_subcat_percentile": subcat_percentile,
        "brand_median": brand_median,
    }


def build_training_rows(history: pd.DataFrame, product_ids: list, subcat_stats: dict, brand_stats: dict):
    """Walk each product's chronological history and emit one MODE-1 row
    per observation that has >= MIN_HISTORY_FOR_MODE1 strictly-prior
    observations, and one MODE-2 row per observation (every observation is
    eligible for Mode 2, since Mode 2 never looks at the product's own
    history at all).

    Returns (mode1_df, mode2_df) of features (model columns + explanation
    extras) plus bookkeeping columns (product_id, timestamp).
    """
    mode1_rows = []
    mode2_rows = []

    subset = history[history.product_id.isin(product_ids)]
    for product_id, group in subset.groupby("product_id"):
        group = group.sort_values("timestamp")
        prices = group["price"].to_numpy()
        subcat = group["subcategory"].iloc[0]
        brand = group["brand"].iloc[0]
        timestamps = group["timestamp"].to_list()

        s_stats = subcat_stats.get(subcat)
        b_stats = brand_stats.get(subcat, {}).get(brand)

        for i in range(len(prices)):
            current_price = prices[i]
            prior_prices = prices[:i]

            # MODE 2 is always computable.
            if s_stats is not None:
                feats2 = mode2_features(current_price, s_stats, b_stats)
                feats2["product_id"] = product_id
                feats2["timestamp"] = timestamps[i]
                mode2_rows.append(feats2)

            # MODE 1 only once enough PRIOR history exists.
            if len(prior_prices) >= config.MIN_HISTORY_FOR_MODE1:
                feats1 = mode1_features_from_prior(current_price, prior_prices, s_stats)
                feats1["product_id"] = product_id
                feats1["timestamp"] = timestamps[i]
                mode1_rows.append(feats1)

    mode1_df = pd.DataFrame(mode1_rows)
    mode2_df = pd.DataFrame(mode2_rows)
    return mode1_df, mode2_df


if __name__ == "__main__":
    from .preprocess import load_history, split_products
    from .category_stats import compute_subcategory_stats, compute_brand_stats

    history = load_history()
    split = split_products(history)
    train_history = history[history.product_id.isin(split.train_ids)]

    subcat_stats = compute_subcategory_stats(train_history)
    brand_stats = compute_brand_stats(train_history)

    m1, m2 = build_training_rows(history, split.train_ids, subcat_stats, brand_stats)
    print("Mode1 training rows:", len(m1), "from products:", m1.product_id.nunique() if len(m1) else 0)
    print("Mode2 training rows:", len(m2), "from products:", m2.product_id.nunique() if len(m2) else 0)
    print()
    print(m1[MODE1_FEATURE_NAMES].describe().T if len(m1) else "no mode1 rows")
    print()
    print(m2[MODE2_FEATURE_NAMES].describe().T if len(m2) else "no mode2 rows")
