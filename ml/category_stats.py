"""
Subcategory-level and brand-within-subcategory pricing statistics.

These statistics are the reference distributions used by MODE 2
(market/category) features, and also as category *context* features inside
MODE 1. They are fit ONCE from the TRAINING split's observations and then
reused unchanged at validation/test/inference time -- exactly like fitting
a StandardScaler on a training set. This is standard practice, not a
leakage of a specific product's own future prices into its own row: no
product's Mode-1 features ever use that same product's own future
observations (see features.py for how that rule is enforced).

Limitation documented explicitly: because repeated price checkpoints of a
heavily-tracked product are all included in these distributions, a product
tracked 380 times contributes far more rows than one tracked twice. This
means the subcategory distribution is closer to "distribution of price
checkpoints seen" than "distribution of distinct products". This is
disclosed in MODEL_README.md.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config


def _price_stats(prices: np.ndarray) -> dict:
    prices = np.asarray(prices, dtype=float)
    prices = prices[~np.isnan(prices)]
    if len(prices) == 0:
        return None
    return {
        "n_obs": int(len(prices)),
        "n_products": None,  # filled by caller when relevant
        "mean": float(np.mean(prices)),
        "median": float(np.median(prices)),
        "std": float(np.std(prices, ddof=1)) if len(prices) > 1 else 0.0,
        "min": float(np.min(prices)),
        "max": float(np.max(prices)),
        "p25": float(np.percentile(prices, 25)),
        "p50": float(np.percentile(prices, 50)),
        "p75": float(np.percentile(prices, 75)),
        "p95": float(np.percentile(prices, 95)),
        "iqr": float(np.percentile(prices, 75) - np.percentile(prices, 25)),
        # Sorted prices are stored (bounded dataset size) so inference can
        # compute an exact empirical percentile-of-score for a submitted
        # price against the *training* distribution, consistently.
        "sorted_prices": [float(p) for p in np.sort(prices)],
    }


def compute_subcategory_stats(train_history: pd.DataFrame) -> dict:
    stats = {}
    for subcat, group in train_history.groupby("subcategory"):
        s = _price_stats(group["price"].values)
        if s is not None:
            s["n_products"] = int(group["product_id"].nunique())
            stats[subcat] = s
    return stats


def compute_brand_stats(train_history: pd.DataFrame) -> dict:
    """Brand-within-subcategory stats, only kept when the sample is large
    enough (config.MIN_OBS_FOR_BRAND_STATS observations) to be meaningful.
    """
    stats = {}
    for (subcat, brand), group in train_history.groupby(["subcategory", "brand"]):
        n_products = group["product_id"].nunique()
        if len(group) < config.MIN_OBS_FOR_BRAND_STATS:
            continue
        if n_products < config.MIN_PRODUCTS_FOR_BRAND_STATS:
            # e.g. "Sony" in Headphones has 380 observations but they all
            # come from a single product -> not a brand distribution.
            continue
        s = _price_stats(group["price"].values)
        if s is None:
            continue
        s["n_products"] = int(n_products)
        stats.setdefault(subcat, {})[brand] = s
    return stats


def percentile_of_score(sorted_prices: list, price: float) -> float:
    """Empirical percentile (0-100) of `price` within `sorted_prices`.

    Uses mean rank of ties (equivalent to scipy.stats.percentileofscore
    with kind='mean'), implemented locally so inference has no dependency
    surprises.
    """
    arr = np.asarray(sorted_prices, dtype=float)
    if len(arr) == 0:
        return 50.0
    left = np.searchsorted(arr, price, side="left")
    right = np.searchsorted(arr, price, side="right")
    rank = (left + right) / 2.0
    return float(100.0 * rank / len(arr))


def save_stats(subcat_stats: dict, brand_stats: dict) -> None:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.CATEGORY_STATS_PATH, "w") as f:
        json.dump(subcat_stats, f, indent=2)
    with open(config.BRAND_STATS_PATH, "w") as f:
        json.dump(brand_stats, f, indent=2)


def load_stats() -> tuple[dict, dict]:
    with open(config.CATEGORY_STATS_PATH) as f:
        subcat_stats = json.load(f)
    with open(config.BRAND_STATS_PATH) as f:
        brand_stats = json.load(f)
    return subcat_stats, brand_stats


if __name__ == "__main__":
    from .preprocess import load_history, split_products

    history = load_history()
    split = split_products(history)
    train_history = history[history.product_id.isin(split.train_ids)]

    subcat_stats = compute_subcategory_stats(train_history)
    brand_stats = compute_brand_stats(train_history)

    for subcat, s in subcat_stats.items():
        print(f"{subcat}: n_obs={s['n_obs']} n_products={s['n_products']} "
              f"median={s['median']:.0f} mean={s['mean']:.0f} std={s['std']:.0f} "
              f"p25={s['p25']:.0f} p75={s['p75']:.0f} p95={s['p95']:.0f}")
    print()
    for subcat, brands in brand_stats.items():
        for brand, s in brands.items():
            print(f"{subcat} / {brand}: n_obs={s['n_obs']} n_products={s['n_products']} median={s['median']:.0f}")
