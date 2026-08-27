"""
predict_product(product_data) -- the single function a website (or anything
else) needs to call. Everything about Isolation Forest, feature
engineering, and mode selection is hidden behind this function.

Usage:
    from ml.predict import predict_product

    result = predict_product({
        "product_id": "B0784BMDRW",     # optional
        "product_name": "...",
        "price": 1299,
        "brand": "boAt",
        "category": "Headphones & Earbuds",
        "subcategory": "Wireless Earphones / TWS",
        "platform": "Amazon",
    })

Artifacts are loaded once per process (module-level cache) so repeated
calls are fast and every call is deterministic given the same input and
the same saved artifacts.
"""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np

from . import config
from .category_stats import percentile_of_score
from .features import mode1_features_from_prior, mode2_features, MODE1_FEATURE_NAMES, MODE2_FEATURE_NAMES
from .scoring import classify
from .config import normalize_brand


class ArtifactsNotFoundError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_artifacts():
    required = [
        config.MODEL_MODE1_PATH,
        config.MODEL_MODE2_PATH,
        config.CATEGORY_STATS_PATH,
        config.BRAND_STATS_PATH,
        config.PRODUCT_HISTORY_PATH,
        config.MODEL_FEATURES_PATH,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise ArtifactsNotFoundError(
            "Missing trained artifacts: "
            + ", ".join(str(p) for p in missing)
            + ". Run `python -m ml.train` first."
        )

    mode1_model = joblib.load(config.MODEL_MODE1_PATH)
    mode2_model = joblib.load(config.MODEL_MODE2_PATH)

    with open(config.CATEGORY_STATS_PATH) as f:
        subcat_stats = json.load(f)
    with open(config.BRAND_STATS_PATH) as f:
        brand_stats = json.load(f)
    with open(config.PRODUCT_HISTORY_PATH) as f:
        product_history = json.load(f)
    with open(config.MODEL_FEATURES_PATH) as f:
        feature_meta = json.load(f)

    return {
        "mode1_model": mode1_model,
        "mode2_model": mode2_model,
        "subcat_stats": subcat_stats,
        "brand_stats": brand_stats,
        "product_history": product_history,
        "feature_meta": feature_meta,
    }


class InvalidProductInputError(ValueError):
    pass


def _validate_input(product_data: dict) -> dict:
    if "price" not in product_data or product_data["price"] is None:
        raise InvalidProductInputError("`price` is required.")
    try:
        price = float(product_data["price"])
    except (TypeError, ValueError):
        raise InvalidProductInputError(f"`price` must be numeric, got {product_data['price']!r}")
    if not np.isfinite(price) or price <= 0:
        raise InvalidProductInputError(f"`price` must be a positive finite number, got {price}")

    subcategory = product_data.get("subcategory")
    if not subcategory:
        raise InvalidProductInputError("`subcategory` is required (one of the known subcategories).")

    cleaned = dict(product_data)
    cleaned["price"] = price
    cleaned["brand"] = normalize_brand(product_data.get("brand")) if product_data.get("brand") else None
    return cleaned


def _explain_mode1(current_price, feats, subcat_stats) -> list:
    reasons = []
    hist_median = feats["hist_median"]
    hist_min, hist_max = feats["hist_min"], feats["hist_max"]
    pct = feats["current_hist_percentile"]
    diff_median_pct = feats["current_vs_hist_median_pct"] * 100

    if current_price > hist_max:
        reasons.append(
            f"Current price \u20b9{current_price:,.0f} is above the product's observed historical "
            f"maximum of \u20b9{hist_max:,.0f}."
        )
    elif current_price < hist_min:
        reasons.append(
            f"Current price \u20b9{current_price:,.0f} is below the product's observed historical "
            f"minimum of \u20b9{hist_min:,.0f}."
        )

    if abs(diff_median_pct) >= 15:
        if diff_median_pct > 0:
            ratio = current_price / hist_median if hist_median else None
            if ratio is not None:
                reasons.append(
                    f"Current price is approximately {ratio:.1f}\u00d7 the product's historical "
                    f"median (\u20b9{hist_median:,.0f})."
                )
        else:
            reasons.append(
                f"Current price is {abs(diff_median_pct):.0f}% below the product's historical "
                f"median (\u20b9{hist_median:,.0f}); this describes the price difference, not "
                f"overpricing or fraud."
            )

    if pct >= 90 or pct <= 10:
        reasons.append(f"Current price falls at the {pct:.0f}th percentile of this product's own price history.")

    if not reasons:
        reasons.append(
            f"Current price \u20b9{current_price:,.0f} is close to the product's historical median "
            f"(\u20b9{hist_median:,.0f}) and within its normal observed range."
        )
    return reasons


def _explain_mode2(current_price, feats, subcategory) -> list:
    reasons = []
    subcat_median = feats["subcat_median"]
    subcat_p95 = feats["subcat_p95"]
    subcat_p25 = feats["subcat_p25"]
    pct = feats["current_subcat_percentile"]
    diff_median_pct = feats["current_vs_subcat_median_pct"] * 100

    if pct >= 95:
        reasons.append(
            f"Current price is above the 95th percentile of the {subcategory} category "
            f"(category 95th percentile: \u20b9{subcat_p95:,.0f})."
        )
    elif pct <= 5:
        reasons.append(
            f"Current price is below the 5th percentile of the {subcategory} category "
            f"(category 25th percentile: \u20b9{subcat_p25:,.0f})."
        )

    if abs(diff_median_pct) >= 25:
        if diff_median_pct > 0:
            ratio = current_price / subcat_median if subcat_median else None
            if ratio is not None:
                reasons.append(
                    f"Current price is approximately {ratio:.1f}\u00d7 the {subcategory} category "
                    f"median (\u20b9{subcat_median:,.0f})."
                )
        else:
            reasons.append(
                f"Current price is {abs(diff_median_pct):.0f}% below the {subcategory} category "
                f"median (\u20b9{subcat_median:,.0f}); this describes the price difference, not "
                f"overpricing or fraud."
            )

    if feats["brand_available"] < 0.5:
        reasons.append(
            "No reliable brand-specific pricing distribution was available for this product; "
            "comparison is against the overall subcategory."
        )

    if not reasons:
        reasons.append(
            f"Current price \u20b9{current_price:,.0f} is close to the {subcategory} category median "
            f"(\u20b9{subcat_median:,.0f})."
        )
    return reasons


def predict_product(product_data: dict) -> dict:
    """Core inference entry point.

    Args:
        product_data: dict with at least `price` and `subcategory`.
            Optional: product_id, product_name, brand, category, platform, url.
            If `product_id` matches a known product AND that product has
            >= config.MIN_HISTORY_FOR_MODE1 historical observations, MODE 1
            (historical-product) is used. Otherwise MODE 2 (market/category).

    Returns:
        JSON-serializable dict:
        {
            "anomaly_score": float 0-1,
            "mode": "historical" | "market",
            "classification": str,
            "current_price": float,
            "reference_values": {...},   # only real, non-fabricated values
            "reasons": [str, ...],
            "warnings": [str, ...],      # e.g. unknown subcategory fallback
        }
    """
    artifacts = _load_artifacts()
    warnings = []

    cleaned = _validate_input(product_data)
    price = cleaned["price"]
    subcategory = cleaned.get("subcategory")
    brand = cleaned.get("brand")
    product_id = cleaned.get("product_id")

    subcat_stats = artifacts["subcat_stats"].get(subcategory)
    if subcat_stats is None:
        known = ", ".join(sorted(artifacts["subcat_stats"].keys()))
        raise InvalidProductInputError(
            f"Unknown subcategory {subcategory!r}. Known subcategories: {known}. "
            f"The model has no reference pricing distribution for an unrecognized subcategory, "
            f"so a score cannot be produced honestly."
        )

    # --- Decide mode -------------------------------------------------
    product_record = artifacts["product_history"].get(product_id) if product_id else None
    use_mode1 = product_record is not None and product_record["n_obs"] >= config.MIN_HISTORY_FOR_MODE1

    if product_id and product_record is None:
        warnings.append(f"product_id {product_id!r} not found in historical dataset; using market/category mode.")
    elif product_record is not None and not use_mode1:
        warnings.append(
            f"product_id {product_id!r} has only {product_record['n_obs']} historical observations "
            f"(< {config.MIN_HISTORY_FOR_MODE1} required for historical mode); using market/category mode."
        )

    if use_mode1:
        # ---------------- MODE 1: historical product ----------------
        prior_prices = np.array(product_record["prices"], dtype=float)
        feats = mode1_features_from_prior(price, prior_prices, subcat_stats)
        X = np.array([[feats[name] for name in MODE1_FEATURE_NAMES]], dtype=float)
        anomaly_score = float(artifacts["mode1_model"].anomaly_score(X)[0])

        reference_values = {
            "subcategory": subcategory,
            "historical_n_obs": int(feats["n_hist_obs"]),
            "historical_mean": round(feats["hist_mean"], 2),
            "historical_median": round(feats["hist_median"], 2),
            "historical_min": round(feats["hist_min"], 2),
            "historical_max": round(feats["hist_max"], 2),
            "historical_std": round(feats["hist_std"], 2),
            "category_median": round(subcat_stats["median"], 2),
        }
        reasons = _explain_mode1(price, feats, subcat_stats)
        mode = "historical"

    else:
        # ---------------- MODE 2: market / category ----------------
        brand_stats = None
        if brand:
            brand_stats = artifacts["brand_stats"].get(subcategory, {}).get(brand)
        feats = mode2_features(price, subcat_stats, brand_stats)
        X = np.array([[feats[name] for name in MODE2_FEATURE_NAMES]], dtype=float)
        anomaly_score = float(artifacts["mode2_model"].anomaly_score(X)[0])

        reference_values = {
            "subcategory": subcategory,
            "category_n_obs": subcat_stats["n_obs"],
            "category_n_products": subcat_stats["n_products"],
            "category_median": round(subcat_stats["median"], 2),
            "category_mean": round(subcat_stats["mean"], 2),
            "category_std": round(subcat_stats["std"], 2),
            "category_p25": round(subcat_stats["p25"], 2),
            "category_p75": round(subcat_stats["p75"], 2),
            "category_p95": round(subcat_stats["p95"], 2),
        }
        if brand_stats is not None:
            reference_values["brand_median_in_category"] = round(brand_stats["median"], 2)
            reference_values["brand_n_products"] = brand_stats["n_products"]
        else:
            pass  # covered below (kept for clarity of the branch)
            # Explicitly OMIT brand reference fields rather than filling
            # zeros/None-as-a-number -- there is no brand-level number to
            # report.
            pass

        reasons = _explain_mode2(price, feats, subcategory)
        mode = "market"

    return {
        "anomaly_score": round(anomaly_score, 4),
        "mode": mode,
        "classification": classify(anomaly_score),
        "current_price": price,
        "reference_values": reference_values,
        "reasons": reasons,
        "warnings": warnings,
    }
