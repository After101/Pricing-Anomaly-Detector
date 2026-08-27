"""
Test suite for the pricing anomaly system.

Run as:  python -m pytest ml/tests/test_predict.py -v
     or: python -m ml.tests.test_predict     (runs as a plain script too)

Covers the 10 required scenarios:
  1. Known product with sufficient history            -> test_known_product_sufficient_history
  2. Known product with insufficient history           -> test_known_product_insufficient_history
  3. Completely unseen product                         -> test_completely_unseen_product
  4. Missing optional brand                             -> test_missing_optional_brand
  5. Invalid price                                      -> test_invalid_price
  6. Unknown subcategory                                -> test_unknown_subcategory
  7. Unknown brand                                       -> test_unknown_brand
  8. Product priced near category median                -> test_price_near_category_median
  9. Product priced extreme relative to category         -> test_price_extreme_vs_category
 10. Historical product with extreme current price       -> test_historical_product_extreme_current_price

Plus:
  - correct mode selection
  - no future leakage (spot check)
  - no fabricated features (spot check on Mode 2 with no brand)
  - stable 0-1 scoring
  - deterministic inference
  - correct JSON response (json.dumps roundtrip)
  - model artifact loading works independently of training (fresh process)
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from ml import config
from ml.predict import predict_product, InvalidProductInputError
from ml.preprocess import load_history, load_split_manifest


@pytest.fixture(scope="module")
def history():
    return load_history()


@pytest.fixture(scope="module")
def split():
    return load_split_manifest()


@pytest.fixture(scope="module")
def counts(history):
    return history.groupby("product_id").size()


# ---------------------------------------------------------------------
# 1. Known product with sufficient history -> MODE 1
# ---------------------------------------------------------------------
def test_known_product_sufficient_history(history, split, counts):
    deep_products = [pid for pid in split.train_ids if counts[pid] >= config.MIN_HISTORY_FOR_MODE1]
    assert deep_products, "Expected at least one deep-history training product"
    pid = deep_products[0]
    row = history[history.product_id == pid].iloc[-1]

    result = predict_product({
        "product_id": pid,
        "price": row.price,
        "brand": row.brand,
        "subcategory": row.subcategory,
    })
    assert result["mode"] == "historical"
    assert 0.0 <= result["anomaly_score"] <= 1.0
    assert "historical_median" in result["reference_values"]
    assert result["warnings"] == []


# ---------------------------------------------------------------------
# 2. Known product with insufficient history -> falls back to MODE 2
# ---------------------------------------------------------------------
def test_known_product_insufficient_history(history, split, counts):
    shallow_products = [pid for pid in split.train_ids if counts[pid] < config.MIN_HISTORY_FOR_MODE1]
    assert shallow_products, "Expected at least one shallow-history training product"
    pid = shallow_products[0]
    row = history[history.product_id == pid].iloc[-1]

    result = predict_product({
        "product_id": pid,
        "price": row.price,
        "brand": row.brand,
        "subcategory": row.subcategory,
    })
    assert result["mode"] == "market"
    assert any("insufficient" in w.lower() or str(config.MIN_HISTORY_FOR_MODE1) in w for w in result["warnings"])
    # No historical fields should be fabricated for this product.
    assert "historical_median" not in result["reference_values"]


# ---------------------------------------------------------------------
# 3. Completely unseen product (no product_id) -> MODE 2
# ---------------------------------------------------------------------
def test_completely_unseen_product():
    result = predict_product({
        "product_name": "Totally New Earbuds Nobody Has Seen",
        "price": 1599,
        "brand": "BrandNewCo",
        "subcategory": "Wireless Earphones / TWS",
    })
    assert result["mode"] == "market"
    assert "historical_median" not in result["reference_values"]
    assert "brand_median_in_category" not in result["reference_values"]  # unknown brand -> no fabricated brand stat


# ---------------------------------------------------------------------
# 4. Missing optional brand
# ---------------------------------------------------------------------
def test_missing_optional_brand():
    result = predict_product({
        "price": 1599,
        "subcategory": "Wireless Earphones / TWS",
        # brand intentionally omitted
    })
    assert result["mode"] == "market"
    assert 0.0 <= result["anomaly_score"] <= 1.0


# ---------------------------------------------------------------------
# 5. Invalid price
# ---------------------------------------------------------------------
@pytest.mark.parametrize("bad_price", [None, -100, 0, "not_a_number", float("nan"), float("inf")])
def test_invalid_price(bad_price):
    with pytest.raises(InvalidProductInputError):
        predict_product({"price": bad_price, "subcategory": "Wireless Earphones / TWS"})


# ---------------------------------------------------------------------
# 6. Unknown subcategory
# ---------------------------------------------------------------------
def test_unknown_subcategory():
    with pytest.raises(InvalidProductInputError):
        predict_product({"price": 1599, "subcategory": "Smart Watches"})


# ---------------------------------------------------------------------
# 7. Unknown brand (known subcategory) -> should still work, no fabricated brand stats
# ---------------------------------------------------------------------
def test_unknown_brand():
    result = predict_product({
        "price": 1599,
        "brand": "TotallyMadeUpBrandXYZ",
        "subcategory": "Wireless Earphones / TWS",
    })
    assert result["mode"] == "market"
    assert "brand_median_in_category" not in result["reference_values"]
    assert any("brand" in r.lower() for r in result["reasons"])


# ---------------------------------------------------------------------
# 8. Product priced near category median -> low-ish score
# ---------------------------------------------------------------------
def test_price_near_category_median():
    result = predict_product({
        "price": 1499,  # ~= Wireless Earphones/TWS training median
        "subcategory": "Wireless Earphones / TWS",
    })
    assert result["anomaly_score"] < 0.4, f"Expected a low score near the category median, got {result['anomaly_score']}"


# ---------------------------------------------------------------------
# 9. Product priced extreme relative to category -> high score
# ---------------------------------------------------------------------
def test_price_extreme_vs_category():
    result = predict_product({
        "price": 39999,  # far beyond category p95
        "subcategory": "Wireless Earphones / TWS",
    })
    assert result["anomaly_score"] > 0.85, f"Expected a high score for an extreme price, got {result['anomaly_score']}"
    assert result["classification"] == "Highly unusual pricing"


# ---------------------------------------------------------------------
# 10. Historical product whose CURRENT price is extreme vs its own history
# ---------------------------------------------------------------------
def test_historical_product_extreme_current_price(history, split, counts):
    deep_products = [pid for pid in split.train_ids if counts[pid] >= config.MIN_HISTORY_FOR_MODE1]
    pid = deep_products[0]
    row = history[history.product_id == pid].iloc[-1]
    hist_max = history[history.product_id == pid]["price"].max()

    extreme_price = hist_max * 5  # deliberately far outside the product's own range
    result = predict_product({
        "product_id": pid,
        "price": extreme_price,
        "brand": row.brand,
        "subcategory": row.subcategory,
    })
    assert result["mode"] == "historical"
    assert result["anomaly_score"] > 0.8, f"Expected a high score, got {result['anomaly_score']}"
    assert any("maximum" in r.lower() or "above" in r.lower() for r in result["reasons"])


# ---------------------------------------------------------------------
# Extra: mode-selection correctness across the threshold boundary
# ---------------------------------------------------------------------
def test_mode_selection_boundary(history, split, counts):
    at_or_above = [pid for pid in split.train_ids if counts[pid] >= config.MIN_HISTORY_FOR_MODE1]
    below = [pid for pid in split.train_ids if counts[pid] < config.MIN_HISTORY_FOR_MODE1]
    assert at_or_above and below

    for pid in at_or_above[:2]:
        row = history[history.product_id == pid].iloc[-1]
        r = predict_product({"product_id": pid, "price": row.price, "subcategory": row.subcategory})
        assert r["mode"] == "historical"

    for pid in below[:2]:
        row = history[history.product_id == pid].iloc[-1]
        r = predict_product({"product_id": pid, "price": row.price, "subcategory": row.subcategory})
        assert r["mode"] == "market"


# ---------------------------------------------------------------------
# Extra: no future leakage -- an early observation's Mode-1 features must
# only reflect PRIOR observations, never the product's full/future history.
# ---------------------------------------------------------------------
def test_no_future_leakage(history, split, counts):
    from ml.features import mode1_features_from_prior
    from ml.category_stats import load_stats

    subcat_stats, _ = load_stats()
    deep_products = [pid for pid in split.train_ids if counts[pid] >= config.MIN_HISTORY_FOR_MODE1]
    pid = deep_products[0]
    group = history[history.product_id == pid].sort_values("timestamp")
    prices = group["price"].to_numpy()
    subcat = group["subcategory"].iloc[0]

    # Take a mid-sequence point. Features computed from only the prior
    # slice must NOT depend on later prices.
    i = config.MIN_HISTORY_FOR_MODE1 + 5
    feats_prior_only = mode1_features_from_prior(prices[i], prices[:i], subcat_stats.get(subcat))
    feats_with_future_leak = mode1_features_from_prior(prices[i], prices[: i + 50], subcat_stats.get(subcat))

    # These should generally differ once future information is (wrongly) included,
    # proving the feature values are sensitive to which slice is passed in --
    # i.e. our production code path (which passes prices[:i]) is the one
    # that determines leakage-safety, not an accident of the math being
    # insensitive to it.
    assert feats_prior_only["hist_median"] != feats_with_future_leak["hist_median"] or \
           feats_prior_only["n_hist_obs"] != feats_with_future_leak["n_hist_obs"]


# ---------------------------------------------------------------------
# Extra: no fabricated features for a brand-new product
# ---------------------------------------------------------------------
def test_no_fabricated_features_for_new_product():
    result = predict_product({
        "price": 1599,
        "subcategory": "Headphones",
        # no product_id, no brand
    })
    ref = result["reference_values"]
    forbidden_keys = {"historical_mean", "historical_median", "historical_min", "historical_max",
                       "brand_median_in_category", "brand_n_products"}
    assert forbidden_keys.isdisjoint(ref.keys())


# ---------------------------------------------------------------------
# Extra: stable 0-1 scoring range across many random-ish prices
# ---------------------------------------------------------------------
@pytest.mark.parametrize("price", [199, 499, 999, 1499, 2999, 5999, 9999, 19999, 49999, 99999])
def test_score_always_in_unit_interval(price):
    result = predict_product({"price": price, "subcategory": "Wireless Earphones / TWS"})
    assert 0.0 <= result["anomaly_score"] <= 1.0


# ---------------------------------------------------------------------
# Extra: deterministic inference (same input -> same output, repeated calls)
# ---------------------------------------------------------------------
def test_deterministic_inference():
    payload = {"price": 3499, "brand": "boAt", "subcategory": "Wireless Earphones / TWS"}
    r1 = predict_product(dict(payload))
    r2 = predict_product(dict(payload))
    assert r1 == r2


# ---------------------------------------------------------------------
# Extra: response is valid, complete JSON
# ---------------------------------------------------------------------
def test_json_serializable_response():
    result = predict_product({"price": 1299, "subcategory": "Headphones"})
    encoded = json.dumps(result)
    decoded = json.loads(encoded)
    assert decoded["anomaly_score"] == result["anomaly_score"]
    for key in ["anomaly_score", "mode", "classification", "current_price", "reference_values", "reasons", "warnings"]:
        assert key in decoded


# ---------------------------------------------------------------------
# Extra: model artifacts load and predict correctly in a FRESH process,
# independent of whether training just ran in this process.
# ---------------------------------------------------------------------
def test_artifact_loading_in_fresh_process():
    code = (
        "from ml.predict import predict_product; "
        "import json; "
        "r = predict_product({'price': 1499, 'subcategory': 'Wireless Earphones / TWS'}); "
        "print(json.dumps(r))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(config.PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert 0.0 <= result["anomaly_score"] <= 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
