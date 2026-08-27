"""
Full training pipeline. Run as:

    python -m ml.train

Produces, under ml/artifacts/:
    pricing_anomaly_model.joblib        (Mode 2 / market model -- always usable)
    pricing_anomaly_model_mode1.joblib  (Mode 1 / historical-product model)
    pricing_anomaly_model_mode2.joblib  (same object as pricing_anomaly_model.joblib)
    pricing_preprocessor.joblib         (dict of both modes' StandardScalers)
    model_features.json                 (feature name lists per mode + thresholds)
    category_stats.json                 (subcategory pricing distributions, train-fit)
    brand_stats.json                    (brand-within-subcategory distributions, train-fit)
    product_history.json                (per-product sorted price history, for lookups)
    train_val_test_products.json        (which product_ids went in which split)
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from . import config
from .preprocess import load_history, split_products, save_split_manifest
from .category_stats import compute_subcategory_stats, compute_brand_stats, save_stats
from .features import build_training_rows, MODE1_FEATURE_NAMES, MODE2_FEATURE_NAMES
from .model import fit_mode_model


def build_product_history_lookup(history: pd.DataFrame) -> dict:
    """Per-product sorted price history + metadata, used by predict.py to
    look up a product_id's OWN history at inference time. This intentionally
    includes ALL 54 products (train+val+test): in a deployed system, any
    product we have genuinely observed before should get Mode-1 treatment
    from its real history. Held-out evaluation of "new product" behaviour
    is done separately by *not* passing a product_id (see evaluate.py),
    which mirrors how a truly new listing would be submitted.
    """
    lookup = {}
    for product_id, group in history.groupby("product_id"):
        group = group.sort_values("timestamp")
        lookup[product_id] = {
            "product_name": group["product_name"].iloc[-1],
            "brand": group["brand"].iloc[-1],
            "subcategory": group["subcategory"].iloc[-1],
            "category": group["category"].iloc[-1],
            "n_obs": int(len(group)),
            "prices": [float(p) for p in group["price"].tolist()],
            # Per-observation timestamps, parallel to "prices", so the
            # website can plot a real date-vs-price chart. Added for the
            # Streamlit integration (historical price chart requirement) --
            # this does not change any feature, statistic, or model
            # training in any way; it's purely an additional export field.
            "timestamps": [ts.isoformat() for ts in group["timestamp"].tolist()],
            "last_timestamp": group["timestamp"].iloc[-1].isoformat(),
        }
    return lookup


def main():
    print("=" * 70)
    print("STEP 1: Load & clean historical data")
    print("=" * 70)
    history = load_history()
    print(f"Loaded {len(history)} observations across {history.product_id.nunique()} products")

    print()
    print("=" * 70)
    print("STEP 2: Product-level train/val/test split")
    print("=" * 70)
    split = split_products(history)
    save_split_manifest(split)
    print(f"Train: {len(split.train_ids)} products | Val: {len(split.val_ids)} products | Test: {len(split.test_ids)} products")
    print(f"Train obs: {history[history.product_id.isin(split.train_ids)].shape[0]}")
    print(f"Val obs:   {history[history.product_id.isin(split.val_ids)].shape[0]}")
    print(f"Test obs:  {history[history.product_id.isin(split.test_ids)].shape[0]}")

    print()
    print("=" * 70)
    print("STEP 3: Fit subcategory & brand statistics (TRAIN split only)")
    print("=" * 70)
    train_history = history[history.product_id.isin(split.train_ids)]
    subcat_stats = compute_subcategory_stats(train_history)
    brand_stats = compute_brand_stats(train_history)
    save_stats(subcat_stats, brand_stats)
    for subcat, s in subcat_stats.items():
        print(f"  {subcat}: n_obs={s['n_obs']} n_products={s['n_products']} median={s['median']:.0f}")
    n_brand_groups = sum(len(v) for v in brand_stats.values())
    print(f"  Brand-within-subcategory groups retained (>= {config.MIN_PRODUCTS_FOR_BRAND_STATS} products, "
          f">= {config.MIN_OBS_FOR_BRAND_STATS} obs): {n_brand_groups}")

    print()
    print("=" * 70)
    print("STEP 4: Build leakage-safe training rows")
    print("=" * 70)
    mode1_train_df, mode2_train_df = build_training_rows(history, split.train_ids, subcat_stats, brand_stats)
    print(f"  Mode 1 (historical) training rows: {len(mode1_train_df)} from {mode1_train_df.product_id.nunique()} products")
    print(f"  Mode 2 (market) training rows: {len(mode2_train_df)} from {mode2_train_df.product_id.nunique()} products")

    print()
    print("=" * 70)
    print("STEP 5: Train Isolation Forest models (one per mode)")
    print("=" * 70)
    X1 = mode1_train_df[MODE1_FEATURE_NAMES].to_numpy(dtype=float)
    X2 = mode2_train_df[MODE2_FEATURE_NAMES].to_numpy(dtype=float)

    mode1_model = fit_mode_model("historical", MODE1_FEATURE_NAMES, X1)
    mode2_model = fit_mode_model("market", MODE2_FEATURE_NAMES, X2)
    print(f"  Mode 1 model trained on {X1.shape[0]} rows x {X1.shape[1]} features")
    print(f"  Mode 2 model trained on {X2.shape[0]} rows x {X2.shape[1]} features")

    print()
    print("=" * 70)
    print("STEP 6: Save artifacts")
    print("=" * 70)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(mode1_model, config.MODEL_MODE1_PATH)
    joblib.dump(mode2_model, config.MODEL_MODE2_PATH)
    # "pricing_anomaly_model.joblib" as requested by spec: the model every
    # submitted product can always fall back to is the Mode 2 (market) model.
    joblib.dump(mode2_model, config.MODEL_PRIMARY_PATH)

    joblib.dump(
        {"mode1_scaler": mode1_model.scaler, "mode2_scaler": mode2_model.scaler},
        config.PREPROCESSOR_PATH,
    )

    with open(config.MODEL_FEATURES_PATH, "w") as f:
        json.dump(
            {
                "mode1_features": MODE1_FEATURE_NAMES,
                "mode2_features": MODE2_FEATURE_NAMES,
                "min_history_for_mode1": config.MIN_HISTORY_FOR_MODE1,
                "min_obs_for_brand_stats": config.MIN_OBS_FOR_BRAND_STATS,
                "min_products_for_brand_stats": config.MIN_PRODUCTS_FOR_BRAND_STATS,
                "isolation_forest_params": config.ISOLATION_FOREST_PARAMS,
            },
            f,
            indent=2,
        )

    product_history = build_product_history_lookup(history)
    with open(config.PRODUCT_HISTORY_PATH, "w") as f:
        json.dump(product_history, f, indent=2)

    print(f"  Saved: {config.MODEL_MODE1_PATH.name}")
    print(f"  Saved: {config.MODEL_MODE2_PATH.name}")
    print(f"  Saved: {config.MODEL_PRIMARY_PATH.name}")
    print(f"  Saved: {config.PREPROCESSOR_PATH.name}")
    print(f"  Saved: {config.MODEL_FEATURES_PATH.name}")
    print(f"  Saved: {config.CATEGORY_STATS_PATH.name}")
    print(f"  Saved: {config.BRAND_STATS_PATH.name}")
    print(f"  Saved: {config.PRODUCT_HISTORY_PATH.name}")
    print(f"  Saved: {config.SPLIT_MANIFEST_PATH.name}")

    print()
    print("Training complete.")
    return {
        "history": history,
        "split": split,
        "subcat_stats": subcat_stats,
        "brand_stats": brand_stats,
        "mode1_model": mode1_model,
        "mode2_model": mode2_model,
        "mode1_train_df": mode1_train_df,
        "mode2_train_df": mode2_train_df,
    }


if __name__ == "__main__":
    main()
