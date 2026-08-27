"""
Data loading, cleaning, and product-level splitting.

Nothing in this module computes ML features — it only prepares a clean,
sorted DataFrame and decides which product_ids belong to train/val/test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


def load_history() -> pd.DataFrame:
    """Load and clean the raw historical price-checkpoint CSV.

    Cleaning performed here:
      - parse timestamp to timezone-aware datetime
      - coerce price to numeric, drop rows with missing/invalid price
      - normalize brand-name casing (Boat/boAt -> boAt, etc.)
      - sort rows by (product_id, timestamp) so "prior history" is a simple
        prefix slice per product
    """
    df = pd.read_csv(config.HISTORY_CSV)

    # format="ISO8601" is required here: the raw timestamps mix
    # microsecond-precision and second-precision ISO strings, and pandas'
    # format auto-detection silently produces NaT for the second-precision
    # ones unless the parser is told to expect (mixed) ISO8601 explicitly.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601", errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["timestamp", "price", "product_id", "subcategory"])
    dropped = before - len(df)
    if dropped:
        print(f"[preprocess] Dropped {dropped} rows with missing timestamp/price/id/subcategory")

    df = df[df["price"] > 0].copy()

    df["brand_raw"] = df["brand"]
    df["brand"] = df["brand_raw"].apply(config.normalize_brand)

    df = df.sort_values(["product_id", "timestamp"]).reset_index(drop=True)
    return df


@dataclass
class ProductSplit:
    train_ids: list
    val_ids: list
    test_ids: list


def split_products(history: pd.DataFrame) -> ProductSplit:
    """Split PRODUCT IDs (not rows) into train/val/test.

    Stratified on two axes so each split is representative:
      - subcategory (Wireless Earphones/TWS, Headphones, Earphones)
      - "depth bucket": whether the product has >= MIN_HISTORY_FOR_MODE1
        total observations (deep-history vs shallow-history products)

    All observations for a given product_id end up in exactly one split.
    """
    product_meta = (
        history.groupby("product_id")
        .agg(subcategory=("subcategory", "first"), n_obs=("price", "size"))
        .reset_index()
    )
    product_meta["depth_bucket"] = np.where(
        product_meta["n_obs"] >= config.MIN_HISTORY_FOR_MODE1, "deep", "shallow"
    )
    product_meta["stratum"] = product_meta["subcategory"] + "|" + product_meta["depth_bucket"]

    rng = np.random.RandomState(config.RANDOM_SEED)
    train_ids, val_ids, test_ids = [], [], []

    for _, group in product_meta.groupby("stratum"):
        ids = group["product_id"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, round(n * config.TEST_FRACTION)) if n >= 3 else (1 if n > 1 else 0)
        n_val = max(1, round(n * config.VAL_FRACTION)) if n >= 5 else (1 if n - n_test > 1 else 0)
        n_val = min(n_val, max(0, n - n_test - 1))
        n_test = min(n_test, max(0, n - 1)) if n > 1 else 0

        test_ids += ids[:n_test]
        val_ids += ids[n_test:n_test + n_val]
        train_ids += ids[n_test + n_val:]

    # Safety net: if stratification left a split empty (can happen with a
    # very small stratum), rebalance a couple of products from train.
    if not test_ids:
        test_ids = [train_ids.pop()]
    if not val_ids:
        val_ids = [train_ids.pop()]

    split = ProductSplit(train_ids=sorted(train_ids), val_ids=sorted(val_ids), test_ids=sorted(test_ids))

    assert set(split.train_ids).isdisjoint(split.val_ids)
    assert set(split.train_ids).isdisjoint(split.test_ids)
    assert set(split.val_ids).isdisjoint(split.test_ids)

    return split


def save_split_manifest(split: ProductSplit) -> None:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.SPLIT_MANIFEST_PATH, "w") as f:
        json.dump(
            {"train": split.train_ids, "val": split.val_ids, "test": split.test_ids},
            f,
            indent=2,
        )


def load_split_manifest() -> ProductSplit:
    with open(config.SPLIT_MANIFEST_PATH) as f:
        data = json.load(f)
    return ProductSplit(train_ids=data["train"], val_ids=data["val"], test_ids=data["test"])


if __name__ == "__main__":
    history = load_history()
    split = split_products(history)
    print(f"Total products: {history['product_id'].nunique()}")
    print(f"Train: {len(split.train_ids)}  Val: {len(split.val_ids)}  Test: {len(split.test_ids)}")
    print(f"Train obs: {history[history.product_id.isin(split.train_ids)].shape[0]}")
    print(f"Val obs:   {history[history.product_id.isin(split.val_ids)].shape[0]}")
    print(f"Test obs:  {history[history.product_id.isin(split.test_ids)].shape[0]}")
