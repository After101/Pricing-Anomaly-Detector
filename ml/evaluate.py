"""
Honest evaluation of the trained models.

This is UNSUPERVISED anomaly detection -- there are no ground-truth
fraud/scam labels in this dataset, and this module does NOT fabricate any.
What it reports instead:

  1. Anomaly score distributions on train/val/test (per mode)
  2. The highest-scoring ("most anomalous") examples for manual inspection
  3. Correlation between the Isolation Forest score and two simple
     statistical baselines (subcategory z-score; own-history z-score)
  4. Behaviour on genuinely unseen test products, simulating the
     "brand-new listing" scenario (product_id withheld from the model)

Run as:  python -m ml.evaluate
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from . import config
from .preprocess import load_history, load_split_manifest
from .category_stats import load_stats
from .features import build_training_rows, MODE1_FEATURE_NAMES, MODE2_FEATURE_NAMES
from .baselines import baseline_subcategory_zscore, baseline_own_history_zscore
from .predict import predict_product, ArtifactsNotFoundError


def _score_rows(model, df, feature_names):
    X = df[feature_names].to_numpy(dtype=float)
    return model.anomaly_score(X)


def evaluate():
    history = load_history()
    split = load_split_manifest()
    subcat_stats, brand_stats = load_stats()

    mode1_model = joblib.load(config.MODEL_MODE1_PATH)
    mode2_model = joblib.load(config.MODEL_MODE2_PATH)

    report_lines = []

    def log(line=""):
        print(line)
        report_lines.append(line)

    log("=" * 78)
    log("EVALUATION REPORT (unsupervised -- no fabricated labels/accuracy)")
    log("=" * 78)

    splits = {
        "train": split.train_ids,
        "val": split.val_ids,
        "test": split.test_ids,
    }

    all_scored = {}
    for split_name, ids in splits.items():
        m1, m2 = build_training_rows(history, ids, subcat_stats, brand_stats)
        if len(m1):
            m1 = m1.assign(anomaly_score=_score_rows(mode1_model, m1, MODE1_FEATURE_NAMES))
        if len(m2):
            m2 = m2.assign(anomaly_score=_score_rows(mode2_model, m2, MODE2_FEATURE_NAMES))
        all_scored[split_name] = (m1, m2)

    log("\n--- Anomaly score distribution by split & mode ---")
    for split_name, (m1, m2) in all_scored.items():
        for mode_name, df in [("MODE 1 (historical)", m1), ("MODE 2 (market)", m2)]:
            if len(df) == 0:
                log(f"{split_name:6s} | {mode_name:22s} | (no rows)")
                continue
            s = df["anomaly_score"]
            log(
                f"{split_name:6s} | {mode_name:22s} | n={len(s):5d}  "
                f"mean={s.mean():.3f}  median={s.median():.3f}  "
                f"p90={s.quantile(.9):.3f}  p99={s.quantile(.99):.3f}  max={s.max():.3f}"
            )

    log("\n--- Top 10 highest-scoring MODE 1 rows across all splits (manual review) ---")
    m1_all = pd.concat([df for _, (df, _) in all_scored.items() if len(df)], ignore_index=True)
    top1 = m1_all.sort_values("anomaly_score", ascending=False).head(10)
    for _, row in top1.iterrows():
        log(
            f"  product={row.product_id}  price={row.current_price:.0f}  "
            f"hist_median={row.hist_median:.0f}  hist_range=[{row.hist_min:.0f},{row.hist_max:.0f}]  "
            f"score={row.anomaly_score:.3f}"
        )

    log("\n--- Top 10 highest-scoring MODE 2 rows across all splits (manual review) ---")
    m2_all = pd.concat([df for _, (_, df) in all_scored.items() if len(df)], ignore_index=True)
    top2 = m2_all.sort_values("anomaly_score", ascending=False).head(10)
    for _, row in top2.iterrows():
        log(
            f"  product={row.product_id}  price={row.current_price:.0f}  "
            f"subcat_median={row.subcat_median:.0f}  score={row.anomaly_score:.3f}"
        )

    log("\n--- Baseline comparison (Pearson correlation with Isolation Forest score) ---")
    m2_all = m2_all.assign(
        baseline1=[baseline_subcategory_zscore(r.current_price, r.subcat_median, r.subcat_std) for r in m2_all.itertuples()]
    )
    corr_m2 = np.corrcoef(m2_all["anomaly_score"], m2_all["baseline1"])[0, 1]
    log(f"  MODE 2: corr(IsolationForest score, subcategory-zscore baseline) = {corr_m2:.3f}")

    m1_all["baseline_own_history"] = [
        baseline_own_history_zscore(r.current_price, r.hist_mean, r.hist_std) for r in m1_all.itertuples()
    ]
    corr_m1_hist = np.corrcoef(m1_all["anomaly_score"], m1_all["baseline_own_history"])[0, 1]
    log(f"  MODE 1: corr(IsolationForest score, own-history-zscore baseline) = {corr_m1_hist:.3f}")

    log("\n  Interpretation: a HIGH correlation (>0.7) means the ML model mostly")
    log("  reproduces what the simple baseline already tells you. A MODERATE")
    log("  correlation (0.3-0.7) means the model agrees on the clearest cases but")
    log("  diverges on others -- i.e. it is using additional signal (e.g. combining")
    log("  multiple reference points, volatility, data-volume confidence) beyond")
    log("  the single-formula baseline.")

    log("\n--- Unseen-product generalization check (MODE 2, test-split products) ---")
    log("  Simulates a brand-new listing: product_id is withheld from the call,")
    log("  so the system cannot look up history it has never legitimately seen.")
    test_products = history[history.product_id.isin(split.test_ids)]
    sample_rows = test_products.groupby("product_id").tail(1)
    for _, row in sample_rows.iterrows():
        result = predict_product({
            "product_name": row["product_name"],
            "price": row["price"],
            "brand": row["brand"],
            "category": row["category"],
            "subcategory": row["subcategory"],
            "platform": row["platform"],
        })
        log(
            f"  {row.product_id}  price={row.price:.0f}  subcat={row.subcategory:26s}  "
            f"score={result['anomaly_score']:.3f}  ({result['classification']})"
        )

    with open(config.REPORTS_DIR / "evaluation_report.txt", "w") as f:
        f.write("\n".join(report_lines))

    return all_scored


if __name__ == "__main__":
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    evaluate()
