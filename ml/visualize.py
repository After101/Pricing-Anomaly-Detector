"""
Generates the six required plots under reports/:
  1. Price distributions by subcategory
  2. Anomaly score distribution (train, by mode)
  3. Example historical product with its price history
  4. Example high-anomaly product
  5. Example normal product
  6. Baseline vs Isolation Forest comparison

Run as: python -m ml.visualize
"""
from __future__ import annotations

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .preprocess import load_history, load_split_manifest
from .category_stats import load_stats
from .features import build_training_rows, MODE1_FEATURE_NAMES, MODE2_FEATURE_NAMES
from .baselines import baseline_subcategory_zscore

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "font.size": 10,
    "figure.dpi": 130,
})

COLORS = {
    "Wireless Earphones / TWS": "#2563eb",
    "Headphones": "#16a34a",
    "Earphones": "#d97706",
}


def plot_price_distributions(history: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    subcats = history["subcategory"].unique()
    data = [history.loc[history.subcategory == s, "price"].values for s in subcats]
    bp = ax.boxplot(data, labels=subcats, showfliers=True, patch_artist=True)
    for patch, s in zip(bp["boxes"], subcats):
        patch.set_facecolor(COLORS.get(s, "#999999"))
        patch.set_alpha(0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Price (\u20b9, log scale)")
    ax.set_title("Price Distribution by Subcategory\n(all 6,820 observations)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(config.REPORTS_DIR / "01_price_distribution_by_subcategory.png")
    plt.close(fig)


def plot_anomaly_score_distribution(history, split, subcat_stats, brand_stats):
    mode1_model = joblib.load(config.MODEL_MODE1_PATH)
    mode2_model = joblib.load(config.MODEL_MODE2_PATH)

    m1, m2 = build_training_rows(history, split.train_ids, subcat_stats, brand_stats)
    s1 = mode1_model.anomaly_score(m1[MODE1_FEATURE_NAMES].to_numpy(dtype=float))
    s2 = mode2_model.anomaly_score(m2[MODE2_FEATURE_NAMES].to_numpy(dtype=float))
    m1 = m1.assign(anomaly_score=s1)
    m2 = m2.assign(anomaly_score=s2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(s1, bins=40, color="#2563eb", alpha=0.75)
    axes[0].axvline(np.median(s1), color="black", linestyle="--", linewidth=1, label=f"median={np.median(s1):.2f}")
    axes[0].set_title("MODE 1 (Historical Product)\nAnomaly Score Distribution -- Train")
    axes[0].set_xlabel("Pricing Anomaly Score")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].hist(s2, bins=40, color="#16a34a", alpha=0.75)
    axes[1].axvline(np.median(s2), color="black", linestyle="--", linewidth=1, label=f"median={np.median(s2):.2f}")
    axes[1].set_title("MODE 2 (Market/Category)\nAnomaly Score Distribution -- Train")
    axes[1].set_xlabel("Pricing Anomaly Score")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(config.REPORTS_DIR / "02_anomaly_score_distribution.png")
    plt.close(fig)
    return m1, m2, s1, s2


def plot_example_historical_product(history, split, counts):
    deep = [pid for pid in split.train_ids if counts[pid] >= config.MIN_HISTORY_FOR_MODE1]
    pid = sorted(deep, key=lambda p: -counts[p])[0]
    group = history[history.product_id == pid].sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(group["timestamp"], group["price"], marker="o", markersize=2, linewidth=1, color="#2563eb")
    ax.axhline(group["price"].median(), color="black", linestyle="--", linewidth=1, label=f"median = \u20b9{group['price'].median():.0f}")
    ax.set_title(f"Example Historical Product Price Trend\n{group['product_name'].iloc[-1][:65]}")
    ax.set_ylabel("Price (\u20b9)")
    ax.set_xlabel("Date")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(config.REPORTS_DIR / "03_example_historical_product_price_trend.png")
    plt.close(fig)
    return pid


def plot_high_and_normal_anomaly_examples(history, subcat_stats):
    from .predict import predict_product

    # High-anomaly: real product (Apple AirPods Pro) treated as a new listing.
    high_pid = "ACCG7XG5ZQ47ZHFR"
    high_row = history[history.product_id == high_pid].iloc[-1]
    high_result = predict_product({
        "product_name": high_row.product_name,
        "price": high_row.price,
        "brand": high_row.brand,
        "subcategory": high_row.subcategory,
    })

    # Normal example: price near category median, new listing.
    normal_subcat = "Wireless Earphones / TWS"
    normal_price = subcat_stats[normal_subcat]["median"]
    normal_result = predict_product({
        "product_name": "Example normally-priced product",
        "price": normal_price,
        "subcategory": normal_subcat,
    })

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, result, title, color in [
        (axes[0], normal_result, "Normal Example\n(priced at category median)", "#16a34a"),
        (axes[1], high_result, f"High-Anomaly Example\n{high_row.product_name[:40]}", "#dc2626"),
    ]:
        ax.barh(["Anomaly\nScore"], [result["anomaly_score"]], color=color)
        ax.set_xlim(0, 1)
        ax.set_title(f"{title}\nprice=\u20b9{result['current_price']:.0f}  score={result['anomaly_score']:.2f}")
        ax.text(result["anomaly_score"] + 0.02, 0, f"{result['anomaly_score']:.2f}", va="center")

    plt.tight_layout()
    fig.savefig(config.REPORTS_DIR / "04_05_normal_vs_high_anomaly_examples.png")
    plt.close(fig)
    return normal_result, high_result


def plot_baseline_vs_isolation_forest(m2_scored):
    m2_scored = m2_scored.copy()
    m2_scored["baseline"] = [
        baseline_subcategory_zscore(r.current_price, r.subcat_median, r.subcat_std)
        for r in m2_scored.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(m2_scored["baseline"], m2_scored["anomaly_score"], alpha=0.2, s=10, color="#2563eb")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="y = x (perfect agreement)")
    corr = np.corrcoef(m2_scored["baseline"], m2_scored["anomaly_score"])[0, 1]
    ax.set_xlabel("Baseline: subcategory z-score (0-1)")
    ax.set_ylabel("Isolation Forest anomaly score (0-1)")
    ax.set_title(f"MODE 2: Isolation Forest vs. Simple Baseline\n(Pearson r = {corr:.2f}, train rows)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(config.REPORTS_DIR / "06_baseline_vs_isolation_forest.png")
    plt.close(fig)
    return corr


def main():
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    history = load_history()
    split = load_split_manifest()
    counts = history.groupby("product_id").size()
    subcat_stats, brand_stats = load_stats()

    print("Plot 1: price distributions by subcategory...")
    plot_price_distributions(history)

    print("Plot 2: anomaly score distribution...")
    m1, m2, s1, s2 = plot_anomaly_score_distribution(history, split, subcat_stats, brand_stats)

    print("Plot 3: example historical product price trend...")
    plot_example_historical_product(history, split, counts)

    print("Plots 4-5: normal vs high-anomaly examples...")
    plot_high_and_normal_anomaly_examples(history, subcat_stats)

    print("Plot 6: baseline vs isolation forest...")
    corr = plot_baseline_vs_isolation_forest(m2)

    print(f"\nAll plots saved under {config.REPORTS_DIR}")
    print(f"Baseline-vs-model correlation (Mode 2, train): {corr:.3f}")


if __name__ == "__main__":
    main()
