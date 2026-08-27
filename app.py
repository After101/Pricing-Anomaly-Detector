"""
Pricing Anomaly Detector -- Streamlit UI.

This file is PRESENTATION ONLY. It never trains, retrains, or re-derives
any scoring logic. All inference goes through the existing, already-trained
pipeline:

    ml.predict.predict_product(product_data) -> structured result dict

Run from the project root with:

    streamlit run app.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ml import config
from ml.predict import predict_product, InvalidProductInputError, ArtifactsNotFoundError

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pricing Anomaly Detector",
    page_icon="\U0001F4CA",
    layout="centered",
)

DISCLAIMER = (
    "Pricing Anomaly Score indicates how unusual the current price is relative "
    "to the available pricing reference. It is not a probability of fraud or a "
    "guarantee that a product is incorrectly priced."
)

KNOWN_SUBCATEGORIES = ["Wireless Earphones / TWS", "Headphones", "Earphones"]
PLATFORMS = ["Amazon", "Flipkart", "Other"]


# ---------------------------------------------------------------------------
# Cached loaders (read-only access to existing artifacts -- no training)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_product_history() -> dict:
    if not config.PRODUCT_HISTORY_PATH.exists():
        return {}
    with open(config.PRODUCT_HISTORY_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_category_stats() -> dict:
    if not config.CATEGORY_STATS_PATH.exists():
        return {}
    with open(config.CATEGORY_STATS_PATH) as f:
        return json.load(f)


def artifacts_available() -> bool:
    required = [
        config.MODEL_MODE1_PATH,
        config.MODEL_MODE2_PATH,
        config.CATEGORY_STATS_PATH,
        config.BRAND_STATS_PATH,
        config.PRODUCT_HISTORY_PATH,
    ]
    return all(p.exists() for p in required)


# ---------------------------------------------------------------------------
# Safe wrapper around predict_product -- the ONLY place errors are caught
# ---------------------------------------------------------------------------
def run_prediction(product_data: dict):
    """Returns (result_dict, error_message) -- exactly one is None."""
    try:
        result = predict_product(product_data)
        return result, None
    except InvalidProductInputError as e:
        return None, str(e)
    except ArtifactsNotFoundError as e:
        return None, f"The trained model could not be loaded: {e}"
    except Exception as e:  # noqa: BLE001 -- deliberate catch-all at the UI boundary
        return None, "Something went wrong while analyzing this product. Please check your inputs and try again."


# ---------------------------------------------------------------------------
# Amazon URL handling (scraper intentionally decoupled from the model)
# ---------------------------------------------------------------------------
AMAZON_URL_RE = re.compile(r"^https?://(www\.)?amazon\.(in|com|co\.uk|de|ca)/", re.I)


def try_scrape_amazon(url: str):
    """Looks for a real Amazon scraper in the project. None is bundled with
    this project (only a PriceDiff historical-checkpoint scraper exists,
    which talks to pricediff.in, not Amazon), so this always reports
    unavailable rather than fabricating scraped data. If a real scraper
    module is added later (e.g. ml/amazon_scraper.py exposing
    scrape_amazon_product(url) -> dict), wiring it in here is the only
    change needed -- the rest of the app is unaffected.
    """
    try:
        from ml.amazon_scraper import scrape_amazon_product  # type: ignore
    except ImportError:
        return None, "No Amazon scraper is available in this build."

    try:
        data = scrape_amazon_product(url)
        return data, None
    except Exception as e:  # noqa: BLE001
        return None, f"The Amazon scraper failed: {e}"


# ---------------------------------------------------------------------------
# Result rendering (uses ONLY what predict_product() returned)
# ---------------------------------------------------------------------------
def render_result(result: dict, product_name: str | None, product_id: str | None):
    st.markdown("---")

    # --- Product header ---
    st.subheader(product_name or "Submitted Product")
    st.metric("Current Price", f"\u20b9{result['current_price']:,.0f}")

    # --- Score ---
    score = result["anomaly_score"]
    st.markdown("### Pricing Anomaly Score")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.progress(min(max(score, 0.0), 1.0))
    with col2:
        st.markdown(f"## {score:.3f}")
    st.caption("0 = relatively typical pricing \u00b7 1 = highly unusual pricing")

    classification = result["classification"]
    if score >= 0.90:
        st.error(f"**{classification}**")
    elif score >= 0.75:
        st.warning(f"**{classification}**")
    elif score >= 0.50:
        st.info(f"**{classification}**")
    else:
        st.success(f"**{classification}**")

    # --- Analysis mode ---
    mode = result["mode"]
    st.markdown("### Analysis Mode")
    if mode == "historical":
        st.markdown("**Historical Product Analysis**")
        st.caption(
            "This product has sufficient historical observations, so its current "
            "price is compared with its own observed pricing behaviour."
        )
    else:
        st.markdown("**Market / Category Analysis**")
        st.caption(
            "This product has insufficient or no historical observations, so its "
            "current price is compared with the observed pricing distribution of its "
            "market subcategory."
        )

    # --- Warnings from predict_product (e.g. unknown product_id, insufficient history) ---
    if result.get("warnings"):
        with st.expander("Notes"):
            for w in result["warnings"]:
                st.write(f"- {w}")

    # --- Reference values (exactly what predict_product returned; no fabrication) ---
    ref = result["reference_values"]
    st.markdown("### Reference Values")
    ref_display = {
        "subcategory": "Subcategory",
        "historical_n_obs": "Historical Observations",
        "historical_median": "Historical Median",
        "historical_mean": "Historical Mean",
        "historical_min": "Historical Minimum",
        "historical_max": "Historical Maximum",
        "historical_std": "Historical Std. Deviation",
        "category_median": "Category Median",
        "category_mean": "Category Mean",
        "category_std": "Category Std. Deviation",
        "category_p25": "Category 25th Percentile",
        "category_p75": "Category 75th Percentile",
        "category_p95": "Category 95th Percentile",
        "category_n_obs": "Category Observations",
        "category_n_products": "Category Products",
        "brand_median_in_category": "Brand Median (in category)",
        "brand_n_products": "Brand Products (in category)",
    }
    rows = []
    currency_keys = {
        "historical_median", "historical_mean", "historical_min", "historical_max", "historical_std",
        "category_median", "category_mean", "category_std", "category_p25", "category_p75", "category_p95",
        "brand_median_in_category",
    }
    for key, label in ref_display.items():
        if key in ref and ref[key] is not None:
            val = ref[key]
            if key in currency_keys:
                rows.append((label, f"\u20b9{float(val):,.0f}"))
            else:
                rows.append((label, f"{val}"))
    for label, val in rows:
        c1, c2 = st.columns([2, 1])
        c1.write(label)
        c2.write(val)

    # --- Why this score? ---
    st.markdown("### Why This Score?")
    for r in result.get("reasons", []):
        st.write(f"- {r}")

    # --- Visualizations ---
    if mode == "historical" and product_id:
        render_historical_chart(product_id, result["current_price"])
    elif mode == "market":
        render_market_visualization(ref, result["current_price"])

    # --- Disclaimer ---
    st.markdown("---")
    st.caption(DISCLAIMER)


def render_historical_chart(product_id: str, current_price: float):
    history_lookup = load_product_history()
    record = history_lookup.get(product_id)
    if not record or "timestamps" not in record or not record["prices"]:
        return  # no real historical data available -- don't fabricate a chart

    st.markdown("### Historical Price Trend")
    dates = [datetime.fromisoformat(ts) for ts in record["timestamps"]]
    prices = record["prices"]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(dates, prices, marker="o", markersize=2, linewidth=1, color="#2563eb", label="Historical price")
    ax.scatter([dates[-1]], [current_price], color="#dc2626", s=60, zorder=5, label="Current price entered")
    ax.set_ylabel("Price (\u20b9)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_market_visualization(ref: dict, current_price: float):
    required = ["category_p25", "category_median", "category_p75", "category_p95"]
    if not all(k in ref and ref[k] is not None for k in required):
        return  # don't fabricate a chart from missing values

    st.markdown("### Market Context")
    p25, median, p75, p95 = ref["category_p25"], ref["category_median"], ref["category_p75"], ref["category_p95"]
    points = [("P25", p25), ("Median", median), ("P75", p75), ("P95", p95), ("Current", current_price)]

    fig, ax = plt.subplots(figsize=(7, 1.8))
    xs = [v for _, v in points]
    x_min, x_max = min(xs) * 0.8, max(xs) * 1.15
    ax.set_xlim(x_min, x_max)
    ax.set_xscale("log")
    ax.axhline(0, color="#cccccc", linewidth=1, zorder=1)

    for label, value in points[:4]:
        ax.scatter([value], [0], color="#2563eb", s=40, zorder=3)
        ax.annotate(f"{label}\n\u20b9{value:,.0f}", (value, 0), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8)

    current_color = "#dc2626" if current_price > p95 or current_price < p25 else "#16a34a"
    ax.scatter([current_price], [0], color=current_color, s=90, zorder=4, marker="D")
    ax.annotate(f"Current\n\u20b9{current_price:,.0f}", (current_price, 0), textcoords="offset points",
                xytext=(0, -28), ha="center", fontsize=8, fontweight="bold", color=current_color)

    ax.set_yticks([])
    ax.set_xlabel("Price (\u20b9, log scale)", fontsize=8)
    plt.xticks(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Prices near the extremes of the observed category distribution may warrant "
        "closer inspection. The anomaly score combines multiple pricing characteristics "
        "and should not be interpreted as a direct measure of fairness or fraud."
    )


# ---------------------------------------------------------------------------
# Demo examples (real data only, real inference calls)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_demo_examples():
    """Builds three demo cases from real dataset entries. Each one calls the
    real predict_product() pipeline when selected -- nothing here is a
    pre-baked result.
    """
    history_lookup = load_product_history()
    examples = {}

    # 1) Known historical product, typical current price (its own last observed price).
    deep = [(pid, r) for pid, r in history_lookup.items() if r["n_obs"] >= config.MIN_HISTORY_FOR_MODE1]
    if deep:
        pid, r = sorted(deep, key=lambda kv: -kv[1]["n_obs"])[0]
        examples["Known product \u2014 price near its historical median (Historical mode)"] = {
            "product_id": pid,
            "product_name": r["product_name"],
            "price": r["prices"][-1],
            "brand": r["brand"],
            "category": r["category"],
            "subcategory": r["subcategory"],
            "platform": "Amazon",
        }

    # 2) Unseen product, typical market pricing (product_id withheld on purpose).
    shallow = [(pid, r) for pid, r in history_lookup.items() if r["n_obs"] < config.MIN_HISTORY_FOR_MODE1]
    typical_pick = None
    for pid, r in shallow:
        if r["subcategory"] == "Wireless Earphones / TWS" and 800 <= r["prices"][-1] <= 2000:
            typical_pick = (pid, r)
            break
    if typical_pick:
        pid, r = typical_pick
        examples["Unseen product \u2014 typical market pricing (Market mode)"] = {
            "product_id": None,  # withheld deliberately -- simulates a brand-new listing
            "product_name": r["product_name"] + " (treated as a new listing)",
            "price": r["prices"][-1],
            "brand": r["brand"],
            "category": r["category"],
            "subcategory": r["subcategory"],
            "platform": "Amazon",
        }

    # 3) Unseen/extreme product, highly unusual market pricing.
    extreme_pick = None
    for pid, r in history_lookup.items():
        if r["subcategory"] == "Wireless Earphones / TWS" and r["prices"][-1] > 15000:
            extreme_pick = (pid, r)
            break
    if extreme_pick:
        pid, r = extreme_pick
        examples["Unseen product \u2014 highly unusual market pricing (Market mode)"] = {
            "product_id": None,  # withheld deliberately
            "product_name": r["product_name"] + " (treated as a new listing)",
            "price": r["prices"][-1],
            "brand": r["brand"],
            "category": r["category"],
            "subcategory": r["subcategory"],
            "platform": "Amazon",
        }

    return examples


# ---------------------------------------------------------------------------
# Session state defaults for the manual-input form
# ---------------------------------------------------------------------------
DEFAULT_FORM = {
    "product_name": "",
    "price": None,
    "brand": "",
    "subcategory": "-- Select --",
    "platform": "Amazon",
    "product_id": "",
    "url": "",
}
for k, v in DEFAULT_FORM.items():
    st.session_state.setdefault(f"form_{k}", v)


def apply_demo_to_form(example: dict):
    st.session_state["form_product_name"] = example.get("product_name") or ""
    st.session_state["form_price"] = float(example["price"])
    st.session_state["form_brand"] = example.get("brand") or ""
    st.session_state["form_subcategory"] = example.get("subcategory") or "-- Select --"
    st.session_state["form_platform"] = example.get("platform") or "Amazon"
    st.session_state["form_product_id"] = example.get("product_id") or ""
    st.session_state["form_url"] = ""


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("Pricing Anomaly Detector")
st.write(
    "Analyze an electronics product's pricing behaviour using historical and "
    "market data. Covers Wireless Earphones / TWS, Headphones, and Earphones."
)

if not artifacts_available():
    st.error(
        "The trained model artifacts were not found under `ml/artifacts/`. "
        "Run `python -m ml.train` from the project root first, then restart this app."
    )
    st.stop()

tab_amazon, tab_manual = st.tabs(["Amazon Product", "Manual Input"])

# --- TAB 1: Amazon Product -------------------------------------------------
with tab_amazon:
    st.write("Paste an Amazon product URL to analyze its pricing.")
    amazon_url = st.text_input("Amazon product URL", key="amazon_url_input")
    analyze_amazon = st.button("Analyze", key="analyze_amazon_btn")

    if analyze_amazon:
        if not amazon_url or not amazon_url.strip():
            st.warning("Please enter a product URL.")
        elif not AMAZON_URL_RE.match(amazon_url.strip()):
            st.warning(
                "That doesn't look like a valid Amazon product URL "
                "(expected something like https://www.amazon.in/...). "
                "You can also use the **Manual Input** tab instead."
            )
        else:
            scraped_data, scrape_error = try_scrape_amazon(amazon_url.strip())
            if scraped_data is None:
                st.info(
                    f"{scrape_error} Please use the **Manual Input** tab to enter "
                    "the product's details directly -- the pricing model works "
                    "identically either way."
                )
            else:
                result, error = run_prediction(scraped_data)
                if error:
                    st.error(error)
                else:
                    render_result(result, scraped_data.get("product_name"), scraped_data.get("product_id"))

# --- TAB 2: Manual Input -----------------------------------------------------
with tab_manual:
    demo_examples = build_demo_examples()
    if demo_examples:
        st.write("**Load a demo example** (built from real dataset entries):")
        demo_choice = st.selectbox(
            "Demo example", ["-- None --"] + list(demo_examples.keys()), key="demo_select"
        )
        if st.button("Load Demo Example", key="load_demo_btn") and demo_choice != "-- None --":
            apply_demo_to_form(demo_examples[demo_choice])
            st.rerun()
        st.markdown("---")

    st.write("Or enter product details manually:")

    product_name = st.text_input("Product name", key="form_product_name")
    price = st.number_input(
        "Current price (\u20b9)", min_value=0.0, step=1.0, key="form_price", format="%.2f"
    )
    brand = st.text_input("Brand (optional)", key="form_brand")
    subcategory = st.selectbox(
        "Subcategory", ["-- Select --"] + KNOWN_SUBCATEGORIES, key="form_subcategory"
    )
    platform = st.selectbox("Platform (optional)", PLATFORMS, key="form_platform")
    product_id = st.text_input("Product ID (optional)", key="form_product_id")
    url = st.text_input("Product URL (optional)", key="form_url")

    analyze_manual = st.button("Analyze", key="analyze_manual_btn", type="primary")

    if analyze_manual:
        errors = []
        if not product_name or not product_name.strip():
            errors.append("Please enter a product name.")
        if price is None or price <= 0:
            errors.append("Please enter a valid, positive current price.")
        if subcategory == "-- Select --":
            errors.append("Please select a subcategory.")

        if errors:
            for e in errors:
                st.warning(e)
        else:
            product_data = {
                "product_id": product_id.strip() or None,
                "product_name": product_name.strip(),
                "price": price,
                "brand": brand.strip() or None,
                "category": "Headphones & Earbuds",
                "subcategory": subcategory,
                "platform": platform,
                "url": url.strip() or None,
            }
            result, error = run_prediction(product_data)
            if error:
                st.error(error)
            else:
                render_result(result, product_name.strip(), product_id.strip() or None)

st.markdown("---")
st.caption("This is a demonstration project for educational purposes.")
