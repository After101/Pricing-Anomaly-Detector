"""
Central configuration for the pricing-anomaly-detection pipeline.

All thresholds are documented here with the reasoning behind them so that
anyone reviewing the project can see exactly why a number was chosen,
rather than a number appearing "magically" inside feature code.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

HISTORY_CSV = DATA_DIR / "pricediff_history.csv"
PRODUCTS_CSV = DATA_DIR / "pricediff_products.csv"

MODEL_MODE1_PATH = ARTIFACTS_DIR / "pricing_anomaly_model_mode1.joblib"
MODEL_MODE2_PATH = ARTIFACTS_DIR / "pricing_anomaly_model_mode2.joblib"
# Kept for compatibility with the exact filename requested in the spec.
# This holds the market/category (Mode 2) model, which is the model every
# submitted product can always fall back to.
MODEL_PRIMARY_PATH = ARTIFACTS_DIR / "pricing_anomaly_model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "pricing_preprocessor.joblib"
MODEL_FEATURES_PATH = ARTIFACTS_DIR / "model_features.json"
CATEGORY_STATS_PATH = ARTIFACTS_DIR / "category_stats.json"
BRAND_STATS_PATH = ARTIFACTS_DIR / "brand_stats.json"
PRODUCT_HISTORY_PATH = ARTIFACTS_DIR / "product_history.json"
CALIBRATION_PATH = ARTIFACTS_DIR / "score_calibration.joblib"
SPLIT_MANIFEST_PATH = ARTIFACTS_DIR / "train_val_test_products.json"

# ---------------------------------------------------------------------------
# Historical-depth threshold (Mode 1 vs Mode 2 routing)
# ---------------------------------------------------------------------------
# The observation-count histogram of this dataset has a sharp natural gap:
#   24 products have between 50 and 384 observations
#   30 products have between 1 and 5 observations
# There is nothing between 5 and 50. We set the Mode-1 threshold at 30
# observations of *prior* history (i.e. history available strictly before
# the price point being scored) because:
#   - it sits inside the natural gap (so the choice isn't sensitive to
#     small changes in the exact cutoff)
#   - 30 observations is enough to get a reasonably stable mean/median/std
#     estimate for a single product's own price behaviour
# This constant is used both when constructing point-in-time training rows
# (leakage-safe) and at inference time.
MIN_HISTORY_FOR_MODE1 = 30

# ---------------------------------------------------------------------------
# Brand-within-subcategory statistics
# ---------------------------------------------------------------------------
# We only build a brand-specific price distribution inside a subcategory if
# there are at least this many training-set observations for that
# brand+subcategory pair. Below this we consider the sample too small to be
# a meaningful distribution (e.g. a "Sony Headphones" distribution built
# from a single Sony product would just describe that one listing, not
# "Sony's pricing", so we fall back to subcategory-level statistics).
MIN_OBS_FOR_BRAND_STATS = 30

# In addition to a minimum observation count, we require a minimum number
# of DISTINCT products for that brand within the subcategory. This is the
# rule that actually matters most in this dataset: several brands (Sony,
# Portronics, Bigben, ZEBRONICS...) have only ONE product each, tracked
# for hundreds of checkpoints. Without this check, "Sony's pricing
# distribution" would really just be "this one Sony SKU's price history
# repeated many times", which is exactly the kind of misleading,
# meaningful-looking-but-empty statistic the project spec warns against.
MIN_PRODUCTS_FOR_BRAND_STATS = 3

# ---------------------------------------------------------------------------
# Brand-name normalization
# ---------------------------------------------------------------------------
# The raw scrape contains case-inconsistent brand strings for the same
# real-world brand (e.g. "Boat" vs "boAt", "OnePlus" vs "Oneplus"). We
# normalize to a canonical capitalization before computing any brand-level
# statistics. This map is intentionally explicit and small — we do NOT
# attempt fuzzy/automatic brand matching, since that risks silently merging
# genuinely different brands.
BRAND_NORMALIZATION = {
    "boat": "boAt",
    "oneplus": "OnePlus",
}


def normalize_brand(raw_brand: str) -> str:
    if raw_brand is None:
        return "Unknown"
    key = str(raw_brand).strip().lower()
    return BRAND_NORMALIZATION.get(key, str(raw_brand).strip())


# ---------------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------------
TRAIN_FRACTION = 0.75
VAL_FRACTION = 0.10
TEST_FRACTION = 0.15
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Isolation Forest hyperparameters
# ---------------------------------------------------------------------------
# `contamination` is treated as an *assumption about how much of the
# training population is unusual*, not a validated fraud rate. 5% is a
# conventional default for anomaly-detection tasks with no labels; it only
# affects the model's internal decision boundary between "raw normal" and
# "raw anomalous" -- the 0-1 score reported to the user comes from a
# separate, explicit percentile-rank calibration (see scoring.py), so this
# value does not need to be exact.
ISOLATION_FOREST_PARAMS = dict(
    n_estimators=200,
    max_samples="auto",
    contamination=0.05,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

# ---------------------------------------------------------------------------
# Column exclusions
# ---------------------------------------------------------------------------
# Columns that must NEVER be fed into the ML feature matrix, because they
# are identifiers/free text/urls, not price-relevant signal, and including
# them (especially product_id) would let a model "memorize" a product
# instead of learning pricing behaviour.
EXCLUDED_RAW_COLUMNS = {
    "product_id",
    "product_name",
    "product_url",
    "pricediff_url",
    "price_history_id",
    "timestamp",
    "currency",
    "mrp",
    "discount_percent",
}
