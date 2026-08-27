"""
Produces the four required demonstration predictions using real data only:

  A) One historical product -> historical mode -> score
  B) One completely unseen test product -> market mode -> score
  C) One normal-looking product -> low score
  D) One deliberately selected extreme-price test case -> high score

Run as: python -m ml.demo
"""
import json

from .preprocess import load_history, load_split_manifest
from .predict import predict_product


def main():
    history = load_history()
    split = load_split_manifest()
    counts = history.groupby("product_id").size()

    results = {}

    # --- A) Historical product with deep history, in TRAIN split ---
    train_deep = [pid for pid in split.train_ids if counts[pid] >= 30]
    pid_a = sorted(train_deep, key=lambda p: -counts[p])[0]
    row_a = history[history.product_id == pid_a].iloc[-1]
    print("=" * 78)
    print("A) HISTORICAL PRODUCT (in training set, deep history) -> MODE 1")
    print("=" * 78)
    print(f"Product: {row_a.product_name[:70]}")
    print(f"Brand: {row_a.brand} | Subcategory: {row_a.subcategory} | Observed history: {counts[pid_a]} checkpoints")
    result_a = predict_product({
        "product_id": pid_a,
        "product_name": row_a.product_name,
        "price": row_a.price,
        "brand": row_a.brand,
        "category": row_a.category,
        "subcategory": row_a.subcategory,
        "platform": row_a.platform,
    })
    print(json.dumps(result_a, indent=2))
    results["A_historical_product"] = result_a

    # --- B) Completely unseen TEST product, treated as a brand-new listing ---
    test_ids = split.test_ids
    pid_b = test_ids[0]
    row_b = history[history.product_id == pid_b].iloc[-1]
    print()
    print("=" * 78)
    print("B) COMPLETELY UNSEEN TEST PRODUCT (no product_id passed) -> MODE 2")
    print("=" * 78)
    print(f"Product: {row_b.product_name[:70]}  (product_id={pid_b} withheld from the call)")
    print(f"Brand: {row_b.brand} | Subcategory: {row_b.subcategory} | Actual price: {row_b.price}")
    result_b = predict_product({
        # product_id deliberately NOT passed -- simulates a genuinely new listing
        "product_name": row_b.product_name,
        "price": row_b.price,
        "brand": row_b.brand,
        "category": row_b.category,
        "subcategory": row_b.subcategory,
        "platform": row_b.platform,
    })
    print(json.dumps(result_b, indent=2))
    results["B_unseen_test_product"] = result_b

    # --- C) Normal-looking new product, priced near its category median ---
    print()
    print("=" * 78)
    print("C) NORMAL-LOOKING NEW PRODUCT (priced near category median) -> LOW SCORE")
    print("=" * 78)
    result_c = predict_product({
        "product_name": "Unbranded Wireless Earbuds (new listing)",
        "price": 1049,  # close to Wireless Earphones/TWS category median of ~1499, inside IQR
        "brand": "SomeNewBrand",
        "category": "Headphones & Earbuds",
        "subcategory": "Wireless Earphones / TWS",
        "platform": "amazon",
    })
    print(json.dumps(result_c, indent=2))
    results["C_normal_new_product"] = result_c

    # --- D) Deliberately extreme-price test case ---
    print()
    print("=" * 78)
    print("D) DELIBERATELY EXTREME-PRICE PRODUCT -> HIGH SCORE")
    print("=" * 78)
    result_d = predict_product({
        "product_name": "Unbranded Wireless Earbuds (suspiciously priced)",
        "price": 29999,  # far above category p95 (~9,949) with an unknown brand
        "brand": "SomeNewBrand",
        "category": "Headphones & Earbuds",
        "subcategory": "Wireless Earphones / TWS",
        "platform": "amazon",
    })
    print(json.dumps(result_d, indent=2))
    results["D_extreme_price_product"] = result_d

    with open("reports/demo_predictions.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    main()
