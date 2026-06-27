"""Standalone v2.3 training script.

Trains only the v2.3 model (symmetric pairs + proba aggregation + no weights)
using the same temporal split and historical features as v2.2. Skips all other
model training to run in a few minutes instead of the full week7 pipeline.

Usage:
    python3.11 notebooks/train_v2_3.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pairwise_trainer import (
    PairwiseModelTrainer,
    compute_historical_features,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")


def main() -> None:
    # --- Load labeled data ---
    labeled_path = os.path.join(DATA_DIR, "labeled_training_set.json")
    if not os.path.exists(labeled_path):
        print("ERROR: labeled_training_set.json not found. Run week5_data_labeling.py first.")
        sys.exit(1)

    print("Loading labeled training data...")
    with open(labeled_path) as f:
        labeled_data = json.load(f)
    print(f"  {len(labeled_data):,} scenarios")

    # --- Load cook logs and POS sales for historical features ---
    print("Loading cook logs and POS sales for historical features...")
    with open(os.path.join(DATA_DIR, "cook_logs.json")) as f:
        cook_logs = json.load(f)
    with open(os.path.join(DATA_DIR, "pos_sales.json")) as f:
        pos_sales = json.load(f)

    # --- Temporal split (same cutoff as v2.2) ---
    all_dates = sorted(set(s["features"]["date"] for s in labeled_data))
    split_idx = int(len(all_dates) * 0.67)
    cutoff_date = all_dates[split_idx]
    train_scenarios = [s for s in labeled_data if s["features"]["date"] < cutoff_date]
    test_scenarios  = [s for s in labeled_data if s["features"]["date"] >= cutoff_date]
    print(f"\nTemporal split: {cutoff_date} → train={len(train_scenarios):,}  test={len(test_scenarios):,}")

    # --- Load v2.2 results for comparison ---
    v22_report_path = os.path.join(OUTPUT_DIR, "v2_2_temporal_report.json")
    v22_test_acc = None
    if os.path.exists(v22_report_path):
        with open(v22_report_path) as f:
            v22_report = json.load(f)
        v22_test_acc = v22_report.get("temporal_split", {}).get("test_top1_accuracy")
        print(f"v2.2 holdout top-1 (baseline): {v22_test_acc}%")

    # --- Historical features from training period only (no leakage) ---
    print("\nComputing historical features from TRAIN period only...")
    hist_train = compute_historical_features(cook_logs, pos_sales, cutoff_date=cutoff_date)

    # =========================================================================
    # v2.3: symmetric pairs + proba aggregation + no weights
    # =========================================================================
    print("\n" + "=" * 60)
    print("  v2.3: SYMMETRIC PAIRS + PROBA AGGREGATION (no weights)")
    print("=" * 60)

    print("\nBuilding pairwise data (symmetric=True, use_weights=False)...")
    pw_v23 = PairwiseModelTrainer(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42,
        use_proba=True,
    )
    pw_v23.prepare_data(train_scenarios, hist_train, use_weights=False, symmetric=True)

    n = pw_v23.X.shape[0]
    label_dist = pw_v23.y.value_counts().to_dict()
    print(f"  Train pairs: {n:,}  balance: {100*label_dist.get(1,0)/n:.1f}% class-1")

    # Skip full 5-fold CV — 715K pairs × 5 folds × 200 trees is very slow.
    # The holdout top-1 is the metric we care about; use a single 20% val
    # split for a quick sanity check on pairwise accuracy.
    import numpy as np
    rng = np.random.default_rng(42)
    n_total = len(pw_v23.X)
    val_idx  = rng.choice(n_total, size=int(n_total * 0.2), replace=False)
    train_idx = np.setdiff1d(np.arange(n_total), val_idx)
    X_t, y_t = pw_v23.X.iloc[train_idx], pw_v23.y.iloc[train_idx]
    X_v, y_v = pw_v23.X.iloc[val_idx],   pw_v23.y.iloc[val_idx]

    print("Training on 80% split for quick pairwise validation...")
    pw_v23.model.fit(X_t, y_t)
    val_acc = (pw_v23.model.predict(X_v) == y_v).mean() * 100
    print(f"  Pairwise val accuracy (20% holdout): {val_acc:.1f}%")

    print("Training final model on ALL train pairs...")
    pw_v23.train()
    pw_v23.evaluate()

    # Top features
    print("\nTop 10 feature importances:")
    for feat, imp in list(pw_v23.feature_importance.items())[:10]:
        bar = "█" * int(imp * 40)
        print(f"  {feat:25s}: {imp:.4f} {bar}")

    print("\nEvaluating on holdout test set...")
    test_result = pw_v23.evaluate_ranking_accuracy(test_scenarios)
    full_result = pw_v23.evaluate_ranking_accuracy(labeled_data)

    test_acc = test_result["ranking_top1_accuracy"]
    print(f"  Test top-1:  {test_acc:.1f}%", end="")
    if v22_test_acc is not None:
        delta = test_acc - v22_test_acc
        print(f"  (v2.2: {v22_test_acc:.1f}%  Δ={delta:+.1f}pp)")
    else:
        print()
    print(f"  Full top-1:  {full_result['ranking_top1_accuracy']:.1f}%")

    # --- Save ---
    model_path = pw_v23.save_model(os.path.join(MODELS_DIR, "v2_3_pairwise_symmetric.pkl"))
    print(f"\nModel saved → {model_path}")

    pw_v23.report["improvements"] = {
        "symmetric_pairs": True,
        "use_proba_aggregation": True,
        "sample_weights": False,
        "n_estimators": 200,
        "note": (
            "Symmetric augmentation removes item-position bias and balances "
            "label distribution. Proba aggregation preserves confidence signal "
            "and avoids intransitive rank cycles. No weights gives edge/near-tie "
            "pairs equal contribution during training."
        ),
    }
    pw_v23.report["temporal_split"] = {
        "cutoff_date": cutoff_date,
        "train_scenarios": len(train_scenarios),
        "test_scenarios": len(test_scenarios),
        "pairwise_val_accuracy": round(val_acc, 1),
        "test_top1_accuracy": test_acc,
        "full_top1_accuracy": full_result["ranking_top1_accuracy"],
        "v22_test_top1_for_comparison": v22_test_acc,
        "delta_vs_v22": round(test_acc - v22_test_acc, 1) if v22_test_acc else None,
    }

    report_path = os.path.join(OUTPUT_DIR, "v2_3_symmetric_report.json")
    pw_v23.save_report(report_path)
    print(f"Report saved → {report_path}")

    print("\n" + "=" * 60)
    if v22_test_acc is not None:
        delta = test_acc - v22_test_acc
        print(f"  v2.3 holdout: {test_acc:.1f}%  |  v2.2 holdout: {v22_test_acc:.1f}%  |  Δ={delta:+.1f}pp")
        if delta >= 2:
            print("  ✅ Meaningful improvement — proceed to v0.3 diagnostic eval")
        elif delta >= 0:
            print("  ↗  Slight improvement — worth deploying")
        else:
            print("  ⬇  Regression on holdout — investigate before deploying")
    print("=" * 60)


if __name__ == "__main__":
    main()
