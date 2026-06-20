"""Week 7: v2 Model Training Pipeline.

This script:
1. Loads the labeled training set from Week 5-6.
2. Separates into "informative" scenarios (items have different waste outcomes)
   and "tiebreaker" scenarios (all items had the same waste — label is heuristic).
3. Trains a RandomForest classifier on all data.
4. Reports accuracy on both full and informative-only subsets.
5. Saves model, feature importances, and evaluation report.
"""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_trainer import ModelTrainer, build_feature_matrix
from src.pairwise_trainer import PairwiseModelTrainer, compute_historical_features
from src.cook_scheduler import AssociateBaseline


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    output_dir = os.path.join(project_root, "output")
    models_dir = os.path.join(project_root, "models")

    # --- Load labeled data ---
    labeled_path = os.path.join(data_dir, "labeled_training_set.json")
    if not os.path.exists(labeled_path):
        print("ERROR: labeled_training_set.json not found. Run week5_data_labeling.py first.")
        sys.exit(1)

    print("Loading labeled training data...")
    with open(labeled_path) as f:
        labeled_data = json.load(f)

    informative = [s for s in labeled_data if s.get("informative", True)]
    tiebreaker = [s for s in labeled_data if not s.get("informative", True)]

    print(f"  Total scenarios:       {len(labeled_data):,}")
    print(f"  Informative (signal):  {len(informative):,} ({100*len(informative)/len(labeled_data):.1f}%)")
    print(f"  Tiebreaker (noise):    {len(tiebreaker):,} ({100*len(tiebreaker)/len(labeled_data):.1f}%)")

    # --- Train on ALL data (model learns both signal and heuristic patterns) ---
    print("\n--- Training on ALL scenarios ---")
    trainer = ModelTrainer(n_estimators=200, max_depth=12, random_state=42)
    trainer.prepare_data(labeled_data)
    print(f"  Feature matrix: {trainer.X.shape[0]} samples × {trainer.X.shape[1]} features")

    print("  Running 5-fold cross-validation...")
    cv_scores = trainer.cross_validate(cv=5)
    print(f"  CV Mean accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    print("  Training final model...")
    trainer.train()
    trainer.evaluate()

    # --- Evaluate on INFORMATIVE scenarios only (the meaningful metric) ---
    print("\n--- Evaluating on INFORMATIVE scenarios only ---")
    if len(informative) >= 10:
        trainer_info = ModelTrainer(n_estimators=200, max_depth=12, random_state=42)
        trainer_info.prepare_data(informative)
        print(f"  Feature matrix: {trainer_info.X.shape[0]} samples × {trainer_info.X.shape[1]} features")

        print("  Running 5-fold cross-validation...")
        cv_info = trainer_info.cross_validate(cv=5)
        print(f"  CV Mean accuracy: {cv_info.mean()*100:.1f}% ± {cv_info.std()*100:.1f}%")

        print("  Training final model...")
        trainer_info.train()
        trainer_info.evaluate()
    else:
        print("  Not enough informative scenarios for separate training.")
        trainer_info = None

    # --- Print full results ---
    trainer.print_summary()

    if trainer_info:
        print("\n" + "=" * 60)
        print("  INFORMATIVE-ONLY MODEL RESULTS")
        print("=" * 60)
        r = trainer_info.report
        print(f"\n  Samples: {r['training_set']['n_samples']:,}")
        print(f"  CV accuracy: {r['cross_validation']['mean_accuracy']:.1f}% "
              f"± {r['cross_validation']['std_accuracy']:.1f}%")
        print(f"  Training accuracy: {r['accuracy']:.1f}%")
        print(f"\n  Top 5 features:")
        for feat, imp in list(r["feature_importance_top10"].items())[:5]:
            print(f"    {feat:30s}: {imp:.4f}")

    # --- Save outputs (use the all-data model as primary) ---
    model_path = trainer.save_model(os.path.join(models_dir, "v2_ranking_model.pkl"))
    print(f"\nModel saved to: {model_path}")

    report_path = trainer.save_report(os.path.join(output_dir, "v2_training_report.json"))
    print(f"Training report saved to: {report_path}")

    fi_path = os.path.join(output_dir, "feature_importance.json")
    with open(fi_path, "w") as f:
        json.dump(trainer.feature_importance, f, indent=2)
    print(f"Feature importance saved to: {fi_path}")

    # --- Evaluate associate baseline ---
    print("\n--- Evaluating Associate Baseline ---")
    associate = AssociateBaseline(seed=42)
    associate_correct = 0
    for scenario in labeled_data:
        features = scenario["features"]
        expected_first = scenario["optimal_first_item"]
        # Build oven_events-like dicts from scenario features
        oven_events = []
        for item in ["pizza", "wings_2h", "wings_4h", "baked_goods"]:
            if f"{item}_forecast_demand" in features:
                oven_events.append({
                    "item": item,
                    "forecast_demand": features[f"{item}_forecast_demand"],
                    "hold_time_hours": features[f"{item}_hold_time"],
                    "time_remaining": features[f"{item}_time_remaining"],
                })
        pick = associate.pick_first_item(oven_events)
        if pick == expected_first:
            associate_correct += 1
    associate_accuracy = 100 * associate_correct / len(labeled_data)
    print(f"  Associate baseline accuracy: {associate_accuracy:.1f}%")
    print(f"  (Simulates: 40% expiration, 30% habit/pizza, 20% random, 10% demand)")

    # --- Compare v1 vs v2 ---
    print("\n" + "=" * 60)
    print("  BASELINE vs v1 vs v2 COMPARISON")
    print("=" * 60)

    labeling_report_path = os.path.join(output_dir, "labeling_report.json")
    if os.path.exists(labeling_report_path):
        with open(labeling_report_path) as f:
            labeling_report = json.load(f)
        v1_accuracy = labeling_report.get("v1_agreement_pct", 0)
    else:
        v1_accuracy = 65.5

    v2_all_acc = trainer.report["cross_validation"]["mean_accuracy"]
    v2_info_acc = (trainer_info.report["cross_validation"]["mean_accuracy"]
                   if trainer_info else v2_all_acc)

    print(f"\n  Associate baseline (current state):       {associate_accuracy:.1f}%")
    print(f"  v1 accuracy (rule-based heuristic):       {v1_accuracy:.1f}%")
    print(f"  v2 accuracy (ML, all scenarios):          {v2_all_acc:.1f}%")
    print(f"  v2 accuracy (ML, informative only):       {v2_info_acc:.1f}%")

    print(f"\n  v2 vs associate baseline: +{v2_all_acc - associate_accuracy:.1f} pp")
    print(f"  v2 vs v1 heuristic:       +{v2_all_acc - v1_accuracy:.1f} pp")

    print(f"\n  On INFORMATIVE scenarios (different waste outcomes):")
    print(f"    v2 informative CV accuracy: {v2_info_acc:.1f}%")

    if v2_info_acc >= 75:
        print(f"    ✅ Target met: ≥75% on informative scenarios")
    elif v2_info_acc >= 60:
        print(f"    ✅ Solid performance (>60%) on outcome-driven scenarios")
    else:
        print(f"    ⚠️  Below 60% — limited learnable signal in current features")

    print("\n" + "=" * 60)

    # =========================================================================
    # v2.1: PAIRWISE RANKING MODEL + HISTORICAL FEATURES
    # =========================================================================
    print("\n\n")
    print("=" * 60)
    print("  v2.1: PAIRWISE RANKING + HISTORICAL FEATURES")
    print("=" * 60)

    # Load cook logs and sales for historical feature computation
    print("\nLoading cook logs and POS sales for historical features...")
    with open(os.path.join(data_dir, "cook_logs.json")) as f:
        cook_logs = json.load(f)
    with open(os.path.join(data_dir, "pos_sales.json")) as f:
        pos_sales = json.load(f)

    print("Computing historical aggregate features...")
    historical = compute_historical_features(cook_logs, pos_sales)
    for item, stats in historical.items():
        print(f"  {item:15s}: avg writeoff = {stats['overall_avg_writeoff']:.2f}")

    print("\nBuilding pairwise training data...")
    pw_trainer = PairwiseModelTrainer(
        n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42
    )
    pw_trainer.prepare_data(labeled_data, historical, use_weights=False)
    print(f"  Pairwise samples: {pw_trainer.X.shape[0]:,} (from {len(labeled_data):,} scenarios)")
    print(f"  Features per pair: {pw_trainer.X.shape[1]}")
    print(f"  Label balance: {pw_trainer.y.value_counts().to_dict()}")

    print("\nRunning 5-fold cross-validation (pairwise)...")
    pw_cv = pw_trainer.cross_validate(cv=5)
    print(f"  Pairwise CV accuracy: {pw_cv.mean()*100:.1f}% ± {pw_cv.std()*100:.1f}%")

    print("\nTraining final pairwise model...")
    pw_trainer.train()
    pw_trainer.evaluate()
    pw_trainer.print_summary()

    # Evaluate ranking accuracy (top-1 match with labels)
    print("\nEvaluating ranking accuracy (top-1 item prediction)...")
    ranking_eval = pw_trainer.evaluate_ranking_accuracy(labeled_data)
    print(f"  Top-1 ranking accuracy: {ranking_eval['ranking_top1_accuracy']:.1f}%")
    print(f"  Scenarios evaluated: {ranking_eval['total_evaluated']:,}")

    # Save pairwise model
    pw_model_path = pw_trainer.save_model(os.path.join(models_dir, "v2_1_pairwise_model.pkl"))
    print(f"\nPairwise model saved to: {pw_model_path}")

    pw_report_path = os.path.join(output_dir, "v2_1_pairwise_report.json")
    pw_trainer.report["ranking_accuracy"] = ranking_eval
    pw_trainer.save_report(pw_report_path)
    print(f"Pairwise report saved to: {pw_report_path}")

    # =========================================================================
    # v2.2: TEMPORAL SPLIT + SOFT LABELS
    # =========================================================================
    print("\n\n")
    print("=" * 60)
    print("  v2.2: TEMPORAL SPLIT + SOFT LABELS (SAMPLE WEIGHTS)")
    print("=" * 60)

    # Determine temporal split point
    # Our data spans 30 days. Train on first 20 days, test on last 10.
    all_dates = sorted(set(s["features"]["date"] for s in labeled_data))
    split_idx = int(len(all_dates) * 0.67)  # ~20 days train, ~10 days test
    cutoff_date = all_dates[split_idx]

    train_scenarios = [s for s in labeled_data if s["features"]["date"] < cutoff_date]
    test_scenarios = [s for s in labeled_data if s["features"]["date"] >= cutoff_date]

    print(f"\n  Date range: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} days)")
    print(f"  Cutoff: {cutoff_date}")
    print(f"  Train: {len(train_scenarios):,} scenarios (days before {cutoff_date})")
    print(f"  Test:  {len(test_scenarios):,} scenarios (days from {cutoff_date} onward)")

    # Compute historical features ONLY from training period (no leakage)
    print("\n  Computing historical features from TRAIN period only...")
    hist_train = compute_historical_features(cook_logs, pos_sales, cutoff_date=cutoff_date)
    for item, stats in hist_train.items():
        print(f"    {item:15s}: avg writeoff = {stats['overall_avg_writeoff']:.2f}")

    # Build pairwise data with soft labels (sample weights)
    print("\n  Building pairwise data with sample weights...")
    pw_v22 = PairwiseModelTrainer(
        n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42
    )
    pw_v22.prepare_data(train_scenarios, hist_train, use_weights=True)

    print(f"    Train pairs: {pw_v22.X.shape[0]:,}")
    print(f"    Features: {pw_v22.X.shape[1]}")
    print(f"    Weight range: [{pw_v22.sample_weights.min():.2f}, {pw_v22.sample_weights.max():.2f}]")
    print(f"    Mean weight: {pw_v22.sample_weights.mean():.3f}")
    print(f"    Low-weight pairs (<0.5): {(pw_v22.sample_weights < 0.5).sum():,} "
          f"({100*(pw_v22.sample_weights < 0.5).mean():.1f}%)")

    # Cross-validate on training set
    print("\n  Running 5-fold CV on TRAIN set (with sample weights during fit)...")
    pw_v22_cv = pw_v22.cross_validate(cv=5)
    print(f"    Train CV accuracy: {pw_v22_cv.mean()*100:.1f}% ± {pw_v22_cv.std()*100:.1f}%")

    # Train final model
    print("  Training final model with sample weights...")
    pw_v22.train()
    pw_v22.evaluate()

    # Evaluate on HELD-OUT TEST SET (temporal split — the honest metric)
    print("\n  Evaluating on HELD-OUT TEST SET (temporal)...")
    test_ranking = pw_v22.evaluate_ranking_accuracy(test_scenarios)
    print(f"    Test top-1 accuracy: {test_ranking['ranking_top1_accuracy']:.1f}%")
    print(f"    Test scenarios: {test_ranking['total_evaluated']:,}")

    # Also evaluate on full dataset for comparison
    full_ranking = pw_v22.evaluate_ranking_accuracy(labeled_data)
    print(f"    Full dataset top-1: {full_ranking['ranking_top1_accuracy']:.1f}%")

    # Save v2.2 model
    pw_v22_model_path = pw_v22.save_model(os.path.join(models_dir, "v2_2_pairwise_temporal.pkl"))
    print(f"\n  Model saved to: {pw_v22_model_path}")

    # Save report with temporal eval
    pw_v22.report["temporal_split"] = {
        "cutoff_date": cutoff_date,
        "train_scenarios": len(train_scenarios),
        "test_scenarios": len(test_scenarios),
        "train_cv_accuracy": round(pw_v22_cv.mean() * 100, 1),
        "test_top1_accuracy": test_ranking["ranking_top1_accuracy"],
        "full_top1_accuracy": full_ranking["ranking_top1_accuracy"],
    }
    pw_v22.report["sample_weights"] = {
        "enabled": True,
        "min_weight": 0.3,
        "max_weight": 1.0,
        "mean_weight": round(float(pw_v22.sample_weights.mean()), 3),
        "low_weight_pct": round(100 * float((pw_v22.sample_weights < 0.5).mean()), 1),
    }
    pw_v22_report_path = os.path.join(output_dir, "v2_2_temporal_report.json")
    pw_v22.save_report(pw_v22_report_path)
    print(f"  Report saved to: {pw_v22_report_path}")

    pw_v22.print_summary()

    # --- Final comparison ---
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON: Associate vs v1 vs v2 vs v2.1 vs v2.2")
    print("=" * 60)

    print(f"\n  {'Model':<42s} {'CV/Train':>10s} {'Test':>10s}")
    print(f"  {'-'*42} {'-'*10} {'-'*10}")
    print(f"  {'Associate baseline (current state)':<42s} {'—':>10s} {associate_accuracy:>9.1f}%")
    print(f"  {'v1 (rule-based heuristic)':<42s} {'—':>10s} {v1_accuracy:>9.1f}%")
    print(f"  {'v2 (multiclass RF)':<42s} {v2_all_acc:>9.1f}% {'—':>10s}")
    print(f"  {'v2.1 (pairwise GBM, no split)':<42s} "
          f"{pw_cv.mean()*100:>9.1f}% "
          f"{ranking_eval['ranking_top1_accuracy']:>9.1f}%")
    print(f"  {'v2.2 (pairwise + temporal + weights)':<42s} "
          f"{pw_v22_cv.mean()*100:>9.1f}% "
          f"{test_ranking['ranking_top1_accuracy']:>9.1f}%")

    print(f"\n  Key metrics (vs. associate baseline):")
    print(f"    Associate baseline:                     {associate_accuracy:.1f}%")
    print(f"    v2.2 test (honest, temporal):            {test_ranking['ranking_top1_accuracy']:.1f}%")
    print(f"    v2.2 improvement over associate:        +{test_ranking['ranking_top1_accuracy'] - associate_accuracy:.1f} pp")
    print(f"    v2.2 improvement over v1:               +{test_ranking['ranking_top1_accuracy'] - v1_accuracy:.1f} pp")

    if test_ranking['ranking_top1_accuracy'] - associate_accuracy >= 20:
        print(f"\n  ✅ STRONG IMPACT: ≥20pp improvement over current associate behavior!")
    elif test_ranking['ranking_top1_accuracy'] - associate_accuracy >= 10:
        print(f"\n  ✅ Meaningful improvement over associate baseline")
    else:
        print(f"\n  ⚠️  Limited improvement over associate baseline")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
