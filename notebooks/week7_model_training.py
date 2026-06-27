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
from src.pairwise_trainer import PairwiseModelTrainer, OVEN_ITEMS, compute_historical_features
from src.lambdarank_trainer import LambdaRankTrainer
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
        for item in OVEN_ITEMS:
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

    # =========================================================================
    # v2.3: SYMMETRIC PAIRS + PROBA AGGREGATION (no sample weights)
    # Implements three improvements over v2.2:
    #   1. Symmetric pair augmentation — removes item-position bias from
    #      OVEN_ITEMS ordering and perfectly balances the 69/31 label skew.
    #   2. Probability-based rank aggregation — preserves confidence, avoids
    #      intransitive cycles from hard 0/1 win-counting.
    #   3. No sample weights — edge/near-tie pairs get equal weight so the
    #      model isn't starved of signal on hard waste-avoidance decisions.
    # Same temporal split and historical features as v2.2.
    # =========================================================================
    print("\n\n")
    print("=" * 60)
    print("  v2.3: SYMMETRIC PAIRS + PROBA AGGREGATION (no weights)")
    print("=" * 60)

    print(f"\n  Reusing v2.2 temporal split: cutoff={cutoff_date}, "
          f"train={len(train_scenarios):,}, test={len(test_scenarios):,}")
    print("  Reusing v2.2 historical features (train-only, no leakage).")

    print("\n  Building pairwise data (symmetric=True, use_weights=False)...")
    pw_v23 = PairwiseModelTrainer(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42,
        use_proba=True,
    )
    pw_v23.prepare_data(train_scenarios, hist_train, use_weights=False, symmetric=True)

    n_orig = pw_v22.X.shape[0]
    n_sym = pw_v23.X.shape[0]
    label_dist = pw_v23.y.value_counts().to_dict()
    print(f"    Train pairs (symmetric): {n_sym:,}  (v2.2 had {n_orig:,}, ×{n_sym/n_orig:.2f}x)")
    print(f"    Label distribution: {label_dist}  (balance: {100*label_dist.get(1,0)/n_sym:.1f}% class-1)")

    print("\n  Running 5-fold CV on TRAIN set...")
    pw_v23_cv = pw_v23.cross_validate(cv=5)
    print(f"    Train CV accuracy: {pw_v23_cv.mean()*100:.1f}% ± {pw_v23_cv.std()*100:.1f}%")

    print("  Training final model (use_proba=True)...")
    pw_v23.train()
    pw_v23.evaluate()

    print("\n  Evaluating on HELD-OUT TEST SET (temporal)...")
    test_ranking_v23 = pw_v23.evaluate_ranking_accuracy(test_scenarios)
    full_ranking_v23 = pw_v23.evaluate_ranking_accuracy(labeled_data)
    print(f"    Test top-1 accuracy:  {test_ranking_v23['ranking_top1_accuracy']:.1f}%  "
          f"(v2.2: {test_ranking['ranking_top1_accuracy']:.1f}%  "
          f"Δ={test_ranking_v23['ranking_top1_accuracy']-test_ranking['ranking_top1_accuracy']:+.1f}pp)")
    print(f"    Full dataset top-1:   {full_ranking_v23['ranking_top1_accuracy']:.1f}%  "
          f"(v2.2: {full_ranking['ranking_top1_accuracy']:.1f}%)")

    pw_v23_model_path = pw_v23.save_model(os.path.join(models_dir, "v2_3_pairwise_symmetric.pkl"))
    print(f"\n  Model saved to: {pw_v23_model_path}")

    pw_v23.report["improvements"] = {
        "symmetric_pairs": True,
        "use_proba_aggregation": True,
        "sample_weights": False,
        "note": ("Symmetric augmentation balances 69/31 label skew and removes item-position "
                 "bias. Proba aggregation preserves confidence and avoids intransitive rank "
                 "cycles. No weights lets edge/near-tie pairs contribute equally."),
    }
    pw_v23.report["temporal_split"] = {
        "cutoff_date": cutoff_date,
        "train_scenarios": len(train_scenarios),
        "test_scenarios": len(test_scenarios),
        "train_cv_accuracy": round(pw_v23_cv.mean() * 100, 1),
        "test_top1_accuracy": test_ranking_v23["ranking_top1_accuracy"],
        "full_top1_accuracy": full_ranking_v23["ranking_top1_accuracy"],
    }
    pw_v23_report_path = os.path.join(output_dir, "v2_3_symmetric_report.json")
    pw_v23.save_report(pw_v23_report_path)
    print(f"  Report saved to: {pw_v23_report_path}")

    # =========================================================================
    # v3: LightGBM LambdaRank (listwise NDCG optimisation, group = scenario)
    # =========================================================================
    print("\n\n")
    print("=" * 60)
    print("  v3: LightGBM LambdaRank (listwise, group=scenario)")
    print("=" * 60)
    print(f"\n  Reusing v2.2 temporal split: cutoff={cutoff_date}, "
          f"train={len(train_scenarios):,}, test={len(test_scenarios):,}")
    print("  Reusing v2.2 historical features (train-only, no leakage).")
    print("\n  Key differences vs pairwise GBM:")
    print("    - One row per (scenario, item) — no C(n,2) explosion")
    print("    - Relevance = reverse rank position (graded, linear label_gain)")
    print("    - LambdaRank loss optimises NDCG directly")
    print("    - Inference: model.predict() scores → sort desc (no win-counting)")

    print("\n  Building listwise training data...")
    v3_trainer = LambdaRankTrainer(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    v3_trainer.prepare_data(train_scenarios, hist_train)
    print(f"    Scenarios (train): {len(v3_trainer.group):,}")
    print(f"    Item-rows  (train): {len(v3_trainer.X):,}")
    print(f"    Features per row:  {len(v3_trainer.feature_names)}")
    print(f"    Relevance range:   [0, {max(v3_trainer.y)}]")
    avg_group = sum(v3_trainer.group) / len(v3_trainer.group)
    print(f"    Avg items/scenario: {avg_group:.1f}")

    print("\n  Training final model...")
    v3_trainer.train()
    print("  Done.")

    # --- Evaluate on TRAINING set (in-sample NDCG) ---
    print("\n  Evaluating on TRAIN set (in-sample)...")
    train_eval_v3 = v3_trainer.evaluate(labeled_data=train_scenarios)
    print(f"    Train NDCG@1: {train_eval_v3['ndcg_at_1']:.4f}")
    print(f"    Train NDCG@3: {train_eval_v3['ndcg_at_3']:.4f}")
    print(f"    Train NDCG@5: {train_eval_v3['ndcg_at_5']:.4f}")
    print(f"    Train Top-1 accuracy: "
          f"{train_eval_v3['ranking_accuracy']['ranking_top1_accuracy']:.1f}%")

    # --- Evaluate on HELD-OUT TEST SET (temporal, honest) ---
    print("\n  Evaluating on HELD-OUT TEST SET (temporal)...")
    test_eval_v3 = v3_trainer.evaluate(labeled_data=test_scenarios)
    test_ndcg1_v3 = test_eval_v3["ndcg_at_1"]
    test_ndcg3_v3 = test_eval_v3["ndcg_at_3"]
    test_ndcg5_v3 = test_eval_v3["ndcg_at_5"]
    test_top1_v3 = test_eval_v3["ranking_accuracy"]["ranking_top1_accuracy"]
    print(f"    Test NDCG@1: {test_ndcg1_v3:.4f}")
    print(f"    Test NDCG@3: {test_ndcg3_v3:.4f}")
    print(f"    Test NDCG@5: {test_ndcg5_v3:.4f}")
    print(f"    Test Top-1 accuracy: {test_top1_v3:.1f}%  "
          f"(v2.2: {test_ranking['ranking_top1_accuracy']:.1f}%  "
          f"Δ={test_top1_v3-test_ranking['ranking_top1_accuracy']:+.1f}pp)")

    # --- Evaluate on full dataset ---
    print("\n  Evaluating on FULL dataset...")
    full_eval_v3 = v3_trainer.evaluate(labeled_data=labeled_data)
    full_top1_v3 = full_eval_v3["ranking_accuracy"]["ranking_top1_accuracy"]
    print(f"    Full Top-1 accuracy: {full_top1_v3:.1f}%")

    # --- Save model ---
    v3_model_path = v3_trainer.save_model(os.path.join(models_dir, "v3_lambdarank.pkl"))
    print(f"\n  Model saved to: {v3_model_path}")

    # --- Save report (test-set eval is the definitive record) ---
    v3_trainer.report.update({
        "temporal_split": {
            "cutoff_date": cutoff_date,
            "train_scenarios": len(train_scenarios),
            "test_scenarios": len(test_scenarios),
        },
        "test_ndcg_at_1": test_ndcg1_v3,
        "test_ndcg_at_3": test_ndcg3_v3,
        "test_ndcg_at_5": test_ndcg5_v3,
        "test_top1_accuracy": test_top1_v3,
        "full_top1_accuracy": full_top1_v3,
        "train_ndcg_at_1": train_eval_v3["ndcg_at_1"],
        "train_ndcg_at_3": train_eval_v3["ndcg_at_3"],
        "train_ndcg_at_5": train_eval_v3["ndcg_at_5"],
        "train_top1_accuracy": train_eval_v3["ranking_accuracy"]["ranking_top1_accuracy"],
        "improvements": {
            "listwise_structure": True,
            "ndcg_optimised_directly": True,
            "no_pairwise_expansion": True,
            "no_win_counting": True,
            "note": (
                "LGBMRanker(objective=lambdarank) with group=scenario and graded "
                "reverse-rank relevance. One row per (scenario, item) instead of "
                "C(n,2) pairs. Eliminates win-counting at inference. Linear label_gain "
                "prevents 2^label blow-up across 28 grades."
            ),
        },
    })
    v3_report_path = os.path.join(output_dir, "v3_lambdarank_report.json")
    v3_trainer.save_report(v3_report_path)
    print(f"  Report saved to: {v3_report_path}")

    v3_trainer.print_summary()

    # --- Final comparison: all models ---
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON: Associate vs v1 vs v2.x vs v3")
    print("=" * 60)

    print(f"\n  {'Model':<56s} {'CV/Train':>10s} {'Test':>10s}")
    print(f"  {'-'*56} {'-'*10} {'-'*10}")
    print(f"  {'Associate baseline':<56s} {'—':>10s} {associate_accuracy:>9.1f}%")
    print(f"  {'v1 (rule-based heuristic)':<56s} {'—':>10s} {v1_accuracy:>9.1f}%")
    print(f"  {'v2.1 (pairwise GBM, no split)':<56s} "
          f"{pw_cv.mean()*100:>9.1f}% "
          f"{ranking_eval['ranking_top1_accuracy']:>9.1f}%")
    print(f"  {'v2.2 (pairwise + temporal + weights, hard wins)':<56s} "
          f"{pw_v22_cv.mean()*100:>9.1f}% "
          f"{test_ranking['ranking_top1_accuracy']:>9.1f}%")
    print(f"  {'v2.3 (symmetric + proba, no weights)':<56s} "
          f"{pw_v23_cv.mean()*100:>9.1f}% "
          f"{test_ranking_v23['ranking_top1_accuracy']:>9.1f}%")
    print(f"  {'v3  (LightGBM lambdarank, listwise)':<56s} "
          f"{'—':>10s} "
          f"{test_top1_v3:>9.1f}%")

    delta_v3_v22 = test_top1_v3 - test_ranking['ranking_top1_accuracy']
    delta_v3_assoc = test_top1_v3 - associate_accuracy
    print(f"\n  v3 vs v2.2 on holdout:          {delta_v3_v22:+.1f}pp")
    print(f"  v3 vs associate baseline:        {delta_v3_assoc:+.1f}pp")
    print(f"  v3 test NDCG@1/3/5:             "
          f"{test_ndcg1_v3:.4f} / {test_ndcg3_v3:.4f} / {test_ndcg5_v3:.4f}")

    if delta_v3_v22 > 0:
        print(f"\n  v3 beats v2.2 by {delta_v3_v22:+.1f}pp on temporal holdout.")
    elif delta_v3_v22 == 0:
        print(f"\n  v3 matches v2.2 on temporal holdout.")
    else:
        print(f"\n  v3 trails v2.2 by {abs(delta_v3_v22):.1f}pp on temporal holdout "
              f"— app/API stays on v2.2 pending further tuning.")

    print(f"\n  NOTE: App/API still serves v2.2 (models/v2_2_pairwise_temporal.pkl).")
    print(f"        Swap app/utils.py:load_model() to v3_lambdarank.pkl when ready.")


if __name__ == "__main__":
    main()
