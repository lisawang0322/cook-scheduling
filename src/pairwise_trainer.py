"""v2.1 Pairwise Ranking Model — Learning to Rank for cook order prediction.

Instead of multiclass ("which item first?"), this model learns pairwise preferences:
"given items A and B in context C, should A be cooked before B?"

This directly addresses the pizza/wings_2h confusion by giving the model
dedicated training signal for each pair comparison.

Also adds historical aggregate features (avg write-off per item/hour/store_type)
to provide distinguishing context between items with similar time characteristics.
"""

import json
import os
import pickle
from collections import defaultdict
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report


# Fixed order of oven items
OVEN_ITEMS = ["pizza", "wings_2h", "wings_4h", "baked_goods"]


def compute_historical_features(cook_logs: list[dict[str, Any]],
                                pos_sales: list[dict[str, Any]],
                                cutoff_date: str | None = None) -> dict[str, dict]:
    """Compute historical aggregate features from cook logs.

    Args:
        cook_logs: All cook events.
        pos_sales: All POS sale records.
        cutoff_date: If provided (ISO format YYYY-MM-DD), only use events
            strictly before this date. This prevents data leakage in
            temporal train/test splits.

    Returns a nested dict: historical[item][key] = value
    Keys include:
    - avg_writeoff_by_hour[hour] — avg waste for this item at this hour
    - avg_writeoff_by_store_type[store_type] — avg waste by store type
    - avg_demand_by_hour[hour] — avg demand for this item at this hour
    - overall_avg_writeoff — global average waste for this item
    """
    # Index sales by event
    sales_by_event: dict[str, int] = defaultdict(int)
    for sale in pos_sales:
        sales_by_event[sale["cook_event_id"]] += sale["quantity"]

    # Collect per-item stats
    item_writeoff_by_hour: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    item_writeoff_by_store: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    item_demand_by_hour: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    item_all_writeoffs: dict[str, list] = defaultdict(list)

    for event in cook_logs:
        if event.get("cook_type") != "initial":
            continue
        item = event["item"]
        if item not in OVEN_ITEMS:
            continue
        if cutoff_date and event.get("date", "") >= cutoff_date:
            continue

        hour = event["window_start_hour"]
        store_type = event["store_type"]
        cooked = event["cooked_qty"]
        sold = sales_by_event.get(event["cook_event_id"], 0)
        writeoff = max(0, cooked - sold)

        item_writeoff_by_hour[item][hour].append(writeoff)
        item_writeoff_by_store[item][store_type].append(writeoff)
        item_demand_by_hour[item][hour].append(event["forecast_demand"])
        item_all_writeoffs[item].append(writeoff)

    # Compute averages
    historical = {}
    for item in OVEN_ITEMS:
        historical[item] = {
            "avg_writeoff_by_hour": {
                h: np.mean(vals) if vals else 0
                for h, vals in item_writeoff_by_hour[item].items()
            },
            "avg_writeoff_by_store_type": {
                st: np.mean(vals) if vals else 0
                for st, vals in item_writeoff_by_store[item].items()
            },
            "avg_demand_by_hour": {
                h: np.mean(vals) if vals else 0
                for h, vals in item_demand_by_hour[item].items()
            },
            "overall_avg_writeoff": (
                np.mean(item_all_writeoffs[item]) if item_all_writeoffs[item] else 0
            ),
        }

    return historical


def build_pairwise_data(labeled_data: list[dict[str, Any]],
                        historical: dict[str, dict],
                        compute_weights: bool = False) -> tuple[pd.DataFrame, pd.Series, np.ndarray | None]:
    """Convert labeled scenarios into pairwise comparison samples.

    For each decision point with items [A, B, C], generate pairs:
    (A,B), (A,C), (B,C) with binary label: 1 if left item should go first.

    Features for each pair:
    - Context features (shared): decision_hour, store_type, is_weekend, day_of_week
    - Item A features: demand, lcu, urgency, hold_time, historical avg writeoff
    - Item B features: same
    - Difference features: A_urgency - B_urgency, A_demand - B_demand, etc.

    If compute_weights=True, returns sample weights based on priority score margin.
    High-margin pairs (obvious decisions) get weight 1.0.
    Low-margin pairs (nearly tied) get lower weight (min 0.3).
    """
    rows = []
    labels = []
    weights = []

    store_map = {"urban": 2, "suburban": 1, "highway": 0}
    day_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6,
    }

    for scenario in labeled_data:
        features = scenario["features"]
        optimal_order = scenario["optimal_order"]

        # Find which items are present
        present_items = [
            item for item in OVEN_ITEMS
            if f"{item}_forecast_demand" in features
        ]

        if len(present_items) < 2:
            continue

        # Context features
        decision_hour = features["decision_hour"]
        store_type = features["store_type"]
        is_weekend = int(features["is_weekend"])
        day_of_week = day_map.get(features["day_of_week"], 0)
        store_encoded = store_map.get(store_type, 1)

        # Generate all pairs
        for item_a, item_b in combinations(present_items, 2):
            # Determine label: does item_a come before item_b in optimal order?
            idx_a = optimal_order.index(item_a) if item_a in optimal_order else 99
            idx_b = optimal_order.index(item_b) if item_b in optimal_order else 99
            label = 1 if idx_a < idx_b else 0

            # Compute sample weight from rank distance + write-off difference
            if compute_weights:
                rank_gap = abs(idx_a - idx_b)
                total_items = len(optimal_order)
                # Rank-based margin [0, 1]
                rank_margin = rank_gap / max(1, total_items - 1)

                # Write-off difference: if items have different waste, label is confident
                a_wo = features.get(f"{item_a}_writeoff", 0)
                b_wo = features.get(f"{item_b}_writeoff", 0)
                a_cooked = features.get(f"{item_a}_cooked_qty", 1)
                b_cooked = features.get(f"{item_b}_cooked_qty", 1)
                a_waste_rate = a_wo / max(1, a_cooked)
                b_waste_rate = b_wo / max(1, b_cooked)
                waste_diff = abs(a_waste_rate - b_waste_rate)
                # Normalize waste diff to [0, 1] (cap at 0.5 diff = max signal)
                waste_signal = min(1.0, waste_diff / 0.5)

                # Combined weight: blend rank margin + waste signal
                # Weight range: [0.2, 1.0]
                combined = 0.5 * rank_margin + 0.5 * waste_signal
                weight = 0.2 + 0.8 * combined
                weights.append(weight)

            # Item A features
            a_demand = features[f"{item_a}_forecast_demand"]
            a_lcu = features[f"{item_a}_lcu"]
            a_hold = features[f"{item_a}_hold_time"]
            a_time_rem = features[f"{item_a}_time_remaining"]
            a_cooked = features[f"{item_a}_cooked_qty"]
            a_urgency = 1.0 / max(0.01, a_time_rem)
            a_density = a_demand / max(1, a_lcu)

            # Item B features
            b_demand = features[f"{item_b}_forecast_demand"]
            b_lcu = features[f"{item_b}_lcu"]
            b_hold = features[f"{item_b}_hold_time"]
            b_time_rem = features[f"{item_b}_time_remaining"]
            b_cooked = features[f"{item_b}_cooked_qty"]
            b_urgency = 1.0 / max(0.01, b_time_rem)
            b_density = b_demand / max(1, b_lcu)

            # Historical features for item A
            hist_a = historical.get(item_a, {})
            a_hist_wo_hour = hist_a.get("avg_writeoff_by_hour", {}).get(decision_hour, 0)
            a_hist_wo_store = hist_a.get("avg_writeoff_by_store_type", {}).get(store_type, 0)
            a_hist_demand_hour = hist_a.get("avg_demand_by_hour", {}).get(decision_hour, 0)
            a_hist_wo_overall = hist_a.get("overall_avg_writeoff", 0)

            # Historical features for item B
            hist_b = historical.get(item_b, {})
            b_hist_wo_hour = hist_b.get("avg_writeoff_by_hour", {}).get(decision_hour, 0)
            b_hist_wo_store = hist_b.get("avg_writeoff_by_store_type", {}).get(store_type, 0)
            b_hist_demand_hour = hist_b.get("avg_demand_by_hour", {}).get(decision_hour, 0)
            b_hist_wo_overall = hist_b.get("overall_avg_writeoff", 0)

            # Demand vs historical average (is this unusual?)
            a_demand_vs_avg = a_demand - a_hist_demand_hour if a_hist_demand_hour else 0
            b_demand_vs_avg = b_demand - b_hist_demand_hour if b_hist_demand_hour else 0

            row = {
                # Context
                "decision_hour": decision_hour,
                "store_type_encoded": store_encoded,
                "is_weekend": is_weekend,
                "day_of_week": day_of_week,
                "num_items": features["num_oven_items"],

                # Item A
                "a_demand": a_demand,
                "a_lcu": a_lcu,
                "a_hold_time": a_hold,
                "a_time_remaining": a_time_rem,
                "a_cooked_qty": a_cooked,
                "a_urgency": round(a_urgency, 4),
                "a_demand_density": round(a_density, 4),
                "a_hist_wo_hour": round(a_hist_wo_hour, 3),
                "a_hist_wo_store": round(a_hist_wo_store, 3),
                "a_hist_wo_overall": round(a_hist_wo_overall, 3),
                "a_demand_vs_avg": round(a_demand_vs_avg, 2),

                # Item B
                "b_demand": b_demand,
                "b_lcu": b_lcu,
                "b_hold_time": b_hold,
                "b_time_remaining": b_time_rem,
                "b_cooked_qty": b_cooked,
                "b_urgency": round(b_urgency, 4),
                "b_demand_density": round(b_density, 4),
                "b_hist_wo_hour": round(b_hist_wo_hour, 3),
                "b_hist_wo_store": round(b_hist_wo_store, 3),
                "b_hist_wo_overall": round(b_hist_wo_overall, 3),
                "b_demand_vs_avg": round(b_demand_vs_avg, 2),

                # Difference features (A - B)
                "diff_urgency": round(a_urgency - b_urgency, 4),
                "diff_demand_density": round(a_density - b_density, 4),
                "diff_hold_time": a_hold - b_hold,
                "diff_time_remaining": round(a_time_rem - b_time_rem, 2),
                "diff_demand": a_demand - b_demand,
                "diff_hist_wo_hour": round(a_hist_wo_hour - b_hist_wo_hour, 3),
                "diff_hist_wo_store": round(a_hist_wo_store - b_hist_wo_store, 3),
                "diff_hist_wo_overall": round(a_hist_wo_overall - b_hist_wo_overall, 3),
                "diff_demand_vs_avg": round(a_demand_vs_avg - b_demand_vs_avg, 2),
            }

            rows.append(row)
            labels.append(label)

    X = pd.DataFrame(rows)
    y = pd.Series(labels, name="a_preferred")
    w = np.array(weights) if compute_weights else None
    return X, y, w


class PairwiseModelTrainer:
    """Trains a pairwise ranking model (Learning to Rank approach)."""

    def __init__(self, n_estimators: int = 300, max_depth: int = 5,
                 learning_rate: float = 0.1, random_state: int = 42):
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            subsample=0.8,
        )
        self.random_state = random_state
        self.X: pd.DataFrame | None = None
        self.y: pd.Series | None = None
        self.sample_weights: np.ndarray | None = None
        self.cv_scores: np.ndarray | None = None
        self.feature_importance: dict[str, float] | None = None
        self.report: dict[str, Any] | None = None
        self.historical: dict[str, dict] | None = None

    def prepare_data(self, labeled_data: list[dict[str, Any]],
                     historical: dict[str, dict],
                     use_weights: bool = False) -> None:
        """Build pairwise feature matrix with optional sample weights."""
        self.historical = historical
        self.X, self.y, self.sample_weights = build_pairwise_data(
            labeled_data, historical, compute_weights=use_weights
        )

    def cross_validate(self, cv: int = 5) -> np.ndarray:
        """Run stratified k-fold cross-validation."""
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        self.cv_scores = cross_val_score(
            self.model, self.X, self.y, cv=skf, scoring="accuracy"
        )
        return self.cv_scores

    def train(self) -> None:
        """Train the model on all data, using sample weights if available."""
        self.model.fit(self.X, self.y, sample_weight=self.sample_weights)

        # Feature importance
        importances = self.model.feature_importances_
        feature_names = list(self.X.columns)
        self.feature_importance = dict(sorted(
            zip(feature_names, importances),
            key=lambda x: -x[1]
        ))

    def evaluate(self) -> dict[str, Any]:
        """Generate evaluation report."""
        if self.cv_scores is None:
            self.cross_validate()
        if self.feature_importance is None:
            self.train()

        y_pred = self.model.predict(self.X)
        class_report = classification_report(self.y, y_pred, output_dict=True)

        self.report = {
            "model_type": "pairwise_gradient_boosting",
            "cross_validation": {
                "cv_folds": len(self.cv_scores),
                "scores": [round(s, 4) for s in self.cv_scores.tolist()],
                "mean_accuracy": round(self.cv_scores.mean() * 100, 1),
                "std_accuracy": round(self.cv_scores.std() * 100, 1),
            },
            "training_set": {
                "n_samples": len(self.X),
                "n_features": len(self.X.columns),
                "label_distribution": self.y.value_counts().to_dict(),
            },
            "training_accuracy": round(class_report["accuracy"] * 100, 1),
            "feature_importance_top10": dict(list(self.feature_importance.items())[:10]),
        }
        return self.report

    def rank_items(self, scenario_features: dict[str, Any],
                   present_items: list[str]) -> list[str]:
        """Use the pairwise model to produce a full ranking for a scenario.

        For each pair, predict preference. Aggregate wins to produce ranking.
        """
        if len(present_items) < 2:
            return present_items

        # Count wins for each item
        wins = defaultdict(int)
        store_map = {"urban": 2, "suburban": 1, "highway": 0}
        day_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6,
        }

        decision_hour = scenario_features["decision_hour"]
        store_type = scenario_features["store_type"]

        for item_a, item_b in combinations(present_items, 2):
            # Build pair features
            a_demand = scenario_features.get(f"{item_a}_forecast_demand", 0)
            a_lcu = scenario_features.get(f"{item_a}_lcu", 1)
            a_hold = scenario_features.get(f"{item_a}_hold_time", 2)
            a_time_rem = scenario_features.get(f"{item_a}_time_remaining", 2)
            a_cooked = scenario_features.get(f"{item_a}_cooked_qty", 0)
            a_urgency = 1.0 / max(0.01, a_time_rem)
            a_density = a_demand / max(1, a_lcu)

            b_demand = scenario_features.get(f"{item_b}_forecast_demand", 0)
            b_lcu = scenario_features.get(f"{item_b}_lcu", 1)
            b_hold = scenario_features.get(f"{item_b}_hold_time", 2)
            b_time_rem = scenario_features.get(f"{item_b}_time_remaining", 2)
            b_cooked = scenario_features.get(f"{item_b}_cooked_qty", 0)
            b_urgency = 1.0 / max(0.01, b_time_rem)
            b_density = b_demand / max(1, b_lcu)

            hist_a = self.historical.get(item_a, {})
            hist_b = self.historical.get(item_b, {})
            a_hist_wo_hour = hist_a.get("avg_writeoff_by_hour", {}).get(decision_hour, 0)
            a_hist_wo_store = hist_a.get("avg_writeoff_by_store_type", {}).get(store_type, 0)
            a_hist_wo_overall = hist_a.get("overall_avg_writeoff", 0)
            a_hist_demand_hour = hist_a.get("avg_demand_by_hour", {}).get(decision_hour, 0)
            b_hist_wo_hour = hist_b.get("avg_writeoff_by_hour", {}).get(decision_hour, 0)
            b_hist_wo_store = hist_b.get("avg_writeoff_by_store_type", {}).get(store_type, 0)
            b_hist_wo_overall = hist_b.get("overall_avg_writeoff", 0)
            b_hist_demand_hour = hist_b.get("avg_demand_by_hour", {}).get(decision_hour, 0)

            row = pd.DataFrame([{
                "decision_hour": decision_hour,
                "store_type_encoded": store_map.get(store_type, 1),
                "is_weekend": int(scenario_features.get("is_weekend", False)),
                "day_of_week": day_map.get(scenario_features.get("day_of_week", "Monday"), 0),
                "num_items": scenario_features.get("num_oven_items", len(present_items)),
                "a_demand": a_demand, "a_lcu": a_lcu, "a_hold_time": a_hold,
                "a_time_remaining": a_time_rem, "a_cooked_qty": a_cooked,
                "a_urgency": round(a_urgency, 4), "a_demand_density": round(a_density, 4),
                "a_hist_wo_hour": round(a_hist_wo_hour, 3),
                "a_hist_wo_store": round(a_hist_wo_store, 3),
                "a_hist_wo_overall": round(a_hist_wo_overall, 3),
                "a_demand_vs_avg": round(a_demand - a_hist_demand_hour, 2),
                "b_demand": b_demand, "b_lcu": b_lcu, "b_hold_time": b_hold,
                "b_time_remaining": b_time_rem, "b_cooked_qty": b_cooked,
                "b_urgency": round(b_urgency, 4), "b_demand_density": round(b_density, 4),
                "b_hist_wo_hour": round(b_hist_wo_hour, 3),
                "b_hist_wo_store": round(b_hist_wo_store, 3),
                "b_hist_wo_overall": round(b_hist_wo_overall, 3),
                "b_demand_vs_avg": round(b_demand - b_hist_demand_hour, 2),
                "diff_urgency": round(a_urgency - b_urgency, 4),
                "diff_demand_density": round(a_density - b_density, 4),
                "diff_hold_time": a_hold - b_hold,
                "diff_time_remaining": round(a_time_rem - b_time_rem, 2),
                "diff_demand": a_demand - b_demand,
                "diff_hist_wo_hour": round(a_hist_wo_hour - b_hist_wo_hour, 3),
                "diff_hist_wo_store": round(a_hist_wo_store - b_hist_wo_store, 3),
                "diff_hist_wo_overall": round(a_hist_wo_overall - b_hist_wo_overall, 3),
                "diff_demand_vs_avg": round((a_demand - a_hist_demand_hour) - (b_demand - b_hist_demand_hour), 2),
            }])

            pred = self.model.predict(row)[0]
            if pred == 1:
                wins[item_a] += 1
            else:
                wins[item_b] += 1

        # Rank by number of wins (descending)
        return sorted(present_items, key=lambda x: -wins.get(x, 0))

    def evaluate_ranking_accuracy(self, labeled_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate full ranking accuracy: does pairwise model's top-1 match the label?"""
        correct = 0
        total = 0

        for scenario in labeled_data:
            features = scenario["features"]
            expected_first = scenario["optimal_first_item"]

            present_items = [
                item for item in OVEN_ITEMS
                if f"{item}_forecast_demand" in features
            ]
            if len(present_items) < 2:
                continue

            predicted_order = self.rank_items(features, present_items)
            if predicted_order[0] == expected_first:
                correct += 1
            total += 1

        return {
            "ranking_top1_accuracy": round(100 * correct / total, 1) if total else 0,
            "total_evaluated": total,
        }

    def save_model(self, path: str) -> str:
        """Save trained model and historical data."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "historical": self.historical}, f)
        return path

    def save_report(self, path: str) -> str:
        """Save evaluation report."""
        if self.report is None:
            self.evaluate()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.report, f, indent=2, default=str)
        return path

    def print_summary(self) -> None:
        """Print results."""
        if self.report is None:
            self.evaluate()

        r = self.report
        cv = r["cross_validation"]

        print("=" * 60)
        print("  v2.1 PAIRWISE MODEL — RESULTS")
        print("=" * 60)

        print(f"\n--- Training Data ---")
        print(f"  Pairwise samples: {r['training_set']['n_samples']:,}")
        print(f"  Features: {r['training_set']['n_features']}")
        print(f"  Label balance: {r['training_set']['label_distribution']}")

        print(f"\n--- Cross-Validation ({cv['cv_folds']}-fold) ---")
        print(f"  Pairwise accuracy: {cv['mean_accuracy']:.1f}% ± {cv['std_accuracy']:.1f}%")

        print(f"\n--- Training Accuracy ---")
        print(f"  {r['training_accuracy']:.1f}%")

        print(f"\n--- Top 10 Feature Importances ---")
        for feat, imp in r["feature_importance_top10"].items():
            bar = "█" * int(imp * 40)
            print(f"  {feat:25s}: {imp:.4f} {bar}")

        print("\n" + "=" * 60)
