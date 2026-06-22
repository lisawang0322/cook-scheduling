"""Week 7: v2 ML Model Training — RandomForest classifier for cook order prediction.

Trains a model to predict the optimal first item to cook at each decision point,
given scenario features (time, store type, per-item demand/urgency/LCU).

The model uses a fixed feature vector with slots for each possible oven item.
Items not present at a decision point get zero-filled features.
"""

import json
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

from src.pairwise_trainer import OVEN_ITEMS  # single source of truth for item list

# Per-item feature columns (appended with item name prefix)
ITEM_FEATURES = ["forecast_demand", "lcu", "hold_time", "time_remaining", "cooked_qty"]


def build_feature_matrix(labeled_data: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.Series]:
    """Convert labeled scenarios into a fixed-width feature matrix and target labels.

    Returns:
        (X, y) where X is a DataFrame of numeric features and y is the target labels.
    """
    rows = []
    labels = []

    for scenario in labeled_data:
        features = scenario["features"]
        row = {}

        # Global features
        row["decision_hour"] = features["decision_hour"]
        row["is_weekend"] = int(features["is_weekend"])
        row["num_oven_items"] = features["num_oven_items"]

        # Encode store type as numeric
        store_map = {"urban": 2, "suburban": 1, "highway": 0}
        row["store_type_encoded"] = store_map.get(features["store_type"], 1)

        # Encode day of week as numeric (Monday=0 ... Sunday=6)
        day_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6,
        }
        row["day_of_week_encoded"] = day_map.get(features["day_of_week"], 0)

        # Per-item features (fixed slots for all oven items)
        for item in OVEN_ITEMS:
            present = f"{item}_forecast_demand" in features
            row[f"{item}_present"] = int(present)
            for feat in ITEM_FEATURES:
                key = f"{item}_{feat}"
                row[key] = features.get(key, 0)

        # Per-item derived features
        demands = []
        times = []
        for item in OVEN_ITEMS:
            if f"{item}_forecast_demand" in features:
                demand = features[f"{item}_forecast_demand"]
                lcu = features[f"{item}_lcu"]
                time_rem = features[f"{item}_time_remaining"]
                hold = features[f"{item}_hold_time"]

                demands.append(demand)
                times.append(time_rem)

                # Urgency (1/time_remaining)
                row[f"{item}_urgency"] = round(1.0 / max(0.01, time_rem), 4)
                # Demand density (demand / LCU)
                row[f"{item}_demand_density"] = round(demand / max(1, lcu), 4)
                # Waste penalty (1 + LCU/demand)
                row[f"{item}_waste_penalty"] = round(1.0 + lcu / max(1, demand), 4)
                # v1-style score
                urgency = 1.0 / max(0.01, time_rem)
                density = demand / max(1, lcu)
                penalty = 1.0 + lcu / max(1, demand)
                row[f"{item}_v1_score"] = round(urgency * density * penalty, 4)
            else:
                row[f"{item}_urgency"] = 0
                row[f"{item}_demand_density"] = 0
                row[f"{item}_waste_penalty"] = 0
                row[f"{item}_v1_score"] = 0

        # Global derived features
        if demands:
            row["max_demand"] = max(demands)
            row["min_time_remaining"] = min(times)
            row["demand_spread"] = max(demands) - min(demands)
            row["urgency_spread"] = max(times) - min(times)
        else:
            row["max_demand"] = 0
            row["min_time_remaining"] = 0
            row["demand_spread"] = 0
            row["urgency_spread"] = 0

        rows.append(row)
        labels.append(scenario["optimal_first_item"])

    X = pd.DataFrame(rows)
    y = pd.Series(labels, name="optimal_first_item")
    return X, y


class ModelTrainer:
    """Trains and evaluates a v2 RandomForest cook-order prediction model."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 12,
                 random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",  # Handle class imbalance (pizza dominates)
            n_jobs=-1,
        )
        self.random_state = random_state
        self.X: pd.DataFrame | None = None
        self.y: pd.Series | None = None
        self.cv_scores: np.ndarray | None = None
        self.feature_importance: dict[str, float] | None = None
        self.report: dict[str, Any] | None = None

    def prepare_data(self, labeled_data: list[dict[str, Any]]) -> None:
        """Build feature matrix from labeled scenarios."""
        self.X, self.y = build_feature_matrix(labeled_data)

    def cross_validate(self, cv: int = 5) -> np.ndarray:
        """Run stratified k-fold cross-validation."""
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        self.cv_scores = cross_val_score(
            self.model, self.X, self.y, cv=skf, scoring="accuracy"
        )
        return self.cv_scores

    def train(self) -> None:
        """Train the model on all data."""
        self.model.fit(self.X, self.y)

        # Extract feature importance
        importances = self.model.feature_importances_
        feature_names = list(self.X.columns)
        self.feature_importance = dict(sorted(
            zip(feature_names, importances),
            key=lambda x: -x[1]
        ))

    def evaluate(self) -> dict[str, Any]:
        """Generate full evaluation report."""
        if self.cv_scores is None:
            self.cross_validate()
        if self.feature_importance is None:
            self.train()

        # Predictions on training set (for classification report)
        y_pred = self.model.predict(self.X)
        class_report = classification_report(self.y, y_pred, output_dict=True)

        # Confusion matrix
        labels = sorted(self.y.unique())
        cm = confusion_matrix(self.y, y_pred, labels=labels)

        self.report = {
            "cross_validation": {
                "cv_folds": len(self.cv_scores),
                "scores": [round(s, 4) for s in self.cv_scores.tolist()],
                "mean_accuracy": round(self.cv_scores.mean() * 100, 1),
                "std_accuracy": round(self.cv_scores.std() * 100, 1),
            },
            "training_set": {
                "n_samples": len(self.X),
                "n_features": len(self.X.columns),
                "class_distribution": self.y.value_counts().to_dict(),
            },
            "classification_report": {
                k: v for k, v in class_report.items()
                if k not in ["accuracy"]
            },
            "accuracy": round(class_report["accuracy"] * 100, 1),
            "feature_importance_top10": dict(list(self.feature_importance.items())[:10]),
            "confusion_matrix": {
                "labels": labels,
                "matrix": cm.tolist(),
            },
        }

        return self.report

    def predict(self, scenarios: list[dict[str, Any]]) -> list[str]:
        """Predict optimal first item for new scenarios."""
        X_new, _ = build_feature_matrix(scenarios)
        return self.model.predict(X_new).tolist()

    def save_model(self, path: str) -> str:
        """Save trained model to pickle file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        return path

    def save_report(self, path: str) -> str:
        """Save evaluation report to JSON."""
        if self.report is None:
            self.evaluate()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.report, f, indent=2, default=str)
        return path

    def print_summary(self) -> None:
        """Print human-readable training summary."""
        if self.report is None:
            self.evaluate()

        r = self.report
        cv = r["cross_validation"]

        print("=" * 60)
        print("  v2 MODEL TRAINING — RESULTS")
        print("=" * 60)

        print(f"\n--- Training Data ---")
        print(f"  Samples: {r['training_set']['n_samples']:,}")
        print(f"  Features: {r['training_set']['n_features']}")
        print(f"  Class distribution:")
        for cls, count in sorted(r['training_set']['class_distribution'].items()):
            pct = 100 * count / r['training_set']['n_samples']
            print(f"    {cls:15s}: {count:,} ({pct:.1f}%)")

        print(f"\n--- Cross-Validation ({cv['cv_folds']}-fold) ---")
        print(f"  Scores: {cv['scores']}")
        print(f"  Mean accuracy: {cv['mean_accuracy']:.1f}% ± {cv['std_accuracy']:.1f}%")

        print(f"\n--- Training Accuracy ---")
        print(f"  {r['accuracy']:.1f}%")

        print(f"\n--- Top 10 Feature Importances ---")
        for feat, imp in r["feature_importance_top10"].items():
            bar = "█" * int(imp * 50)
            print(f"  {feat:30s}: {imp:.4f} {bar}")

        print(f"\n--- Confusion Matrix ---")
        labels = r["confusion_matrix"]["labels"]
        matrix = r["confusion_matrix"]["matrix"]
        print(f"  {'':15s} " + " ".join(f"{l:>10s}" for l in labels))
        for i, row in enumerate(matrix):
            print(f"  {labels[i]:15s} " + " ".join(f"{v:10d}" for v in row))

        print("\n" + "=" * 60)
