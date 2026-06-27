"""v3 Listwise LambdaRank Model — LightGBM LambdaRank for cook-order prediction.

Instead of expanding each scenario into C(n,2) pairwise rows and predicting
"should A go before B?", this model treats each scenario as a query group and
each present oven item as a document.  LightGBM optimises NDCG directly via the
lambdarank loss, handling the listwise structure natively and eliminating the
post-hoc win-counting / proba-accumulation step.

Data shape:
  - One row per (scenario, present_item) — not per pair.
  - group[i] = number of items in scenario i (must be contiguous).
  - y[row] = graded relevance derived from optimal_order position:
      rel = (n_items - 1) - position_in_optimal_order  (highest = cook first)

Inference:
  - model.predict(X_query) → raw scores per item.
  - Sort items by score descending → cook order.
"""

import json
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

from src.pairwise_trainer import OVEN_ITEMS, compute_historical_features  # single source of truth


# Maximum number of distinct relevance grades across all scenarios.
# With 28 oven items per scenario, grades run 0..27.  We supply a linear
# label_gain=[0,1,...,N_MAX_GRADES-1] to avoid the default 2^label
# blow-up that makes grade-27 items astronomically more important.
N_MAX_GRADES = len(OVEN_ITEMS)  # 28


def build_listwise_data(
    labeled_data: list[dict[str, Any]],
    historical: dict[str, dict],
) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    """Convert labeled scenarios into a listwise dataset for LGBMRanker.

    Returns:
        X        — DataFrame, one row per (scenario, present_item).
                   Rows for the same scenario are contiguous.
        y        — int ndarray of graded relevance labels (0 = lowest priority).
        group    — list[int] of per-scenario row counts (sums to len(X)).
    """
    store_map = {"urban": 2, "suburban": 1, "highway": 0}
    day_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6,
    }

    rows: list[dict] = []
    labels: list[int] = []
    group: list[int] = []

    for scenario in labeled_data:
        features = scenario["features"]
        optimal_order: list[str] = scenario["optimal_order"]

        present_items = [
            item for item in OVEN_ITEMS
            if f"{item}_forecast_demand" in features
        ]
        if not present_items:
            continue

        # Build rank_lookup: item → position in optimal_order (0 = best).
        rank_lookup: dict[str, int] = {
            item: idx for idx, item in enumerate(optimal_order)
        }
        n_items = len(present_items)
        n_ranked = len(optimal_order)

        decision_hour = features["decision_hour"]
        store_type = features["store_type"]
        store_encoded = store_map.get(store_type, 1)
        is_weekend = int(features["is_weekend"])
        day_of_week = day_map.get(features["day_of_week"], 0)

        scenario_rows: list[dict] = []
        scenario_labels: list[int] = []

        for item in present_items:
            demand = features[f"{item}_forecast_demand"]
            lcu = features[f"{item}_lcu"]
            hold = features[f"{item}_hold_time"]
            time_rem = features[f"{item}_time_remaining"]
            cooked = features.get(f"{item}_cooked_qty", 0)
            urgency = round(1.0 / max(0.01, time_rem), 6)
            demand_density = round(demand / max(1, lcu), 6)

            hist = historical.get(item, {})
            hist_wo_hour = hist.get("avg_writeoff_by_hour", {}).get(decision_hour, 0)
            hist_wo_store = hist.get("avg_writeoff_by_store_type", {}).get(store_type, 0)
            hist_wo_overall = hist.get("overall_avg_writeoff", 0)
            hist_demand_hour = hist.get("avg_demand_by_hour", {}).get(decision_hour, 0)
            demand_vs_avg = round(demand - hist_demand_hour, 4)

            row = {
                "decision_hour": decision_hour,
                "store_type_encoded": store_encoded,
                "is_weekend": is_weekend,
                "day_of_week": day_of_week,
                "num_items": n_items,
                # Per-item signals
                "demand": demand,
                "lcu": lcu,
                "hold_time": hold,
                "time_remaining": time_rem,
                "cooked_qty": cooked,
                "urgency": urgency,
                "demand_density": demand_density,
                # Historical waste / demand signals
                "hist_wo_hour": round(hist_wo_hour, 4),
                "hist_wo_store": round(hist_wo_store, 4),
                "hist_wo_overall": round(hist_wo_overall, 4),
                "demand_vs_avg": demand_vs_avg,
            }
            scenario_rows.append(row)

            # Graded relevance: best item gets grade (n_ranked - 1), worst gets 0.
            # Items absent from optimal_order (shouldn't happen) get 0.
            pos = rank_lookup.get(item, n_ranked - 1)
            rel = max(0, (n_ranked - 1) - pos)
            scenario_labels.append(rel)

        rows.extend(scenario_rows)
        labels.extend(scenario_labels)
        group.append(len(scenario_rows))

    X = pd.DataFrame(rows)
    y = np.array(labels, dtype=np.int32)
    return X, y, group


class LambdaRankTrainer:
    """Trains a v3 LightGBM LambdaRank cook-order model.

    Drop-in replacement for PairwiseModelTrainer at the ranking interface:
    rank_items() has the same signature and return type.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        # Linear label_gain avoids 2^label explosion at 28 grades.
        label_gain = list(range(N_MAX_GRADES))

        self.model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
            label_gain=label_gain,
            verbose=-1,
        )
        self.eval_at = [1, 3, 5]
        self.random_state = random_state

        self.X: pd.DataFrame | None = None
        self.y: np.ndarray | None = None
        self.group: list[int] | None = None
        self.feature_names: list[str] | None = None
        self.historical: dict[str, dict] | None = None
        self.feature_importance: dict[str, float] | None = None
        self.report: dict[str, Any] | None = None

    def prepare_data(
        self,
        labeled_data: list[dict[str, Any]],
        historical: dict[str, dict],
    ) -> None:
        """Build listwise feature matrix from labeled scenarios."""
        self.historical = historical
        self.X, self.y, self.group = build_listwise_data(labeled_data, historical)
        self.feature_names = list(self.X.columns)

    def train(self) -> None:
        """Fit LGBMRanker on all prepared data."""
        self.model.fit(
            self.X,
            self.y,
            group=self.group,
        )
        importances = self.model.feature_importances_
        self.feature_importance = dict(sorted(
            zip(self.feature_names, importances.tolist()),
            key=lambda x: -x[1],
        ))

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _compute_ndcg_at_k(
        self,
        labeled_data: list[dict[str, Any]],
        k_values: list[int],
    ) -> dict[str, float]:
        """Compute NDCG@k for each k in k_values over all scenarios."""
        store_map = {"urban": 2, "suburban": 1, "highway": 0}
        day_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6,
        }

        ndcg_accum: dict[int, list[float]] = {k: [] for k in k_values}

        for scenario in labeled_data:
            features = scenario["features"]
            optimal_order: list[str] = scenario["optimal_order"]

            present_items = [
                item for item in OVEN_ITEMS
                if f"{item}_forecast_demand" in features
            ]
            if not present_items:
                continue

            n_ranked = len(optimal_order)
            rank_lookup = {item: idx for idx, item in enumerate(optimal_order)}

            decision_hour = features["decision_hour"]
            store_type = features["store_type"]

            item_rows = []
            true_rels = []
            for item in present_items:
                demand = features[f"{item}_forecast_demand"]
                lcu = features[f"{item}_lcu"]
                hold = features[f"{item}_hold_time"]
                time_rem = features[f"{item}_time_remaining"]
                cooked = features.get(f"{item}_cooked_qty", 0)
                urgency = 1.0 / max(0.01, time_rem)
                density = demand / max(1, lcu)

                hist = self.historical.get(item, {})
                hist_wo_hour = hist.get("avg_writeoff_by_hour", {}).get(decision_hour, 0)
                hist_wo_store = hist.get("avg_writeoff_by_store_type", {}).get(store_type, 0)
                hist_wo_overall = hist.get("overall_avg_writeoff", 0)
                hist_demand_hour = hist.get("avg_demand_by_hour", {}).get(decision_hour, 0)

                item_rows.append({
                    "decision_hour": decision_hour,
                    "store_type_encoded": store_map.get(store_type, 1),
                    "is_weekend": int(features["is_weekend"]),
                    "day_of_week": day_map.get(features["day_of_week"], 0),
                    "num_items": features["num_oven_items"],
                    "demand": demand,
                    "lcu": lcu,
                    "hold_time": hold,
                    "time_remaining": time_rem,
                    "cooked_qty": cooked,
                    "urgency": round(urgency, 6),
                    "demand_density": round(density, 6),
                    "hist_wo_hour": round(hist_wo_hour, 4),
                    "hist_wo_store": round(hist_wo_store, 4),
                    "hist_wo_overall": round(hist_wo_overall, 4),
                    "demand_vs_avg": round(demand - hist_demand_hour, 4),
                })
                pos = rank_lookup.get(item, n_ranked - 1)
                true_rels.append(max(0, (n_ranked - 1) - pos))

            X_q = pd.DataFrame(item_rows)
            scores = self.model.predict(X_q)

            # Sort items by predicted score descending
            predicted_order = np.argsort(-scores)
            true_rels_arr = np.array(true_rels, dtype=float)

            # NDCG@k
            ideal_rels = np.sort(true_rels_arr)[::-1]
            for k in k_values:
                top_k = predicted_order[:k]
                dcg = sum(
                    (2 ** true_rels_arr[idx] - 1) / np.log2(r + 2)
                    for r, idx in enumerate(top_k)
                )
                idcg = sum(
                    (2 ** rel - 1) / np.log2(r + 2)
                    for r, rel in enumerate(ideal_rels[:k])
                )
                ndcg_accum[k].append(dcg / idcg if idcg > 0 else 0.0)

        return {
            f"ndcg_at_{k}": round(float(np.mean(ndcg_accum[k])), 4)
            for k in k_values
        }

    def evaluate_ranking_accuracy(
        self, labeled_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Evaluate top-1 accuracy: does model's top item match optimal_first_item?

        Same definition as PairwiseModelTrainer.evaluate_ranking_accuracy for
        apples-to-apples comparison with v2.2.
        """
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
            if predicted_order and predicted_order[0] == expected_first:
                correct += 1
            total += 1

        return {
            "ranking_top1_accuracy": round(100 * correct / total, 1) if total else 0,
            "total_evaluated": total,
        }

    def evaluate(
        self,
        labeled_data: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Compute NDCG@1/3/5 and top-1 accuracy; populate self.report."""
        if labeled_data is None:
            # Re-construct scenarios from stored X/y/group for a rough
            # training-set measure — full NDCG requires the original dicts.
            # Callers should pass labeled_data directly for honest reporting.
            raise ValueError(
                "Pass labeled_data to evaluate(); "
                "LambdaRankTrainer requires original scenario dicts for NDCG."
            )

        ndcg = self._compute_ndcg_at_k(labeled_data, self.eval_at)
        top1 = self.evaluate_ranking_accuracy(labeled_data)

        self.report = {
            "model_type": "lambdarank_lightgbm",
            "training_set": {
                "n_scenarios": len(self.group) if self.group else 0,
                "n_rows": len(self.X) if self.X is not None else 0,
                "n_features": len(self.feature_names) if self.feature_names else 0,
            },
            **ndcg,
            "ranking_accuracy": top1,
            "feature_importance_top10": (
                dict(list(self.feature_importance.items())[:10])
                if self.feature_importance else {}
            ),
        }
        return self.report

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def rank_items(
        self,
        scenario_features: dict[str, Any],
        present_items: list[str],
    ) -> list[str]:
        """Rank present_items for a single scenario using the trained model.

        Same signature as PairwiseModelTrainer.rank_items — drop-in compatible.
        Returns items sorted from highest to lowest predicted score.
        """
        if len(present_items) < 2:
            return present_items

        store_map = {"urban": 2, "suburban": 1, "highway": 0}
        day_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6,
        }

        decision_hour = scenario_features["decision_hour"]
        store_type = scenario_features["store_type"]

        item_rows = []
        for item in present_items:
            demand = scenario_features.get(f"{item}_forecast_demand", 0)
            lcu = scenario_features.get(f"{item}_lcu", 1)
            hold = scenario_features.get(f"{item}_hold_time", 2)
            time_rem = scenario_features.get(f"{item}_time_remaining", 2)
            cooked = scenario_features.get(f"{item}_cooked_qty", 0)
            urgency = 1.0 / max(0.01, time_rem)
            density = demand / max(1, lcu)

            hist = (self.historical or {}).get(item, {})
            hist_wo_hour = hist.get("avg_writeoff_by_hour", {}).get(decision_hour, 0)
            hist_wo_store = hist.get("avg_writeoff_by_store_type", {}).get(store_type, 0)
            hist_wo_overall = hist.get("overall_avg_writeoff", 0)
            hist_demand_hour = hist.get("avg_demand_by_hour", {}).get(decision_hour, 0)

            item_rows.append({
                "decision_hour": decision_hour,
                "store_type_encoded": store_map.get(store_type, 1),
                "is_weekend": int(scenario_features.get("is_weekend", False)),
                "day_of_week": day_map.get(scenario_features.get("day_of_week", "Monday"), 0),
                "num_items": scenario_features.get("num_oven_items", len(present_items)),
                "demand": demand,
                "lcu": lcu,
                "hold_time": hold,
                "time_remaining": time_rem,
                "cooked_qty": cooked,
                "urgency": round(urgency, 6),
                "demand_density": round(density, 6),
                "hist_wo_hour": round(hist_wo_hour, 4),
                "hist_wo_store": round(hist_wo_store, 4),
                "hist_wo_overall": round(hist_wo_overall, 4),
                "demand_vs_avg": round(demand - hist_demand_hour, 4),
            })

        X_q = pd.DataFrame(item_rows)
        scores = self.model.predict(X_q)
        return [item for item, _ in sorted(
            zip(present_items, scores), key=lambda x: -x[1]
        )]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str) -> str:
        """Pickle model + historical + metadata to path."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "historical": self.historical,
                "feature_names": self.feature_names,
                "model_type": "lambdarank_lightgbm",
            }, f)
        return path

    def save_report(self, path: str) -> str:
        """Write evaluation report to JSON."""
        if self.report is None:
            raise RuntimeError("Call evaluate() before save_report().")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.report, f, indent=2, default=str)
        return path

    def print_summary(self) -> None:
        """Print human-readable training summary."""
        if self.report is None:
            raise RuntimeError("Call evaluate() before print_summary().")

        r = self.report
        ts = r["training_set"]
        ra = r["ranking_accuracy"]

        print("=" * 60)
        print("  v3 MODEL — LightGBM LambdaRank — RESULTS")
        print("=" * 60)

        print(f"\n--- Training Data ---")
        print(f"  Scenarios: {ts['n_scenarios']:,}")
        print(f"  Item-rows: {ts['n_rows']:,}")
        print(f"  Features:  {ts['n_features']}")

        print(f"\n--- NDCG (on eval set) ---")
        for k in self.eval_at:
            key = f"ndcg_at_{k}"
            print(f"  NDCG@{k}: {r.get(key, 0):.4f}")

        print(f"\n--- Top-1 Ranking Accuracy ---")
        print(f"  {ra['ranking_top1_accuracy']:.1f}%  ({ra['total_evaluated']:,} scenarios)")

        print(f"\n--- Top 10 Feature Importances ---")
        for feat, imp in r["feature_importance_top10"].items():
            bar = "█" * max(1, int(imp / max(r["feature_importance_top10"].values()) * 30))
            print(f"  {feat:20s}: {imp:6.0f} {bar}")

        print("\n" + "=" * 60)
