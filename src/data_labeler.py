"""Week 5–6: Data Labeling — Prepare labeled training data for v2 ML model.

For each decision point (moment when multiple oven items need cooking), this module:
1. Extracts scenario features (time, store, demand, item properties).
2. Determines the optimal cook order based on actual write-off outcomes.
3. Labels: "given these features, the optimal first item to cook was X."
4. Filters to high + medium confidence scenarios only.

The labeled dataset is used to train the v2 RandomForest classifier.
"""

import json
import os
from collections import defaultdict
from typing import Any

from src.cook_scheduler import CookSchedulerV1


class DataLabeler:
    """Creates labeled training data from cook logs and write-off outcomes."""

    def __init__(
        self,
        cook_logs: list[dict[str, Any]],
        pos_sales: list[dict[str, Any]],
        write_off_logs: list[dict[str, Any]],
        quality_report: dict[str, Any] | None = None,
    ):
        self.cook_logs = cook_logs
        self.pos_sales = pos_sales
        self.write_off_logs = write_off_logs
        self.quality_report = quality_report

        # Index write-offs and sales by cook_event_id
        self.writeoff_by_event: dict[str, dict[str, Any]] = {
            w["cook_event_id"]: w for w in self.write_off_logs
        }
        self.sales_by_event: dict[str, int] = defaultdict(int)
        for sale in self.pos_sales:
            self.sales_by_event[sale["cook_event_id"]] += sale["quantity"]

        self.labeled_data: list[dict[str, Any]] = []

    def _get_confidence(self, event: dict[str, Any]) -> str:
        """Determine confidence level for a cook event (mirrors DataValidator logic)."""
        event_id = event["cook_event_id"]
        cooked_qty = event["cooked_qty"]
        sold_qty = self.sales_by_event.get(event_id, 0)
        inferred_writeoff = max(0, cooked_qty - sold_qty)

        writeoff_entry = self.writeoff_by_event.get(event_id)
        if writeoff_entry is None:
            return "medium"  # Gap

        logged_writeoff = writeoff_entry["logged_writeoff_qty"]
        difference = abs(logged_writeoff - inferred_writeoff)

        if difference <= 1:
            return "high"
        elif difference <= 2:
            return "medium"
        else:
            return "low"

    def _get_writeoff(self, event: dict[str, Any]) -> int:
        """Get inferred write-off for a cook event."""
        event_id = event["cook_event_id"]
        cooked_qty = event["cooked_qty"]
        sold_qty = self.sales_by_event.get(event_id, 0)
        return max(0, cooked_qty - sold_qty)

    def _get_decision_points(self) -> list[dict[str, Any]]:
        """Group initial cook events into decision points.

        A decision point: same store, same date, same window_start_hour,
        with at least 2 oven items competing for the oven.
        """
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for event in self.cook_logs:
            if event.get("cook_type") != "initial":
                continue
            key = (event["store_id"], event["date"], event["window_start_hour"])
            groups[key].append(event)

        decision_points = []
        for (store_id, date, start_hour), events in groups.items():
            oven_events = [e for e in events if e["item"] in CookSchedulerV1.OVEN_ITEMS]
            if len(oven_events) < 2:
                continue

            # Check confidence: all oven events must be high or medium
            confidences = [self._get_confidence(e) for e in oven_events]
            if "low" in confidences:
                continue  # Skip decision points with low-confidence data

            decision_points.append({
                "store_id": store_id,
                "date": date,
                "decision_hour": start_hour,
                "events": events,
                "oven_events": oven_events,
            })

        return decision_points

    def _extract_features(self, dp: dict[str, Any]) -> dict[str, Any]:
        """Extract scenario features from a decision point."""
        oven_events = dp["oven_events"]
        first_event = oven_events[0]

        # Time features
        features = {
            "store_id": dp["store_id"],
            "store_type": first_event["store_type"],
            "date": dp["date"],
            "day_of_week": first_event["day_of_week"],
            "is_weekend": first_event["is_weekend"],
            "decision_hour": dp["decision_hour"],
            "num_oven_items": len(oven_events),
        }

        # Per-item features (flattened for ML consumption)
        for event in oven_events:
            item = event["item"]
            window_end = event["window_end_hour"]
            if window_end <= event["window_start_hour"]:
                window_end += 24
            time_remaining = window_end - (dp["decision_hour"] + 0.25)

            features[f"{item}_forecast_demand"] = event["forecast_demand"]
            features[f"{item}_lcu"] = event["lowest_cookable_unit"]
            features[f"{item}_hold_time"] = event["hold_time_hours"]
            features[f"{item}_exact_multiples"] = event["exact_multiples"]
            features[f"{item}_time_remaining"] = round(time_remaining, 2)
            features[f"{item}_cooked_qty"] = event["cooked_qty"]
            features[f"{item}_writeoff"] = self._get_writeoff(event)

        return features

    def _determine_optimal_order(self, oven_events: list[dict[str, Any]]) -> list[str]:
        """Determine the optimal cook order using a composite priority score.

        The label reflects what a domain expert would recommend given observable
        conditions: urgency (time pressure), waste risk (perishability × volume),
        and actual outcomes (write-off adjustment).

        Score = urgency_score + waste_risk_score + outcome_bonus

        Where:
        - urgency_score = 1 / time_remaining (higher when window is ending soon)
        - waste_risk_score = hold_time_penalty × demand_density
        - outcome_bonus = -waste_ratio (reward items that were well-managed)

        This creates labels that are both learnable from features AND
        informed by actual outcomes.
        """
        scored = []
        for event in oven_events:
            wo = self._get_writeoff(event)
            cooked = event["cooked_qty"]
            waste_ratio = wo / max(1, cooked)

            # Time remaining from decision point (start of window + 15 min)
            window_end = event["window_end_hour"]
            window_start = event["window_start_hour"]
            if window_end <= window_start:
                window_end += 24
            time_remaining = window_end - (window_start + 0.25)

            # Component scores
            urgency = 1.0 / max(0.1, time_remaining)
            demand_density = event["forecast_demand"] / max(1, event["lowest_cookable_unit"])
            hold_penalty = 1.0 / max(1, event["hold_time_hours"])  # Shorter hold = more urgent

            # Composite: urgency-driven + waste-risk + outcome adjustment
            priority = (
                urgency * 2.0             # Time pressure is most important
                + demand_density * 0.3    # More batches = more oven time needed
                + hold_penalty * 1.0      # Perishability
                - waste_ratio * 1.5       # Penalize items that ended up wasted
            )

            scored.append((event["item"], -priority))  # Negative for ascending sort

        scored.sort(key=lambda x: x[1])
        return [item for item, _ in scored]

    def label(self) -> list[dict[str, Any]]:
        """Generate labeled training data.

        Returns:
            List of labeled scenarios, each with features and optimal_order label.
            Each scenario includes an 'informative' flag indicating whether items
            have different waste outcomes (True = clear signal for learning).
        """
        decision_points = self._get_decision_points()
        self.labeled_data = []

        for dp in decision_points:
            oven_events = dp["oven_events"]
            features = self._extract_features(dp)
            optimal_order = self._determine_optimal_order(oven_events)

            # Determine if scenario is informative (items have different waste ratios)
            waste_ratios = set()
            for event in oven_events:
                wo = self._get_writeoff(event)
                cooked = event["cooked_qty"]
                waste_ratios.add(round(wo / max(1, cooked), 3))

            self.labeled_data.append({
                "scenario_id": f"{dp['store_id']}_{dp['date']}_{dp['decision_hour']}",
                "features": features,
                "optimal_first_item": optimal_order[0],
                "optimal_order": optimal_order,
                "num_items_ranked": len(optimal_order),
                "informative": len(waste_ratios) > 1,
            })

        return self.labeled_data

    def get_informative_scenarios(self) -> list[dict[str, Any]]:
        """Return only scenarios where items have different waste outcomes.

        These are the scenarios where the label carries actual signal
        (not just tiebreaker-driven).
        """
        if not self.labeled_data:
            self.label()
        return [s for s in self.labeled_data if s["informative"]]

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics for the labeled dataset."""
        if not self.labeled_data:
            self.label()

        total = len(self.labeled_data)

        # Distribution of optimal first items
        first_item_dist: dict[str, int] = defaultdict(int)
        for scenario in self.labeled_data:
            first_item_dist[scenario["optimal_first_item"]] += 1

        # Distribution by store type
        store_type_dist: dict[str, int] = defaultdict(int)
        for scenario in self.labeled_data:
            store_type_dist[scenario["features"]["store_type"]] += 1

        # Distribution by number of items ranked
        rank_size_dist: dict[int, int] = defaultdict(int)
        for scenario in self.labeled_data:
            rank_size_dist[scenario["num_items_ranked"]] += 1

        # V1 agreement: how often does optimal match v1's top pick?
        v1_agreement = 0
        for scenario in self.labeled_data:
            features = scenario["features"]
            ranked = CookSchedulerV1.rank_from_features(features)
            if ranked and ranked[0] == scenario["optimal_first_item"]:
                v1_agreement += 1

        return {
            "total_labeled_scenarios": total,
            "first_item_distribution": dict(sorted(first_item_dist.items())),
            "store_type_distribution": dict(sorted(store_type_dist.items())),
            "rank_size_distribution": dict(sorted(rank_size_dist.items())),
            "v1_agreement_pct": round(100 * v1_agreement / total, 1) if total > 0 else 0,
        }

    def save(self, path: str) -> str:
        """Save labeled training set to JSON."""
        if not self.labeled_data:
            self.label()

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.labeled_data, f, indent=2)

        return path
