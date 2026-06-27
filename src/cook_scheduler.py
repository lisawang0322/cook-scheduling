"""v1 Rule-Based Cook Scheduler — Earliest-Deadline-First with demand-weighted priority.

At each decision point (oven is free), the scheduler scores each pending oven item
and recommends cooking the highest-scoring item first. Taquitos are on the roller
grill and cook in parallel (excluded from oven ranking).

Priority Score:
    score = urgency × demand_density × waste_penalty

Where:
    urgency        = 1 / time_until_window_ends  (hours remaining)
    demand_density = forecast_demand / LCU       (batches needed)
    waste_penalty  = 1 + (LCU / forecast_demand) (over-cook risk)

Tiebreaker: shorter hold_time_hours → higher priority.
"""

import json
import os
import random
from collections import defaultdict
from typing import Any


class CookSchedulerV1:
    """v1 deterministic priority-score cook scheduler."""

    # All hot food items that compete for scheduling priority
    OVEN_ITEMS = {
        "wings_bone_in", "wings_boneless",
        "chicken_strip", "chicken_bite", "quesadilla", "chicken_sandwich",
        "potato_wedge", "waffle_tot", "hash_brown",
        "empanada", "chimichanga", "jamaican_turnover", "jamaican_patty", "pupusa",
        "beef_mini_taco", "garlic_knot", "kolache",
        "croissant", "breakfast_sandwich", "sweet_croissant", "danish",
        "pizza_slice", "pizza_stuffed",
        "hot_dog", "sausage", "taquito", "buffalo_roller", "corn_dog",
    }
    # Grill items (subset — noted for equipment routing, still scheduled)
    GRILL_ITEMS = {"hot_dog", "sausage", "taquito", "buffalo_roller", "corn_dog"}

    @classmethod
    def pending_from_features(cls, features: dict) -> tuple[float, list[dict[str, Any]]]:
        """Build pending items for v1 ranking from a scenario features dict.

        Uses OVEN_ITEMS order and the same window reconstruction as
        data_labeler.py v1-agreement check so eval harness metrics match
        output/labeling_report.json v1_agreement_pct.
        """
        decision_hour = features["decision_hour"] + 0.25
        pending: list[dict[str, Any]] = []
        for item in cls.OVEN_ITEMS:
            if f"{item}_forecast_demand" not in features:
                continue
            pending.append({
                "item": item,
                "forecast_demand": features[f"{item}_forecast_demand"],
                "lcu": features[f"{item}_lcu"],
                "hold_time_hours": features[f"{item}_hold_time"],
                "exact_multiples": features.get(f"{item}_exact_multiples", True),
                "window_start_hour": features["decision_hour"],
                "window_end_hour": int(
                    features["decision_hour"]
                    + features[f"{item}_time_remaining"]
                    + 0.25
                ),
            })
        return decision_hour, pending

    @classmethod
    def rank_from_features(cls, features: dict) -> list[str]:
        """Return v1 ranked item IDs for a scenario features dict."""
        decision_hour, pending = cls.pending_from_features(features)
        if not pending:
            return []
        ranked = cls().rank_items(decision_hour, pending)
        return [r["item"] for r in ranked]

    def score_item(self, item: str, forecast_demand: int, lcu: int,
                   hold_time_hours: int, time_remaining_hours: float) -> float:
        """Calculate priority score for a single item at a decision point.

        Args:
            item: Item name.
            forecast_demand: Forecast demand for this window (LCU-valid).
            lcu: Lowest cookable unit.
            hold_time_hours: Hold time in hours.
            time_remaining_hours: Hours until the current window ends.

        Returns:
            Priority score (higher = cook sooner).
        """
        if time_remaining_hours <= 0:
            # Window already passed — maximum urgency
            urgency = 10.0
        else:
            urgency = 1.0 / time_remaining_hours

        demand_density = forecast_demand / max(1, lcu)
        waste_penalty = 1.0 + (lcu / max(1, forecast_demand))

        return urgency * demand_density * waste_penalty

    def rank_items(self, decision_point_hour: float,
                   pending_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rank pending oven items by priority score at a given decision point.

        Args:
            decision_point_hour: Current time as fractional hour (e.g., 10.25 = 10:15 AM).
            pending_items: List of dicts with keys:
                item, forecast_demand, lcu, hold_time_hours, exact_multiples,
                window_start_hour, window_end_hour

        Returns:
            Sorted list of items with added 'score' and 'rank' fields.
        """
        scored = []
        for entry in pending_items:
            # Skip grill items from oven ranking
            if entry["item"] in self.GRILL_ITEMS:
                continue

            window_end = entry["window_end_hour"]
            # Handle overnight windows (e.g., end_hour=2 means 26 in linear time)
            if window_end <= entry["window_start_hour"]:
                window_end += 24

            time_remaining = window_end - decision_point_hour
            score = self.score_item(
                item=entry["item"],
                forecast_demand=entry["forecast_demand"],
                lcu=entry["lcu"],
                hold_time_hours=entry["hold_time_hours"],
                time_remaining_hours=time_remaining,
            )
            scored.append({
                **entry,
                "score": round(score, 4),
                "time_remaining_hours": round(time_remaining, 2),
            })

        # Sort by score desc; tiebreak hold_time asc, then canonical OVEN_ITEMS order
        item_order = type(self)._ITEM_ORDER
        scored.sort(key=lambda x: (
            -x["score"],
            x["hold_time_hours"],
            item_order.get(x["item"], 9999),
        ))

        for i, entry in enumerate(scored):
            entry["rank"] = i + 1

        return scored

    def explain(self, ranked_item: dict[str, Any]) -> str:
        """Generate a template-based explanation for a ranked item.

        Returns a human-readable string explaining why this item has its priority.
        """
        item = ranked_item["item"]
        demand = ranked_item["forecast_demand"]
        lcu = ranked_item["lcu"]
        time_rem = ranked_item["time_remaining_hours"]
        rank = ranked_item["rank"]
        score = ranked_item["score"]
        batches = demand // lcu if lcu > 0 else demand

        if rank == 1:
            urgency_text = "COOK NOW"
        elif time_rem <= 1.0:
            urgency_text = "urgent"
        elif time_rem <= 2.0:
            urgency_text = "soon"
        else:
            urgency_text = "can wait"

        return (
            f"#{rank} {item} [{urgency_text}]: "
            f"window ends in {time_rem:.1f}h, "
            f"demand={demand} ({batches} batch{'es' if batches != 1 else ''} of {lcu}), "
            f"score={score:.2f}"
        )

    def schedule_window(self, decision_point_hour: float,
                        cook_events: list[dict[str, Any]]) -> dict[str, Any]:
        """Given cook events active at a decision point, produce a full schedule.

        Args:
            decision_point_hour: Current time as fractional hour.
            cook_events: List of cook log entries active at this time.

        Returns:
            Dict with 'oven_ranking', 'grill_items', and 'explanations'.
        """
        # Build pending items from cook events
        pending = []
        grill = []
        for event in cook_events:
            entry = {
                "item": event["item"],
                "forecast_demand": event["forecast_demand"],
                "lcu": event["lowest_cookable_unit"],
                "hold_time_hours": event["hold_time_hours"],
                "exact_multiples": event["exact_multiples"],
                "window_start_hour": event["window_start_hour"],
                "window_end_hour": event["window_end_hour"],
                "cook_event_id": event["cook_event_id"],
                "equipment": event.get("equipment", "oven"),
            }
            if event["item"] in self.GRILL_ITEMS:
                grill.append(entry)
            else:
                pending.append(entry)

        ranked = self.rank_items(decision_point_hour, pending)
        explanations = [self.explain(r) for r in ranked]

        return {
            "decision_point_hour": decision_point_hour,
            "oven_ranking": ranked,
            "grill_items": grill,
            "explanations": explanations,
        }


CookSchedulerV1._ITEM_ORDER = {
    item: i for i, item in enumerate(CookSchedulerV1.OVEN_ITEMS)
}


class AssociateBaseline:
    """Simulates real associate cook-order decision-making.

    Based on observations, associates use a mix of:
    - Expiration proximity (40%): pick the item closest to expiring
    - Habit/familiarity (30%): default to pizza (most common, always visible)
    - Random/convenience (20%): whatever is closest in the freezer
    - Demand-driven (10%): pick highest-demand item

    This mimics the inconsistent, partially-informed decisions made in stores
    and serves as the realistic baseline ("what happens today without a system").
    """

    OVEN_ITEMS = {
        "wings_bone_in", "wings_boneless",
        "chicken_strip", "chicken_bite", "quesadilla", "chicken_sandwich",
        "potato_wedge", "waffle_tot", "hash_brown",
        "empanada", "chimichanga", "jamaican_turnover", "jamaican_patty", "pupusa",
        "beef_mini_taco", "garlic_knot", "kolache",
        "croissant", "breakfast_sandwich", "sweet_croissant", "danish",
        "pizza_slice", "pizza_stuffed",
        "hot_dog", "sausage", "taquito", "buffalo_roller", "corn_dog",
    }

    # How often each strategy is used
    STRATEGY_WEIGHTS = {
        "expiration": 0.40,   # pick item closest to window end
        "habit": 0.30,       # default to familiar high-volume items
        "random": 0.20,      # random choice
        "demand": 0.10,      # pick highest demand item
    }

    # Habit preferences: associates default to familiar high-volume items
    HABIT_PRIORITY = [
        "taquito", "hot_dog", "sausage", "pizza_slice",
        "wings_bone_in", "chicken_sandwich", "corn_dog",
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def pick_first_item(self, oven_events: list[dict[str, Any]]) -> str:
        """Simulate which item an associate would cook first.

        Returns the item name chosen by the associate's decision process.
        """
        if not oven_events:
            return ""
        if len(oven_events) == 1:
            return oven_events[0]["item"]

        # Roll for strategy
        roll = self.rng.random()
        cumulative = 0.0
        strategy = "random"
        for strat, weight in self.STRATEGY_WEIGHTS.items():
            cumulative += weight
            if roll <= cumulative:
                strategy = strat
                break

        if strategy == "expiration":
            # Pick item with shortest time remaining (closest to window end)
            # Associates often grab "the one about to expire" even if it's
            # already cooked — here we model it as shortest hold time first
            return min(oven_events, key=lambda e: e["hold_time_hours"])["item"]

        elif strategy == "habit":
            # Default to pizza if present, else familiar order
            items_present = {e["item"] for e in oven_events}
            for preferred in self.HABIT_PRIORITY:
                if preferred in items_present:
                    return preferred
            return oven_events[0]["item"]

        elif strategy == "demand":
            # Pick highest demand item (associates sometimes check the forecast)
            return max(oven_events, key=lambda e: e["forecast_demand"])["item"]

        else:  # random
            return self.rng.choice(oven_events)["item"]

    def rank_items(self, oven_events: list[dict[str, Any]]) -> list[str]:
        """Produce a full ranking by repeatedly picking from remaining items."""
        remaining = list(oven_events)
        order = []
        while remaining:
            pick = self.pick_first_item(remaining)
            order.append(pick)
            remaining = [e for e in remaining if e["item"] != pick]
        return order


class SchedulerEvaluator:
    """Evaluates the v1 scheduler against actual write-off outcomes."""

    def __init__(self, cook_logs: list[dict[str, Any]],
                 write_off_logs: list[dict[str, Any]]):
        self.cook_logs = cook_logs
        self.write_off_logs = write_off_logs
        self.scheduler = CookSchedulerV1()

        # Index write-offs by cook_event_id
        self.writeoff_by_event: dict[str, dict[str, Any]] = {
            w["cook_event_id"]: w for w in self.write_off_logs
        }

        self.results: list[dict[str, Any]] = []
        self.report: dict[str, Any] | None = None

    def _get_decision_points(self) -> list[dict[str, Any]]:
        """Group cook events into decision points (same store, same day, same window start).

        A decision point is a moment when an associate must choose what to cook.
        We approximate this as: for each store-day, at each unique window_start_hour,
        which oven items are pending?
        """
        # Group by (store_id, date, window_start_hour)
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for event in self.cook_logs:
            # Only consider initial cooks for scheduling decisions
            if event.get("cook_type") != "initial":
                continue
            key = (event["store_id"], event["date"], event["window_start_hour"])
            groups[key].append(event)

        decision_points = []
        for (store_id, date, start_hour), events in groups.items():
            # Filter to oven items only for ranking decisions
            oven_events = [e for e in events if e["item"] in CookSchedulerV1.OVEN_ITEMS]
            if len(oven_events) < 2:
                # Need at least 2 oven items to make a ranking decision
                continue
            decision_points.append({
                "store_id": store_id,
                "date": date,
                "decision_hour": start_hour + 0.25,  # 15 min into window
                "events": events,  # Include all items for schedule_window
                "oven_events": oven_events,
            })

        return decision_points

    def _get_writeoff_for_event(self, event: dict[str, Any]) -> int:
        """Get the actual write-off for a cook event."""
        wo = self.writeoff_by_event.get(event["cook_event_id"])
        if wo is None:
            return 0
        return wo.get("inferred_writeoff_qty", 0)

    def evaluate(self) -> dict[str, Any]:
        """Run evaluation: does the v1 ranking correlate with lower write-offs?

        For each decision point, compare:
        - v1's recommended first-cook item
        - Which item actually had the highest write-off (should have been cooked later)
        - Which item had the lowest write-off (was managed well)
        """
        decision_points = self._get_decision_points()
        self.results = []

        correct_top1 = 0
        total_decisions = 0
        writeoff_savings_vs_random = 0.0

        for dp in decision_points:
            events = dp["events"]
            oven_events = dp["oven_events"]
            decision_hour = dp["decision_hour"]

            # Get v1 ranking
            schedule = self.scheduler.schedule_window(decision_hour, events)
            ranked = schedule["oven_ranking"]

            if len(ranked) < 2:
                continue

            total_decisions += 1

            # Get actual write-offs for each ranked oven item
            event_writeoffs = {}
            for event in oven_events:
                wo = self._get_writeoff_for_event(event)
                event_writeoffs[event["cook_event_id"]] = wo

            # The item v1 says to cook first
            v1_first = ranked[0]
            v1_first_writeoff = event_writeoffs.get(v1_first["cook_event_id"], 0)

            # The item with the lowest actual write-off (best outcome)
            best_event_id = min(event_writeoffs, key=event_writeoffs.get)
            best_writeoff = event_writeoffs[best_event_id]

            # Average write-off (random baseline)
            avg_writeoff = (sum(event_writeoffs.values()) / len(event_writeoffs)
                           if event_writeoffs else 0)

            # Did v1 pick the item with lowest write-off?
            is_correct = v1_first["cook_event_id"] == best_event_id
            if is_correct:
                correct_top1 += 1

            # Savings vs random
            savings = avg_writeoff - v1_first_writeoff
            writeoff_savings_vs_random += savings

            self.results.append({
                "store_id": dp["store_id"],
                "date": dp["date"],
                "decision_hour": decision_hour,
                "v1_first_item": v1_first["item"],
                "v1_first_score": v1_first["score"],
                "v1_first_writeoff": v1_first_writeoff,
                "best_item_writeoff": best_writeoff,
                "avg_writeoff": round(avg_writeoff, 2),
                "is_correct_top1": is_correct,
                "savings_vs_random": round(savings, 2),
                "num_oven_items": len(ranked),
                "explanations": schedule["explanations"],
            })

        # Summary metrics
        top1_accuracy = correct_top1 / total_decisions if total_decisions > 0 else 0
        avg_savings = writeoff_savings_vs_random / total_decisions if total_decisions > 0 else 0

        # Write-off comparison
        v1_writeoffs = [r["v1_first_writeoff"] for r in self.results]
        best_writeoffs = [r["best_item_writeoff"] for r in self.results]
        avg_v1_wo = sum(v1_writeoffs) / len(v1_writeoffs) if v1_writeoffs else 0
        avg_best_wo = sum(best_writeoffs) / len(best_writeoffs) if best_writeoffs else 0

        self.report = {
            "total_decision_points": total_decisions,
            "top1_accuracy_pct": round(top1_accuracy * 100, 1),
            "avg_v1_first_writeoff": round(avg_v1_wo, 2),
            "avg_best_writeoff": round(avg_best_wo, 2),
            "avg_savings_vs_random": round(avg_savings, 2),
            "total_savings_vs_random": round(writeoff_savings_vs_random, 1),
            "v1_valid_rankings_pct": 100.0,  # v1 always produces a valid ranking
        }

        return self.report

    def save_report(self, path: str) -> str:
        """Save evaluation report to JSON."""
        if self.report is None:
            self.evaluate()

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        output = {
            "summary": self.report,
            "sample_decisions": self.results[:20],  # First 20 for inspection
        }
        with open(path, "w") as f:
            json.dump(output, f, indent=2)

        return path

    def print_summary(self) -> None:
        """Print a human-readable evaluation summary."""
        if self.report is None:
            self.evaluate()

        r = self.report
        print("=" * 60)
        print("  v1 COOK SCHEDULER — EVALUATION REPORT")
        print("=" * 60)
        print(f"\nTotal decision points evaluated: {r['total_decision_points']}")
        print(f"\n--- Accuracy ---")
        print(f"  Top-1 accuracy (v1 picks lowest-writeoff item): {r['top1_accuracy_pct']:.1f}%")
        print(f"  v1 always produces valid ranking: {r['v1_valid_rankings_pct']:.0f}%")
        print(f"\n--- Write-Off Impact ---")
        print(f"  Avg write-off for v1's #1 pick:    {r['avg_v1_first_writeoff']:.2f} units")
        print(f"  Avg write-off for best possible:   {r['avg_best_writeoff']:.2f} units")
        print(f"  Avg savings vs. random selection:  {r['avg_savings_vs_random']:.2f} units/decision")
        print(f"  Total savings vs. random:          {r['total_savings_vs_random']:.1f} units")

        # Show a few example explanations
        if self.results:
            print(f"\n--- Sample Explanations (first 3 decisions) ---")
            for i, result in enumerate(self.results[:3]):
                print(f"\n  Decision #{i+1} ({result['store_id']}, {result['date']}, "
                      f"{result['decision_hour']:.2f}h):")
                for exp in result["explanations"]:
                    print(f"    {exp}")
                print(f"    -> v1 picked: {result['v1_first_item']} "
                      f"(writeoff={result['v1_first_writeoff']}, "
                      f"{'correct' if result['is_correct_top1'] else 'not optimal'})")

        print("\n" + "=" * 60)
