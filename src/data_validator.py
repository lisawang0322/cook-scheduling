import json
import os
from collections import defaultdict
from typing import Any


class DataValidator:
    """Validates data quality by comparing inferred vs. logged write-offs."""

    CONFIDENCE_THRESHOLDS = {
        "high": 1,      # logged = inferred (or ±1 unit)
        "medium": 2,    # ±2 unit counting errors or gaps
        "low": 3,       # ±3+ unit major discrepancies
    }

    def __init__(
        self,
        cook_logs: list[dict[str, Any]],
        pos_sales: list[dict[str, Any]],
        write_off_logs: list[dict[str, Any]],
    ):
        self.cook_logs = cook_logs
        self.pos_sales = pos_sales
        self.write_off_logs = write_off_logs

        # Index write-offs and sales by cook_event_id
        self.writeoff_by_event: dict[str, dict[str, Any]] = {
            w["cook_event_id"]: w for w in self.write_off_logs
        }
        self.sales_by_event: dict[str, int] = defaultdict(int)
        for sale in self.pos_sales:
            self.sales_by_event[sale["cook_event_id"]] += sale["quantity"]

        self.report: dict[str, Any] | None = None

    def _classify_confidence(
        self, cook_event: dict[str, Any]
    ) -> dict[str, Any]:
        """Classify a single cook event's data quality confidence level."""
        event_id = cook_event["cook_event_id"]
        cooked_qty = cook_event["cooked_qty"]
        sold_qty = self.sales_by_event.get(event_id, 0)
        inferred_writeoff = max(0, cooked_qty - sold_qty)

        writeoff_entry = self.writeoff_by_event.get(event_id)

        if writeoff_entry is None:
            # Gap — no write-off logged
            return {
                "cook_event_id": event_id,
                "item": cook_event["item"],
                "store_type": cook_event["store_type"],
                "window": cook_event["window"],
                "date": cook_event["date"],
                "cooked_qty": cooked_qty,
                "sold_qty": sold_qty,
                "inferred_writeoff": inferred_writeoff,
                "logged_writeoff": None,
                "difference": None,
                "confidence": "medium",
                "quality_issue": "gap",
            }

        logged_writeoff = writeoff_entry["logged_writeoff_qty"]
        difference = abs(logged_writeoff - inferred_writeoff)

        if difference <= 1:
            confidence = "high"
            quality_issue = "accurate"
        elif difference <= 2:
            confidence = "medium"
            quality_issue = "counting_error"
        else:
            confidence = "low"
            quality_issue = "major_discrepancy"

        return {
            "cook_event_id": event_id,
            "item": cook_event["item"],
            "store_type": cook_event["store_type"],
            "window": cook_event["window"],
            "date": cook_event["date"],
            "cooked_qty": cooked_qty,
            "sold_qty": sold_qty,
            "inferred_writeoff": inferred_writeoff,
            "logged_writeoff": logged_writeoff,
            "difference": difference,
            "confidence": confidence,
            "quality_issue": quality_issue,
        }

    def validate(self) -> dict[str, Any]:
        """Run full validation and produce quality report."""
        classifications = [self._classify_confidence(c) for c in self.cook_logs]
        total = len(classifications)

        # Confidence breakdown
        confidence_counts: dict[str, int] = defaultdict(int)
        for c in classifications:
            confidence_counts[c["confidence"]] += 1

        # Quality issue breakdown
        issue_counts: dict[str, int] = defaultdict(int)
        for c in classifications:
            issue_counts[c["quality_issue"]] += 1

        # By store type breakdown
        store_type_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for c in classifications:
            store_type_stats[c["store_type"]][c["confidence"]] += 1
            store_type_stats[c["store_type"]]["total"] += 1

        # Build report
        self.report = {
            "total_cook_events": total,
            "confidence_breakdown": {
                level: {
                    "count": confidence_counts.get(level, 0),
                    "percentage": round(
                        100 * confidence_counts.get(level, 0) / total, 1
                    )
                    if total > 0
                    else 0,
                }
                for level in ["high", "medium", "low"]
            },
            "quality_issues": {
                issue: {
                    "count": issue_counts.get(issue, 0),
                    "percentage": round(
                        100 * issue_counts.get(issue, 0) / total, 1
                    )
                    if total > 0
                    else 0,
                }
                for issue in ["accurate", "gap", "counting_error", "major_discrepancy"]
            },
            "by_store_type": {
                st: {
                    "total": stats["total"],
                    "high": stats.get("high", 0),
                    "medium": stats.get("medium", 0),
                    "low": stats.get("low", 0),
                    "high_pct": round(100 * stats.get("high", 0) / stats["total"], 1)
                    if stats["total"] > 0
                    else 0,
                    "medium_pct": round(100 * stats.get("medium", 0) / stats["total"], 1)
                    if stats["total"] > 0
                    else 0,
                    "low_pct": round(100 * stats.get("low", 0) / stats["total"], 1)
                    if stats["total"] > 0
                    else 0,
                }
                for st, stats in sorted(store_type_stats.items())
            },
            "usable_for_training_pct": round(
                100
                * (confidence_counts.get("high", 0) + confidence_counts.get("medium", 0))
                / total,
                1,
            )
            if total > 0
            else 0,
            "classifications": classifications,
        }

        return self.report

    def save_report(self, path: str) -> str:
        """Save the quality report to a JSON file (without per-event classifications)."""
        if self.report is None:
            self.validate()

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # Save a summary version (without the large classifications list)
        summary = {k: v for k, v in self.report.items() if k != "classifications"}
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

        return path

    def print_summary(self) -> None:
        """Print a human-readable summary of the quality report."""
        if self.report is None:
            self.validate()

        r = self.report
        print("=" * 60)
        print("  DATA QUALITY REPORT")
        print("=" * 60)
        print(f"\nTotal cook events: {r['total_cook_events']}")

        print("\n--- Confidence Breakdown ---")
        for level in ["high", "medium", "low"]:
            info = r["confidence_breakdown"][level]
            print(f"  {level.capitalize():10s}: {info['count']:5d}  ({info['percentage']:.1f}%)")

        print(f"\n  Usable for training: {r['usable_for_training_pct']:.1f}%")

        print("\n--- Quality Issues ---")
        for issue in ["accurate", "gap", "counting_error", "major_discrepancy"]:
            info = r["quality_issues"][issue]
            label = issue.replace("_", " ").capitalize()
            print(f"  {label:25s}: {info['count']:5d}  ({info['percentage']:.1f}%)")

        print("\n--- By Store Type ---")
        for st, stats in r["by_store_type"].items():
            print(f"\n  {st.capitalize()} (n={stats['total']}):")
            print(f"    High:   {stats['high']:5d} ({stats['high_pct']:.1f}%)")
            print(f"    Medium: {stats['medium']:5d} ({stats['medium_pct']:.1f}%)")
            print(f"    Low:    {stats['low']:5d} ({stats['low_pct']:.1f}%)")

        print("\n" + "=" * 60)
