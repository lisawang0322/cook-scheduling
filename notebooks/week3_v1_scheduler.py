"""Week 3–4: v1 Rule-Based Cook Scheduler — Evaluation Pipeline.

This script:
1. Loads the synthetic cook logs and write-off logs generated in Week 1-2.
2. Runs the v1 priority-score scheduler against all decision points.
3. Evaluates how well v1's recommendations correlate with actual write-off outcomes.
4. Saves the evaluation report to output/v1_eval_report.json.
"""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cook_scheduler import CookSchedulerV1, SchedulerEvaluator


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    output_dir = os.path.join(project_root, "output")

    # --- Load data ---
    print("Loading cook logs and write-off logs...")
    cook_logs_path = os.path.join(data_dir, "cook_logs.json")
    writeoff_logs_path = os.path.join(data_dir, "write_off_logs.json")

    if not os.path.exists(cook_logs_path):
        print("ERROR: cook_logs.json not found. Run week1_data_generation.py first.")
        sys.exit(1)

    with open(cook_logs_path) as f:
        cook_logs = json.load(f)
    with open(writeoff_logs_path) as f:
        write_off_logs = json.load(f)

    print(f"  Cook events loaded: {len(cook_logs):,}")
    print(f"  Write-off logs loaded: {len(write_off_logs):,}")

    # --- Demo: Score a single scenario ---
    print("\n" + "=" * 60)
    print("  DEMO: v1 Scheduler on a sample scenario")
    print("=" * 60)

    scheduler = CookSchedulerV1()

    # Create a sample scenario: urban store, 10:15 AM, multiple items due
    sample_events = [
        {
            "cook_event_id": "demo-pizza",
            "item": "pizza",
            "forecast_demand": 12,
            "lowest_cookable_unit": 6,
            "hold_time_hours": 2,
            "exact_multiples": True,
            "window_start_hour": 10,
            "window_end_hour": 12,
            "equipment": "oven",
        },
        {
            "cook_event_id": "demo-wings2h",
            "item": "wings_2h",
            "forecast_demand": 10,
            "lowest_cookable_unit": 5,
            "hold_time_hours": 2,
            "exact_multiples": True,
            "window_start_hour": 10,
            "window_end_hour": 12,
            "equipment": "oven",
        },
        {
            "cook_event_id": "demo-wings4h",
            "item": "wings_4h",
            "forecast_demand": 16,
            "lowest_cookable_unit": 8,
            "hold_time_hours": 4,
            "exact_multiples": True,
            "window_start_hour": 10,
            "window_end_hour": 14,
            "equipment": "oven",
        },
        {
            "cook_event_id": "demo-taquitos",
            "item": "taquitos",
            "forecast_demand": 6,
            "lowest_cookable_unit": 2,
            "hold_time_hours": 4,
            "exact_multiples": False,
            "window_start_hour": 10,
            "window_end_hour": 14,
            "equipment": "roller_grill",
        },
    ]

    schedule = scheduler.schedule_window(10.25, sample_events)

    print(f"\n  Decision point: 10:15 AM (urban store)")
    print(f"\n  Oven ranking:")
    for exp in schedule["explanations"]:
        print(f"    {exp}")
    if schedule["grill_items"]:
        print(f"\n  Roller grill (parallel):")
        for g in schedule["grill_items"]:
            print(f"    taquitos — cook immediately on grill (demand={g['forecast_demand']})")

    # --- Full evaluation ---
    print("\n\nRunning full evaluation on all decision points...")
    evaluator = SchedulerEvaluator(cook_logs, write_off_logs)
    evaluator.evaluate()
    evaluator.print_summary()

    # --- Save report ---
    report_path = os.path.join(output_dir, "v1_eval_report.json")
    evaluator.save_report(report_path)
    print(f"\nEvaluation report saved to: {report_path}")


if __name__ == "__main__":
    main()
