"""
Week 1: Synthetic Data Generation & Quality Assessment
=======================================================
Generates 180 days of synthetic cook scheduling data across 3 store types
and 4 dayparts, then validates data quality for training readiness.
"""

import os
import sys

# Ensure project root is on the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.synthetic_data_generator import SyntheticDataGenerator
from src.data_validator import DataValidator


def main():
    # ── Step 1: Generate synthetic data ──────────────────────────────
    print("Generating 180 days of synthetic cook scheduling data...")
    generator = SyntheticDataGenerator(
        seed=42,
        num_days=180,
        output_dir=os.path.join(project_root, "data"),
    )
    data = generator.generate()

    print(f"  Cook events:    {len(data['cook_logs']):,}")
    print(f"  POS sales:      {len(data['pos_sales']):,}")
    print(f"  Write-off logs: {len(data['write_off_logs']):,}")

    # Save raw data
    paths = generator.save()
    print("\nData saved to:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    # ── Step 2: Validate data quality ────────────────────────────────
    print("\nRunning data quality validation...")
    validator = DataValidator(
        cook_logs=data["cook_logs"],
        pos_sales=data["pos_sales"],
        write_off_logs=data["write_off_logs"],
    )
    validator.validate()
    validator.print_summary()

    # ── Step 3: Save quality report ──────────────────────────────────
    report_path = os.path.join(project_root, "output", "quality_report.json")
    validator.save_report(report_path)
    print(f"\nQuality report saved to: {report_path}")


if __name__ == "__main__":
    main()
