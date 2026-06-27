"""Build llm_eval_set_holdout.json — stratified sample from the ML temporal holdout.

The ML temporal holdout is the set of labeled scenarios with date >= 2025-05-01
(730 scenarios), the same partition used to evaluate v2.2 at 68.9% top-1 and
v3 at 66.2% top-1.  Running the LLM eval harness on this set gives an
apples-to-apples accuracy comparison against those published numbers.

Stratification:
  - Primary: store_type (3) × hour_band (4) = 12 cells
  - Allocation: proportional to cell size, minimum 2 per non-empty cell
  - num_oven_items (19 vs 28) is preserved naturally (no forced balance)
  - Random seed 42 for reproducibility

Target size: ~150 examples
  At n=150 with p≈0.66: σ = sqrt(0.66×0.34/150) ≈ 3.9pp → 95% CI ≈ ±7.6pp
  Enough to detect a ≥10pp gap between LLM and v2.2/v3 with high confidence.

Each example includes:
  - scenario_text   — plain-language situation (v0.3 CSV format; sent to LLM in native mode)
  - features        — hidden numeric block for ML baselines
  - cook_now        — formula-derived first item (= optimal_first_item)
  - holdout_clean   — True (no ML training leakage)

Outputs (same schema as v0.3):
  data/llm_eval_set_holdout.json   ← runner-compatible {metadata, examples[]}
  data/llm_eval_set_holdout.csv    ← id, input, expected, tag, source, notes

Run:
  python scripts/build_eval_set_holdout.py

Then evaluate with:
  # Natural language (default native mode — LLM reads scenario_text)
  python notebooks/week9_llm_eval_runner.py \\
    --eval-set=holdout --prompt-version=v0.3

  # Fair comparison (LLM gets same numeric table as ML)
  python notebooks/week9_llm_eval_runner.py \\
    --eval-set=holdout --prompt-version=v0.2 --input-mode=features

Composition:
  - ~150 modal examples — stratified sample from ML temporal holdout (holdout_clean=True)
  - 23 edge + 15 OOS + 9 adversarial — imported from llm_eval_set_v0.3.json (holdout_clean=False)
  Modal formula_top1_accuracy is comparable to v2.2/v3; guardrails test edge ranking + refusal.
"""

import copy
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.utils import ITEM_DISPLAY_NAMES
from src.pairwise_trainer import OVEN_ITEMS

RANDOM_SEED     = 42
HOLDOUT_CUTOFF  = "2025-05-01"
TARGET_N        = 150
MIN_PER_CELL    = 2

JSON_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "llm_eval_set_holdout.json")
CSV_OUTPUT_PATH  = os.path.join(PROJECT_ROOT, "data", "llm_eval_set_holdout.csv")
V03_JSON_PATH    = os.path.join(PROJECT_ROOT, "data", "llm_eval_set_v0.3.json")
V03_IMPORT_TAGS  = ("edge", "OOS", "adversarial")


# ---------------------------------------------------------------------------
# Helpers — match the hour_band definition used in v0.1/v0.2 eval sets
# ---------------------------------------------------------------------------

def hour_band(h: int) -> str:
    if 6 <= h <= 10:
        return "morning"
    if 11 <= h <= 14:
        return "lunch"
    if 15 <= h <= 18:
        return "afternoon"
    return "evening"


def cell_key(scenario: dict) -> tuple[str, str]:
    f = scenario["features"]
    return (f["store_type"], hour_band(f["decision_hour"]))


def present_items(features: dict) -> list[str]:
    found = {k[: -len("_forecast_demand")] for k in features if k.endswith("_forecast_demand")}
    ordered = [item for item in OVEN_ITEMS if item in found]
    extras = sorted(found - set(ordered))
    return ordered + extras


def present_items_count(features: dict) -> int:
    return len(present_items(features))


def _clock_phrase(hour: int) -> str:
    if hour == 0:
        return "midnight"
    if hour == 12:
        return "noon"
    if hour < 12:
        return f"{hour} AM"
    return f"{hour - 12} PM"


def _hold_phrase(hold: int | float) -> str:
    h = int(hold)
    if h <= 2:
        return "two hours"
    if h <= 4:
        return "four hours"
    return "most of the day"


def _time_remaining_phrase(tr: float) -> str:
    if tr <= 0.25:
        return "about 15 minutes left in the cook window"
    if tr <= 0.5:
        return "about half an hour left in the cook window"
    if tr <= 0.75:
        return "about 45 minutes left in the cook window"
    if tr <= 1.0:
        return "about an hour left in the cook window"
    if tr <= 1.75:
        return "about an hour and 45 minutes left in the cook window"
    if tr <= 3.0:
        return f"about {tr:.0f} hours left in the cook window"
    return "plenty of time left in the cook window"


def _tray_phrase(lcu: int) -> str:
    if lcu == 1:
        return ", cooks one at a time"
    return f", cooks {lcu} to a tray"


def _opening_context(features: dict) -> str:
    store = features["store_type"]
    hour = features["decision_hour"]
    day = features["day_of_week"]
    clock = _clock_phrase(hour)

    if 5 <= hour <= 10:
        return f"It's {clock} on a {day} at your {store} store. The morning rush is starting."
    if 11 <= hour <= 14:
        return f"It's {clock} on a {day} — {store} store, lunch peak."
    if 15 <= hour <= 16:
        return f"{store.capitalize()} store, {clock} on a {day}. Afternoon traffic is steady."
    return f"It's {clock} on a {day} at your {store} store. The evening rush is building."


def format_scenario_text(features: dict) -> str:
    """Plain-language scenario for the LLM (v0.3-style prose, not a data table)."""
    items = present_items(features)
    opening = _opening_context(features)

    clauses = []
    for item in items:
        display = ITEM_DISPLAY_NAMES.get(item, item.replace("_", " "))
        demand = features[f"{item}_forecast_demand"]
        hold = features[f"{item}_hold_time"]
        tr = features[f"{item}_time_remaining"]
        lcu = features[f"{item}_lcu"]
        unit = "unit" if demand == 1 else "units"
        clauses.append(
            f"{display} — forecasting {demand} {unit}, {_time_remaining_phrase(tr)}, "
            f"stays good {_hold_phrase(hold)} once cooked{_tray_phrase(lcu)}"
        )

    n = len(items)
    board = (
        f"You've got {n} items on the board that all need the oven. "
        f"Here's what's queued: " + "; ".join(clauses) + "."
    )
    closing = "One oven. What goes in first?"
    return f"{opening} {board} {closing}"


def format_input_text(features: dict) -> str:
    """Numeric feature-table user-turn message (--input-mode=features)."""
    items = present_items(features)
    lines = []
    for item in items:
        demand = features[f"{item}_forecast_demand"]
        lcu    = features[f"{item}_lcu"]
        hold   = features[f"{item}_hold_time"]
        tr     = features[f"{item}_time_remaining"]
        lines.append(
            f"  {item:<20s} — need {demand} units, "
            f"{tr}hr left in window, stays good {hold}hr once cooked, "
            f"cooks {lcu} to a tray"
        )
    return (
        f"Store: {features['store_type']} | "
        f"Day: {features['day_of_week']} (weekend={features['is_weekend']}) | "
        f"Hour: {features['decision_hour']}:00\n"
        f"Items present:\n" + "\n".join(lines)
    )


# ---------------------------------------------------------------------------
# Proportional stratified sampling
# ---------------------------------------------------------------------------

def proportional_allocation(
    cell_sizes: dict[tuple, int],
    target: int,
    min_per_cell: int,
) -> dict[tuple, int]:
    """Allocate target samples proportionally to cell sizes.

    Guarantees at least min_per_cell per non-empty cell.
    Uses the Hamilton (largest-remainder) method for integer rounding.
    """
    total = sum(cell_sizes.values())
    if total == 0:
        return {}

    # Raw (fractional) allocations
    raw = {cell: target * size / total for cell, size in cell_sizes.items()}

    # Floor, capped at cell size
    alloc = {cell: max(min_per_cell, min(size, int(raw[cell])))
             for cell, size in cell_sizes.items()}

    # Distribute remaining quota by largest remainder, respecting cell caps
    remaining = target - sum(alloc.values())
    remainders = sorted(
        [(raw[cell] - int(raw[cell]), cell) for cell in cell_sizes],
        reverse=True,
    )
    for _, cell in remainders:
        if remaining <= 0:
            break
        cap = cell_sizes[cell]
        if alloc[cell] < cap:
            alloc[cell] += 1
            remaining -= 1

    return alloc


def build_holdout_examples(
    holdout: list[dict],
    allocation: dict[tuple, int],
    rng: random.Random,
) -> list[dict]:
    """Sample from each cell and convert to eval-set format."""
    # Group holdout scenarios by cell
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for s in holdout:
        cells[cell_key(s)].append(s)

    examples = []
    for cell, n in sorted(allocation.items()):
        store_type, hband = cell
        pool = cells.get(cell, [])
        if not pool:
            print(f"  WARNING: empty cell {cell}, skipping")
            continue

        sampled = rng.sample(pool, min(n, len(pool)))

        for s in sampled:
            features = s["features"]
            optimal_order = s["optimal_order"]
            optimal_first = s["optimal_first_item"]
            n_items = present_items_count(features)

            scenario_text = format_scenario_text(features)

            examples.append({
                "eval_id":            s["scenario_id"],
                "source":             "synthetic_logs",
                "source_detail":      (
                    f"labeled_training_set.json | scenario_id={s['scenario_id']} | "
                    f"holdout partition (date>={HOLDOUT_CUTOFF})"
                ),
                "eval_tags":          ["accuracy", "holdout"],
                "csv_tag":            "modal",
                "store_type":         features["store_type"],
                "decision_hour":      features["decision_hour"],
                "hour_band":          hour_band(features["decision_hour"]),
                "num_items":          n_items,
                "scenario_text":      scenario_text,
                "cook_now":           optimal_first,
                "cook_now_set":       [optimal_first],
                "must_precede":       [],
                "features":           features,
                "optimal_order":      optimal_order,
                "optimal_first_item": optimal_first,
                "formula_first_item": optimal_first,
                "formula_agrees":     True,
                "holdout_clean":      True,
                "rationale": (
                    f"{features['store_type']} store, "
                    f"{hour_band(features['decision_hour'])} ({features['decision_hour']}:00), "
                    f"{n_items} items. Formula first: {optimal_first}."
                ),
            })

    return examples


def load_v03_guardrail_examples() -> list[dict]:
    """Import edge, OOS, and adversarial cases from llm_eval_set_v0.3.json."""
    if not os.path.exists(V03_JSON_PATH):
        raise FileNotFoundError(
            f"v0.3 eval set not found: {V03_JSON_PATH}. "
            "Run: python scripts/build_eval_set_v0_3.py"
        )

    with open(V03_JSON_PATH) as f:
        v03 = json.load(f)

    imported: list[dict] = []
    for ex in v03["examples"]:
        tag = ex.get("csv_tag")
        if tag not in V03_IMPORT_TAGS:
            continue

        row = copy.deepcopy(ex)
        row["holdout_clean"] = False
        row["eval_tags"] = list(dict.fromkeys(
            (ex.get("eval_tags") or []) + ["holdout_guardrail"]
        ))
        row["source_detail"] = (
            f"{ex.get('source_detail') or 'llm_eval_set_v0.3.json'} | "
            f"imported into holdout eval set ({tag})"
        )
        imported.append(row)

    by_tag = Counter(ex["csv_tag"] for ex in imported)
    print(f"  Imported from v0.3: {dict(sorted(by_tag.items()))} ({len(imported)} total)")
    return imported


# ---------------------------------------------------------------------------
# CSV serialisation (matches v0.3: id, input, expected, tag, source, notes)
# ---------------------------------------------------------------------------

def to_csv_row(ex: dict) -> dict:
    return {
        "id":       ex["eval_id"],
        "input":    ex["scenario_text"],
        "expected": ex.get("cook_now") or "",
        "tag":      ex["csv_tag"],
        "source":   ex["source"],
        "notes":    ex.get("rationale", ""),
    }


def write_csv(examples: list[dict], path: str) -> None:
    fieldnames = ["id", "input", "expected", "tag", "source", "notes"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ex in examples:
            writer.writerow(to_csv_row(ex))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Building llm_eval_set_holdout (target ~{TARGET_N} stratified examples)...")
    rng = random.Random(RANDOM_SEED)

    # Load holdout
    labeled_path = os.path.join(PROJECT_ROOT, "data", "labeled_training_set.json")
    with open(labeled_path) as f:
        all_data = json.load(f)
    holdout = [s for s in all_data if s["features"]["date"] >= HOLDOUT_CUTOFF]
    print(f"  Holdout scenarios (>={HOLDOUT_CUTOFF}): {len(holdout):,}")

    # Compute cell sizes
    cell_sizes: dict[tuple, int] = defaultdict(int)
    for s in holdout:
        cell_sizes[cell_key(s)] += 1

    print("\n  Cell sizes (store_type × hour_band):")
    for (st, hb), n in sorted(cell_sizes.items()):
        print(f"    {st:10s} × {hb:10s}: {n}")

    # Proportional allocation
    allocation = proportional_allocation(cell_sizes, TARGET_N, MIN_PER_CELL)

    print(f"\n  Allocation (total = {sum(allocation.values())}):")
    for (st, hb), n in sorted(allocation.items()):
        print(f"    {st:10s} × {hb:10s}: {n}  (pool={cell_sizes[(st, hb)]})")

    # Sample holdout modal cases
    holdout_examples = build_holdout_examples(holdout, allocation, rng)
    print(f"\n  Sampled holdout modal: {len(holdout_examples)} examples")

    # Append v0.3 guardrail cases (edge / OOS / adversarial)
    print("  Loading v0.3 guardrail cases (edge, OOS, adversarial)...")
    guardrail_examples = load_v03_guardrail_examples()
    examples = holdout_examples + guardrail_examples
    print(f"  Combined total: {len(examples)} examples")

    # Coverage report
    stores      = Counter(e["store_type"] for e in examples if e.get("store_type"))
    hbands      = Counter(e["hour_band"]  for e in examples if e.get("hour_band"))
    num_items_d = Counter(e["num_items"]  for e in examples)
    first_items = Counter(e["optimal_first_item"] for e in examples if e.get("optimal_first_item"))
    csv_tags    = Counter(e["csv_tag"] for e in examples)
    sources     = Counter(e["source"] for e in examples)
    holdout_clean_n = sum(1 for e in examples if e.get("holdout_clean"))

    print("\n  Coverage:")
    print(f"    CSV tags    : {dict(sorted(csv_tags.items()))}")
    print(f"    Sources     : {dict(sorted(sources.items()))}")
    print(f"    holdout_clean (ML-comparable modal): {holdout_clean_n}")
    print(f"    Store types : {dict(sorted(stores.items()))}")
    print(f"    Hour bands  : {dict(sorted(hbands.items()))}")
    print(f"    Item counts : {dict(sorted(num_items_d.items()))}")
    print(f"    Top 5 first items (ranking examples):")
    ranking_n = sum(1 for e in examples if e.get("optimal_first_item"))
    for item, cnt in first_items.most_common(5):
        print(f"      {item:25s}: {cnt}  ({100*cnt/ranking_n:.1f}%)")

    # --- Write JSON ---
    metadata = {
        "version":                "holdout",
        "created":                "2026-06-27",
        "description":            (
            f"Holdout eval set: {len(holdout_examples)} modal examples stratified from the ML "
            f"temporal holdout (date>={HOLDOUT_CUTOFF}, holdout_clean=True) plus "
            f"{len(guardrail_examples)} guardrail cases from llm_eval_set_v0.3.json "
            f"(edge/OOS/adversarial, holdout_clean=False). "
            f"Modal formula_top1_accuracy is comparable to v2.2/v3."
        ),
        "total_examples":          len(examples),
        "modal_holdout_n":         len(holdout_examples),
        "guardrail_n":             len(guardrail_examples),
        "guardrail_by_tag":        dict(Counter(e["csv_tag"] for e in guardrail_examples)),
        "target_n":                TARGET_N,
        "stratification":          "proportional by store_type × hour_band (modal slice only)",
        "sources":                 dict(sources),
        "csv_tag_distribution":    dict(csv_tags),
        "holdout_clean_n":         holdout_clean_n,
        "v03_import_tags":         list(V03_IMPORT_TAGS),
        "store_type_distribution": dict(stores),
        "hour_band_distribution":  dict(hbands),
        "item_count_distribution": {str(k): v for k, v in sorted(num_items_d.items())},
        "holdout_cutoff":          HOLDOUT_CUTOFF,
        "random_seed":             RANDOM_SEED,
        "ml_reference_accuracy": {
            "v2_2_top1_pct":  68.9,
            "v3_top1_pct":    66.2,
            "n_scenarios":    730,
            "note": (
                "v2.2 pairwise GBM temporal holdout top-1 (authoritative selection metric). "
                "v3 LightGBM LambdaRank same holdout. "
                "LLM formula_top1_accuracy on this eval set is directly comparable."
            ),
        },
        "notes": (
            "CSV columns match v0.3: id, input (scenario_text), expected (cook_now), tag, source, notes. "
            "Modal slice (holdout_clean=True): cook_now = formula label from labeled_training_set. "
            "Guardrail slice (from v0.3): edge/adversarial ranked with JTBD labels; OOS expects refusal. "
            "Report formula_top1_accuracy on modal slice only for ML comparison. "
            "Recommended: python notebooks/week9_llm_eval_runner.py "
            "--eval-set=holdout --prompt-version=v0.3"
        ),
    }

    output = {"metadata": metadata, "examples": examples}
    os.makedirs(os.path.dirname(JSON_OUTPUT_PATH), exist_ok=True)
    with open(JSON_OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved JSON → {JSON_OUTPUT_PATH}")
    print(f"  File size: {os.path.getsize(JSON_OUTPUT_PATH) / 1024 / 1024:.1f} MB")

    write_csv(examples, CSV_OUTPUT_PATH)
    print(f"  Saved CSV  → {CSV_OUTPUT_PATH}")

    # Sample scenario preview
    sample = examples[0]
    preview = sample["scenario_text"][:280] + "..."
    print(f"\n  Sample scenario_text ({sample['eval_id']}):")
    print(f"    {preview}")

    print(f"\n  To run the LLM eval harness:")
    print(f"    # Natural language (default)")
    print(f"    python notebooks/week9_llm_eval_runner.py \\")
    print(f"      --eval-set=holdout --prompt-version=v0.3")
    print(f"    # Fair numeric comparison vs ML")
    print(f"    python notebooks/week9_llm_eval_runner.py \\")
    print(f"      --eval-set=holdout --prompt-version=v0.2 --input-mode=features")
    print(f"\n  Expected: ~{len(examples)} eval rows "
          f"({len(holdout_examples)} modal + {len(guardrail_examples)} guardrail)")
    print(f"  ML reference: v2.2={metadata['ml_reference_accuracy']['v2_2_top1_pct']}%  "
          f"v3={metadata['ml_reference_accuracy']['v3_top1_pct']}%")


if __name__ == "__main__":
    main()
