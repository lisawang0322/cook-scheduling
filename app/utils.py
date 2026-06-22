"""Shared utilities for the Streamlit demo app.

Handles model loading, scenario comparison, and plain-language explanation generation.
"""

import json
import os
import pickle
import random
from typing import Any

import numpy as np
import pandas as pd

# Add project root to path
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.cook_scheduler import CookSchedulerV1, AssociateBaseline
from src.pairwise_trainer import PairwiseModelTrainer, OVEN_ITEMS


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "_loss":
            module = "sklearn._loss.loss"
        return super().find_class(module, name)


# --- Constants ---

ITEM_DISPLAY_NAMES = {
    "wings_bone_in":      "Bone-in Wings",
    "wings_boneless":     "Boneless Wings",
    "chicken_strip":      "Chicken Strips",
    "chicken_bite":       "Chicken Bites",
    "quesadilla":         "Mini Chicken Quesadilla",
    "chicken_sandwich":   "Crispy Chicken Sandwich",
    "potato_wedge":       "Seasoned Potato Wedges",
    "waffle_tot":         "Waffle Potato Tots",
    "hash_brown":         "Hash Brown Patties",
    "empanada":           "Savory Empanada",
    "chimichanga":        "Beef Chimichanga",
    "jamaican_turnover":  "Jamaican-Style Turnover",
    "jamaican_patty":     "Jamaican-Style Patty",
    "pupusa":             "Stuffed Corn Cake",
    "beef_mini_taco":     "Mini Beef Taco Bites",
    "garlic_knot":        "Artisan Garlic Knots",
    "kolache":            "Sausage Kolache",
    "croissant":          "Breakfast Croissant",
    "breakfast_sandwich": "Breakfast Sandwich",
    "sweet_croissant":    "Sweet Pastry Croissant",
    "danish":             "Glazed Danish Pastry",
    "pizza_slice":        "Pizza Slice",
    "pizza_stuffed":      "Stuffed Pizza Pocket",
    "hot_dog":            "Beef Hot Dog",
    "sausage":            "Smoked Sausage Link",
    "taquito":            "Chicken Taquito",
    "buffalo_roller":     "Buffalo Chicken Roller",
    "corn_dog":           "Corn Dog",
}

ITEM_EMOJIS = {
    "wings_bone_in":      "�",
    "wings_boneless":     "🍗",
    "chicken_strip":      "🍗",
    "chicken_bite":       "🍗",
    "quesadilla":         "🫓",
    "chicken_sandwich":   "🥪",
    "potato_wedge":       "🍟",
    "waffle_tot":         "🧇",
    "hash_brown":         "🥔",
    "empanada":           "🥟",
    "chimichanga":        "🌯",
    "jamaican_turnover":  "🥟",
    "jamaican_patty":     "🫔",
    "pupusa":             "🫓",
    "beef_mini_taco":     "🌮",
    "garlic_knot":        "🫓",
    "kolache":            "�",
    "croissant":          "🥐",
    "breakfast_sandwich": "🥚",
    "sweet_croissant":    "🥐",
    "danish":             "🥐",
    "pizza_slice":        "🍕",
    "pizza_stuffed":      "🍕",
    "hot_dog":            "🌭",
    "sausage":            "🌭",
    "taquito":            "🌯",
    "buffalo_roller":     "🌯",
    "corn_dog":           "🌭",
}

STORE_TYPE_LABELS = {
    "urban": "Urban (high-traffic)",
    "suburban": "Suburban (medium-traffic)",
    "highway": "Highway (low-traffic)",
}

HOUR_LABELS = {
    6: "6 AM (Opening)", 7: "7 AM", 8: "8 AM", 9: "9 AM",
    10: "10 AM", 11: "11 AM", 12: "12 PM (Lunch)", 13: "1 PM",
    14: "2 PM", 15: "3 PM", 16: "4 PM", 17: "5 PM",
    18: "6 PM (Dinner)", 19: "7 PM", 20: "8 PM", 21: "9 PM",
    22: "10 PM", 23: "11 PM",
}

ITEM_COOK_TIMES = {
    "wings_bone_in":      25,
    "wings_boneless":     20,
    "chicken_strip":      15,
    "chicken_bite":       15,
    "quesadilla":          8,
    "chicken_sandwich":    8,
    "potato_wedge":       12,
    "waffle_tot":         10,
    "hash_brown":         10,
    "empanada":           15,
    "chimichanga":        12,
    "jamaican_turnover":  15,
    "jamaican_patty":     10,
    "pupusa":             10,
    "beef_mini_taco":      8,
    "garlic_knot":        10,
    "kolache":             8,
    "croissant":           5,
    "breakfast_sandwich":  5,
    "sweet_croissant":     8,
    "danish":             10,
    "pizza_slice":        10,
    "pizza_stuffed":       8,
    "hot_dog":            15,
    "sausage":            20,
    "taquito":            20,
    "buffalo_roller":     15,
    "corn_dog":           15,
}


# --- Data Loading ---

def load_model():
    """Load the v2.2 pairwise model and historical features."""
    model_path = os.path.join(PROJECT_ROOT, "models", "v2_2_pairwise_temporal.pkl")
    with open(model_path, "rb") as f:
        data = _CompatUnpickler(f).load()
    return data["model"], data["historical"]


def load_labeled_data():
    """Load labeled training scenarios."""
    data_path = os.path.join(PROJECT_ROOT, "data", "labeled_training_set.json")
    with open(data_path) as f:
        return json.load(f)


def load_labeling_report():
    """Load the labeling report for aggregate stats."""
    report_path = os.path.join(PROJECT_ROOT, "output", "labeling_report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            return json.load(f)
    return None


# --- Scenario Comparison ---

def build_scenario_events(features: dict) -> list[dict]:
    """Convert scenario features dict into oven_events list."""
    events = []
    for item in OVEN_ITEMS:
        if f"{item}_forecast_demand" in features:
            events.append({
                "item": item,
                "forecast_demand": features[f"{item}_forecast_demand"],
                "lcu": features[f"{item}_lcu"],
                "hold_time_hours": features[f"{item}_hold_time"],
                "time_remaining": features[f"{item}_time_remaining"],
                "cooked_qty": features.get(f"{item}_cooked_qty", 0),
                "window_start_hour": features["decision_hour"],
                "window_end_hour": (features["decision_hour"]
                                    + features[f"{item}_hold_time"]),
                "lowest_cookable_unit": features[f"{item}_lcu"],
                "exact_multiples": True,
                "equipment": "oven",
            })
    return events


def get_v1_ranking(features: dict) -> list[str]:
    """Get v1 scheduler ranking for a scenario."""
    scheduler = CookSchedulerV1()
    events = build_scenario_events(features)
    decision_hour = features["decision_hour"] + 0.25
    ranked = scheduler.rank_items(decision_hour, events)
    return [r["item"] for r in ranked]


def get_associate_pick(features: dict, seed: int = 42) -> str:
    """Get associate baseline pick for a scenario."""
    associate = AssociateBaseline(seed=seed)
    events = build_scenario_events(features)
    return associate.pick_first_item(events)


def get_v22_ranking(features: dict, model, historical: dict) -> list[str]:
    """Get v2.2 pairwise model ranking for a scenario."""
    trainer = PairwiseModelTrainer()
    trainer.model = model
    trainer.historical = historical

    present_items = [
        item for item in OVEN_ITEMS
        if f"{item}_forecast_demand" in features
    ]
    if len(present_items) < 2:
        return present_items

    scenario_features = {
        "decision_hour": features["decision_hour"],
        "store_type": features["store_type"],
        "is_weekend": features.get("is_weekend", False),
        "day_of_week": features.get("day_of_week", "Monday"),
        "num_oven_items": len(present_items),
    }
    for item in present_items:
        scenario_features[f"{item}_forecast_demand"] = features[f"{item}_forecast_demand"]
        scenario_features[f"{item}_lcu"] = features[f"{item}_lcu"]
        scenario_features[f"{item}_hold_time"] = features[f"{item}_hold_time"]
        scenario_features[f"{item}_time_remaining"] = features[f"{item}_time_remaining"]
        scenario_features[f"{item}_cooked_qty"] = features.get(f"{item}_cooked_qty", 0)

    return trainer.rank_items(scenario_features, present_items)


def compare_scenario(scenario: dict, model, historical: dict) -> dict:
    """Run all three approaches on a single scenario and return comparison."""
    features = scenario["features"]
    optimal_first = scenario["optimal_first_item"]
    optimal_order = scenario["optimal_order"]

    v1_ranking = get_v1_ranking(features)
    associate_pick = get_associate_pick(features)
    v22_ranking = get_v22_ranking(features, model, historical)

    return {
        "features": features,
        "optimal_first": optimal_first,
        "optimal_order": optimal_order,
        "associate_pick": associate_pick,
        "associate_correct": associate_pick == optimal_first,
        "v1_ranking": v1_ranking,
        "v1_correct": v1_ranking[0] == optimal_first if v1_ranking else False,
        "v22_ranking": v22_ranking,
        "v22_correct": v22_ranking[0] == optimal_first if v22_ranking else False,
    }


# --- Explanation Generation ---

# Item-specific demand peak descriptions
ITEM_PEAK_DESCRIPTIONS = {
    "wings_bone_in":      "Bone-in wings peak during dinner hours (5–9 PM)",
    "wings_boneless":     "Boneless wings peak during dinner hours (5–9 PM)",
    "chicken_strip":      "Chicken strips peak at lunch and dinner (11 AM–7 PM)",
    "chicken_bite":       "Chicken bites peak at lunch and dinner (11 AM–7 PM)",
    "quesadilla":         "Quesadillas peak at lunch (11 AM–4 PM)",
    "chicken_sandwich":   "Chicken sandwiches peak at lunch and dinner (11 AM–8 PM)",
    "potato_wedge":       "Potato wedges peak at lunch and dinner (11 AM–8 PM)",
    "waffle_tot":         "Waffle tots peak in the morning and at lunch (7 AM–1 PM)",
    "hash_brown":         "Hash browns peak in the morning (6–10 AM)",
    "empanada":           "Empanadas peak midday (10 AM–5 PM)",
    "chimichanga":        "Chimichangas peak at lunch (11 AM–5 PM)",
    "jamaican_turnover":  "Jamaican turnovers peak in the morning and at lunch (8 AM–3 PM)",
    "jamaican_patty":     "Jamaican patties peak midday (10 AM–4 PM)",
    "pupusa":             "Pupusas peak at lunch (11 AM–5 PM)",
    "beef_mini_taco":     "Mini beef tacos peak midday (10 AM–4 PM)",
    "garlic_knot":        "Garlic knots peak at dinner (5–9 PM)",
    "kolache":            "Kolaches peak in the morning (7–11 AM)",
    "croissant":          "Breakfast croissants peak in the morning (6–10 AM)",
    "breakfast_sandwich": "Breakfast sandwiches peak in the morning (6–10 AM)",
    "sweet_croissant":    "Sweet pastry croissants peak in the morning (7–11 AM)",
    "danish":             "Danish pastries peak in the morning (7–11 AM)",
    "pizza_slice":        "Pizza slices peak at lunch (11 AM–2 PM)",
    "pizza_stuffed":      "Stuffed pizza pockets peak at lunch (11 AM–4 PM)",
    "hot_dog":            "Hot dogs peak at lunch and dinner (11 AM–8 PM)",
    "sausage":            "Smoked sausages peak at lunch and dinner (11 AM–8 PM)",
    "taquito":            "Taquitos sell consistently all day (8 AM–9 PM)",
    "buffalo_roller":     "Buffalo rollers peak in the afternoon and evening (4–9 PM)",
    "corn_dog":           "Corn dogs peak at lunch and dinner (11 AM–8 PM)",
}

# Peak hour ranges per item (start, end) for explanation logic
_ITEM_PEAK_HOURS: dict[str, tuple[int, int]] = {
    "wings_bone_in": (17, 21),    "wings_boneless": (17, 21),
    "chicken_strip": (11, 19),    "chicken_bite": (11, 19),
    "quesadilla": (11, 16),       "chicken_sandwich": (11, 20),
    "potato_wedge": (11, 20),     "waffle_tot": (7, 13),
    "hash_brown": (6, 10),        "empanada": (10, 17),
    "chimichanga": (11, 17),      "jamaican_turnover": (8, 15),
    "jamaican_patty": (10, 16),   "pupusa": (11, 17),
    "beef_mini_taco": (10, 16),   "garlic_knot": (17, 21),
    "kolache": (7, 11),           "croissant": (6, 10),
    "breakfast_sandwich": (6, 10), "sweet_croissant": (7, 11),
    "danish": (7, 11),            "pizza_slice": (11, 14),
    "pizza_stuffed": (11, 16),    "hot_dog": (11, 20),
    "sausage": (11, 20),          "taquito": (8, 21),
    "buffalo_roller": (16, 21),   "corn_dog": (11, 20),
}

ITEM_WASTE_DESCRIPTIONS = {
    ("pizza_slice", "highway"): "Pizza slices tend to waste more at highway stores",
    ("pizza_slice", "urban"): "Pizza slices sell well at urban stores (low waste risk)",
    ("wings_bone_in", "urban"): "Bone-in wings sell quickly at urban stores",
    ("wings_bone_in", "highway"): "Bone-in wings can over-produce at highway stores",
    ("wings_boneless", "urban"): "Boneless wings are popular at urban stores",
    ("beef_mini_taco", "highway"): "Mini beef tacos move slower at highway stores",
    ("beef_mini_taco", "urban"): "Mini beef tacos sell well at urban stores",
    ("taquito", "highway"): "Taquitos sell consistently at highway stores",
    ("empanada", "highway"): "Empanadas tend to waste more at highway stores",
    ("empanada", "urban"): "Empanadas sell well at urban stores",
}


def generate_explanation(ranking: list[str], features: dict) -> list[str]:
    """Generate plain-language explanations for why items are ranked this way.

    No technical jargon — just business-relevant reasoning.
    """
    explanations = []
    hour = features["decision_hour"]
    store_type = features["store_type"]

    for i, item in enumerate(ranking):
        reasons = []
        demand = features.get(f"{item}_forecast_demand", 0)
        time_rem = features.get(f"{item}_time_remaining", 2)
        display = ITEM_DISPLAY_NAMES.get(item, item)

        if i == 0:
            if time_rem <= 1.0:
                reasons.append("needs to be cooked soon (window closing)")
            if demand >= 8:
                reasons.append(f"high demand expected ({demand} units)")

            # Check if in peak hours using the generic mapping
            peak = _ITEM_PEAK_HOURS.get(item)
            if peak and peak[0] <= hour <= peak[1]:
                peak_desc = ITEM_PEAK_DESCRIPTIONS.get(item, "")
                if peak_desc:
                    # Extract the peak window description from the full sentence
                    if "peak" in peak_desc:
                        reasons.append(f"now in peak window ({peak_desc.split('peak')[1].strip()})")
                    else:
                        reasons.append("peak demand window is now")

            waste_key = (item, store_type)
            if waste_key in ITEM_WASTE_DESCRIPTIONS:
                desc = ITEM_WASTE_DESCRIPTIONS[waste_key]
                if "sells well" in desc or "sell quickly" in desc or "popular" in desc or "consistently" in desc:
                    reasons.append(f"moves well at {store_type} stores")

            if not reasons:
                reasons.append(f"best balance of demand ({demand}) and timing")

            explanations.append(
                f"**Cook {display} first** — {'; '.join(reasons)}."
            )
        else:
            if time_rem > 2.0:
                reasons.append("still has time before window ends")
            if demand < 4:
                reasons.append("lower demand right now")

            waste_key = (item, store_type)
            if waste_key in ITEM_WASTE_DESCRIPTIONS:
                desc = ITEM_WASTE_DESCRIPTIONS[waste_key]
                if "waste" in desc or "over-produce" in desc or "slower" in desc:
                    reasons.append("tends to over-produce at this store type")

            if not reasons:
                reasons.append("can wait without risk")

            explanations.append(
                f"**{display}** (#{i+1}) — {'; '.join(reasons)}."
            )

    return explanations


def compute_waste_savings(scenario: dict) -> float:
    """Estimate waste savings if v2.2 had been followed instead of associate."""
    features = scenario["features"]
    # Use write-off data if available
    items = [item for item in OVEN_ITEMS if f"{item}_forecast_demand" in features]
    total_waste = sum(features.get(f"{item}_writeoff", 0) for item in items)
    # Conservative estimate: correct ordering prevents ~30% of waste
    if total_waste > 0:
        return total_waste * 0.3
    return 0.5  # Default small improvement estimate


# --- Aggregate Stats ---

def compute_aggregate_metrics(labeled_data: list[dict], model, historical: dict) -> dict:
    """Compute aggregate comparison metrics across all scenarios."""
    associate_baseline = AssociateBaseline(seed=42)

    associate_correct = 0
    v1_correct = 0
    v22_correct = 0

    by_store = {"urban": {"a": 0, "v1": 0, "v22": 0, "n": 0},
                "suburban": {"a": 0, "v1": 0, "v22": 0, "n": 0},
                "highway": {"a": 0, "v1": 0, "v22": 0, "n": 0}}

    by_hour = {}

    for scenario in labeled_data:
        features = scenario["features"]
        optimal_first = scenario["optimal_first_item"]
        hour = features["decision_hour"]
        store_type = features["store_type"]

        # Associate
        events = build_scenario_events(features)
        a_pick = associate_baseline.pick_first_item(events)
        a_correct = int(a_pick == optimal_first)
        associate_correct += a_correct

        # v1
        v1_rank = get_v1_ranking(features)
        v1_ok = int(v1_rank[0] == optimal_first) if v1_rank else 0
        v1_correct += v1_ok

        # v2.2
        v22_rank = get_v22_ranking(features, model, historical)
        v22_ok = int(v22_rank[0] == optimal_first) if v22_rank else 0
        v22_correct += v22_ok

        # By store
        by_store[store_type]["a"] += a_correct
        by_store[store_type]["v1"] += v1_ok
        by_store[store_type]["v22"] += v22_ok
        by_store[store_type]["n"] += 1

        # By hour
        if hour not in by_hour:
            by_hour[hour] = {"a": 0, "v1": 0, "v22": 0, "n": 0}
        by_hour[hour]["a"] += a_correct
        by_hour[hour]["v1"] += v1_ok
        by_hour[hour]["v22"] += v22_ok
        by_hour[hour]["n"] += 1

    n = len(labeled_data)
    return {
        "total_scenarios": n,
        "associate_accuracy": round(100 * associate_correct / n, 1),
        "v1_accuracy": round(100 * v1_correct / n, 1),
        "v22_accuracy": round(100 * v22_correct / n, 1),
        "by_store": {
            st: {
                "associate": round(100 * d["a"] / d["n"], 1) if d["n"] else 0,
                "v1": round(100 * d["v1"] / d["n"], 1) if d["n"] else 0,
                "v22": round(100 * d["v22"] / d["n"], 1) if d["n"] else 0,
                "n": d["n"],
            }
            for st, d in by_store.items()
        },
        "by_hour": {
            h: {
                "associate": round(100 * d["a"] / d["n"], 1) if d["n"] else 0,
                "v1": round(100 * d["v1"] / d["n"], 1) if d["n"] else 0,
                "v22": round(100 * d["v22"] / d["n"], 1) if d["n"] else 0,
                "n": d["n"],
            }
            for h, d in sorted(by_hour.items())
        },
    }


# --- Story Scenarios ---

STORY_SCENARIOS = [
    {
        "name": "Monday 6 AM — Urban Store (Morning Rush)",
        "description": "Early morning, breakfast items are in demand. Croissants and breakfast sandwiches lead.",
    },
    {
        "name": "Friday 12 PM — Suburban Store (Lunch Peak)",
        "description": "Lunch rush at a busy suburban location. Pizza slices and chicken items spike.",
    },
    {
        "name": "Saturday 6 PM — Highway Store (Dinner)",
        "description": "Weekend dinner at a highway rest stop. Wings and hot dogs are in highest demand.",
    },
    {
        "name": "Wednesday 2 PM — Urban Store (Afternoon Lull)",
        "description": "Slow afternoon. Lower demand across all items — priority ordering matters most.",
    },
    {
        "name": "Sunday 8 AM — Suburban Store (Weekend Morning)",
        "description": "Weekend brunch crowd. Waffle tots, hash browns, and breakfast croissants in demand.",
    },
]


def find_story_scenario(labeled_data: list[dict], story_idx: int) -> dict | None:
    """Find a labeled scenario that matches a story description."""
    conditions = [
        # Monday 6 AM urban (morning rush)
        lambda f: f["day_of_week"] == "Monday" and f["decision_hour"] == 6
        and f["store_type"] == "urban",
        # Friday 12 PM suburban (lunch peak)
        lambda f: f["day_of_week"] == "Friday" and f["decision_hour"] == 12
        and f["store_type"] == "suburban",
        # Saturday 18 (6 PM) highway (dinner)
        lambda f: f["day_of_week"] == "Saturday" and f["decision_hour"] == 18
        and f["store_type"] == "highway",
        # Wednesday 14 (2 PM) urban (afternoon lull)
        lambda f: f["day_of_week"] == "Wednesday" and f["decision_hour"] == 14
        and f["store_type"] == "urban",
        # Sunday 8 AM suburban (weekend morning)
        lambda f: f["day_of_week"] == "Sunday" and f["decision_hour"] == 8
        and f["store_type"] == "suburban",
    ]

    if story_idx >= len(conditions):
        return None

    condition = conditions[story_idx]
    for scenario in labeled_data:
        if condition(scenario["features"]):
            return scenario
    return None


_FORECAST_BASE_DEMAND = {
    "wings_bone_in":      5,
    "wings_boneless":     6,
    "chicken_strip":      5,
    "chicken_bite":       5,
    "quesadilla":         3,
    "chicken_sandwich":   5,
    "potato_wedge":       6,
    "waffle_tot":         7,
    "hash_brown":         4,
    "empanada":           3,
    "chimichanga":        3,
    "jamaican_turnover":  2,
    "jamaican_patty":     2,
    "pupusa":             2,
    "beef_mini_taco":     5,
    "garlic_knot":        3,
    "kolache":            3,
    "croissant":          5,
    "breakfast_sandwich": 5,
    "sweet_croissant":    3,
    "danish":             4,
    "pizza_slice":        5,
    "pizza_stuffed":      3,
    "hot_dog":            5,
    "sausage":            5,
    "taquito":            6,
    "buffalo_roller":     3,
    "corn_dog":           4,
}

_FORECAST_STORE_MULT = {
    "urban": 1.4,
    "suburban": 1.0,
    "highway": 0.7,
}

# --- Reusable time-of-day demand curves (multiplier per hour 0-23) ---
_C_MORNING      = {0:0.1,1:0.1,2:0.1,3:0.1,4:0.15,5:0.25,6:0.8,7:1.5,8:1.6,9:1.4,10:1.0,11:0.6,12:0.4,13:0.3,14:0.25,15:0.2,16:0.2,17:0.2,18:0.2,19:0.2,20:0.15,21:0.1,22:0.1,23:0.1}
_C_LUNCH        = {0:0.1,1:0.1,2:0.1,3:0.1,4:0.1,5:0.15,6:0.3,7:0.5,8:0.6,9:0.8,10:1.0,11:1.4,12:1.7,13:1.5,14:1.2,15:0.9,16:0.7,17:0.8,18:0.9,19:0.8,20:0.6,21:0.4,22:0.3,23:0.15}
_C_DINNER       = {0:0.15,1:0.1,2:0.1,3:0.1,4:0.1,5:0.15,6:0.3,7:0.4,8:0.5,9:0.6,10:0.7,11:0.9,12:1.0,13:1.0,14:0.9,15:0.9,16:1.0,17:1.4,18:1.6,19:1.7,20:1.5,21:1.2,22:0.7,23:0.4}
_C_ALL_DAY      = {0:0.2,1:0.15,2:0.1,3:0.1,4:0.1,5:0.2,6:0.4,7:0.7,8:0.8,9:0.9,10:1.0,11:1.1,12:1.2,13:1.1,14:1.0,15:0.9,16:1.0,17:1.2,18:1.3,19:1.2,20:1.0,21:0.8,22:0.5,23:0.3}
_C_LUNCH_DINNER = {0:0.1,1:0.1,2:0.1,3:0.1,4:0.1,5:0.15,6:0.3,7:0.5,8:0.6,9:0.8,10:1.0,11:1.3,12:1.5,13:1.4,14:1.1,15:0.9,16:1.0,17:1.3,18:1.5,19:1.4,20:1.0,21:0.7,22:0.4,23:0.2}
_C_MORNING_LUNCH= {0:0.1,1:0.1,2:0.1,3:0.1,4:0.15,5:0.3,6:0.7,7:1.2,8:1.4,9:1.3,10:1.1,11:1.2,12:1.0,13:0.8,14:0.6,15:0.5,16:0.4,17:0.4,18:0.4,19:0.3,20:0.2,21:0.15,22:0.1,23:0.1}
_C_MIDDAY       = {0:0.1,1:0.1,2:0.1,3:0.1,4:0.1,5:0.2,6:0.3,7:0.5,8:0.7,9:0.9,10:1.2,11:1.4,12:1.5,13:1.4,14:1.3,15:1.1,16:1.0,17:0.8,18:0.6,19:0.5,20:0.3,21:0.2,22:0.15,23:0.1}
_C_AFT_DINNER   = {0:0.1,1:0.1,2:0.1,3:0.1,4:0.1,5:0.15,6:0.3,7:0.4,8:0.5,9:0.6,10:0.7,11:0.9,12:1.0,13:1.1,14:1.2,15:1.3,16:1.4,17:1.5,18:1.5,19:1.4,20:1.2,21:1.0,22:0.6,23:0.3}

_FORECAST_ITEM_TIME_CURVES = {
    "wings_bone_in":      _C_DINNER,
    "wings_boneless":     _C_DINNER,
    "chicken_strip":      _C_LUNCH_DINNER,
    "chicken_bite":       _C_LUNCH_DINNER,
    "quesadilla":         _C_LUNCH,
    "chicken_sandwich":   _C_LUNCH_DINNER,
    "potato_wedge":       _C_LUNCH_DINNER,
    "waffle_tot":         _C_MORNING_LUNCH,
    "hash_brown":         _C_MORNING,
    "empanada":           _C_MIDDAY,
    "chimichanga":        _C_LUNCH,
    "jamaican_turnover":  _C_MORNING_LUNCH,
    "jamaican_patty":     _C_MIDDAY,
    "pupusa":             _C_MIDDAY,
    "beef_mini_taco":     _C_LUNCH,
    "garlic_knot":        _C_AFT_DINNER,
    "kolache":            _C_MORNING,
    "croissant":          _C_MORNING,
    "breakfast_sandwich": _C_MORNING,
    "sweet_croissant":    _C_MORNING,
    "danish":             _C_MORNING,
    "pizza_slice":        _C_LUNCH,
    "pizza_stuffed":      _C_LUNCH,
    "hot_dog":            _C_ALL_DAY,
    "sausage":            _C_ALL_DAY,
    "taquito":            _C_ALL_DAY,
    "buffalo_roller":     _C_AFT_DINNER,
    "corn_dog":           _C_LUNCH_DINNER,
}

_FORECAST_ITEM_LCU = {
    "wings_bone_in":      (5,  True),
    "wings_boneless":     (8,  True),
    "chicken_strip":      (3,  True),
    "chicken_bite":       (10, True),
    "quesadilla":         (5,  True),
    "chicken_sandwich":   (1,  True),
    "potato_wedge":       (10, True),
    "waffle_tot":         (10, True),
    "hash_brown":         (2,  True),
    "empanada":           (2,  True),
    "chimichanga":        (2,  True),
    "jamaican_turnover":  (2,  True),
    "jamaican_patty":     (1,  False),
    "pupusa":             (2,  True),
    "beef_mini_taco":     (8,  True),
    "garlic_knot":        (2,  True),
    "kolache":            (2,  True),
    "croissant":          (1,  True),
    "breakfast_sandwich": (1,  True),
    "sweet_croissant":    (6,  True),
    "danish":             (6,  True),
    "pizza_slice":        (6,  True),
    "pizza_stuffed":      (2,  True),
    "hot_dog":            (2,  True),
    "sausage":            (2,  True),
    "taquito":            (2,  True),
    "buffalo_roller":     (2,  False),
    "corn_dog":           (2,  True),
}


ITEM_LCU: dict[str, int] = {item: lcu for item, (lcu, _) in _FORECAST_ITEM_LCU.items()}

ITEM_HOLD_TIMES: dict[str, int] = {
    "wings_bone_in": 2, "wings_boneless": 2,
    "chicken_strip": 2, "chicken_bite": 2, "quesadilla": 2, "chicken_sandwich": 2,
    "potato_wedge": 2, "waffle_tot": 2, "hash_brown": 2,
    "empanada": 2, "chimichanga": 2, "jamaican_turnover": 2, "jamaican_patty": 2, "pupusa": 2,
    "beef_mini_taco": 4,
    "garlic_knot": 2, "kolache": 2,
    "croissant": 4, "breakfast_sandwich": 2, "sweet_croissant": 4, "danish": 4,
    "pizza_slice": 2, "pizza_stuffed": 2,
    "hot_dog": 4, "sausage": 4, "taquito": 4, "buffalo_roller": 4, "corn_dog": 4,
}


def generate_forecast_demand(item: str, hour: int, store_type: str, is_weekend: bool) -> int:
    """Generate forecast demand for an item based on time of day, store type, and day type.

    Uses the same demand model as the synthetic data generator (deterministic, no noise).
    Returns a quantity rounded up to the nearest valid cookable unit (LCU).
    """
    base = _FORECAST_BASE_DEMAND.get(item, 6)
    store_mult = _FORECAST_STORE_MULT.get(store_type, 1.0)
    weekend_mult = 1.3 if is_weekend else 1.0
    tod_mult = _FORECAST_ITEM_TIME_CURVES.get(item, {}).get(hour, 0.5)

    raw = max(1, round(base * store_mult * weekend_mult * tod_mult))

    lcu, exact = _FORECAST_ITEM_LCU.get(item, (1, False))
    if exact:
        return max(lcu, ((raw + lcu - 1) // lcu) * lcu)
    return max(lcu, raw)


_FOUR_HOUR_ITEMS = {
    "beef_mini_taco", "croissant", "sweet_croissant", "danish",
    "hot_dog", "sausage", "taquito", "buffalo_roller", "corn_dog",
}


def compute_queue_timing(item: str, decision_hour: float) -> dict[str, Any]:
    """Calculate cook time, hold time, ready time, and expiry for a given item and decision point hour."""
    cook_time = ITEM_COOK_TIMES.get(item, 15)  # default 15 mins
    hold_time_hours = 4 if item in _FOUR_HOUR_ITEMS else 2
    
    # Calculate times
    hour_part = int(decision_hour)
    min_part = int((decision_hour - hour_part) * 60)
    
    # Convert decision_hour to minutes from midnight
    now_mins = hour_part * 60 + min_part
    ready_mins = now_mins + cook_time
    expiry_mins = now_mins + int(hold_time_hours * 60)
    
    def mins_to_str(m: int) -> str:
        h = (m // 60) % 24
        mins = m % 60
        ampm = "AM" if h < 12 else "PM"
        h_12 = h if 1 <= h <= 12 else h - 12 if h > 12 else 12
        return f"{h_12}:{mins:02d} {ampm}"
        
    return {
        "cook_time_mins": cook_time,
        "hold_time_hours": hold_time_hours,
        "ready_time_str": mins_to_str(ready_mins),
        "expiry_time_str": mins_to_str(expiry_mins)
    }


def get_auto_scenario() -> dict[str, Any]:
    """Return a single highly realistic scenario for the auto-loaded tablet view."""
    try:
        labeled_data = load_labeled_data()
        # Find a Friday 12 PM Suburban scenario (Lunch Peak)
        for s in labeled_data:
            f = s["features"]
            if f["day_of_week"] == "Friday" and f["decision_hour"] == 12 and f["store_type"] == "suburban":
                return s
        # Fallback to the first scenario if not found
        if labeled_data:
            return labeled_data[0]
    except Exception:
        pass
        
    # Return a high-fidelity mock scenario if loading fails
    return {
        "optimal_first_item": "pizza_slice",
        "optimal_order": ["pizza_slice", "wings_bone_in", "taquito"],
        "features": {
            "decision_hour": 12,
            "store_type": "suburban",
            "day_of_week": "Friday",
            "is_weekend": False,
            "num_oven_items": 3,
            "pizza_slice_forecast_demand": 18,
            "pizza_slice_lcu": 6,
            "pizza_slice_hold_time": 2,
            "pizza_slice_time_remaining": 1.75,
            "wings_bone_in_forecast_demand": 10,
            "wings_bone_in_lcu": 5,
            "wings_bone_in_hold_time": 2,
            "wings_bone_in_time_remaining": 1.75,
            "taquito_forecast_demand": 12,
            "taquito_lcu": 2,
            "taquito_hold_time": 4,
            "taquito_time_remaining": 3.75,
        }
    }


def log_associate_action(record: dict[str, Any]) -> str:
    """Log an associate override or confirmation to output/associate_overrides.json."""
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "associate_overrides.json")
    
    actions = load_associate_actions()
    actions.append(record)
    
    with open(log_path, "w") as f:
        json.dump(actions, f, indent=2)
        
    return log_path


def load_associate_actions() -> list[dict[str, Any]]:
    """Load associate overrides and confirmations from output/associate_overrides.json."""
    log_path = os.path.join(PROJECT_ROOT, "output", "associate_overrides.json")
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                return json.load(f)
        except Exception:
            return []
    return []

