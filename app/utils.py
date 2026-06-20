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


# --- Constants ---

ITEM_DISPLAY_NAMES = {
    "pizza": "Pizza",
    "wings_2h": "Wings (2hr hold)",
    "wings_4h": "Wings (4hr hold)",
    "baked_goods": "Baked Goods",
}

ITEM_EMOJIS = {
    "pizza": "🍕",
    "wings_2h": "🍗",
    "wings_4h": "🍗",
    "baked_goods": "🧁",
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
    "pizza": 12,        # 12 minutes
    "wings_2h": 18,     # 18 minutes
    "wings_4h": 18,     # 18 minutes
    "baked_goods": 10,   # 10 minutes
    "taquitos": 20,     # 20 minutes (roller grill)
}


# --- Data Loading ---

def load_model():
    """Load the v2.2 pairwise model and historical features."""
    model_path = os.path.join(PROJECT_ROOT, "models", "v2_2_pairwise_temporal.pkl")
    with open(model_path, "rb") as f:
        data = pickle.load(f)
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
    "pizza": "Pizza demand peaks during lunch hours (11 AM–2 PM)",
    "wings_2h": "Wings demand peaks during dinner hours (5–9 PM)",
    "wings_4h": "Wings (4hr) demand peaks in the evening (5–9 PM)",
    "baked_goods": "Baked goods demand peaks in the morning (7–10 AM)",
}

ITEM_WASTE_DESCRIPTIONS = {
    ("pizza", "highway"): "Pizza tends to waste more at highway stores",
    ("pizza", "urban"): "Pizza sells well at urban stores (low waste risk)",
    ("wings_2h", "urban"): "Wings tend to waste more at urban stores",
    ("wings_2h", "highway"): "Wings sell quickly at highway stores",
    ("baked_goods", "highway"): "Baked goods waste more at highway stores",
    ("baked_goods", "urban"): "Baked goods sell well at urban stores",
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
        hold = features.get(f"{item}_hold_time", 2)

        if i == 0:
            # Top-ranked item: explain why it should go first
            if time_rem <= 1.0:
                reasons.append("needs to be cooked soon (window closing)")
            if demand >= 8:
                reasons.append(f"high demand expected ({demand} units)")

            # Check if this item is in its peak hours
            if item == "pizza" and 11 <= hour <= 14:
                reasons.append("lunch rush is starting")
            elif item in ("wings_2h", "wings_4h") and 17 <= hour <= 21:
                reasons.append("dinner demand is picking up")
            elif item == "baked_goods" and 6 <= hour <= 10:
                reasons.append("morning customers want fresh baked goods")

            # Store-specific waste context
            waste_key = (item, store_type)
            if waste_key in ITEM_WASTE_DESCRIPTIONS:
                desc = ITEM_WASTE_DESCRIPTIONS[waste_key]
                if "sells well" in desc or "sell quickly" in desc:
                    reasons.append(f"this item sells well at {store_type} stores")

            if not reasons:
                reasons.append(f"best balance of demand ({demand}) and timing")

            explanations.append(
                f"**Cook {ITEM_DISPLAY_NAMES[item]} first** — {'; '.join(reasons)}."
            )
        else:
            # Lower-ranked items: brief reason for waiting
            if time_rem > 2.0:
                reasons.append("still has time before window ends")
            if demand < 4:
                reasons.append("lower demand right now")

            waste_key = (item, store_type)
            if waste_key in ITEM_WASTE_DESCRIPTIONS:
                desc = ITEM_WASTE_DESCRIPTIONS[waste_key]
                if "waste" in desc:
                    reasons.append("tends to waste at this store type anyway")

            if not reasons:
                reasons.append("can wait without risk")

            explanations.append(
                f"**{ITEM_DISPLAY_NAMES[item]}** (#{i+1}) — {'; '.join(reasons)}."
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
        "description": "Fresh morning, baked goods demand is high. Associate would default to pizza.",
    },
    {
        "name": "Friday 12 PM — Suburban Store (Lunch Peak)",
        "description": "Lunch rush at a busy suburban location. Pizza demand spikes.",
    },
    {
        "name": "Saturday 6 PM — Highway Store (Dinner)",
        "description": "Weekend dinner at a highway rest stop. Wings demand is highest.",
    },
    {
        "name": "Wednesday 2 PM — Urban Store (Afternoon Lull)",
        "description": "Slow afternoon. Lower demand across all items — priority ordering matters most.",
    },
    {
        "name": "Sunday 8 AM — Suburban Store (Weekend Morning)",
        "description": "Weekend brunch crowd. Baked goods and pizza both in demand.",
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
    "pizza": 6,
    "wings_2h": 5,
    "wings_4h": 8,
    "baked_goods": 15,
}

_FORECAST_STORE_MULT = {
    "urban": 1.4,
    "suburban": 1.0,
    "highway": 0.7,
}

_FORECAST_ITEM_TIME_CURVES = {
    "pizza": {
        0: 0.15, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.2,
        6: 0.4, 7: 0.6, 8: 0.7, 9: 0.8,
        10: 1.1, 11: 1.5, 12: 1.8, 13: 1.6, 14: 1.2,
        15: 0.9, 16: 0.8, 17: 0.9, 18: 1.0, 19: 1.0,
        20: 0.8, 21: 0.6, 22: 0.4, 23: 0.2,
    },
    "wings_2h": {
        0: 0.2, 1: 0.15, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.15,
        6: 0.3, 7: 0.4, 8: 0.5, 9: 0.6,
        10: 0.7, 11: 0.9, 12: 1.0, 13: 1.0, 14: 0.9,
        15: 0.9, 16: 1.0, 17: 1.3, 18: 1.6, 19: 1.7,
        20: 1.5, 21: 1.2, 22: 0.7, 23: 0.4,
    },
    "wings_4h": {
        0: 0.2, 1: 0.15, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.15,
        6: 0.3, 7: 0.4, 8: 0.5, 9: 0.6,
        10: 0.8, 11: 0.9, 12: 1.0, 13: 1.0, 14: 1.0,
        15: 1.0, 16: 1.1, 17: 1.2, 18: 1.4, 19: 1.5,
        20: 1.3, 21: 1.0, 22: 0.6, 23: 0.3,
    },
    "baked_goods": {
        0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.2,
        6: 0.8, 7: 1.3, 8: 1.5, 9: 1.4,
        10: 1.2, 11: 1.0, 12: 0.9, 13: 0.8, 14: 0.7,
        15: 0.7, 16: 0.7, 17: 0.6, 18: 0.5, 19: 0.4,
        20: 0.3, 21: 0.2, 22: 0.15, 23: 0.1,
    },
}

_FORECAST_ITEM_LCU = {
    "pizza": (6, True),
    "wings_2h": (5, True),
    "wings_4h": (8, True),
    "baked_goods": (1, False),
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


def compute_queue_timing(item: str, decision_hour: float) -> dict[str, Any]:
    """Calculate cook time, hold time, ready time, and expiry for a given item and decision point hour."""
    cook_time = ITEM_COOK_TIMES.get(item, 15)  # default 15 mins
    hold_time_hours = 4 if item == "wings_4h" else 24 if item == "baked_goods" else 2
    
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
        "optimal_first_item": "pizza",
        "optimal_order": ["pizza", "wings_2h", "baked_goods"],
        "features": {
            "decision_hour": 12,
            "store_type": "suburban",
            "day_of_week": "Friday",
            "is_weekend": False,
            "num_oven_items": 3,
            "pizza_forecast_demand": 18,
            "pizza_lcu": 6,
            "pizza_hold_time": 2,
            "pizza_time_remaining": 1.75,
            "wings_2h_forecast_demand": 12,
            "wings_2h_lcu": 5,
            "wings_2h_hold_time": 2,
            "wings_2h_time_remaining": 1.75,
            "baked_goods_forecast_demand": 15,
            "baked_goods_lcu": 1,
            "baked_goods_hold_time": 24,
            "baked_goods_time_remaining": 23.75,
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

