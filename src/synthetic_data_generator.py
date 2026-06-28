import json
import os
import random
import uuid
from datetime import datetime, timedelta
from typing import Any


# Reusable time-of-day demand curves (shared by SyntheticDataGenerator)
_C_MORNING      = {0:0.1,1:0.1,2:0.1,3:0.1,4:0.15,5:0.25,6:0.8,7:1.5,8:1.6,9:1.4,10:1.0,11:0.6,12:0.4,13:0.3,14:0.25,15:0.2,16:0.2,17:0.2,18:0.2,19:0.2,20:0.15,21:0.1,22:0.1,23:0.1}
_C_LUNCH        = {0:0.1,1:0.1,2:0.1,3:0.1,4:0.1,5:0.15,6:0.3,7:0.5,8:0.6,9:0.8,10:1.0,11:1.4,12:1.7,13:1.5,14:1.2,15:0.9,16:0.7,17:0.8,18:0.9,19:0.8,20:0.6,21:0.4,22:0.3,23:0.15}
_C_DINNER       = {0:0.15,1:0.1,2:0.1,3:0.1,4:0.1,5:0.15,6:0.3,7:0.4,8:0.5,9:0.6,10:0.7,11:0.9,12:1.0,13:1.0,14:0.9,15:0.9,16:1.0,17:1.4,18:1.6,19:1.7,20:1.5,21:1.2,22:0.7,23:0.4}
_C_ALL_DAY      = {0:0.2,1:0.15,2:0.1,3:0.1,4:0.1,5:0.2,6:0.4,7:0.7,8:0.8,9:0.9,10:1.0,11:1.1,12:1.2,13:1.1,14:1.0,15:0.9,16:1.0,17:1.2,18:1.3,19:1.2,20:1.0,21:0.8,22:0.5,23:0.3}
_C_LUNCH_DINNER = {0:0.1,1:0.1,2:0.1,3:0.1,4:0.1,5:0.15,6:0.3,7:0.5,8:0.6,9:0.8,10:1.0,11:1.3,12:1.5,13:1.4,14:1.1,15:0.9,16:1.0,17:1.3,18:1.5,19:1.4,20:1.0,21:0.7,22:0.4,23:0.2}
_C_MORNING_LUNCH= {0:0.1,1:0.1,2:0.1,3:0.1,4:0.15,5:0.3,6:0.7,7:1.2,8:1.4,9:1.3,10:1.1,11:1.2,12:1.0,13:0.8,14:0.6,15:0.5,16:0.4,17:0.4,18:0.4,19:0.3,20:0.2,21:0.15,22:0.1,23:0.1}
_C_MIDDAY       = {0:0.1,1:0.1,2:0.1,3:0.1,4:0.1,5:0.2,6:0.3,7:0.5,8:0.7,9:0.9,10:1.2,11:1.4,12:1.5,13:1.4,14:1.3,15:1.1,16:1.0,17:0.8,18:0.6,19:0.5,20:0.3,21:0.2,22:0.15,23:0.1}
_C_AFT_DINNER   = {0:0.1,1:0.1,2:0.1,3:0.1,4:0.1,5:0.15,6:0.3,7:0.4,8:0.5,9:0.6,10:0.7,11:0.9,12:1.0,13:1.1,14:1.2,15:1.3,16:1.4,17:1.5,18:1.5,19:1.4,20:1.2,21:1.0,22:0.6,23:0.3}


class SyntheticDataGenerator:
    """Generates synthetic cook scheduling data for hot food items."""

    ITEM_PROPERTIES = {
        "wings_bone_in":      {"hold_time_hours": 2, "lowest_cookable_unit": 5,  "exact_multiples": True,  "unit": "pieces",     "equipment": "oven"},
        "wings_boneless":     {"hold_time_hours": 2, "lowest_cookable_unit": 8,  "exact_multiples": True,  "unit": "pieces",     "equipment": "oven"},
        "chicken_strip":      {"hold_time_hours": 2, "lowest_cookable_unit": 3,  "exact_multiples": True,  "unit": "pieces",     "equipment": "oven"},
        "chicken_bite":       {"hold_time_hours": 2, "lowest_cookable_unit": 10, "exact_multiples": True,  "unit": "pieces",     "equipment": "oven"},
        "quesadilla":         {"hold_time_hours": 2, "lowest_cookable_unit": 5,  "exact_multiples": True,  "unit": "pieces",     "equipment": "oven"},
        "chicken_sandwich":   {"hold_time_hours": 2, "lowest_cookable_unit": 1,  "exact_multiples": True,  "unit": "sandwiches", "equipment": "oven"},
        "potato_wedge":       {"hold_time_hours": 2, "lowest_cookable_unit": 10, "exact_multiples": True,  "unit": "pieces",     "equipment": "oven"},
        "waffle_tot":         {"hold_time_hours": 2, "lowest_cookable_unit": 10, "exact_multiples": True,  "unit": "pieces",     "equipment": "oven"},
        "hash_brown":         {"hold_time_hours": 2, "lowest_cookable_unit": 2,  "exact_multiples": True,  "unit": "pieces",     "equipment": "oven"},
        "empanada":           {"hold_time_hours": 2, "lowest_cookable_unit": 2,  "exact_multiples": True,  "unit": "pieces",     "equipment": "oven"},
        "chimichanga":        {"hold_time_hours": 2, "lowest_cookable_unit": 2,  "exact_multiples": True,  "unit": "pieces",     "equipment": "oven"},
        "jamaican_turnover":  {"hold_time_hours": 2, "lowest_cookable_unit": 2,  "exact_multiples": True,  "unit": "pieces",     "equipment": "oven"},
        "jamaican_patty":     {"hold_time_hours": 2, "lowest_cookable_unit": 1,  "exact_multiples": False, "unit": "pieces",     "equipment": "oven"},
        "pupusa":             {"hold_time_hours": 2, "lowest_cookable_unit": 2,  "exact_multiples": True,  "unit": "pieces",     "equipment": "oven"},
        "beef_mini_taco":     {"hold_time_hours": 4, "lowest_cookable_unit": 8,  "exact_multiples": True,  "unit": "pieces",     "equipment": "oven"},
        "garlic_knot":        {"hold_time_hours": 2, "lowest_cookable_unit": 2,  "exact_multiples": True,  "unit": "pieces",     "equipment": "oven"},
        "kolache":            {"hold_time_hours": 2, "lowest_cookable_unit": 2,  "exact_multiples": True,  "unit": "pieces",     "equipment": "oven"},
        "croissant":          {"hold_time_hours": 4, "lowest_cookable_unit": 1,  "exact_multiples": True,  "unit": "pieces",     "equipment": "oven"},
        "breakfast_sandwich": {"hold_time_hours": 2, "lowest_cookable_unit": 1,  "exact_multiples": True,  "unit": "sandwiches", "equipment": "oven"},
        "sweet_croissant":    {"hold_time_hours": 4, "lowest_cookable_unit": 6,  "exact_multiples": True,  "unit": "pieces",     "equipment": "oven"},
        "danish":             {"hold_time_hours": 4, "lowest_cookable_unit": 6,  "exact_multiples": True,  "unit": "pieces",     "equipment": "oven"},
        "pizza_slice":        {"hold_time_hours": 2, "lowest_cookable_unit": 6,  "exact_multiples": True,  "unit": "slices",     "equipment": "oven"},
        "pizza_stuffed":      {"hold_time_hours": 2, "lowest_cookable_unit": 2,  "exact_multiples": True,  "unit": "pieces",     "equipment": "oven"},
        "hot_dog":            {"hold_time_hours": 4, "lowest_cookable_unit": 2,  "exact_multiples": True,  "unit": "pieces",     "equipment": "grill"},
        "sausage":            {"hold_time_hours": 4, "lowest_cookable_unit": 2,  "exact_multiples": True,  "unit": "pieces",     "equipment": "grill"},
        "taquito":            {"hold_time_hours": 4, "lowest_cookable_unit": 2,  "exact_multiples": True,  "unit": "pieces",     "equipment": "grill"},
        "buffalo_roller":     {"hold_time_hours": 4, "lowest_cookable_unit": 2,  "exact_multiples": False, "unit": "pieces",     "equipment": "grill"},
        "corn_dog":           {"hold_time_hours": 4, "lowest_cookable_unit": 2,  "exact_multiples": True,  "unit": "pieces",     "equipment": "grill"},
    }

    # Probability that actual demand exceeds forecast in a given window,
    # triggering a mid-window restock cook.
    SELLTHROUGH_PROBABILITY = 0.15

    STORE_TYPES = ["urban", "suburban", "highway"]

    # Forecast cycle: 6 AM to 6 AM next day (24 hours, store runs 24/7)
    OPERATING_START = 6   # 6 AM
    OPERATING_END = 30    # 6 AM next day (represented as hour 30 for window math)

    # Demand multipliers by store type (urban highest, highway lowest)
    STORE_DEMAND_MULTIPLIER = {
        "urban": 1.4,
        "suburban": 1.0,
        "highway": 0.7,
    }

    # Base demand per window (units expected to sell in one hold-time window)
    BASE_DEMAND_PER_WINDOW = {
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

    # Default time-of-day demand multiplier (fallback for items without specific curve)
    TIME_OF_DAY_CURVE = {
        0: 0.2, 1: 0.15, 2: 0.1, 3: 0.1, 4: 0.15, 5: 0.2,
        6: 0.5, 7: 0.7, 8: 0.8, 9: 0.9,
        10: 1.0, 11: 1.3, 12: 1.5, 13: 1.4, 14: 1.1,
        15: 0.9, 16: 0.9, 17: 1.1, 18: 1.3, 19: 1.4,
        20: 1.2, 21: 0.8, 22: 0.5, 23: 0.3,
    }

    # Item-specific time-of-day demand curves (references module-level curve constants)
    ITEM_TIME_CURVES = {
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

    # Item-specific waste propensity by store type (higher = more unsold units)
    ITEM_WASTE_MULTIPLIER = {
        "wings_bone_in":      {"urban": 0.9, "suburban": 1.0, "highway": 1.2},
        "wings_boneless":     {"urban": 0.9, "suburban": 1.0, "highway": 1.1},
        "chicken_strip":      {"urban": 1.0, "suburban": 1.0, "highway": 1.1},
        "chicken_bite":       {"urban": 1.0, "suburban": 1.0, "highway": 1.1},
        "quesadilla":         {"urban": 0.9, "suburban": 1.0, "highway": 1.2},
        "chicken_sandwich":   {"urban": 0.9, "suburban": 1.0, "highway": 1.1},
        "potato_wedge":       {"urban": 1.0, "suburban": 1.0, "highway": 0.9},
        "waffle_tot":         {"urban": 1.0, "suburban": 1.0, "highway": 1.0},
        "hash_brown":         {"urban": 1.0, "suburban": 1.0, "highway": 1.0},
        "empanada":           {"urban": 0.8, "suburban": 1.0, "highway": 1.4},
        "chimichanga":        {"urban": 0.9, "suburban": 1.0, "highway": 1.3},
        "jamaican_turnover":  {"urban": 0.8, "suburban": 1.0, "highway": 1.4},
        "jamaican_patty":     {"urban": 0.8, "suburban": 1.0, "highway": 1.5},
        "pupusa":             {"urban": 0.8, "suburban": 1.0, "highway": 1.5},
        "beef_mini_taco":     {"urban": 0.9, "suburban": 1.0, "highway": 1.2},
        "garlic_knot":        {"urban": 1.0, "suburban": 1.0, "highway": 1.1},
        "kolache":            {"urban": 1.0, "suburban": 1.0, "highway": 1.0},
        "croissant":          {"urban": 1.0, "suburban": 1.0, "highway": 1.1},
        "breakfast_sandwich": {"urban": 1.0, "suburban": 1.0, "highway": 1.0},
        "sweet_croissant":    {"urban": 1.0, "suburban": 1.0, "highway": 1.2},
        "danish":             {"urban": 1.0, "suburban": 1.0, "highway": 1.2},
        "pizza_slice":        {"urban": 0.8, "suburban": 1.0, "highway": 1.3},
        "pizza_stuffed":      {"urban": 0.9, "suburban": 1.0, "highway": 1.2},
        "hot_dog":            {"urban": 1.0, "suburban": 1.0, "highway": 0.9},
        "sausage":            {"urban": 1.0, "suburban": 1.0, "highway": 0.9},
        "taquito":            {"urban": 0.9, "suburban": 1.0, "highway": 1.0},
        "buffalo_roller":     {"urban": 1.0, "suburban": 1.0, "highway": 1.1},
        "corn_dog":           {"urban": 1.0, "suburban": 1.0, "highway": 0.9},
    }

    # Write-off quality distribution
    WRITEOFF_QUALITY = {
        "accurate": 0.60,
        "gap": 0.15,
        "counting_error": 0.20,
        "major_discrepancy": 0.05,
    }

    def __init__(self, seed: int = 42, num_days: int = 180, output_dir: str = "data"):
        self.seed = seed
        self.num_days = num_days
        self.output_dir = output_dir
        self.rng = random.Random(seed)
        self.start_date = datetime(2025, 1, 1)

        self.cook_logs: list[dict[str, Any]] = []
        self.pos_sales: list[dict[str, Any]] = []
        self.write_off_logs: list[dict[str, Any]] = []

    def _is_weekend(self, date: datetime) -> bool:
        return date.weekday() >= 5

    def _get_item_windows(self, item: str) -> list[dict[str, int]]:
        """Get the forecast windows for an item. Each window = one hold-time period."""
        hold = self.ITEM_PROPERTIES[item]["hold_time_hours"]
        windows = []
        current = self.OPERATING_START
        while current < self.OPERATING_END:
            end = min(current + hold, self.OPERATING_END)
            windows.append({"start_hour": current, "end_hour": end})
            current = end
        return windows

    def _get_window_label(self, window: dict[str, int]) -> str:
        """Generate a human-readable label for a window, e.g. '06:00-10:00'."""
        sh = window["start_hour"] % 24
        eh = window["end_hour"] % 24
        return f"{sh:02d}:00-{eh:02d}:00"

    def _get_time_of_day_mult(self, window: dict[str, int], item: str | None = None) -> float:
        """Get the time-of-day demand multiplier based on window midpoint.

        Uses item-specific curve if available, otherwise falls back to default.
        """
        midpoint = (window["start_hour"] + window["end_hour"]) // 2
        hour = midpoint % 24
        if item and item in self.ITEM_TIME_CURVES:
            return self.ITEM_TIME_CURVES[item].get(hour, 0.2)
        return self.TIME_OF_DAY_CURVE.get(hour, 0.2)

    def _get_lcu_valid_qty(self, item: str, raw: int) -> int:
        """Round a raw quantity up to a valid cookable amount respecting LCU rules."""
        props = self.ITEM_PROPERTIES[item]
        lcu = props["lowest_cookable_unit"]
        if props["exact_multiples"]:
            return max(lcu, ((raw + lcu - 1) // lcu) * lcu)
        else:
            return max(lcu, raw)

    def _get_demand(self, item: str, window: dict[str, int], store_type: str, date: datetime) -> int:
        """Calculate the post-rounding forecast for a given item/window/store/date.

        The upstream API provides fractional hourly forecasts which are rounded
        to valid cookable quantities before reaching this prototype. This method
        simulates that final rounded forecast:
        - For exact-multiple items: result is always a multiple of LCU.
        - For non-exact-multiple items: result is any integer >= LCU.
        """
        base = self.BASE_DEMAND_PER_WINDOW[item]
        store_mult = self.STORE_DEMAND_MULTIPLIER[store_type]
        weekend_mult = 1.3 if self._is_weekend(date) else 1.0
        tod_mult = self._get_time_of_day_mult(window, item)
        noise = self.rng.uniform(0.8, 1.2)
        raw = max(1, round(base * store_mult * weekend_mult * tod_mult * noise))
        return self._get_lcu_valid_qty(item, raw)

    def _generate_cook_timestamp(self, date: datetime, window: dict[str, int]) -> datetime:
        """Generate a cook timestamp near the start of the window."""
        start_hour = window["start_hour"]
        # Cook happens within the first 15 minutes of the window
        cook_time = date.replace(hour=0, minute=0, second=0)
        cook_time += timedelta(hours=start_hour, minutes=self.rng.randint(0, 15))
        return cook_time

    def _generate_sale_timestamps(
        self, date: datetime, window: dict[str, int], quantity: int
    ) -> list[datetime]:
        """Generate POS sale timestamps spread throughout the window."""
        start_hour = window["start_hour"]
        end_hour = window["end_hour"]
        window_minutes = (end_hour - start_hour) * 60
        window_start = date.replace(hour=0, minute=0, second=0) + timedelta(hours=start_hour)

        timestamps = []
        for _ in range(quantity):
            # Sales distributed throughout window (Gaussian centered at midpoint)
            midpoint = window_minutes / 2
            spread = window_minutes / 4  # σ = quarter of window
            offset = int(self.rng.gauss(midpoint, spread))
            offset = max(5, min(window_minutes - 5, offset))
            sale_time = window_start + timedelta(minutes=offset)
            timestamps.append(sale_time)
        return sorted(timestamps)

    def _generate_writeoff_entry(
        self,
        cook_event_id: str,
        item: str,
        store_type: str,
        date: datetime,
        window: dict[str, int],
        window_label: str,
        cooked_qty: int,
        sold_qty: int,
    ) -> dict[str, Any] | None:
        """Generate a write-off log entry with realistic quality issues."""
        inferred_writeoff = max(0, cooked_qty - sold_qty)

        # Determine quality category
        roll = self.rng.random()
        cumulative = 0.0
        quality_type = "accurate"
        for qtype, prob in self.WRITEOFF_QUALITY.items():
            cumulative += prob
            if roll <= cumulative:
                quality_type = qtype
                break

        if quality_type == "gap":
            # No write-off logged at all
            return None

        # Calculate logged write-off based on quality type
        if quality_type == "accurate":
            logged_writeoff = inferred_writeoff
        elif quality_type == "counting_error":
            error = self.rng.choice([-2, -1, 1, 2])
            logged_writeoff = max(0, inferred_writeoff + error)
        else:  # major_discrepancy
            error = self.rng.choice([-5, -4, -3, 3, 4, 5])
            logged_writeoff = max(0, inferred_writeoff + error)

        # Write-off logged at end of window + delay (30-120 min)
        end_hour = window["end_hour"]
        base_time = date.replace(hour=0, minute=0, second=0) + timedelta(hours=end_hour)
        writeoff_time = base_time + timedelta(minutes=self.rng.randint(30, 120))

        return {
            "writeoff_id": str(uuid.UUID(int=self.rng.getrandbits(128))),
            "cook_event_id": cook_event_id,
            "item": item,
            "store_type": store_type,
            "date": date.strftime("%Y-%m-%d"),
            "window": window_label,
            "logged_writeoff_qty": logged_writeoff,
            "inferred_writeoff_qty": inferred_writeoff,
            "writeoff_timestamp": writeoff_time.isoformat(),
            "quality_type": quality_type,
            "delay_minutes": self.rng.randint(30, 120),
        }

    def _make_cook_log(self, cook_event_id, store_id, store_type, item,
                       date, window_label, window, demand, cooked_qty,
                       cook_timestamp, cook_type):
        """Create a cook log entry dict."""
        props = self.ITEM_PROPERTIES[item]
        return {
            "cook_event_id": cook_event_id,
            "store_id": store_id,
            "store_type": store_type,
            "item": item,
            "date": date.strftime("%Y-%m-%d"),
            "window": window_label,
            "window_start_hour": window["start_hour"] % 24,
            "window_end_hour": window["end_hour"] % 24,
            "day_of_week": date.strftime("%A"),
            "is_weekend": self._is_weekend(date),
            "forecast_demand": demand,
            "cooked_qty": cooked_qty,
            "cook_timestamp": cook_timestamp.isoformat(),
            "cook_type": cook_type,
            "hold_time_hours": props["hold_time_hours"],
            "lowest_cookable_unit": props["lowest_cookable_unit"],
            "exact_multiples": props["exact_multiples"],
            "equipment": props["equipment"],
        }

    def generate(self) -> dict[str, list[dict[str, Any]]]:
        """Generate all synthetic data for the configured period."""
        self.cook_logs.clear()
        self.pos_sales.clear()
        self.write_off_logs.clear()

        for day_offset in range(self.num_days):
            date = self.start_date + timedelta(days=day_offset)

            for store_type in self.STORE_TYPES:
                store_id = f"{store_type}_{self.rng.randint(1000, 9999)}"

                for item in self.ITEM_PROPERTIES:
                    windows = self._get_item_windows(item)

                    for window in windows:
                        window_label = self._get_window_label(window)
                        forecast = self._get_demand(item, window, store_type, date)

                        # --- Initial cook ---
                        initial_qty = forecast
                        cook_event_id = str(uuid.UUID(int=self.rng.getrandbits(128)))
                        cook_timestamp = self._generate_cook_timestamp(date, window)

                        self.cook_logs.append(self._make_cook_log(
                            cook_event_id, store_id, store_type, item,
                            date, window_label, window, forecast, initial_qty,
                            cook_timestamp, "initial"
                        ))

                        # --- Determine actual demand (may exceed forecast) ---
                        # ~15% of windows have a sell-through requiring restock
                        restock_qty = 0
                        restock_event_id = None
                        if self.rng.random() < self.SELLTHROUGH_PROBABILITY:
                            # Actual demand exceeds forecast by 20-80%
                            excess_pct = self.rng.uniform(0.2, 0.8)
                            excess_raw = max(1, round(forecast * excess_pct))
                            restock_qty = self._get_lcu_valid_qty(item, excess_raw)

                            # Restock cook happens mid-window
                            window_minutes = (window["end_hour"] - window["start_hour"]) * 60
                            restock_offset = self.rng.randint(
                                window_minutes // 3, 2 * window_minutes // 3
                            )
                            restock_time = (
                                date.replace(hour=0, minute=0, second=0)
                                + timedelta(hours=window["start_hour"], minutes=restock_offset)
                            )

                            restock_event_id = str(uuid.UUID(int=self.rng.getrandbits(128)))
                            self.cook_logs.append(self._make_cook_log(
                                restock_event_id, store_id, store_type, item,
                                date, window_label, window, restock_qty, restock_qty,
                                restock_time, "restock"
                            ))

                        total_cooked = initial_qty + restock_qty

                        # --- Sold quantity ---
                        # Apply item-specific waste propensity by store type
                        waste_mult = self.ITEM_WASTE_MULTIPLIER.get(
                            item, {}).get(store_type, 1.0)

                        if restock_qty > 0:
                            # Demand consumed most/all of the initial + restock
                            actual_demand = initial_qty + self.rng.randint(
                                1, restock_qty
                            )
                        else:
                            # Base demand noise, adjusted by waste propensity
                            # Higher waste_mult → demand undershoots forecast more
                            demand_adjustment = self.rng.randint(-2, 2)
                            waste_reduction = round((waste_mult - 1.0) * forecast * 0.3)
                            actual_demand = forecast + demand_adjustment - waste_reduction
                        sold_qty = max(0, min(total_cooked, actual_demand))

                        # --- POS sales (linked to initial cook; restock sales after restock time) ---
                        sale_timestamps = self._generate_sale_timestamps(
                            date, window, sold_qty
                        )
                        for ts in sale_timestamps:
                            # Link sale to restock event if it happened after restock
                            linked_event = cook_event_id
                            if restock_event_id and ts > restock_time:
                                linked_event = restock_event_id
                            self.pos_sales.append(
                                {
                                    "sale_id": str(uuid.UUID(int=self.rng.getrandbits(128))),
                                    "cook_event_id": linked_event,
                                    "store_id": store_id,
                                    "store_type": store_type,
                                    "item": item,
                                    "date": date.strftime("%Y-%m-%d"),
                                    "window": window_label,
                                    "quantity": 1,
                                    "sale_timestamp": ts.isoformat(),
                                }
                            )

                        # --- Write-off (for initial cook; restock write-off separate) ---
                        # Split sold qty between initial and restock
                        initial_sold = min(initial_qty, sold_qty)
                        restock_sold = sold_qty - initial_sold

                        writeoff = self._generate_writeoff_entry(
                            cook_event_id, item, store_type, date,
                            window, window_label, initial_qty, initial_sold,
                        )
                        if writeoff is not None:
                            self.write_off_logs.append(writeoff)

                        if restock_qty > 0:
                            writeoff_r = self._generate_writeoff_entry(
                                restock_event_id, item, store_type, date,
                                window, window_label, restock_qty, restock_sold,
                            )
                            if writeoff_r is not None:
                                self.write_off_logs.append(writeoff_r)

        return {
            "cook_logs": self.cook_logs,
            "pos_sales": self.pos_sales,
            "write_off_logs": self.write_off_logs,
        }

    def save(self, output_dir: str | None = None) -> dict[str, str]:
        """Save generated data to JSON files."""
        out = output_dir or self.output_dir
        os.makedirs(out, exist_ok=True)

        paths = {}
        for name, data in [
            ("cook_logs", self.cook_logs),
            ("pos_sales", self.pos_sales),
            ("write_off_logs", self.write_off_logs),
        ]:
            path = os.path.join(out, f"{name}.json")
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            paths[name] = path

        return paths
