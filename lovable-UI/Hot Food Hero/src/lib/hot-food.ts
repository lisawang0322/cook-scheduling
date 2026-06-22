// Shared types, item definitions, ranking logic, and time utilities for the
// 7-Eleven Hot Food Assistant (tablet + scenario simulator).

import { DEFAULT_STORE_CONFIG, getDaypartRule, type StoreConfig } from "./store-config";

export type CookItem = {
  id: string;
  name: string;
  emoji: string;
  recommended: number;
  lcu: number;
  cookTimeMin: number;
  holdTimeHr: number;
};

export type Scenario = {
  id: string;
  storeId: string;
  storeType: string;
  dayLabel: string;
  time: string; // "12:30 PM"
  reason: string;
  items: CookItem[];
};

export type OvenItemId =
  | "wings_bone_in" | "wings_boneless"
  | "chicken_strip" | "chicken_bite" | "quesadilla" | "chicken_sandwich"
  | "potato_wedge" | "waffle_tot" | "hash_brown"
  | "empanada" | "chimichanga" | "jamaican_turnover" | "jamaican_patty" | "pupusa"
  | "beef_mini_taco" | "garlic_knot" | "kolache"
  | "croissant" | "breakfast_sandwich" | "sweet_croissant" | "danish"
  | "pizza_slice" | "pizza_stuffed"
  | "hot_dog" | "sausage" | "taquito" | "buffalo_roller" | "corn_dog";

export const ITEM_DEFS: Record<
  OvenItemId,
  { name: string; emoji: string; lcu: number; cookTimeMin: number; holdTimeHr: number; peakHours: [number, number]; peakLabel: string }
> = {
  // Wings
  wings_bone_in:        { name: "Bone-in Wings",             emoji: "🍗", lcu: 5,  cookTimeMin: 25, holdTimeHr: 2, peakHours: [17, 21], peakLabel: "dinner" },
  wings_boneless:       { name: "Boneless Wings",            emoji: "🍗", lcu: 8,  cookTimeMin: 20, holdTimeHr: 2, peakHours: [17, 21], peakLabel: "dinner" },
  // Chicken
  chicken_strip:        { name: "Chicken Strips",            emoji: "🍗", lcu: 3,  cookTimeMin: 15, holdTimeHr: 2, peakHours: [11, 19], peakLabel: "lunch & dinner" },
  chicken_bite:         { name: "Chicken Bites",             emoji: "�", lcu: 10, cookTimeMin: 15, holdTimeHr: 2, peakHours: [11, 19], peakLabel: "lunch & dinner" },
  quesadilla:           { name: "Mini Chicken Quesadilla",   emoji: "🫓", lcu: 5,  cookTimeMin: 8,  holdTimeHr: 2, peakHours: [11, 16], peakLabel: "lunch" },
  chicken_sandwich:     { name: "Crispy Chicken Sandwich",   emoji: "🥪", lcu: 1,  cookTimeMin: 8,  holdTimeHr: 2, peakHours: [11, 20], peakLabel: "lunch & dinner" },
  // Potatoes & Tots
  potato_wedge:         { name: "Seasoned Potato Wedges",    emoji: "🍟", lcu: 10, cookTimeMin: 12, holdTimeHr: 2, peakHours: [11, 20], peakLabel: "lunch & dinner" },
  waffle_tot:           { name: "Waffle Potato Tots",        emoji: "🧇", lcu: 10, cookTimeMin: 10, holdTimeHr: 2, peakHours: [7,  13], peakLabel: "morning & lunch" },
  hash_brown:           { name: "Hash Brown Patties",        emoji: "🥔", lcu: 2,  cookTimeMin: 10, holdTimeHr: 2, peakHours: [6,  10], peakLabel: "morning" },
  // Latin & International
  empanada:             { name: "Savory Empanada",           emoji: "🥟", lcu: 2,  cookTimeMin: 15, holdTimeHr: 2, peakHours: [10, 17], peakLabel: "midday" },
  chimichanga:          { name: "Beef Chimichanga",          emoji: "�", lcu: 2,  cookTimeMin: 12, holdTimeHr: 2, peakHours: [11, 17], peakLabel: "lunch" },
  jamaican_turnover:    { name: "Jamaican-Style Turnover",   emoji: "🥟", lcu: 2,  cookTimeMin: 15, holdTimeHr: 2, peakHours: [8,  15], peakLabel: "morning & lunch" },
  jamaican_patty:       { name: "Jamaican-Style Patty",      emoji: "🫔", lcu: 1,  cookTimeMin: 10, holdTimeHr: 2, peakHours: [10, 16], peakLabel: "midday" },
  pupusa:               { name: "Stuffed Corn Cake",         emoji: "🫓", lcu: 2,  cookTimeMin: 10, holdTimeHr: 2, peakHours: [11, 17], peakLabel: "lunch" },
  // Small Bites & Snacks
  beef_mini_taco:       { name: "Mini Beef Taco Bites",      emoji: "🌮", lcu: 8,  cookTimeMin: 8,  holdTimeHr: 4, peakHours: [10, 16], peakLabel: "midday" },
  garlic_knot:          { name: "Artisan Garlic Knots",      emoji: "🫓", lcu: 2,  cookTimeMin: 10, holdTimeHr: 2, peakHours: [17, 21], peakLabel: "dinner" },
  kolache:              { name: "Sausage Kolache",           emoji: "🌭", lcu: 2,  cookTimeMin: 8,  holdTimeHr: 2, peakHours: [7,  11], peakLabel: "morning" },
  // Breakfast
  croissant:            { name: "Breakfast Croissant",       emoji: "🥐", lcu: 1,  cookTimeMin: 5,  holdTimeHr: 4, peakHours: [6,  10], peakLabel: "morning" },
  breakfast_sandwich:   { name: "Breakfast Sandwich",        emoji: "�", lcu: 1,  cookTimeMin: 5,  holdTimeHr: 2, peakHours: [6,  10], peakLabel: "morning" },
  sweet_croissant:      { name: "Sweet Pastry Croissant",    emoji: "🥐", lcu: 6,  cookTimeMin: 8,  holdTimeHr: 4, peakHours: [7,  11], peakLabel: "morning" },
  danish:               { name: "Glazed Danish Pastry",      emoji: "🥐", lcu: 6,  cookTimeMin: 10, holdTimeHr: 4, peakHours: [7,  11], peakLabel: "morning" },
  // Pizza
  pizza_slice:          { name: "Pizza Slice",               emoji: "🍕", lcu: 6,  cookTimeMin: 10, holdTimeHr: 2, peakHours: [11, 14], peakLabel: "lunch" },
  pizza_stuffed:        { name: "Stuffed Pizza Pocket",      emoji: "🍕", lcu: 2,  cookTimeMin: 8,  holdTimeHr: 2, peakHours: [11, 16], peakLabel: "lunch" },
  // Grill
  hot_dog:              { name: "Beef Hot Dog",              emoji: "🌭", lcu: 2,  cookTimeMin: 15, holdTimeHr: 4, peakHours: [11, 20], peakLabel: "lunch & dinner" },
  sausage:              { name: "Smoked Sausage Link",       emoji: "🌭", lcu: 2,  cookTimeMin: 20, holdTimeHr: 4, peakHours: [11, 20], peakLabel: "lunch & dinner" },
  taquito:              { name: "Chicken Taquito",           emoji: "🌯", lcu: 2,  cookTimeMin: 20, holdTimeHr: 4, peakHours: [8,  21], peakLabel: "all day" },
  buffalo_roller:       { name: "Buffalo Chicken Roller",    emoji: "🌯", lcu: 2,  cookTimeMin: 15, holdTimeHr: 4, peakHours: [16, 21], peakLabel: "afternoon & dinner" },
  corn_dog:             { name: "Corn Dog",                  emoji: "🌭", lcu: 2,  cookTimeMin: 15, holdTimeHr: 4, peakHours: [11, 20], peakLabel: "lunch & dinner" },
};


// ---------- Time utilities ----------
export function formatHour(hour24: number): string {
  const h = ((hour24 % 12) + 12) % 12 || 12;
  const ap = hour24 >= 12 && hour24 < 24 ? "PM" : "AM";
  return `${h}:00 ${ap}`;
}
export function addMinutes(time: string, mins: number) {
  const m = time.match(/(\d+):(\d+)\s*(AM|PM)/i);
  if (!m) return time;
  let h = parseInt(m[1], 10) % 12;
  if (m[3].toUpperCase() === "PM") h += 12;
  const date = new Date();
  date.setHours(h, parseInt(m[2], 10) + mins, 0, 0);
  let hh = date.getHours();
  const mm = date.getMinutes();
  const ap = hh >= 12 ? "PM" : "AM";
  hh = hh % 12 || 12;
  return `${hh}:${mm.toString().padStart(2, "0")} ${ap}`;
}
export function addHours(time: string, hours: number) {
  return addMinutes(time, Math.round(hours * 60));
}

// ---------- Simulator input type ----------
export type SimulatorInput = {
  storeType: "Urban" | "Suburban" | "Highway";
  dayLabel: string;
  hour: number; // 6..23
  taquitosOn: boolean;
};

// ---------- Fractional hourly demand model ----------
// Base demand per hour by store type and item. Peak-hour multipliers are applied
// on top. This simulates the fractional hourly forecast the AI team would send.

const STORE_BASE: Record<SimulatorInput["storeType"], Record<OvenItemId, number>> = {
  Urban: {
    wings_bone_in: 8,  wings_boneless: 10,
    chicken_strip: 5,  chicken_bite: 6,  quesadilla: 4,  chicken_sandwich: 6,
    potato_wedge: 5,   waffle_tot: 6,    hash_brown: 4,
    empanada: 5,       chimichanga: 4,   jamaican_turnover: 5, jamaican_patty: 3, pupusa: 4,
    beef_mini_taco: 6, garlic_knot: 4,   kolache: 3,
    croissant: 4,      breakfast_sandwich: 5, sweet_croissant: 3, danish: 3,
    pizza_slice: 8,    pizza_stuffed: 4,
    hot_dog: 5,        sausage: 4,       taquito: 7,  buffalo_roller: 4, corn_dog: 3,
  },
  Suburban: {
    wings_bone_in: 5,  wings_boneless: 6,
    chicken_strip: 5,  chicken_bite: 5,  quesadilla: 3,  chicken_sandwich: 5,
    potato_wedge: 6,   waffle_tot: 7,    hash_brown: 4,
    empanada: 3,       chimichanga: 3,   jamaican_turnover: 2, jamaican_patty: 2, pupusa: 2,
    beef_mini_taco: 5, garlic_knot: 3,   kolache: 3,
    croissant: 5,      breakfast_sandwich: 5, sweet_croissant: 3, danish: 4,
    pizza_slice: 5,    pizza_stuffed: 3,
    hot_dog: 5,        sausage: 5,       taquito: 6,  buffalo_roller: 3, corn_dog: 4,
  },
  Highway: {
    wings_bone_in: 6,  wings_boneless: 7,
    chicken_strip: 5,  chicken_bite: 5,  quesadilla: 3,  chicken_sandwich: 6,
    potato_wedge: 8,   waffle_tot: 5,    hash_brown: 5,
    empanada: 2,       chimichanga: 2,   jamaican_turnover: 2, jamaican_patty: 1, pupusa: 1,
    beef_mini_taco: 4, garlic_knot: 3,   kolache: 4,
    croissant: 5,      breakfast_sandwich: 6, sweet_croissant: 3, danish: 3,
    pizza_slice: 6,    pizza_stuffed: 3,
    hot_dog: 7,        sausage: 6,       taquito: 7,  buffalo_roller: 4, corn_dog: 4,
  },
};

const WEEKEND = new Set(["Saturday", "Sunday"]);

function hourMultiplier(hour: number, peak: [number, number]) {
  const [s, e] = peak;
  if (hour >= s && hour <= e) return 1.8;
  const dist = hour < s ? s - hour : hour - e;
  if (dist === 1) return 1.1;
  if (dist === 2) return 0.7;
  return 0.3;
}

// Returns the fractional (float) hourly demand — no rounding applied here.
// This is the simulated equivalent of one hour's worth of AI-team forecast data.
export function forecastDemand(
  id: OvenItemId,
  storeType: SimulatorInput["storeType"],
  dayLabel: string,
  hour: number,
): number {
  const def = ITEM_DEFS[id];
  const base = STORE_BASE[storeType][id];
  const peakMul = hourMultiplier(hour, def.peakHours);
  const dayMul = WEEKEND.has(dayLabel) ? 1.25 : 1.0;
  const lateNightMul =
    storeType === "Highway" && hour >= 21 && (id === "wings_bone_in" || id === "wings_boneless") ? 1.2 : 1;
  return base * peakMul * dayMul * lateNightMul; // fractional — intentionally not rounded
}

// ---------- Daypart helpers ----------

// Returns all daypart start-hours for an item given its hold time.
// The 24-hour cycle starts at 6 AM (store open), wraps through midnight.
// e.g. holdTimeHr=2  → [6,8,10,12,14,16,18,20,22,0,2,4]
//      holdTimeHr=4  → [6,10,14,18,22,2]
//      holdTimeHr=24 → [6]
function getDaypartStarts(holdTimeHr: number): number[] {
  const count = Math.round(24 / holdTimeHr);
  return Array.from({ length: count }, (_, i) => (6 + i * holdTimeHr) % 24);
}

// Returns the start-hour of the daypart that contains `hour`.
function getCurrentDaypartStart(hour: number, holdTimeHr: number): number {
  const hoursFromOpen = ((hour - 6) + 24) % 24;
  const idx = Math.floor(hoursFromOpen / holdTimeHr);
  return (6 + idx * holdTimeHr) % 24;
}

// ---------- LCU rounding ----------
// Rounds a fractional demand value to the nearest LCU boundary using midpoint rule:
//   if value < nearestLower + 0.5 × LCU → round down
//   otherwise → round up
function lcuRound(fractional: number, lcu: number): number {
  if (fractional <= 0) return 0;
  const lower = Math.floor(fractional / lcu) * lcu;
  return fractional < lower + 0.5 * lcu ? lower : lower + lcu;
}

// ---------- Allocation engine ----------
// Implements the full backend allocation algorithm for a single item.
// Returns a map of { daypartStartHour → allocatedQty } for every daypart in the day.

function allocateItem(
  id: OvenItemId,
  input: SimulatorInput,
  config: StoreConfig,
): Map<number, number> {
  const def = ITEM_DEFS[id];
  const daypartStarts = getDaypartStarts(def.holdTimeHr);

  // Step 1: fractional forecast per daypart = sum of hourly values in the window.
  const daypartForecast = new Map<number, number>();
  for (const dpStart of daypartStarts) {
    let total = 0;
    for (let i = 0; i < def.holdTimeHr; i++) {
      const h = (dpStart + i) % 24;
      total += forecastDemand(id, input.storeType, input.dayLabel, h);
    }
    daypartForecast.set(dpStart, total);
  }

  // Step 2: identify eligible dayparts (not DO NOT COOK).
  const eligible = daypartStarts.filter(
    (dp) => !getDaypartRule(config, id, dp).doNotCook,
  );

  // Step 3: daily macro = sum of all 24 fractional hourly forecasts.
  let macro = 0;
  for (let h = 0; h < 24; h++) {
    macro += forecastDemand(id, input.storeType, input.dayLabel, h);
  }

  // Step 4: sum of min presentations for eligible dayparts.
  const sumMinPres = eligible.reduce(
    (sum, dp) => sum + getDaypartRule(config, id, dp).minPresentation,
    0,
  );

  // Step 5: remaining = macro - sumMinPres (Sprint 1: sumMinPres = 0).
  let remaining = macro - sumMinPres;

  // Step 6: sort eligible dayparts by their fractional forecast descending.
  const sorted = [...eligible].sort(
    (a, b) => (daypartForecast.get(b) ?? 0) - (daypartForecast.get(a) ?? 0),
  );

  // Step 7: allocation loop.
  const result = new Map<number, number>(daypartStarts.map((dp) => [dp, 0]));

  const isPizza = id === "pizza_slice";
  const pizzaCap = 2 * def.lcu; // 12 slices = 2 whole pizzas

  for (const dp of sorted) {
    if (remaining >= 0) {
      const raw = daypartForecast.get(dp) ?? 0;
      const minPres = getDaypartRule(config, id, dp).minPresentation;
      const rounded = lcuRound(raw, def.lcu);
      const adjusted = rounded + minPres;

      if (isPizza) {
        // Pizza cap check: skip daypart if allocation would exceed 2 pizzas.
        if (adjusted <= pizzaCap) {
          result.set(dp, adjusted);
          remaining -= adjusted;
        }
        // else: skip — result stays 0, remaining unchanged, loop continues.
      } else {
        result.set(dp, adjusted);
        remaining -= adjusted;
      }
    }
    // remaining < 0: daypart stays 0 (already initialized).
  }

  return result;
}

// ---------- Public forecast API ----------

export type ForecastRow = {
  id: OvenItemId;
  rawDaypartForecast: number; // fractional demand for the current daypart (scoring signal)
  rounded: number;            // LCU-rounded quantity allocated to the current daypart
  included: boolean;          // rounded > 0
};

export function generateForecasts(
  input: SimulatorInput,
  config: StoreConfig = DEFAULT_STORE_CONFIG,
): ForecastRow[] {
  return (Object.keys(ITEM_DEFS) as OvenItemId[]).map((id) => {
    const def = ITEM_DEFS[id];
    const dpStart = getCurrentDaypartStart(input.hour, def.holdTimeHr);
    const allocation = allocateItem(id, input, config);
    const rounded = allocation.get(dpStart) ?? 0;

    // Raw fractional forecast for this daypart (used in scoring).
    let rawDaypartForecast = 0;
    for (let i = 0; i < def.holdTimeHr; i++) {
      rawDaypartForecast += forecastDemand(id, input.storeType, input.dayLabel, (dpStart + i) % 24);
    }

    return { id, rawDaypartForecast, rounded, included: rounded > 0 };
  });
}

// ---------- v1 deterministic ranking ----------

function isInPeak(hour: number, [start, end]: [number, number]) {
  return hour >= start && hour <= end;
}

function reasonFor(top: CookItem, hour: number): string {
  const def = (Object.entries(ITEM_DEFS).find(([, d]) => d.name === top.name) ?? [])[1];
  const peak = def && isInPeak(hour, def.peakHours);
  const windowText = top.holdTimeHr <= 2
    ? `its ${top.holdTimeHr}-hour window closes soonest`
    : `it has the strongest demand right now`;
  if (peak) {
    return `${top.name} is first — ${def!.peakLabel} demand is peaking and ${windowText}.`;
  }
  return `${top.name} is first — ${windowText}.`;
}

export function computeScenario(
  input: SimulatorInput,
  config: StoreConfig = DEFAULT_STORE_CONFIG,
): Scenario {
  const forecasts = generateForecasts(input, config);
  const included = forecasts
    .filter((f) => f.included)
    .map(({ id, rawDaypartForecast, rounded }) => {
      const def = ITEM_DEFS[id];
      // Normalize to per-hour so daypart windows (2h vs 24h) are comparable in scoring.
      const hourlyDemand = rawDaypartForecast / def.holdTimeHr;
      const urgency = 1 / Math.max(0.25, def.holdTimeHr);
      const demandDensity = hourlyDemand / def.lcu;
      const wastePenalty = 1 + def.lcu / Math.max(1, hourlyDemand);
      const score = urgency * demandDensity * wastePenalty;
      return {
        id,
        def,
        score,
        item: {
          id,
          name: def.name,
          emoji: def.emoji,
          recommended: rounded, // already LCU-aligned from allocation engine
          lcu: def.lcu,
          cookTimeMin: def.cookTimeMin,
          holdTimeHr: def.holdTimeHr,
        } as CookItem,
      };
    });

  included.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    return a.def.holdTimeHr - b.def.holdTimeHr;
  });

  const time = formatHour(input.hour);
  const items = included.map((x) => x.item);
  const top = items[0];

  return {
    id: `sim-${Date.now()}`,
    storeId:
      input.storeType === "Urban" ? "#1234" : input.storeType === "Suburban" ? "#7788" : "#0421",
    storeType: input.storeType,
    dayLabel: input.dayLabel,
    time,
    reason: top
      ? reasonFor(top, input.hour)
      : "Forecast is quiet across all items at this hour — nothing to cook yet.",
    items,
  };
}

// ---------- Default starting scenario (Friday lunch, urban) ----------
export const DEFAULT_SIM_INPUT: SimulatorInput = {
  storeType: "Urban",
  dayLabel: "Friday",
  hour: 12,
  taquitosOn: false,
};

export const DEFAULT_SCENARIO: Scenario = computeScenario(DEFAULT_SIM_INPUT);

