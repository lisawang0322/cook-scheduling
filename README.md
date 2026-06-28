# Hot Food Cook Sequencing System

An AI system that helps store associates decide which hot food item to cook first during overlapping daypart windows. Built across Weeks 1–9 using a three-tier model stack (rule-based → ML → LLM benchmark) with a reproducible evaluation harness.

> **Sprint 1 Status (Week 9) + Jun 2026 updates:** All deliverables complete. **Selection metric (Tier 1 holdout):** v1 58.6% • v2.2 **68.9%** • v2.3 66.0% • v3 66.2% (730-scenario temporal holdout — authoritative). **Holdout head-to-head eval** (197 examples: 150 modal from ML holdout + 47 guardrails, Jun 27 2026): on modal slice formula top-1 — v2.2 **67.3%** • v2.3 64.7% • v1 56.0% • LLM v0.3 **26.7%** (hypothesis confirmed: idealized human judgment underperforms ML). **JTBD v0.3 DIAGNOSTIC eval** (110 examples, 2–5 items, Jun 26 2026): LLM **95.8%** JTBD / **78.9%** formula (native) → **90.4%** formula (fair `--input-mode=features`) ≈ v1 **92.6%** • v2.2 **91.6%** — *diagnostic only; confounded by small-item prose set.* See [`EVAL_METHODOLOGY.md`](EVAL_METHODOLOGY.md). Fair-and-Robust harness: bootstrap CIs, McNemar, multi-seed floor, `--eval-set=holdout`, hardened LLM parsing.

---

## TABLE OF CONTENTS

1. [Problem Statement](#problem-statement)
2. [Solution Overview](#solution-overview)
3. [Assumptions](#assumptions)
4. [Project Timeline](#project-timeline)
5. [Architecture](#architecture)
6. [Data Structure](#data-structure)
7. [Success Criteria](#success-criteria)
8. [Key Risks & Mitigations](#key-risks--mitigations)
9. [Cost & ROI](#cost--roi)
10. [Getting Started](#getting-started)
11. [Project Structure](#project-structure)

---

## PROBLEM STATEMENT

### The Customer

New-hire store associate (<6 months on job) at a convenience store with an active hot food program. Age 18–55, high school education, English may not be first language, minimal food-service training. Cooks solo while simultaneously handling the register, restocking shelves, and serving customers.

### The Moment

It is 11:47 AM during lunch peak (11 AM–1 PM). Food Planner shows five items due in overlapping 15-minute windows — pizza (5-min cook), wings (12-min cook), beef mini tacos (6-min cook), waffle tots (10-min cook), and hot dogs — with one oven available. A customer is waiting at the register. The associate has roughly 30 seconds to decide which item to load first before the next task pulls them away. Food Planner tells them what to cook and when, but not which to load first given oven constraints, and not what to deprioritize if they are already running behind.

### The Job-to-be-Done

When I'm a store associate during lunch peak (11 AM–1 PM) with 5 items due in overlapping 15-minute windows, one oven, and customers waiting, I want to know which item to cook first — accounting for cook time and daypart close time — so I can make one decision and start cooking without guessing, hit every daypart on time (no stockouts, no angry customers), and avoid waste (no items expiring in the warmer). But first I need to trust the ranking enough to follow it instead of defaulting to habit, and the system needs to explain why in plain language, not a score.

I also want to stop feeling like I'm constantly catching up and falling behind during peak hours, so I can stop worrying about my job security — but first I need to complete the tasks in front of me at least 50% faster.

And when I execute a ranked cook sequence and hit every daypart on time, I want my shift manager to see that I'm handling tasks with speed and competence, so I can prove I'm reliable — but first I need the system to explain its reasoning so I can defend my decisions if questioned, and my performance needs to be visible to the manager.

### The Current Pain

Associates guess which item to cook first based on habit or memory. This leads to:
- **Stockouts:** Some dayparts miss minimum presentation time; customers leave empty-handed
- **Waste:** Items expire in the warmer; food is discarded
- **Stress:** Associates feel behind because they're constantly guessing
- **Inefficiency:** No systematic way to optimize cook order

---

## SOLUTION OVERVIEW

### What We're Building

A ranking system (v1 rule-based, v2 ML-trained) that tells associates **which item to cook first** when multiple items have overlapping daypart windows.

**Associate sees:**
```
🔴 Cook NOW: Wings (Batch 1 of 4)
   Hold time: 1 hour | Cook time: 12 min
   Ready: 6:00 AM | Expires: 7:00 AM
   
🟡 NEXT (at 6:48 AM): Wings (Batch 2 of 4)
   Hold time: 1 hour | Cook time: 12 min
   Ready: 7:00 AM | Expires: 8:00 AM
   
🟢 THEN (at 7:08 AM): Wings (Batch 3 of 4)
   Hold time: 1 hour | Cook time: 12 min
   Ready: 7:20 AM | Expires: 8:20 AM
```

The system generates this schedule by:
1. Taking Food Planner's forecast (quantities, daypart windows, hold times)
2. Calculating how many cooks of each item are needed to cover the daypart
3. Determining the optimal cook order to minimize waste and missed dayparts
4. Explaining the recommendation in plain language ("Cook wings now because they expire soonest")

### Three-Tier Model Stack

| Tier | Model | Role | 28-item holdout | Holdout eval (modal 150) | 5-item shared eval |
|---|---|---|---|---|---|
| Floor | `AssociateBaseline` | Simulates realistic (flawed) associate behavior | 8.9% | 14.0% | 52.4% |
| ML v1 | Rule-based heuristic | Urgency × demand_density × waste_penalty | 58.6% | 56.0% | 78.6% |
| ML v2.2 | Pairwise GBM (temporal split) | Learned from labeled historical data | **68.9%** | **67.3%** | **81.0%** |
| ML v2.3 | Symmetric pairwise + proba agg | v2.2 variant: balanced pairs, no sample weights | 66.0% | 64.7% | — |
| ML v3 | LightGBM LambdaRank (listwise) | Optimises NDCG directly; group = scenario | **66.2%** top-1 / NDCG@1 0.723 | — | — |
| Ceiling | LLM v0.3 (claude-sonnet-4-6) | Idealized associate judgment proxy, plain-language | — | **26.7%** | 64.0% (v0.2) |

*730-scenario holdout = temporal split ≥ 2025-05-01 (authoritative ML selection metric). **Holdout eval modal 150** = stratified sample from that holdout + 47 v0.3 guardrails (197 total); compare formula top-1 on modal slice only. 5-item shared eval = Sprint 1 50-example set. App/API still serve v2.2.*

**Why predictive ML as primary model:** This is a classification task with deterministic outputs — correct rank order given a scenario. Predictive ML is narrow, well-scoped, and measured by accuracy metrics. It fits this constraint-satisfaction problem without hallucination risk.

**Why LLM as benchmark (not primary model):** The LLM serves as an *idealized human-judgment proxy* — testing whether reasoning like an experienced associate converges on formula-optimal rankings at scale. On the **150-example modal holdout** (19–28 items, plain-language), LLM v0.3 reaches **26.7%** formula top-1 vs v2.2 **67.3%** — confirming that even a strong LLM underperforms learned ML on the real selection task. The smaller v0.3 diagnostic set (2–5 items) is useful for guardrail/refusal testing but confounded for selection (prose input, easy item count).

### Agency Level: Augmentation

The system **recommends** a cook sequence; the associate **executes**; **full override at all times**. No autonomous action is taken. Food safety decisions cannot be delegated, and trust is earned incrementally through consistently correct recommendations.

---

## ASSUMPTIONS

This section documents every assumption baked into the synthetic data generator, validator, and overall project design. These should be revisited when real data becomes available.

### Data Generation Assumptions

**Item Properties (hardcoded in `SyntheticDataGenerator.ITEM_PROPERTIES`):**

28 items are modeled across three hold-time / equipment groups:

| Group | Hold Time | Equipment | Items (ID: LCU) |
|-------|-----------|-----------|------------------|
| Wings & chicken | 2 hr | Oven | wings_bone_in:5, wings_boneless:8, chicken_strip:3, chicken_bite:10, quesadilla:5, chicken_sandwich:1 |
| Sides | 2 hr | Oven | potato_wedge:10, waffle_tot:10, hash_brown:2 |
| Handheld & ethnic | 2 hr | Oven | empanada:2, chimichanga:2, jamaican_turnover:2, jamaican_patty:1†, pupusa:2 |
| Bakery & breakfast | 2 hr | Oven | garlic_knot:2, kolache:2, breakfast_sandwich:1, pizza_slice:6, pizza_stuffed:2 |
| Slow oven | 4 hr | Oven | beef_mini_taco:8, croissant:1, sweet_croissant:6, danish:6 |
| Roller grill | 4 hr | Roller grill | hot_dog:2, sausage:2, taquito:2, buffalo_roller:2†, corn_dog:2 |

† = non-exact multiples (all other items use exact multiples of their LCU)

- **28 item types** are modeled. Baked goods (24h hold) removed from scope.
- Hold times and LCUs are domain-expert estimates, not from actual retailer operational data.
- All data is **entirely synthetic** — no real store data is used.

**Cooking Equipment:**
- **Oven:** 23 items (wings, chicken, sides, ethnic handhelds, bakery/breakfast, pizza, slow-oven pastries). These compete for oven time.
- **Roller grill:** 5 items (hot_dog, sausage, taquito, buffalo_roller, corn_dog). Operate independently of the oven.
- Since the equipment is separate, **grill items can be cooked in parallel** with any oven item. The scheduling constraint only applies among oven items.
- Oven capacity is assumed but not explicitly modeled (no upper bound constraint in the prototype).
- Associate lead time for switching between equipment is negligible in this model.

**Lowest Cookable Unit (LCU) & Exact Multiples — explained:**
- LCU is the minimum number of units that must go into the cooking equipment together.
- "Exact multiples" means the equipment must receive exactly LCU × N units (e.g., pizza must be 6, 12, 18…). You cannot cook 7 slices of pizza.
- When exact multiples are required, the cooked quantity is rounded **up** to the nearest multiple of LCU that meets or exceeds demand.
- This applies to **both initial cooks and restock cooks** — every cook event must produce a valid LCU quantity.

**Mid-Window Sell-Throughs & Restocks:**
- ~15% of windows experience a sell-through where actual demand exceeds forecast.
- When a sell-through occurs, the associate performs a **restock cook** mid-window.
- The restock quantity is also LCU-valid (e.g., if 4 more pizzas are needed, 6 are cooked because LCU=6).
- Restock cooks happen in the middle third of the window (1/3 to 2/3 through).
- Each cook event is tagged as `cook_type: "initial"` or `cook_type: "restock"`.
- This produces ~13–15% more cook events beyond the base 19,980 initial cooks.

**Store Types & Demand:**

| Store Type | Demand Multiplier | Rationale |
|------------|-------------------|-----------|
| Urban | 1.4× | Higher foot traffic, denser population |
| Suburban | 1.0× (baseline) | Average traffic |
| Highway | 0.7× | Lower, more sporadic traffic |

- Exactly 3 store types are modeled; real variation is continuous, not categorical.
- Demand multipliers are estimates. Actual urban/highway ratios may differ.
- Weekend demand is assumed to be **30% higher** than weekdays (uniform across all items and store types).
- Per-event demand noise is **±20%** (uniform random, `rng.uniform(0.8, 1.2)`).

**Forecast Windows (item-specific dayparts):**

Forecasts are generated on a **24-hour cycle from 6 AM to 6 AM** (the store operates 24/7). Each item’s forecast window length equals its hold time:

| Item | Hold Time | Windows per Day | First Window |
|------|-----------|-----------------|---------------|
| Pizza | 2 hours | 12 | 06:00–08:00 |
| Wings (2h) | 2 hours | 12 | 06:00–08:00 |
| Wings (4h) | 4 hours | 6 | 06:00–10:00 |
| Taquitos | 4 hours | 6 | 06:00–10:00 |
| Baked goods | 24 hours | 1 | 06:00–06:00 |

- **One cook event per item per window.** The forecast for a window IS the cooked quantity (no separate “cooks needed” multiplier).
- Windows are contiguous and non-overlapping. A 4-hour item’s windows are: 06–10, 10–14, 14–18, 18–22, 22–02, 02–06.
- No seasonal variation is modeled (e.g., summer vs. winter patterns).
- Demand varies by time-of-day via a curve (peak at 11am–2pm and 5pm–9pm, low overnight).

**Cook Quantity Calculation:**
- The prototype receives **whole-unit forecasts** (already rounded from the fractional hourly forecast provided by the upstream API). The rounding logic itself is out of scope.
- For exact-multiple items: `forecast = ceil(raw / LCU) × LCU` (e.g., raw demand of 7 pizza → forecast of 12)
- For non-exact-multiple items: `forecast = max(LCU, raw)`
- **One cook per window:** `cooked_qty = forecast_demand` (the forecast is what gets cooked)
- Sold quantity = `min(cooked_qty, demand ± 2)`, clamped ≥ 0.

**Timestamps:**
- Cook timestamps are within the first 15 minutes of each window (associate cooks at window start to have food ready).
- POS sale timestamps are Gaussian-distributed around the window midpoint (σ = window_length/4).
- Write-off timestamps are at end-of-window + 30–120 min delay (associate logs disposal after expiry).
- **Timing delays in write-off logging are expected behavior, not errors.** The validator does not penalize them.

**Granularity:**
- One cook event per item per window per store type per day.
- Window counts vary by item: 2h-hold items have 12 windows/day, 4h-hold have 6, 24h-hold have 1.
- Base: 180 days × 3 stores × (12+12+6+6+1 windows) = **19,980 initial cooks**.
- Plus ~15% sell-through restocks = **~23,000 total cook events**.
- Store IDs are randomly generated each day — they do **not** represent persistent stores tracked across days.

### Write-Off Quality Assumptions

Quality issues are injected at generation time with these probabilities:

| Quality Type | Probability | Behavior |
|--------------|-------------|----------|
| Accurate | 60% | Logged write-off = inferred write-off |
| Gap | 15% | No write-off logged at all (record missing) |
| Counting error | 20% | Logged write-off off by ±1 or ±2 units |
| Major discrepancy | 5% | Logged write-off off by ±3 to ±5 units |

- These are the **generation-side** probabilities. The **validator** independently classifies confidence based on the actual numerical difference, which produces different final percentages (see below).
- Counting errors use `rng.choice([-2, -1, 1, 2])` — the error is never zero.
- Major discrepancies use `rng.choice([-5, -4, -3, 3, 4, 5])`.
- Logged write-offs are clamped to ≥ 0 (cannot be negative).

### Validation Assumptions

**Confidence Classification (independent of generator labels):**

| Confidence | Rule | Maps to |
|------------|------|---------|
| High | \|logged − inferred\| ≤ 1 | Accurate logging |
| Medium | \|logged − inferred\| ≤ 2 **OR** gap (no log) | Counting errors or missing data |
| Low | \|logged − inferred\| ≥ 3 | Major discrepancies |

- **Important nuance:** The validator classifies confidence from the actual difference, not the generator's `quality_type` label. This means some generator "counting errors" (±1) are classified as **high** confidence by the validator, since ±1 is within the high-confidence threshold. This is why observed high confidence (~66%) exceeds the generator's 60% accurate rate. Restock events tend to have larger write-offs (due to LCU over-cooking), which increases low-confidence counts.
- Inferred write-off = `max(0, cooked_qty − sold_qty)`, using POS sales aggregated by `cook_event_id`.
- Gaps (missing write-off logs) are always classified as **medium** confidence, never low.
- The validator does **not** account for timing — only quantity differences matter.

### Observed vs. Target Distribution

| Metric | Design Target | Actual (28-item retrain) | Explanation |
|--------|---------------|--------------------------|-------------|
| Total cook events | — | **175,051** | 28-item set, 180 days, 3 store types |
| POS sales records | — | **766,229** | Linked to cook events by quantity sold |
| Write-off log entries | — | **148,422** | 175,051 minus ~15% gaps |
| High confidence | ~60% | **70.5%** | More diverse items create clearer quantity differentials |
| Medium confidence | ~30% | **23.2%** | Gaps + counting errors |
| Low confidence | ~10% | **6.2%** | Lower than Sprint 1 due to more diverse item mix |
| Usable for training | ~90% | **93.8%** | High + medium confidence combined |

### General Project Assumptions

- **No real data:** All data is synthetic. Model performance on synthetic data does not guarantee production performance.
- **Forecast rounding is out of scope:** The upstream API provides fractional hourly forecasts; a separate rounding module converts these to whole units. This prototype only operates on the **post-rounding whole-unit forecast**. The rounding logic is not implemented here.
- **Single oven constraint** is assumed but not yet modeled in the scheduler.
- **Store operates 24/7.** Forecasts run on a 24-hour cycle starting at 6 AM (6 AM Day 1 → 6 AM Day 2).
- **No seasonal effects:** Demand patterns are static across the 180-day period (Jan 1 – Jun 29, 2025).
- **No store-level learning:** Each store type behaves identically (same base demand, same quality distribution).
- **Reproducibility:** All random generation uses `seed=42` via `random.Random(seed)` for deterministic output.
- **Python stdlib only:** The current implementation uses no external dependencies (json, os, random, uuid, datetime, collections). Future weeks (v2 ML) will require scikit-learn, pandas, numpy.
- **Write-off timing delays (30–120 min) are expected** and should not be treated as data quality issues. They reflect realistic associate logging behavior.
- **Minimum presentation limits** exist per item (minimum units that must be displayed at all times) but are not yet modeled in the prototype. These will factor into the v1/v2 scheduler logic.

---

## PROJECT TIMELINE

### WEEK 1–2: Synthetic Data Generation & Validation (Setup Phase)

**Objective:** Create 6 months of realistic synthetic store data and assess data quality.

**Tasks:**
1. Define item properties (hold times: 2, 4, 24 hours; LCU constraints; demand patterns)
2. Generate item-specific forecast windows (window length = hold time, 6 AM to 6 AM cycle)
3. For each item/window/store/day combination:
   - Generate whole-unit forecast demand (already LCU-valid, based on store type, time-of-day, day of week, ±20% noise)
   - Cooked quantity = forecast (one cook per window)
   - Simulate POS sales (Gaussian across window)
   - Generate write-off logs with realistic quality issues (60/15/20/5 distribution)

**Deliverables:**
- `data/cook_logs.json` — 175,051 cook events (28-item set, 180 days)
- `data/pos_sales.json` — 766,229 individual sale records linked to cook events
- `data/write_off_logs.json` — 148,422 write-off entries
- `output/quality_report.json` — Data quality assessment with confidence breakdown

**Observed Results (28-item retrain):**
- ✅ 70.5% high-confidence data (inferred ≈ logged, ±1 unit)
- ✅ 23.2% medium-confidence data (gaps or ±2 counting errors)
- ✅ 6.2% low-confidence data (major discrepancies, ±3+ units)
- ✅ 93.8% usable for training (high + medium combined)
- ✅ Quality breakdown by store type: highway 71.5% high, suburban 70.5%, urban 69.7%

---

### WEEK 3–4: v1 Rule-Based System (Simple Ranking)

**Objective:** Build a deterministic rule-based ranking that associates can understand and trust.

**Heuristic:** Earliest-Deadline-First with demand-weighted priority.

At each decision point (oven is free), the associate needs to decide: *"Which item do I cook next?"* The v1 rule scores each item and cooks the highest-scoring one first.

**Priority Score:**
```
score(item, t) = urgency × demand_density × waste_penalty

Where:
  urgency         = 1 / time_until_window_ends     (hours remaining; lower = more urgent)
  demand_density  = forecast_demand / LCU          (how many batches needed; higher = more work)
  waste_penalty   = 1 + (LCU / forecast_demand)    (higher when LCU is large relative to demand,
                                                     meaning over-cooking is likely → cook sooner
                                                     to avoid compounding waste)
```

**Intuition:**
- Items whose window is about to end get high urgency (you'll miss the window if you wait).
- Items with high demand relative to their batch size need more oven time total.
- Items with large LCU (e.g., wings_4h at 8 per batch) carry more waste risk per cook, so it's better to cook them earlier when demand is more certain.

**Tiebreaker:** Shorter hold time → higher priority (perishable items can't recover from a missed cook).

**Note:** Taquitos use the **roller grill** (not the oven), so they cook in parallel with any oven item. The v1 priority score only ranks oven items (pizza, wings, baked goods) against each other. Taquitos are always scheduled immediately on the grill when their window starts.

**Example scenario** (urban store, 10:15 AM, oven items only):
| Item | Window Ends | Urgency | Demand | LCU | Demand Density | Score |
|------|-------------|---------|--------|-----|----------------|-------|
| Pizza | 12:00 (1.75h) | 0.57 | 12 | 6 | 2.0 | 1.71 |
| Wings 2h | 12:00 (1.75h) | 0.57 | 10 | 5 | 2.0 | 1.71 |
| Wings 4h | 14:00 (3.75h) | 0.27 | 16 | 8 | 2.0 | 0.80 |

→ **Cook pizza or wings_2h first** (highest score, tiebreak by hold time). Meanwhile, taquitos cook on the roller grill in parallel.

**Tasks:**
1. Implement `src/cook_scheduler.py` with v1 priority scoring
2. Generate template-based explanations (no LLM): e.g., *"Cook pizza now: window ends in 1.75h, demand is 12 slices (2 batches of 6)"*
3. Test on all ~23,000 scenarios
4. Measure: does v1 ranking minimize simulated write-offs?

**Deliverables:**
- `src/cook_scheduler.py` — v1 priority-score ranking logic
- Eval report: v1 ranking vs. actual write-off outcomes

**Success Criteria:**
- ✅ v1 works on >90% of high-confidence scenarios (produces a valid, non-degenerate ranking)
- ✅ Explanations are clear and actionable
- ✅ Baseline established for v2 comparison

---

### WEEK 5–6: Data Labeling & Outcome Simulation (Training Data Prep)

**Objective:** Prepare labeled training data for v2 by calculating what the optimal cook order should have been.

**What is a "decision point"?**
A decision point is a moment where an associate must choose which oven item to cook first. It occurs when 2+ oven items share the same window start hour at the same store on the same day. Taquitos (roller grill) are excluded.

**Approach (Hybrid):**
1. Group initial cook events into decision points (same store, date, window_start_hour)
2. Filter: exclude any decision point containing a low-confidence event
3. Score each item using a composite priority function combining observable features AND actual outcomes
4. Label: "For this scenario, the optimal cook order was [item1, item2, item3]"

**Labeling Logic — Composite Priority Score:**

The label for each scenario is determined by scoring each item and sorting highest-first:

```python
priority = (
    urgency * 2.0             # 1/time_remaining — time pressure is most important
    + demand_density * 0.3    # demand/LCU — more batches = more oven time needed
    + hold_penalty * 1.0      # 1/hold_time — shorter hold = more perishable
    - waste_ratio * 1.5       # writeoff/cooked_qty — penalize items that were wasted
)
```

This creates labels that are:
- **Learnable** — urgency, demand density, and hold time are directly observable input features
- **Outcome-informed** — the waste_ratio component adjusts rankings based on what actually happened
- **Deterministic** — same inputs always produce the same label (0 contradictions)

**Labeling Iterations:**

| Attempt | Strategy | v1 Agreement | Issue |
|---------|----------|--------------|-------|
| 1 | Rank by raw write-off ascending, tiebreak by hold time | 29.2% | Most items have 0 write-off; tiebreaker is arbitrary noise |
| 2 | Rank by waste_ratio, tiebreak by time_remaining + demand_density | 65.5% | Tiebreaker aligned too closely with v1 (nothing new to learn) |
| 3 | Composite score: urgency + demand_density + hold_penalty - waste_ratio | 34.2% | Balance of learnable structure and outcome signal |

Attempt 3 was chosen because it creates labels that differ from v1 (34.2% agreement) while being learnable from input features (model achieves 64.2% vs v1's 34.2%).

**Informative vs. Tiebreaker Scenarios:**
- **Informative (73.2%):** Items have different waste ratios — the label carries actual outcome signal
- **Tiebreaker (26.8%):** All items had identical waste ratios — label is purely feature-driven

**Tasks:**
1. Match write-off logs to cook events (by cook_event_id)
2. Calculate confidence for each event (|logged - inferred| threshold)
3. Compute composite priority score incorporating actual outcomes
4. Create labeled dataset: scenario features → optimal cook order

**Deliverables:**
- `data/labeled_training_set.json` — 2,164 labeled decision-point scenarios (28-item set)
- `output/labeling_report.json` — Quality assurance report

**Success Criteria:**
- ✅ 2,164 labeled scenarios ready for training (28-item set; fewer decision points per store due to larger item universe)
- ✅ All labels validated by composite scoring (no contradictions)
- ✅ v1 agreement: 58.6%

---

### WEEK 7: Train v2 ML Model (Supervised Learning)

**Objective:** Build a supervised learning model that learns patterns from historical cook outcomes.

**Model:** RandomForestClassifier (scikit-learn)
- `n_estimators=200`, `max_depth=12`, `class_weight="balanced"`, `random_state=42`
- **Input features (49 total):**
  - Global: decision_hour, is_weekend, day_of_week, store_type, num_oven_items
  - Per item (×4 oven items): forecast_demand, lcu, hold_time, time_remaining, cooked_qty, presence flag
  - Per item derived: urgency (1/time_remaining), demand_density, waste_penalty, v1_score
  - Cross-item: max_demand, min_time_remaining, demand_spread, urgency_spread
- **Output:** Classification — which item to cook first (pizza | wings_2h | wings_4h | baked_goods)

**Feature Matrix Design:**

Decision points have 2–4 items, but the model needs a fixed-width input. Solution: allocate fixed slots for all 4 oven items with a `_present` flag. Missing items get zero-filled features.

**Training Iterations:**

| Iteration | Labeling | Features | CV Accuracy | Notes |
|-----------|----------|----------|-------------|-------|
| 1 | Raw write-off tiebreak | 33 basic features | 42.2% | Labels were noise; model couldn’t learn |
| 2 | Waste ratio + demand density tiebreak | 49 features (added urgency, v1_score) | 43.7% | Still noisy labels; v1 agreement too high (65.5%) |
| 3 | Composite priority score | 49 features | **64.2%** | Labels now learnable + outcome-informed |

**Key insight:** The labeling strategy matters more than feature engineering. When labels are driven by noise (random write-off variance), no amount of features helps. When labels encode a learnable pattern (urgency + hold time + outcome adjustment), the model succeeds.

**Final Results (seed=42):**

```
All scenarios (5,290):
  5-fold CV accuracy: 64.2% ± 1.5%
  Training accuracy:  70.2%

Informative scenarios only (3,870):
  5-fold CV accuracy: 61.2% ± 1.5%
  Training accuracy:  69.3%

v1 vs v2:
  v1 accuracy: 34.2%
  v2 accuracy: 64.2%  (✅ +30.0 pp improvement)
```

**Top Feature Importances:**

| Feature | Importance | Why It Matters |
|---------|------------|----------------|
| baked_goods_waste_penalty | 0.091 | Unique LCU=1 item, very different waste profile |
| baked_goods_cooked_qty | 0.089 | High volume (24h window) = distinctive |
| num_oven_items | 0.082 | 2 vs 3 vs 4 items changes the decision dynamics |
| baked_goods_urgency | 0.073 | 24h window = very low urgency (always ranked last) |
| baked_goods_demand_density | 0.066 | High density (33 units / LCU=1) |

**Confusion Matrix:**
- baked_goods: 384/384 correct (100%) — trivially separable
- pizza: 2,414/3,442 correct (70.1%) — main class
- wings_2h: 916/1,464 correct (62.6%) — confused with pizza (same hold time)
- wings_4h: never labeled as optimal first (longer window = lower priority)

**Why 75% wasn’t reached:**
Pizza and wings_2h share identical time characteristics (2h hold, same window boundaries). When both have 0 waste, the composite score difference is driven only by demand_density (demand/LCU), which varies stochastically. This creates inherent classification noise between these two classes. A production model with real historical data would have richer distinguishing features (actual sales velocity, time-of-day patterns per item).

**Tasks:**
1. ✅ Build fixed-width feature matrix from variable-item decision points
2. ✅ Train RandomForest on all 5,290 scenarios
3. ✅ Stratified 5-fold cross-validation
4. ✅ Extract and analyze feature importance
5. ✅ Save model to `models/v2_ranking_model.pkl`

**Deliverables:**
- `models/v2_ranking_model.pkl` — Trained model
- `output/v2_training_report.json` — Full evaluation (CV scores, confusion matrix, per-class metrics)
- `output/feature_importance.json` — All 49 features ranked by importance
- `src/model_trainer.py` — ModelTrainer class with train/evaluate/predict/save
- `requirements.txt` — numpy, pandas, scikit-learn

**Success Criteria:**
- ✅ v2 outperforms v1 by +30 percentage points (64.2% vs 34.2%)
- ✅ Model learns meaningful patterns (baked_goods features dominate; time/demand features matter)
- ✅ Feature importance aligns with domain knowledge
- ⚠️ CV accuracy 64.2% (below 75% target — explained by pizza/wings_2h class overlap)

---

### TRAINING LOG

Full iteration history documenting each experiment, what was tried, and what was learned.

#### Iteration 1: Multiclass RF + Raw Write-Off Labels

| Dimension | Detail |
|-----------|--------|
| **Labeling** | Rank items by raw write-off ascending; tiebreak by hold_time |
| **Model** | RandomForestClassifier, n=200, depth=12, balanced |
| **Features** | 33 (global + per-item basics) |
| **Result** | CV: 42.2%, Training: 50.2% |
| **Problem** | Most items have 0 write-off → label determined by arbitrary tiebreaker → model can't learn |

**Lesson:** When the signal (write-off difference) is absent in >70% of scenarios, labels become noise.

---

#### Iteration 2: Multiclass RF + Demand-Density Tiebreaker

| Dimension | Detail |
|-----------|--------|
| **Labeling** | Waste ratio primary; tiebreak by time_remaining then demand_density |
| **Model** | RandomForestClassifier, n=200, depth=12, balanced |
| **Features** | 49 (added urgency, demand_density, waste_penalty, v1_score per item) |
| **Result** | CV: 43.7%, Training: 51.1% |
| **Problem** | Tiebreaker aligned too closely with v1 logic (65.5% v1 agreement) — model learns to mimic v1 but can't beat it |

**Lesson:** If labels are ~v1, the model at best reproduces v1. Need labels that diverge from v1 but are still learnable.

---

#### Iteration 3: Multiclass RF + Composite Priority Labels

| Dimension | Detail |
|-----------|--------|
| **Labeling** | Composite score: `urgency×2 + demand_density×0.3 + hold_penalty×1 - waste_ratio×1.5` |
| **Model** | RandomForestClassifier, n=200, depth=12, balanced |
| **Features** | 49 features |
| **Result** | CV: **64.2%**, Training: 70.2%, v1 agreement: 34.2% |
| **Improvement** | +30pp over v1 |
| **Remaining issue** | Pizza vs wings_2h confusion (same 2h hold, same window → nearly identical features) |

**Lesson:** Composite labels that blend observable structure + outcome signal are learnable. But multiclass framing loses pairwise signal.

---

#### Iteration 4 (v2.1): Pairwise GBM + Historical Features ✅

| Dimension | Detail |
|-----------|--------|
| **Labeling** | Same composite priority (convert to pairwise: "A before B?") |
| **Model** | GradientBoostingClassifier, n=300, depth=5, lr=0.1, subsample=0.8 |
| **Features** | 36 per pair (A features, B features, difference features, historical aggregates) |
| **Training samples** | 539,991 pairs (from 2,164 scenarios, 28-item retrain); *Sprint 1: 11,466 pairs from 5,290 scenarios* |
| **Result** | Pairwise CV: **79.5%** ±0.1%, Top-1 ranking: **77.1%** (n=2,164) |
| **Sprint 1 result** | Pairwise CV: 86.8%, Top-1: 76.5% (5-item, no temporal guard) |
| **Improvement** | +18.5pp over v1 (58.6%), +13pp over v2 multiclass |

**Key changes that drove the improvement:**

1. **Pairwise reframing** — "Should A go before B?" is a cleaner learning target than "Which of 28 items goes first?" Each confusing pair gets dedicated training signal.

2. **Historical features** — Computing `avg_writeoff_by_hour[item][hour]` and `avg_writeoff_by_store_type[item][store]` from all cook logs gave the model context that distinguishes otherwise-similar items.

3. **Gradient Boosting** — Sequential tree building captures interaction effects better than Random Forest's averaging. Lower depth (5 vs 12) with more trees (300 vs 200) reduces overfitting.

4. **Difference features** — `diff_hold_time`, `diff_urgency`, `diff_demand_density` directly encode the pairwise comparison the model needs to make.

**Top features learned (28-item retrain):**

```
diff_demand_density      : 0.551  (demand/LCU ratio difference A-B — dominant signal)
diff_urgency             : 0.197  (urgency difference A-B)
diff_hold_time           : 0.051  (hold time difference A-B)
diff_time_remaining      : 0.048  (time remaining difference A-B)
a_demand                 : 0.026  (item A's forecast demand)
```

**Interpretation:** Across 28 heterogeneous items, demand density difference is the strongest pairwise signal (+0.354pp over Sprint 1 where historical write-off features dominated). With 28 items, demand/LCU ratios vary much more widely than in the 5-item set, making the instantaneous features more discriminative.

---

#### Iteration 5 (v2.2): Temporal Split + Soft Labels

| Dimension | Detail |
|-----------|--------|
| **Labeling** | Same composite priority; add sample weights based on rank gap + waste difference |
| **Model** | GradientBoostingClassifier, same hyperparams as v2.1 |
| **Training samples** | 357,822 pairs (train partition only; 1,434 scenarios pre-cutoff) |
| **Split** | Temporal: cutoff 2025-05-01; train 1,434 scenarios, test 730 scenarios |
| **Historical features** | Computed ONLY from training period (prevents data leakage) |
| **Weights** | High-confidence pairs (clear demand/urgency gap): weight 1.0; near-tied pairs (same urgency, both 0 waste): weight 0.33 (min 0.3); mean weight 0.523; 52.8% of pairs down-weighted |
| **Result** | Pairwise CV: **79.6%** ±0.1%, Training acc: 79.2%, Honest test: **68.9%**, Full top-1: **72.2%** |
| **Sprint 1 result** | CV 85.4%, Honest test 74.3% (5-item set, 1,747 test scenarios) |
| **vs v2.1** | -8.2pp honest test (the "honesty + scale tax" — v2.1 had no temporal guard; 28-item problem is harder than 5-item) |

**Top features learned (28-item retrain):**

```
diff_demand_density      : 0.450  (demand/LCU ratio difference — dominant)
diff_urgency             : 0.130  (urgency difference A-B)
diff_demand              : 0.088  (raw demand difference)
diff_hold_time           : 0.081  (hold time difference)
a_cooked_qty             : 0.041  (already-cooked quantity for item A)
diff_hist_wo_overall     : 0.018  (historical write-off diff — present but lower than Sprint 1)
```

**Lesson:** The temporal split reveals the model's true generalization ability. The larger drop vs Sprint 1 (-5.4pp: 74.3% → 68.9%) reflects both the honesty tax and the harder 28-item problem (more item confusion, lower base rate). The +60pp gap over associate proves strong lift regardless.

---

#### Iteration 6: 28-Item Expansion + Associate Baseline

| Dimension | Detail |
|-----------|--------|
| **Scope** | Expanded from 5 items to 28 items; 175K cook events, 3 store types, 180 days |
| **Data generation** | Item-specific time-of-day demand curves + item-specific waste propensity by store type |
| **Morning items** | `breakfast_sandwich`, `hash_brown`, `kolache`, `waffle_tot` peak 6–10 AM |
| **Lunch/dinner items** | `chicken_sandwich`, `wings_bone_in`, `wings_boneless`, `quesadilla` peak 11 AM–8 PM |
| **All-day items** | `pizza_slice`, `pizza_stuffed`, `beef_mini_taco`, `empanada`, `jamaican_patty` etc. distributed across dayparts |
| **Waste patterns** | Urban stores: higher wing write-offs; highway stores: higher baked-goods write-offs |
| **Associate baseline** | `AssociateBaseline` class simulating realistic (flawed) associate decision-making |

**Associate Baseline — Observed Behavior Model:**

Associates don't use a formula. Based on store observations, their decision process is:
- **40% expiration-driven:** Grab whatever is closest to expiring (shortest hold time). "The frozen pizza is about to go bad, cook that first."
- **30% habit/familiarity:** Default to pizza because it's the most common item, always visible, fastest to prepare mentally.
- **20% random/convenience:** Whatever is physically closest in the freezer. No decision logic at all.
- **10% demand-checking:** Occasionally glance at the Food Planner forecast and pick the highest-demand item.

This produces **52.4% accuracy** on the 50-example shared eval — better than pure random (~33%) but far below what's achievable with systematic decision-making.

**Final results with enriched data (28-item set, retrained Jun 2026):**

```
Associate baseline (current state):         8.9%  (28 items; habit/random ≈ 1/28 base rate)
v1 (rule-based heuristic):                 58.6%
v2.1 (pairwise GBM, no temporal split):   77.1%  (top-1, 539,991 pairs)
v2.2 (pairwise + temporal + weights):      68.9%  (honest temporal test, 730 holdout scenarios)
```

**Why associate accuracy dropped to 8.9%:** With 28 competing items, habit/random behavior converges near random chance (~1/28 = 3.6% base rate). The urgency-based habit picking provides only a small lift above floor.

**Why v2.2 beats v1 by +10.3pp:** The pairwise GBM learns *item × hour × store_type* interaction patterns and historical write-off signals that the static urgency × density formula cannot capture across 28 heterogeneous items.

---

#### Iteration 7 (v3): LightGBM LambdaRank — Listwise NDCG Optimisation

| Dimension | Detail |
|-----------|--------|
| **Model** | `LGBMRanker(objective="lambdarank", metric="ndcg")` |
| **Structure** | One row per (scenario, item) — no C(n,2) pairwise expansion |
| **Group** | `group = scenario` (contiguous item rows per query group) |
| **Relevance** | Graded reverse-rank: `rel = (n_items − 1) − position_in_optimal_order`; linear `label_gain=[0..27]` |
| **Loss** | LambdaRank: optimises NDCG directly instead of approximating ranking via independent binary classifications |
| **Inference** | `model.predict()` scores per item → sort descending; no win-counting or proba accumulation |
| **Split** | Same temporal split as v2.2: cutoff 2025-05-01; train 1,434 scenarios, test 730 scenarios |
| **Historical features** | Train-period only (no leakage), same as v2.2 |
| **Result** | Test NDCG@1 **0.723** / @3 **0.753** / @5 **0.797**; Top-1 **66.2%** (730-scenario holdout); full top-1 **71.6%** (2,164 scenarios) |
| **App/API status** | Still serving v2.2 — swap `load_model()` in `app/utils.py` when v3 is validated |

---

#### Iteration 8 (v2.3): Symmetric Pairwise + Probability Aggregation

| Dimension | Detail |
|-----------|--------|
| **Changes vs v2.2** | Symmetric pair augmentation (removes item-position bias); probability-based rank aggregation (avoids intransitive win-count cycles); no sample weights (edge/near-tie pairs weighted equally) |
| **Split** | Same temporal split as v2.2 (cutoff 2025-05-01) |
| **Result** | Test top-1 **66.0%** (730 holdout, −2.9pp vs v2.2); holdout eval modal slice **64.7%** |
| **Status** | Experimental — v2.2 remains selection winner |

---

#### Summary Table

| Version | Approach | Pairwise CV | Honest Test | vs Associate |
|---------|----------|-------------|-------------|--------------|
| Associate | Mix of expiration/habit/random/demand | — | 8.9% | baseline |
| v1 | Rule-based (urgency × density × penalty) | — | 58.6% | +49.7 |
| v2.1 | Pairwise GBM + historical (no temporal split) | 79.5% | 77.1% | +68.2 |
| **v2.2** | **Pairwise GBM + temporal + soft labels** | **79.6%** | **68.9%** | **+60.0** |
| v2.3 | Symmetric pairwise + proba agg (no weights) | 79.5% | 66.0% | — |
| v3 | LightGBM LambdaRank (listwise, NDCG) | — | **66.2%** top-1 / NDCG@1 0.723 | — |

**Note:** v2.1's 77.1% is slightly inflated (historical features used test-period data). v2.2's 68.9% is the honest metric with no data leakage. v3 achieves 66.2% top-1 on the same 730-scenario holdout (−2.7pp vs v2.2) but optimises NDCG directly (NDCG@1 0.723 / @3 0.753 / @5 0.797) with 11× fewer training rows and no win-counting. Full metrics in `output/v3_lambdarank_report.json`.

---

### WEEK 8–9: Demo, LLM Benchmark & Evaluation ✅

#### Streamlit Demo

**How to Run:**
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

**App Pages:**

| Page | What It Shows |
|------|---------------|
| 📊 Scenario Comparison | Pick a scenario → see Associate vs v1 vs v2.2 side-by-side with plain-language explanations |
| 📈 Impact Dashboard | Aggregate KPIs: accuracy by store/hour, projected waste reduction |
| 🎛️ What-If Simulator | Adjust store type, hour, demand → see recommendation change in real-time |

#### LLM Benchmark (Sprint 1 Eval Plan)

**Setup:**
```bash
export ANTHROPIC_API_KEY=your_key
python notebooks/week9_llm_eval_runner.py --prompt-version=v0.1
python notebooks/week9_llm_eval_runner.py --prompt-version=v0.2
python scripts/compare_llm_versions.py v0.1 v0.2
```

**Results:**

| Comparator | Overall Top-1 | Ranking | Refusal (OOS+adv) | Kendall τ |
|---|---|---|---|---|
| associate_floor | 52.4% | 52.4% | n/a | 0.436 |
| v1_rules | 78.6% | 78.6% | n/a | 0.730 |
| v2_2_ml | **81.0%** | **81.0%** | n/a | **0.762** |
| llm_v0.1_zero_shot | 50.0% | 45.2% | 75.0% | 0.381 |
| **llm_v0.2_zero_shot** | **64.0%** | **61.9%** | **75.0%** | **0.476** |

**Category breakdown (v0.2):**
| Category | v0.1 | v0.2 | Δ |
|---|---|---|---|
| modal (30) | 46.7% | 60.0% | +13.3pp |
| edge (12) | 41.7% | 66.7% | +25.0pp |
| OOS (5) | 100.0% | 100.0% | 0 |
| adversarial (3) | 33.3% | 33.3% | 0 |
| divergence (n=4) | 0.0% | 0.0% | open question |

**Open question — divergence cases:** The LLM consistently ranks high-demand baked_goods above pizza/wings even when baked_goods has a 23hr window. This may reflect genuine associate intuition that diverges from the formula. Requires field validation before treating as a prompt failure.

**Eval harness features (Fair-and-Robust update Jun 2026):**
- `--eval-set=v0.1|v0.2|v0.3|holdout` selects eval set; `--prompt-version=vX.Y` selects prompt
- **`--eval-set=holdout`** — 197 examples (150 modal from ML temporal holdout + 47 v0.3 guardrails); use **modal slice formula top-1** for head-to-head ML vs LLM selection comparison
- `--assoc-seeds=N` (default 20): multi-seed associate floor → reports mean ± std (not single draw)
- `--llm-samples=k` (default 1): repeated LLM runs → mean ± std, parse-failure rate (cost-gated)
- `--input-mode=native|features|prose`: controls LLM input fairness; `features` mode feeds numeric table (removes prose advantage)
- **LLM parse hardening (Jun 27 2026):** `max_tokens=1024`, robust JSON extraction/repair, deterministic retries, v1 fallback — 0 parse failures on latest holdout run
- **Bootstrap 95% CIs** (`bootstrap_ci`, n=2,000, seed=42) on cook-now, formula top-1, set-recall, MPV-rate
- **McNemar paired significance matrix** between all comparators (same examples → paired test appropriate)
- **Scale stratification** (`item_count_band`: small 2-5 / medium 6-12 / large 13-28) — compare within band
- **Holdout-clean slice** (`holdout_clean` flag, cutoff 2025-05-01) — v2.2 selection metrics only on clean examples
- **Selection scorecard** (`selection_scorecard` block): primary metric + CI, guardrail pass/fail, recommendation
- v0.3 routing, JTBD metrics, dual-label breakdown: unchanged
- `--no-llm` flag for dry-run against ML baselines only; preserves existing LLM preds when re-running ML only

**Holdout head-to-head eval (Jun 27 2026, `--eval-set=holdout --prompt-version=v0.3`):**

| Comparator | Modal formula top-1 (n=150) | OOS refusal | MPV | Parse failures |
|---|---|---|---|---|
| associate_floor | 14.0% | — | 15 | 0 |
| v1_rules | 56.0% | — | 23 | 0 |
| v2_2_ml | **67.3%** | — | 11 | 0 |
| v2_3_ml | 64.7% | — | 12 | 0 |
| llm_v0.3_zero_shot | **26.7%** | 93.3% (14/15) | 1 | **0** |

**Interpretation:** LLM underperforms v2.2 by **−40.6pp** on the modal holdout slice — supporting the hypothesis that idealized human judgment (LLM proxy) does not beat learned ML at production scale (19–28 items). LLM passes MPV/refusal guardrails but fails latency (~11s vs 3s budget). Reports: `output/llm_eval_v0.3_holdout_report.json`.

**Deliverables:**
- `prompts/v0.1_system_prompt.md`, `prompts/v0.2_system_prompt.md`, `prompts/v0.3_system_prompt.md`
- `data/llm_eval_set_v0.1.json` — 50 examples (modal 30 / edge 12 / OOS 5 / adv 3)
- `data/llm_eval_set_holdout.json` — 197 examples (150 modal holdout + 47 guardrails); built by `scripts/build_eval_set_holdout.py`
- `output/llm_eval_v0.1_report.json`, `output/llm_eval_v0.2_report.json`, `output/llm_eval_v0.3_holdout_report.json`
- `output/llm_eval_version_comparison.json`
- `SPRINT1_SUMMARY.md`

---

### WEEK 10–11: Evaluation & Documentation (Rigor Phase)

**Objective:** Rigorously test v2 on unseen data and document all limitations.

**Evaluation Plan:**
1. **Test on holdout data:** 20% of labeled scenarios, unseen during training
2. **Measure accuracy:** Top-1, Top-2, Top-3 (does model predict correct item in top N?)
3. **Measure by store type:** Does v2 perform equally on urban/highway/suburban?
4. **Measure by confidence:** Does v2 perform better on high-confidence data?
5. **Failure analysis:** Where does v2 fail? Why?

**Tasks:**
1. Build evaluation harness
2. Run all tests
3. Generate detailed eval report
4. Document assumptions and limitations
5. Write README + architecture documentation

**Deliverables:**
- Eval report with tables, charts, and breakdown
- Failure analysis (top 10 failure modes)
- Architecture documentation
- Assumptions & limitations document

**Success Criteria:**
- ✅ Test accuracy ≥ 75%
- ✅ Performance breakdown by store type documented
- ✅ Honest assessment of what works and what doesn't
- ✅ Clear documentation of synthetic data limitations

---

### WEEK 12: Final Presentation & Validation (Demo & Panel)

**Objective:** Present findings to stakeholders with working demo and rigorous evaluation.

**Presentation Structure:**

1. **Problem Statement (2 min)**
   - Store associates guess which item to cook first
   - Leads to stockouts and waste

2. **Solution Overview (3 min)**
   - Predictive ML ranking system
   - v1 simple rules vs. v2 learned model
   - Augmentation (recommend, don't automate)

3. **Data & Methodology (3 min)**
   - ~23,000 synthetic cook events (19,980 initial + ~3,000 restocks from sell-throughs)
   - Hybrid labeling (inferred vs. logged write-offs)
   - Training on ~5,290 labeled decision-point scenarios

4. **Demo (5 min)**
   - Show v1 ranking
   - Show v2 ranking
   - Compare outcomes
   - Explain why v2 is better

5. **Results (3 min)**
   - Cross-validation accuracy: ___%
   - Test accuracy: ___%
   - Write-off reduction: __% vs. v1
   - On-time delivery rate: __%

6. **Failures & Mitigations (2 min)**
   - What could go wrong and how we address it

7. **Next Steps & Production Path (2 min)**
   - What would be needed to deploy in real stores

**Failure Modes & Panel Questions:**

| Risk | What the Panel Asks | Our Answer |
|------|---|---|
| **Mis-Governance** | "Won't associates just ignore the system?" | We designed for trust via clear explanations. We'll track adherence (target: ≥80%). Success measured by actual behavior change, not model accuracy alone. |
| **Data Quality** | "Your write-off data is synthetic. How do we know it's realistic?" | We injected realistic logging errors: 60% accurate, 15% gaps, 20% counting errors, 5% major discrepancies. Validator independently confirms 66.4% high / 23.7% medium / 10.0% low confidence. Production v2 will use real warmer data. |
| **Mis-Priced Economics** | "How much will this cost to maintain?" | v1 has zero cost (hard-coded). v2 retraining is ~2 hours/week. Real baseline (6 months) would require infrastructure investment, but payoff is documented: 10–15% waste reduction. |

**Deliverables:**
- Presentation slides
- Working demo
- Evaluation report
- Architecture documentation
- README

---

## ARCHITECTURE

### System Components

```
┌─────────────────────────────────────────────────────────┐
│ Food Planner Output                                      │
│ (Forecast: quantities, hold times, demand windows)      │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ Feature Engineering                                      │
│ (urgency, demand_density, hold_time, store_type, hour)  │
└──────┬──────────────────┬──────────────────┬────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
  ┌─────────┐       ┌──────────┐      ┌───────────┐
  │   v1    │       │  v2.2    │      │ LLM       │
  │  Rules  │       │  ML      │      │ Benchmark │
  │  58.6%  │       │  68.9%   │      │ (ceiling) │
  └────┬────┘       └────┬─────┘      └─────┬─────┘
       │                 │                  │
       └────────┬─────────┘                 │
                ▼                           │
   ┌─────────────────────────┐     ┌────────▼────────┐
   │ Associate UI            │     │ Eval Harness    │
   │ (Ranked cook sequence)  │     │ (50-ex shared   │
   │ Streamlit app           │     │  eval set)      │
   └─────────────────────────┘     └─────────────────┘
```

**Three-tier evaluation** (see [`EVAL_METHODOLOGY.md`](EVAL_METHODOLOGY.md) for full protocol):
- **Tier 1 — Selection (authoritative):** ML holdout (730 scenarios, temporal split ≥ 2025-05-01). v1 58.6%, v2.2 **68.9%** — current selection winner (+10.3pp). **Head-to-head holdout eval** (150 modal + guardrails): v2.2 **67.3%** vs LLM v0.3 **26.7%** on modal formula top-1 (Jun 27 2026).
- **Tier 2 — Guardrails:** `must_precede_violation_rate` < 5%, refusal accuracy ≥ 90% (on ≥ 20 examples), latency budget, parse failure = 0%.
- **Tier 3 — Diagnostic only (never decisive):** v0.1/v0.2 formula-label evals + **JTBD v0.3** (110 examples, 2–5 items, plain-language). v0.3 headline (Jun 26 2026): LLM **95.8%** JTBD but only **78.9%** formula (native/prose); **90.4%** formula with fair `--input-mode=features` ≈ v1 **92.6%** / v2.2 **91.6%** — confounded by small-item set; do not use for selection.

### Data Flow

```
1. Generate Synthetic Data (Week 1–2)  ✅
   └─> Cook logs + POS sales + Write-off logs

2. Validate Quality (Week 2)  ✅
   └─> Data quality report (high/medium/low confidence)

3. Build v1 (Week 3–4)  ✅
   └─> Rule-based ranking — 58.6% top-1 (28-item set, 2,164 decision points)

4. Label Data (Week 5–6)  ✅
   └─> 2,164 labeled decision-point scenarios (28-item set)

5. Train v2 → v2.2 (Week 7)  ✅
   └─> Pairwise GBM — 68.9% honest test (28-item, temporal split)

6. LLM Benchmark (Week 8–9)  ✅
   └─> v0.1/v0.2: 50-ex shared eval; v0.1 50% → v0.2 64% top-1
   └─> v0.3 JTBD eval (Jun 2026): 110 plain-language scenarios, cook-now metrics (diagnostic)
   └─> Holdout head-to-head (Jun 27 2026): LLM 26.7% vs v2.2 67.3% on 150-ex modal slice

7. Demo (Streamlit, Week 8)  ✅
   └─> Associate vs v1 vs v2.2 side-by-side; 3 pages
```

---

## DATA STRUCTURE

### Cook Event (`data/cook_logs.json`)

```json
{
  "cook_event_id": "a1b2c3d4-...",
  "store_id": "urban_4521",
  "store_type": "urban",
  "item": "wings_2h",
  "date": "2025-01-15",
  "window": "10:00-12:00",
  "window_start_hour": 10,
  "window_end_hour": 12,
  "day_of_week": "Wednesday",
  "is_weekend": false,
  "forecast_demand": 15,
  "cooked_qty": 15,
  "cook_timestamp": "2025-01-15T10:08:00",
  "cook_type": "initial",
  "hold_time_hours": 2,
  "lowest_cookable_unit": 5,
  "exact_multiples": true,
  "equipment": "oven"
}
```

### POS Sale (`data/pos_sales.json`)

```json
{
  "sale_id": "e5f6g7h8-...",
  "cook_event_id": "a1b2c3d4-...",
  "store_id": "urban_4521",
  "store_type": "urban",
  "item": "wings_2h",
  "date": "2025-01-15",
  "window": "10:00-12:00",
  "quantity": 1,
  "sale_timestamp": "2025-01-15T11:04:00"
}
```

### Write-Off Event (`data/write_off_logs.json`)

```json
{
  "writeoff_id": "i9j0k1l2-...",
  "cook_event_id": "a1b2c3d4-...",
  "item": "wings_2h",
  "store_type": "urban",
  "date": "2025-01-15",
  "window": "10:00-12:00",
  "logged_writeoff_qty": 3,
  "inferred_writeoff_qty": 3,
  "writeoff_timestamp": "2025-01-15T12:47:00",
  "quality_type": "accurate",
  "delay_minutes": 47
}
```

### Validator Classification (in-memory, saved to `output/quality_report.json`)

```json
{
  "cook_event_id": "a1b2c3d4-...",
  "item": "wings_2h",
  "store_type": "urban",
  "window": "10:00-12:00",
  "date": "2025-01-15",
  "cooked_qty": 15,
  "sold_qty": 12,
  "inferred_writeoff": 3,
  "logged_writeoff": 3,
  "difference": 0,
  "confidence": "high",
  "quality_issue": "accurate"
}
```

---

## SUCCESS CRITERIA

### Week 12 Targets

| Metric | Baseline | Target | How We Measure |
|--------|----------|--------|---|
| **Model Accuracy** | — | ≥75% on test set | Cross-validation on holdout data |
| **Cook Decision Time** | ~60 sec | ≤5 sec | Time from oven-free to load decision |
| **Daypart Hit Rate** | ~25% | ≥75% | % of dayparts where min presentation is reached before window closes |
| **Write-Off Rate** | ~3 units/hr | ≤1 unit/hr | Write-offs per store per hour vs. forecast |
| **Correct Sequence Rate** | — | ≥70% of ranked decisions | AI rank results in zero daypart miss vs. baseline |
| **Associate Adherence** | — | ≥80% follow system | Track overrides (target ≤20% override rate) |
| **Pilot Usage** | — | ≥6 ranked sequences/store/day at ≥60% of pilot stores | Usage instrumentation |
| **Data Quality** | — | 95% usable | High + medium confidence scenarios |
| **Demo Robustness** | — | Zero errors | Run end-to-end 10+ times without failures |
| **Documentation** | — | Complete | README, assumptions, limitations all documented |

### Evaluation on High-Confidence Scenarios

For the 15,308 high-confidence scenarios (66.4% of data):
- ✅ v2 should match or beat v1 on >70%
- ✅ v2 should reduce write-offs by >10%
- ✅ v2 should maintain >95% on-time rate

---

## KEY RISKS & MITIGATIONS

### Risk 1: Adoption Risk (Associates Don't Follow System)

**Failure Mode:** Associates don't trust the ranking and load by habit anyway. Override data dries up, the feedback loop dies, and the model never improves. Associate loads wings last — they miss their window — and the stockout is invisible to management.

**Mitigation:**
- ✅ Template-based plain-language explanations (clear reasoning associates can defend to their manager)
- ✅ Planned adherence tracking (measure % following system; counter-metric: override rate ≤20%)
- ✅ Trust-building: target 80%+ adherence as success metric, not 100%
- ✅ Override logging is mandatory — every override is the most valuable training signal; without it the feedback loop dies

### Risk 2: Data Quality (Noisy Write-Off Records Corrupt Training Labels)

**Failure Mode:** Write-off records contain missing entries, manual errors, and misattribution. Noisy training labels cause the model to recommend confidently wrong sequences. Trust collapses faster than it was built.

**Mitigation:**
- ✅ Realistic synthetic generation (injected 60/15/20/5 quality distribution; independently validated)
- ✅ Confidence classification (high/medium/low) — exclude low-confidence events from training
- ✅ Honest documentation (flag synthetic data limitations upfront)
- ✅ Production path: requires 6+ months of labeled execution data from 13,000+ stores before production training; audit protocols required for write-off records before use
- ✅ Stratified by store type (acknowledge different stores have different patterns)

### Risk 3: Governance Risk at Scale (No Attribution Path for Failures)

**Failure Mode:** Model errors compound across stores. There is no attribution path between a model failure and an associate override. Corporate cannot tell whether a daypart miss was caused by the model or the associate, so the system gets blamed and pulled.

**Mitigation:**
- ✅ Override logging creates an auditable per-decision record (was the ranked sequence followed or overridden?)
- ✅ Per-store accuracy tracking (identify stores where model diverges from outcomes)
- ✅ Phased rollout — pilot 50 urban stores first; validate attribution before scaling to full chain

### Risk 4: Mis-Priced Economics (Cost Exceeds Benefit)

**Failure Mode:** Building a real baseline and maintaining the system costs more than the value from waste reduction.

**Mitigation:**
- ✅ Simple v1 (zero inference cost)
- ✅ Lightweight v2 (GBM, not deep learning; ~$0.021/call, $12.87/store/month at pilot)
- ✅ Documented ROI: $180/store/month in write-off savings vs. $12.87 cost ≈ 13x ROI at pilot, ~22x at full chain
- ✅ Clear production requirements (6-month baseline, weekly retraining, monitoring)

---

## COST & ROI

| Item | Detail |
|------|--------|
| **Per-call cost** | ~$0.021 (inference $0.000004 + API gateway $0.0000035 + human spot-check $0.02 at 1% review rate) |
| **Usage shape** | 12 calls/store/day (one decision per ~2-hour cook cycle); predictable and flat — no token explosion risk |
| **Cost at pilot (50 stores)** | ~$12.87/store/month |
| **Cost at full chain (13,000 stores)** | ~$7.72/store/month (fixed costs amortize: hosting $150/mo, observability $40/mo, feature store $60/mo) |
| **Write-off savings** | ~$180/store/month (3 units/hr → 1 unit/hr reduction) |
| **ROI at pilot** | ~13x |
| **ROI at full chain** | ~22x |

Internal tool — no SaaS pricing. Value case: $180/store/month savings vs. $12.87 cost at pilot.

---

## GETTING STARTED

### Prerequisites

```bash
pip install -r requirements.txt   # scikit-learn, numpy, pandas, streamlit, anthropic
```

Python 3.10+ required (uses `X | Y` union type syntax).

### Run the Streamlit Demo

```bash
streamlit run app/app.py
```

### Run with Docker (Backend + Frontend)

```bash
docker-compose up -d
```

- **Backend API:** http://localhost:8000 (health check at `/health`)
- **Frontend UI:** http://localhost:5173

To stop:

```bash
docker-compose down
```

### Run the Eval Harness

```bash
# Holdout head-to-head (SELECTION — 150 modal + 47 guardrails)
python3.11 notebooks/week9_llm_eval_runner.py --eval-set=holdout --prompt-version=v0.3 --no-llm   # ML only (~1 min)
python3.11 notebooks/week9_llm_eval_runner.py --eval-set=holdout --prompt-version=v0.3             # full LLM + ML (~30 min)

# v0.1/v0.2 sets — formula-label accuracy (legacy metric, backward-compat)
python3.11 notebooks/week9_llm_eval_runner.py --no-llm                        # ML baselines, v0.1 set
python3.11 notebooks/week9_llm_eval_runner.py --eval-set=v0.2 --no-llm        # ML baselines, v0.2 set

# v0.3 JTBD plain-language set (DIAGNOSTIC — 95 ranking + 15 refusal after expansion)
python3.11 notebooks/week9_llm_eval_runner.py --eval-set=v0.3 --prompt-version=v0.3 --no-llm

# Multi-seed associate floor (default: 20 seeds)
python3.11 notebooks/week9_llm_eval_runner.py --eval-set=v0.3 --no-llm --assoc-seeds=20
```

### Run the Full LLM Evaluation (requires Anthropic key)

```bash
cp .env.example .env   # add ANTHROPIC_API_KEY to .env (gitignored)
export ANTHROPIC_API_KEY=your_key   # or rely on .env via python-dotenv

# Native mode (prose for LLM; baseline comparison)
python3.11 notebooks/week9_llm_eval_runner.py --eval-set=v0.3 --prompt-version=v0.3

# Features-controlled mode (LLM gets numeric table — removes prose advantage)
python3.11 notebooks/week9_llm_eval_runner.py --eval-set=v0.3 --prompt-version=v0.3 \
    --input-mode=features

# Multi-sample LLM variance (3 runs, cost-gated)
python3.11 notebooks/week9_llm_eval_runner.py --eval-set=v0.3 --prompt-version=v0.3 \
    --llm-samples=3

# Compare native vs features-mode to quantify prose advantage
python3.11 scripts/compare_reports.py output/llm_eval_v0.3_v0_3_report.json \
                                       output/llm_eval_v0.3_v0_3_features_report.json
```

### Rebuild Eval Sets

```bash
python scripts/build_eval_set_holdout.py  # data/llm_eval_set_holdout.json + .csv (150 modal + 47 guardrails)
python scripts/build_eval_set_v0_1.py   # data/llm_eval_set_v0.1.json + .csv
python scripts/build_eval_set_v0_3.py   # data/llm_eval_set_v0.3.json + .csv (plain-language)
# v0.2 JSON is committed; builder script was removed — restore from git if missing
```

### JTBD Eval Metrics (v0.3 set)

| Metric | What it measures | Goal |
|---|---|---|
| **cook_now_accuracy** | Correct first item (in urgent set) | Headline — maximize |
| **cook_now_set_recall** | Fraction of urgent items in top-k | Maximize |
| **must_precede_violations** | Near-expiry item ranked after a long-hold item | **0** (waste proxy) |
| **refusal_accuracy** | Refusal-only examples (15 OOS) correctly refused | Maximize (≥ 90% on ≥ 20 ex) |
| **kendall_tau** | Full ordering quality (2–5 items, now meaningful) | Secondary |

**v0.3 scenario mix (110 total, expanded Jun 2026):** modal 40 • edge 23 • stockout 10 • no_demand 3 • triage 10 • OOS 15 (refusal-only) • adversarial 9 (ranking under override/injection). 16 divergence examples where JTBD answer differs from formula.

> ⚠️ **v0.3 is DIAGNOSTIC ONLY.** Confounds: LLM gets prose, ML gets numeric features; N=95 ranking gives ~±7pp CI; JTBD labels are unvalidated hypotheses on divergence cases. Do not use v0.3 results for model selection. See [`EVAL_METHODOLOGY.md`](EVAL_METHODOLOGY.md).

**v0.3 results (Jun 26 2026, expanded 110-ex set, 95 ranking + 15 refusal, `--assoc-seeds=20`):**

| Comparator | JTBD% | Formula% | Formula CI 95% | Set Recall | MPV | Refusal | Kendall τ |
|---|---|---|---|---|---|---|---|
| associate_floor | 66.3% ±4.4 | 50.5% | [41.0–60.0%] | 0.668 | 41 | — | 0.246 |
| **v1_rules** | 89.5% | **92.6%** | [87.4–97.9%] | 0.900 | 8 | — | 0.851 |
| v2_2_ml | 83.2% | 91.6% | [85.3–96.8%] | 0.837 | 16 | — | 0.835 |
| llm (native/prose) | **95.8%** | 78.9% | [70.5–87.4%] | 0.953 | **2** | **93.3%** | 0.675 |
| llm (`--input-mode=features`) | 93.6% | **90.4%** | [84.0–95.7%] | — | 1 | 93.3% | — |

**Dual-label breakdown (79 agrees / 16 divergence):**

| Slice | LLM JTBD (native) | LLM Formula (native) | LLM Formula (features) | v1 Formula | v2.2 Formula |
|---|---|---|---|---|---|
| Overall | 95.8% | 78.9% | **90.4%** | 92.6% | 91.6% |
| Labels agree | 96.2% | 93.7% | 98.7% | 94.9% | 91.1% |
| Labels diverge | 93.8% | 6.2% | 50.0% | 81.2% | 93.8% |

**Interpretation:** LLM wins JTBD (prose input + JTBD labels) but loses formula top-1 in native mode (p=0.009 vs v1, p=0.031 vs v2.2). With fair input (`--input-mode=features`), LLM formula **90.4%** ≈ v1/v2.2 (p=1.0) — the "LLM > ML" story was input+label confound. v1 vs v2.2 still not significant (p=1.0). LLM passes guardrails (2 MPV, 93% refusal on 15 ex); ML models fail MPV threshold on this set.

Reports: `output/llm_eval_v0.3_v0_3_report.json` (native), `output/llm_eval_v0.3_v0_3_features_report.json` (fair).

**Compare two reports with CIs and significance:**
```bash
python scripts/compare_reports.py output/llm_eval_v0.3_v0_3_report.json \
                                   output/llm_eval_v0.3_v0_3_features_report.json
```

### Retrain v2.2 Model

```bash
python notebooks/week7_model_training.py
```

---

## PROJECT STRUCTURE

```
7eleven-cook-scheduling/
├── data/
│   ├── cook_logs.json                  # 175,051 cook events (28-item set; gitignored — regenerate locally)
│   ├── pos_sales.json                  # 766,229 sale records (gitignored)
│   ├── write_off_logs.json             # 148,422 write-off entries (gitignored)
│   ├── labeled_training_set.json       # 2,164 labeled decision-point scenarios (gitignored)
│   ├── llm_eval_set_v0.1.json          # 50-example shared eval set (formula-label, 5-item universe)
│   ├── llm_eval_set_v0.1.csv           # Same, CSV format
│   ├── llm_eval_set_v0.2.json          # 53-example 28-item eval set (formula-label)
│   ├── llm_eval_set_v0.2.csv           # Same, CSV format
│   ├── llm_eval_set_v0.3.json          # 110-example JTBD plain-language eval (Jun 2026)
│   ├── llm_eval_set_v0.3.csv           # Same, CSV format
│   ├── llm_eval_set_holdout.json       # 197-ex holdout eval (150 modal + 47 guardrails)
│   ├── llm_eval_set_holdout.csv        # Same, CSV format
│   └── interview_notes.md              # 15 simulated associate vignettes
│
├── prompts/
│   ├── v0.1_system_prompt.md          # LLM prompt v0.1 (initial benchmark)
│   ├── v0.2_system_prompt.md          # LLM prompt v0.2 (associate-voice tightening)
│   └── v0.3_system_prompt.md          # LLM prompt v0.3 (plain-language input)
│
├── src/
│   ├── cook_scheduler.py               # v1 rule-based ranker + AssociateBaseline
│   ├── data_labeler.py                 # Composite priority labeling
│   ├── pairwise_trainer.py             # v2.1/v2.2 pairwise GBM training
│   ├── lambdarank_trainer.py           # v3 LightGBM LambdaRank (listwise, group=scenario)
│   ├── llm_ranker.py                   # LLMRanker (shared, uses ANTHROPIC_MODEL env)
│   ├── synthetic_data_generator.py
│   └── data_validator.py
│
├── notebooks/
│   ├── week1_data_generation.py
│   ├── week3_v1_scheduler.py
│   ├── week5_data_labeling.py
│   ├── week7_model_training.py         # v2 → v2.1 → v2.2 → v3 full pipeline
│   ├── week8_llm_benchmark.py          # LLM ranker standalone benchmark
│   └── week9_llm_eval_runner.py        # ⭐ Main eval harness (v0.1/v0.2/v0.3, JTBD metrics)
│
├── scripts/
│   ├── build_eval_set_v0_1.py          # Builds llm_eval_set_v0.1.json/.csv
│   ├── build_eval_set_v0_3.py          # Builds llm_eval_set_v0.3.json/.csv (JTBD, 110 examples)
│   ├── build_eval_set_holdout.py       # Builds llm_eval_set_holdout.json/.csv (197 examples)
│   ├── compare_llm_versions.py         # LLM prompt A/B diff (legacy)
│   └── compare_reports.py              # ⭐ General report diff with CIs + significance
│
├── models/
│   ├── v2_ranking_model.pkl            # v2 multiclass baseline
│   ├── v2_1_pairwise_model.pkl         # v2.1 pairwise (no temporal split)
│   ├── v2_2_pairwise_temporal.pkl      # v2.2 — current app/API model
│   ├── v2_3_pairwise_symmetric.pkl     # v2.3 symmetric pairwise (experimental)
│   └── v3_lambdarank.pkl               # v3 LightGBM LambdaRank (generated by pipeline)
│
├── output/
│   ├── v1_eval_report.json             # v1: 58.6% top-1 (28-item set)
│   ├── v2_2_temporal_report.json       # v2.2: 68.9% honest test (28-item set)
│   ├── v3_lambdarank_report.json       # v3: NDCG@1/3/5 + top-1 (generated by pipeline)
│   ├── llm_eval_v0.1_report.json       # LLM v0.1: 50.0% top-1
│   ├── llm_eval_v0.2_report.json       # LLM v0.2: 64.0% top-1
│   ├── llm_eval_v0.3_v0_3_report.json          # JTBD v0.3 native (prose) — Jun 26 2026
│   ├── llm_eval_v0.3_v0_3_features_report.json  # JTBD v0.3 fair input (--input-mode=features)
│   ├── llm_eval_v0.3_holdout_report.json       # Holdout head-to-head — Jun 27 2026
│   ├── llm_eval_v0.3_holdout_predictions.json  # Per-example holdout predictions
│   ├── v2_3_symmetric_report.json              # v2.3 training report
│   ├── llm_eval_v0.1_predictions.json  # Per-example predictions
│   ├── llm_eval_v0.2_predictions.json
│   ├── llm_eval_v0.3_v0_3_predictions.json
│   ├── llm_eval_version_comparison.json
│   └── feature_importance.json
│
├── app/                                # Streamlit demo
│   ├── app.py
│   └── pages/
│       ├── 1_Scenario_Comparison.py
│       ├── 2_Impact_Dashboard.py
│       └── 3_What_If_Simulator.py
│
├── EVAL_METHODOLOGY.md                 # ⭐ Authoritative eval protocol (selection vs diagnostic)
├── SPRINT1_SUMMARY.md                  # Sprint 1 results summary with caveats
├── ARCHITECTURE_DECISIONS.md           # Design decisions + two-tier eval strategy
└── requirements.txt
```

---

## TIMELINE SUMMARY

| Week | Phase | Deliverable | Actual Result |
|------|-------|---|---|
| 1–2 | Data Generation | ~23,000 cook events, quality report | ✅ Sprint 1: 66.4% high, 90.0% usable |
| 3–4 | v1 Baseline | Rule-based ranker | ✅ Sprint 1: 71.8% (5-item) → **58.6%** (28-item retrain) |
| 5–6 | Data Labeling | 5,290 → 2,164 labeled scenarios | ✅ 28-item: 2,164 scenarios, 0 contradictions |
| 7 | Model Training | v2 → v2.2 pairwise GBM | ✅ Sprint 1: 74.3% (5-item) → **68.9%** (28-item retrain) |
| 8–9 | Demo + LLM Eval | Streamlit app, 50-ex eval, v0.1→v0.2 | ✅ Sprint 1 complete (⭐ SPRINT1_SUMMARY.md) |
| Post-S1 | Item Expansion | 28-item set, retrained v2.2 | ✅ 175K events, 2,164 scenarios, v2.2 68.9% test, +60pp vs associate |
| Post-S1 | v2.3 + v3 | Symmetric pairwise + LambdaRank | ✅ v2.3 66.0% / v3 66.2% on 730 holdout; v2.2 remains selection winner |
| Post-S1 | Holdout head-to-head eval | 197-ex set, LLM vs ML on modal slice | ✅ LLM 26.7% vs v2.2 67.3% formula top-1; hypothesis confirmed |

---

## WHAT SUCCESS LOOKS LIKE

**By Week 12, you'll have:**

1. ✅ A working prototype that takes Food Planner output and recommends a cook sequence
2. ✅ v1 (simple rules) that works for >90% of scenarios
3. ✅ v2 (ML-trained) that outperforms v1 on >70% of scenarios
4. ✅ Rigorous evaluation showing ≥75% accuracy on unseen test data
5. ✅ Honest documentation of limitations (synthetic data, assumptions, production requirements)
6. ✅ Clear failure mode analysis and mitigation strategies
7. ✅ Working demo you can show stakeholders
8. ✅ Roadmap for production deployment (6-month baseline, weekly retraining)

**The panel will see:**
- You understand the problem deeply (realistic JTBD, specific moment)
- You chose the right AI approach (predictive ML, not LLM)
- You built it responsibly (tested on synthetic data, flagged limitations)
- You measured results rigorously (cross-validation, holdout test set)
- You're honest about what you don't know (data quality issues, production requirements)

---

**Last Updated:** June 27, 2026  
**Project Status:** Sprint 1 complete (Weeks 1–9) + 28-item expansion + JTBD v0.3 diagnostic eval + **holdout head-to-head eval** (Jun 27 2026).

**Selection decision (Tier 1 holdout, authoritative):** `v2_2_ml` **68.9%** vs v1 58.6% (+10.3pp, 730-scenario temporal holdout). Head-to-head holdout eval confirms LLM v0.3 **26.7%** on 150-ex modal slice (−40.6pp vs v2.2) — idealized human judgment underperforms ML.

**JTBD v0.3 diagnostic eval (110 examples, Jun 26 2026):** LLM **95.8%** JTBD / **78.9%** formula (native) → **90.4%** formula (fair input) ≈ v1 **92.6%** • v2.2 **91.6%**. Useful for guardrails/refusal; not for selection (small-item prose confound). See [`EVAL_METHODOLOGY.md`](EVAL_METHODOLOGY.md).

**Holdout eval report:** `output/llm_eval_v0.3_holdout_report.json` — run with `python notebooks/week9_llm_eval_runner.py --eval-set=holdout --prompt-version=v0.3`.

See [`SPRINT1_SUMMARY.md`](SPRINT1_SUMMARY.md) for full results history. See [`EVAL_METHODOLOGY.md`](EVAL_METHODOLOGY.md) for decision framework, guardrails, and statistical protocol.
