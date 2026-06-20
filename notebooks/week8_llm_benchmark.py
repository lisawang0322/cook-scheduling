"""Week 8 — LLM Benchmark: ML vs. Rule-Based vs. LLM for Cook Sequencing.

Compares four approaches on the same held-out test set (temporal split,
cutoff 2025-05-01, 1,747 scenarios) using identical metrics:

  1. Random (floor)
  2. v1 CookSchedulerV1 (deterministic rules)
  3. v2.2 PairwiseModelTrainer (pairwise GradientBoosting, best ML candidate)
  4. LLM zero-shot (Anthropic Claude, observable features only)
  5. LLM few-shot  (3 training-set examples, no test leakage)

Accuracy metric: top-1 ranking correctness, matching evaluate_ranking_accuracy()
used in week7_model_training.py for a fair apples-to-apples comparison.

Additional metrics: Kendall's tau (full ranking quality), latency,
LLM output variance across N repeated runs on a held-out variance subset.

IMPORTANT — label interpretation:
  Ground-truth labels come from the deterministic formula in
  data_labeler.py:_determine_optimal_order(). Accuracy measures recovery
  of that domain-expert logic, not real-world optimality. LLM convergence
  on similar rankings validates that labels encode learnable domain patterns.
  See ARCHITECTURE_DECISIONS.md § "Labeling Methodology and Benchmark Interpretation".

Usage:
  export ANTHROPIC_API_KEY=your_key
  python notebooks/week8_llm_benchmark.py

Configuration constants below control cost (SUBSAMPLE_N) and variance depth.
"""

import json
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import kendalltau

# --- Path setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.cook_scheduler import CookSchedulerV1
from src.pairwise_trainer import PairwiseModelTrainer, OVEN_ITEMS
from src.llm_ranker import LLMRanker, extract_observable_features

# --- Configuration ---
CUTOFF_DATE = "2025-05-01"      # Temporal split date (matches week7 training)
SUBSAMPLE_N = 200               # Scenarios to evaluate (controls LLM cost)
                                # Set to None to evaluate the full 1,747-scenario test set
FEW_SHOT_ENABLED = True         # Run LLM few-shot variant (additional API calls)
N_FEW_SHOT_EXAMPLES = 3        # Training-set examples injected into few-shot prompt
N_VARIANCE_RUNS = 3             # Repeated LLM runs to measure output variance
VARIANCE_SUBSET_N = 30          # Scenarios used for variance measurement
LLM_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
RANDOM_SEED = 42


# ===========================================================================
# Helpers
# ===========================================================================

def load_data() -> tuple[list[dict], list[dict], list[dict]]:
    """Load labeled data and split into train/test by temporal cutoff."""
    data_path = os.path.join(PROJECT_ROOT, "data", "labeled_training_set.json")
    with open(data_path) as f:
        all_data = json.load(f)

    train = [s for s in all_data if s["features"]["date"] < CUTOFF_DATE]
    test  = [s for s in all_data if s["features"]["date"] >= CUTOFF_DATE]
    return all_data, train, test


def load_v22_model() -> PairwiseModelTrainer:
    """Load the v2.2 pairwise model and attach to a PairwiseModelTrainer instance."""
    model_path = os.path.join(PROJECT_ROOT, "models", "v2_2_pairwise_temporal.pkl")
    with open(model_path, "rb") as f:
        pkl = pickle.load(f)
    trainer = PairwiseModelTrainer()
    trainer.model = pkl["model"]
    trainer.historical = pkl["historical"]
    return trainer


def get_present_items(features: dict[str, Any]) -> list[str]:
    return [item for item in OVEN_ITEMS if f"{item}_forecast_demand" in features]


def v1_rank(scheduler: CookSchedulerV1, features: dict[str, Any]) -> list[str]:
    """Reconstruct pending_items format and run v1 rule-based ranker."""
    decision_hour = features["decision_hour"] + 0.25
    pending = []
    for item in get_present_items(features):
        time_rem = features[f"{item}_time_remaining"]
        pending.append({
            "item": item,
            "forecast_demand": features[f"{item}_forecast_demand"],
            "lcu": features[f"{item}_lcu"],
            "hold_time_hours": features[f"{item}_hold_time"],
            "exact_multiples": features.get(f"{item}_exact_multiples", True),
            "window_start_hour": features["decision_hour"],
            "window_end_hour": features["decision_hour"] + 0.25 + time_rem,
        })
    ranked = scheduler.rank_items(decision_hour, pending)
    return [r["item"] for r in ranked]


def random_rank(features: dict[str, Any], rng: random.Random) -> list[str]:
    items = get_present_items(features)
    rng.shuffle(items)
    return items


def compute_tau(pred: list[str], truth: list[str]) -> float | None:
    """Kendall's tau between predicted and ground-truth ranking."""
    if len(pred) < 2 or set(pred) != set(truth):
        return None
    truth_rank = {item: i for i, item in enumerate(truth)}
    pred_rank  = {item: i for i, item in enumerate(pred)}
    items = list(truth)
    x = [truth_rank[i] for i in items]
    y = [pred_rank[i]  for i in items]
    tau, _ = kendalltau(x, y)
    return round(float(tau), 4)


def select_few_shot_examples(
    train_scenarios: list[dict],
    n: int = 3,
) -> list[tuple[dict, list[str]]]:
    """Pick n diverse training examples for few-shot prompt.

    Diversity criterion: different store types and hours.
    Returns list of (observable_features, optimal_order) tuples.
    """
    targets = [
        ("urban",    6),
        ("suburban", 12),
        ("highway",  18),
    ]
    examples = []
    used = set()
    for store_type, hour in targets[:n]:
        for s in train_scenarios:
            sid = s["scenario_id"]
            f = s["features"]
            if (f["store_type"] == store_type
                    and f["decision_hour"] == hour
                    and sid not in used):
                obs = extract_observable_features(f)
                examples.append((obs, s["optimal_order"]))
                used.add(sid)
                break
    # If we couldn't find all targeted examples, pad with random training scenarios
    rng = random.Random(RANDOM_SEED)
    remaining = [s for s in train_scenarios if s["scenario_id"] not in used]
    rng.shuffle(remaining)
    while len(examples) < n and remaining:
        s = remaining.pop()
        obs = extract_observable_features(s["features"])
        examples.append((obs, s["optimal_order"]))
    return examples[:n]


def run_comparator(
    name: str,
    scenarios: list[dict],
    rank_fn,  # callable(features) -> list[str] | None
) -> dict[str, Any]:
    """Run a single comparator over all scenarios and collect results."""
    top1_correct = 0
    taus: list[float] = []
    latencies: list[float] = []
    parse_failures = 0

    for scenario in scenarios:
        features = scenario["features"]
        truth_order = scenario["optimal_order"]
        truth_first = scenario["optimal_first_item"]

        t0 = time.time()
        pred = rank_fn(features)
        latency_ms = (time.time() - t0) * 1000

        if pred is None:
            parse_failures += 1
            continue

        latencies.append(latency_ms)
        if pred[0] == truth_first:
            top1_correct += 1

        tau = compute_tau(pred, truth_order)
        if tau is not None:
            taus.append(tau)

    evaluated = len(scenarios) - parse_failures
    return {
        "name": name,
        "top1_accuracy": round(100 * top1_correct / evaluated, 1) if evaluated else 0,
        "kendall_tau_mean": round(float(np.mean(taus)), 4) if taus else None,
        "latency_median_ms": round(float(np.median(latencies)), 1) if latencies else None,
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 1) if latencies else None,
        "parse_failures": parse_failures,
        "evaluated": evaluated,
    }


def run_slice_analysis(
    scenarios: list[dict],
    comparators: dict[str, Any],  # name -> rank_fn
    slice_key: str,               # feature key, e.g. "store_type"
) -> dict[str, dict[str, float]]:
    """Top-1 accuracy by slice value for each comparator."""
    slices: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for scenario in scenarios:
        features = scenario["features"]
        slice_val = str(features.get(slice_key, "unknown"))
        truth_first = scenario["optimal_first_item"]

        for name, rank_fn in comparators.items():
            pred = rank_fn(features)
            if pred is not None:
                slices[slice_val][name].append(int(pred[0] == truth_first))

    result = {}
    for slice_val, comp_results in slices.items():
        result[slice_val] = {
            name: round(100 * np.mean(correct), 1)
            for name, correct in comp_results.items()
            if correct
        }
    return dict(sorted(result.items()))


def measure_llm_variance(
    ranker: LLMRanker,
    scenarios: list[dict],
    n_runs: int,
    few_shot_examples: list[tuple] | None = None,
) -> dict[str, Any]:
    """Run LLM n_runs times per scenario and measure disagreement rate."""
    disagreements = 0
    total = 0
    parse_failures = 0

    for scenario in scenarios:
        features = scenario["features"]
        runs: list[list[str]] = []

        for _ in range(n_runs):
            pred, _ = ranker.rank(features, few_shot_examples)
            if pred is None:
                parse_failures += 1
            else:
                runs.append(pred)

        if len(runs) >= 2:
            reference = runs[0]
            for run in runs[1:]:
                if run != reference:
                    disagreements += 1
                total += 1

    return {
        "scenarios_tested": len(scenarios),
        "n_runs_per_scenario": n_runs,
        "disagreement_rate_pct": round(100 * disagreements / total, 1) if total else 0,
        "parse_failures": parse_failures,
    }


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("=" * 62)
    print("  WEEK 8 — LLM BENCHMARK: ML vs. Rules vs. LLM")
    print("=" * 62)

    # --- Load data ---
    print("\n[1/6] Loading data and splitting at cutoff", CUTOFF_DATE)
    _, train_scenarios, test_scenarios = load_data()
    print(f"  Train scenarios : {len(train_scenarios):,}")
    print(f"  Test scenarios  : {len(test_scenarios):,}")

    # Subsample test set for cost control (reproducible)
    rng = random.Random(RANDOM_SEED)
    eval_scenarios = test_scenarios[:]
    if SUBSAMPLE_N and len(eval_scenarios) > SUBSAMPLE_N:
        rng.shuffle(eval_scenarios)
        eval_scenarios = eval_scenarios[:SUBSAMPLE_N]
    print(f"  Evaluating      : {len(eval_scenarios):,} scenarios (seed={RANDOM_SEED})")

    # Variance subset (smaller, drawn from eval set)
    variance_scenarios = eval_scenarios[:VARIANCE_SUBSET_N]

    # --- Load models ---
    print("\n[2/6] Loading v1 and v2.2 models")
    scheduler = CookSchedulerV1()
    pw_model = load_v22_model()
    print("  v1  : CookSchedulerV1 loaded")
    print("  v2.2: PairwiseModelTrainer loaded (models/v2_2_pairwise_temporal.pkl)")

    # --- LLM setup ---
    print("\n[3/6] Initialising LLM ranker (Anthropic /", LLM_MODEL, ")")
    ranker = LLMRanker(model=LLM_MODEL, max_retries=1)
    few_shot_examples = select_few_shot_examples(train_scenarios, N_FEW_SHOT_EXAMPLES)
    print(f"  Few-shot examples selected from train set:")
    for obs, order in few_shot_examples:
        print(f"    {obs['store_type']:10s} @ hour {obs['decision_hour']:02d}  →  {order}")

    # --- Run comparators ---
    print(f"\n[4/6] Running comparators on {len(eval_scenarios)} scenarios...")

    rand_rng = random.Random(RANDOM_SEED)

    def rank_random(f):  return random_rank(f, rand_rng)
    def rank_v1(f):      return v1_rank(scheduler, f)
    def rank_v22(f):     return pw_model.rank_items(f, get_present_items(f))
    def rank_llm_zs(f):  pred, _ = ranker.rank(f, None); return pred
    def rank_llm_fs(f):  pred, _ = ranker.rank(f, few_shot_examples); return pred

    results: list[dict] = []

    print("  → Random baseline...")
    results.append(run_comparator("random", eval_scenarios, rank_random))

    print("  → v1 rule-based...")
    results.append(run_comparator("v1_rules", eval_scenarios, rank_v1))

    print("  → v2.2 ML...")
    results.append(run_comparator("v2_2_ml", eval_scenarios, rank_v22))

    print(f"  → LLM zero-shot ({len(eval_scenarios)} API calls)...")
    results.append(run_comparator("llm_zero_shot", eval_scenarios, rank_llm_zs))

    if FEW_SHOT_ENABLED:
        print(f"  → LLM few-shot ({len(eval_scenarios)} API calls)...")
        results.append(run_comparator("llm_few_shot", eval_scenarios, rank_llm_fs))

    # --- Variance measurement ---
    print(f"\n[5/6] Measuring LLM variance ({VARIANCE_SUBSET_N} scenarios × {N_VARIANCE_RUNS} runs)...")
    variance = measure_llm_variance(ranker, variance_scenarios, N_VARIANCE_RUNS)
    print(f"  Zero-shot disagreement rate: {variance['disagreement_rate_pct']}%")

    # --- Slice analysis ---
    print("\n[5b/6] Slice analysis by store_type and decision_hour...")
    slice_fns = {
        "v1_rules":      rank_v1,
        "v2_2_ml":       rank_v22,
        "llm_zero_shot": rank_llm_zs,
    }
    slices_store = run_slice_analysis(eval_scenarios, slice_fns, "store_type")
    slices_hour  = run_slice_analysis(eval_scenarios, slice_fns, "decision_hour")
    slices_items = run_slice_analysis(eval_scenarios, slice_fns, "num_oven_items")

    # --- Print summary table ---
    print("\n" + "=" * 62)
    print("  BENCHMARK RESULTS")
    print("=" * 62)
    print(f"\n  {'Comparator':<20s} {'Top-1%':>7s} {'Kendall τ':>10s} {'Latency(ms)':>12s} {'Failures':>9s}")
    print(f"  {'-'*20} {'-'*7} {'-'*10} {'-'*12} {'-'*9}")
    for r in results:
        tau_str = f"{r['kendall_tau_mean']:.3f}" if r["kendall_tau_mean"] is not None else "  n/a"
        lat_str = f"{r['latency_median_ms']:.1f}" if r["latency_median_ms"] is not None else "n/a"
        print(
            f"  {r['name']:<20s} {r['top1_accuracy']:>6.1f}% "
            f"{tau_str:>10s} {lat_str:>11s} ms "
            f"{r['parse_failures']:>7d}"
        )

    print(f"\n  LLM variance (zero-shot, {VARIANCE_SUBSET_N} scenarios × {N_VARIANCE_RUNS} runs):")
    print(f"    Disagreement rate : {variance['disagreement_rate_pct']}%")

    print("\n  Slice analysis — Top-1% by store_type:")
    print(f"  {'Store':>10s}", end="")
    for name in slice_fns:
        print(f"  {name:>14s}", end="")
    print()
    for store, scores in slices_store.items():
        print(f"  {store:>10s}", end="")
        for name in slice_fns:
            print(f"  {scores.get(name, 0):>13.1f}%", end="")
        print()

    # --- Save report ---
    report = {
        "metadata": {
            "run_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "cutoff_date": CUTOFF_DATE,
            "test_scenarios_total": len(test_scenarios),
            "subsample_n": len(eval_scenarios),
            "llm_model": LLM_MODEL,
            "random_seed": RANDOM_SEED,
            "few_shot_enabled": FEW_SHOT_ENABLED,
            "n_few_shot_examples": N_FEW_SHOT_EXAMPLES,
            "n_variance_runs": N_VARIANCE_RUNS,
            "variance_subset_n": VARIANCE_SUBSET_N,
            "label_note": (
                "Accuracy measures recovery of domain-expert labeling formula "
                "(data_labeler.py:_determine_optimal_order), not real-world optimality. "
                "LLM convergence validates that labels encode learnable domain patterns."
            ),
        },
        "results": results,
        "llm_variance": variance,
        "slice_analysis": {
            "by_store_type": slices_store,
            "by_decision_hour": slices_hour,
            "by_num_oven_items": slices_items,
        },
        "few_shot_examples": [
            {
                "store_type": obs["store_type"],
                "decision_hour": obs["decision_hour"],
                "optimal_order": order,
            }
            for obs, order in few_shot_examples
        ],
    }

    output_path = os.path.join(PROJECT_ROOT, "output", "llm_benchmark_report.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[6/6] Report saved → {output_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()
