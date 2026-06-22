"""Week 9 — LLM Evaluation Runner: v0.1 Prompt on 50-Example Curated Set.

Runs the v0.1 system prompt (prompts/v0.1_system_prompt.md) against the
curated 50-example evaluation set (data/llm_eval_set_v0.1.json) and compares
LLM zero-shot against v1 rules and v2.2 ML across three breakdowns:

  - By source       (synthetic_logs / synthetic_constructed / simulated_interview)
  - By eval_tags    (accuracy / edge_case / divergence)
  - By store_type   (urban / suburban / highway)
  - By hour_band    (morning / lunch / afternoon / evening)

Metrics: top-1 accuracy (matches week8 metric for fair comparison),
         Kendall's tau (full ranking quality), parse failures.

LLM is run zero-shot with the v0.1 system prompt body extracted from
prompts/v0.1_system_prompt.md. The few-shot examples in week8 are NOT
used here — this tests the prompt alone.

Usage:
  export ANTHROPIC_API_KEY=your_key
  python notebooks/week9_llm_eval_runner.py

  To skip LLM (dry run, v1+v2.2 only):
  python notebooks/week9_llm_eval_runner.py --no-llm
"""

import json
import os
import pickle
import random
import re
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import kendalltau

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.cook_scheduler import CookSchedulerV1, AssociateBaseline
from src.pairwise_trainer import PairwiseModelTrainer, OVEN_ITEMS

EVAL_SET_PATH   = os.path.join(PROJECT_ROOT, "data",   "llm_eval_set_v0.1.json")
MODEL_PATH      = os.path.join(PROJECT_ROOT, "models",  "v2_2_pairwise_temporal.pkl")
LLM_MODEL       = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
RANDOM_SEED     = 42

# Prompt version from --prompt-version=vX.Y CLI arg (default: v0.1)
PROMPT_VERSION  = next(
    (a.split("=", 1)[1] for a in sys.argv if a.startswith("--prompt-version=")),
    "v0.1",
)
PROMPT_PATH      = os.path.join(PROJECT_ROOT, "prompts", f"{PROMPT_VERSION}_system_prompt.md")
OUTPUT_PATH      = os.path.join(PROJECT_ROOT, "output",  f"llm_eval_{PROMPT_VERSION}_report.json")
PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "output",  f"llm_eval_{PROMPT_VERSION}_predictions.json")

NO_LLM = "--no-llm" in sys.argv


# ===========================================================================
# Prompt extraction
# ===========================================================================

def load_system_prompt() -> str:
    """Extract the prompt body from the ```...``` fenced block in the versioned prompt file."""
    with open(PROMPT_PATH) as f:
        content = f.read()
    match = re.search(r"## Prompt Body\s*```\s*(.*?)```", content, re.DOTALL)
    if not match:
        raise ValueError(
            f"Could not find '## Prompt Body' fenced block in {PROMPT_PATH}. "
            "Ensure the prompt is wrapped in a triple-backtick code block under that heading."
        )
    return match.group(1).strip()


# ===========================================================================
# Data loading
# ===========================================================================

def load_eval_set() -> tuple[dict, list[dict]]:
    with open(EVAL_SET_PATH) as f:
        data = json.load(f)
    return data["metadata"], data["examples"]


def load_v22_model() -> PairwiseModelTrainer:
    with open(MODEL_PATH, "rb") as f:
        pkl = pickle.load(f)
    trainer = PairwiseModelTrainer()
    trainer.model = pkl["model"]
    trainer.historical = pkl["historical"]
    return trainer


# ===========================================================================
# Rankers
# ===========================================================================

def present_items(features: dict) -> list[str]:
    found = {k[: -len("_forecast_demand")] for k in features if k.endswith("_forecast_demand")}
    ordered = [item for item in OVEN_ITEMS if item in found]
    extras = sorted(found - set(ordered))
    return ordered + extras


def rank_v1(scheduler: CookSchedulerV1, features: dict) -> list[str] | None:
    decision_hour = features["decision_hour"] + 0.25
    pending = []
    for item in present_items(features):
        pending.append({
            "item": item,
            "forecast_demand": features[f"{item}_forecast_demand"],
            "lcu": features[f"{item}_lcu"],
            "hold_time_hours": features[f"{item}_hold_time"],
            "exact_multiples": features.get(f"{item}_exact_multiples", True),
            "window_start_hour": features["decision_hour"],
            "window_end_hour": features["decision_hour"] + 0.25 + features[f"{item}_time_remaining"],
        })
    ranked = scheduler.rank_items(decision_hour, pending)
    return [r["item"] for r in ranked] if ranked else None


def rank_v22(trainer: PairwiseModelTrainer, features: dict) -> list[str] | None:
    items = present_items(features)
    return trainer.rank_items(features, items) if items else None


def build_oven_events(features: dict) -> list[dict]:
    """Reconstruct the oven-event list AssociateBaseline expects."""
    return [
        {
            "item": item,
            "hold_time_hours": features[f"{item}_hold_time"],
            "forecast_demand": features[f"{item}_forecast_demand"],
        }
        for item in present_items(features)
    ]


def rank_associate(associate: AssociateBaseline, features: dict) -> list[str] | None:
    """Realistic associate floor (habit/expiration/random mix) on the same set."""
    events = build_oven_events(features)
    return associate.rank_items(events) if events else None


def build_llm_user_prompt(features: dict) -> str:
    """Format a decision scenario into the user-turn message.

    Associate-legible view: no demand_density (a formula term). LCU is shown as
    the physical tray size an associate knows, not as a computed ratio.
    """
    items_lines = []
    for item in present_items(features):
        demand = features[f"{item}_forecast_demand"]
        lcu = features[f"{item}_lcu"]
        hold = features[f"{item}_hold_time"]
        tr = features[f"{item}_time_remaining"]
        items_lines.append(
            f"  {item:<12s} — need {demand} units, "
            f"{tr}hr left in window, stays good {hold}hr once cooked, "
            f"cooks {lcu} to a tray"
        )

    return (
        f"Decision point:\n"
        f"  Store type : {features['store_type']}\n"
        f"  Day        : {features['day_of_week']} (weekend={features['is_weekend']})\n"
        f"  Hour       : {features['decision_hour']}:00\n\n"
        f"Items present:\n" + "\n".join(items_lines)
    )


def is_refusal_response(text: str) -> bool:
    """Return True if the LLM returned a refusal/error rather than a ranking."""
    if '"error"' in text:
        return True
    ltext = text.lower()
    return any(p in ltext for p in [
        "i can only", "i'm unable", "i cannot help", "out of scope",
        "not able to", "only help with cook", "only assist with cook",
    ])


def parse_llm_response(text: str, present: list[str]) -> list[str] | None:
    """Parse LLM JSON response into ranked list; returns None on failure."""
    json_match = re.search(r'\{[^{}]*"ranked_queue"[^{}]*\}', text, re.DOTALL)
    if not json_match:
        return None
    try:
        data = json.loads(json_match.group())
        ranked = data.get("ranked_queue", [])
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(ranked, list):
        return None
    if set(ranked) != set(present) or len(ranked) != len(present):
        return None
    return ranked


class V01LLMRanker:
    """LLM ranker using the v0.1 system prompt extracted from prompts/."""

    def __init__(self, model: str = LLM_MODEL):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("pip install anthropic") from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. "
                "Export it before running: export ANTHROPIC_API_KEY=your_key"
            )
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.system_prompt = load_system_prompt()

    def rank(self, features: dict) -> list[str] | None:
        present = present_items(features)
        user_msg = build_llm_user_prompt(features)
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=128,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
        except Exception as exc:
            if exc.__class__.__name__ == "NotFoundError":
                raise RuntimeError(
                    f"Anthropic model not found: '{self.model}'. "
                    "Set ANTHROPIC_MODEL to an available model alias (example: "
                    "claude-3-5-haiku-latest) and rerun."
                ) from exc
            raise
        return parse_llm_response(message.content[0].text, present)

    def check_refusal(self, input_text: str) -> tuple[bool, str]:
        """Send an OOS/adversarial input; return (is_correct_refusal, raw_response)."""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=128,
                system=self.system_prompt,
                messages=[{"role": "user", "content": input_text}],
            )
            raw = message.content[0].text
            return is_refusal_response(raw), raw
        except Exception as exc:
            return False, str(exc)


# ===========================================================================
# Evaluation helpers
# ===========================================================================

def compute_tau(pred: list[str], truth: list[str]) -> float | None:
    if len(pred) < 2 or set(pred) != set(truth):
        return None
    truth_rank = {item: i for i, item in enumerate(truth)}
    pred_rank  = {item: i for i, item in enumerate(pred)}
    x = [truth_rank[item] for item in truth]
    y = [pred_rank[item]  for item in truth]
    tau, _ = kendalltau(x, y)
    return round(float(tau), 4)


def evaluate_comparator(
    name: str,
    examples: list[dict],
    rank_fn,
    refusal_fn=None,  # fn(input_text: str) -> (bool, str) | None; None = skip OOS/adv
) -> dict[str, Any]:
    ranking_correct = 0
    ranking_evaluated = 0
    refusal_correct = 0
    refusal_evaluated = 0
    taus: list[float] = []
    latencies: list[float] = []
    parse_failures = 0

    per_example: list[dict] = []

    for ex in examples:
        features    = ex["features"]
        truth_order = ex.get("optimal_order") or []
        truth_first = ex.get("optimal_first_item")
        is_refusal_ex = any(t in ("OOS", "adversarial") for t in ex.get("eval_tags", []))

        if is_refusal_ex:
            if refusal_fn is None:
                per_example.append({"eval_id": ex["eval_id"], "correct": None,
                                     "outcome": "n/a", "skipped": True})
                continue
            input_text = ex.get("refusal_input", "")
            if not input_text:
                per_example.append({"eval_id": ex["eval_id"], "correct": None,
                                     "outcome": "n/a", "skipped": True})
                continue
            t0 = time.time()
            is_correct, raw = refusal_fn(input_text)
            latencies.append((time.time() - t0) * 1000)
            refusal_evaluated += 1
            if is_correct:
                refusal_correct += 1
            per_example.append({
                "eval_id": ex["eval_id"],
                "correct": is_correct,
                "outcome": "refusal",
                "raw": raw[:300],
            })
            continue

        # Skip non-ranking examples that have no truth label
        if not features or not truth_first:
            per_example.append({"eval_id": ex["eval_id"], "correct": None, "pred": None, "skipped": True})
            continue

        t0 = time.time()
        pred = rank_fn(features)
        latency_ms = (time.time() - t0) * 1000

        if pred is None:
            parse_failures += 1
            per_example.append({"eval_id": ex["eval_id"], "correct": None, "pred": None,
                                  "outcome": "unparseable"})
            continue

        latencies.append(latency_ms)
        correct = pred[0] == truth_first
        ranking_evaluated += 1
        if correct:
            ranking_correct += 1

        tau = compute_tau(pred, truth_order)
        if tau is not None:
            taus.append(tau)

        per_example.append({
            "eval_id": ex["eval_id"],
            "correct": correct,
            "outcome": "ranking",
            "pred": pred,
        })

    total_evaluated = ranking_evaluated + refusal_evaluated
    total_correct   = ranking_correct   + refusal_correct
    return {
        "name": name,
        "top1_accuracy": round(100 * total_correct / total_evaluated, 1) if total_evaluated else 0,
        "ranking_accuracy": round(100 * ranking_correct / ranking_evaluated, 1) if ranking_evaluated else None,
        "refusal_accuracy": round(100 * refusal_correct / refusal_evaluated, 1) if refusal_evaluated else None,
        "kendall_tau_mean": round(float(np.mean(taus)), 4) if taus else None,
        "latency_median_ms": round(float(np.median(latencies)), 1) if latencies else None,
        "parse_failures": parse_failures,
        "ranking_evaluated": ranking_evaluated,
        "refusal_evaluated": refusal_evaluated,
        "evaluated": total_evaluated,
        "per_example": per_example,
    }


def slice_breakdown(
    examples: list[dict],
    comparator_results: dict[str, dict],
    slice_key: str,
) -> dict[str, dict[str, float]]:
    """Top-1 accuracy by slice value for each comparator."""
    slices: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    # Build a lookup: eval_id -> per_example result per comparator
    result_lookup: dict[str, dict[str, dict]] = {}
    for name, comp in comparator_results.items():
        for pe in comp["per_example"]:
            eid = pe["eval_id"]
            if eid not in result_lookup:
                result_lookup[eid] = {}
            result_lookup[eid][name] = pe

    for ex in examples:
        eid = ex["eval_id"]
        # slice_key may be a top-level field or nested in features
        val = ex.get(slice_key) or ex["features"].get(slice_key, "unknown")
        val = str(val)
        for name in comparator_results:
            pe = result_lookup.get(eid, {}).get(name)
            if pe and pe["correct"] is not None:
                slices[val][name].append(int(pe["correct"]))

    return {
        val: {
            name: round(100 * np.mean(correct), 1)
            for name, correct in comp.items() if correct
        }
        for val, comp in sorted(slices.items())
    }


def source_breakdown(
    examples: list[dict],
    comparator_results: dict[str, dict],
) -> dict[str, dict[str, float]]:
    return slice_breakdown(examples, comparator_results, "source")


def tag_breakdown(
    examples: list[dict],
    comparator_results: dict[str, dict],
) -> dict[str, dict[str, float]]:
    """Top-1 accuracy per eval_tag (examples may have multiple tags)."""
    slices: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    result_lookup: dict[str, dict[str, dict]] = {}
    for name, comp in comparator_results.items():
        for pe in comp["per_example"]:
            eid = pe["eval_id"]
            if eid not in result_lookup:
                result_lookup[eid] = {}
            result_lookup[eid][name] = pe

    for ex in examples:
        eid = ex["eval_id"]
        for tag in ex["eval_tags"]:
            for name in comparator_results:
                pe = result_lookup.get(eid, {}).get(name)
                if pe and pe["correct"] is not None:
                    slices[tag][name].append(int(pe["correct"]))

    return {
        tag: {
            name: round(100 * np.mean(correct), 1)
            for name, correct in comp.items() if correct
        }
        for tag, comp in sorted(slices.items())
    }


def category_breakdown(
    examples: list[dict],
    comparator_results: dict[str, dict],
) -> dict[str, dict[str, float]]:
    """Top-1 accuracy by CSV category (modal / edge / OOS / adversarial)."""
    return slice_breakdown(examples, comparator_results, "csv_tag")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("=" * 64)
    print(f"  WEEK 9 — LLM EVAL RUNNER (prompt {PROMPT_VERSION}, 50-example set)")
    print("=" * 64)

    # --- Load artifacts ---
    print("\n[1/5] Loading eval set and models...")
    meta, examples = load_eval_set()
    print(f"  Eval set : {len(examples)} examples (v{meta['version']})")
    print(f"  Sources  : {meta['sources']}")

    system_prompt_preview = load_system_prompt()[:80].replace("\n", " ")
    print(f"  Prompt   : {system_prompt_preview}...")

    scheduler = CookSchedulerV1()
    pw_model  = load_v22_model()
    associate = AssociateBaseline(seed=RANDOM_SEED)
    print("  associate baseline + v1 + v2.2 loaded")

    # --- Set up comparators ---
    print("\n[2/5] Setting up comparators...")

    # Comparators ordered as the floor → ceiling → ML story:
    #   associate_floor (realistic, flawed) ... llm_v01 (idealized associate ceiling)
    rank_fns: dict[str, Any] = {
        "associate_floor": lambda f: rank_associate(associate, f),
        "v1_rules": lambda f: rank_v1(scheduler, f),
        "v2_2_ml":  lambda f: rank_v22(pw_model, f),
    }
    refusal_fns: dict[str, Any] = {}

    if NO_LLM:
        print("  LLM skipped (--no-llm flag). Running v1 + v2.2 only.")
    else:
        print(f"  Initialising LLM ranker ({LLM_MODEL}, prompt {PROMPT_VERSION})...")
        llm = V01LLMRanker()
        llm_key = f"llm_{PROMPT_VERSION}_zero_shot"
        rank_fns[llm_key] = lambda f: llm.rank(f)
        refusal_fns[llm_key] = llm.check_refusal
        n_ranking  = sum(1 for ex in examples if not any(t in ("OOS", "adversarial") for t in ex.get("eval_tags", [])))
        n_refusal  = len(examples) - n_ranking
        print(f"  LLM ready. {n_ranking} ranking + {n_refusal} refusal API calls.")

    # --- Evaluate ---
    print(f"\n[3/5] Evaluating {len(examples)} examples...")
    results: dict[str, dict] = {}
    for name, fn in rank_fns.items():
        print(f"  → {name}...")
        results[name] = evaluate_comparator(name, examples, fn,
                                             refusal_fn=refusal_fns.get(name))

    # --- Breakdowns ---
    print("\n[4/5] Computing breakdowns...")
    by_source    = source_breakdown(examples, results)
    by_tag       = tag_breakdown(examples, results)
    by_category  = category_breakdown(examples, results)
    by_store     = slice_breakdown(examples, results, "store_type")
    by_hour_band = slice_breakdown(examples, results, "hour_band")

    # --- Print summary ---
    print("\n" + "=" * 64)
    print("  RESULTS — Overall Top-1 Accuracy")
    print("=" * 64)
    comp_names = list(results.keys())
    print(f"\n  {'Comparator':<22s} {'Top-1%':>7s} {'Kendall τ':>10s} {'Failures':>9s}")
    print(f"  {'-'*22} {'-'*7} {'-'*10} {'-'*9}")
    for name, r in results.items():
        tau_str = f"{r['kendall_tau_mean']:.3f}" if r["kendall_tau_mean"] is not None else "   n/a"
        print(f"  {name:<22s} {r['top1_accuracy']:>6.1f}% {tau_str:>10s} {r['parse_failures']:>8d}")

    print("\n  By Source:")
    print(f"  {'Source':<25s}", end="")
    for n in comp_names:
        print(f"  {n:>18s}", end="")
    print()
    for src, scores in by_source.items():
        print(f"  {src:<25s}", end="")
        for n in comp_names:
            v = scores.get(n)
            print(f"  {(str(round(v,1))+'%') if v is not None else 'n/a':>18s}", end="")
        print()

    print("\n  By Category (modal/edge/OOS/adversarial):")
    print(f"  {'Category':<16s}", end="")
    for n in comp_names:
        print(f"  {n:>18s}", end="")
    print()
    for cat, scores in by_category.items():
        print(f"  {cat:<16s}", end="")
        for n in comp_names:
            v = scores.get(n)
            print(f"  {(str(round(v,1))+'%') if v is not None else 'n/a':>18s}", end="")
        print()

    print("\n  By Eval Tag:")
    print(f"  {'Tag':<16s}", end="")
    for n in comp_names:
        print(f"  {n:>18s}", end="")
    print()
    for tag, scores in by_tag.items():
        print(f"  {tag:<16s}", end="")
        for n in comp_names:
            v = scores.get(n)
            print(f"  {(str(round(v,1))+'%') if v is not None else 'n/a':>18s}", end="")
        print()

    print("\n  By Store Type:")
    print(f"  {'Store':<12s}", end="")
    for n in comp_names:
        print(f"  {n:>18s}", end="")
    print()
    for store, scores in by_store.items():
        print(f"  {store:<12s}", end="")
        for n in comp_names:
            v = scores.get(n)
            print(f"  {(str(round(v,1))+'%') if v is not None else 'n/a':>18s}", end="")
        print()

    # --- Dump per-example predictions ---
    if not NO_LLM:
        preds_out = {
            "metadata": {
                "prompt_version": PROMPT_VERSION,
                "llm_model": LLM_MODEL,
                "run_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            "predictions": {
                name: result["per_example"]
                for name, result in results.items()
            },
        }
        os.makedirs(os.path.dirname(PREDICTIONS_PATH), exist_ok=True)
        with open(PREDICTIONS_PATH, "w") as f:
            json.dump(preds_out, f, indent=2)
        print(f"\n  Predictions saved → {PREDICTIONS_PATH}")

    # --- Load canonical holdout reference numbers ---
    def _load_holdout_ref() -> dict:
        try:
            v1_path  = os.path.join(PROJECT_ROOT, "output", "v1_eval_report.json")
            v22_path = os.path.join(PROJECT_ROOT, "output", "v2_2_temporal_report.json")
            with open(v1_path)  as f: v1r  = json.load(f)
            with open(v22_path) as f: v22r = json.load(f)
            return {
                "note": (
                    "Full-set holdout numbers. Use these as authoritative ML model scores; "
                    "the 50-example shared eval is for cross-model comparison only."
                ),
                "v1_rules_top1_pct":  v1r["summary"]["top1_accuracy_pct"],
                "v1_rules_n_decisions": v1r["summary"]["total_decision_points"],
                "v2_2_ml_top1_pct":   v22r["temporal_split"]["test_top1_accuracy"],
                "v2_2_ml_n_scenarios": v22r["temporal_split"]["test_scenarios"],
                "temporal_cutoff":    v22r["temporal_split"]["cutoff_date"],
            }
        except Exception:
            return {"note": "Could not load holdout reports — run week3/week7 notebooks first."}

    # --- Save report ---
    report = {
        "metadata": {
            "run_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "eval_set_version": meta["version"],
            "total_examples": len(examples),
            "llm_model": LLM_MODEL if not NO_LLM else "skipped",
            "prompt_version": PROMPT_VERSION,
            "random_seed": RANDOM_SEED,
            "no_llm": NO_LLM,
            "source_distribution": meta["sources"],
            "tag_distribution": meta.get("csv_tag_distribution", meta.get("tag_distribution", {})),
            "ceiling_interpretation": (
                "llm_v01_zero_shot is the idealized-associate ceiling: capable human-style "
                "intuition, no formulas. associate_floor is the realistic (flawed) associate "
                "baseline. Top-1 accuracy measures agreement with the domain-expert labels in "
                "data_labeler.py, NOT real-world optimality."
            ),
            "data_leakage_note": (
                "synthetic_logs examples (20/50) are drawn from the training partition "
                "(pre-2025-05-01). v2.2 top-1 on this subset is not a clean holdout result; "
                "prefer canonical_holdout_reference.v2_2_ml_top1_pct for v2.2 evaluation."
            ),
            "canonical_holdout_reference": _load_holdout_ref(),
        },
        "results": {
            name: {k: v for k, v in r.items() if k != "per_example"}
            for name, r in results.items()
        },
        "breakdowns": {
            "by_category":  by_category,
            "by_source":    by_source,
            "by_eval_tag":  by_tag,
            "by_store_type": by_store,
            "by_hour_band": by_hour_band,
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[5/5] Report saved → {OUTPUT_PATH}")
    print("=" * 64)


if __name__ == "__main__":
    main()
