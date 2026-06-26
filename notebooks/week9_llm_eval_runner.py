"""LLM Evaluation Runner — multi-eval-set, multi-prompt, JTBD-aligned metrics.

Supports three eval sets:
  v0.1  50-example legacy set (5-item universe, data/llm_eval_set_v0.1.json)
  v0.2  53-example 28-item set (data/llm_eval_set_v0.2.json)
  v0.3  32-example JTBD plain-language set (data/llm_eval_set_v0.3.json)

Metrics (v0.3 JTBD-aligned, also computed for v0.1/v0.2 where ground truth allows):
  cook_now_accuracy       — did the model pick the right first item?
  cook_now_set_recall     — fraction of urgent items that landed in the top-k
  must_precede_violations — A-before-B safety violations (goal = 0)
  refusal_accuracy        — OOS / adversarial examples (trust dimension)
  kendall_tau_mean        — full-order quality (secondary)

Usage:
  export ANTHROPIC_API_KEY=your_key

  # v0.1 / v0.2 sets (data format, legacy metrics)
  python notebooks/week9_llm_eval_runner.py --prompt-version=v0.2
  python notebooks/week9_llm_eval_runner.py --eval-set=v0.2 --prompt-version=v0.2

  # v0.3 JTBD plain-language set
  python notebooks/week9_llm_eval_runner.py --eval-set=v0.3 --prompt-version=v0.3
  python notebooks/week9_llm_eval_runner.py --eval-set=v0.3 --no-llm  # ML baselines only
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

MODEL_PATH      = os.path.join(PROJECT_ROOT, "models", "v2_2_pairwise_temporal.pkl")
LLM_MODEL       = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
RANDOM_SEED     = 42

# Eval set version from --eval-set=vX.Y CLI arg (default: v0.1)
EVAL_SET_VERSION = next(
    (a.split("=", 1)[1] for a in sys.argv if a.startswith("--eval-set=")),
    "v0.1",
)
EVAL_SET_PATH = os.path.join(PROJECT_ROOT, "data", f"llm_eval_set_{EVAL_SET_VERSION}.json")

# Prompt version from --prompt-version=vX.Y CLI arg (default: v0.1)
PROMPT_VERSION  = next(
    (a.split("=", 1)[1] for a in sys.argv if a.startswith("--prompt-version=")),
    "v0.1",
)
PROMPT_PATH      = os.path.join(PROJECT_ROOT, "prompts", f"{PROMPT_VERSION}_system_prompt.md")

# Output filenames encode both prompt and eval-set versions to avoid collisions
_eval_suffix     = "" if EVAL_SET_VERSION == "v0.1" else f"_{EVAL_SET_VERSION.replace('.', '_')}"
OUTPUT_PATH      = os.path.join(PROJECT_ROOT, "output", f"llm_eval_{PROMPT_VERSION}{_eval_suffix}_report.json")
PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "output", f"llm_eval_{PROMPT_VERSION}{_eval_suffix}_predictions.json")

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
    if not os.path.exists(EVAL_SET_PATH):
        raise FileNotFoundError(
            f"Eval set not found: {EVAL_SET_PATH}. "
            f"Run: python scripts/build_eval_set_{EVAL_SET_VERSION.replace('.', '_')}.py"
        )
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


def try_load_v22_model() -> PairwiseModelTrainer | None:
    """Load v2.2 model; return None if pickle is missing or corrupt."""
    if not os.path.exists(MODEL_PATH):
        print(f"  WARNING: v2.2 model not found at {MODEL_PATH}; skipping v2_2_ml")
        return None
    try:
        return load_v22_model()
    except Exception as exc:
        print(f"  WARNING: v2.2 model unavailable ({type(exc).__name__}: {exc}); skipping v2_2_ml")
        return None


# ===========================================================================
# Rankers
# ===========================================================================

def present_items(features: dict) -> list[str]:
    found = {k[: -len("_forecast_demand")] for k in features if k.endswith("_forecast_demand")}
    ordered = [item for item in OVEN_ITEMS if item in found]
    extras = sorted(found - set(ordered))
    return ordered + extras


def _extract_features(ex_or_features: dict) -> dict:
    """Accept either a full example dict or a flat features dict."""
    if "features" in ex_or_features and isinstance(ex_or_features["features"], dict):
        return ex_or_features["features"]
    return ex_or_features


def rank_v1(scheduler: CookSchedulerV1, ex_or_features: dict) -> list[str] | None:
    features = _extract_features(ex_or_features)
    if not present_items(features):
        return None
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


def rank_v22(trainer: PairwiseModelTrainer, ex_or_features: dict) -> list[str] | None:
    features = _extract_features(ex_or_features)
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


def rank_associate(associate: AssociateBaseline, ex_or_features: dict) -> list[str] | None:
    """Realistic associate floor (habit/expiration/random mix) on the same set."""
    features = _extract_features(ex_or_features)
    events = build_oven_events(features)
    return associate.rank_items(events) if events else None


def build_llm_user_prompt_from_features(features: dict) -> str:
    """Format a numeric decision scenario into the user-turn message (v0.1/v0.2 format).

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


def build_llm_user_prompt(example: dict) -> str:
    """Build the LLM user message for any eval-set version.

    For v0.3 plain-language examples: sends scenario_text verbatim, then
    appends the canonical item IDs so the LLM knows what strings to put in
    ranked_queue.

    For v0.1/v0.2: falls back to the numeric feature-table format.
    """
    scenario_text = example.get("scenario_text")
    features = example.get("features") or {}

    if scenario_text:
        items = present_items(features)
        if items:
            id_hint = (
                "\n\nItem IDs to use in your ranked_queue (use these exact strings):\n  "
                + ", ".join(items)
            )
        else:
            id_hint = ""
        return scenario_text + id_hint
    return build_llm_user_prompt_from_features(features)


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

    def rank(self, ex_or_features: dict) -> list[str] | None:
        """Rank items for an example.

        Accepts either a full example dict (eval_id, features, scenario_text…)
        or a bare features dict (backward compat with v0.1/v0.2 callers).
        For v0.3 plain-language examples, scenario_text is sent verbatim.
        """
        features = _extract_features(ex_or_features)
        present = present_items(features)
        # Full example dict → use scenario_text if available
        user_msg = build_llm_user_prompt(ex_or_features)
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=256,
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
# Evaluation helpers — JTBD-aligned metrics
# ===========================================================================

def compute_cook_now_accuracy(pred: list[str], cook_now: str | None, cook_now_set: list[str]) -> bool | None:
    """True if pred[0] is in cook_now_set (or equals cook_now when set is empty)."""
    if not pred:
        return None
    urgent = cook_now_set if cook_now_set else ([cook_now] if cook_now else [])
    if not urgent:
        return None
    return pred[0] in urgent


def compute_cook_now_set_recall(pred: list[str], cook_now_set: list[str]) -> float | None:
    """Fraction of urgent items that appear in pred[:k] where k = len(cook_now_set)."""
    if not pred or not cook_now_set:
        return None
    k = len(cook_now_set)
    top_k = set(pred[:k])
    return sum(1 for item in cook_now_set if item in top_k) / k


def compute_must_precede_violations(pred: list[str], must_precede: list[list[str]]) -> int:
    """Count [A, B] pairs where A appears *after* B in pred (safety violation)."""
    if not pred or not must_precede:
        return 0
    pos = {item: i for i, item in enumerate(pred)}
    violations = 0
    for pair in must_precede:
        if len(pair) != 2:
            continue
        a, b = pair
        if a in pos and b in pos and pos[a] > pos[b]:
            violations += 1
    return violations


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

    # JTBD metrics
    cook_now_correct_list: list[bool] = []
    set_recall_list: list[float] = []
    must_precede_total_violations = 0

    per_example: list[dict] = []

    for ex in examples:
        features    = _extract_features(ex)
        truth_order = ex.get("optimal_order") or []
        truth_first = ex.get("optimal_first_item")
        cook_now      = ex.get("cook_now") or truth_first
        cook_now_set  = ex.get("cook_now_set") or ([cook_now] if cook_now else [])
        must_precede  = ex.get("must_precede") or []
        is_refusal_ex = any(t in ("OOS", "adversarial") for t in ex.get("eval_tags", []))

        if is_refusal_ex:
            if refusal_fn is None:
                per_example.append({"eval_id": ex["eval_id"], "correct": None,
                                     "outcome": "n/a", "skipped": True})
                continue
            input_text = ex.get("refusal_input") or ex.get("scenario_text", "")
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

        # Skip non-ranking examples with no truth label and no numeric features
        if not features or not cook_now:
            per_example.append({"eval_id": ex["eval_id"], "correct": None, "pred": None, "skipped": True})
            continue

        t0 = time.time()
        # Pass the full example dict so LLM rankers can access scenario_text
        pred = rank_fn(ex)
        latency_ms = (time.time() - t0) * 1000

        if pred is None:
            parse_failures += 1
            per_example.append({"eval_id": ex["eval_id"], "correct": None, "pred": None,
                                  "outcome": "unparseable"})
            continue

        latencies.append(latency_ms)

        # JTBD headline: cook_now_accuracy
        cn_correct = compute_cook_now_accuracy(pred, cook_now, cook_now_set)
        if cn_correct is not None:
            cook_now_correct_list.append(cn_correct)

        # cook_now_set recall
        sr = compute_cook_now_set_recall(pred, cook_now_set)
        if sr is not None:
            set_recall_list.append(sr)

        # must_precede violations
        mp_viol = compute_must_precede_violations(pred, must_precede)
        must_precede_total_violations += mp_viol

        # Classic top-1 vs formula optimal (secondary for v0.3)
        correct = pred[0] == truth_first if truth_first else cn_correct
        ranking_evaluated += 1
        if correct:
            ranking_correct += 1

        tau = compute_tau(pred, truth_order)
        if tau is not None:
            taus.append(tau)

        per_example.append({
            "eval_id":                ex["eval_id"],
            "correct":                correct,
            "cook_now_correct":       cn_correct,
            "cook_now_set_recall":    sr,
            "must_precede_violations": mp_viol,
            "outcome":                "ranking",
            "pred":                   pred,
        })

    total_evaluated = ranking_evaluated + refusal_evaluated
    total_correct   = ranking_correct   + refusal_correct
    cn_n            = len(cook_now_correct_list)
    return {
        "name":                       name,
        "top1_accuracy":              round(100 * total_correct / total_evaluated, 1) if total_evaluated else 0,
        "ranking_accuracy":           round(100 * ranking_correct / ranking_evaluated, 1) if ranking_evaluated else None,
        "refusal_accuracy":           round(100 * refusal_correct / refusal_evaluated, 1) if refusal_evaluated else None,
        # JTBD-aligned metrics
        "cook_now_accuracy":          round(100 * sum(cook_now_correct_list) / cn_n, 1) if cn_n else None,
        "cook_now_set_recall":        round(float(np.mean(set_recall_list)), 4) if set_recall_list else None,
        "must_precede_violations":    must_precede_total_violations,
        "must_precede_violation_rate": round(must_precede_total_violations / ranking_evaluated, 4) if ranking_evaluated else None,
        # Standard metrics
        "kendall_tau_mean":           round(float(np.mean(taus)), 4) if taus else None,
        "latency_median_ms":          round(float(np.median(latencies)), 1) if latencies else None,
        "parse_failures":             parse_failures,
        "ranking_evaluated":          ranking_evaluated,
        "refusal_evaluated":          refusal_evaluated,
        "evaluated":                  total_evaluated,
        "per_example":                per_example,
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
    print(f"  LLM EVAL RUNNER  eval-set={EVAL_SET_VERSION}  prompt={PROMPT_VERSION}")
    print("=" * 64)

    # --- Load artifacts ---
    print("\n[1/5] Loading eval set and models...")
    meta, examples = load_eval_set()
    print(f"  Eval set  : {len(examples)} examples ({meta['version']})")
    print(f"  Eval path : {EVAL_SET_PATH}")
    if "sources" in meta:
        print(f"  Sources   : {meta['sources']}")

    system_prompt_preview = load_system_prompt()[:80].replace("\n", " ")
    print(f"  Prompt    : {system_prompt_preview}...")

    scheduler = CookSchedulerV1()
    pw_model  = try_load_v22_model()
    associate = AssociateBaseline(seed=RANDOM_SEED)
    print("  associate baseline + v1 loaded" + (" + v2.2 loaded" if pw_model else " (v2.2 skipped)"))

    # --- Set up comparators ---
    print("\n[2/5] Setting up comparators...")

    rank_fns: dict[str, Any] = {
        "associate_floor": lambda ex: rank_associate(associate, ex),
        "v1_rules":        lambda ex: rank_v1(scheduler, ex),
    }
    if pw_model is not None:
        rank_fns["v2_2_ml"] = lambda ex: rank_v22(pw_model, ex)

    refusal_fns: dict[str, Any] = {}

    if NO_LLM:
        print("  LLM skipped (--no-llm flag). Running ML baselines only.")
    else:
        print(f"  Initialising LLM ranker ({LLM_MODEL}, prompt {PROMPT_VERSION})...")
        llm = V01LLMRanker()
        llm_key = f"llm_{PROMPT_VERSION}_zero_shot"
        rank_fns[llm_key] = lambda ex: llm.rank(ex)
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
    print("\n" + "=" * 72)
    print("  RESULTS — JTBD Metrics")
    print("=" * 72)
    comp_names = list(results.keys())
    print(f"\n  {'Comparator':<22s} {'CookNow%':>9s} {'SetRecall':>10s} {'MPViol':>7s} {'τ':>8s} {'Fail':>5s}")
    print(f"  {'-'*22} {'-'*9} {'-'*10} {'-'*7} {'-'*8} {'-'*5}")
    for name, r in results.items():
        cn  = f"{r['cook_now_accuracy']:.1f}%" if r.get("cook_now_accuracy") is not None else "   n/a"
        sr  = f"{r['cook_now_set_recall']:.3f}"  if r.get("cook_now_set_recall") is not None else "    n/a"
        mpv = str(r.get("must_precede_violations", 0))
        tau = f"{r['kendall_tau_mean']:.3f}" if r.get("kendall_tau_mean") is not None else "    n/a"
        fail = str(r.get("parse_failures", 0))
        print(f"  {name:<22s} {cn:>9s} {sr:>10s} {mpv:>7s} {tau:>8s} {fail:>5s}")

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
    # Always save ML predictions; when --no-llm, merge LLM predictions from existing file.
    existing_llm_preds: dict = {}
    ml_comparator_names = {k for k in rank_fns if not k.startswith("llm_")}
    if NO_LLM and os.path.exists(PREDICTIONS_PATH):
        try:
            with open(PREDICTIONS_PATH) as f:
                existing = json.load(f)
            for name, preds in existing.get("predictions", {}).items():
                if name not in ml_comparator_names:
                    existing_llm_preds[name] = preds
        except Exception:
            pass

    ml_predictions = {
        name: result["per_example"]
        for name, result in results.items()
    }
    preds_out = {
        "metadata": {
            "prompt_version": PROMPT_VERSION,
            "llm_model": LLM_MODEL if not NO_LLM else "skipped (ml-only re-run)",
            "run_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "predictions": {**ml_predictions, **existing_llm_preds},
    }
    os.makedirs(os.path.dirname(PREDICTIONS_PATH), exist_ok=True)
    with open(PREDICTIONS_PATH, "w") as f:
        json.dump(preds_out, f, indent=2)
    print(f"\n  Predictions saved → {PREDICTIONS_PATH}")

    # --- Load canonical holdout reference numbers ---
    def _load_holdout_ref() -> dict:
        try:
            lab_path = os.path.join(PROJECT_ROOT, "output", "labeling_report.json")
            v22_path = os.path.join(PROJECT_ROOT, "output", "v2_2_temporal_report.json")
            with open(lab_path) as f: labr = json.load(f)
            with open(v22_path) as f: v22r = json.load(f)
            return {
                "note": (
                    "Accuracy vs composite priority labels (data_labeler.py). "
                    "v1 = label agreement on full dataset; v2.2 = temporal holdout test (>=2025-05-01). "
                    "See output/v1_eval_report.json for v1 write-off optimality."
                ),
                "v1_rules_top1_pct":   labr["v1_agreement_pct"],
                "v1_rules_n_decisions": labr["total_labeled_scenarios"],
                "v2_2_ml_top1_pct":    v22r["temporal_split"]["test_top1_accuracy"],
                "v2_2_ml_n_scenarios":  v22r["temporal_split"]["test_scenarios"],
                "temporal_cutoff":     v22r["temporal_split"]["cutoff_date"],
            }
        except Exception:
            return {"note": "Could not load holdout reports — run week3/week7 notebooks first."}

    # --- Save report ---
    report = {
        "metadata": {
            "run_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "eval_set_version": meta["version"],
            "eval_set_path": EVAL_SET_PATH,
            "total_examples": len(examples),
            "llm_model": LLM_MODEL if not NO_LLM else "skipped",
            "prompt_version": PROMPT_VERSION,
            "random_seed": RANDOM_SEED,
            "no_llm": NO_LLM,
            "source_distribution": meta.get("sources", {}),
            "tag_distribution": meta.get("csv_tag_distribution", {}),
            "jtbd_metrics": (
                "cook_now_accuracy: did model pick the right first item? "
                "cook_now_set_recall: fraction of urgent items in top-k. "
                "must_precede_violations: safety constraint violations (goal=0). "
                "refusal_accuracy: OOS/adversarial correctness. "
                "kendall_tau: full-order quality (secondary)."
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
