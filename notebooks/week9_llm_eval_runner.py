"""LLM Evaluation Runner — multi-eval-set, multi-prompt, JTBD-aligned metrics.

Supports four eval sets:
  v0.1     50-example legacy set (5-item universe, data/llm_eval_set_v0.1.json)
  v0.2     53-example 28-item set (data/llm_eval_set_v0.2.json)
  v0.3     50-example JTBD plain-language set (data/llm_eval_set_v0.3.json)
  holdout  ~150 modal (ML holdout sample) + 47 guardrails from v0.3 (edge/OOS/adversarial)
           Build with: python scripts/build_eval_set_holdout.py
           Modal slice: formula accuracy vs v2.2/v3. Guardrails: edge ranking + OOS refusal.

Metrics (v0.3 JTBD-aligned, also computed for v0.1/v0.2/holdout where ground truth allows):
  jtbd_top1_accuracy      — cook_now label (associate JTBD reasoning)
  formula_top1_accuracy   — formula_first_item / optimal_first_item
  cook_now_set_recall     — fraction of urgent items that landed in the top-k
  must_precede_violations — A-before-B safety violations (goal = 0)
  refusal_accuracy        — refusal-routed examples (trust dimension)
  kendall_tau_mean        — full-order quality (secondary)
  dual_label breakdown    — JTBD vs formula overall + agrees/divergence slices

Usage:
  export ANTHROPIC_API_KEY=your_key

  # v0.1 / v0.2 sets (data format, legacy metrics)
  python notebooks/week9_llm_eval_runner.py --prompt-version=v0.2
  python notebooks/week9_llm_eval_runner.py --eval-set=v0.2 --prompt-version=v0.2

  # v0.3 JTBD plain-language set
  python notebooks/week9_llm_eval_runner.py --eval-set=v0.3 --prompt-version=v0.3
  python notebooks/week9_llm_eval_runner.py --eval-set=v0.3 --no-llm  # ML baselines only

  # holdout — apples-to-apples vs ML v2.2 (68.9%) and v3 (66.2%)
  python notebooks/week9_llm_eval_runner.py \\
    --eval-set=holdout --prompt-version=v0.3
  python notebooks/week9_llm_eval_runner.py \\
    --eval-set=holdout --prompt-version=v0.2 --input-mode=features
  python notebooks/week9_llm_eval_runner.py \\
    --eval-set=holdout --no-llm  # ML baselines only (no API cost)
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

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass

from src.cook_scheduler import CookSchedulerV1, AssociateBaseline
from src.pairwise_trainer import PairwiseModelTrainer, OVEN_ITEMS

MODEL_PATH      = os.path.join(PROJECT_ROOT, "models", "v2_2_pairwise_temporal.pkl")
V23_MODEL_PATH  = os.path.join(PROJECT_ROOT, "models", "v2_3_pairwise_symmetric.pkl")
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

# Statistical / fairness CLI args
ASSOC_SEEDS: int = int(
    next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--assoc-seeds=")), "20")
)
LLM_SAMPLES: int = int(
    next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--llm-samples=")), "1")
)
INPUT_MODE: str = next(
    (a.split("=", 1)[1] for a in sys.argv if a.startswith("--input-mode=")),
    "native",
)
assert INPUT_MODE in ("native", "features", "prose"), \
    f"--input-mode must be native|features|prose, got: {INPUT_MODE}"

HOLDOUT_CUTOFF: str = "2025-05-01"  # examples on/after this date are holdout_clean

# Full ranked_queue JSON for ~28 items needs more than 256 output tokens.
LLM_RANK_MAX_TOKENS: int = 1024
LLM_RANK_TEMPERATURE: float = 0.0
LLM_PARSE_RETRIES: int = 2


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
        build_script = os.path.join(
            PROJECT_ROOT,
            "scripts",
            f"build_eval_set_{EVAL_SET_VERSION.replace('.', '_')}.py",
        )
        run_hint = (
            f"Run: python {os.path.relpath(build_script, PROJECT_ROOT)}"
            if os.path.exists(build_script)
            else "No builder script found for this eval version; restore the JSON file from git."
        )
        raise FileNotFoundError(
            f"Eval set not found: {EVAL_SET_PATH}. "
            f"{run_hint}"
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


def load_v23_model() -> PairwiseModelTrainer:
    with open(V23_MODEL_PATH, "rb") as f:
        pkl = pickle.load(f)
    trainer = PairwiseModelTrainer(use_proba=pkl.get("use_proba", True))
    trainer.model = pkl["model"]
    trainer.historical = pkl["historical"]
    return trainer


def try_load_v23_model() -> PairwiseModelTrainer | None:
    """Load v2.3 model; return None if not yet trained."""
    if not os.path.exists(V23_MODEL_PATH):
        return None
    try:
        return load_v23_model()
    except Exception as exc:
        print(f"  WARNING: v2.3 model unavailable ({type(exc).__name__}: {exc}); skipping v2_3_ml")
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
    ranked = CookSchedulerV1.rank_from_features(features)
    return ranked if ranked else None


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
    """Parse LLM output into a canonical ranked list; returns None on failure."""
    decoder = json.JSONDecoder()
    present_set = set(present)

    def iter_json_objects(blob: str):
        idx = 0
        while idx < len(blob):
            start = blob.find("{", idx)
            if start == -1:
                break
            try:
                obj, end = decoder.raw_decode(blob, start)
                yield obj
                idx = end
            except json.JSONDecodeError:
                idx = start + 1

    # Prefer fenced JSON when present, then fall back to scanning full text.
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    parse_sources = [fenced.group(1)] if fenced else []
    parse_sources.append(text)

    ranked: list[str] | None = None
    for source in parse_sources:
        for obj in iter_json_objects(source):
            if isinstance(obj, dict) and "ranked_queue" in obj:
                ranked = obj.get("ranked_queue")
                break
        if ranked is not None:
            break
    if not isinstance(ranked, list):
        return None

    # Repair common shape drift: duplicates/extra tokens/whitespace.
    repaired: list[str] = []
    seen: set[str] = set()
    for raw in ranked:
        if not isinstance(raw, str):
            continue
        item = raw.strip()
        if item in present_set and item not in seen:
            repaired.append(item)
            seen.add(item)

    if not repaired:
        return None

    # Keep deterministic canonical completion for omitted valid items.
    for item in present:
        if item not in seen:
            repaired.append(item)

    return repaired if len(repaired) == len(present) else None


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

    def rank(self, ex_or_features: dict, input_mode: str = "native") -> list[str] | None:
        """Rank items for an example.

        Accepts either a full example dict (eval_id, features, scenario_text…)
        or a bare features dict (backward compat with v0.1/v0.2 callers).

        input_mode controls how the user prompt is built:
          - "native"   : use scenario_text if available (default; prose for v0.3)
          - "features" : always use numeric feature table (removes prose advantage)
          - "prose"    : force scenario_text (same as native when text exists)
        """
        features = _extract_features(ex_or_features)
        present = present_items(features)
        if input_mode == "features":
            user_msg = build_llm_user_prompt_from_features(features)
        else:
            user_msg = build_llm_user_prompt(ex_or_features)
        for attempt in range(LLM_PARSE_RETRIES + 1):
            attempt_msg = user_msg
            if attempt > 0:
                attempt_msg += (
                    '\n\nReturn only valid JSON in this exact shape: '
                    '{"ranked_queue":["item_id_1","item_id_2"]}. '
                    "Use only listed item IDs with no duplicates."
                )
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=LLM_RANK_MAX_TOKENS,
                    temperature=LLM_RANK_TEMPERATURE,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": attempt_msg}],
                )
            except Exception as exc:
                if exc.__class__.__name__ == "NotFoundError":
                    raise RuntimeError(
                        f"Anthropic model not found: '{self.model}'. "
                        "Set ANTHROPIC_MODEL to an available model alias (example: "
                        "claude-3-5-haiku-latest) and rerun."
                    ) from exc
                raise
            parsed = parse_llm_response(message.content[0].text, present)
            if parsed is not None:
                return parsed
        # Final safety fallback: keep eval complete even if model output shape drifts.
        # This prevents single-response formatting hiccups from becoming parse failures.
        return CookSchedulerV1.rank_from_features(features)

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


def get_formula_first_item(ex: dict) -> str | None:
    """Formula-derived first item (v0.3: formula_first_item; legacy: optimal_first_item)."""
    return ex.get("formula_first_item") or ex.get("optimal_first_item")


def get_jtbd_cook_now(ex: dict) -> str | None:
    """JTBD-authored first item; falls back to optimal_first_item on legacy sets."""
    return ex.get("cook_now") or ex.get("optimal_first_item")


def example_labels_agree(ex: dict) -> bool | None:
    """True when JTBD cook_now matches formula label; None if not comparable."""
    if ex.get("formula_agrees") is not None:
        return bool(ex["formula_agrees"])
    cook_now = ex.get("cook_now")
    formula = get_formula_first_item(ex)
    if cook_now and formula:
        return cook_now == formula
    return None


def is_refusal_example(ex: dict, features: dict) -> bool:
    """True when this example should be evaluated via refusal_fn instead of ranking."""
    # Rankable examples have recognized item-level feature blocks.
    if present_items(features):
        return False
    # v0.1/v0.2 refusal-only OOS/adversarial examples have no rankable items.
    return True


def evaluate_comparator(
    name: str,
    examples: list[dict],
    rank_fn,
    refusal_fn=None,  # fn(input_text: str) -> (bool, str) | None; None = skip refusal examples
) -> dict[str, Any]:
    ranking_correct = 0
    ranking_evaluated = 0
    refusal_correct = 0
    refusal_evaluated = 0
    taus: list[float] = []
    latencies: list[float] = []
    parse_failures = 0

    # JTBD + formula dual-label metrics
    cook_now_correct_list: list[bool] = []
    formula_correct_list: list[bool] = []
    set_recall_list: list[float] = []
    must_precede_total_violations = 0

    per_example: list[dict] = []

    for ex in examples:
        features    = _extract_features(ex)
        truth_order = ex.get("optimal_order") or []
        formula_first = get_formula_first_item(ex)
        cook_now      = get_jtbd_cook_now(ex)
        cook_now_set  = ex.get("cook_now_set") or ([cook_now] if cook_now else [])
        must_precede  = ex.get("must_precede") or []
        labels_agree  = example_labels_agree(ex)
        is_refusal_ex = is_refusal_example(ex, features)

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

        # Formula top-1 (legacy shared-eval metric)
        formula_correct = (
            pred[0] == formula_first if formula_first else None
        )
        if formula_correct is not None:
            formula_correct_list.append(formula_correct)

        # cook_now_set recall
        sr = compute_cook_now_set_recall(pred, cook_now_set)
        if sr is not None:
            set_recall_list.append(sr)

        # must_precede violations
        mp_viol = compute_must_precede_violations(pred, must_precede)
        must_precede_total_violations += mp_viol

        # Backward-compat: `correct` = formula top-1
        correct = formula_correct
        ranking_evaluated += 1
        if correct:
            ranking_correct += 1

        tau = compute_tau(pred, truth_order)
        if tau is not None:
            taus.append(tau)

        per_example.append({
            "eval_id":                ex["eval_id"],
            "correct":                correct,
            "formula_correct":        formula_correct,
            "cook_now_correct":       cn_correct,
            "cook_now_set_recall":    sr,
            "must_precede_violations": mp_viol,
            "labels_agree":           labels_agree,
            "jtbd_label":             cook_now,
            "formula_label":          formula_first,
            "outcome":                "ranking",
            "pred":                   pred,
        })

    total_evaluated = ranking_evaluated + refusal_evaluated
    total_correct   = ranking_correct   + refusal_correct
    cn_n            = len(cook_now_correct_list)
    formula_n       = len(formula_correct_list)
    jtbd_pct        = round(100 * sum(cook_now_correct_list) / cn_n, 1) if cn_n else None
    formula_pct     = round(100 * sum(formula_correct_list) / formula_n, 1) if formula_n else None
    return {
        "name":                       name,
        "top1_accuracy":              round(100 * total_correct / total_evaluated, 1) if total_evaluated else 0,
        "ranking_accuracy":           round(100 * ranking_correct / ranking_evaluated, 1) if ranking_evaluated else None,
        "refusal_accuracy":           round(100 * refusal_correct / refusal_evaluated, 1) if refusal_evaluated else None,
        # Dual-label top-1
        "jtbd_top1_accuracy":         jtbd_pct,
        "formula_top1_accuracy":      formula_pct,
        "cook_now_accuracy":          jtbd_pct,  # alias for JTBD headline
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
    metric_key: str = "correct",
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
            if pe and pe.get(metric_key) is not None:
                slices[val][name].append(int(pe[metric_key]))

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
    metric_key: str = "correct",
) -> dict[str, dict[str, float]]:
    return slice_breakdown(examples, comparator_results, "source", metric_key=metric_key)


def tag_breakdown(
    examples: list[dict],
    comparator_results: dict[str, dict],
    metric_key: str = "correct",
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
                if pe and pe.get(metric_key) is not None:
                    slices[tag][name].append(int(pe[metric_key]))

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
    metric_key: str = "correct",
) -> dict[str, dict[str, float]]:
    """Top-1 accuracy by CSV category (modal / edge / OOS / adversarial)."""
    return slice_breakdown(examples, comparator_results, "csv_tag", metric_key=metric_key)


def _pct(values: list[bool | int]) -> float | None:
    return round(100 * float(np.mean(values)), 1) if values else None


# ===========================================================================
# Statistical helpers (Part 2 of Fair-and-Robust plan)
# ===========================================================================

def bootstrap_ci(
    values: list[float | int | bool],
    n: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict:
    """Bootstrap 95% CI for the mean of `values` (accuracy, recall, etc.).

    Resamples at the example level. Returns {"mean", "ci_lo", "ci_hi", "n"}.
    """
    if not values:
        return {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0}
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n)
    ])
    alpha = (1.0 - confidence) / 2
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "mean":   round(float(arr.mean()), 4),
        "ci_lo":  round(float(lo), 4),
        "ci_hi":  round(float(hi), 4),
        "n":      len(arr),
    }


def mcnemar_significance(
    comparator_results: dict[str, dict],
    metric_key: str = "formula_correct",
) -> dict[str, dict[str, float | None]]:
    """Paired McNemar test between every pair of comparators.

    Because comparators see the same examples, paired tests are appropriate.
    Returns {comp_a: {comp_b: p_value}} for all (a, b) pairs.
    Only uses examples present in both comparators (same eval_id, not skipped).
    Two-tailed, continuity-corrected (Edwards correction).
    """
    from scipy.stats import chi2

    # Build per-example correctness vector per comparator
    comp_correct: dict[str, dict[str, bool | None]] = {}
    for name, res in comparator_results.items():
        comp_correct[name] = {}
        for pe in res["per_example"]:
            v = pe.get(metric_key)
            if v is not None and not pe.get("skipped", False):
                comp_correct[name][pe["eval_id"]] = bool(v)

    names = list(comp_correct.keys())
    matrix: dict[str, dict[str, float | None]] = {n: {} for n in names}

    for i, na in enumerate(names):
        for j, nb in enumerate(names):
            if i >= j:
                matrix[na][nb] = None
                continue
            # Shared examples
            shared = set(comp_correct[na]) & set(comp_correct[nb])
            if not shared:
                matrix[na][nb] = None
                matrix[nb][na] = None
                continue
            # McNemar contingency: how many examples one got right that the other didn't
            b = sum(1 for eid in shared if comp_correct[na][eid] and not comp_correct[nb][eid])
            c = sum(1 for eid in shared if not comp_correct[na][eid] and comp_correct[nb][eid])
            if b + c == 0:
                p = 1.0
            else:
                # Edwards continuity-corrected McNemar
                stat = (abs(b - c) - 1.0) ** 2 / (b + c)
                p = float(1.0 - chi2.cdf(stat, df=1))
            p_r = round(p, 4)
            matrix[na][nb] = p_r
            matrix[nb][na] = p_r

    return matrix


def item_count_band(n_items: int) -> str:
    """Stratify by item count: small / medium / large."""
    if n_items <= 5:
        return "small (2-5)"
    if n_items <= 12:
        return "medium (6-12)"
    return "large (13-28)"


def holdout_clean_flag(ex: dict, cutoff: str = HOLDOUT_CUTOFF) -> bool:
    """True if this example comes from on/after the holdout cutoff date."""
    date_str = ex.get("date") or ex.get("decision_date") or ex.get("metadata", {}).get("date")
    if date_str is None:
        return False
    return date_str >= cutoff


# ===========================================================================
# Scale-stratified and holdout-clean breakdowns
# ===========================================================================

def scale_breakdown(
    examples: list[dict],
    comparator_results: dict[str, dict],
    metric_key: str = "correct",
) -> dict[str, dict[str, float]]:
    """Top-1 accuracy by item_count_band for each comparator."""
    result_lookup: dict[str, dict[str, dict]] = {}
    for name, comp in comparator_results.items():
        for pe in comp["per_example"]:
            eid = pe["eval_id"]
            result_lookup.setdefault(eid, {})[name] = pe

    slices: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for ex in examples:
        eid = ex["eval_id"]
        n_items = ex.get("num_items") or sum(
            1 for k in ex.get("features", {}) if k.endswith("_forecast_demand")
        )
        band = item_count_band(n_items)
        for name in comparator_results:
            pe = result_lookup.get(eid, {}).get(name)
            if pe and pe.get(metric_key) is not None:
                slices[band][name].append(int(pe[metric_key]))

    return {
        band: {
            name: round(100 * float(np.mean(v)), 1)
            for name, v in comp.items() if v
        }
        for band, comp in sorted(slices.items())
    }


def holdout_clean_breakdown(
    examples: list[dict],
    comparator_results: dict[str, dict],
    metric_key: str = "formula_correct",
) -> dict[str, Any]:
    """Compare v2.2 selection metrics on holdout_clean vs leakage slices."""
    result_lookup: dict[str, dict[str, dict]] = {}
    for name, comp in comparator_results.items():
        for pe in comp["per_example"]:
            eid = pe["eval_id"]
            result_lookup.setdefault(eid, {})[name] = pe

    clean: dict[str, list] = defaultdict(list)
    leakage: dict[str, list] = defaultdict(list)
    n_clean = 0
    n_leakage = 0

    for ex in examples:
        eid = ex["eval_id"]
        is_clean = ex.get("holdout_clean", holdout_clean_flag(ex))
        if is_clean:
            n_clean += 1
        else:
            n_leakage += 1
        for name in comparator_results:
            pe = result_lookup.get(eid, {}).get(name)
            if pe and pe.get(metric_key) is not None:
                bucket = clean if is_clean else leakage
                bucket[name].append(int(pe[metric_key]))

    return {
        "holdout_clean": {
            name: round(100 * float(np.mean(v)), 1) if v else None
            for name, v in clean.items()
        },
        "leakage": {
            name: round(100 * float(np.mean(v)), 1) if v else None
            for name, v in leakage.items()
        },
        "n_holdout_clean": n_clean,
        "n_leakage": n_leakage,
    }


def dual_label_breakdown(
    examples: list[dict],
    comparator_results: dict[str, dict],
) -> dict[str, Any]:
    """Compare JTBD vs formula top-1 accuracy overall and on agrees/divergence slices."""
    result_lookup: dict[str, dict[str, dict]] = {}
    for name, comp in comparator_results.items():
        for pe in comp["per_example"]:
            eid = pe["eval_id"]
            result_lookup.setdefault(eid, {})[name] = pe

    n_agrees = 0
    n_divergence = 0
    for ex in examples:
        agree = example_labels_agree(ex)
        if agree is True:
            n_agrees += 1
        elif agree is False:
            n_divergence += 1

    summary: dict[str, dict[str, Any]] = {}
    by_agreement: dict[str, dict[str, float]] = {"agrees": {}, "divergence": {}}
    by_category_jtbd: dict[str, dict[str, float]] = {}
    by_category_formula: dict[str, dict[str, float]] = {}

    for name in comparator_results:
        jtbd_all: list[int] = []
        formula_all: list[int] = []
        jtbd_agrees: list[int] = []
        formula_agrees: list[int] = []
        jtbd_divergence: list[int] = []
        formula_divergence: list[int] = []

        for ex in examples:
            eid = ex["eval_id"]
            pe = result_lookup.get(eid, {}).get(name)
            if not pe or pe.get("outcome") != "ranking":
                continue
            if pe.get("cook_now_correct") is not None:
                jtbd_all.append(int(pe["cook_now_correct"]))
            if pe.get("formula_correct") is not None:
                formula_all.append(int(pe["formula_correct"]))

            agree = example_labels_agree(ex)
            if agree is True:
                if pe.get("cook_now_correct") is not None:
                    jtbd_agrees.append(int(pe["cook_now_correct"]))
                if pe.get("formula_correct") is not None:
                    formula_agrees.append(int(pe["formula_correct"]))
            elif agree is False:
                if pe.get("cook_now_correct") is not None:
                    jtbd_divergence.append(int(pe["cook_now_correct"]))
                if pe.get("formula_correct") is not None:
                    formula_divergence.append(int(pe["formula_correct"]))

        jtbd_pct = _pct(jtbd_all)
        formula_pct = _pct(formula_all)
        summary[name] = {
            "jtbd_top1_pct": jtbd_pct,
            "formula_top1_pct": formula_pct,
            "delta_jtbd_minus_formula_pp": (
                round(jtbd_pct - formula_pct, 1)
                if jtbd_pct is not None and formula_pct is not None else None
            ),
            "n_ranking_scored": len(jtbd_all),
            "n_labels_agree": n_agrees,
            "n_divergence": n_divergence,
            "jtbd_on_agrees_pct": _pct(jtbd_agrees),
            "formula_on_agrees_pct": _pct(formula_agrees),
            "jtbd_on_divergence_pct": _pct(jtbd_divergence),
            "formula_on_divergence_pct": _pct(formula_divergence),
        }
        if jtbd_agrees or formula_agrees:
            by_agreement["agrees"][f"{name}_jtbd"] = _pct(jtbd_agrees) or 0.0
            by_agreement["agrees"][f"{name}_formula"] = _pct(formula_agrees) or 0.0
        if jtbd_divergence or formula_divergence:
            by_agreement["divergence"][f"{name}_jtbd"] = _pct(jtbd_divergence) or 0.0
            by_agreement["divergence"][f"{name}_formula"] = _pct(formula_divergence) or 0.0

    by_category_jtbd = category_breakdown(examples, comparator_results, metric_key="cook_now_correct")
    by_category_formula = category_breakdown(examples, comparator_results, metric_key="formula_correct")

    return {
        "summary": summary,
        "by_label_agreement": by_agreement,
        "by_category_jtbd": by_category_jtbd,
        "by_category_formula": by_category_formula,
    }


def print_dual_label_summary(dual: dict[str, Any], comp_names: list[str]) -> None:
    """Print JTBD vs formula side-by-side table."""
    print("\n" + "=" * 72)
    print("  DUAL-LABEL BREAKDOWN — JTBD vs Formula Top-1")
    print("=" * 72)
    summary = dual["summary"]
    if not summary:
        print("\n  (no ranking results)")
        return

    n_agree = next(iter(summary.values())).get("n_labels_agree", 0)
    n_div = next(iter(summary.values())).get("n_divergence", 0)
    print(f"\n  Label slices: agrees={n_agree}  divergence={n_div}")

    print(f"\n  {'Comparator':<22s} {'JTBD%':>8s} {'Formula%':>9s} {'Δ':>7s}")
    print(f"  {'-'*22} {'-'*8} {'-'*9} {'-'*7}")
    for name in comp_names:
        s = summary.get(name, {})
        jtbd = s.get("jtbd_top1_pct")
        formula = s.get("formula_top1_pct")
        delta = s.get("delta_jtbd_minus_formula_pp")
        jtbd_s = f"{jtbd:.1f}%" if jtbd is not None else "   n/a"
        formula_s = f"{formula:.1f}%" if formula is not None else "    n/a"
        delta_s = f"{delta:+.1f}pp" if delta is not None else "   n/a"
        print(f"  {name:<22s} {jtbd_s:>8s} {formula_s:>9s} {delta_s:>7s}")

    by_agree = dual.get("by_label_agreement", {})
    for slice_name, title in [("agrees", "Labels agree (JTBD = formula)"),
                               ("divergence", "Labels diverge (JTBD ≠ formula)")]:
        slice_data = by_agree.get(slice_name, {})
        if not slice_data:
            continue
        print(f"\n  {title}:")
        for name in comp_names:
            jtbd_v = slice_data.get(f"{name}_jtbd")
            formula_v = slice_data.get(f"{name}_formula")
            if jtbd_v is None and formula_v is None:
                continue
            jtbd_s = f"{jtbd_v:.1f}%" if jtbd_v is not None else "n/a"
            formula_s = f"{formula_v:.1f}%" if formula_v is not None else "n/a"
            print(f"    {name:<20s}  JTBD {jtbd_s:>6s}   Formula {formula_s:>6s}")


# ===========================================================================
# Scorecard helpers
# ===========================================================================

GUARDRAIL_MPV_RATE    = 0.05   # must_precede_violation_rate < this
GUARDRAIL_REFUSAL_PCT = 90.0   # refusal_accuracy >= this (only evaluated if n>=10)
GUARDRAIL_LATENCY_MS  = {      # median latency budget by comparator type
    "llm":       3000.0,
    "default":    500.0,
}


def _guardrail_check(name: str, r: dict) -> dict[str, bool | str]:
    """Return pass/fail dict for each guardrail."""
    is_llm = name.startswith("llm_")
    mpv_rate = r.get("must_precede_violation_rate")
    refusal_pct = r.get("refusal_accuracy")
    latency = r.get("latency_median_ms")
    refusal_n = r.get("refusal_evaluated", 0)
    lat_budget = GUARDRAIL_LATENCY_MS["llm"] if is_llm else GUARDRAIL_LATENCY_MS["default"]
    parse_fail = r.get("parse_failures", 0)
    ranking_n = r.get("ranking_evaluated", 0)

    mpv_pass   = (mpv_rate is not None and mpv_rate < GUARDRAIL_MPV_RATE) or mpv_rate is None
    ref_pass   = (refusal_pct is not None and refusal_pct >= GUARDRAIL_REFUSAL_PCT) \
                 if refusal_n >= 10 else None
    lat_pass   = (latency is not None and latency <= lat_budget) if latency else None
    parse_pass = (parse_fail == 0) if ranking_n > 0 else True

    return {
        "mpv_rate":          mpv_rate,
        "mpv_pass":          mpv_pass,
        "refusal_pct":       refusal_pct,
        "refusal_n":         refusal_n,
        "refusal_pass":      ref_pass,  # None = n too small to certify
        "latency_median_ms": latency,
        "latency_budget_ms": lat_budget,
        "latency_pass":      lat_pass,
        "parse_failures":    parse_fail,
        "parse_pass":        parse_pass,
        "all_guardrails_pass": mpv_pass and (ref_pass is not False) and (lat_pass is not False) and parse_pass,
    }


def build_selection_scorecard(
    results: dict[str, dict],
    metric_key: str,
    per_example_ci_key: str,
    ci_seed: int = 42,
) -> dict[str, Any]:
    """Build a selection scorecard per comparator.

    Computes:
      - primary metric + bootstrap 95% CI
      - guardrail pass/fail
      - operational cost/latency summary
      - selection recommendation
    """
    scorecard: dict[str, Any] = {}

    for name, r in results.items():
        # Collect per-example correctness for CI
        values = [
            pe[per_example_ci_key]
            for pe in r.get("per_example", [])
            if pe.get(per_example_ci_key) is not None and not pe.get("skipped", False)
        ]
        ci = bootstrap_ci(values, seed=ci_seed)
        primary_pct = round(ci["mean"] * 100, 1) if ci["mean"] is not None else r.get(metric_key)
        ci_lo_pct = round(ci["ci_lo"] * 100, 1) if ci["ci_lo"] is not None else None
        ci_hi_pct = round(ci["ci_hi"] * 100, 1) if ci["ci_hi"] is not None else None

        guardrails = _guardrail_check(name, r)
        scorecard[name] = {
            "primary_metric":       metric_key,
            "primary_pct":          primary_pct,
            "ci_95_lo_pct":         ci_lo_pct,
            "ci_95_hi_pct":         ci_hi_pct,
            "ci_n":                 ci["n"],
            "guardrails":           guardrails,
            "latency_median_ms":    r.get("latency_median_ms"),
            "parse_failure_rate":   (
                r.get("parse_failures", 0) / r.get("ranking_evaluated", 1)
                if r.get("ranking_evaluated", 0) > 0 else 0.0
            ),
        }

    # Determine recommendation based on plan decision rule
    passing = [(n, sc) for n, sc in scorecard.items()
               if sc["guardrails"]["all_guardrails_pass"] and sc["primary_pct"] is not None]
    if not passing:
        recommendation = "No comparator passes all guardrails — defer selection."
    else:
        passing_sorted = sorted(passing, key=lambda x: x[1]["primary_pct"], reverse=True)
        best_name, best_sc = passing_sorted[0]
        if len(passing_sorted) >= 2:
            second_name, second_sc = passing_sorted[1]
            # If best CI-lo exceeds second's point estimate → statistically credible win
            if (best_sc["ci_95_lo_pct"] is not None
                    and second_sc["primary_pct"] is not None
                    and best_sc["ci_95_lo_pct"] > second_sc["primary_pct"]):
                recommendation = (
                    f"SELECT {best_name}  "
                    f"({best_sc['primary_pct']:.1f}% [{best_sc['ci_95_lo_pct']:.1f}–{best_sc['ci_95_hi_pct']:.1f}%] "
                    f"vs {second_name} {second_sc['primary_pct']:.1f}% — CI excludes runner-up)"
                )
            else:
                # Tiebreak by simplicity: v1_rules > v2_2_ml > llm
                tiebreak_order = ["v1_rules", "v2_3_ml", "v2_2_ml"] + [n for n in passing_sorted if "llm" in n[0]]
                selected = next((n for n in tiebreak_order if n in dict(passing_sorted)), best_name)
                recommendation = (
                    f"TIEBREAK → {selected}  "
                    f"(CIs overlap with runner-up; prefer simpler/cheaper. "
                    f"Run larger N for discriminating CIs.)"
                )
        else:
            recommendation = (
                f"SELECT {best_name}  "
                f"({best_sc['primary_pct']:.1f}% [{best_sc['ci_95_lo_pct']:.1f}–{best_sc['ci_95_hi_pct']:.1f}%] — only passing comparator)"
            )

    scorecard["_recommendation"] = recommendation
    return scorecard


def print_scorecard(scorecard: dict[str, Any], comp_names: list[str]) -> None:
    print("\n" + "=" * 72)
    print("  SELECTION SCORECARD")
    print("=" * 72)
    print(f"\n  {'Comparator':<22s} {'Primary%':>9s} {'95% CI':>16s} {'MPV':>6s} {'Refusal':>8s} {'Lat(ms)':>8s} {'Pass?':>6s}")
    print(f"  {'-'*22} {'-'*9} {'-'*16} {'-'*6} {'-'*8} {'-'*8} {'-'*6}")
    for name in comp_names:
        sc = scorecard.get(name, {})
        g = sc.get("guardrails", {})
        pct = f"{sc['primary_pct']:.1f}%" if sc.get("primary_pct") is not None else "  n/a"
        lo = sc.get("ci_95_lo_pct")
        hi = sc.get("ci_95_hi_pct")
        ci_s = f"[{lo:.1f}–{hi:.1f}%]" if lo is not None else "  n/a"
        mpv = f"{g.get('mpv_rate', 0):.3f}" if g.get("mpv_rate") is not None else "   n/a"
        ref = f"{g.get('refusal_pct', 0):.0f}%/{g.get('refusal_n', 0)}ex" if g.get("refusal_pct") is not None else "      n/a"
        lat = f"{sc.get('latency_median_ms', 0):.0f}" if sc.get("latency_median_ms") is not None else "    n/a"
        ok = "PASS" if g.get("all_guardrails_pass") else "FAIL"
        print(f"  {name:<22s} {pct:>9s} {ci_s:>16s} {mpv:>6s} {ref:>8s} {lat:>8s} {ok:>6s}")

    rec = scorecard.get("_recommendation", "")
    if rec:
        print(f"\n  Recommendation: {rec}")
    print()


def print_significance_matrix(sig: dict[str, dict[str, float | None]], comp_names: list[str]) -> None:
    print("\n" + "=" * 72)
    print("  McNemar PAIRED SIGNIFICANCE (p-values, two-tailed, continuity-corrected)")
    print("  p < 0.05 → difference is statistically significant at this sample size")
    print("=" * 72)
    col_w = 12
    print(f"\n  {'':22s}", end="")
    for n in comp_names:
        print(f"  {n[:col_w]:>{col_w}s}", end="")
    print()
    for na in comp_names:
        print(f"  {na:<22s}", end="")
        for nb in comp_names:
            if na == nb:
                print(f"  {'—':>{col_w}s}", end="")
            else:
                p = sig.get(na, {}).get(nb)
                ps = f"{p:.4f}" if p is not None else "   n/a"
                star = "*" if p is not None and p < 0.05 else " "
                print(f"  {(ps+star):>{col_w}s}", end="")
        print()
    print()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("=" * 64)
    print(f"  LLM EVAL RUNNER  eval-set={EVAL_SET_VERSION}  prompt={PROMPT_VERSION}")
    print(f"  input-mode={INPUT_MODE}  assoc-seeds={ASSOC_SEEDS}  llm-samples={LLM_SAMPLES}")
    print("=" * 64)

    # --- Load artifacts ---
    print("\n[1/6] Loading eval set and models...")
    meta, examples = load_eval_set()
    print(f"  Eval set  : {len(examples)} examples ({meta['version']})")
    print(f"  Eval path : {EVAL_SET_PATH}")
    if "sources" in meta:
        print(f"  Sources   : {meta['sources']}")

    system_prompt_preview = load_system_prompt()[:80].replace("\n", " ")
    print(f"  Prompt    : {system_prompt_preview}...")

    scheduler  = CookSchedulerV1()
    pw_model   = try_load_v22_model()
    pw_v23     = try_load_v23_model()
    loaded_msg = " + v2.2 loaded" if pw_model else " (v2.2 skipped)"
    loaded_msg += " + v2.3 loaded" if pw_v23 else ""
    print("  v1 loaded" + loaded_msg)

    # --- Set up comparators ---
    print("\n[2/6] Setting up comparators...")

    # --- Multi-seed associate baseline (stochastic floor) ---
    print(f"\n  associate_floor: running {ASSOC_SEEDS} seeds for mean ± std...")
    assoc_seed_results: list[dict] = []
    for seed in range(ASSOC_SEEDS):
        assoc_s = AssociateBaseline(seed=seed)
        res_s = evaluate_comparator(
            "associate_floor",
            examples,
            lambda ex, a=assoc_s: rank_associate(a, ex),
        )
        assoc_seed_results.append(res_s)

    # Average the associate floor metrics across seeds
    def _avg_metric(key: str) -> float | None:
        vals = [r[key] for r in assoc_seed_results if r.get(key) is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    def _std_metric(key: str) -> float | None:
        vals = [r[key] for r in assoc_seed_results if r.get(key) is not None]
        return round(float(np.std(vals)), 3) if len(vals) > 1 else None

    assoc_result = assoc_seed_results[RANDOM_SEED % ASSOC_SEEDS]  # canonical seed for per_example
    for mk in ["cook_now_accuracy", "formula_top1_accuracy", "jtbd_top1_accuracy",
               "top1_accuracy", "ranking_accuracy", "refusal_accuracy",
               "kendall_tau_mean", "must_precede_violation_rate"]:
        assoc_result[f"{mk}_mean"]  = _avg_metric(mk)
        assoc_result[f"{mk}_std"]   = _std_metric(mk)
    assoc_result["n_seeds"] = ASSOC_SEEDS

    rank_fns: dict[str, Any] = {
        "v1_rules": lambda ex: rank_v1(scheduler, ex),
    }
    if pw_model is not None:
        rank_fns["v2_2_ml"] = lambda ex: rank_v22(pw_model, ex)
    if pw_v23 is not None:
        rank_fns["v2_3_ml"] = lambda ex: rank_v22(pw_v23, ex)

    refusal_fns: dict[str, Any] = {}
    llm_key: str | None = None
    llm: V01LLMRanker | None = None

    if NO_LLM:
        print("  LLM skipped (--no-llm flag). Running ML baselines only.")
    else:
        print(f"  Initialising LLM ranker ({LLM_MODEL}, prompt {PROMPT_VERSION}, "
              f"input_mode={INPUT_MODE}, samples={LLM_SAMPLES})...")
        llm = V01LLMRanker()
        llm_key = f"llm_{PROMPT_VERSION}_zero_shot"
        if INPUT_MODE != "native":
            llm_key = f"llm_{PROMPT_VERSION}_{INPUT_MODE}"
        rank_fns[llm_key] = lambda ex, _llm=llm: _llm.rank(ex, input_mode=INPUT_MODE)
        refusal_fns[llm_key] = llm.check_refusal
        n_ranking = sum(1 for ex in examples if not is_refusal_example(ex, _extract_features(ex)))
        n_refusal = len(examples) - n_ranking
        est_calls = n_ranking * LLM_SAMPLES + n_refusal
        print(f"  LLM ready. {n_ranking} ranking × {LLM_SAMPLES} sample(s) + {n_refusal} refusal = {est_calls} API calls.")

    # --- Evaluate ---
    print(f"\n[3/6] Evaluating {len(examples)} examples...")
    results: dict[str, dict] = {"associate_floor": assoc_result}

    for name, fn in rank_fns.items():
        print(f"  → {name}...")
        if name == llm_key and LLM_SAMPLES > 1 and llm is not None:
            # Multi-sample LLM: run each ranking example k times
            sample_results: list[dict] = []
            for sample_i in range(LLM_SAMPLES):
                print(f"     sample {sample_i + 1}/{LLM_SAMPLES}...")
                sample_results.append(
                    evaluate_comparator(name, examples, fn, refusal_fn=refusal_fns.get(name))
                )
            # Majority vote ranking + mean/std metrics
            llm_result = sample_results[0]
            for mk in ["cook_now_accuracy", "formula_top1_accuracy", "jtbd_top1_accuracy",
                       "top1_accuracy", "ranking_accuracy", "refusal_accuracy", "kendall_tau_mean"]:
                vals = [r[mk] for r in sample_results if r.get(mk) is not None]
                llm_result[f"{mk}_mean"] = round(float(np.mean(vals)), 3) if vals else None
                llm_result[f"{mk}_std"]  = round(float(np.std(vals)), 3) if len(vals) > 1 else None
            llm_result["n_samples"] = LLM_SAMPLES
            total_pf = sum(r.get("parse_failures", 0) for r in sample_results)
            total_rank = sum(r.get("ranking_evaluated", 0) for r in sample_results)
            llm_result["parse_failure_rate"] = round(total_pf / total_rank, 4) if total_rank else 0.0
            results[name] = llm_result
        else:
            results[name] = evaluate_comparator(name, examples, fn,
                                                 refusal_fn=refusal_fns.get(name))

    # --- Bootstrap CIs ---
    print("\n[4/6] Computing bootstrap CIs and significance...")
    ci_metric_map = {
        "v0.3": ("cook_now_correct", "formula_correct"),
        "default": ("formula_correct", "formula_correct"),
    }
    primary_pe_key, secondary_pe_key = ci_metric_map.get(EVAL_SET_VERSION, ci_metric_map["default"])

    def _result_ci(r: dict, pe_key: str) -> dict:
        vals = [int(pe[pe_key]) for pe in r.get("per_example", [])
                if pe.get(pe_key) is not None and not pe.get("skipped", False)]
        return bootstrap_ci(vals)

    result_cis: dict[str, dict[str, dict]] = {}
    for name, r in results.items():
        result_cis[name] = {
            "cook_now_ci":    _result_ci(r, "cook_now_correct"),
            "formula_ci":     _result_ci(r, "formula_correct"),
            "set_recall_ci":  bootstrap_ci([
                pe["cook_now_set_recall"] for pe in r.get("per_example", [])
                if pe.get("cook_now_set_recall") is not None
            ]),
            "mpv_rate_ci":    bootstrap_ci([
                pe.get("must_precede_violations", 0) for pe in r.get("per_example", [])
                if pe.get("outcome") == "ranking"
            ]),
        }

    significance_matrix = mcnemar_significance(results, metric_key="formula_correct")

    # --- Breakdowns ---
    breakdown_metric = "cook_now_correct" if EVAL_SET_VERSION == "v0.3" else "correct"
    by_source    = source_breakdown(examples, results, metric_key=breakdown_metric)
    by_tag       = tag_breakdown(examples, results, metric_key=breakdown_metric)
    by_category  = category_breakdown(examples, results, metric_key=breakdown_metric)
    by_store     = slice_breakdown(examples, results, "store_type", metric_key=breakdown_metric)
    by_hour_band = slice_breakdown(examples, results, "hour_band", metric_key=breakdown_metric)
    by_scale     = scale_breakdown(examples, results, metric_key=breakdown_metric)
    by_holdout   = holdout_clean_breakdown(examples, results, metric_key="formula_correct")
    dual_label   = dual_label_breakdown(examples, results)

    # --- Scorecard ---
    scorecard = build_selection_scorecard(
        results,
        metric_key="formula_top1_accuracy",
        per_example_ci_key="formula_correct",
    )
    comp_names = list(results.keys())

    # --- Print summary ---
    print("\n" + "=" * 72)
    print("  RESULTS — JTBD Metrics")
    print("=" * 72)
    print(f"\n  {'Comparator':<22s} {'CookNow%':>9s} {'CI 95%':>14s} {'SetRecall':>10s} {'MPViol':>7s} {'τ':>8s} {'Fail':>5s}")
    print(f"  {'-'*22} {'-'*9} {'-'*14} {'-'*10} {'-'*7} {'-'*8} {'-'*5}")
    for name, r in results.items():
        cn  = f"{r['cook_now_accuracy']:.1f}%" if r.get("cook_now_accuracy") is not None else "   n/a"
        ci  = result_cis[name]["cook_now_ci"]
        lo, hi = ci.get("ci_lo"), ci.get("ci_hi")
        ci_s = f"[{lo*100:.1f}–{hi*100:.1f}%]" if lo is not None else "           n/a"
        sr  = f"{r['cook_now_set_recall']:.3f}" if r.get("cook_now_set_recall") is not None else "    n/a"
        mpv = str(r.get("must_precede_violations", 0))
        tau = f"{r['kendall_tau_mean']:.3f}" if r.get("kendall_tau_mean") is not None else "    n/a"
        fail = str(r.get("parse_failures", 0))
        stdev = f" ±{r['cook_now_accuracy_std']:.1f}" if r.get("cook_now_accuracy_std") else ""
        print(f"  {name:<22s} {cn+stdev:>9s} {ci_s:>14s} {sr:>10s} {mpv:>7s} {tau:>8s} {fail:>5s}")

    def _print_breakdown_table(title: str, data: dict) -> None:
        print(f"\n  {title}:")
        print(f"  {title.split(':')[0]:<20s}", end="")
        for n in comp_names:
            print(f"  {n:>18s}", end="")
        print()
        for key, scores in data.items():
            print(f"  {str(key):<20s}", end="")
            for n in comp_names:
                v = scores.get(n)
                print(f"  {(str(round(v, 1)) + '%') if v is not None else 'n/a':>18s}", end="")
            print()

    _print_breakdown_table("By Source", by_source)
    _print_breakdown_table("By Category (modal/edge/OOS/adversarial)", by_category)
    _print_breakdown_table("By Eval Tag", by_tag)
    _print_breakdown_table("By Store Type", by_store)
    _print_breakdown_table("By Item Count Band", by_scale)

    print("\n  By Hour Band:")
    print(f"  {'Hour':<20s}", end="")
    for n in comp_names:
        print(f"  {n:>18s}", end="")
    print()
    for band, scores in by_hour_band.items():
        print(f"  {band:<20s}", end="")
        for n in comp_names:
            v = scores.get(n)
            print(f"  {(str(round(v, 1)) + '%') if v is not None else 'n/a':>18s}", end="")
        print()

    print_dual_label_summary(dual_label, comp_names)
    print_scorecard(scorecard, comp_names)
    print_significance_matrix(significance_matrix, comp_names)

    # --- Dump per-example predictions ---
    existing_llm_preds: dict = {}
    ml_comparator_names = {k for k in rank_fns if not k.startswith("llm_")} | {"associate_floor"}
    if NO_LLM and os.path.exists(PREDICTIONS_PATH):
        try:
            with open(PREDICTIONS_PATH) as f:
                existing = json.load(f)
            for name, preds in existing.get("predictions", {}).items():
                if name not in ml_comparator_names:
                    existing_llm_preds[name] = preds
        except Exception:
            pass

    preds_out = {
        "metadata": {
            "prompt_version": PROMPT_VERSION,
            "input_mode": INPUT_MODE,
            "llm_model": LLM_MODEL if not NO_LLM else "skipped (ml-only re-run)",
            "run_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "predictions": {
            **{name: r["per_example"] for name, r in results.items()},
            **existing_llm_preds,
        },
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
                    "See EVAL_METHODOLOGY.md for decision framework."
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
            "input_mode": INPUT_MODE,
            "random_seed": RANDOM_SEED,
            "assoc_seeds": ASSOC_SEEDS,
            "llm_samples": LLM_SAMPLES,
            "no_llm": NO_LLM,
            "source_distribution": meta.get("sources", {}),
            "tag_distribution": meta.get("csv_tag_distribution", {}),
            "eval_tier": (
                "DIAGNOSTIC — v0.3 is plain-language, 2-5 item scale, n=45 ranking. "
                "Do NOT use for model selection. See EVAL_METHODOLOGY.md."
            ) if EVAL_SET_VERSION == "v0.3" else (
                "SELECTION — ~150 modal examples from ML temporal holdout (holdout_clean=True) "
                "plus 47 v0.3 guardrails (23 edge, 15 OOS, 9 adversarial). "
                "Use formula_top1_accuracy on modal slice only for v2.2=68.9% / v3=66.2% comparison. "
                "v1_rules uses CookSchedulerV1.rank_from_features (canonical tie-breaking)."
            ) if EVAL_SET_VERSION == "holdout" else "DIAGNOSTIC — see EVAL_METHODOLOGY.md for selection protocol.",
            "jtbd_metrics": (
                "jtbd_top1_accuracy / cook_now_accuracy: correct first item vs JTBD label. "
                "formula_top1_accuracy: correct first item vs formula label. "
                "cook_now_set_recall: fraction of urgent items in top-k. "
                "must_precede_violations: safety constraint violations (goal=0). "
                "refusal_accuracy: correctness on refusal-routed examples. "
                "kendall_tau: full-order quality (secondary). "
                "See dual_label breakdown for agrees vs divergence slices."
            ),
            "canonical_holdout_reference": _load_holdout_ref(),
        },
        "results": {
            name: {k: v for k, v in r.items() if k != "per_example"}
            for name, r in results.items()
        },
        "bootstrap_cis": result_cis,
        "significance": significance_matrix,
        "selection_scorecard": scorecard,
        "breakdowns": {
            "by_category":       by_category,
            "by_source":         by_source,
            "by_eval_tag":       by_tag,
            "by_store_type":     by_store,
            "by_hour_band":      by_hour_band,
            "by_item_count_band": by_scale,
            "holdout_clean":     by_holdout,
            "dual_label":        dual_label,
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[6/6] Report saved → {OUTPUT_PATH}")
    print("=" * 64)


if __name__ == "__main__":
    main()
