"""LLM-based cook order ranker using Anthropic's Claude.

Provides zero-shot and few-shot ranking using only observable-at-decision-time
features (forecast_demand, lcu, hold_time, time_remaining). Post-hoc outcome
features (writeoff, cooked_qty) are explicitly excluded to prevent leakage.

Used for benchmarking against the v1 rule-based and v2.2 ML approaches.
The benchmark measures how well each approach recovers the domain-expert
labeling logic encoded in data_labeler.py:_determine_optimal_order().
"""

import json
import os
import re
import time
from typing import Any

from src.pairwise_trainer import OVEN_ITEMS  # single source of truth for item list

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_PATH = os.path.join(PROJECT_ROOT, "prompts", "v0.1_system_prompt.md")


def load_system_prompt(prompt_path: str = PROMPT_PATH) -> str:
    """Load the associate-framed system prompt body from the versioned prompt file.

    The associate-intuition prompt in prompts/v0.1_system_prompt.md is the single
    source of truth for the LLM's persona. The prior hardcoded analytical prompt
    (explicit formulas, demand_density, LCU constants) was retired so the LLM
    benchmarks idealized associate reasoning rather than data-scientist reasoning.
    """
    with open(prompt_path) as f:
        content = f.read()
    match = re.search(r"## Prompt Body\s*```\s*(.*?)```", content, re.DOTALL)
    if not match:
        raise ValueError(
            f"Could not find '## Prompt Body' fenced block in {prompt_path}. "
            "Ensure the prompt is wrapped in a triple-backtick code block under that heading."
        )
    return match.group(1).strip()


def extract_observable_features(features: dict[str, Any]) -> dict[str, Any]:
    """Extract only observable-at-decision-time features.

    Excludes writeoff and cooked_qty, which are post-hoc outcomes
    only known after the cook event completes.
    """
    obs: dict[str, Any] = {
        "store_type": features["store_type"],
        "day_of_week": features["day_of_week"],
        "is_weekend": features["is_weekend"],
        "decision_hour": features["decision_hour"],
        "items": {},
    }
    for item in OVEN_ITEMS:
        if f"{item}_forecast_demand" in features:
            demand = features[f"{item}_forecast_demand"]
            lcu = features[f"{item}_lcu"]
            obs["items"][item] = {
                "forecast_demand": demand,
                "lcu": lcu,
                "hold_time": features[f"{item}_hold_time"],
                "time_remaining": features[f"{item}_time_remaining"],
            }
    return obs


def _format_items_table(obs: dict[str, Any]) -> str:
    """Format items the way an associate would see them at the hot bar.

    Deliberately avoids the formula term demand_density. LCU is shown as the
    physical tray/batch size an associate actually knows, not as a ratio.
    """
    lines = []
    for item, props in obs["items"].items():
        lines.append(
            f"  {item:<12s} — need {props['forecast_demand']} units, "
            f"{props['time_remaining']}hr left in window, "
            f"stays good {props['hold_time']}hr once cooked, "
            f"cooks {props['lcu']} to a tray"
        )
    return "\n".join(lines)


def _format_decision_block(obs: dict[str, Any]) -> str:
    return (
        f"Decision point:\n"
        f"  Store type : {obs['store_type']}\n"
        f"  Day        : {obs['day_of_week']} (weekend={obs['is_weekend']})\n"
        f"  Hour       : {obs['decision_hour']}:00\n\n"
        f"Items present:\n{_format_items_table(obs)}"
    )


def build_prompt(
    obs: dict[str, Any],
    few_shot_examples: list[tuple[dict, list[str]]] | None = None,
) -> str:
    """Build the full user-turn prompt.

    Args:
        obs: Observable features from extract_observable_features().
        few_shot_examples: Optional list of (obs_features, optimal_order) tuples
            drawn exclusively from the training partition (pre-cutoff). Never
            pass test-set scenarios here to avoid ground-truth leakage.
    """
    parts: list[str] = []

    if few_shot_examples:
        parts.append("Examples of correct rankings:\n")
        for ex_obs, ex_order in few_shot_examples:
            present = list(ex_obs["items"].keys())
            ranked = [item for item in ex_order if item in present]
            example_block = _format_decision_block(ex_obs)
            parts.append(example_block)
            parts.append(f'\nResponse: {{"ranked_queue": {json.dumps(ranked)}}}\n')
        parts.append("---\nNow rank the following:\n")

    parts.append(_format_decision_block(obs))
    return "\n".join(parts)


def parse_response(text: str, present_items: list[str]) -> list[str] | None:
    """Parse LLM response into a validated ranked item list.

    Returns None if the response cannot be parsed, contains unknown items,
    or is missing any present item.
    """
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
    if set(ranked) != set(present_items) or len(ranked) != len(present_items):
        return None

    return ranked


class LLMRanker:
    """Anthropic Claude-based cook order ranker for benchmarking."""

    def __init__(
        self,
        model: str | None = None,
        max_retries: int = 1,
    ):
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package required: pip install anthropic"
            ) from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable not set.\n"
                "Export it before running: export ANTHROPIC_API_KEY=your_key"
            )

        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self.max_retries = max_retries
        self.system_prompt = load_system_prompt()

    def rank(
        self,
        features: dict[str, Any],
        few_shot_examples: list[tuple[dict, list[str]]] | None = None,
    ) -> tuple[list[str] | None, float]:
        """Rank oven items using the LLM.

        Args:
            features: Raw scenario features from labeled_training_set.json.
            few_shot_examples: Optional list of (obs_features, optimal_order)
                tuples drawn from the training set only.

        Returns:
            (ranked_items, latency_ms) where ranked_items is None on parse failure.
        """
        obs = extract_observable_features(features)
        present_items = list(obs["items"].keys())
        prompt = build_prompt(obs, few_shot_examples)

        start = time.time()
        ranked: list[str] | None = None

        for attempt in range(self.max_retries + 1):
            message = self.client.messages.create(
                model=self.model,
                max_tokens=128,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = message.content[0].text
            ranked = parse_response(response_text, present_items)
            if ranked is not None:
                break
            if attempt < self.max_retries:
                item_list = ", ".join(f'"{i}"' for i in present_items)
                prompt += (
                    f"\n\nYour previous response could not be parsed. "
                    f"Respond ONLY with valid JSON containing exactly these items: "
                    f"[{item_list}] in your preferred order."
                )

        latency_ms = round((time.time() - start) * 1000, 1)
        return ranked, latency_ms
