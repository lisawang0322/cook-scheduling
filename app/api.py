"""FastAPI backend — HTTP bridge between Hot Food Hero frontend and Python ML pipeline.

Start with:
    uvicorn app.api:app --reload --port 8000
    (run from project root)
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.utils import (
    compute_aggregate_metrics,
    find_story_scenario,
    generate_explanation,
    generate_forecast_demand,
    get_v22_ranking,
    load_labeled_data,
    load_model,
    log_associate_action,
    STORY_SCENARIOS,
)

_model = None
_historical = None
_labeled_data: list = []
_metrics_cache: dict | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _historical, _labeled_data
    _model, _historical = load_model()
    _labeled_data = load_labeled_data()
    yield


app = FastAPI(title="Hot Food Scheduler API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Request models ----------


class ItemInput(BaseModel):
    id: str              # backend item id: e.g. pizza_slice | wings_bone_in | taquito | ...
    forecast_demand: int
    lcu: int
    hold_time: float
    time_remaining: float


class RankRequest(BaseModel):
    store_type: str      # "urban" | "suburban" | "highway"
    day_of_week: str     # "Monday" … "Sunday"
    is_weekend: bool
    decision_hour: int   # 0..23
    items: list[ItemInput]


def apply_ml_guardrails(ranking: list[str], decision_hour: int, present_item_ids: set[str]) -> list[str]:
    """Apply lightweight business guardrails on top of ML ranking.

    Placeholder for Sprint 2 rules (e.g. morning-only items deprioritized after 11 AM).
    Currently a no-op pass-through — ML ranking is trusted as-is.
    """
    if not ranking:
        return ranking
    return ranking


# ---------- Endpoints ----------


@app.get("/health")
def health():
    return {"ok": True, "model_loaded": _model is not None}


@app.post("/api/rank")
def rank(req: RankRequest):
    """Run v2.2 ML ranking for one Sprint 1 prototype scenario."""
    features: dict = {
        "decision_hour": req.decision_hour,
        "store_type": req.store_type,
        "day_of_week": req.day_of_week,
        "is_weekend": req.is_weekend,
        "num_oven_items": len(req.items),
    }
    for item in req.items:
        # Use backend demand model (same distribution the ML model was trained on).
        # Frontend demand comes from a different scale (daypart allocation engine)
        # and would push ML inputs out of distribution.
        demand = generate_forecast_demand(
            item.id, req.decision_hour, req.store_type, req.is_weekend
        )
        features[f"{item.id}_forecast_demand"] = demand
        features[f"{item.id}_lcu"] = item.lcu
        features[f"{item.id}_hold_time"] = item.hold_time
        features[f"{item.id}_time_remaining"] = item.time_remaining
        features[f"{item.id}_cooked_qty"] = 0

    v22_ranking = get_v22_ranking(features, _model, _historical)
    v22_ranking = apply_ml_guardrails(
        v22_ranking,
        req.decision_hour,
        {item.id for item in req.items},
    )
    explanations = generate_explanation(v22_ranking, features)

    return {
        "v22_ranking": v22_ranking,
        "explanations": explanations,
    }


@app.get("/api/scenarios")
def get_scenarios():
    """Return story scenarios sourced from the real labeled dataset."""
    result = []
    for i, story in enumerate(STORY_SCENARIOS):
        scenario = find_story_scenario(_labeled_data, i)
        if scenario:
            result.append({
                "id": str(i),
                "name": story["name"],
                "description": story["description"],
                "features": scenario["features"],
                "optimal_first": scenario["optimal_first_item"],
                "optimal_order": scenario["optimal_order"],
            })
    return result


@app.get("/api/metrics")
def get_metrics():
    """Aggregate top-1 accuracy across all 1,747 labeled scenarios (cached after first call)."""
    global _metrics_cache
    if _metrics_cache is None:
        _metrics_cache = compute_aggregate_metrics(_labeled_data, _model, _historical)
    return _metrics_cache


@app.post("/api/log-action")
def log_action(record: dict):
    """Append an associate confirmation or override to output/associate_overrides.json."""
    path = log_associate_action(record)
    return {"ok": True, "path": path}
