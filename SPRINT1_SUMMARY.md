# Sprint 1 Evaluation Summary — Hot Food Cook Order AI

**Author:** Lisa Wang | **Period:** Weeks 1–9 + Post-Sprint (Jun 2026) | **Updated:** June 27, 2026  
**LLM model:** claude-sonnet-4-6 | **Protocol:** [`EVAL_METHODOLOGY.md`](EVAL_METHODOLOGY.md)

---

## 1. Prototype Overview

### What Was Built

An end-to-end **hot food cook-sequencing prototype** that tells store associates which item to load into the oven first when multiple dayparts overlap. Delivered across five layers:

| Layer | Component | Status |
|---|---|---|
| Data | Synthetic cook logs, POS sales, write-offs; 2,164 labeled scenarios (28-item universe) | ✅ |
| Features | Urgency, demand density, hold time, store type, hour, historical waste | ✅ |
| ML v1 | Rule-based ranker (urgency × density × waste penalty) | ✅ |
| ML v2.2 | Pairwise gradient boosting with temporal holdout split | ✅ **selection winner** |
| ML v2.3 / v3 | Symmetric pairwise + LambdaRank experiments | ✅ evaluated |
| LLM | Zero-shot associate framing (prompts v0.1 → v0.3) | ✅ benchmarked |
| Eval harness | Unified runner: floor → ML → LLM; bootstrap CIs, McNemar, guardrails | ✅ |
| Demo | Streamlit app (3 pages) + Docker stack (FastAPI + Hot Food Hero UI) | ✅ |

**Dataset:** 175K+ synthetic events → 2,164 decision-point scenarios. **730-scenario temporal holdout** (date ≥ 2025-05-01) is the authoritative ML test partition.

### Current Architecture & Workflow

```
Food Planner forecast
        │
        ▼
Feature engineering (per-item urgency, demand, hold time, store/hour context)
        │
   ┌────┴────┬────────────┐
   ▼         ▼            ▼
 v1 rules  v2.2 ML    LLM benchmark
 (58.6%)   (68.9%)   (26.7% modal)
   │         │            │
   └────┬────┘            │
        ▼                 ▼
 Ranked cook sequence   Eval harness
 + plain-language       (197-ex holdout set,
   explanation            v0.3 diagnostic set)
        │
        ▼
 Associate UI (recommend — associate executes, full override)
```

**Workflow:** decision scenario → feature extraction → rank items → surface top recommendation with explanation → associate cooks (or overrides). ML serves production inference (~100 ms); LLM is evaluated as an idealized human-judgment ceiling, not the shipping path.

**Agency model:** Augmentation only — the system recommends; the associate decides.

---

## 2. Evaluation Results

> Full versioned tables (all eval sets, tiers, and breakdowns): [`VERSIONED_EVAL_RESULTS.md`](VERSIONED_EVAL_RESULTS.md)

### Overall Performance

Two eval tiers matter for decisions (see [`EVAL_METHODOLOGY.md`](EVAL_METHODOLOGY.md)):

**A) ML model selection — 730-scenario temporal holdout (authoritative)**

| Comparator | Formula top-1 | Notes |
|---|---|---|
| associate_floor | ~8.9% | Simulated flawed associate behavior |
| v1_rules | 58.6% | +49.7pp vs floor |
| **v2_2_ml** | **68.9%** | **Selection winner** (+10.3pp vs v1) |
| v2_3_ml | 66.0% | −2.9pp vs v2.2 |
| v3 LambdaRank | 66.2% | NDCG@1 0.723; −2.7pp vs v2.2 |

**B) Head-to-head at scale — holdout eval set (Jun 27 2026, n=197)**

150 modal examples (19–28 items, stratified from ML holdout) + 47 guardrails. **Use modal slice only** for accuracy comparison.

| Comparator | Modal formula top-1 | Refusal | Parse failures | Latency |
|---|---|---|---|---|
| associate_floor | 14.0% | — | 0 | 0 ms |
| v1_rules | 56.7% | — | 0 | 0 ms |
| **v2_2_ml** | **67.3%** | — | 0 | ~100 ms |
| v2_3_ml | 64.7% | — | 0 | ~98 ms |
| LLM v0.3 | **26.7%** | 93.3% (14/15) | 0 | ~11 s |

**Headline:** LLM underperforms v2.2 by **−40.6pp** on modal holdout — hypothesis confirmed (idealized human judgment does not beat ML at production scale).

*Legacy 5-item shared eval (superseded): v1 78.6%, v2.2 81.0%, LLM v0.2 64.0% on n=50.*

### Performance by Category / Tag

**Holdout eval set — formula top-1 by category** (`output/llm_eval_v0.3_holdout_report.json`):

| Category | associate | v1 | v2.2 | v2.3 | LLM v0.3 |
|---|---|---|---|---|---|
| **modal** (n=150) | 14.0% | 56.7% | **67.3%** | 64.7% | 26.7% |
| edge (n=23) | 34.8% | 30.4% | 73.9% | 69.6% | 82.6% |
| adversarial (n=9) | 55.6% | 66.7% | 88.9% | 88.9% | 66.7% |
| OOS (n=15) | — | — | — | — | **93.3%** refusal |

**By hour band (modal slice):** LLM collapses in **evening** (11.7% vs v2.2 39.0%); strongest at **morning** (59.5% vs v2.2 91.9%). All comparators struggle in evening — hardest operational window.

**By item count:** LLM **26.7%** on large (13–28 items) vs **83.0%** on small (2–5 guardrail cases) — scale confound explains inflated diagnostic-set scores.

**v0.3 diagnostic set (110 ex, 2–5 items — not for selection):**

| Comparator | JTBD top-1 | Formula top-1 | Refusal |
|---|---|---|---|
| v1_rules | 89.5% | 92.6% | — |
| v2_2_ml | 83.2% | 91.6% | — |
| LLM native | **95.8%** | 78.9% | 93.3% |
| LLM `--input-mode=features` | 93.6% | **90.4%** | 93.3% |

Fair input ablation: LLM formula ≈ v1/v2.2 (McNemar p=1.0) — no LLM edge when prose advantage is removed.

### Key Metrics & Observations

| Metric | v2.2 ML | LLM v0.3 | Observation |
|---|---|---|---|
| Formula top-1 (modal 150) | **67.3%** | 26.7% | ML wins decisively at scale |
| Kendall τ (holdout, all ranking) | 0.53 | 0.44 | ML produces better full orderings |
| MPV rate (holdout eval) | 0.060 | **0.006** | LLM safer on must-precede constraints |
| Refusal accuracy | n/a | **93.3%** | LLM handles OOS inputs well |
| Latency median | ~100 ms | ~11 s | LLM fails 3 s budget |
| Parse failures | 0 | **0** | Fixed via parser hardening (Jun 27) |

**McNemar (holdout eval):** LLM vs v2.2 p < 0.001 — difference is statistically significant at n=182 ranking examples.

**Selection decision:** Ship **v2.2 ML**. LLM remains a diagnostic/trust benchmark, not the ranking engine.

---

## 3. Failure Modes

### Specific Examples Where the System Underperformed

**1. LLM at production scale (modal holdout)**  
On 19–28 item scenarios, the LLM frequently returns plausible-but-wrong first picks — e.g., prioritizing high-demand items with longer hold windows over near-expiry items the formula labels urgent. Evening scenarios are worst (11.7% modal accuracy).

**2. JTBD divergence cases (label vs intuition)**  
Example **E06** (v0.3): pre-ordered 40 danishes (4 hr hold) vs pizza slices expiring in <2 hr. Formula says cook **danish** first (demand volume); JTBD label says **pizza_slice** (expiry urgency). LLM follows JTBD (80% on divergence tag in holdout eval); v1/v2.2 follow formula (40%). Requires field validation — neither label is proven ground truth.

**3. LLM adversarial compliance**  
On v0.2 eval, `hand_ADV_03`-style prompts ("all equally urgent — pick highest demand") still get ranked instead of refused. LLM adversarial accuracy: 33.3% (v0.2) → 66.7% (holdout guardrails) — improved but not solved.

**4. v1 / ML on edge guardrails**  
v1_rules scores **30.4%** on edge cases in holdout eval vs v2.2 **73.9%** — rule formula misses waste-avoidance nuance that pairwise ML learns.

**5. Evening hour band (all comparators)**  
v2.2 drops to **39.0%** formula top-1 in evening vs **91.9%** morning — likely label noise + operational complexity at daypart transitions; needs failure analysis.

**6. Infrastructure (resolved)**  
Initial LLM runs had **137/182 parse failures** at `max_tokens=256`. Fixed Jun 27: robust JSON parsing, retries, v1 fallback → **0 failures**.

### Known Limitations

- **Labels are formula-derived** (`src/data_labeler.py`) — accuracy measures convergence with composite priority logic, not confirmed waste reduction in stores.
- **Synthetic data only** — 66.4% high-confidence events; production warmer/POS patterns may differ.
- **No real associate adherence study** — model accuracy ≠ behavior change.
- **Guardrail certification incomplete** — MPV thresholds fail on mixed 197-ex eval set for ML; LLM fails latency.
- **Bootstrap CI on 730 holdout** not yet run — selection margin (+10.3pp) not yet CI-certified.
- **Single LLM model, zero-shot** — no fine-tuning, no latency optimization.

---

## 4. Iterations

### ML Model Evolution (v1 → v2.2 → v2.3 → v3)

| Version | Change | 730-holdout top-1 | Δ vs prior |
|---|---|---|---|
| v1 rules | Urgency × density × waste penalty baseline | 58.6% | — |
| v2.1 pairwise | Pairwise GBM + historical features (no temporal split) | 77.1%* | +18.5pp (*inflated — leakage) |
| **v2.2** | Temporal split + soft sample weights + train-only historical features | **68.9%** | Honest metric |
| v2.3 | Symmetric pairs + proba aggregation, no weights | 66.0% | **−2.9pp** regression |
| v3 | LightGBM LambdaRank, NDCG-optimised, listwise | 66.2% | **−2.7pp** vs v2.2 |

**Evidence:** Temporal split (v2.2) was the critical honesty improvement — v2.1's 77.1% dropped to 68.9% when test-period leakage was removed. v2.3/v3 did not beat v2.2 on top-1 despite architectural improvements (NDCG@1 0.723 for v3).

### LLM Prompt Evolution (v0.1 → v0.2 → v0.3)

| Prompt | Change | 50-ex shared eval | Holdout modal (150) |
|---|---|---|---|
| v0.1 | Initial zero-shot | 50.0% formula | — |
| v0.2 | Associate-voice tightening | 64.0% (+14pp); edge +25pp | — |
| v0.3 | Plain-language JTBD scenarios | 78.9% formula (diagnostic, 2–5 items) | **26.7%** (production scale) |

**Evidence of improvement:** v0.2 prompt materially improved edge/modal on small eval sets. **Evidence of regression at scale:** v0.3 diagnostic success does not transfer — prose + small N masked a **−40pp gap** vs v2.2 on modal holdout.

### Eval Harness Evolution

| Change | Impact |
|---|---|
| JTBD metrics + dual-label breakdown | Surfaces formula vs JTBD label conflicts |
| `--eval-set=holdout` + stratified 150-ex modal sample | Apples-to-apples ML vs LLM at 19–28 items |
| v1 `rank_from_features()` canonical tie-breaking | v1 modal 56.7% (was inflated ~80% with arbitrary ties) |
| LLM parse hardening (1024 tokens, JSON repair, retries) | Parse failures 137 → **0** |
| Fair input ablation (`--input-mode=features`) | Quantified prose confound on v0.3 diagnostic set |

---

## 5. Sprint 2 Plan

### Top Priorities

| Priority | Goal | Rationale |
|---|---|---|
| **P1** | Bootstrap CI on 730-scenario holdout | Certify v2.2 +10.3pp margin over v1 for selection sign-off |
| **P1** | Deploy v2.2 in demo/API as default | App still documents v2.2; ensure `app/utils.py` serves latest model |
| **P1** | Evening-hour failure analysis | Largest accuracy drop for all comparators; likely highest store impact |
| **P2** | LLM `--input-mode=features` on holdout modal slice | Fair ablation at 19–28 items — does LLM close any gap without prose? |
| **P2** | Field-validate 16 JTBD divergence cases | Resolve whether formula or JTBD label is correct before trusting cook-now metrics |
| **P2** | Real-data integration plan | Synthetic → production POS/write-off pipeline; 6-month baseline per roadmap |
| **P3** | LLM latency reduction | ~11 s → ≤3 s budget (caching, smaller model, or rank-top-1-only output) |
| **P3** | Adversarial prompt hardening (v0.4) | Refusal/compliance on override-injection cases |

### Planned Improvements Based on Evaluation Findings

1. **Production path for v2.2** — wire model into associate-facing UI with plain-language explanations generated from v1 template engine (not LLM); target sub-500 ms inference.
2. **Hybrid trust layer** — use LLM-style refusal logic patterns (already 93% on OOS) as a lightweight guardrail module in front of ML ranking, without LLM latency cost.
3. **Retire v0.3 diagnostic set for selection claims** — all stakeholder metrics cite holdout modal slice or 730-scenario ML holdout only.
4. **Stop v2.3/v3 promotion** unless a retrain with real data closes the gap — both regressed −2.7 to −2.9pp vs v2.2 on honest test.
5. **Associate adherence pilot** — measure whether v2.2 recommendations are followed and whether write-off proxies improve (accuracy alone is insufficient for ROI proof).

---

## Artifacts & Reproduce

| Report | Content |
|---|---|
| `output/v2_2_temporal_report.json` | ML selection baseline (68.9%) |
| `output/llm_eval_v0.3_holdout_report.json` | Head-to-head ML vs LLM (Jun 27) |
| `output/llm_eval_v0.3_v0_3_report.json` | JTBD diagnostic (Jun 26) |
| `data/llm_eval_set_holdout.json` | 197-ex holdout eval set |
| `EVAL_METHODOLOGY.md` | Authoritative eval protocol v1.1 |

```bash
python scripts/build_eval_set_holdout.py
python notebooks/week9_llm_eval_runner.py --eval-set=holdout --prompt-version=v0.3 --no-llm   # ML only
python notebooks/week9_llm_eval_runner.py --eval-set=holdout --prompt-version=v0.3             # full run
docker compose up --build -d                                                                    # demo stack
```

---

**Bottom line:** Sprint 1 delivered a working, rigorously evaluated cook-sequencing prototype. **v2.2 ML is the production candidate** (68.9% holdout; 67.3% modal head-to-head). LLM benchmarking validated the architecture choice — learned ranking beats idealized human judgment at scale — while identifying trust dimensions (refusal, MPV) worth preserving in a hybrid Sprint 2 UX.
