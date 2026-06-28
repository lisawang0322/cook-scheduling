# Versioned Evaluation Results — Hot Food Cook Order AI

**Author:** Lisa Wang | **Updated:** June 27, 2026  
**LLM model:** claude-sonnet-4-6 | **Protocol:** [`EVAL_METHODOLOGY.md`](EVAL_METHODOLOGY.md)

Numbers are decision-relevant only when interpreted against the tier definitions in [`EVAL_METHODOLOGY.md`](EVAL_METHODOLOGY.md). See [`SPRINT1_SUMMARY.md`](SPRINT1_SUMMARY.md) for prototype overview, failure modes, iterations, and Sprint 2 plan.

---

## Tier 1 — ML Model Selection (730-scenario holdout, authoritative)

Temporal split: date ≥ 2025-05-01, 730 test scenarios, 19–28 items each. Formula-label top-1 accuracy.

| Version | Approach | Test top-1 | vs associate (~8.9%) |
|---|---|---|---|
| associate_floor | Simulated flawed associate behavior | ~8.9% | — |
| v1 rules | Urgency × density × waste penalty | **58.6%** | +49.7pp |
| **v2.2 ML** | Pairwise GBM + temporal split + soft labels | **68.9%** | **+60.0pp** |
| v2.3 ML | Symmetric pairs + proba aggregation | 66.0% | −2.9pp vs v2.2 |
| v3 ML | LightGBM LambdaRank (listwise NDCG) | 66.2% | −2.7pp vs v2.2 |

**Selection winner:** **v2.2 ML** — best honest generalization on the full holdout.

| v3 detail | Value |
|---|---|
| NDCG@1 (test) | 0.723 |
| NDCG@3 (test) | 0.753 |
| NDCG@5 (test) | 0.797 |
| Full-dataset top-1 | 71.6% |

*Sprint 1 5-item baseline (superseded): v1 71.8%, v2.2 74.3%, n=1,747.*

**Reports:** `output/v2_2_temporal_report.json`, `output/v2_3_symmetric_report.json`, `output/v3_lambdarank_report.json`

---

## Tier 1 — Head-to-Head at Scale (holdout eval set, Jun 27 2026)

**Set:** 197 examples = **150 modal** (stratified from ML holdout, 19–28 items) + **47 guardrails** (23 edge + 15 OOS + 9 adversarial from v0.3).  
**Built by:** `python scripts/build_eval_set_holdout.py`  
**Run:** `python notebooks/week9_llm_eval_runner.py --eval-set=holdout --prompt-version=v0.3`  
**Metric:** Formula top-1 on **modal slice only** (do not blend guardrail slices for selection).

| Comparator | Modal formula top-1 | Refusal | Parse failures | Latency |
|---|---|---|---|---|
| associate_floor | 14.0% | — | 0 | 0 ms |
| v1_rules | 56.7% | — | 0 | 0 ms |
| **v2_2_ml** | **67.3%** | — | 0 | ~100 ms |
| v2_3_ml | 64.7% | — | 0 | ~98 ms |
| LLM v0.3 (native prose) | **26.7%** | 93.3% (14/15) | 0 | ~11 s |

**Key finding:** LLM underperforms v2.2 by **−40.6pp** on the modal slice — **hypothesis confirmed**: idealized human judgment does not beat learned ML at production item counts. LLM passes MPV (1 violation) and refusal guardrails but fails latency (~11s vs 3s budget).

**Report:** `output/llm_eval_v0.3_holdout_report.json`

### By category (formula top-1)

| Category | associate | v1 | v2.2 | v2.3 | LLM v0.3 |
|---|---|---|---|---|---|
| **modal** (n=150) | 14.0% | 56.7% | **67.3%** | 64.7% | 26.7% |
| edge (n=23) | 34.8% | 30.4% | 73.9% | 69.6% | 82.6% |
| adversarial (n=9) | 55.6% | 66.7% | 88.9% | 88.9% | 66.7% |
| OOS (n=15) | — | — | — | — | **93.3%** refusal |

### By hour band (modal slice)

| Hour band | v2.2 | LLM v0.3 |
|---|---|---|
| morning | 91.9% | 59.5% |
| lunch | 88.9% | 55.6% |
| afternoon | 93.8% | 43.8% |
| **evening** | **39.0%** | **11.7%** |

### By item count band

| Band | v2.2 | LLM v0.3 | Notes |
|---|---|---|---|
| large (13–28) | 67.3% | 26.7% | Modal holdout — selection metric |
| small (2–5) | 78.1% | 83.0% | Guardrail cases only — do not use for selection |

### Key metrics (holdout eval, all ranking n=182)

| Metric | v2.2 ML | LLM v0.3 |
|---|---|---|
| Formula top-1 (overall blended) | 69.2% | 35.7% |
| Kendall τ mean | 0.53 | 0.44 |
| MPV violation rate | 0.060 | **0.006** |
| McNemar vs v2.2 | — | p < 0.001 |

---

## Tier 3 — Diagnostic Eval Sets (do not use for selection)

| Eval set | N | Scale | LLM prompt | LLM formula top-1 | v2.2 | Notes |
|---|---|---|---|---|---|---|
| **v0.1 shared** | 50 | 5-item | v0.1 | 50.0% | 81.0% | Sprint 1 baseline |
| **v0.2 shared** | 53 | 28-item | v0.2 | 64.0% | 81.0% | v0.2 +13pp modal vs v0.1 |
| **v0.3 JTBD** | 110 | 2–5 items | v0.3 native | 78.9% | 91.6% | LLM **95.8%** JTBD; prose confound |
| **v0.3 fair input** | 110 | 2–5 items | v0.3 + `--input-mode=features` | **90.4%** | 91.6% | No LLM edge (McNemar p=1.0) |

v0.3 high scores reflect **small-item scale + prose input + JTBD labels** — not comparable to the 730-scenario holdout. Fair input ablation shows LLM ≈ v1/v2.2; the "LLM wins" story was a confound.

### 50-ex shared eval (legacy, all comparators)

| Comparator | Overall top-1 | Ranking | Refusal | Kendall τ |
|---|---|---|---|---|
| associate_floor | 52.4% | 52.4% | n/a | 0.436 |
| v1_rules | 78.6% | 78.6% | n/a | 0.730 |
| v2_2_ml | **81.0%** | **81.0%** | n/a | **0.762** |
| llm_v0.1_zero_shot | 50.0% | 45.2% | 75.0% | 0.381 |
| llm_v0.2_zero_shot | **64.0%** | **61.9%** | **75.0%** | **0.476** |

### LLM by category (v0.2, 50-ex set)

| Category | v0.1 | v0.2 | Δ |
|---|---|---|---|
| modal (30) | 46.7% | **60.0%** | +13.3pp |
| edge (12) | 41.7% | **66.7%** | +25.0pp |
| OOS (5) | 100.0% | **100.0%** | 0 |
| adversarial (3) | 33.3% | 33.3% | 0 |
| divergence (n=4) | 0.0% | 0.0% | open question |

### v0.3 diagnostic (110 ex, Jun 26 2026)

| Comparator | JTBD top-1 | Formula top-1 | Refusal | MPV |
|---|---|---|---|---|
| associate_floor | 66.3% ±4.4 | 50.5% | — | 41 |
| v1_rules | 89.5% | **92.6%** | — | 8 |
| v2_2_ml | 83.2% | 91.6% | — | 16 |
| LLM native | **95.8%** | 78.9% | **93.3%** | **2** |
| LLM `--input-mode=features` | 93.6% | **90.4%** | 93.3% | 1 |

**Dual-label breakdown (79 agrees / 16 divergence):**

| Slice | LLM JTBD (native) | LLM Formula (native) | LLM Formula (features) | v1 Formula | v2.2 Formula |
|---|---|---|---|---|---|
| Overall | 95.8% | 78.9% | **90.4%** | 92.6% | 91.6% |
| Labels agree | 96.2% | 93.7% | 98.7% | 94.9% | 91.1% |
| Labels diverge | 93.8% | 6.2% | 50.0% | 81.2% | 93.8% |

**Reports:** `output/llm_eval_v0.1_report.json`, `output/llm_eval_v0.2_report.json`, `output/llm_eval_v0.3_v0_3_report.json`, `output/llm_eval_v0.3_v0_3_features_report.json`

---

## Prompt & Harness Evolution

| Version | Change | Outcome |
|---|---|---|
| LLM v0.1 | Initial zero-shot benchmark | 50% on 50-ex set |
| LLM v0.2 | Associate-voice tightening | 64% overall; +25pp edge cases |
| LLM v0.3 | Plain-language JTBD scenarios | Strong on diagnostic set; **26.7%** modal holdout |
| Harness | `--eval-set=holdout`, parse hardening, v1 tie-break fix | 0 parse failures; fair ML vs LLM comparison |

**Harness features:** bootstrap 95% CIs, McNemar paired significance, multi-seed associate floor (`--assoc-seeds=20`), `--input-mode=native|features`, scale stratification, selection scorecard, holdout-clean slice.

**Parse hardening (Jun 27 2026):** `max_tokens=1024`, robust JSON extraction/repair, deterministic retries, v1 fallback — parse failures **137 → 0** on holdout run.

---

## Reproduce

```bash
# Rebuild holdout eval set
python scripts/build_eval_set_holdout.py

# ML baselines only (~1 min)
python notebooks/week9_llm_eval_runner.py --eval-set=holdout --prompt-version=v0.3 --no-llm

# Full head-to-head (LLM + ML, ~30 min)
python notebooks/week9_llm_eval_runner.py --eval-set=holdout --prompt-version=v0.3

# v0.3 diagnostic (not for selection)
python notebooks/week9_llm_eval_runner.py --eval-set=v0.3 --prompt-version=v0.3

# Fair input ablation
python notebooks/week9_llm_eval_runner.py --eval-set=v0.3 --prompt-version=v0.3 --input-mode=features
```

---

## Summary

| Question | Answer |
|---|---|
| **Which ML model ships?** | **v2.2** (68.9% on 730 holdout; 67.3% on 150-ex modal head-to-head) |
| **Does LLM replace ML?** | **No** — 26.7% modal vs 67.3% v2.2 at production scale |
| **Which eval set for selection claims?** | 730-scenario ML holdout + holdout eval **modal slice only** |
| **Which eval sets are diagnostic only?** | v0.1, v0.2, v0.3 (especially v0.3 at 2–5 items) |
