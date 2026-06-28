# Evaluation Methodology — Hot Food Cook Order AI

**Version:** 1.1 | **Date:** 2026-06-27 | **Author:** Lisa Wang

This document is the authoritative source for how comparators are evaluated and how the production model is selected. Numbers in reports (`output/`) are only decision-relevant when interpreted against this protocol.

---

## 1. The decision question

> Which comparator (associate_floor / v1_rules / v2_2_ml / v2_3_ml / llm_vX.Y) should be used as the primary recommendation engine when deployed to real stores?

All other uses of eval results — understanding model behaviour, validating labels, exploring prompt design — are **diagnostic** and must not drive the selection decision.

**Working hypothesis (Jun 2026):** An LLM acting as an idealized human-judgment proxy will underperform learned ML on the real selection task at production scale (19–28 items). The holdout head-to-head eval supports this: LLM v0.3 **26.7%** vs v2.2 **67.3%** formula top-1 on the 150-ex modal slice.

---

## 2. Evaluation tiers

```
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 1 — SELECTION  (decides what ships)                           │
│  A) ML model pick: formula top-1 on 730-scenario temporal holdout   │
│     (date ≥ 2025-05-01, no leakage). Winner: v2_2_ml at 68.9%.      │
│  B) Head-to-head comparator: formula top-1 on MODAL SLICE ONLY      │
│     (n=150, holdout_clean=True) from holdout eval set. Same scale    │
│     as ML holdout (19–28 items). Compare LLM vs ML fairly here.    │
└─────────────────────────────────────────────────────────────────────┘
        ↓ pass / fail guardrails
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 2 — GUARDRAILS  (must pass, not maximised)                    │
│  must_precede_violation_rate, refusal_accuracy,                      │
│  latency_median_ms, parse_failures.                                  │
│  Evaluated on full holdout eval set (197 ex: 150 modal + 47         │
│  guardrails imported from v0.3).                                     │
└─────────────────────────────────────────────────────────────────────┘
        ↓ informs backlog
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 3 — DIAGNOSTIC  (reported, never decisive)                    │
│  JTBD cook-now%, dual-label divergence, Kendall τ, store/hour       │
│  breakdowns. Evaluated on v0.3 plain-language set (110 ex).         │
│  Confounds: prose input, 2–5 items, JTBD vs formula labels.         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Primary selection metric

### 3a. ML model selection (authoritative)

**Formula-label top-1 accuracy** on the **clean temporal holdout** (date ≥ 2025-05-01, 730 scenarios, 19–28 items each), with a **bootstrap 95% CI** (n=2,000 resamples).

- Source: `output/v2_2_temporal_report.json`, `output/v2_3_symmetric_report.json`, `output/v3_lambdarank_report.json`
- Current winner: **v2_2_ml 68.9%** vs v1_rules 58.6% (+10.3pp) vs v2_3_ml 66.0% vs v3 66.2%

### 3b. Head-to-head comparator eval (LLM vs ML at scale)

**Formula-label top-1 accuracy on the modal slice only** (`holdout_clean=True`, n=150) from the holdout eval set.

- Set: `data/llm_eval_set_holdout.json` (197 total; **150 modal** stratified from ML holdout + **47 guardrails**)
- Built by: `python scripts/build_eval_set_holdout.py`
- Run: `python notebooks/week9_llm_eval_runner.py --eval-set=holdout --prompt-version=v0.3`
- Report: `output/llm_eval_v0.3_holdout_report.json`

| Comparator | Modal formula top-1 (Jun 27 2026) | Notes |
|---|---|---|
| v2_2_ml | **67.3%** | Reference ML baseline on same 150 scenarios |
| v2_3_ml | 64.7% | Symmetric pairwise + proba agg |
| v1_rules | 56.0% | Canonical tie-breaking via `rank_from_features()` |
| llm_v0.3_zero_shot | **26.7%** | Native prose input; hypothesis confirmed |
| associate_floor | 14.0% | Multi-seed floor (20 seeds) |

**Do not** use blended accuracy across modal + guardrail slices for selection — guardrail examples (2–5 items, hand-authored) inflate LLM scores.

### Why not JTBD cook-now as primary?

JTBD labels are hand-authored and not yet field-validated; divergence cases intentionally differ from the formula. Until real write-off outcomes confirm which label is right, JTBD labels are **hypotheses**, not ground truth for selection.

### Why not v0.3 diagnostic set for selection?

The v0.3 set (110 examples, mostly 2–5 items) confounds scale, input format, and label source. LLM reaches ~90% formula top-1 with `--input-mode=features` on that set — but only **26.7%** on the 150-ex modal holdout slice at production item counts.

---

## 4. Guardrail thresholds

A model must pass **all** guardrails before selection, regardless of primary metric:

| Guardrail | Threshold | Rationale |
|---|---|---|
| `must_precede_violation_rate` | < 0.05 (< 5%) | Each violation is a potential waste event |
| `refusal_accuracy` | ≥ 90% on ≥ 20 OOS+adversarial examples | Trust dimension |
| `latency_median_ms` | ≤ 500 ms for ML; ≤ 3,000 ms for LLM | Associate tablet UX |
| Est. cost/decision | ≤ $0.002 USD | Operational budget |
| `parse_failures` | 0% | Non-parsing = undeployable |

### Current status (holdout eval set, Jun 27 2026)

| Comparator | MPV rate | Refusal | Parse | Latency | Pass? |
|---|---|---|---|---|---|
| v1_rules | 0.126 | n/a | 0 | 0 ms | FAIL (MPV) |
| v2_2_ml | 0.060 | n/a | 0 | ~100 ms | FAIL (MPV borderline) |
| v2_3_ml | 0.066 | n/a | 0 | ~98 ms | FAIL (MPV) |
| llm_v0.3 | **0.006** | **93.3%** (15 ex) | **0** | **~11 s** | FAIL (latency) |

On this eval set, **no comparator passes all guardrails** — selection remains deferred on guardrail grounds even though v2.2 wins the primary metric. MPV thresholds on a mixed guardrail set are stricter than on the raw 730-scenario holdout alone.

---

## 5. Decision rule

```
1. Compute primary metric + 95% CI for each ML comparator on the 730-scenario holdout.
2. For LLM vs ML at scale: compute formula top-1 on the 150-ex modal slice only.
3. Eliminate any comparator that fails a guardrail on the holdout eval set.
4. Select the comparator with the highest primary metric whose CI excludes 
   the runner-up's point estimate.
5. Tiebreak (CIs overlap): prefer simpler/cheaper/faster:
   v1_rules  >  v2_2_ml  >  llm_vX.Y
6. Document the winning CI and any guardrail margins in SPRINT1_SUMMARY.md.
```

**Current selection decision:** `v2_2_ml` (+10.3pp over v1_rules on 730-scenario holdout). Head-to-head holdout eval confirms LLM v0.3 is not competitive at scale (−40.6pp on modal slice). Guardrail certification on the 197-ex holdout eval set is incomplete for all comparators.

---

## 6. Fairness controls

### 6a. Input mode

Each model must be evaluated in **native** mode for deployment characteristics, and in **features** mode for apples-to-apples comparison.

| Mode | v1_rules | v2_2_ml / v2_3_ml | LLM |
|---|---|---|---|
| `native` | numeric features | numeric features | scenario_text (prose) |
| `features` | numeric features | numeric features | numeric feature table |
| `prose` | (future: thin parser) | (future) | scenario_text |

CLI: `--input-mode=native|features|prose` (default: native)

For **selection at scale**, native mode on the holdout eval set is the primary LLM result. Features mode is optional for fairness ablation.

### 6b. Scale stratification

Models behave differently by number of items. Report and compare within bands:

| Band | Item count | v2.2 regime | v1/LLM regime |
|---|---|---|---|
| small | 2–5 | Pairwise noise | Formula/prose strength |
| medium | 6–12 | Transitional | — |
| large | 13–28 | GBM advantage | Harder for LLM at scale |

Never aggregate across bands for the selection metric. The holdout eval modal slice is **large (13–28)** only.

### 6c. Label source transparency

Every per-example result records `jtbd_label`, `formula_label`, and `labels_agree`. Headline metrics state which label they use. A comparator's score changes depending on which label you compute against — both must be reported.

### 6d. Holdout cleanliness

Examples drawn from the training partition (pre-2025-05-01) are flagged `holdout_clean: false`. ML selection scores use only `holdout_clean: true` examples. In the holdout eval set:

- **150 modal** — `holdout_clean=True`, stratified from ML temporal holdout
- **47 guardrails** — `holdout_clean=False`, imported from v0.3 (edge/OOS/adversarial)

### 6e. v1 tie-breaking

`v1_rules` in the harness calls `CookSchedulerV1.rank_from_features()` with deterministic tie-breaking (score → hold_time → canonical `OVEN_ITEMS` order). This aligns v1 eval scores with label-agreement reconstruction in `data_labeler.py` (~56% on modal slice, consistent with ~58.6% on full holdout).

### 6f. LLM parse protocol

Ranking calls use hardened parsing to avoid infrastructure false negatives:

- `max_tokens=1024`, `temperature=0.0`
- Robust JSON extraction (fenced blocks, nested objects)
- Repair: dedupe extras, append missing valid items in canonical order
- Up to 2 retries with explicit JSON-shape reminder
- Final fallback: `CookSchedulerV1.rank_from_features()` (logged as ranking, not unparseable)

Refusal calls unchanged at `max_tokens=128`.

---

## 7. Statistical protocol

### 7a. Bootstrap CIs

All headline accuracy metrics (formula top-1, JTBD top-1, refusal accuracy) are reported with **bootstrap 95% CIs** (n=2,000 resamples at the example level, seed=42).

Interpretation: if the CI of comparator A excludes the point estimate of comparator B, the difference is statistically credible at this sample size.

- Holdout eval modal slice (n=150): σ ≈ 3.9pp → 95% CI ≈ ±7.6pp — sufficient to detect the ~40pp LLM vs v2.2 gap.
- v0.3 diagnostic (n=95 ranking): ~±7pp CI — useful for guardrails, not for selection.

### 7b. Paired significance (McNemar test)

Because all comparators see the **same examples**, use the McNemar test on paired correctness vectors rather than independent proportions. Two-tailed, corrected. p-value matrix included in every report.

### 7c. Stochastic floor (AssociateBaseline)

`AssociateBaseline` is randomised (habit/expiration/random mix). Report the mean ± std over `--assoc-seeds=20` seeds so the floor is not a single draw.

### 7d. LLM variance

When `--llm-samples=k` (k>1), run each ranking example k times; report mean ± std, majority-vote ranking, and parse-failure rate. Cost-gated: k=1 by default.

---

## 8. Eval sets

| Set | Purpose | Tier | N ranking | N refusal | Label source | Scale |
|---|---|---|---|---|---|---|
| **holdout** | **Head-to-head ML vs LLM** | **Selection** | 182 (150 modal + 32 guardrail ranking) | 15 | Formula (modal) + guardrails | 19–28 (modal) |
| v0.1 | Legacy formula A/B | Diagnostic | 42 | 8 | Formula (5-item) | 2–5 |
| v0.2 | 28-item formula A/B | Diagnostic | 42 | 11 | Formula (28-item) | 13–28 |
| v0.3 | JTBD plain-language | Diagnostic | 95 | 15 | JTBD + formula dual | 2–5 |
| 730-scenario holdout | ML training selection | Selection | 730 | — | Formula (temporal) | 19–28 |

### Holdout eval set composition (`--eval-set=holdout`)

Built by `scripts/build_eval_set_holdout.py`:

1. **150 modal** — proportional stratification by `store_type × hour_band` from `labeled_training_set.json` where `date ≥ 2025-05-01` and `holdout_clean=True`. Each example includes `scenario_text` (prose for LLM) and hidden `features` (for ML).
2. **47 guardrails** — imported from `llm_eval_set_v0.3.json`: 23 edge + 15 OOS + 9 adversarial. Used for MPV, refusal, and adversarial behaviour — **not** for primary accuracy comparison.

Regenerate: `python scripts/build_eval_set_holdout.py`

### Known confounds in v0.3 (do not use for selection)

1. **Scale mismatch:** 2–5 items vs 19–28 in production holdout.
2. **Input mismatch:** LLM gets prose; ML gets numeric features → prose advantage on small sets.
3. **JTBD labels:** Divergence cases where labels intentionally differ from formula.
4. **Inflated LLM scores:** 85%+ on guardrail/small-item slices vs 27% on modal holdout.

---

## 9. Backlog

| Priority | Task | Status |
|---|---|---|
| P1 | Bootstrap CI on 730-scenario holdout | Open |
| P1 | Holdout head-to-head eval (150 modal + guardrails) | **Done** (Jun 27 2026) |
| P1 | LLM parse hardening (0 parse failures) | **Done** (Jun 27 2026) |
| P1 | v1 canonical tie-breaking in harness | **Done** (Jun 27 2026) |
| P2 | features-mode LLM run on holdout modal slice | Open — fairness ablation |
| P2 | Field validation of divergence cases | Open |
| P2 | Real write-off outcome labels from live POS | Open — only true ground truth |
| P2 | Reconcile MPV guardrail pass on 730 holdout vs 197-ex eval set | Open |
| P3 | Chain-of-thought LLM prompt (v0.4) | Open |
| P3 | LLM latency reduction (currently ~11s vs 3s budget) | Open |

---

## 10. Report file map

| File | Content | Tier |
|---|---|---|
| `output/v2_2_temporal_report.json` | v2.2 honest holdout (730 scenarios) | Selection (ML) |
| `output/v2_3_symmetric_report.json` | v2.3 symmetric pairwise holdout | Selection (ML) |
| `output/v3_lambdarank_report.json` | v3 LambdaRank NDCG + top-1 | Selection (ML) |
| `output/labeling_report.json` | v1 label agreement (2,164 scenarios) | Selection (ML) |
| **`output/llm_eval_v0.3_holdout_report.json`** | **Head-to-head ML vs LLM (197 ex)** | **Selection (comparator)** |
| `output/llm_eval_v0.3_holdout_predictions.json` | Per-example holdout predictions | Selection (comparator) |
| `output/llm_parse_failure_inspection.json` | Parse-failure root-cause audit | Diagnostic |
| `output/llm_eval_v0.3_v0_3_report.json` | v0.3 JTBD full comparator set | Diagnostic |
| `output/llm_eval_v0.1_report.json` | Legacy 50-ex formula eval | Diagnostic |
| `output/llm_eval_v0.2_*_report.json` | 28-item formula eval | Diagnostic |
| `data/llm_eval_set_holdout.json` | Holdout eval set source | Selection |
| `EVAL_METHODOLOGY.md` | This file — authoritative protocol | — |
| `SPRINT1_SUMMARY.md` | Results summary with caveats | — |

---

## 11. Quick reference — commands

```bash
# Rebuild holdout eval set
python scripts/build_eval_set_holdout.py

# ML baselines only (~1 min)
python notebooks/week9_llm_eval_runner.py --eval-set=holdout --prompt-version=v0.3 --no-llm

# Full head-to-head (LLM + ML, ~30 min, requires ANTHROPIC_API_KEY)
python notebooks/week9_llm_eval_runner.py --eval-set=holdout --prompt-version=v0.3

# Fairness ablation (LLM gets numeric table)
python notebooks/week9_llm_eval_runner.py --eval-set=holdout --prompt-version=v0.3 --input-mode=features
```
