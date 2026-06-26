# Evaluation Methodology — 7-Eleven Cook Order AI

**Version:** 1.0 | **Date:** 2026-06-26 | **Author:** Lisa Wang

This document is the authoritative source for how comparators are evaluated and how the production model is selected. Numbers in reports (`output/`) are only decision-relevant when interpreted against this protocol.

---

## 1. The decision question

> Which comparator (associate_floor / v1_rules / v2_2_ml / llm_vX.Y) should be used as the primary recommendation engine when deployed to real stores?

All other uses of eval results — understanding model behaviour, validating labels, exploring prompt design — are **diagnostic** and must not drive the selection decision.

---

## 2. Evaluation tiers

```
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 1 — SELECTION  (decides what ships)                           │
│  Primary metric + guardrails. CI must exclude runner-up.            │
│  Source: clean temporal holdout (date ≥ 2025-05-01, n=730)          │
│  Formula-label accuracy, scale-stratified, no leakage.              │
└─────────────────────────────────────────────────────────────────────┘
        ↓ pass / fail guardrails
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 2 — GUARDRAILS  (must pass, not maximised)                    │
│  must_precede_violation_rate, refusal_accuracy,                      │
│  latency_median_ms, estimated cost/decision.                         │
│  Evaluated on the same holdout + v0.3 refusal set.                  │
└─────────────────────────────────────────────────────────────────────┘
        ↓ informs backlog
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 3 — DIAGNOSTIC  (reported, never decisive)                    │
│  JTBD cook-now%, dual-label divergence, Kendall τ, store/hour       │
│  breakdowns. Evaluated on v0.3 plain-language set (n=45 ranking).   │
│  Confounds: prose input, small N, JTBD vs formula labels.           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Primary selection metric

**Formula-label top-1 accuracy** on the **clean temporal holdout** (date ≥ 2025-05-01, 730 scenarios, 19–28 items each), with a **bootstrap 95% CI** (n=2,000 resamples).

- This is what `v2_2_ml_top1_pct` in `output/v2_2_temporal_report.json` measures.
- For LLM comparators, re-run on the same holdout features (input-mode=features) to eliminate the prose-format advantage.
- `v2_2_ml` is the current selection winner: **68.9% [CI: run harness]** vs v1_rules 58.6%.

### Why not JTBD cook-now as primary?

JTBD labels are hand-authored and not yet field-validated; 7 of 45 examples intentionally diverge from the formula. Until real write-off outcomes can confirm which label is right, JTBD labels are **hypotheses**, not ground truth for selection.

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

Current status:
- v1_rules and v2_2_ml: all ML guardrails pass. MPV rate: v1=0.11, v2.2=0.18 (borderline — both below 0.05 threshold on holdout).
- LLM v0.3 zero-shot: refusal_accuracy 80% on 5 examples (n too small to certify); latency ~1,400 ms median. **Refusal guardrail not yet certifiable.**

---

## 5. Decision rule

```
1. Compute primary metric + 95% CI for each comparator on the clean holdout.
2. Eliminate any comparator that fails a guardrail.
3. Select the comparator with the highest primary metric whose CI excludes 
   the runner-up's point estimate.
4. Tiebreak (CIs overlap): prefer simpler/cheaper/faster:
   v1_rules  >  v2_2_ml  >  llm_vX.Y
5. Document the winning CI and any guardrail margins in SPRINT1_SUMMARY.md.
```

**Current selection decision:** `v2_2_ml` (+10.3pp over v1_rules on 730-scenario holdout). CI not yet bootstrapped on holdout — add this in Phase 1.

---

## 6. Fairness controls

### 6a. Input mode

Each model must be evaluated in **native** mode for deployment characteristics, and in **features** mode for apples-to-apples comparison.

| Mode | v1_rules | v2_2_ml | LLM |
|---|---|---|---|
| `native` | numeric features | numeric features | scenario_text (prose) |
| `features` | numeric features | numeric features | numeric feature table |
| `prose` | (future: thin parser) | (future) | scenario_text |

CLI: `--input-mode=native|features|prose` (default: native)

### 6b. Scale stratification

Models behave differently by number of items. Report and compare within bands:

| Band | Item count | v2.2 regime | v1/LLM regime |
|---|---|---|---|
| small | 2–5 | Pairwise noise | Formula/prose strength |
| medium | 6–12 | Transitional | — |
| large | 13–28 | GBM advantage | Harder for formula |

Never aggregate across bands for the selection metric.

### 6c. Label source transparency

Every per-example result records `jtbd_label`, `formula_label`, and `labels_agree`. Headline metrics state which label they use. A comparator's score changes depending on which label you compute against — both must be reported.

### 6d. Holdout cleanliness

Examples drawn from the training partition (pre-2025-05-01) are flagged `holdout_clean: false`. v2.2 selection scores are computed only on `holdout_clean: true` examples. Leakage examples are reported separately as a diagnostic slice.

---

## 7. Statistical protocol

### 7a. Bootstrap CIs

All headline accuracy metrics (formula top-1, JTBD top-1, refusal accuracy) are reported with **bootstrap 95% CIs** (n=2,000 resamples at the example level, seed=42).

Interpretation: if the CI of comparator A excludes the point estimate of comparator B, the difference is statistically credible at this sample size.

Current v0.3 n=45 ranking examples: a 6.7pp gap (88.9% vs 82.2%) has a wide CI — likely not credible. Run on n≥120 for meaningful CIs.

### 7b. Paired significance (McNemar test)

Because all comparators see the **same examples**, use the McNemar test on paired correctness vectors rather than independent proportions. Two-tailed, corrected. p-value matrix included in every report.

### 7c. Stochastic floor (AssociateBaseline)

`AssociateBaseline` is randomised (habit/expiration/random mix). Report the mean ± std over `--assoc-seeds=20` seeds so the floor is not a single draw.

### 7d. LLM variance

When `--llm-samples=k` (k>1), run each ranking example k times; report mean ± std, majority-vote ranking, and parse-failure rate. Cost-gated: k=1 by default.

---

## 8. Eval sets

| Set | Purpose | Tier | N ranking | N refusal | Label source |
|---|---|---|---|---|---|
| v0.1 | Legacy formula A/B | Diagnostic | 42 | 8 | Formula (5-item) |
| v0.2 | 28-item formula A/B | Diagnostic | 42 | 11 | Formula (28-item) |
| v0.3 | JTBD plain-language | **Diagnostic** | 45 | 5 | JTBD + formula dual |
| holdout | **Selection** | Selection | 730 | — | Formula (temporal) |

### Known confounds in v0.3 (do not use for selection)

1. **Input mismatch:** LLM gets prose; ML gets numeric features → prose advantage is unquantified.
2. **Small N:** 45 ranking examples gives ~±7pp CI — not discriminating enough at typical accuracy gaps.
3. **JTBD labels:** 7 divergence cases where labels intentionally differ from formula. Until field-validated, these are ambiguous.
4. **Leakage not tested:** v0.3 examples are hand-authored, no training-partition overlap, but scale does not represent the holdout regime (2-5 vs 19-28 items).

---

## 9. Backlog (not yet done)

| Priority | Task | Unblocks |
|---|---|---|
| P1 | Bootstrap CI on 730-scenario holdout | Section 5 decision rule |
| P1 | Grow v0.3 to ≥120 ranking + ≥20 refusal examples | Useful CIs on diagnostic tier |
| P1 | features-mode LLM run on holdout | Apples-to-apples vs ML |
| P2 | Field validation of 7 divergence cases | JTBD label credibility |
| P2 | Real write-off outcome labels from live POS | Only true ground truth |
| P3 | Chain-of-thought LLM prompt (v0.4) | May improve LLM refusal + adversarial |

---

## 10. Report file map

| File | Content | Tier |
|---|---|---|
| `output/v2_2_temporal_report.json` | v2.2 honest holdout (730 scenarios) | Selection |
| `output/labeling_report.json` | v1 label agreement (2,164 scenarios) | Selection |
| `output/llm_eval_v0.3_v0_3_report.json` | v0.3 JTBD full comparator set | Diagnostic |
| `output/llm_eval_v0.1_report.json` | Legacy 50-ex formula eval | Diagnostic |
| `output/llm_eval_v0.2_*_report.json` | 28-item formula eval | Diagnostic |
| `EVAL_METHODOLOGY.md` | This file — authoritative protocol | — |
| `SPRINT1_SUMMARY.md` | Results summary with caveats | — |
