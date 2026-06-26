"""Compare two eval report JSON files and print a diff table with CIs and significance.

Generalizes compare_llm_versions.py to work with any pair of reports from
week9_llm_eval_runner.py — not just LLM prompt A/B comparisons.

Usage:
  # Compare two report files by path
  python scripts/compare_reports.py output/llm_eval_v0.3_v0_3_report.json \\
                                     output/llm_eval_v0.3_v0_3_features_report.json

  # Compare by shorthand name (looks in output/ directory)
  python scripts/compare_reports.py v0.3_v0_3 v0.3_v0_3_features

Options:
  --comparator=NAME   Focus diff on a specific comparator (default: all shared)
  --metric=KEY        Primary metric to diff (default: formula_top1_accuracy)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def resolve_path(arg: str) -> str:
    if os.path.exists(arg):
        return arg
    # Try output/<arg>_report.json
    candidate = os.path.join(OUTPUT_DIR, f"llm_eval_{arg}_report.json")
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        f"Report not found: {arg!r}. "
        f"Tried {candidate!r}. Pass a full path or a report shorthand like 'v0.3_v0_3'."
    )


def load_report(arg: str) -> tuple[str, dict]:
    path = resolve_path(arg)
    with open(path) as f:
        return path, json.load(f)


# ---------------------------------------------------------------------------
# Diff helpers
# ---------------------------------------------------------------------------

def delta_str(a: float | None, b: float | None, pct: bool = True) -> str:
    if a is None or b is None:
        return "n/a"
    d = b - a
    suffix = "pp" if pct else ""
    return f"{'+' if d >= 0 else ''}{d:.1f}{suffix}"


def fmt_pct(v: float | None) -> str:
    return f"{v:.1f}%" if v is not None else "n/a"


def fmt_ci(ci_block: dict | None) -> str:
    if not ci_block:
        return "n/a"
    lo, hi = ci_block.get("ci_lo"), ci_block.get("ci_hi")
    if lo is None or hi is None:
        return "n/a"
    return f"[{lo*100:.1f}–{hi*100:.1f}%]"


# ---------------------------------------------------------------------------
# Report diff
# ---------------------------------------------------------------------------

def compare_reports(
    path_a: str,
    report_a: dict,
    path_b: str,
    report_b: dict,
    target_comparator: str | None = None,
    primary_metric: str = "formula_top1_accuracy",
) -> dict[str, Any]:
    meta_a = report_a.get("metadata", {})
    meta_b = report_b.get("metadata", {})
    results_a = report_a.get("results", {})
    results_b = report_b.get("results", {})
    cis_a = report_a.get("bootstrap_cis", {})
    cis_b = report_b.get("bootstrap_cis", {})
    sig_a = report_a.get("significance", {})

    shared_names = set(results_a) & set(results_b)
    if target_comparator:
        shared_names = {n for n in shared_names if target_comparator in n}

    print(f"\n{'='*72}")
    print("  REPORT COMPARISON")
    print(f"{'='*72}")
    print(f"  Report A: {os.path.basename(path_a)}")
    print(f"    eval-set={meta_a.get('eval_set_version')}  "
          f"input-mode={meta_a.get('input_mode', 'native')}  "
          f"date={meta_a.get('run_date', '')[:10]}")
    print(f"  Report B: {os.path.basename(path_b)}")
    print(f"    eval-set={meta_b.get('eval_set_version')}  "
          f"input-mode={meta_b.get('input_mode', 'native')}  "
          f"date={meta_b.get('run_date', '')[:10]}")
    print(f"  Shared comparators: {sorted(shared_names)}")
    print(f"  Primary metric: {primary_metric}")

    comparison: dict[str, Any] = {
        "report_a": path_a,
        "report_b": path_b,
        "meta_a": {k: meta_a.get(k) for k in ("eval_set_version", "input_mode", "run_date")},
        "meta_b": {k: meta_b.get(k) for k in ("eval_set_version", "input_mode", "run_date")},
        "primary_metric": primary_metric,
        "comparators": {},
    }

    def _metric(results: dict, name: str, key: str) -> float | None:
        return results.get(name, {}).get(key)

    METRICS = [
        ("formula_top1_accuracy",   "Formula Top-1 %",  True),
        ("cook_now_accuracy",        "JTBD Cook-Now %",  True),
        ("cook_now_set_recall",      "Set Recall",       False),
        ("must_precede_violation_rate", "MPV Rate",      False),
        ("refusal_accuracy",         "Refusal % ",       True),
        ("kendall_tau_mean",         "Kendall τ",        False),
        ("latency_median_ms",        "Latency (ms)",     False),
        ("parse_failures",           "Parse Failures",   False),
    ]

    for name in sorted(shared_names):
        ra = results_a.get(name, {})
        rb = results_b.get(name, {})
        ci_a = cis_a.get(name, {})
        ci_b = cis_b.get(name, {})

        print(f"\n  ── Comparator: {name} ──")
        print(f"  {'Metric':<35s} {'Report A':>12s} {'CI(A)':>16s} {'Report B':>12s} {'CI(B)':>16s} {'Δ':>8s}")
        print(f"  {'-'*35} {'-'*12} {'-'*16} {'-'*12} {'-'*16} {'-'*8}")

        comp_diff: dict[str, Any] = {}
        for mk, label, is_pct in METRICS:
            va = _metric(results_a, name, mk)
            vb = _metric(results_b, name, mk)
            # Which CI sub-key maps to this metric?
            ci_key = None
            if mk in ("formula_top1_accuracy", "formula_correct"):
                ci_key = "formula_ci"
            elif mk in ("cook_now_accuracy", "cook_now_correct"):
                ci_key = "cook_now_ci"
            elif mk == "cook_now_set_recall":
                ci_key = "set_recall_ci"
            elif mk == "must_precede_violation_rate":
                ci_key = "mpv_rate_ci"
            ci_a_s = fmt_ci(ci_a.get(ci_key)) if ci_key else "n/a"
            ci_b_s = fmt_ci(ci_b.get(ci_key)) if ci_key else "n/a"
            fmt_a = fmt_pct(va) if is_pct else (f"{va:.4f}" if va is not None else "n/a")
            fmt_b = fmt_pct(vb) if is_pct else (f"{vb:.4f}" if vb is not None else "n/a")
            d = delta_str(va, vb, pct=is_pct)
            print(f"  {label:<35s} {fmt_a:>12s} {ci_a_s:>16s} {fmt_b:>12s} {ci_b_s:>16s} {d:>8s}")
            comp_diff[mk] = {"a": va, "b": vb, "delta": round(vb - va, 4) if va is not None and vb is not None else None}

        comparison["comparators"][name] = comp_diff

    # Category breakdown diff
    print(f"\n  {'By Category':<35s} {'Report A':>12s} {'Report B':>12s} {'Δ':>8s}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*8}")
    cats = sorted(set(report_a.get("breakdowns", {}).get("by_category", {}))
                  | set(report_b.get("breakdowns", {}).get("by_category", {})))
    for cat in cats:
        for name in sorted(shared_names):
            va = report_a["breakdowns"].get("by_category", {}).get(cat, {}).get(name)
            vb = report_b["breakdowns"].get("by_category", {}).get(cat, {}).get(name)
            d = delta_str(va, vb, pct=True)
            print(f"  {cat+'/'+name:<35s} {fmt_pct(va):>12s} {fmt_pct(vb):>12s} {d:>8s}")

    # Scale breakdown diff
    print(f"\n  {'By Scale Band':<35s} {'Report A':>12s} {'Report B':>12s} {'Δ':>8s}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*8}")
    bands = sorted(set(report_a.get("breakdowns", {}).get("by_item_count_band", {}))
                   | set(report_b.get("breakdowns", {}).get("by_item_count_band", {})))
    for band in bands:
        for name in sorted(shared_names):
            va = report_a["breakdowns"].get("by_item_count_band", {}).get(band, {}).get(name)
            vb = report_b["breakdowns"].get("by_item_count_band", {}).get(band, {}).get(name)
            d = delta_str(va, vb, pct=True)
            print(f"  {band+'/'+name:<35s} {fmt_pct(va):>12s} {fmt_pct(vb):>12s} {d:>8s}")

    # McNemar cross-report note
    print(f"\n  NOTE: McNemar significance shown within each report (requires same eval set).")
    if "significance" in report_a:
        print("  Significance matrix for Report A:")
        for na, row in sorted(report_a["significance"].items()):
            for nb, p in sorted(row.items()):
                if p is not None and na < nb:
                    star = "*" if p < 0.05 else ""
                    print(f"    {na} vs {nb}: p={p:.4f}{star}")

    # Save JSON comparison
    out_path = os.path.join(OUTPUT_DIR, "report_comparison.json")
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Saved → {out_path}")
    return comparison


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=", 1)[0][2:]: a.split("=", 1)[1]
            for a in sys.argv[1:] if a.startswith("--") and "=" in a}

    if len(args) < 2:
        print("Usage: compare_reports.py <report_a> <report_b> [--comparator=NAME] [--metric=KEY]")
        print("  Each arg can be a file path or a shorthand (e.g. 'v0.3_v0_3').")
        sys.exit(1)

    path_a, report_a = load_report(args[0])
    path_b, report_b = load_report(args[1])
    target_comparator = opts.get("comparator")
    primary_metric = opts.get("metric", "formula_top1_accuracy")

    compare_reports(path_a, report_a, path_b, report_b, target_comparator, primary_metric)


if __name__ == "__main__":
    main()
