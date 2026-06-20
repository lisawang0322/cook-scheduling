"""Compare llm_eval_vX.Y_report.json files and produce a diff table.

Usage:
  python scripts/compare_llm_versions.py v0.1 v0.2
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")


def load_report(version: str) -> dict:
    path = os.path.join(OUTPUT_DIR, f"llm_eval_{version}_report.json")
    with open(path) as f:
        return json.load(f)


def llm_key(report: dict) -> str:
    """Find the LLM comparator key in results."""
    for k in report["results"]:
        if k.startswith("llm_"):
            return k
    return ""


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: compare_llm_versions.py <v_a> <v_b>")
        sys.exit(1)

    va, vb = sys.argv[1], sys.argv[2]
    ra, rb = load_report(va), load_report(vb)
    ka, kb = llm_key(ra), llm_key(rb)

    print(f"\n{'='*64}")
    print(f"  LLM PROMPT COMPARISON: {va} → {vb}")
    print(f"{'='*64}")
    print(f"  Model  : {ra['metadata']['llm_model']}")
    print(f"  Dataset: eval set v{ra['metadata']['eval_set_version']} ({ra['metadata']['total_examples']} examples)")

    # Overall
    ra_res, rb_res = ra["results"][ka], rb["results"][kb]
    delta_top1 = round(rb_res["top1_accuracy"] - ra_res["top1_accuracy"], 1)
    sign = "+" if delta_top1 >= 0 else ""
    print(f"\n  {'Metric':<30s} {va:>10s} {vb:>10s} {'Δ':>8s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8}")

    def row(label, a, b):
        if a is None or b is None:
            d = "n/a"
        else:
            d = f"{'+' if b-a >= 0 else ''}{round(b-a, 1)}"
        av = f"{a:.1f}%" if a is not None else "n/a"
        bv = f"{b:.1f}%" if b is not None else "n/a"
        print(f"  {label:<30s} {av:>10s} {bv:>10s} {d:>8s}")

    row("Overall Top-1", ra_res["top1_accuracy"], rb_res["top1_accuracy"])
    row("  Ranking accuracy", ra_res.get("ranking_accuracy"), rb_res.get("ranking_accuracy"))
    row("  Refusal accuracy (OOS+adv)", ra_res.get("refusal_accuracy"), rb_res.get("refusal_accuracy"))

    tau_a = ra_res.get("kendall_tau_mean")
    tau_b = rb_res.get("kendall_tau_mean")
    if tau_a and tau_b:
        td = f"{'+' if tau_b-tau_a >= 0 else ''}{round(tau_b-tau_a,3)}"
        print(f"  {'Kendall τ (mean)':<30s} {tau_a:>10.3f} {tau_b:>10.3f} {td:>8s}")

    row("  Parse failures", ra_res.get("parse_failures"), rb_res.get("parse_failures"))

    # By category
    print(f"\n  {'By Category':<30s} {va:>10s} {vb:>10s} {'Δ':>8s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8}")
    cats = sorted(set(ra["breakdowns"]["by_category"]) | set(rb["breakdowns"]["by_category"]))
    for cat in cats:
        a = ra["breakdowns"]["by_category"].get(cat, {}).get(ka)
        b = rb["breakdowns"]["by_category"].get(cat, {}).get(kb)
        row(f"  {cat}", a, b)

    # By eval tag
    print(f"\n  {'By Eval Tag':<30s} {va:>10s} {vb:>10s} {'Δ':>8s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8}")
    tags = sorted(set(ra["breakdowns"]["by_eval_tag"]) | set(rb["breakdowns"]["by_eval_tag"]))
    for tag in tags:
        a = ra["breakdowns"]["by_eval_tag"].get(tag, {}).get(ka)
        b = rb["breakdowns"]["by_eval_tag"].get(tag, {}).get(kb)
        row(f"  {tag}", a, b)

    # Save JSON
    comparison = {
        "versions": {va: ka, vb: kb},
        "model": ra["metadata"]["llm_model"],
        "eval_set_version": ra["metadata"]["eval_set_version"],
        "overall": {
            va: ra_res["top1_accuracy"],
            vb: rb_res["top1_accuracy"],
            "delta": round(rb_res["top1_accuracy"] - ra_res["top1_accuracy"], 1),
        },
        "ranking_accuracy": {
            va: ra_res.get("ranking_accuracy"),
            vb: rb_res.get("ranking_accuracy"),
        },
        "refusal_accuracy": {
            va: ra_res.get("refusal_accuracy"),
            vb: rb_res.get("refusal_accuracy"),
        },
        "kendall_tau": {
            va: tau_a,
            vb: tau_b,
        },
        "by_category": {
            cat: {
                va: ra["breakdowns"]["by_category"].get(cat, {}).get(ka),
                vb: rb["breakdowns"]["by_category"].get(cat, {}).get(kb),
            }
            for cat in cats
        },
        "by_eval_tag": {
            tag: {
                va: ra["breakdowns"]["by_eval_tag"].get(tag, {}).get(ka),
                vb: rb["breakdowns"]["by_eval_tag"].get(tag, {}).get(kb),
            }
            for tag in tags
        },
    }

    out_path = os.path.join(OUTPUT_DIR, "llm_eval_version_comparison.json")
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
