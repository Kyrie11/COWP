#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def fmt(x: Any) -> str:
    return "None" if x is None else f"{float(x):.6f}"


def metric_row(name: str, val: Any, op: str, thr: float, eps: float = 1e-9) -> bool:
    ok = False if val is None else ((float(val) <= thr + eps) if op == "<=" else (float(val) + eps >= thr))
    print(f"{name:34s} {fmt(val):>10s} {op} {thr:.6f}  {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Check OC-RAP v37 tail-budget regime targets from closed-loop/audit JSON outputs.")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--safe-nup-min", type=float, default=1.0)
    ap.add_argument("--safe-int-max", type=float, default=0.0)
    ap.add_argument("--near-miss-max", type=float, default=0.05)
    ap.add_argument("--near-nup-min", type=float, default=0.995)
    ap.add_argument("--contact-miss-max", type=float, default=0.05)
    ap.add_argument("--contact-pcd-min", type=float, default=0.472)
    ap.add_argument("--contact-fra-max", type=float, default=0.25)
    ap.add_argument("--contact-drs-min", type=float, default=0.75)
    ap.add_argument("--contact-nup-min", type=float, default=0.98)
    ap.add_argument("--contact-int-max", type=float, default=0.06)
    ap.add_argument("--offline-contact-nup-min", type=float, default=0.95)
    ap.add_argument("--offline-contact-int-max", type=float, default=0.12)
    ap.add_argument("--offline-contact-pcd-min", type=float, default=0.53)
    ap.add_argument("--offline-contact-fra-max", type=float, default=0.14)
    ap.add_argument("--offline-contact-drs-min", type=float, default=0.85)
    ap.add_argument("--near-int-max", type=float, default=0.02)
    args = ap.parse_args()

    root = args.run_dir
    files = {
        "safe": root / "closed_loop_safe_fast_v37.json",
        "near": root / "audit_near_contact_selected_topk_v37_v37.json",
        "contact": root / "audit_contact_selected_topk_v37_v37.json",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        print("missing files:", *missing, sep="\n  ")
        return 2
    safe, near, contact = (load(files[k]) for k in ("safe", "near", "contact"))

    print("===== selected-topk / closed-loop targets =====")
    ok_all = True
    ok_all &= metric_row("safe_intervention", safe.get("intervention_rate"), "<=", args.safe_int_max)
    ok_all &= metric_row("safe_NUP", safe.get("closed_loop_bounded_NUP"), ">=", args.safe_nup_min)
    ok_all &= metric_row("near_NUP", near.get("closed_loop_bounded_NUP"), ">=", args.near_nup_min)
    ok_all &= metric_row("near_intervention", near.get("intervention_rate"), "<=", args.near_int_max)
    ok_all &= metric_row("near_paper_PCD_miss", near.get("closed_loop_audit_paper_pcd_selector_miss_rate"), "<=", args.near_miss_max)
    ok_all &= metric_row("contact_paper_PCD_miss", contact.get("closed_loop_audit_paper_pcd_selector_miss_rate"), "<=", args.contact_miss_max)
    ok_all &= metric_row("contact_PCD", contact.get("closed_loop_post_contact_deployability"), ">=", args.contact_pcd_min)
    ok_all &= metric_row("contact_FRA", contact.get("closed_loop_FRA_exec"), "<=", args.contact_fra_max)
    ok_all &= metric_row("contact_DRS", contact.get("closed_loop_DRS"), ">=", args.contact_drs_min)
    ok_all &= metric_row("contact_NUP", contact.get("closed_loop_bounded_NUP"), ">=", args.contact_nup_min)
    ok_all &= metric_row("contact_intervention", contact.get("intervention_rate"), "<=", args.contact_int_max)

    print("contact macro_counts:", contact.get("macro_counts"))
    print("contact reason_counts:", contact.get("selection_reason_counts"))
    print("contact paper miss best macros:", contact.get("audit_paper_pcd_miss_best_macro_counts"))
    print("near macro_counts:", near.get("macro_counts"))
    print("near paper miss best macros:", near.get("audit_paper_pcd_miss_best_macro_counts"))

    offline = root / "eval_contact_v37_v37.json"
    if offline.exists():
        d = load(offline)
        o = d.get("methods", {}).get("ocrap", {})
        print("===== offline contact sanity =====")
        ok_all &= metric_row("offline_contact_NUP", o.get("bounded_NUP"), ">=", args.offline_contact_nup_min)
        ok_all &= metric_row("offline_contact_intervention", o.get("intervention_rate"), "<=", args.offline_contact_int_max)
        ok_all &= metric_row("offline_contact_PCD", o.get("post_contact_deployability"), ">=", args.offline_contact_pcd_min)
        ok_all &= metric_row("offline_contact_FRA", o.get("FRA_exec"), "<=", args.offline_contact_fra_max)
        ok_all &= metric_row("offline_contact_DRS", o.get("DRS"), ">=", args.offline_contact_drs_min)
        scalar_path = root / "eval_contact_v37_scalar.json"
        if scalar_path.exists():
            sc = load(scalar_path).get("methods", {}).get("ocrap", {})
            print("offline contact scalar OCRAP:", {k: sc.get(k) for k in ["FRA_exec", "DRS", "bounded_NUP", "post_contact_deployability", "intervention_rate", "selection_reason_counts"]})
        print("offline contact OCRAP:", {k: o.get(k) for k in ["FRA_exec", "DRS", "bounded_NUP", "post_contact_deployability", "intervention_rate", "selection_reason_counts"]})
    else:
        print("offline contact sanity skipped: missing", offline)

    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
