#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def check(name: str, value: Any, op: str, threshold: float, *, required: bool = True) -> bool:
    if value is None:
        ok = not required
        print(f"{name:42s} {'missing':>12s} {op} {threshold:.6f}  {'SKIP' if ok else 'FAIL'}")
        return ok
    v = float(value)
    ok = v <= threshold + 1e-9 if op == "<=" else v + 1e-9 >= threshold
    print(f"{name:42s} {v:12.6f} {op} {threshold:.6f}  {'PASS' if ok else 'FAIL'}")
    return ok


def paired_delta(report: dict[str, Any] | None, metric: str) -> tuple[float | None, tuple[float, float] | None]:
    if not report:
        return None, None
    row = (report.get("metrics") or {}).get(metric)
    if not row:
        return None, None
    ci = row.get("bootstrap_95ci")
    return float(row.get("paired_delta")), (float(ci[0]), float(ci[1])) if ci else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Check v40 development targets and paired physical-effect requirements.")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--near-paired", type=Path)
    ap.add_argument("--contact-paired", type=Path)
    ap.add_argument("--publication", action="store_true", help="Require paired CIs to exclude zero and use stricter miss limits.")
    args = ap.parse_args()
    r = args.run_dir
    paths = {
        "safe": r / "closed_loop_safe_fast_v40.json",
        "near": r / "audit_near_contact_selected_topk_v40_v40.json",
        "contact": r / "audit_contact_selected_topk_v40_v40.json",
        "offline": r / "eval_contact_v40_v40.json",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        print("missing files:", *missing, sep="\n  ")
        return 2
    safe, near, contact, offline = (load(paths[k]) for k in ("safe", "near", "contact", "offline"))
    near_pair = load(args.near_paired) if args.near_paired and args.near_paired.exists() else None
    contact_pair = load(args.contact_paired) if args.contact_paired and args.contact_paired.exists() else None
    miss_max = 0.025 if args.publication else 0.034
    ok = True

    print("===== safe: strict nominal preservation =====")
    ok &= check("safe intervention", safe.get("intervention_rate"), "<=", 0.0)
    ok &= check("safe intervention episodes", safe.get("intervention_episode_count", 0), "<=", 0.0)
    ok &= check("safe bounded NUP", safe.get("closed_loop_bounded_NUP"), ">=", 0.999)

    print("===== near-contact: low-cost physical margin recovery =====")
    ok &= check("near paper-PCD miss", near.get("closed_loop_audit_paper_pcd_selector_miss_rate"), "<=", miss_max)
    ok &= check("near PCD", near.get("closed_loop_post_contact_deployability"), ">=", 0.54)
    ok &= check("near FRA", near.get("closed_loop_FRA_exec"), "<=", 0.12)
    ok &= check("near DRS", near.get("closed_loop_DRS"), ">=", 0.88)
    ok &= check("near bounded NUP", near.get("closed_loop_bounded_NUP"), ">=", 0.995)
    ok &= check("near intervention", near.get("intervention_rate"), "<=", 0.020)
    ok &= check("near intervention episode rate", near.get("intervention_episode_rate"), "<=", 0.012)
    ok &= check("near max intervention run", near.get("max_intervention_run_length"), "<=", 1.0)
    d_clear, ci_clear = paired_delta(near_pair, "min_clearance_m_min")
    d_ttc, ci_ttc = paired_delta(near_pair, "ttc_s_min")
    ok &= check("near paired min-clearance delta", d_clear, ">=", 0.10, required=args.publication)
    ok &= check("near paired min-TTC delta", d_ttc, ">=", 0.20, required=args.publication)
    if args.publication and ci_clear:
        ok &= check("near clearance CI lower bound", ci_clear[0], ">=", 0.0)
    if args.publication and ci_ttc:
        ok &= check("near TTC CI lower bound", ci_ttc[0], ">=", 0.0)

    print("===== contact: secondary-harm reduction with bounded episodes =====")
    ok &= check("contact paper-PCD miss", contact.get("closed_loop_audit_paper_pcd_selector_miss_rate"), "<=", miss_max)
    ok &= check("contact PCD", contact.get("closed_loop_post_contact_deployability"), ">=", 0.52)
    ok &= check("contact FRA", contact.get("closed_loop_FRA_exec"), "<=", 0.16)
    ok &= check("contact DRS", contact.get("closed_loop_DRS"), ">=", 0.84)
    ok &= check("contact bounded NUP", contact.get("closed_loop_bounded_NUP"), ">=", 0.985)
    ok &= check("contact intervention", contact.get("intervention_rate"), "<=", 0.040)
    ok &= check("contact intervention episode rate", contact.get("intervention_episode_rate"), "<=", 0.025)
    ok &= check("contact max intervention run", contact.get("max_intervention_run_length"), "<=", 2.0)
    d_secondary, ci_secondary = paired_delta(contact_pair, "secondary_overlap_event")
    d_stop, ci_stop = paired_delta(contact_pair, "new_stable_stop_event")
    ok &= check("contact secondary-overlap delta", d_secondary, "<=", -0.02, required=args.publication)
    ok &= check("contact new-stable-stop delta", d_stop, ">=", 0.02, required=args.publication)
    if args.publication and ci_secondary:
        ok &= check("contact overlap CI upper bound", ci_secondary[1], "<=", 0.0)
    if args.publication and ci_stop:
        ok &= check("contact stable-stop CI lower bound", ci_stop[0], ">=", 0.0)

    print("===== offline sanity only (not publication evidence) =====")
    o = (offline.get("methods") or {}).get("ocrap", {})
    ok &= check("offline contact PCD", o.get("post_contact_deployability"), ">=", 0.56)
    ok &= check("offline contact FRA", o.get("FRA_exec"), "<=", 0.10)
    ok &= check("offline contact DRS", o.get("DRS"), ">=", 0.90)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
