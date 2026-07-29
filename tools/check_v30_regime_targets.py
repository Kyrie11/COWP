#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def fmt(x):
    return "None" if x is None else f"{float(x):.6f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Check OC-RAP v30 regime targets from closed-loop/audit JSON outputs.")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--safe-nup-min", type=float, default=1.0)
    ap.add_argument("--safe-int-max", type=float, default=0.0)
    ap.add_argument("--near-miss-max", type=float, default=0.05)
    ap.add_argument("--near-nup-min", type=float, default=0.995)
    ap.add_argument("--contact-miss-max", type=float, default=0.0833)
    ap.add_argument("--contact-pcd-min", type=float, default=0.472)
    ap.add_argument("--contact-fra-max", type=float, default=0.25)
    ap.add_argument("--contact-drs-min", type=float, default=0.75)
    args = ap.parse_args()
    root = args.run_dir
    files = {
        "safe": root / "closed_loop_safe_fast_v30.json",
        "near": root / "audit_near_contact_selected_topk_v30_v30.json",
        "contact": root / "audit_contact_selected_topk_v30_v30.json",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        print("missing files:", *missing, sep="\n  ")
        return 2
    safe, near, contact = (load(files[k]) for k in ("safe", "near", "contact"))
    rows = [
        ("safe_intervention", safe.get("intervention_rate"), "<=", args.safe_int_max),
        ("safe_NUP", safe.get("closed_loop_bounded_NUP"), ">=", args.safe_nup_min),
        ("near_NUP", near.get("closed_loop_bounded_NUP"), ">=", args.near_nup_min),
        ("near_paper_PCD_miss", near.get("closed_loop_audit_paper_pcd_selector_miss_rate"), "<=", args.near_miss_max),
        ("contact_paper_PCD_miss", contact.get("closed_loop_audit_paper_pcd_selector_miss_rate"), "<=", args.contact_miss_max),
        ("contact_PCD", contact.get("closed_loop_post_contact_deployability"), ">=", args.contact_pcd_min),
        ("contact_FRA", contact.get("closed_loop_FRA_exec"), "<=", args.contact_fra_max),
        ("contact_DRS", contact.get("closed_loop_DRS"), ">=", args.contact_drs_min),
    ]
    ok_all = True
    for name, val, op, thr in rows:
        ok = False if val is None else ((float(val) <= thr) if op == "<=" else (float(val) >= thr))
        ok_all &= ok
        print(f"{name:24s} {fmt(val):>10s} {op} {thr:.6f}  {'PASS' if ok else 'FAIL'}")
    print("contact macro_counts:", contact.get("macro_counts"))
    print("contact paper miss best macros:", contact.get("audit_paper_pcd_miss_best_macro_counts"))
    print("near macro_counts:", near.get("macro_counts"))
    print("near paper miss best macros:", near.get("audit_paper_pcd_miss_best_macro_counts"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
