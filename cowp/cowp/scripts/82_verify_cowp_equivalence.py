from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _rows(obj: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if "scenario_results" in obj:
        return {str(r["scenario_id"]): r for r in obj["scenario_results"]}
    ids = [str(x) for x in obj.get("scenario_ids_resolved", [])]
    metrics = list(obj.get("standard_metrics", []))
    diags = list(obj.get("scenario_diagnostics", []))
    return {
        sid: {
            "scenario_id": sid,
            "standard_metrics": metrics[i] if i < len(metrics) else {},
            "diagnostics": diags[i] if i < len(diags) else {},
        }
        for i, sid in enumerate(ids)
    }


def _same(a: Any, b: Any, atol: float) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        fa, fb = float(a), float(b)
        if np.isnan(fa) and np.isnan(fb):
            return True
        return bool(np.isclose(fa, fb, rtol=0.0, atol=atol, equal_nan=True))
    return a == b


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify that v16.8.29 COWP base path remains v16.8.28-equivalent on the dev manifest.")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--atol", type=float, default=1e-7)
    args = ap.parse_args()
    ref = _rows(json.load(open(args.reference, encoding="utf-8")))
    cand = _rows(json.load(open(args.candidate, encoding="utf-8")))
    if set(ref) != set(cand):
        raise SystemExit(f"scenario set mismatch: ref={len(ref)} candidate={len(cand)} common={len(set(ref)&set(cand))}")

    # New v16.8.29 fields are intentionally ignored here. Every field that already
    # existed in the v16.8.28 reference must stay unchanged on the unmodified COWP
    # method. This catches accidental semantic changes in the speed refactor.
    mismatches: list[dict[str, Any]] = []
    checked = 0
    for sid in sorted(ref):
        rr, cc = ref[sid], cand[sid]
        for section in ("standard_metrics", "diagnostics"):
            rsec = rr.get(section) or {}
            csec = cc.get(section) or {}
            for key, rv in rsec.items():
                if key not in csec:
                    mismatches.append({"scenario_id": sid, "section": section, "key": key, "reference": rv, "candidate": "<missing>"})
                    continue
                checked += 1
                cv = csec[key]
                if not _same(rv, cv, args.atol):
                    mismatches.append({"scenario_id": sid, "section": section, "key": key, "reference": rv, "candidate": cv})
                    if len(mismatches) >= 50:
                        break
            if len(mismatches) >= 50:
                break
        if len(mismatches) >= 50:
            break
    out = {
        "schema_version": "cowp_v16_8_29_base_equivalence_v1",
        "scenarios": len(ref),
        "fields_checked": checked,
        "atol": args.atol,
        "passed": not mismatches,
        "mismatch_count_capped": len(mismatches),
        "mismatches": mismatches,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if mismatches:
        raise SystemExit("v16.8.29 COWP base equivalence FAILED")


if __name__ == "__main__":
    main()
