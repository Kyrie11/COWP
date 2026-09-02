from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cheap screen for v16.8.6 priority-commitment proposals before a 1200-scene/full rebuild."
    )
    ap.add_argument("--paired-probe", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-any-valid", type=float, default=0.97)
    ap.add_argument("--min-any-ncf", type=float, default=0.38)
    ap.add_argument("--max-false-safe-floor", type=float, default=0.58)
    ap.add_argument("--max-pbtr-floor", type=float, default=0.52)
    ap.add_argument("--min-pbtr-improvement", type=float, default=0.04)
    ap.add_argument("--min-hard-recovery", type=float, default=0.12)
    ap.add_argument("--max-ncf-loss", type=float, default=0.05)
    ap.add_argument("--min-phr-scene-rate", type=float, default=0.02)
    ap.add_argument("--min-phr-priority-ncf-scene-rate", type=float, default=0.005)
    args = ap.parse_args()

    p = json.loads(Path(args.paired_probe).read_text(encoding="utf-8"))
    old = p.get("old", {})
    new = p.get("new", {})
    paired = p.get("paired", {})
    pairing = p.get("pairing_completeness", {})

    old_pbtr = float(old.get("best_case_pbtr_lower_bound", 1.0))
    new_pbtr = float(new.get("best_case_pbtr_lower_bound", 1.0))
    pbtr_improvement = old_pbtr - new_pbtr
    checks = {
        "pairing_complete": bool(pairing.get("complete", False)) and int(pairing.get("build_error_count", 0)) == 0,
        "any_valid": float(new.get("any_valid_scene_rate", 0.0)) >= args.min_any_valid,
        "any_ncf": float(new.get("any_ncf_scene_rate", 0.0)) >= args.min_any_ncf,
        "false_safe_floor": float(new.get("best_case_selected_false_safe_lower_bound", 1.0)) <= args.max_false_safe_floor,
        "pbtr_floor": new_pbtr <= args.max_pbtr_floor,
        "pbtr_improvement": pbtr_improvement >= args.min_pbtr_improvement,
        "hard_recovery": float(paired.get("hard_scene_ncf_recovery_rate", 0.0)) >= args.min_hard_recovery,
        "ncf_not_destroyed": float(paired.get("ncf_loss_rate", 1.0)) <= args.max_ncf_loss,
        "priority_hold_release_generated": float(new.get("scene_with_priority_hold_release_rate", 0.0)) >= args.min_phr_scene_rate,
        "priority_hold_release_yields_priority_ncf": float(new.get("scene_with_priority_hold_release_priority_ncf_rate", 0.0)) >= args.min_phr_priority_ncf_scene_rate,
    }
    result = {
        "schema_version": "cowp_v16_8_6_priority_commitment_micro_screen_v1",
        "paired_probe": str(Path(args.paired_probe).resolve()),
        "screen_pass": bool(all(checks.values())),
        "checks": checks,
        "observed": {
            "old_pbtr_floor": old_pbtr,
            "new_pbtr_floor": new_pbtr,
            "paired_pbtr_floor_improvement": pbtr_improvement,
            "new_any_valid_scene_rate": float(new.get("any_valid_scene_rate", 0.0)),
            "new_any_ncf_scene_rate": float(new.get("any_ncf_scene_rate", 0.0)),
            "new_false_safe_floor": float(new.get("best_case_selected_false_safe_lower_bound", 1.0)),
            "hard_scene_ncf_recovery_rate": float(paired.get("hard_scene_ncf_recovery_rate", 0.0)),
            "ncf_loss_rate": float(paired.get("ncf_loss_rate", 1.0)),
            "phr_scene_rate": float(new.get("scene_with_priority_hold_release_rate", 0.0)),
            "phr_priority_ncf_scene_rate": float(new.get("scene_with_priority_hold_release_priority_ncf_rate", 0.0)),
        },
        "thresholds": {
            "min_any_valid": args.min_any_valid,
            "min_any_ncf": args.min_any_ncf,
            "max_false_safe_floor": args.max_false_safe_floor,
            "max_pbtr_floor": args.max_pbtr_floor,
            "min_pbtr_improvement": args.min_pbtr_improvement,
            "min_hard_recovery": args.min_hard_recovery,
            "max_ncf_loss": args.max_ncf_loss,
            "min_phr_scene_rate": args.min_phr_scene_rate,
            "min_phr_priority_ncf_scene_rate": args.min_phr_priority_ncf_scene_rate,
        },
        "next_action": (
            "Run the 400-hard + 800-random strict v16.8.6 proposal probe; do not full-rebuild yet."
            if all(checks.values())
            else "Do not full-rebuild. Inspect PHR generation/yield and proposal rejection diagnostics first."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["screen_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
