from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Source-attributed promotion screen for the experimental v16.8.25 "
            "Multi-Conflict Feasibility Corridor (MCFC) proposal family."
        )
    )
    ap.add_argument("--paired-probe", required=True)
    ap.add_argument("--source-ablation", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-delta-any-ncf", type=float, default=0.03)
    ap.add_argument("--min-delta-priority-ncf", type=float, default=0.02)
    ap.add_argument("--max-delta-false-safe-floor", type=float, default=-0.03)
    ap.add_argument("--max-delta-pbtr-floor", type=float, default=-0.02)
    ap.add_argument("--min-corridor-ncf-candidates", type=int, default=5)
    ap.add_argument("--min-corridor-priority-ncf-candidates", type=int, default=3)
    ap.add_argument("--max-ncf-loss-rate", type=float, default=0.02)
    ap.add_argument("--strict", action="store_true", help="Exit nonzero when the promotion screen fails.")
    args = ap.parse_args()

    paired = _load(args.paired_probe)
    ablation = _load(args.source_ablation)
    new = paired.get("new", {})
    paired_stats = paired.get("paired", {})
    inc = ablation.get("multi_conflict_corridor_increment", {})

    checks = {
        # Reuse the general proposal-bank validity/coverage gates first.
        "paired_probe_general_gate": bool(paired.get("promote_to_full_rebuild", False)),
        # Require that the newly enabled source actually exists and supplies
        # certified NCF candidates, rather than letting unrelated bank changes
        # satisfy the global gate.
        "mcfc_emits_ncf_candidates": int(new.get("multi_conflict_corridor_global_ncf_candidate_count", 0))
        >= int(args.min_corridor_ncf_candidates),
        "mcfc_emits_priority_ncf_candidates": int(new.get("multi_conflict_corridor_priority_ncf_candidate_count", 0))
        >= int(args.min_corridor_priority_ncf_candidates),
        # Require source-attributed scene-level gains large enough to matter at
        # paper scale.  These are deliberately stronger than a >0 smoke gate.
        "mcfc_any_ncf_gain": float(inc.get("delta_any_ncf_scene_rate", 0.0)) >= float(args.min_delta_any_ncf),
        "mcfc_priority_ncf_gain": float(inc.get("delta_priority_ncf_scene_rate", 0.0))
        >= float(args.min_delta_priority_ncf),
        "mcfc_false_safe_floor_drop": float(inc.get("delta_false_safe_floor", 0.0))
        <= float(args.max_delta_false_safe_floor),
        "mcfc_pbtr_floor_drop": float(inc.get("delta_pbtr_floor", 0.0)) <= float(args.max_delta_pbtr_floor),
        "no_material_ncf_regression": float(paired_stats.get("ncf_loss_rate", 1.0)) <= float(args.max_ncf_loss_rate),
    }
    promoted = bool(all(checks.values()))
    result = {
        "schema_version": "cowp_v16_8_25_mcfc_source_attributed_screen_v1",
        "paired_probe": str(Path(args.paired_probe).resolve()),
        "source_ablation": str(Path(args.source_ablation).resolve()),
        "thresholds": {
            "min_delta_any_ncf": args.min_delta_any_ncf,
            "min_delta_priority_ncf": args.min_delta_priority_ncf,
            "max_delta_false_safe_floor": args.max_delta_false_safe_floor,
            "max_delta_pbtr_floor": args.max_delta_pbtr_floor,
            "min_corridor_ncf_candidates": args.min_corridor_ncf_candidates,
            "min_corridor_priority_ncf_candidates": args.min_corridor_priority_ncf_candidates,
            "max_ncf_loss_rate": args.max_ncf_loss_rate,
        },
        "observed": {
            "corridor_global_ncf_candidate_count": int(new.get("multi_conflict_corridor_global_ncf_candidate_count", 0)),
            "corridor_priority_ncf_candidate_count": int(new.get("multi_conflict_corridor_priority_ncf_candidate_count", 0)),
            "delta_any_ncf_scene_rate": float(inc.get("delta_any_ncf_scene_rate", 0.0)),
            "delta_priority_ncf_scene_rate": float(inc.get("delta_priority_ncf_scene_rate", 0.0)),
            "delta_false_safe_floor": float(inc.get("delta_false_safe_floor", 0.0)),
            "delta_pbtr_floor": float(inc.get("delta_pbtr_floor", 0.0)),
            "ncf_loss_rate": float(paired_stats.get("ncf_loss_rate", 1.0)),
        },
        "gate_checks": checks,
        "promote_mcfc_to_full_rebuild": promoted,
        "interpretation": (
            "Promotion means MCFC passes both the global paired proposal-bank gate and a source-attributed "
            "effect-size gate. It is permission to rebuild/retrain, not evidence of final closed-loop benefit."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.strict and not promoted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
