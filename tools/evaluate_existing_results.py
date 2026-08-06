#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize an existing COWP v16.8.x result directory.")
    ap.add_argument("--result-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    root = args.result_dir

    mech = load(root / "eval/learned_offline/mechanism_verification.json")
    causal = load(root / "eval/causal_protocol_audit.json")
    cache = load(root / "eval/cache_sufficiency_current.json")
    natural = load(root / "eval/learned_offline/natural_basis_gate.json")
    effectiveness = load(root / "eval/learned_offline/natural_effectiveness_gate.json")

    report = {
        "natural_basis_gate": natural["pass"],
        "natural_effectiveness_gate": effectiveness["pass"],
        "causal_engineering_audit": causal["engineering_pass"],
        "fresh_v16_8_3_label_protocol": causal["fresh_v16_8_3_label_protocol_pass"],
        "calibration_feasible": mech["calibration_feasible"],
        "proposal_feasible": mech["proposal_feasible"],
        "mechanism_gate": mech["pass"],
        "metrics": {
            "PairWitnessAUPRC": mech["witness_auprc"],
            "ProtectedBCOTAUPRC": mech["priority_bcot_false_safe_auprc"],
            "ProtectedRootTransportAUPRC": mech["priority_root_transport_auprc"],
            "PriorityNCFRecall": mech["priority_accept_ncf_recall"],
            "PriorityNCFPrecision": mech["priority_accept_ncf_precision"],
            "AcceptedCandidateRate": mech["learned_accepted_candidate_rate"],
            "FallbackRate": mech["fallback_rate"],
            "PBTRImprovement": mech["priority_transfer_improvement"],
            "FalseSafeImprovement": mech["global_false_safe_improvement"],
            "CalibrationProposalFSRFloor": mech[
                "proposal_best_case_selected_false_safe_lower_bound"
            ],
            "CalibrationProposalPBTRFloor": mech[
                "proposal_best_case_pbtr_lower_bound"
            ],
        },
        "cache_decision": cache["decision"]["overall"],
        "root_cause": (
            "The learned mechanism ranks existing proposals well, but the fixed proposal "
            "bank is mathematically infeasible for the configured FSR/PBTR gates. The "
            "current cache is an old-base root-conditioned overlay, so v16.8.3 RMR-BCTE "
            "and repaired offline filtering were not materialized."
        ),
        "paper_claim_ready": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
