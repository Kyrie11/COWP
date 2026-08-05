from __future__ import annotations

import argparse
import json
from pathlib import Path


def _metric(row: dict, name: str, default: float) -> float:
    try:
        value = row.get(name, default)
        return float(default if value is None else value)
    except Exception:
        return float(default)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Select a priority-aware BCOT risk budget on the calibration split. "
            "The protected-priority certificate is primary; global NCF metrics "
            "remain secondary anti-degeneration constraints."
        )
    )
    ap.add_argument("--input", required=True, help="learned_offline JSON containing bcot_risk_budget_sweep")
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-priority-ncf-recall", type=float, default=0.25)
    ap.add_argument("--min-priority-ncf-precision", type=float, default=0.35)
    ap.add_argument("--max-priority-burden-transfer", type=float, default=0.50)
    ap.add_argument("--min-global-ncf-recall", type=float, default=0.18)
    ap.add_argument("--min-accepted-rate", type=float, default=0.08)
    ap.add_argument("--max-fallback", type=float, default=0.30)
    ap.add_argument("--max-selected-global-false-safe", type=float, default=0.55)
    ap.add_argument("--min-ep", type=float, default=0.0)
    args = ap.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = payload.get("bcot_risk_budget_sweep", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("No bcot_risk_budget_sweep was found")
    bad_semantics = [
        idx for idx, row in enumerate(rows)
        if str(row.get("CertificateSemantics/Version", "")) != "v16_8_2_decoupled"
        or not bool(row.get("FallbackSemantics/ExplicitAccounting", False))
    ]
    if bad_semantics:
        raise ValueError(
            "Calibration input uses stale certificate/fallback semantics at rows "
            f"{bad_semantics[:8]}; rerun learned-offline evaluation with v16.8.2."
        )

    constraints = {
        "min_priority_ncf_recall": float(args.min_priority_ncf_recall),
        "min_priority_ncf_precision": float(args.min_priority_ncf_precision),
        "max_priority_burden_transfer": float(args.max_priority_burden_transfer),
        "min_global_ncf_recall": float(args.min_global_ncf_recall),
        "min_accepted_rate": float(args.min_accepted_rate),
        "max_fallback": float(args.max_fallback),
        "max_selected_global_false_safe": float(args.max_selected_global_false_safe),
        "min_ep": float(args.min_ep),
    }

    def feasible_row(row: dict) -> bool:
        return (
            _metric(row, "PriorityCertificate/AcceptNCFRecall", 0.0) >= args.min_priority_ncf_recall
            and _metric(row, "PriorityCertificate/AcceptNCFPrecision", 0.0) >= args.min_priority_ncf_precision
            and _metric(row, "PriorityBurdenTransferRate", 1.0) <= args.max_priority_burden_transfer
            and _metric(row, "LearnedAcceptNCFRecall", 0.0) >= args.min_global_ncf_recall
            and _metric(row, "LearnedAcceptedCandidateRate", 0.0) >= args.min_accepted_rate
            and _metric(row, "FallbackRate", 1.0) <= args.max_fallback
            and _metric(row, "SelectedFalseSafeRate", 1.0) <= args.max_selected_global_false_safe
            and _metric(row, "EP", 0.0) >= args.min_ep
        )

    feasible = [row for row in rows if feasible_row(row)]
    proposal_fs_floor = min(
        (_metric(row, "ProposalCoverage/BestCaseSelectedFalseSafeLowerBound", float("inf")) for row in rows),
        default=float("inf"),
    )
    proposal_pbtr_floor = min(
        (_metric(row, "ProposalCoverage/BestCasePBTRLowerBound", float("inf")) for row in rows),
        default=float("inf"),
    )
    proposal_diagnostics_available = bool(
        any(str(row.get("ProposalDiagnostics/Version", "")) == "v16_8_3_proposal_floor" for row in rows)
    )
    proposal_infeasible_reasons: list[str] = []
    if proposal_diagnostics_available and proposal_fs_floor > args.max_selected_global_false_safe:
        proposal_infeasible_reasons.append(
            "best_case_selected_false_safe_lower_bound_exceeds_constraint"
        )
    if proposal_diagnostics_available and proposal_pbtr_floor > args.max_priority_burden_transfer:
        proposal_infeasible_reasons.append("best_case_pbtr_lower_bound_exceeds_constraint")
    proposal_feasible = not proposal_infeasible_reasons
    if feasible:
        selected = min(
            feasible,
            key=lambda row: (
                _metric(row, "PriorityBurdenTransferRate", 1.0),
                _metric(row, "SelectedFalseSafeRate", 1.0),
                _metric(row, "FallbackRate", 1.0),
                -_metric(row, "EP", 0.0),
            ),
        )
        status = "constraints_satisfied"
    else:
        def violation(row: dict) -> tuple[float, float, float, float, float]:
            p_recall_gap = max(0.0, args.min_priority_ncf_recall - _metric(row, "PriorityCertificate/AcceptNCFRecall", 0.0))
            p_precision_gap = max(0.0, args.min_priority_ncf_precision - _metric(row, "PriorityCertificate/AcceptNCFPrecision", 0.0))
            pbtr_gap = max(0.0, _metric(row, "PriorityBurdenTransferRate", 1.0) - args.max_priority_burden_transfer)
            g_recall_gap = max(0.0, args.min_global_ncf_recall - _metric(row, "LearnedAcceptNCFRecall", 0.0))
            accepted_gap = max(0.0, args.min_accepted_rate - _metric(row, "LearnedAcceptedCandidateRate", 0.0))
            fallback_gap = max(0.0, _metric(row, "FallbackRate", 1.0) - args.max_fallback)
            fs_gap = max(0.0, _metric(row, "SelectedFalseSafeRate", 1.0) - args.max_selected_global_false_safe)
            ep_gap = max(0.0, args.min_ep - _metric(row, "EP", 0.0))
            primary = (
                3.0 * p_recall_gap + 2.0 * p_precision_gap + 3.0 * pbtr_gap + 1.5 * g_recall_gap
                + 1.5 * accepted_gap + 1.5 * fallback_gap + fs_gap + ep_gap
            )
            return (
                primary,
                _metric(row, "PriorityBurdenTransferRate", 1.0),
                _metric(row, "SelectedFalseSafeRate", 1.0),
                _metric(row, "FallbackRate", 1.0),
                -_metric(row, "EP", 0.0),
            )

        selected = min(rows, key=violation)
        status = "proposal_infeasible" if not proposal_feasible else "least_violation"

    out = {
        "bcot_risk_budget": _metric(selected, "bcot_risk_budget", 0.35),
        "pair_witness_threshold": _metric(
            selected,
            "pair_witness_threshold",
            _metric(payload, "pair_witness_threshold", 0.70),
        ),
        "status": status,
        "selection_metrics": selected,
        "constraints": constraints,
        "num_operating_points": len(rows),
        "objective": "protected_priority_certificate_with_global_anti_degeneracy",
        "proposal_diagnostics_available": proposal_diagnostics_available,
        "proposal_feasible": proposal_feasible,
        "proposal_infeasible_reasons": proposal_infeasible_reasons,
        "proposal_best_case_selected_false_safe_lower_bound": (
            proposal_fs_floor if proposal_diagnostics_available else None
        ),
        "proposal_best_case_pbtr_lower_bound": (
            proposal_pbtr_floor if proposal_diagnostics_available else None
        ),
    }
    Path(args.output).write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
