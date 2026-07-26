from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SELECTION_KEYS = (
    "EP",
    "FallbackRate",
    "SelectedFalseSafeRate",
    "LearnedAcceptedCandidateRate",
    "LearnedAcceptNCFRecall",
)


def _f(d: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = d.get(key, default)
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return float(default)


def _subset_meta(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "modulo": int(metrics.get("EvaluationSubset/Modulo", 1) or 1),
        "remainder": int(metrics.get("EvaluationSubset/Remainder", 0) or 0),
        "scenes": int(metrics.get("EvaluationSubset/Scenes", metrics.get("num_scenes", 0)) or 0),
        "index_sha256": str(metrics.get("EvaluationSubset/IndexSHA256", "")),
    }


def _disjoint_partition(a: dict[str, Any], b: dict[str, Any]) -> bool:
    # For the deterministic i % modulo partition used by learned_offline, equal
    # modulus and distinct remainders prove disjointness.  A differing hash alone
    # is not treated as proof of disjointness.
    return (
        int(a.get("modulo", 1)) > 1
        and int(a.get("modulo", 1)) == int(b.get("modulo", 1))
        and int(a.get("remainder", 0)) != int(b.get("remainder", 0))
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Verify held-out COWP mechanism/selector quality. Calibration may "
            "choose the operating point but must never supply the reported metrics."
        )
    )
    ap.add_argument("--input", required=True, help="Held-out shared-method evaluation JSON.")
    ap.add_argument("--sweep-input", required=True, help="Calibration-only BCOT budget sweep JSON.")
    ap.add_argument("--method", default="cowp")
    ap.add_argument("--output", required=True)
    ap.add_argument("--calibration-json", required=True)
    ap.add_argument("--min-unique-selection-points", type=int, default=2)
    ap.add_argument("--min-ncf-recall", type=float, default=0.20)
    ap.add_argument("--min-witness-auprc", type=float, default=0.50)
    ap.add_argument("--min-bcot-auprc", type=float, default=0.0)
    ap.add_argument("--min-root-transport-auprc", type=float, default=0.0)
    ap.add_argument("--min-accepted-rate", type=float, default=0.08)
    ap.add_argument("--max-fallback", type=float, default=0.30)
    ap.add_argument("--min-false-safe-improvement", type=float, default=0.03)
    args = ap.parse_args()

    heldout = json.loads(Path(args.input).read_text(encoding="utf-8"))
    sweep_payload = json.loads(Path(args.sweep_input).read_text(encoding="utf-8"))
    calibration = json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))

    main = heldout.get(args.method, {})
    conventional = heldout.get("conventional_safety", {})
    if not isinstance(main, dict) or not main:
        raise ValueError(f"Held-out method {args.method!r} is missing from {args.input}")
    if not isinstance(conventional, dict) or not conventional:
        raise ValueError(f"Held-out conventional_safety is missing from {args.input}")

    rows = sweep_payload.get("bcot_risk_budget_sweep", [])
    sweep_kind = "bcot_risk_budget"
    if not rows:
        sweep_kind = "witness_threshold"
        rows = sweep_payload.get("witness_threshold_sweep", {}).get(args.method, [])
    rows = rows if isinstance(rows, list) else []
    signatures = [
        tuple(round(_f(row.get("metrics", row), key, float("nan")), 8) for key in SELECTION_KEYS)
        for row in rows
    ]
    unique = len(set(signatures)) if signatures else 0

    calibration_status = str(calibration.get("status", "unknown"))
    selected_metrics = calibration.get("selection_metrics", {})
    calibration_subset = _subset_meta(selected_metrics if isinstance(selected_metrics, dict) else {})
    heldout_subset = _subset_meta(main)
    subset_disjoint = _disjoint_partition(calibration_subset, heldout_subset)
    selected_budget = _f(calibration, sweep_kind, _f(calibration, "bcot_risk_budget", float("nan")))
    heldout_budget = _f(main, sweep_kind, _f(main, "bcot_risk_budget", float("nan")))
    operating_point_matches = abs(selected_budget - heldout_budget) <= 1e-9

    recall = _f(main, "LearnedAcceptNCFRecall")
    auprc = _f(main, "WitnessQuality/AUPRC")
    bcot_auprc = _f(main, "BCOT/FalseSafe_AUPRC")
    root_transport_auprc = _f(main, "RootTransport/ConflictConditioned_AUPRC")
    accepted = _f(main, "LearnedAcceptedCandidateRate")
    fallback = _f(main, "FallbackRate", 1.0)
    false_safe = _f(main, "SelectedFalseSafeRate", 1.0)
    conventional_false_safe = _f(conventional, "SelectedFalseSafeRate", 1.0)
    false_safe_gain = conventional_false_safe - false_safe

    report = {
        "evaluation_protocol": "calibration_partition_then_disjoint_heldout_partition",
        "operating_point_kind": sweep_kind,
        "calibrated_operating_point": selected_budget,
        "heldout_operating_point": heldout_budget,
        "operating_point_matches": operating_point_matches,
        "calibration_status": calibration_status,
        "calibration_feasible": calibration_status == "constraints_satisfied",
        "calibration_subset": calibration_subset,
        "heldout_subset": heldout_subset,
        "calibration_heldout_disjoint": subset_disjoint,
        "threshold_points": len(rows),
        "unique_selection_points": unique,
        "threshold_connected_to_selection": unique >= args.min_unique_selection_points,
        "learned_accept_ncf_recall": recall,
        "ncf_recall_pass": recall >= args.min_ncf_recall,
        "witness_auprc": auprc,
        "witness_auprc_pass": auprc >= args.min_witness_auprc,
        "bcot_false_safe_auprc": bcot_auprc,
        "bcot_auprc_pass": bcot_auprc >= args.min_bcot_auprc,
        "root_transport_conflict_auprc": root_transport_auprc,
        "root_transport_auprc_pass": root_transport_auprc >= args.min_root_transport_auprc,
        "learned_accepted_candidate_rate": accepted,
        "accepted_rate_pass": accepted >= args.min_accepted_rate,
        "fallback_rate": fallback,
        "fallback_pass": fallback <= args.max_fallback,
        "selected_false_safe_rate": false_safe,
        "conventional_selected_false_safe_rate": conventional_false_safe,
        "selected_false_safe_improvement": false_safe_gain,
        "false_safe_improvement_pass": false_safe_gain >= args.min_false_safe_improvement,
        "metrics_source": "heldout_input_only",
    }
    report["pass"] = bool(
        report["calibration_feasible"]
        and report["calibration_heldout_disjoint"]
        and report["operating_point_matches"]
        and report["threshold_connected_to_selection"]
        and report["ncf_recall_pass"]
        and report["witness_auprc_pass"]
        and report["bcot_auprc_pass"]
        and report["root_transport_auprc_pass"]
        and report["accepted_rate_pass"]
        and report["fallback_pass"]
        and report["false_safe_improvement_pass"]
    )
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
