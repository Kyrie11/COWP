from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _f(d: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = d.get(key, default)
        return float(default if v is None else v)
    except (TypeError, ValueError):
        return float(default)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize BCOT mechanism and selector readiness.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--sweep-input", default="", help="Optional JSON containing bcot_risk_budget_sweep; --input can remain the shared-method comparison.")
    ap.add_argument("--calibration-json", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--method", default="cowp")
    args = ap.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    calibration = json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))
    selected = calibration.get("selection_metrics", {})
    conventional = payload.get("conventional_safety", {})
    sweep_payload = (
        json.loads(Path(args.sweep_input).read_text(encoding="utf-8"))
        if args.sweep_input else payload
    )
    sweep = sweep_payload.get("bcot_risk_budget_sweep", [])
    operating_key = "bcot_risk_budget"
    if not sweep:
        sweep = sweep_payload.get("witness_threshold_sweep", {})
        operating_key = "witness_threshold"
    if isinstance(sweep, dict):
        sweep = sweep.get(args.method, [])
    sweep = sweep if isinstance(sweep, list) else []

    keys = ["EP", "FallbackRate", "OPR", "HBCR", "SelectedFalseSafeRate",
            "LearnedAcceptedCandidateRate", "LearnedAcceptNCFRecall"]
    deltas = {k: _f(selected, k) - _f(conventional, k) for k in keys}
    feasible = [
        r for r in sweep
        if _f(r, "LearnedAcceptNCFRecall") >= 0.30
        and _f(r, "FallbackRate", 1.0) <= 0.25
        and _f(conventional, "SelectedFalseSafeRate", 1.0) - _f(r, "SelectedFalseSafeRate", 1.0) >= 0.08
    ]
    best_recall = max(sweep, key=lambda r: _f(r, "LearnedAcceptNCFRecall"), default={})
    best_fs = min(sweep, key=lambda r: _f(r, "SelectedFalseSafeRate", 1.0), default={})
    report = {
        "calibrated_operating_point": _f(
            calibration,
            operating_key,
            _f(calibration, "bcot_risk_budget", _f(calibration, "witness_threshold")),
        ),
        "operating_point_kind": operating_key,
        "calibration_status": calibration.get("status", "unknown"),
        "conventional": {k: _f(conventional, k) for k in keys},
        "bcot_calibrated": {k: _f(selected, k) for k in keys},
        "delta_bcot_minus_conventional": deltas,
        "mechanism": {
            "pair_witness_auprc": _f(selected, "WitnessQuality/AUPRC"),
            "bcot_false_safe_auprc": _f(selected, "BCOT/FalseSafe_AUPRC"),
            "bcot_ranking_pair_accuracy": _f(selected, "BCOT/RiskRankingPairAccuracy"),
            "candidate_certificate_false_safe_auprc": _f(selected, "CandidateCertificate/FalseSafe_AUPRC"),
        },
        "sweep": {
            "points": len(sweep),
            "feasible_points": len(feasible),
            "best_ncf_recall": {k: best_recall.get(k) for k in (
                operating_key, "LearnedAcceptNCFRecall", "FallbackRate", "SelectedFalseSafeRate", "EP")},
            "lowest_false_safe": {k: best_fs.get(k) for k in (
                operating_key, "LearnedAcceptNCFRecall", "FallbackRate", "SelectedFalseSafeRate", "EP")},
        },
        "development_gate": {
            "pair_auprc": _f(selected, "WitnessQuality/AUPRC") >= 0.60,
            "bcot_auprc": _f(selected, "BCOT/FalseSafe_AUPRC") >= 0.65,
            "ncf_recall": _f(selected, "LearnedAcceptNCFRecall") >= 0.30,
            "accepted_rate": _f(selected, "LearnedAcceptedCandidateRate") >= 0.10,
            "fallback": _f(selected, "FallbackRate", 1.0) <= 0.25,
            "false_safe_gain": _f(conventional, "SelectedFalseSafeRate", 1.0) - _f(selected, "SelectedFalseSafeRate", 1.0) >= 0.08,
        },
    }
    report["development_gate"]["pass"] = all(report["development_gate"].values())
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
