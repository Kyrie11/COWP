from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _f(d: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = d.get(key, default)
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return float(default)


def _subset(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "modulo": int(metrics.get("EvaluationSubset/Modulo", 1) or 1),
        "remainder": int(metrics.get("EvaluationSubset/Remainder", 0) or 0),
        "scenes": int(metrics.get("EvaluationSubset/Scenes", metrics.get("num_scenes", 0)) or 0),
        "index_sha256": str(metrics.get("EvaluationSubset/IndexSHA256", "")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize held-out BCOT mechanism and selector readiness.")
    ap.add_argument("--input", required=True, help="Held-out shared-method JSON.")
    ap.add_argument("--sweep-input", required=True, help="Calibration-only budget sweep JSON.")
    ap.add_argument("--calibration-json", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--method", default="cowp")
    args = ap.parse_args()

    heldout = json.loads(Path(args.input).read_text(encoding="utf-8"))
    sweep_payload = json.loads(Path(args.sweep_input).read_text(encoding="utf-8"))
    calibration = json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))
    selected = heldout.get(args.method, {})
    conventional = heldout.get("conventional_safety", {})
    if not isinstance(selected, dict) or not selected:
        raise ValueError(f"Held-out method {args.method!r} is missing")
    if not isinstance(conventional, dict) or not conventional:
        raise ValueError("Held-out conventional_safety is missing")

    sweep = sweep_payload.get("bcot_risk_budget_sweep", [])
    operating_key = "bcot_risk_budget"
    if not sweep:
        sweep = sweep_payload.get("witness_threshold_sweep", {})
        operating_key = "witness_threshold"
    if isinstance(sweep, dict):
        sweep = sweep.get(args.method, [])
    sweep = sweep if isinstance(sweep, list) else []

    keys = [
        "EP", "FallbackRate", "OPR", "HBCR", "SelectedFalseSafeRate",
        "LearnedAcceptedCandidateRate", "LearnedAcceptNCFRecall",
    ]
    deltas = {key: _f(selected, key) - _f(conventional, key) for key in keys}
    feasible = [
        row for row in sweep
        if _f(row, "LearnedAcceptNCFRecall") >= 0.30
        and _f(row, "FallbackRate", 1.0) <= 0.25
        and _f(conventional, "SelectedFalseSafeRate", 1.0) - _f(row, "SelectedFalseSafeRate", 1.0) >= 0.08
    ]
    best_recall = max(sweep, key=lambda row: _f(row, "LearnedAcceptNCFRecall"), default={})
    best_fs = min(sweep, key=lambda row: _f(row, "SelectedFalseSafeRate", 1.0), default={})
    calibration_metrics = calibration.get("selection_metrics", {})
    report = {
        "evaluation_protocol": "calibration_partition_then_disjoint_heldout_partition",
        "calibrated_operating_point": _f(calibration, operating_key, _f(calibration, "bcot_risk_budget", _f(calibration, "witness_threshold"))),
        "heldout_operating_point": _f(selected, operating_key, _f(selected, "bcot_risk_budget")),
        "operating_point_kind": operating_key,
        "calibration_status": calibration.get("status", "unknown"),
        "calibration_subset": _subset(calibration_metrics if isinstance(calibration_metrics, dict) else {}),
        "heldout_subset": _subset(selected),
        "conventional": {key: _f(conventional, key) for key in keys},
        "bcot_heldout": {key: _f(selected, key) for key in keys},
        "delta_bcot_minus_conventional": deltas,
        "mechanism_heldout": {
            "pair_witness_auprc": _f(selected, "WitnessQuality/AUPRC"),
            "bcot_false_safe_auprc": _f(selected, "BCOT/FalseSafe_AUPRC"),
            "root_transport_conflict_auprc": _f(selected, "RootTransport/ConflictConditioned_AUPRC"),
            "bcot_ranking_pair_accuracy": _f(selected, "BCOT/RiskRankingPairAccuracy"),
            "candidate_certificate_false_safe_auprc": _f(selected, "CandidateCertificate/FalseSafe_AUPRC"),
        },
        "calibration_sweep": {
            "points": len(sweep),
            "feasible_points": len(feasible),
            "best_ncf_recall": {key: best_recall.get(key) for key in (operating_key, "LearnedAcceptNCFRecall", "FallbackRate", "SelectedFalseSafeRate", "EP")},
            "lowest_false_safe": {key: best_fs.get(key) for key in (operating_key, "LearnedAcceptNCFRecall", "FallbackRate", "SelectedFalseSafeRate", "EP")},
        },
        "development_gate": {
            "pair_auprc": _f(selected, "WitnessQuality/AUPRC") >= 0.60,
            "bcot_auprc": _f(selected, "BCOT/FalseSafe_AUPRC") >= 0.65,
            "root_transport_auprc": _f(selected, "RootTransport/ConflictConditioned_AUPRC") >= 0.65,
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
