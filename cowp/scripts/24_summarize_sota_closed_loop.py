from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METHODS = ("gameformer", "dtpp", "pluto", "plant2", "pdm_closed")


def _load(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _dig(d: dict[str, Any], *keys: str, default=None):
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _method_offline(payload: dict[str, Any], method: str) -> dict[str, Any]:
    if not payload:
        return {}
    if method in payload and isinstance(payload[method], dict):
        return payload[method]
    # Allow a direct metrics JSON as input.
    return payload if payload.get("mode") == "learned_offline" else {}


def _row(method: str, waymax: dict[str, Any], offline: dict[str, Any], *, execution: str) -> dict[str, Any]:
    std = waymax.get("standard_metric_summary", {}) if isinstance(waymax, dict) else {}
    off = _method_offline(offline, method)
    rt = waymax.get("external_policy_runtime_summary", {}) if isinstance(waymax, dict) else {}
    return {
        "Method": method,
        "Execution": execution,
        "ClosedLoopScenes": waymax.get("num_rollouts"),
        "CUR": std.get("CR"),
        "CollisionRate": std.get("CollisionRate"),
        "OffroadRate": std.get("OffroadRate"),
        "WrongWayRate": std.get("WrongWayRate"),
        "OffRouteRate": std.get("OffRouteRate"),
        "EgoProgress": std.get("EP", std.get("WaymaxFinal/ProgressionMetric")),
        "KinematicsInfeasibilityRate": std.get("KinematicsInfeasibilityRate"),
        "LogDivergence": std.get("LogDivergence", std.get("WaymaxMean/LogDivergenceMetric")),
        # Mechanism quantities below are *cached-label candidate audits*, not
        # closed-loop counterfactual ground truth.  Keeping the protocol column
        # adjacent prevents these values from being silently mixed with Waymax.
        "PBTR_offline_audit": off.get("PriorityBurdenTransferRate"),
        "FSR_offline_audit": off.get("SelectedFalseSafeRate", off.get("FSR")),
        "OPR_offline_audit": off.get("OPR"),
        "BTE_CVaR25_offline_audit": off.get("PriorityBurden/BTE_CVaR_25"),
        "NCF_Ret_offline_audit": off.get("PriorityCertificate/NCFSceneRetention", off.get("CertificateCoverage/NCFSceneRetention")),
        "NPR_offline_audit": off.get("PriorityCertificate/NonCoerciveProgressRegret"),
        "MechanismAuditProtocol": "cached_label_candidate_projection" if off else None,
        "ClosedLoopMechanismGT": 0 if waymax else None,
        "FallbackStepRate": rt.get("ClosedLoopFallbackStepRate", waymax.get("ClosedLoopFallbackStepRate")),
        "DirectExecutionStepRate": rt.get("ClosedLoopDirectExecutionStepRate"),
    }


def _cowp_row(path: Path) -> dict[str, Any]:
    p = _load(path)
    std = p.get("standard_metric_summary", {})
    # Merged reference JSONs keep fallback at top level; direct eval JSONs may
    # keep online proxies in policy diagnostics.
    diag = p.get("policy_diagnostic_summary", {})
    return {
        "Method": "COWP",
        "Execution": p.get("method", "cowp"),
        "ClosedLoopScenes": p.get("num_rollouts"),
        "CUR": std.get("CR"),
        "CollisionRate": std.get("CollisionRate"),
        "OffroadRate": std.get("OffroadRate"),
        "WrongWayRate": std.get("WrongWayRate"),
        "OffRouteRate": std.get("OffRouteRate"),
        "EgoProgress": std.get("EP", std.get("WaymaxFinal/ProgressionMetric")),
        "KinematicsInfeasibilityRate": std.get("KinematicsInfeasibilityRate"),
        "LogDivergence": std.get("LogDivergence", std.get("WaymaxMean/LogDivergenceMetric")),
        "PBTR_offline_audit": None,
        "FSR_offline_audit": None,
        "OPR_offline_audit": None,
        "BTE_CVaR25_offline_audit": None,
        "NCF_Ret_offline_audit": None,
        "NPR_offline_audit": None,
        "MechanismAuditProtocol": "COWP online model proxy only unless paired reactive/human audit is supplied",
        "ClosedLoopMechanismGT": 0,
        "FallbackStepRate": p.get("ClosedLoopFallbackStepRate", diag.get("ClosedLoopFallbackStepRate")),
        "DirectExecutionStepRate": None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize five source-faithful WOMD baseline adapters plus COWP without mixing metric protocols.")
    ap.add_argument("--results-root", required=True, help="Root containing <method>/waymax.json and <method>/offline.json")
    ap.add_argument("--cowp-json", default=None, help="Optional COWP Waymax JSON on the identical scenario-ID manifest")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-csv", required=True)
    args = ap.parse_args()

    root = Path(args.results_root)
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        waymax = _load(root / method / "waymax.json")
        offline = _load(root / method / "offline.json")
        execution = "direct" if method in {"gameformer", "pluto", "plant2"} else "candidate_tree"
        if method == "pdm_closed":
            execution = "predictive_rule_proposals"
        rows.append(_row(method, waymax, offline, execution=execution))
    if args.cowp_json:
        rows.append(_cowp_row(Path(args.cowp_json)))

    payload = {
        "schema_version": "cowp_external_sota_summary_v1",
        "important_protocol_note": (
            "Waymax standard metrics are true closed-loop metrics. PBTR/FSR/OPR/BTE/NCF-Ret/NPR for external baselines "
            "come from cached-label nearest/candidate projection and are mechanism audits, not closed-loop counterfactual ground truth. "
            "The paper's causal burden claim still requires the separate reactive-agent and human-audited protocols."
        ),
        "rows": rows,
    }
    jout = Path(args.output_json)
    cout = Path(args.output_csv)
    jout.parent.mkdir(parents=True, exist_ok=True)
    cout.parent.mkdir(parents=True, exist_ok=True)
    jout.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fields = list(rows[0].keys()) if rows else []
    with cout.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
