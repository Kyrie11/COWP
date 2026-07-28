from __future__ import annotations

import argparse
import json
from pathlib import Path


def _method(path: str, name: str) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d[name] if name in d else d


def main() -> None:
    ap = argparse.ArgumentParser(description="Fail-fast gate before expensive full COWP evaluation.")
    ap.add_argument("--cowp-offline", required=True)
    ap.add_argument("--conventional-offline", required=True)
    ap.add_argument("--cowp-online", default=None)
    ap.add_argument("--conventional-online", default=None)
    ap.add_argument("--probe", action="store_true", help="Use looser online tolerances for 50-100 scenario probes.")
    args = ap.parse_args()

    cowp = _method(args.cowp_offline, "cowp")
    conv = _method(args.conventional_offline, "conventional_safety")
    failures: list[str] = []
    checks = {
        "certificate_false_safe_auprc": cowp.get("CandidateCertificate/FalseSafe_AUPRC", 0.0),
        "certificate_ncf_auprc": cowp.get("CandidateCertificate/NCF_AUPRC", 0.0),
        "certificate_ranking": cowp.get("CandidateCertificate/RiskRankingPairAccuracy", 0.0),
        "witness_p50": cowp.get("WitnessProb/p50", 1.0),
        "cowp_selected_false_safe": cowp.get("SelectedFalseSafeRate", 1.0),
        "conv_selected_false_safe": conv.get("SelectedFalseSafeRate", 1.0),
        "cowp_opr": cowp.get("OPR", 0.0),
        "conv_opr": conv.get("OPR", 0.0),
        "cowp_ep": cowp.get("EP", 0.0),
    }
    if checks["certificate_false_safe_auprc"] < 0.52:
        failures.append("FalseSafe_AUPRC < 0.52")
    if checks["certificate_ncf_auprc"] < 0.35:
        failures.append("NCF_AUPRC < 0.35")
    if checks["certificate_ranking"] < 0.60:
        failures.append("certificate ranking < 0.60")
    if checks["witness_p50"] > 0.95:
        failures.append("witness probability remains saturated")
    if checks["cowp_selected_false_safe"] >= checks["conv_selected_false_safe"]:
        failures.append("COWP selected false-safe rate is not below conventional")
    if checks["cowp_opr"] <= checks["conv_opr"]:
        failures.append("COWP OPR is not above conventional")
    if checks["cowp_ep"] < 0.75:
        failures.append("COWP offline EP < 0.75")

    online = {}
    if args.cowp_online and args.conventional_online:
        co = json.loads(Path(args.cowp_online).read_text(encoding="utf-8"))["standard_metric_summary"]
        cv = json.loads(Path(args.conventional_online).read_text(encoding="utf-8"))["standard_metric_summary"]
        online = {"cowp": co, "conventional": cv}
        slack = 0.08 if args.probe else 0.03
        if co.get("CR", 1.0) > cv.get("CR", 1.0) + slack:
            failures.append("online CR regression")
        if co.get("CollisionRate", 1.0) > cv.get("CollisionRate", 1.0) + slack:
            failures.append("online collision regression")
        if co.get("OffroadRate", 1.0) > cv.get("OffroadRate", 1.0) + slack:
            failures.append("online offroad regression")
        if co.get("EP", 0.0) < 0.75:
            failures.append("online EP < 0.75")

    report = {"offline": checks, "online": online, "failures": failures, "pass": not failures}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
