from __future__ import annotations

import argparse
import json
from pathlib import Path


def _num(row: dict, key: str):
    v = row.get(key)
    return float(v) if isinstance(v, (int, float)) else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare COWP and v16.8.25 Certificate-Then-Utility on one shared learned-offline run.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tol", type=float, default=1e-8)
    args = ap.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    base = payload.get("cowp")
    ctu = payload.get("cowp_cert_utility")
    if not isinstance(base, dict) or not isinstance(ctu, dict):
        raise ValueError("Input must contain both 'cowp' and 'cowp_cert_utility' rows from one shared-model-pass evaluation")

    invariant_prefixes = (
        "ProposalCoverage/", "PriorityCertificate/Accept", "BCOT/", "RootTransport/", "OutcomeHead/"
    )
    invariant_exact = {
        "LearnedAcceptedCandidateRate",
        "CertificateCoverage/AnyAcceptedSceneRate",
        "CertificateCoverage/AnyAcceptedNCFSceneRate",
        "CertificateCoverage/EmptySceneRate",
        "CertificateCoverage/NCFSceneRetention",
    }
    invariant_diffs = {}
    for key in sorted(set(base) & set(ctu)):
        if key in invariant_exact or key.startswith(invariant_prefixes):
            a, b = _num(base, key), _num(ctu, key)
            if a is not None and b is not None:
                invariant_diffs[key] = b - a
    bad_invariants = {k: v for k, v in invariant_diffs.items() if abs(v) > float(args.tol)}

    selected_keys = [
        "EP", "CR", "FallbackRate", "PriorityBurdenTransferRate", "FSR",
        "SelectedNCFRate", "SelectedFalseSafeRate",
        "Selector/NCFSelectionRecallGivenAvailable",
        "Selector/FalseSafeExcessAboveProposalFloor",
        "PriorityCertificate/NonCoerciveProgressRegret",
        "SelectedWaymaxCollisionRate", "SelectedWaymaxOffroadRate",
        "SelectedWaymaxUnsafeRate", "SelectedWaymaxOutcomeCoverage",
    ]
    selected = {}
    for key in selected_keys:
        a, b = _num(base, key), _num(ctu, key)
        if a is not None or b is not None:
            selected[key] = {"cowp": a, "ctu": b, "delta_ctu_minus_cowp": None if a is None or b is None else b - a}

    pbtr_key = "PriorityBurdenTransferRate"
    fs_key = "SelectedFalseSafeRate"
    ep0, ep1 = _num(base, "EP"), _num(ctu, "EP")
    pb0, pb1 = _num(base, pbtr_key), _num(ctu, pbtr_key)
    fs0, fs1 = _num(base, fs_key), _num(ctu, fs_key)
    reasons = []
    if bad_invariants:
        verdict = "BUG_certificate_not_invariant"
        reasons.append("CTU changed quantities that should be selector-invariant; do not interpret performance deltas.")
    else:
        noninferior = True
        if pb0 is not None and pb1 is not None and pb1 > pb0 + 0.01:
            noninferior = False; reasons.append("PBTR worsened by >1 percentage point.")
        if fs0 is not None and fs1 is not None and fs1 > fs0 + 0.01:
            noninferior = False; reasons.append("Selected false-safe rate worsened by >1 percentage point.")
        if ep0 is not None and ep1 is not None and ep1 < 0.95 * max(ep0, 1e-9):
            noninferior = False; reasons.append("EP retained <95% of COWP.")
        if noninferior:
            verdict = "WORTH_EXACT_ID_WAYMAX_PROBE"
            reasons.append("Certificate is invariant and learned-offline selection is non-inferior under the preregistered engineering screen.")
        else:
            verdict = "REJECT_CTU_AS_DEFAULT"
            reasons.append("Keep original COWP selector; use the result only to localize the bottleneck.")

    out = {
        "version": "v16.8.25_ctu_probe_v1",
        "verdict": verdict,
        "certificate_invariant": not bad_invariants,
        "bad_invariants": bad_invariants,
        "selected_metric_comparison": selected,
        "engineering_screen_notes": reasons,
        "important_caveat": "Cached SelectedWaymax* metrics are partial candidate-outcome diagnostics, not strict online Waymax evidence. Promotion still requires paired exact-ID Waymax.",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if bad_invariants:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
