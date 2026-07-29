#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Any, Iterable


def load(path: Path) -> dict[str, Any] | None:
    try:
        return json.load(path.open())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def metric(d: dict[str, Any], key: str, default: float | None = None) -> float | None:
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default


def _scene_records(path: Path, summary: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for scene in summary.get("scenes", []) or []:
        if isinstance(scene, dict):
            yield scene
    jsonl = Path(str(path) + ".scenes.jsonl")
    if jsonl.exists():
        for line in jsonl.open():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            scene = row.get("scene", row)
            if isinstance(scene, dict):
                yield scene


def intervention_diagnostics(path: Path, summary: dict[str, Any], threshold: float):
    total = direct = 0
    bad = []
    for scene in _scene_records(path, summary):
        for decision in scene.get("decisions", []) or []:
            macro = str(decision.get("selected_macro", "nominal") or "nominal").lower()
            if macro in {"nominal", "keep", ""}:
                continue
            total += 1
            reason = str(decision.get("selection_reason", ""))
            direct += int("direct_value" in reason)
            dev = float(decision.get("selected_nominal_deviation") or 0.0)
            if dev < threshold:
                bad.append((scene.get("scene_id"), decision.get("step_index"), dev, reason, macro))
    return total, bad, direct


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--base-run", type=Path, required=True)
    ap.add_argument("--mode", choices=["execution", "mechanism", "confirmation"], default="mechanism")
    ap.add_argument("--min-deviation", type=float, default=0.002)
    ap.add_argument("--max-paired-degradation", type=float, default=0.010)
    ap.add_argument("--min-improvement", type=float, default=0.005)
    ap.add_argument("--near-intervention-cap", type=float, default=0.10)
    ap.add_argument("--contact-intervention-cap", type=float, default=0.15)
    args = ap.parse_args()
    failures: list[str] = []
    warnings: list[str] = []
    rows: list[tuple[str, Any]] = []

    for bucket in ("near", "contact"):
        c = load(args.base_run / "calibration" / f"direct_value_risk_{bucket}_v47.json")
        if not c:
            failures.append(f"{bucket}: missing v47 risk certificate")
            continue
        verify = c.get("verify", {}) or {}
        score_t = float(c.get("direct_value_threshold", float("inf")))
        opp_t = float(c.get("direct_value_opportunity_threshold", float("inf")))
        harm_t = float(c.get("direct_value_harm_threshold", float("inf")))
        rows += [
            (f"cal_{bucket}_valid", c.get("valid_for_active_contract")),
            (f"cal_{bucket}_score_threshold", score_t),
            (f"cal_{bucket}_opportunity_threshold", opp_t),
            (f"cal_{bucket}_harm_threshold", harm_t),
            (f"cal_{bucket}_verify_selected", verify.get("num_selected")),
            (f"cal_{bucket}_precision", verify.get("challenge_precision")),
            (f"cal_{bucket}_harm_group_ucb", verify.get("harmful_group_exposure_ucb90")),
            (f"cal_{bucket}_harm_selected_ucb", verify.get("harmful_selected_ucb90")),
        ]
        if not c.get("valid_for_active_contract", False):
            failures.append(f"{bucket}: certificate invalid")
        if not math.isfinite(score_t) or not math.isfinite(opp_t) or not math.isfinite(harm_t):
            failures.append(f"{bucket}: non-finite score/opportunity/harm threshold")

    for bucket in ("near_contact", "contact"):
        v_path = args.run / f"audit_{bucket}_selected_topk_v47_v47.json"
        v = load(v_path)
        if not v:
            failures.append(f"{bucket}: missing v47 audit")
            continue
        total, bad, direct = intervention_diagnostics(v_path, v, args.min_deviation)
        rows += [
            (f"{bucket}_decisions", v.get("num_decisions")),
            (f"{bucket}_pcd", v.get("closed_loop_post_contact_deployability")),
            (f"{bucket}_paper_regret", v.get("closed_loop_audit_paper_selected_PCD_regret")),
            (f"{bucket}_nup", v.get("closed_loop_bounded_NUP")),
            (f"{bucket}_intervention_rate", v.get("intervention_rate")),
            (f"{bucket}_interventions", total),
            (f"{bucket}_direct_interventions", direct),
            (f"{bucket}_active_regime_counts", v.get("active_regime_counts")),
            (f"{bucket}_mean_opportunity", v.get("closed_loop_direct_recovery_opportunity")),
        ]
        if bad:
            failures.append(f"{bucket}: {len(bad)} intervention(s) below actionability threshold")
        if direct <= 0:
            failures.append(f"{bucket}: OC-TRAC direct path never entered final decisions")
        if total > direct:
            warnings.append(f"{bucket}: {total-direct} non-direct intervention(s) occurred")
        nup_floor = 0.995 if bucket == "near_contact" else 0.985
        cap = args.near_intervention_cap if bucket == "near_contact" else args.contact_intervention_cap
        if (metric(v, "closed_loop_bounded_NUP", 0) or 0) < nup_floor:
            failures.append(f"{bucket}: NUP below {nup_floor}")
        if (metric(v, "intervention_rate", 1) or 1) > cap:
            failures.append(f"{bucket}: intervention rate above {cap}")

        if args.mode != "execution":
            s_path = args.run / f"audit_{bucket}_selected_topk_v47_scalar.json"
            scalar = load(s_path)
            if not scalar:
                failures.append(f"{bucket}: missing paired scalar audit")
                continue
            vp = metric(v, "closed_loop_post_contact_deployability", 0) or 0
            sp = metric(scalar, "closed_loop_post_contact_deployability", 0) or 0
            vr = metric(v, "closed_loop_audit_paper_selected_PCD_regret", 1) or 1
            sr = metric(scalar, "closed_loop_audit_paper_selected_PCD_regret", 1) or 1
            pcd_gain = vp - sp
            regret_gain = sr - vr
            rows += [(f"{bucket}_paired_pcd_gain", pcd_gain), (f"{bucket}_paired_regret_reduction", regret_gain)]
            if pcd_gain < -args.max_paired_degradation:
                failures.append(f"{bucket}: PCD degradation exceeds {args.max_paired_degradation}")
            if regret_gain < -args.max_paired_degradation:
                failures.append(f"{bucket}: regret degradation exceeds {args.max_paired_degradation}")
            required = args.min_improvement if args.mode == "confirmation" else 0.0
            if max(pcd_gain, regret_gain) < required:
                failures.append(f"{bucket}: no >= {required} paired PCD/regret improvement")

    print(f"V47 OC-TRAC {args.mode.upper()} GATE")
    for k, v in rows:
        print(f"{k}: {v}")
    for x in warnings:
        print("WARNING:", x)
    for x in failures:
        print("FAIL:", x)
    if failures:
        print("RESULT: FAIL — do not expand")
        return 2
    print("RESULT: PASS — expand only to the next staged gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
