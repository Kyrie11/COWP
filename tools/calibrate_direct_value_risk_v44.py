#!/usr/bin/env python3
"""Fit/verify v45 OC-RAVE opportunity-gated selective recovery certificates.

The deployment rule is reproduced exactly:
  1. apply physical/macro actionability;
  2. retain candidates above a learned opportunity-probability threshold;
  3. choose the highest observation-conditioned score advantage;
  4. challenge nominal only when that advantage exceeds a fitted threshold.

Opportunity and score thresholds are fitted on deterministic fold 0 and
verified on disjoint fold 1.  This is a development risk contract, not a claim
of distribution-free conformal coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ocrap.algorithms.ocmero import oc_mero
from ocrap.data.serialization import load_npz
from ocrap.evaluation.metrics import best_shared_option_index, deployable_recovery_success, post_contact_deployability_score
from ocrap.models.data import iter_sample_paths_many, scalar_metadata_for_path
from ocrap.models.inference import load_model_bundle, predict_sample


def _scalar(d: dict[str, Any], key: str, default: Any) -> Any:
    a = np.asarray(d.get(key, default))
    return a.item() if a.shape == () else a


def _teacher_pcd(d: dict[str, Any], alpha: float, beta: float, top_m: int) -> float:
    m = np.asarray(d["m_star"], dtype=np.float64)
    p = np.asarray(d["root_probs"], dtype=np.float64)
    c = np.asarray(d.get("c_star", np.eye(m.shape[0])), dtype=np.float64)
    rv = np.asarray(d.get("root_valid", np.ones(m.shape[0])), dtype=bool)
    ov = np.asarray(d.get("option_valid", np.ones(m.shape[1])), dtype=bool)
    res = oc_mero(m, p, c, alpha=alpha, beta=beta, option_valid=ov, root_valid=rv,
                  use_lcvar=True, use_obs_kernel=True, top_m=top_m)
    opt = best_shared_option_index(res.q, p, gamma=0.0, root_valid=rv, option_valid=ov)
    drs = deployable_recovery_success(m, p, opt, root_valid=rv)
    rd = float(_scalar(d, "r_dep_star", res.r_dep))
    ro = float(_scalar(d, "r_orc_star", res.r_orc))
    return float(post_contact_deployability_score(drs, rd, max(0.0, ro - rd)))


def _group_deviation(items: list[dict[str, Any]]) -> list[float]:
    nominal = next((x for x in items if x["nominal"]), items[0])
    try:
        ref = np.asarray(nominal["data"]["prefix_states"], dtype=float)[:, :2]
    except Exception:
        return [0.0] * len(items)
    out: list[float] = []
    for x in items:
        try:
            xy = np.asarray(x["data"]["prefix_states"], dtype=float)[:, :2]
            t = min(len(ref), len(xy))
            out.append(0.0 if t <= 0 else float(np.sqrt(np.mean(np.sum((xy[:t] - ref[:t]) ** 2, axis=-1))) / 5.0))
        except Exception:
            out.append(0.0)
    return out


def _fold(scene: str, time_index: int) -> int:
    payload = f"{scene}|{int(time_index)}".encode("utf-8", errors="replace")
    return int.from_bytes(hashlib.sha1(payload).digest()[:8], "big") % 2


def _one_sided_wilson_upper(k: int, n: int, z: float = 1.6448536269514722) -> float:
    if n <= 0:
        return 1.0
    p = float(k) / float(n)
    z2 = z * z
    center = (p + z2 / (2.0 * n)) / (1.0 + z2 / n)
    radius = z * math.sqrt((p * (1.0 - p) / n) + z2 / (4.0 * n * n)) / (1.0 + z2 / n)
    return min(1.0, max(0.0, center + radius))


def _select_top1(groups: list[dict[str, Any]], opportunity_threshold: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for g in groups:
        eligible = [r for r in g["pairs"] if float(r["opportunity"]) >= opportunity_threshold]
        if not eligible:
            continue
        best = sorted(eligible, key=lambda r: (-float(r["pred_adv"]), int(r["candidate"])))[0]
        row = dict(best)
        row["oracle_best_teacher_adv"] = float(g["oracle_best_teacher_adv"])
        row["fold"] = int(g["fold"])
        row["scene"] = g["scene"]
        row["time"] = int(g["time"])
        selected.append(row)
    return selected


def _metrics(all_groups: list[dict[str, Any]], top1_rows: list[dict[str, Any]], score_threshold: float, positive_gain: float) -> dict[str, Any]:
    challenged = [r for r in top1_rows if float(r["pred_adv"]) >= score_threshold]
    positives = [r for r in challenged if float(r["teacher_adv"]) >= positive_gain]
    harmful = [r for r in challenged if float(r["teacher_adv"]) <= 0.0]
    opportunities = [g for g in all_groups if float(g["oracle_best_teacher_adv"]) >= positive_gain]
    captured = [r for r in positives if float(r["oracle_best_teacher_adv"]) >= positive_gain]
    n = len(all_groups)
    return {
        "num_groups": n,
        "num_top1_after_opportunity_gate": len(top1_rows),
        "num_selected": len(challenged),
        "selection_rate": float(len(challenged) / max(1, n)),
        "num_positive_selected": len(positives),
        "challenge_precision": float(len(positives) / len(challenged)) if challenged else None,
        "num_harmful_selected": len(harmful),
        "harmful_selected_rate": float(len(harmful) / len(challenged)) if challenged else None,
        "harmful_group_exposure": float(len(harmful) / max(1, n)),
        "harmful_group_exposure_ucb90": _one_sided_wilson_upper(len(harmful), n),
        "num_opportunities": len(opportunities),
        "positive_challenge_recall": float(len(captured) / len(opportunities)) if opportunities else None,
        "selected_teacher_advantage_mean": float(np.mean([r["teacher_adv"] for r in challenged])) if challenged else None,
        "selected_predicted_advantage_mean": float(np.mean([r["pred_adv"] for r in challenged])) if challenged else None,
        "selected_opportunity_mean": float(np.mean([r["opportunity"] for r in challenged])) if challenged else None,
        "selected_teacher_advantage_min": float(min(r["teacher_adv"] for r in challenged)) if challenged else None,
    }


def _candidate_opportunity_thresholds(groups: list[dict[str, Any]], min_value: float) -> list[float]:
    values = np.asarray([float(r["opportunity"]) for g in groups for r in g["pairs"]], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return []
    qs = np.linspace(0.0, 0.98, 25)
    vals = {float(min_value), *[float(np.quantile(values, q)) for q in qs]}
    vals.update(float(x) for x in values if x >= min_value and values.size <= 200)
    return sorted((x for x in vals if x >= min_value), reverse=True)


def _fit_rule(groups: list[dict[str, Any]], args: argparse.Namespace) -> tuple[float, float, dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for opp_thr in _candidate_opportunity_thresholds(groups, args.min_opportunity):
        top1 = _select_top1(groups, opp_thr)
        score_values = sorted({float(r["pred_adv"]) for r in top1 if np.isfinite(float(r["pred_adv"]))}, reverse=True)
        for score_thr in score_values:
            if score_thr < args.min_score_advantage:
                continue
            m = _metrics(groups, top1, score_thr, args.positive_gain)
            valid = (
                m["num_selected"] >= args.min_fit_selected
                and m["challenge_precision"] is not None and m["challenge_precision"] >= args.min_fit_precision
                and m["harmful_selected_rate"] is not None and m["harmful_selected_rate"] <= args.max_fit_harmful_selected_rate
            )
            if valid:
                row = dict(m)
                row["opportunity_threshold"] = float(opp_thr)
                row["score_threshold"] = float(score_thr)
                candidates.append(row)
    if not candidates:
        empty = _metrics(groups, [], float("inf"), args.positive_gain)
        return float("inf"), float("inf"), empty, []
    candidates.sort(key=lambda x: (
        int(x["num_positive_selected"]),
        float(x["challenge_precision"] or 0.0),
        int(x["num_selected"]),
        float(x["opportunity_threshold"]),
        float(x["score_threshold"]),
    ), reverse=True)
    best = candidates[0]
    return float(best["opportunity_threshold"]), float(best["score_threshold"]), best, candidates[:30]


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit/verify v45 OC-RAVE opportunity-gated risk certificate.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bucket", choices=["near", "contact"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rows-output", type=Path)
    ap.add_argument("--required-min-groups", type=int, default=60)
    ap.add_argument("--macro-ids", default="5,7")
    ap.add_argument("--min-nominal-deviation", type=float, default=0.002)
    ap.add_argument("--max-hard", type=float, default=1.0)
    ap.add_argument("--max-harm", type=float, default=0.70)
    ap.add_argument("--positive-gain", type=float, default=0.015)
    ap.add_argument("--min-opportunity", type=float, default=0.0)
    ap.add_argument("--min-score-advantage", type=float, default=-0.10)
    ap.add_argument("--min-fit-selected", type=int, default=4)
    ap.add_argument("--min-fit-precision", type=float, default=0.55)
    ap.add_argument("--max-fit-harmful-selected-rate", type=float, default=0.20)
    ap.add_argument("--min-verify-selected", type=int, default=2)
    ap.add_argument("--min-verify-precision", type=float, default=0.50)
    ap.add_argument("--max-verify-harmful-group-ucb", type=float, default=0.06)
    args = ap.parse_args()

    macro_ids = {int(x) for x in args.macro_ids.split(",") if x.strip()}
    runtime_cfg = {"selection": {"active_bucket_name": "near_contact" if args.bucket == "near" else "contact"}}
    bundle = load_model_bundle(args.checkpoint, runtime_cfg)
    if bundle is None:
        raise FileNotFoundError(args.checkpoint)
    if not bool(getattr(bundle.model, "direct_recovery_opportunity_head", False)):
        raise ValueError("v44 calibration requires model.direct_recovery_opportunity_head=true")

    alpha = float((bundle.cfg.get("ocmero", {}) or {}).get("alpha", 0.2))
    beta = float((bundle.cfg.get("ocmero", {}) or {}).get("beta", 0.2))
    top_m = int((bundle.cfg.get("ocmero", {}) or {}).get("top_m", 8))
    raw_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    paths = iter_sample_paths_many(args.dataset)
    for i, path in enumerate(paths, 1):
        split = str(scalar_metadata_for_path(path, "split_id", ""))
        if split not in {"calibration", "val"}:
            continue
        d = load_npz(path)
        pred = predict_sample(d, bundle, runtime_cfg)
        if pred.direct_recovery_value is None or pred.direct_recovery_opportunity is None:
            raise ValueError("checkpoint does not expose v44 score and opportunity outputs")
        row = {
            "data": d,
            "scene": str(_scalar(d, "scene_id", path.stem)),
            "time": int(_scalar(d, "time_index", 0)),
            "candidate": int(_scalar(d, "candidate_index", 0)),
            "macro": int(_scalar(d, "prefix_macro_type_id", _scalar(d, "prefix_macro_id", -1))),
            "nominal": bool(float(_scalar(d, "is_nominal", 0)) > 0.5),
            "pred": float(pred.direct_recovery_value),
            "opportunity": float(pred.direct_recovery_opportunity),
            "opportunity_logit": float(pred.direct_recovery_opportunity_logit if pred.direct_recovery_opportunity_logit is not None else math.log(max(float(pred.direct_recovery_opportunity), 1e-6) / max(1.0 - float(pred.direct_recovery_opportunity), 1e-6))),
            "teacher": _teacher_pcd(d, alpha, beta, top_m),
            "hard": float(_scalar(d, "hard_violation", 0.0)),
            "harm": float(_scalar(d, "harm_proxy", 0.0)),
            "feasible": bool(int(_scalar(d, "feasible", 1))),
        }
        raw_groups[(row["scene"], row["time"])].append(row)
        if i == 1 or i % 1000 == 0:
            print({"event": "v44_risk_calibration_progress", "bucket": args.bucket, "seen": i, "total": len(paths)}, flush=True)

    groups: list[dict[str, Any]] = []
    pair_errors: list[float] = []
    skipped = {"no_nominal": 0, "no_eligible": 0}
    for (scene, time_index), items in raw_groups.items():
        items.sort(key=lambda x: x["candidate"])
        for x, dv in zip(items, _group_deviation(items)):
            x["deviation"] = dv
        noms = [x for x in items if x["nominal"]]
        if not noms:
            skipped["no_nominal"] += 1
            continue
        nom = noms[0]
        recs = [x for x in items if (not x["nominal"]) and x["macro"] in macro_ids and x["feasible"]
                and x["hard"] <= args.max_hard and x["harm"] <= args.max_harm
                and x["deviation"] >= args.min_nominal_deviation]
        if not recs:
            skipped["no_eligible"] += 1
            continue
        pairs = []
        for r in recs:
            pred_adv = float(r["pred"] - nom["pred"])
            teacher_adv = float(r["teacher"] - nom["teacher"])
            opp_delta = float(np.clip(r["opportunity_logit"] - nom["opportunity_logit"], -30.0, 30.0))
            opportunity = float(1.0 / (1.0 + math.exp(-opp_delta)))
            pairs.append({
                "candidate": int(r["candidate"]), "macro": int(r["macro"]),
                "deviation": float(r["deviation"]), "pred_adv": pred_adv,
                "opportunity": opportunity, "opportunity_logit_delta": opp_delta,
                "teacher_adv": teacher_adv,
            })
            pair_errors.append(abs(pred_adv - teacher_adv))
        groups.append({
            "scene": scene, "time": int(time_index), "fold": _fold(scene, time_index),
            "pairs": pairs, "oracle_best_teacher_adv": float(max(r["teacher_adv"] for r in pairs)),
        })

    fit_groups = [g for g in groups if g["fold"] == 0]
    verify_groups = [g for g in groups if g["fold"] == 1]
    opp_thr, score_thr, fit_metrics, top_candidates = _fit_rule(fit_groups, args)
    verify_top1 = _select_top1(verify_groups, opp_thr) if np.isfinite(opp_thr) else []
    all_top1 = _select_top1(groups, opp_thr) if np.isfinite(opp_thr) else []
    verify_metrics = _metrics(verify_groups, verify_top1, score_thr, args.positive_gain)
    all_metrics = _metrics(groups, all_top1, score_thr, args.positive_gain)

    warnings: list[str] = []
    if len(groups) < args.required_min_groups:
        warnings.append(f"num_groups < required_min_groups ({len(groups)} < {args.required_min_groups})")
    if not np.isfinite(opp_thr) or not np.isfinite(score_thr):
        warnings.append("no fit-fold opportunity+score rule satisfied precision/risk/coverage constraints")
    if verify_metrics["num_selected"] < args.min_verify_selected:
        warnings.append(f"held-out selections < min_verify_selected ({verify_metrics['num_selected']} < {args.min_verify_selected})")
    if verify_metrics["challenge_precision"] is None or verify_metrics["challenge_precision"] < args.min_verify_precision:
        warnings.append("held-out challenge precision below requirement")
    if verify_metrics["harmful_group_exposure_ucb90"] > args.max_verify_harmful_group_ucb:
        warnings.append("held-out harmful group-exposure UCB exceeds risk budget")

    pair_rows = [r for g in groups for r in g["pairs"]]
    opp_values = np.asarray([r["opportunity"] for r in pair_rows], dtype=float)
    pred_values = np.asarray([r["pred_adv"] for r in pair_rows], dtype=float)
    teacher_values = np.asarray([r["teacher_adv"] for r in pair_rows], dtype=float)
    def _dist(a: np.ndarray) -> dict[str, float | None]:
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {"min": None, "q05": None, "q25": None, "median": None, "q75": None, "q95": None, "max": None}
        return {"min": float(a.min()), "q05": float(np.quantile(a, .05)), "q25": float(np.quantile(a, .25)),
                "median": float(np.quantile(a, .5)), "q75": float(np.quantile(a, .75)),
                "q95": float(np.quantile(a, .95)), "max": float(a.max())}
    corr = None
    if pred_values.size > 1 and np.std(pred_values) > 1e-12 and np.std(teacher_values) > 1e-12:
        corr = float(np.corrcoef(pred_values, teacher_values)[0, 1])
    macro_diag: dict[str, dict[str, float | int | None]] = {}
    for m in sorted({int(r["macro"]) for r in pair_rows}):
        rows_m = [r for r in pair_rows if int(r["macro"]) == m]
        macro_diag[str(m)] = {
            "count": len(rows_m),
            "positive_fraction": float(np.mean([float(r["teacher_adv"]) >= args.positive_gain for r in rows_m])) if rows_m else None,
            "teacher_advantage_mean": float(np.mean([r["teacher_adv"] for r in rows_m])) if rows_m else None,
            "predicted_advantage_mean": float(np.mean([r["pred_adv"] for r in rows_m])) if rows_m else None,
            "opportunity_mean": float(np.mean([r["opportunity"] for r in rows_m])) if rows_m else None,
        }

    valid = (
        len(groups) >= args.required_min_groups and np.isfinite(opp_thr) and np.isfinite(score_thr)
        and verify_metrics["num_selected"] >= args.min_verify_selected
        and verify_metrics["challenge_precision"] is not None
        and verify_metrics["challenge_precision"] >= args.min_verify_precision
        and verify_metrics["harmful_group_exposure_ucb90"] <= args.max_verify_harmful_group_ucb
    )
    result = {
        "method": "oc_rave_regime_expert_relative_opportunity_selective_certificate",
        "bucket": args.bucket,
        "selection_rule": "opportunity_gate_then_highest_score_advantage_then_smallest_candidate_index",
        "dataset": args.dataset, "checkpoint": args.checkpoint,
        "valid_for_deployment": bool(valid),
        "direct_value_uncertainty_mode": "risk_selective",
        "direct_value_opportunity_threshold": float(opp_thr),
        "direct_value_threshold": float(score_thr),
        "direct_value_min_advantage_lcb": float(score_thr),
        "direct_value_score_mode": True, "direct_value_top1_only": True,
        "direct_value_risk_controlled_admission": True,
        "num_scene_time_groups": len(raw_groups), "num_calibration_groups": len(groups),
        "fit_groups": len(fit_groups), "verify_groups": len(verify_groups),
        "all_pair_advantage_mae": float(np.mean(pair_errors)) if pair_errors else None,
        "positive_gain": args.positive_gain, "min_nominal_deviation": args.min_nominal_deviation,
        "max_hard": args.max_hard, "max_harm": args.max_harm, "macro_ids": sorted(macro_ids),
        "constraints": {
            "min_fit_selected": args.min_fit_selected, "min_fit_precision": args.min_fit_precision,
            "max_fit_harmful_selected_rate": args.max_fit_harmful_selected_rate,
            "min_verify_selected": args.min_verify_selected, "min_verify_precision": args.min_verify_precision,
            "max_verify_harmful_group_ucb": args.max_verify_harmful_group_ucb,
        },
        "fit": fit_metrics, "verify": verify_metrics, "all": all_metrics,
        "opportunity_distribution": _dist(opp_values),
        "predicted_advantage_distribution": _dist(pred_values),
        "teacher_advantage_distribution": _dist(teacher_values),
        "pred_teacher_advantage_correlation": corr,
        "macro_diagnostics": macro_diag,
        "top_fit_rule_candidates": top_candidates, "skipped": skipped, "warnings": warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    if args.rows_output is not None:
        args.rows_output.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_rows = all_top1
        diagnostic_score_thr = score_thr
        if not diagnostic_rows and groups:
            min_opp = float(np.min(opp_values)) if opp_values.size else 0.0
            diagnostic_rows = _select_top1(groups, min_opp)
            diagnostic_score_thr = float("inf")
        with args.rows_output.open("w") as f:
            for row in diagnostic_rows:
                out = dict(row)
                out["challenged"] = bool(float(row["pred_adv"]) >= diagnostic_score_thr)
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
