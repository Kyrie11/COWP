#!/usr/bin/env python3
"""Fit and verify the v43 OC-RSC top-1 selective value certificate.

The model deterministically selects the highest-score actionable recovery
candidate in each scene-time group. A score-advantage threshold is fitted on
one deterministic fold and verified on a disjoint fold. The certificate is
valid only when it has non-zero held-out coverage, sufficient positive
precision, and a one-sided upper confidence bound on harmful challenge
*group exposure* below the configured risk budget.

This is deliberately not a residual/additive conformal bound: v42 showed that
such a bound had zero deployment coverage. OC-RSC instead provides an explicit
selective-risk contract and abstains below the verified threshold.
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
from ocrap.evaluation.metrics import (
    best_shared_option_index,
    deployable_recovery_success,
    post_contact_deployability_score,
)
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


def _fold(scene: str, time_index: int, modulus: int = 2) -> int:
    payload = f"{scene}|{int(time_index)}".encode("utf-8", errors="replace")
    return int.from_bytes(hashlib.sha1(payload).digest()[:8], "big") % max(2, int(modulus))


def _one_sided_wilson_upper(k: int, n: int, z: float = 1.6448536269514722) -> float:
    """One-sided Wilson upper bound; returns 1 when n is empty."""
    if n <= 0:
        return 1.0
    p = float(k) / float(n)
    z2 = z * z
    center = (p + z2 / (2.0 * n)) / (1.0 + z2 / n)
    radius = z * math.sqrt((p * (1.0 - p) / n) + z2 / (4.0 * n * n)) / (1.0 + z2 / n)
    return min(1.0, max(0.0, center + radius))


def _metrics(rows: list[dict[str, Any]], threshold: float, positive_gain: float) -> dict[str, Any]:
    n = len(rows)
    selected = [r for r in rows if float(r["pred_adv"]) >= threshold]
    positives = [r for r in selected if float(r["teacher_adv"]) >= positive_gain]
    harmful = [r for r in selected if float(r["teacher_adv"]) <= 0.0]
    opportunities = [r for r in rows if float(r["oracle_best_teacher_adv"]) >= positive_gain]
    captured = [r for r in selected if float(r["teacher_adv"]) >= positive_gain and float(r["oracle_best_teacher_adv"]) >= positive_gain]
    return {
        "num_groups": n,
        "num_selected": len(selected),
        "selection_rate": float(len(selected) / max(1, n)),
        "num_positive_selected": len(positives),
        "challenge_precision": float(len(positives) / max(1, len(selected))) if selected else None,
        "num_harmful_selected": len(harmful),
        "harmful_selected_rate": float(len(harmful) / max(1, len(selected))) if selected else None,
        "harmful_group_exposure": float(len(harmful) / max(1, n)),
        "harmful_group_exposure_ucb90": _one_sided_wilson_upper(len(harmful), n),
        "num_opportunities": len(opportunities),
        "positive_challenge_recall": float(len(captured) / max(1, len(opportunities))) if opportunities else None,
        "selected_teacher_advantage_mean": float(np.mean([r["teacher_adv"] for r in selected])) if selected else None,
        "selected_predicted_advantage_mean": float(np.mean([r["pred_adv"] for r in selected])) if selected else None,
        "selected_teacher_advantage_min": float(min(r["teacher_adv"] for r in selected)) if selected else None,
    }


def _fit_threshold(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    finite = sorted({float(r["pred_adv"]) for r in rows if np.isfinite(float(r["pred_adv"]))}, reverse=True)
    candidates: list[dict[str, Any]] = []
    for threshold in finite:
        if threshold < args.min_score_advantage:
            continue
        m = _metrics(rows, threshold, args.positive_gain)
        precision = m["challenge_precision"]
        harmful_rate = m["harmful_selected_rate"]
        valid = (
            m["num_selected"] >= args.min_fit_selected
            and precision is not None and precision >= args.min_fit_precision
            and harmful_rate is not None and harmful_rate <= args.max_fit_harmful_selected_rate
        )
        if valid:
            row = dict(m)
            row["threshold"] = threshold
            candidates.append(row)
    if not candidates:
        return float("inf"), _metrics(rows, float("inf"), args.positive_gain), []
    # Prefer the rule that recovers the largest number of truly positive groups;
    # break ties by precision, total coverage and finally a stricter threshold.
    candidates.sort(
        key=lambda x: (
            int(x["num_positive_selected"]),
            float(x["challenge_precision"] or 0.0),
            int(x["num_selected"]),
            float(x["threshold"]),
        ),
        reverse=True,
    )
    best = candidates[0]
    return float(best["threshold"]), best, candidates[:20]


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit/verify OC-RSC selective top-1 value certificate.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rows-output", type=Path)
    ap.add_argument("--required-min-groups", type=int, default=60)
    ap.add_argument("--macro-ids", default="2,3,5,7")
    ap.add_argument("--min-nominal-deviation", type=float, default=0.002)
    ap.add_argument("--max-hard", type=float, default=1.0)
    ap.add_argument("--max-harm", type=float, default=0.70)
    ap.add_argument("--positive-gain", type=float, default=0.025)
    ap.add_argument("--min-score-advantage", type=float, default=0.0)
    ap.add_argument("--min-fit-selected", type=int, default=4)
    ap.add_argument("--min-fit-precision", type=float, default=0.60)
    ap.add_argument("--max-fit-harmful-selected-rate", type=float, default=0.10)
    ap.add_argument("--min-verify-selected", type=int, default=2)
    ap.add_argument("--min-verify-precision", type=float, default=0.50)
    ap.add_argument("--max-verify-harmful-group-ucb", type=float, default=0.05)
    args = ap.parse_args()

    macro_ids = {int(x) for x in args.macro_ids.split(",") if x.strip()}
    bundle = load_model_bundle(args.checkpoint, {})
    if bundle is None:
        raise FileNotFoundError(args.checkpoint)
    if str(getattr(bundle.model, "direct_recovery_value_output", "probability")) != "score":
        raise ValueError("v43 risk calibration requires model.direct_recovery_value_output=score")

    alpha = float((bundle.cfg.get("ocmero", {}) or {}).get("alpha", 0.2))
    beta = float((bundle.cfg.get("ocmero", {}) or {}).get("beta", 0.2))
    top_m = int((bundle.cfg.get("ocmero", {}) or {}).get("top_m", 8))
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    paths = iter_sample_paths_many(args.dataset)
    for i, path in enumerate(paths, 1):
        split = str(scalar_metadata_for_path(path, "split_id", ""))
        if split not in {"calibration", "val"}:
            continue
        d = load_npz(path)
        pred = predict_sample(d, bundle, {})
        if pred.direct_recovery_value is None:
            raise ValueError("checkpoint has no direct recovery score head")
        row = {
            "data": d,
            "scene": str(_scalar(d, "scene_id", path.stem)),
            "time": int(_scalar(d, "time_index", 0)),
            "candidate": int(_scalar(d, "candidate_index", 0)),
            "macro": int(_scalar(d, "prefix_macro_type_id", _scalar(d, "prefix_macro_id", -1))),
            "nominal": bool(float(_scalar(d, "is_nominal", 0)) > 0.5),
            "pred": float(pred.direct_recovery_value),
            "teacher": _teacher_pcd(d, alpha, beta, top_m),
            "hard": float(_scalar(d, "hard_violation", 0.0)),
            "harm": float(_scalar(d, "harm_proxy", 0.0)),
            "feasible": bool(int(_scalar(d, "feasible", 1))),
        }
        groups[(row["scene"], row["time"])].append(row)
        if i == 1 or i % 1000 == 0:
            print({"event": "v43_risk_calibration_progress", "seen": i, "total": len(paths)}, flush=True)

    selected_rows: list[dict[str, Any]] = []
    all_pair_errors: list[float] = []
    skipped = {"no_nominal": 0, "no_eligible": 0}
    eligible_counts: list[int] = []
    for (scene, time_index), items in groups.items():
        items.sort(key=lambda x: x["candidate"])
        for x, dv in zip(items, _group_deviation(items)):
            x["deviation"] = dv
        noms = [x for x in items if x["nominal"]]
        if not noms:
            skipped["no_nominal"] += 1
            continue
        nom = noms[0]
        recs = [
            x for x in items
            if (not x["nominal"]) and x["macro"] in macro_ids and x["feasible"]
            and x["hard"] <= args.max_hard and x["harm"] <= args.max_harm
            and x["deviation"] >= args.min_nominal_deviation
        ]
        if not recs:
            skipped["no_eligible"] += 1
            continue
        eligible_counts.append(len(recs))
        pairs: list[tuple[dict[str, Any], float, float]] = []
        for r in recs:
            pred_adv = float(r["pred"] - nom["pred"])
            teacher_adv = float(r["teacher"] - nom["teacher"])
            pairs.append((r, pred_adv, teacher_adv))
            all_pair_errors.append(abs(pred_adv - teacher_adv))
        selected = sorted(pairs, key=lambda z: (-z[1], int(z[0]["candidate"])))[0]
        oracle_best = max(x[2] for x in pairs)
        selected_rows.append({
            "scene": scene,
            "time": int(time_index),
            "fold": _fold(scene, time_index),
            "candidate": int(selected[0]["candidate"]),
            "macro": int(selected[0]["macro"]),
            "deviation": float(selected[0]["deviation"]),
            "pred_adv": float(selected[1]),
            "teacher_adv": float(selected[2]),
            "oracle_best_teacher_adv": float(oracle_best),
        })

    fit_rows = [r for r in selected_rows if int(r["fold"]) == 0]
    verify_rows = [r for r in selected_rows if int(r["fold"]) == 1]
    threshold, fit_metrics, top_candidates = _fit_threshold(fit_rows, args)
    verify_metrics = _metrics(verify_rows, threshold, args.positive_gain)
    all_metrics = _metrics(selected_rows, threshold, args.positive_gain)

    warnings: list[str] = []
    if len(selected_rows) < args.required_min_groups:
        warnings.append(f"num_groups < required_min_groups ({len(selected_rows)} < {args.required_min_groups})")
    if not np.isfinite(threshold):
        warnings.append("no fit-fold threshold satisfied precision/risk/coverage constraints")
    if verify_metrics["num_selected"] < args.min_verify_selected:
        warnings.append(f"held-out selections < min_verify_selected ({verify_metrics['num_selected']} < {args.min_verify_selected})")
    if verify_metrics["challenge_precision"] is None or verify_metrics["challenge_precision"] < args.min_verify_precision:
        warnings.append("held-out challenge precision below requirement")
    if verify_metrics["harmful_group_exposure_ucb90"] > args.max_verify_harmful_group_ucb:
        warnings.append("held-out harmful group-exposure UCB exceeds risk budget")

    valid = (
        len(selected_rows) >= args.required_min_groups
        and np.isfinite(threshold)
        and verify_metrics["num_selected"] >= args.min_verify_selected
        and verify_metrics["challenge_precision"] is not None
        and verify_metrics["challenge_precision"] >= args.min_verify_precision
        and verify_metrics["harmful_group_exposure_ucb90"] <= args.max_verify_harmful_group_ucb
    )
    result = {
        "method": "oc_rsc_disjoint_fit_verify_top1_selective_risk_certificate",
        "selection_rule": "highest_predicted_actionable_advantage_then_smallest_candidate_index",
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "valid_for_deployment": bool(valid),
        "direct_value_uncertainty_mode": "risk_selective",
        "direct_value_threshold": float(threshold),
        "direct_value_min_advantage_lcb": float(threshold),
        "direct_value_score_mode": True,
        "direct_value_top1_only": True,
        "direct_value_risk_controlled_admission": True,
        "num_scene_time_groups": len(groups),
        "num_calibration_groups": len(selected_rows),
        "fit_groups": len(fit_rows),
        "verify_groups": len(verify_rows),
        "eligible_candidates_mean": float(np.mean(eligible_counts)) if eligible_counts else None,
        "all_pair_advantage_mae": float(np.mean(all_pair_errors)) if all_pair_errors else None,
        "positive_gain": args.positive_gain,
        "min_nominal_deviation": args.min_nominal_deviation,
        "max_hard": args.max_hard,
        "max_harm": args.max_harm,
        "macro_ids": sorted(macro_ids),
        "constraints": {
            "min_fit_selected": args.min_fit_selected,
            "min_fit_precision": args.min_fit_precision,
            "max_fit_harmful_selected_rate": args.max_fit_harmful_selected_rate,
            "min_verify_selected": args.min_verify_selected,
            "min_verify_precision": args.min_verify_precision,
            "max_verify_harmful_group_ucb": args.max_verify_harmful_group_ucb,
        },
        "fit": fit_metrics,
        "verify": verify_metrics,
        "all": all_metrics,
        "top_fit_threshold_candidates": top_candidates,
        "skipped": skipped,
        "warnings": warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    if args.rows_output is not None:
        args.rows_output.parent.mkdir(parents=True, exist_ok=True)
        with args.rows_output.open("w") as f:
            for row in selected_rows:
                out = dict(row)
                out["challenged"] = bool(np.isfinite(threshold) and float(row["pred_adv"]) >= threshold)
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    # A non-deployable model is an experimental result, not a tool crash. The
    # staged pipeline reads valid_for_deployment and decides whether to stop.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
