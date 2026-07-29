#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ocrap.algorithms.lcv import finite_sample_upper_quantile
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
    value = d.get(key, default)
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return value


def _teacher_pcd(d: dict[str, Any], *, alpha: float, beta: float, top_m: int) -> float:
    m = np.asarray(d["m_star"], dtype=np.float64)
    p = np.asarray(d["root_probs"], dtype=np.float64)
    c = np.asarray(d.get("c_star", np.eye(m.shape[0])), dtype=np.float64)
    root_valid = np.asarray(d.get("root_valid", np.ones(m.shape[0])), dtype=bool)
    option_valid = np.asarray(d.get("option_valid", np.ones(m.shape[1])), dtype=bool)
    result = oc_mero(
        m,
        p,
        c,
        alpha=alpha,
        beta=beta,
        option_valid=option_valid,
        root_valid=root_valid,
        use_lcvar=True,
        use_obs_kernel=True,
        top_m=top_m,
    )
    option = best_shared_option_index(
        result.q,
        p,
        gamma=0.0,
        root_valid=root_valid,
        option_valid=option_valid,
    )
    drs = deployable_recovery_success(m, p, option, root_valid=root_valid)
    r_dep = float(_scalar(d, "r_dep_star", result.r_dep))
    r_orc = float(_scalar(d, "r_orc_star", result.r_orc))
    return float(post_contact_deployability_score(drs, r_dep, max(0.0, r_orc - r_dep)))


def _load_rows(dataset: str, checkpoint: str, cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    paths = iter_sample_paths_many(dataset)
    bundle = load_model_bundle(checkpoint, cfg)
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    alpha = float((cfg.get("ocmero", {}) or {}).get("alpha", 0.2))
    beta = float((cfg.get("ocmero", {}) or {}).get("beta", 0.2))
    top_m = int((cfg.get("ocmero", {}) or {}).get("top_m", 8))
    for i, path in enumerate(paths, 1):
        if i == 1 or i % 1000 == 0:
            print({"event": "direct_value_calibration_progress", "seen": i, "total": len(paths)}, flush=True)
        split = str(scalar_metadata_for_path(path, "split_id", ""))
        if split not in {"calibration", "val"}:
            continue
        d = load_npz(path)
        pred = predict_sample(d, bundle, cfg)
        if pred.direct_recovery_value is None or pred.direct_recovery_std is None:
            raise ValueError("checkpoint does not contain the v40 direct recovery value/uncertainty head")
        scene_id = str(_scalar(d, "scene_id", path.stem))
        time_index = int(_scalar(d, "time_index", 0))
        macro_id = int(_scalar(d, "prefix_macro_type_id", _scalar(d, "prefix_macro_id", -1)))
        rows_by_split[split].append({
            "scene_id": scene_id,
            "time_index": time_index,
            "candidate_index": int(_scalar(d, "candidate_index", 0)),
            "macro_id": macro_id,
            "is_nominal": bool(float(_scalar(d, "is_nominal", macro_id in {0, 1})) > 0.5),
            "pred_value": float(pred.direct_recovery_value),
            "pred_std": max(1.0e-4, float(pred.direct_recovery_std)),
            "teacher_pcd": _teacher_pcd(d, alpha=alpha, beta=beta, top_m=top_m),
        })
    if rows_by_split.get("calibration"):
        return rows_by_split["calibration"], ["calibration"]
    return rows_by_split.get("val", []), ["val"] if rows_by_split.get("val") else []


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Calibrate a scene-time simultaneous lower-confidence bound for the "
            "direct recovery-value advantage. The nonconformity score is the "
            "maximum standardized advantage over-estimation across all recovery "
            "candidates in a scene-time group, making the bound valid after selection."
        )
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--required-min-groups", type=int, default=50)
    ap.add_argument("--macro-ids", default="2,3,5,7")
    ap.add_argument("--numerical-margin", type=float, default=0.05)
    args = ap.parse_args()
    macro_ids = {int(x.strip()) for x in args.macro_ids.split(",") if x.strip()}
    cfg: dict[str, Any] = {}
    rows, used_splits = _load_rows(args.dataset, args.checkpoint, cfg)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["scene_id"], row["time_index"])].append(row)

    group_scores: list[float] = []
    raw_pair_scores: list[float] = []
    true_advantages: list[float] = []
    pair_count = 0
    skipped_no_nominal = 0
    skipped_no_recovery = 0
    for items in groups.values():
        items.sort(key=lambda x: x["candidate_index"])
        noms = [x for x in items if x["is_nominal"]]
        recs = [x for x in items if x["macro_id"] in macro_ids and not x["is_nominal"]]
        if not noms:
            skipped_no_nominal += 1
            continue
        if not recs:
            skipped_no_recovery += 1
            continue
        nom = noms[0]
        scores = []
        for rec in recs:
            true_adv = float(rec["teacher_pcd"] - nom["teacher_pcd"])
            pred_adv = float(rec["pred_value"] - nom["pred_value"])
            pair_std = math.sqrt(max(1.0e-8, rec["pred_std"] ** 2 + nom["pred_std"] ** 2))
            score = float((pred_adv - true_adv) / pair_std)
            scores.append(score)
            raw_pair_scores.append(score)
            true_advantages.append(true_adv)
            pair_count += 1
        # Max-over-candidates calibration protects the selected candidate, not
        # merely a fixed candidate chosen before seeing the model scores.
        group_scores.append(max(scores))

    warnings: list[str] = []
    z = float("inf")
    if not group_scores:
        warnings.append("no complete scene-time groups with nominal and recovery candidates")
    else:
        z = finite_sample_upper_quantile(
            np.asarray(group_scores, dtype=np.float64),
            float(args.delta),
            numerical_margin=float(args.numerical_margin),
            strict=True,
        )
        z = max(0.0, float(z))
    if len(group_scores) < int(args.required_min_groups):
        warnings.append(
            f"num_groups < required_min_groups ({len(group_scores)} < {int(args.required_min_groups)})"
        )
    empirical_coverage = None
    if np.isfinite(z) and group_scores:
        empirical_coverage = float(np.mean(np.asarray(group_scores) <= z))

    result = {
        "method": "scene_time_simultaneous_conformal_advantage_lcb",
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "splits": used_splits,
        "delta": float(args.delta),
        "direct_value_lcb_z": z,
        "num_rows": len(rows),
        "num_scene_time_groups": len(groups),
        "num_calibration_groups": len(group_scores),
        "num_candidate_pairs": pair_count,
        "skipped_no_nominal": skipped_no_nominal,
        "skipped_no_recovery": skipped_no_recovery,
        "macro_ids": sorted(macro_ids),
        "numerical_margin": float(args.numerical_margin),
        "empirical_group_coverage": empirical_coverage,
        "score_mean": float(np.mean(group_scores)) if group_scores else None,
        "score_p95": float(np.quantile(group_scores, 0.95)) if group_scores else None,
        "pair_score_mean": float(np.mean(raw_pair_scores)) if raw_pair_scores else None,
        "positive_teacher_advantage_fraction": float(np.mean(np.asarray(true_advantages) > 0.0)) if true_advantages else None,
        "warnings": warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, ensure_ascii=False))
    return 0 if np.isfinite(z) else 2


if __name__ == "__main__":
    raise SystemExit(main())
