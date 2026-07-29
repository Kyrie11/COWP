#!/usr/bin/env python3
"""Scene-disjoint joint calibration for v48 OC-TRAC-SR.

The calibrated policy is exactly the deployed policy:
  physical eligibility -> opportunity LCB -> harm UCB -> robust top-1 score
  -> challenge nominal only above a score-advantage threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ocrap.algorithms.ocmero import oc_mero
from ocrap.data.serialization import load_npz
from ocrap.evaluation.metrics import best_shared_option_index, deployable_recovery_success, post_contact_deployability_score
from ocrap.models.data import iter_sample_paths_many, scalar_metadata_for_path
from ocrap.models.inference import load_model_bundle, predict_samples


def _scalar(d: dict[str, Any], key: str, default: Any) -> Any:
    a = np.asarray(d.get(key, default))
    return a.item() if a.shape == () else a


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -30.0, 30.0))
    return float(1.0 / (1.0 + math.exp(-x)))


def _normal_cdf(x: float) -> float:
    return float(0.5 * (1.0 + math.erf(float(np.clip(x, -12.0, 12.0)) / math.sqrt(2.0))))


def _finite_sample_upper_quantile(values: list[float], alpha: float) -> float:
    """One-sided split-conformal finite-sample quantile."""
    a = np.asarray([float(x) for x in values if np.isfinite(x)], dtype=float)
    if a.size == 0:
        return float("inf")
    a.sort()
    rank = int(math.ceil((a.size + 1) * (1.0 - float(alpha))))
    rank = min(max(rank, 1), int(a.size))
    return float(a[rank - 1])


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


def _fold(scene: str, folds: int) -> int:
    return int.from_bytes(hashlib.sha1(scene.encode("utf-8", errors="replace")).digest()[:8], "big") % folds


def _wilson(k: int, n: int, *, upper: bool, z: float = 1.6448536269514722) -> float:
    if n <= 0:
        return 1.0 if upper else 0.0
    p = float(k) / float(n)
    z2 = z * z
    center = (p + z2 / (2.0 * n)) / (1.0 + z2 / n)
    radius = z * math.sqrt((p * (1.0 - p) / n) + z2 / (4.0 * n * n)) / (1.0 + z2 / n)
    return float(min(1.0, center + radius) if upper else max(0.0, center - radius))


def _auc(labels: list[bool], scores: list[float]) -> float | None:
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=float)
    pos = s[y]
    neg = s[~y]
    if pos.size == 0 or neg.size == 0:
        return None
    # Mann-Whitney form, tie-aware and dependency free.
    wins = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((wins + 0.5 * ties) / (pos.size * neg.size))


def _grid(values: list[float], *, minimum: float | None = None, maximum: float | None = None, n: int = 15, reverse: bool = False) -> list[float]:
    a = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if a.size == 0:
        return []
    vals = {float(np.quantile(a, q)) for q in np.linspace(0.0, 1.0, n)}
    if minimum is not None:
        vals.add(float(minimum))
    if maximum is not None:
        vals.add(float(maximum))
    out = [x for x in vals if (minimum is None or x >= minimum) and (maximum is None or x <= maximum)]
    return sorted(out, reverse=reverse)


def _top1(
    groups: list[dict[str, Any]],
    opp_thr: float,
    harm_thr: float,
    supported_macros: set[int],
    *,
    conditional_rank_margin: bool = False,
    policy_first_no_fallback: bool = False,
) -> list[dict[str, Any]]:
    """Select one recovery candidate per group.

    With ``policy_first_no_fallback`` the frozen preference policy chooses its
    top recovery before evidence is checked.  If that candidate is uncertified,
    the policy abstains instead of silently falling through to a lower-ranked
    candidate that Stage-E never trained on.
    """
    out: list[dict[str, Any]] = []
    for g in groups:
        physical = [r for r in g["pairs"] if r["macro"] in supported_macros]
        if not physical:
            continue
        if policy_first_no_fallback:
            best = sorted(physical, key=lambda r: (-r.get("rank_adv", r["pred_adv"]), r["candidate"]))[0]
            alternatives = [float(r.get("rank_adv", r["pred_adv"])) for r in physical if r is not best]
            if not conditional_rank_margin:
                alternatives.append(0.0)
            second = max(alternatives) if alternatives else float(best.get("rank_adv", best["pred_adv"]) - 1.0)
            rank_margin = float(best.get("rank_adv", best["pred_adv"]) - second)
            if best["opportunity"] < opp_thr or best["harm"] > harm_thr:
                continue
        else:
            eligible = [r for r in physical if r["opportunity"] >= opp_thr and r["harm"] <= harm_thr]
            if not eligible:
                continue
            best = sorted(eligible, key=lambda r: (-r.get("rank_adv", r["pred_adv"]), r["candidate"]))[0]
            alternatives = [float(r.get("rank_adv", r["pred_adv"])) for r in eligible if r is not best]
            if not conditional_rank_margin:
                alternatives.append(0.0)
            second = max(alternatives) if alternatives else float(best.get("rank_adv", best["pred_adv"]) - 1.0)
            rank_margin = float(best.get("rank_adv", best["pred_adv"]) - second)
        row = dict(best)
        row["rank_margin"] = rank_margin
        row.update(scene=g["scene"], time=g["time"], fold=g["fold"], oracle_best_teacher_adv=g["oracle_best_teacher_adv"])
        out.append(row)
    return out


def _policy_top1_pairs(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the frozen preference policy's recovery candidate per group.

    Certificate calibration must match the distribution on which the policy is
    actually evaluated.  Fitting residuals over every unused recovery candidate
    made v48.8's global conformal radius exceed the full prediction range.
    """
    out: list[dict[str, Any]] = []
    for g in groups:
        if not g.get("pairs"):
            continue
        best = sorted(
            g["pairs"],
            key=lambda r: (-float(r.get("rank_adv", r["pred_adv"])), int(r["candidate"])),
        )[0]
        row = dict(best)
        row.update(scene=g["scene"], time=g["time"], fold=g["fold"])
        out.append(row)
    return out


def _metrics(groups: list[dict[str, Any]], top1: list[dict[str, Any]], score_thr: float, rank_margin_thr: float, pos_gain: float, neg_gain: float) -> dict[str, Any]:
    selected = [r for r in top1 if r["pred_adv"] >= score_thr and r.get("rank_margin", 0.0) >= rank_margin_thr]
    positive = [r for r in selected if r["teacher_adv"] >= pos_gain]
    harmful = [r for r in selected if r["teacher_adv"] <= -neg_gain]
    opportunities = [g for g in groups if g["oracle_best_teacher_adv"] >= pos_gain]
    macro_counts = Counter(int(r["macro"]) for r in selected)
    max_macro_share = max(macro_counts.values(), default=0) / max(1, len(selected))
    # v48.12 TRIDENT: an absolute diversity cap is invalid when the teacher
    # opportunity distribution itself is concentrated.  Measure whether the
    # learned selector is *more* concentrated than the oracle-positive policy,
    # while retaining the raw share for reporting.
    oracle_macro_counts: Counter[int] = Counter()
    for g in opportunities:
        positive_pairs = [r for r in g.get("pairs", []) if r.get("teacher_adv", -1.0e9) >= pos_gain]
        if not positive_pairs:
            continue
        oracle_best = sorted(
            positive_pairs,
            key=lambda r: (-float(r["teacher_adv"]), int(r["candidate"])),
        )[0]
        oracle_macro_counts[int(oracle_best["macro"])] += 1
    oracle_max_macro_share = (
        max(oracle_macro_counts.values(), default=0) / max(1, sum(oracle_macro_counts.values()))
    )
    macro_excess_share = max(0.0, float(max_macro_share) - float(oracle_max_macro_share))
    return {
        "num_groups": len(groups),
        "num_top1_after_joint_gate": len(top1),
        "num_selected": len(selected),
        "selection_rate": len(selected) / max(1, len(groups)),
        "num_positive_selected": len(positive),
        "precision": len(positive) / len(selected) if selected else None,
        "precision_wilson_lcb90": _wilson(len(positive), len(selected), upper=False) if selected else None,
        "num_harmful_selected": len(harmful),
        "harmful_selected_rate": len(harmful) / len(selected) if selected else None,
        "harmful_selected_ucb90": _wilson(len(harmful), len(selected), upper=True) if selected else 1.0,
        "harmful_group_exposure": len(harmful) / max(1, len(groups)),
        "harmful_group_exposure_ucb90": _wilson(len(harmful), len(groups), upper=True),
        "num_opportunities": len(opportunities),
        "positive_recall": len(positive) / len(opportunities) if opportunities else None,
        "teacher_advantage_mean": float(np.mean([r["teacher_adv"] for r in selected])) if selected else None,
        "teacher_advantage_min": float(min(r["teacher_adv"] for r in selected)) if selected else None,
        "selected_macro_counts": dict(sorted(macro_counts.items())),
        "max_selected_macro_share": float(max_macro_share),
        "oracle_positive_macro_counts": dict(sorted(oracle_macro_counts.items())),
        "oracle_positive_max_macro_share": float(oracle_max_macro_share),
        "selected_macro_excess_share": float(macro_excess_share),
    }


def _fit(
    groups: list[dict[str, Any]],
    args: argparse.Namespace,
    supported_macros: set[int],
) -> tuple[dict[str, float] | None, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = [r for g in groups for r in g["pairs"] if r["macro"] in supported_macros]
    opp_grid = _grid(
        [r["opportunity"] for r in pairs],
        minimum=args.min_opportunity,
        n=args.grid_size,
        reverse=True,
    )
    harm_grid = _grid(
        [r["harm"] for r in pairs],
        maximum=args.max_harm_probability,
        n=args.grid_size,
        reverse=False,
    )
    # Preserve a diagnostic frontier even when every score lies outside the
    # hard probability bounds.  Such rows can never become deployable rules,
    # but they reveal whether failure comes from opportunity saturation, harm
    # saturation, precision, support, or macro concentration.
    diagnostic_probability_violation = False
    if not opp_grid:
        opp_grid = _grid([r["opportunity"] for r in pairs], n=args.grid_size, reverse=True)
        diagnostic_probability_violation = True
    if not harm_grid:
        harm_grid = _grid([r["harm"] for r in pairs], n=args.grid_size, reverse=False)
        diagnostic_probability_violation = True
    candidates: list[dict[str, Any]] = []
    frontier: list[dict[str, Any]] = []
    for opp_thr in opp_grid:
        for harm_thr in harm_grid:
            top1 = _top1(groups, opp_thr, harm_thr, supported_macros, conditional_rank_margin=args.conditional_recovery_ranking, policy_first_no_fallback=args.policy_first_no_fallback)
            score_grid = _grid(
                [r["pred_adv"] for r in top1],
                minimum=args.min_score_advantage,
                n=args.grid_size,
                reverse=True,
            )
            rank_grid = _grid(
                [r.get("rank_margin", 0.0) for r in top1],
                minimum=args.min_rank_margin,
                n=max(7, args.grid_size // 2),
                reverse=True,
            )
            for score_thr in score_grid:
                for rank_margin_thr in rank_grid:
                    m = _metrics(
                        groups, top1, score_thr, rank_margin_thr,
                        args.positive_gain, args.negative_gain,
                    )
                    lcb = float(m["precision_wilson_lcb90"] or 0.0)
                    harm_ucb = float(m["harmful_group_exposure_ucb90"])
                    conditional_harm_ucb = float(m["harmful_selected_ucb90"])
                    mean_adv = float(m["teacher_advantage_mean"] or -1.0)
                    macro_violation = (
                        float(m.get("selected_macro_excess_share", 0.0)) - args.max_macro_excess_share
                        if args.macro_constraint_mode == "opportunity_normalized"
                        else float(m["max_selected_macro_share"]) - args.max_selected_macro_share
                    )
                    probability_deficit = (
                        max(0.0, float(args.min_opportunity) - float(opp_thr))
                        + max(0.0, float(harm_thr) - float(args.max_harm_probability))
                    )
                    deficit = (
                        probability_deficit
                        + max(0, args.min_fit_selected - int(m["num_selected"])) / max(args.min_fit_selected, 1)
                        + max(0.0, args.min_fit_precision_lcb - lcb)
                        + max(0.0, harm_ucb - args.max_fit_harmful_group_ucb)
                        + max(0.0, conditional_harm_ucb - args.max_fit_harmful_selected_ucb)
                        + max(0.0, macro_violation)
                        + max(0.0, -mean_adv)
                    )
                    row = dict(m)
                    row.update(
                        opportunity_threshold=opp_thr,
                        harm_threshold=harm_thr,
                        score_threshold=score_thr,
                        rank_margin_threshold=rank_margin_thr,
                        constraint_deficit=float(deficit),
                    )
                    frontier.append(row)
                    if (
                        opp_thr >= args.min_opportunity
                        and harm_thr <= args.max_harm_probability
                        and m["num_selected"] >= args.min_fit_selected
                        and m["precision_wilson_lcb90"] is not None
                        and m["precision_wilson_lcb90"] >= args.min_fit_precision_lcb
                        and m["harmful_group_exposure_ucb90"] <= args.max_fit_harmful_group_ucb
                        and m["harmful_selected_ucb90"] <= args.max_fit_harmful_selected_ucb
                        and (
                            float(m.get("selected_macro_excess_share", 0.0)) <= args.max_macro_excess_share
                            if args.macro_constraint_mode == "opportunity_normalized"
                            else m["max_selected_macro_share"] <= args.max_selected_macro_share
                        )
                        and m["teacher_advantage_mean"] is not None
                        and m["teacher_advantage_mean"] > 0.0
                    ):
                        candidates.append(row)
    frontier.sort(key=lambda x: (x["constraint_deficit"], -x["num_positive_selected"], -x["num_selected"]))
    if not candidates:
        empty = _metrics(groups, [], float("inf"), float("inf"), args.positive_gain, args.negative_gain)
        return None, empty, [], frontier[:40]
    candidates.sort(
        key=lambda x: (
            x["num_positive_selected"],
            x["precision_wilson_lcb90"],
            -x["harmful_group_exposure_ucb90"],
            -x["harmful_selected_ucb90"],
            x["num_selected"],
            -(
                x.get("selected_macro_excess_share", 0.0)
                if args.macro_constraint_mode == "opportunity_normalized"
                else x["max_selected_macro_share"]
            ),
        ),
        reverse=True,
    )
    best = candidates[0]
    rule = {
        "opportunity_threshold": best["opportunity_threshold"],
        "harm_threshold": best["harm_threshold"],
        "score_threshold": best["score_threshold"],
        "rank_margin_threshold": best["rank_margin_threshold"],
    }
    return rule, best, candidates[:40], frontier[:40]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bucket", choices=["near", "contact"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rows-output", type=Path)
    ap.add_argument("--macro-ids", default="2,3,5,6,7")
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--fit-fold", type=int, default=0)
    ap.add_argument("--required-min-groups", type=int, default=120)
    ap.add_argument("--required-min-scenes", type=int, default=60)
    ap.add_argument("--min-nominal-deviation", type=float, default=0.002)
    ap.add_argument("--max-hard", type=float, default=1.0)
    ap.add_argument("--positive-gain", type=float, default=0.015)
    ap.add_argument("--negative-gain", type=float, default=0.010)
    ap.add_argument("--min-opportunity", type=float, default=0.05)
    ap.add_argument("--max-harm-probability", type=float, default=0.80)
    ap.add_argument("--min-score-advantage", type=float, default=0.0)
    ap.add_argument("--grid-size", type=int, default=15)
    ap.add_argument("--min-fit-selected", type=int, default=12)
    ap.add_argument("--min-fit-precision-lcb", type=float, default=0.50)
    ap.add_argument("--max-fit-harmful-group-ucb", type=float, default=0.08,
                    help="Maximum population harmful-exposure UCB")
    ap.add_argument("--max-fit-harmful-selected-ucb", type=float, default=1.0,
                    help="Maximum conditional harmful-switch UCB among selected actions")
    ap.add_argument("--min-verify-selected", type=int, default=8)
    ap.add_argument("--min-verify-precision-lcb", type=float, default=0.40)
    ap.add_argument("--max-verify-harmful-group-ucb", type=float, default=0.10,
                    help="Maximum population harmful-exposure UCB")
    ap.add_argument("--max-verify-harmful-selected-ucb", type=float, default=1.0,
                    help="Maximum conditional harmful-switch UCB among selected actions")
    ap.add_argument("--max-selected-macro-share", type=float, default=0.85)
    ap.add_argument(
        "--macro-constraint-mode", choices=["absolute", "opportunity_normalized"],
        default="absolute",
        help="Use an absolute selected-macro cap or limit excess concentration over the oracle-positive macro distribution.",
    )
    ap.add_argument("--max-macro-excess-share", type=float, default=0.10)
    ap.add_argument("--risk-source", choices=["direct_delta", "conformal_delta", "delta_distribution", "heads", "ordinal_evidence"], default="direct_delta")
    ap.add_argument("--conformal-alpha", type=float, default=0.10)
    ap.add_argument("--conformal-temperature", type=float, default=0.02)
    ap.add_argument(
        "--conformal-scope", choices=["all_pairs", "policy_top1"],
        default="policy_top1",
        help="Fit residuals on all candidates or on the frozen Stage-P policy top-1 distribution.",
    )
    ap.add_argument("--delta-std-floor", type=float, default=0.03)
    ap.add_argument("--min-rank-margin", type=float, default=0.0)
    ap.add_argument("--conditional-recovery-ranking", action="store_true", help="Compute rank margins only against the second recovery candidate; nominal admission is handled by the evidence gate.")
    ap.add_argument("--policy-first-no-fallback", action="store_true", help="Choose preference top-1 before evidence gating; abstain if it is uncertified instead of falling through to a runner-up.")
    ap.add_argument("--preference-tie-epsilon-near", type=float, default=0.025)
    ap.add_argument("--preference-tie-epsilon-contact", type=float, default=0.010)
    args = ap.parse_args()

    supported_macros = {int(x) for x in args.macro_ids.split(",") if x.strip()}
    runtime_cfg = {"selection": {"active_bucket_name": "near_contact" if args.bucket == "near" else "contact"}}
    bundle = load_model_bundle(args.checkpoint, runtime_cfg)
    if bundle is None:
        raise FileNotFoundError(args.checkpoint)
    if args.risk_source in {"heads", "ordinal_evidence"} and not bool(getattr(bundle.model, "direct_recovery_harm_head", False)):
        raise ValueError("risk-source=heads/ordinal_evidence requires model.direct_recovery_harm_head=true")

    alpha = float((bundle.cfg.get("ocmero", {}) or {}).get("alpha", 0.2))
    beta = float((bundle.cfg.get("ocmero", {}) or {}).get("beta", 0.2))
    top_m = int((bundle.cfg.get("ocmero", {}) or {}).get("top_m", 8))
    raw: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    paths = iter_sample_paths_many(args.dataset)
    for i, path in enumerate(paths, 1):
        split = str(scalar_metadata_for_path(path, "split_id", ""))
        if split and split not in {"calibration", "val"}:
            continue
        d = load_npz(path)
        row = {
            "data": d,
            "scene": str(_scalar(d, "scene_id", path.stem)),
            "time": int(_scalar(d, "time_index", 0)),
            "candidate": int(_scalar(d, "candidate_index", 0)),
            "macro": int(_scalar(d, "prefix_macro_type_id", _scalar(d, "prefix_macro_id", -1))),
            "nominal": bool(float(_scalar(d, "is_nominal", 0)) > 0.5),
            "teacher": _teacher_pcd(d, alpha, beta, top_m),
            "hard": float(_scalar(d, "hard_violation", 0.0)),
            "feasible": bool(int(_scalar(d, "feasible", 1))),
        }
        raw[(row["scene"], row["time"])].append(row)
        if i == 1 or i % 1000 == 0 or i == len(paths):
            print({"event": "v48_calibration_load_progress", "bucket": args.bucket, "seen": i, "total": len(paths)}, flush=True)

    # v48.3: score complete candidate sets jointly so the nominal-anchored set
    # context used during training is also present during calibration/deployment.
    for gi, items in enumerate(raw.values(), 1):
        items.sort(key=lambda x: x["candidate"])
        preds = predict_samples([x["data"] for x in items], bundle, runtime_cfg, shared_scene_features=True)
        for row, pred in zip(items, preds):
            if pred.direct_recovery_value is None or pred.direct_recovery_std is None:
                raise ValueError("checkpoint does not expose value mean/std outputs")
            row["pred"] = float(pred.direct_recovery_value)
            row["pred_std"] = max(float(args.delta_std_floor), float(pred.direct_recovery_std))
            row["rank"] = float(pred.direct_recovery_rank if pred.direct_recovery_rank is not None else pred.direct_recovery_value)
            row["delta"] = None if pred.direct_recovery_delta is None else float(pred.direct_recovery_delta)
            row["delta_std"] = None if pred.direct_recovery_delta_std is None else max(float(args.delta_std_floor), float(pred.direct_recovery_delta_std))
            row["opp_logit"] = None if pred.direct_recovery_opportunity_logit is None else float(pred.direct_recovery_opportunity_logit)
            row["harm_logit"] = None if pred.direct_recovery_harm_logit is None else float(pred.direct_recovery_harm_logit)
            if args.risk_source in {"heads", "ordinal_evidence"} and (row["opp_logit"] is None or row["harm_logit"] is None):
                raise ValueError("risk-source=heads/ordinal_evidence requires opportunity/harm outputs")
        if gi == 1 or gi % 200 == 0 or gi == len(raw):
            print({"event": "v48_calibration_group_score_progress", "bucket": args.bucket, "groups": gi, "total_groups": len(raw)}, flush=True)

    groups: list[dict[str, Any]] = []
    skipped = Counter()
    for (scene, time_index), items in raw.items():
        items.sort(key=lambda x: x["candidate"])
        for x, dev in zip(items, _group_deviation(items)):
            x["deviation"] = dev
        noms = [x for x in items if x["nominal"]]
        if not noms:
            skipped["no_nominal"] += 1
            continue
        nom = noms[0]
        recs = [x for x in items if not x["nominal"] and x["macro"] in supported_macros and x["feasible"] and x["hard"] <= args.max_hard and x["deviation"] >= args.min_nominal_deviation]
        if not recs:
            skipped["no_eligible"] += 1
            continue
        pairs = []
        for r in recs:
            if args.risk_source in {"direct_delta", "conformal_delta"}:
                if r["delta"] is None or r["delta_std"] is None:
                    raise ValueError("risk-source=direct_delta requires direct delta mean/std outputs")
                pred_adv = float(r["delta"])
            elif args.risk_source == "ordinal_evidence":
                opportunity = _sigmoid(float(r["opp_logit"]) - float(nom["opp_logit"]))
                harm = _sigmoid(float(r["harm_logit"]) - float(nom["harm_logit"]))
                pred_adv = opportunity - harm
            else:
                pred_adv = r["pred"] - nom["pred"]
            rank_adv = r["rank"] - nom["rank"]
            delta_std = (
                float(r["delta_std"])
                if args.risk_source in {"direct_delta", "conformal_delta"}
                else max(float(args.delta_std_floor), math.sqrt(r["pred_std"] ** 2 + nom["pred_std"] ** 2))
            )
            if args.risk_source in {"direct_delta", "conformal_delta", "delta_distribution"}:
                opportunity = _normal_cdf((pred_adv - args.positive_gain) / delta_std)
                harm = _normal_cdf((-args.negative_gain - pred_adv) / delta_std)
            elif args.risk_source != "ordinal_evidence":
                opportunity = _sigmoid(float(r["opp_logit"]) - float(nom["opp_logit"]))
                harm = _sigmoid(float(r["harm_logit"]) - float(nom["harm_logit"]))
            head_harm = None
            if r["harm_logit"] is not None and nom["harm_logit"] is not None:
                head_harm = _sigmoid(float(r["harm_logit"]) - float(nom["harm_logit"]))
            pairs.append({
                "candidate": r["candidate"], "macro": r["macro"], "deviation": r["deviation"],
                "pred_adv": pred_adv, "rank_adv": rank_adv, "delta_std": delta_std,
                "opportunity": opportunity, "harm": harm, "head_harm": head_harm,
                "teacher_adv": r["teacher"] - nom["teacher"],
            })
        groups.append({
            "scene": scene, "time": time_index, "fold": _fold(scene, max(2, args.folds)),
            "pairs": pairs, "oracle_best_teacher_adv": max(r["teacher_adv"] for r in pairs),
        })

    fit = [g for g in groups if g["fold"] == args.fit_fold]
    verify = [g for g in groups if g["fold"] != args.fit_fold]
    conformal = None
    if args.risk_source == "conformal_delta":
        if args.conformal_scope == "policy_top1":
            fit_pairs = _policy_top1_pairs(fit)
        else:
            fit_pairs = [r for g in fit for r in g["pairs"]]
        q_over = _finite_sample_upper_quantile(
            [r["pred_adv"] - r["teacher_adv"] for r in fit_pairs], args.conformal_alpha
        )
        q_under = _finite_sample_upper_quantile(
            [r["teacher_adv"] - r["pred_adv"] for r in fit_pairs], args.conformal_alpha
        )
        temp = max(float(args.conformal_temperature), 1.0e-4)
        for g in groups:
            for r in g["pairs"]:
                r["gain_lcb"] = float(r["pred_adv"] - q_over)
                r["gain_ucb"] = float(r["pred_adv"] + q_under)
                # Admission is based on a held-out lower confidence bound.  The
                # harm score uses the same bound conservatively, so a candidate
                # cannot look simultaneously high-opportunity and low-risk only
                # because its learned variance collapsed.
                r["opportunity"] = _sigmoid((r["gain_lcb"] - args.positive_gain) / temp)
                r["harm"] = _sigmoid((-args.negative_gain - r["gain_lcb"]) / temp)
        conformal = {
            "alpha": float(args.conformal_alpha),
            "temperature": float(temp),
            "fit_pair_count": len(fit_pairs),
            "scope": str(args.conformal_scope),
            "overprediction_quantile": float(q_over),
            "underprediction_quantile": float(q_under),
        }
    rule, fit_metrics, candidates, near_miss = _fit(fit, args, supported_macros)
    if rule is None:
        verify_top1: list[dict[str, Any]] = []
        all_top1: list[dict[str, Any]] = []
        score_thr = float("inf")
    else:
        verify_top1 = _top1(verify, rule["opportunity_threshold"], rule["harm_threshold"], supported_macros, conditional_rank_margin=args.conditional_recovery_ranking, policy_first_no_fallback=args.policy_first_no_fallback)
        all_top1 = _top1(groups, rule["opportunity_threshold"], rule["harm_threshold"], supported_macros, conditional_rank_margin=args.conditional_recovery_ranking, policy_first_no_fallback=args.policy_first_no_fallback)
        score_thr = rule["score_threshold"]
    rank_margin_thr = float("inf") if rule is None else rule["rank_margin_threshold"]
    verify_metrics = _metrics(verify, verify_top1, score_thr, rank_margin_thr, args.positive_gain, args.negative_gain)
    all_metrics = _metrics(groups, all_top1, score_thr, rank_margin_thr, args.positive_gain, args.negative_gain)
    near_miss_verify_frontier: list[dict[str, Any]] = []
    for fit_row in near_miss[:20]:
        vtop = _top1(
            verify, float(fit_row["opportunity_threshold"]),
            float(fit_row["harm_threshold"]), supported_macros,
            conditional_rank_margin=args.conditional_recovery_ranking,
            policy_first_no_fallback=args.policy_first_no_fallback,
        )
        vm = _metrics(
            verify, vtop, float(fit_row["score_threshold"]),
            float(fit_row["rank_margin_threshold"]),
            args.positive_gain, args.negative_gain,
        )
        near_miss_verify_frontier.append({
            "fit_constraint_deficit": float(fit_row.get("constraint_deficit", 0.0)),
            "rule": {k: float(fit_row[k]) for k in (
                "opportunity_threshold", "harm_threshold",
                "score_threshold", "rank_margin_threshold",
            )},
            "fit": {k: fit_row.get(k) for k in (
                "num_selected", "precision", "precision_wilson_lcb90",
                "harmful_selected_rate", "harmful_selected_ucb90",
                "positive_recall", "teacher_advantage_mean",
                "max_selected_macro_share",
            )},
            "verify": vm,
        })

    pairs = [r for g in groups for r in g["pairs"]]
    pred = [r["pred_adv"] for r in pairs]
    rank = [r["rank_adv"] for r in pairs]
    teacher = [r["teacher_adv"] for r in pairs]
    corr = float(np.corrcoef(pred, teacher)[0, 1]) if len(pred) > 1 and np.std(pred) > 1e-12 and np.std(teacher) > 1e-12 else None
    rank_corr = float(np.corrcoef(rank, teacher)[0, 1]) if len(rank) > 1 and np.std(rank) > 1e-12 and np.std(teacher) > 1e-12 else None
    unconstrained_top1: list[dict[str, Any]] = []
    top1_correct_labels: list[bool] = []
    top1_strict_labels: list[bool] = []
    top1_rank_margins: list[float] = []
    tie_epsilon = args.preference_tie_epsilon_near if args.bucket == "near" else args.preference_tie_epsilon_contact
    for g in groups:
        ordered = sorted(g["pairs"], key=lambda r: (-r["rank_adv"], r["candidate"]))
        chosen = ordered[0]
        alternative_scores = [float(r["rank_adv"]) for r in ordered[1:]]
        if not args.conditional_recovery_ranking:
            alternative_scores.append(0.0)
        second_score = max(alternative_scores) if alternative_scores else float(chosen["rank_adv"] - 1.0)
        chosen = dict(chosen)
        chosen["rank_margin"] = float(chosen["rank_adv"] - second_score)
        chosen.update(
            scene=g["scene"], time=g["time"], fold=g["fold"],
            oracle_best_teacher_adv=g["oracle_best_teacher_adv"],
        )
        unconstrained_top1.append(chosen)
        if g["oracle_best_teacher_adv"] >= args.positive_gain:
            oracle = max(g["pairs"], key=lambda r: r["teacher_adv"])
            top1_strict_labels.append(chosen["candidate"] == oracle["candidate"])
            top1_correct_labels.append(chosen["teacher_adv"] >= oracle["teacher_adv"] - tie_epsilon)
            top1_rank_margins.append(chosen["rank_margin"])
    top1_pred = [r["rank_adv"] for r in unconstrained_top1]
    top1_teacher = [r["teacher_adv"] for r in unconstrained_top1]
    top1_corr = float(np.corrcoef(top1_pred, top1_teacher)[0, 1]) if len(top1_pred) > 1 and np.std(top1_pred) > 1e-12 and np.std(top1_teacher) > 1e-12 else None
    policy_top1_positive_auc = _auc(
        [r["teacher_adv"] >= args.positive_gain for r in unconstrained_top1],
        [r["pred_adv"] for r in unconstrained_top1],
    )
    policy_top1_harm_auc = _auc(
        [r["teacher_adv"] <= -args.negative_gain for r in unconstrained_top1],
        [r["harm"] for r in unconstrained_top1],
    )
    policy_top1_gain_mae = (
        float(np.mean([abs(float(r["pred_adv"]) - float(r["teacher_adv"])) for r in unconstrained_top1]))
        if unconstrained_top1 else None
    )
    recovery_switches = [r for r in unconstrained_top1 if float(r["rank_adv"]) > 0.0]
    nonpositive_groups = [r for r in unconstrained_top1 if float(r["oracle_best_teacher_adv"]) < args.positive_gain]
    positive_policy_groups = [r for r in unconstrained_top1 if float(r["oracle_best_teacher_adv"]) >= args.positive_gain]
    nonpositive_false_switches = [r for r in nonpositive_groups if float(r["rank_adv"]) > 0.0]
    harmful_ranked_switches = [
        r for r in recovery_switches if float(r["teacher_adv"]) <= -args.negative_gain
    ]
    positive_group_activations = [r for r in positive_policy_groups if float(r["rank_adv"]) > 0.0]
    positive_groups = [g for g in groups if g["oracle_best_teacher_adv"] >= args.positive_gain]
    top1_hits = []
    top1_strict_hits = []
    top1_regrets = []
    for g in positive_groups:
        chosen = max(g["pairs"], key=lambda r: r["rank_adv"])
        oracle = max(g["pairs"], key=lambda r: r["teacher_adv"])
        top1_strict_hits.append(chosen["candidate"] == oracle["candidate"])
        top1_hits.append(chosen["teacher_adv"] >= oracle["teacher_adv"] - tie_epsilon)
        top1_regrets.append(max(0.0, oracle["teacher_adv"] - chosen["teacher_adv"] - tie_epsilon))

    scenes = {g["scene"] for g in groups}
    fit_scenes = {g["scene"] for g in fit}
    verify_scenes = {g["scene"] for g in verify}
    warnings: list[str] = []
    if len(groups) < args.required_min_groups:
        warnings.append("insufficient calibration groups")
    if len(scenes) < args.required_min_scenes:
        warnings.append("insufficient independent calibration scenes")
    if fit_scenes & verify_scenes:
        warnings.append("fit/verify scene leakage")
    if rule is None:
        warnings.append("no joint opportunity-harm-score rule satisfied fit constraints")
    if verify_metrics["num_selected"] < args.min_verify_selected:
        warnings.append("held-out selections below requirement")
    if verify_metrics["precision_wilson_lcb90"] is None or verify_metrics["precision_wilson_lcb90"] < args.min_verify_precision_lcb:
        warnings.append("held-out precision LCB below requirement")
    if verify_metrics["harmful_group_exposure_ucb90"] > args.max_verify_harmful_group_ucb:
        warnings.append("held-out harmful exposure UCB above budget")
    if verify_metrics["harmful_selected_ucb90"] > args.max_verify_harmful_selected_ucb:
        warnings.append("held-out conditional harmful-switch UCB above budget")
    verify_macro_bad = (
        float(verify_metrics.get("selected_macro_excess_share", 0.0)) > args.max_macro_excess_share
        if args.macro_constraint_mode == "opportunity_normalized"
        else verify_metrics["max_selected_macro_share"] > args.max_selected_macro_share
    )
    if verify_macro_bad:
        warnings.append("held-out selections exceed the macro concentration budget")

    valid = not warnings
    result = {
        "method": "v48_scene_disjoint_joint_policy_risk_certificate",
        "bucket": args.bucket,
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "valid_for_deployment": valid,
        "selection_rule": ("physical -> preference top1 -> evidence -> value challenge" if args.policy_first_no_fallback else "physical -> gain-distribution opportunity/harm -> preference top1 -> value challenge"),
        "risk_source": args.risk_source,
        "conformal": conformal,
        "rule": rule,
        "selector_overrides": ({} if rule is None else {
            "direct_value_certificate": True,
            "direct_value_score_mode": True,
            "direct_value_uncertainty_mode": ("conformal_additive" if conformal is not None else "risk_controlled"),
            "direct_value_additive_q": (0.0 if conformal is None else conformal["overprediction_quantile"]),
            "direct_value_top1_only": True,
            "direct_value_policy_first_no_fallback": bool(args.policy_first_no_fallback),
            "direct_value_risk_controlled_admission": True,
            "direct_value_risk_source": args.risk_source,
            "direct_value_positive_gain": args.positive_gain,
            "direct_value_negative_gain": args.negative_gain,
            "direct_value_opportunity_threshold": rule["opportunity_threshold"],
            "direct_value_harm_threshold": rule["harm_threshold"],
            "direct_value_min_advantage_lcb": rule["score_threshold"],
            "direct_value_min_rank_margin": rule["rank_margin_threshold"],
            "direct_value_conditional_rank_margin": bool(args.conditional_recovery_ranking),
            "direct_value_conformal_overprediction_quantile": (None if conformal is None else conformal["overprediction_quantile"]),
            "direct_value_conformal_underprediction_quantile": (None if conformal is None else conformal["underprediction_quantile"]),
            "direct_value_conformal_temperature": (None if conformal is None else conformal["temperature"]),
        }),
        "num_groups": len(groups), "num_scenes": len(scenes), "fit_groups": len(fit), "verify_groups": len(verify),
        "fit_scenes": len(fit_scenes), "verify_scenes": len(verify_scenes), "scene_overlap": len(fit_scenes & verify_scenes),
        "fit": fit_metrics, "verify": verify_metrics, "all": all_metrics,
        "candidate_positive_auc": _auc([r["teacher_adv"] >= args.positive_gain for r in pairs], [r["pred_adv"] for r in pairs]),
        "candidate_harm_auc": _auc([r["teacher_adv"] <= -args.negative_gain for r in pairs], [r["harm"] for r in pairs]),
        "candidate_risk_harm_auc": _auc([r["teacher_adv"] <= -args.negative_gain for r in pairs], [r["harm"] for r in pairs]),
        "candidate_head_harm_auc": _auc(
            [r["teacher_adv"] <= -args.negative_gain for r in pairs if r.get("head_harm") is not None],
            [float(r["head_harm"]) for r in pairs if r.get("head_harm") is not None],
        ),
        "candidate_pred_teacher_correlation": corr,
        "candidate_rank_teacher_correlation": rank_corr,
        "unconstrained_group_top1_correlation": top1_corr,
        "policy_top1_positive_auc": policy_top1_positive_auc,
        "policy_top1_harm_auc": policy_top1_harm_auc,
        "policy_top1_gain_mae": policy_top1_gain_mae,
        "unconstrained_recovery_switch_rate": (len(recovery_switches) / len(unconstrained_top1) if unconstrained_top1 else None),
        "nonpositive_group_false_switch_rate": (len(nonpositive_false_switches) / len(nonpositive_groups) if nonpositive_groups else None),
        "harmful_ranked_switch_rate": (len(harmful_ranked_switches) / len(unconstrained_top1) if unconstrained_top1 else None),
        "positive_group_recovery_activation_rate": (len(positive_group_activations) / len(positive_policy_groups) if positive_policy_groups else None),
        "positive_group_top1_accuracy": (float(np.mean(top1_hits)) if top1_hits else None),
        "positive_group_strict_top1_accuracy": (float(np.mean(top1_strict_hits)) if top1_strict_hits else None),
        "positive_group_top1_regret_mean": (float(np.mean(top1_regrets)) if top1_regrets else None),
        "top1_correctness_rank_margin_auc": _auc(top1_correct_labels, top1_rank_margins),
        "strict_top1_correctness_rank_margin_auc": _auc(top1_strict_labels, top1_rank_margins),
        "preference_tie_epsilon": tie_epsilon,
        "top_fit_candidates": candidates,
        "near_miss_frontier": near_miss,
        "near_miss_verify_frontier": near_miss_verify_frontier,
        "skipped": dict(skipped),
        "warnings": warnings,
        "constraints": vars(args) | {"output": str(args.output), "rows_output": str(args.rows_output) if args.rows_output else None},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.rows_output:
        args.rows_output.parent.mkdir(parents=True, exist_ok=True)
        rows_to_write = all_top1 if rule is not None else unconstrained_top1
        with args.rows_output.open("w", encoding="utf-8") as f:
            for row in rows_to_write:
                row = dict(row)
                row["selected"] = bool(
                    rule is not None
                    and row["pred_adv"] >= score_thr
                    and row.get("rank_margin", 0.0) >= rank_margin_thr
                )
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if valid else 3


if __name__ == "__main__":
    raise SystemExit(main())
