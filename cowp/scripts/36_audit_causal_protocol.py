from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml


def _load(path: str) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _nested(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def main() -> None:
    ap = argparse.ArgumentParser(description="Static and report-backed audit for a causal COWP evaluation protocol.")
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--label-config", required=True)
    ap.add_argument("--train-config", required=True)
    ap.add_argument("--eval-config", required=True)
    ap.add_argument("--cache-alignment-report", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    model_cfg = _load(args.model_config)
    label_cfg = _load(args.label_config)
    train_cfg = _load(args.train_config)
    eval_cfg = _load(args.eval_config)
    merged_model = dict(model_cfg.get("model", {}))
    merged_model.update(train_cfg.get("model", {}))
    planning = eval_cfg.get("planning", {})
    eval_block = eval_cfg.get("eval", {})
    natural = label_cfg.get("natural", {})
    weights = train_cfg.get("loss_weights", {})

    future_source = str(planning.get("online_other_future_source", "constant_velocity")).lower()
    causal_sources = {"constant_velocity", "current_state", "learned", "learned_response"}
    checks: dict[str, bool] = {
        "future_label_encoder_fallback_disabled": not bool(merged_model.get("allow_label_only_state_fallback", False)),
        "explicit_sdc_required": bool(merged_model.get("require_explicit_sdc_index", False)),
        "online_other_future_is_causal": future_source in causal_sources,
        "log_trajectory_fallback_disabled": not bool(planning.get("causal_log_trajectory_fallback", False)),
        "collision_rate_from_waymax_only": str(eval_block.get("reported_collision_source", "")).lower() == "waymax_standard_metrics_only",
        "label_metric_named_as_proxy": str(eval_block.get("label_space_candidate_safety_metric", "")) == "OfflineConventionalUnsafeRate",
        "reactive_protocol_claim_is_honest": (
            bool(eval_block.get("reactive_mixture_implemented", False))
            or str(eval_block.get("actual_non_ego_policy", "")).lower() == "logged_replay"
        ),
        "natural_map_filter_enabled": bool(natural.get("map_filter_enabled", False)),
        "observational_decontamination_enabled": bool(natural.get("obs_decontamination_enabled", False)),
        "typed_source_ce_not_used_as_learning_claim": abs(float(weights.get("branch_source_ce", 0.0))) <= 1e-12,
    }

    alignment = None
    if args.cache_alignment_report:
        alignment = json.loads(Path(args.cache_alignment_report).read_text(encoding="utf-8"))
        finite_rate = _nested(alignment, "rates", "waymax_logdiv_finite", default=None)
        if finite_rate is not None:
            finite_rate = float(finite_rate)
            logdiv_weight = float(weights.get("outcome_logdiv", 0.0))
            checks["missing_logdiv_not_supervised"] = finite_rate > 0.0 or abs(logdiv_weight) <= 1e-12
        checks["critical_mapping_complete"] = float(_nested(alignment, "rates", "critical_unmapped", default=1.0)) <= 0.0
        checks["response_root_indices_in_range"] = float(_nested(alignment, "rates", "response_root_out_of_range", default=1.0)) <= 0.0

    failed = [name for name, ok in checks.items() if not ok]
    report = {
        "pass": not failed,
        "checks": checks,
        "failed_checks": failed,
        "protocol": {
            "online_other_future_source": future_source,
            "actual_non_ego_policy": eval_block.get("actual_non_ego_policy"),
            "reactive_mixture_implemented": bool(eval_block.get("reactive_mixture_implemented", False)),
            "closed_loop_collision_source": eval_block.get("reported_collision_source"),
            "label_space_safety_metric": eval_block.get("label_space_candidate_safety_metric"),
        },
        "interpretation": (
            "Passing this audit removes known causality, coordinate-identity, and metric-naming violations. "
            "It does not establish algorithmic SOTA or supply reactive non-ego ground truth."
        ),
        "cache_alignment_report": args.cache_alignment_report,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
