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
    ap.add_argument(
        "--data-protocol",
        choices=["v15", "v9_reuse", "v16_8_root_conditioned_overlay", "v16_8_2_fresh", "v16_8_3_fresh", "v16_8_4_fresh"],
        default="v15",
    )
    ap.add_argument("--cache-reuse-report", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    model_cfg = _load(args.model_config)
    label_cfg = _load(args.label_config)
    train_cfg = _load(args.train_config)
    eval_cfg = _load(args.eval_config)
    merged_model = dict(model_cfg.get("model", {}))
    merged_model.update(train_cfg.get("model", {}))
    planning = dict(label_cfg.get("planning", {}))
    planning.update(eval_cfg.get("planning", {}))
    eval_block = eval_cfg.get("eval", {})
    natural = label_cfg.get("natural", {})
    weights = train_cfg.get("loss_weights", {})

    future_source = str(planning.get("online_other_future_source", "constant_velocity")).lower()
    causal_sources = {"constant_velocity", "current_state", "learned", "learned_response"}
    engineering_checks: dict[str, bool] = {
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
        "typed_source_ce_not_used_as_learning_claim": abs(float(weights.get("branch_source_ce", 0.0))) <= 1e-12,
        # Backward-compatible for generic protocol audits; v16 configs set these
        # fields explicitly, so an accidental regression still fails in real runs.
        "natural_decoder_is_dynamics_consistent": (
            not str(merged_model.get("natural_decoder_type", "")).strip()
            or str(merged_model.get("natural_decoder_type", "")).lower() in {
                "typed_causal_dynamics", "cnob_dynamics", "cnob"
            }
        ),
        "certificate_fallback_is_explicitly_disabled": (
            not any(k in planning for k in (
                "candidate_cert_allow_hybrid_fallback",
                "candidate_cert_hybrid_fallback_mix",
                "candidate_cert_flat_fallback_mix",
            ))
            or (
                not bool(planning.get("candidate_cert_allow_hybrid_fallback", False))
                and abs(float(planning.get("candidate_cert_hybrid_fallback_mix", 0.0))) <= 1e-12
                and abs(float(planning.get("candidate_cert_flat_fallback_mix", 0.0))) <= 1e-12
            )
        ),
    }

    label_protocol_checks: dict[str, bool] = {
        "natural_map_filter_enabled": bool(natural.get("map_filter_enabled", False)),
        "observational_decontamination_enabled": bool(natural.get("obs_decontamination_enabled", False)),
    }
    reuse_report = None
    overlay_checks: dict[str, bool] = {}
    if args.cache_reuse_report:
        reuse_report = json.loads(Path(args.cache_reuse_report).read_text(encoding="utf-8"))
        materialized = bool(_nested(reuse_report, "decisions", "reuse_as_true_v15_causal_label_dataset", "pass", default=False))
        label_protocol_checks["v15_label_tensors_materialized"] = materialized
        for split in ("train", "val"):
            summary = _nested(reuse_report, split, "overlay_summary", default={}) or {}
            overlay_checks[f"{split}_overlay_complete"] = bool(summary.get("complete", False))
            overlay_checks[f"{split}_overlay_error_free"] = int(summary.get("error_count", 1)) == 0
            missing = _nested(reuse_report, split, "missing_required_key_counts", default={}) or {}
            overlay_checks[f"{split}_required_transport_keys_present"] = all(
                int(v) == 0 for v in missing.values()
            )
        overlay_checks["train_val_filename_disjoint"] = int(
            reuse_report.get("cross_split_filename_overlap", 1)
        ) == 0
        overlay_checks["v9_base_reuse_gate_pass"] = bool(
            _nested(
                reuse_report, "decisions",
                "reuse_for_v14_or_v15_model_with_v9_labels", "pass", default=False,
            )
        )

    alignment = None
    if args.cache_alignment_report:
        alignment = json.loads(Path(args.cache_alignment_report).read_text(encoding="utf-8"))
        finite_rate = _nested(alignment, "rates", "waymax_logdiv_finite", default=None)
        if finite_rate is not None:
            finite_rate = float(finite_rate)
            logdiv_weight = float(weights.get("outcome_logdiv", 0.0))
            engineering_checks["missing_logdiv_not_supervised"] = finite_rate > 0.0 or abs(logdiv_weight) <= 1e-12
        engineering_checks["critical_mapping_complete"] = float(_nested(alignment, "rates", "critical_unmapped", default=1.0)) <= 0.0
        engineering_checks["response_root_indices_in_range"] = float(_nested(alignment, "rates", "response_root_out_of_range", default=1.0)) <= 0.0

    engineering_failed = [name for name, ok in engineering_checks.items() if not ok]
    label_failed = [name for name, ok in label_protocol_checks.items() if not ok]
    overlay_failed = [name for name, ok in overlay_checks.items() if not ok]
    require_v15_labels = args.data_protocol in {"v15", "v16_8_2_fresh", "v16_8_3_fresh", "v16_8_4_fresh"}
    require_overlay = args.data_protocol == "v16_8_root_conditioned_overlay"
    if require_overlay and not args.cache_reuse_report:
        overlay_failed.append("cache_reuse_report_required_for_v16_8_overlay")
    failed = engineering_failed
    if require_v15_labels:
        failed += label_failed
    if require_overlay:
        failed += overlay_failed
    report = {
        "pass": not failed,
        "checks": {**engineering_checks, **label_protocol_checks, **overlay_checks},
        "engineering_pass": not engineering_failed,
        "full_v15_label_protocol_pass": not label_failed,
        "mechanism_overlay_protocol_pass": bool(overlay_checks) and not overlay_failed,
        "fresh_v16_8_2_label_protocol_pass": (not label_failed) if args.data_protocol == "v16_8_2_fresh" else False,
        "fresh_v16_8_3_label_protocol_pass": (not label_failed) if args.data_protocol == "v16_8_3_fresh" else False,
        "fresh_v16_8_4_label_protocol_pass": (not label_failed) if args.data_protocol == "v16_8_4_fresh" else False,
        "data_protocol": args.data_protocol,
        "failed_checks": failed,
        "engineering_failed_checks": engineering_failed,
        "label_protocol_failed_checks": label_failed,
        "overlay_protocol_failed_checks": overlay_failed,
        "protocol": {
            "online_other_future_source": future_source,
            "actual_non_ego_policy": eval_block.get("actual_non_ego_policy"),
            "reactive_mixture_implemented": bool(eval_block.get("reactive_mixture_implemented", False)),
            "closed_loop_collision_source": eval_block.get("reported_collision_source"),
            "label_space_safety_metric": eval_block.get("label_space_candidate_safety_metric"),
        },
        "interpretation": (
            "Passing this audit removes known causality, coordinate-identity, metric-naming, and selected "
            "overlay-integrity violations. The v16_8_root_conditioned_overlay protocol is valid for isolated "
            "RCOT/certificate development on a v9 base, but it is not a fresh v15/v16 causal-label dataset. "
            "The v16_8_2_fresh, v16_8_3_fresh, and v16_8_4_fresh protocols additionally require materialized map-filtered and observationally "
            "decontaminated labels; v16_8_3_fresh records RMR-BCTE provenance, while v16_8_4_fresh requires boundary-consistent "
            "BCS-RMR-BCTE proposal provenance and a matching build fingerprint. "
            "No audit result establishes algorithmic SOTA or supplies reactive non-ego ground truth."
        ),
        "cache_alignment_report": args.cache_alignment_report,
        "cache_reuse_report": args.cache_reuse_report,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
