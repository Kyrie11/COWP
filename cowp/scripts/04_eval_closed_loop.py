from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from cowp.core.config import load_config
from cowp.waymax_eval.rollout import (
    import_policy_fn,
    learned_offline_candidate_eval,
    learned_offline_candidate_eval_budget_sweep,
    learned_offline_candidate_eval_methods,
    learned_offline_candidate_eval_sweep,
    offline_candidate_eval,
    waymax_closed_loop_rollout,
)
from cowp.waymax_eval.policy_wrapper import make_cowp_policy
from cowp.waymax_eval.metrics_cowp import (
    physical_failure_attribution_summary,
    policy_diagnostic_episode_summary,
    policy_diagnostic_scenario_rows,
    policy_diagnostic_summary,
)
from cowp.waymax_eval.metrics_standard import aggregate_waymax_standard_metrics


def _configure_waymax_runtime(args) -> None:
    """Configure JAX/Waymax before any Waymax/JAX import happens.

    PyTorch and Waymax/JAX share the same CUDA devices by default.  Disabling
    JAX preallocation and optionally restricting JAX-visible devices prevents JAX
    from reserving most of the GPU memory that the PyTorch policy also needs.
    """
    if getattr(args, "waymax_device", "auto") == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    if getattr(args, "jax_visible_devices", None):
        os.environ["JAX_VISIBLE_DEVICES"] = str(args.jax_visible_devices)
    if getattr(args, "jax_preallocate", None) is not None:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(args.jax_preallocate).lower()
    if getattr(args, "jax_mem_fraction", None) is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(float(args.jax_mem_fraction))


def _parse_metric_names(value: str | None):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"all", "default", "none"}:
        return None if text.lower() != "none" else []
    return {x.strip() for x in text.split(",") if x.strip()}


def _load_scenario_ids_file(path: str | None) -> list[str] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"scenario id file not found: {p}")
    ids = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"scenario id file is empty: {p}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"scenario id file contains duplicate ids: {p}")
    return ids


def _json_safe(obj):
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate COWP labels/planner offline, learned offline, or real Waymax closed-loop rollout.")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--label-config", default="configs/label.yaml")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--waymax-split", choices=["training", "validation", "testing"], default="validation", help="WOMD split for --mode waymax. Default validation avoids evaluating online rollouts on training.")
    ap.add_argument("--tfexample-glob", default=None, help="Optional Waymax/tf.Example path override for --mode waymax.")
    ap.add_argument("--tfexample-index-jsonl", default=None, help="Optional scenario-id -> TFExample shard index. Exact-ID Waymax uses it to avoid scanning irrelevant record shards.")
    ap.add_argument("--scenario-ids-file", default=None, help="Exact scenario-ID allowlist for --mode waymax. Every requested ID must be resolved; missing IDs are a hard error.")
    ap.add_argument("--num-shards", type=int, default=1, help="Shard online Waymax rollouts across multiple parallel processes.")
    ap.add_argument("--shard-index", type=int, default=0, help="Shard id in [0, num_shards) for --mode waymax.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for learned_offline cache loading.")
    ap.add_argument("--prefetch-factor", type=int, default=2, help="Batches prefetched per learned_offline DataLoader worker.")
    ap.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None, help="Pin learned_offline host tensors. Default: enabled only for CUDA evaluation.")
    ap.add_argument("--learned-subset-modulo", type=int, default=1, help="Deterministic learned_offline partition modulus; use disjoint remainders for calibration and held-out evaluation.")
    ap.add_argument("--learned-subset-remainder", type=int, default=0, help="Deterministic learned_offline partition remainder in [0, modulo).")
    ap.add_argument("--learned-max-scenes", type=int, default=None, help="Optional cap after deterministic learned_offline partitioning.")
    ap.add_argument("--witness-threshold", type=float, default=0.5, help="Pair-level coercion witness threshold; independent of the candidate BCOT risk budget.")
    ap.add_argument("--bcot-risk-budget", type=float, default=None, help="Candidate-level BCOT risk budget. Defaults to planning.candidate_transport_budget.")
    ap.add_argument("--bcot-risk-budget-sweep", default=None, help="Comma-separated candidate BCOT budgets. Keeps the pair-witness threshold fixed and reuses one model/cache pass.")
    ap.add_argument("--witness-threshold-sweep", default=None, help="Comma-separated thresholds for learned_offline diagnostic sweep, e.g. 0.1,0.2,0.3,0.5,0.7.")
    ap.add_argument("--ncf-gate-mode", choices=["hard", "priority", "soft", "none"], default="priority", help="Candidate acceptance gate. priority is the default COWP gate; hard/universal vetoes any predicted witness; soft uses witness only as a score penalty; none disables witness gating.")
    ap.add_argument("--priority-hard-threshold", type=float, default=0.55, help="Online priority proxy threshold for hard P-NCF rejection.")
    ap.add_argument("--secondary-witness-threshold", type=float, default=0.85, help="Severe witness threshold for priority/soft gate diagnostics.")
    ap.add_argument("--secondary-opr-alpha", type=float, default=0.10, help="Severe low-option-preservation threshold used with secondary witness threshold.")
    ap.add_argument("--soft-ncf-penalty", type=float, default=1.5, help="Score penalty weight for non-hard coercion evidence in priority/soft gates.")
    ap.add_argument("--method", default="cowp", help="Evaluation method/internal baseline: cowp, cowp_cert_utility, cowp_fallback_outcome, cowp_recursive_viability, cowp_rvr_pareto_guard, cowp_successor_option_viability, cowp_bihorizon_option_viability, cowp_successor_restore_only, cowp_trihorizon_option_persistence, cowp_sov_recovery_commitment, cowp_sov_dominance_hysteresis, cowp_recovery_option_spectrum_hysteresis, cowp_transition_guarded_rosh, cowp_executable_option_spectrum_hysteresis, cowp_waymax_kinematic_guarded_rosh, cowp_control_projected_option_spectrum_hysteresis, cowp_control_projected_recovery_frontier, cowp_recourse_returnability_bridge, cowp_shift_closed_control_reachable_tube, universal_ncf, soft_burden_cost_only, idm_lattice, conventional_safety, planner_score_only, etc.")
    ap.add_argument("--methods", default=None, help="Comma-separated learned_offline methods evaluated in one shared checkpoint/cache pass.")
    ap.add_argument("--offline-fallback", choices=["conservative", "stop_like"], default="stop_like", help="For learned_offline, what to do when no candidate passes the gate. conservative marks fallback (-1); stop_like selects neutral/yield/stop candidates when present.")
    ap.add_argument("--adaptive-frontier-margin", type=float, default=0.20, help="Scene-adaptive P-NCF frontier margin used when absolute witness calibration rejects every candidate.")
    ap.add_argument("--outcome-risk-penalty", type=float, default=0.0, help="Penalty weight for learned Waymax outcome risk. Keep 0 for checkpoints trained without outcome labels.")
    ap.add_argument("--outcome-risk-threshold", type=float, default=1.10, help="Maximum learned outcome risk allowed in the hard feasibility layer when outcome-risk-penalty > 0.")
    ap.add_argument("--mode", choices=["offline", "learned_offline", "waymax"], default="offline")
    ap.add_argument("--policy-fn", default=None, help="For --mode waymax: optional Python callable spec 'module:function' returning Waymax actions. If omitted with --checkpoint, a COWP checkpoint policy is used.")
    ap.add_argument("--waymax-action-mode", choices=["delta_xy_yaw", "absolute_xy_yaw"], default="absolute_xy_yaw")
    ap.add_argument("--waymax-device", choices=["auto", "cpu", "gpu"], default="auto", help="Runtime device preference for Waymax/JAX. Use cpu to keep Waymax off CUDA and leave the GPU to PyTorch.")
    ap.add_argument("--jax-visible-devices", default=None, help="Comma-separated JAX device ids, relative to CUDA_VISIBLE_DEVICES. Example: 0 keeps Waymax/JAX on the first visible GPU.")
    ap.add_argument("--jax-preallocate", choices=["true", "false"], default="false", help="Set XLA_PYTHON_CLIENT_PREALLOCATE before importing JAX. Default false avoids JAX reserving most GPU memory.")
    ap.add_argument("--jax-mem-fraction", type=float, default=None, help="Optional XLA_PYTHON_CLIENT_MEM_FRACTION, e.g. 0.35. Mainly useful if --jax-preallocate true.")
    ap.add_argument("--keep-rollout-state", action="store_true", help="Keep final Waymax SimulatorState objects in memory. Off by default to avoid GPU memory growth across scenarios.")
    ap.add_argument("--clear-accelerator-cache", action="store_true", help="Best-effort gc/JAX/PyTorch cache cleanup between scenarios. Slower, but useful for diagnosing OOM.")
    ap.add_argument("--num-scenarios", type=int, default=None)
    ap.add_argument("--rollout-horizon-steps", type=int, default=None)
    ap.add_argument("--waymax-standard-metrics", action="store_true")
    ap.add_argument("--waymax-standard-metric-names", default=None, help="Comma-separated Waymax metric class names to compute, or 'all'. Example: OverlapMetric,OffroadMetric,ProgressionMetric,KinematicsInfeasibilityMetric")
    ap.add_argument("--reuse-waymax-env", action=argparse.BooleanOptionalAction, default=True, help="Reuse a compatible Waymax environment object across scenarios. Does not change rollout logic; avoids per-scenario construction overhead.")
    ap.add_argument("--prefilter-waymax-shards", action=argparse.BooleanOptionalAction, default=True, help="Apply modulo sharding before SimulatorState construction. Scenario assignment and metrics are unchanged.")
    ap.add_argument("--jit-waymax-env", action=argparse.BooleanOptionalAction, default=True, help="JIT cached Waymax reset/step functions, with automatic fallback to the original eager path.")
    ap.add_argument("--jit-waymax-metrics", action=argparse.BooleanOptionalAction, default=True, help="JIT Waymax metric compute functions, with automatic per-metric fallback.")
    ap.add_argument("--profile-waymax-runtime", action="store_true", help="Record per-scenario data/reset/policy/env-step/metric timing. Use on a small exact-ID subset first.")
    ap.add_argument("--profile-waymax-sync", action="store_true", help="Synchronize JAX device work around outer Waymax timing sections. Slows execution; profiling only.")
    ap.add_argument("--profile-policy-runtime", action="store_true", help="Record fine-grained COWP policy timings: state/map, candidate build, H2D, model forward, selection, action projection.")
    ap.add_argument("--profile-policy-sync", action="store_true", help="Synchronize the PyTorch CUDA stream around fine-grained policy timing sections. Profiling only.")
    ap.add_argument("--status-every", type=int, default=10, help="When --no-progress is used, print one compact rollout heartbeat every N completed scenarios.")
    ap.add_argument("--output", default="outputs/eval_metrics.json")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()
    _configure_waymax_runtime(args)
    cfg = load_config(args.label_config, args.data_config, args.eval_config)
    protocol_cfg = cfg.get("eval", {}) if isinstance(cfg, dict) else {}
    actual_non_ego_policy = str(protocol_cfg.get("actual_non_ego_policy", "logged_replay"))
    reactive_mixture_implemented = bool(protocol_cfg.get("reactive_mixture_implemented", False))
    if args.mode == "waymax" and actual_non_ego_policy != "logged_replay":
        raise ValueError(
            "This evaluator controls only the SDC and leaves non-ego agents on Waymax/log playback. "
            f"eval.actual_non_ego_policy={actual_non_ego_policy!r} would be a false protocol claim. "
            "Use logged_replay here or implement and validate a real sim-agent actor wrapper first."
        )

    if args.mode == "offline":
        metrics = offline_candidate_eval(args.labels_dir or cfg["outputs"]["labels_dir"], cfg, method=args.method, progress=not args.no_progress)
        payload = {args.method: metrics, "mode": "offline"}
    elif args.mode == "learned_offline":
        if not args.checkpoint:
            raise ValueError("--mode learned_offline requires --checkpoint")
        method_list = [x.strip() for x in str(args.methods or "").split(",") if x.strip()]
        if args.bcot_risk_budget_sweep and method_list:
            raise ValueError(
                "--bcot-risk-budget-sweep currently evaluates one --method. "
                "Run the shared --methods comparison at the calibrated budget in a separate command."
            )
        if args.bcot_risk_budget_sweep:
            budgets = [
                float(x.strip())
                for x in args.bcot_risk_budget_sweep.split(",")
                if x.strip()
            ]
            if args.bcot_risk_budget is not None:
                budgets.append(float(args.bcot_risk_budget))
            sweep = learned_offline_candidate_eval_budget_sweep(
                args.cache_dir or cfg["outputs"]["tensor_cache_dir"],
                args.checkpoint,
                cfg,
                bcot_risk_budgets=budgets,
                witness_threshold=args.witness_threshold,
                batch_size=args.batch_size,
                device=args.device,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                pin_memory=args.pin_memory,
                progress=not args.no_progress,
                gate_mode=args.ncf_gate_mode,
                secondary_witness_threshold=args.secondary_witness_threshold,
                secondary_opr_alpha=args.secondary_opr_alpha,
                priority_hard_threshold=args.priority_hard_threshold,
                soft_ncf_penalty=args.soft_ncf_penalty,
                method=args.method,
                offline_fallback=args.offline_fallback,
                adaptive_frontier_margin=args.adaptive_frontier_margin,
                outcome_risk_penalty=args.outcome_risk_penalty,
                outcome_risk_threshold=args.outcome_risk_threshold,
                subset_modulo=args.learned_subset_modulo,
                subset_remainder=args.learned_subset_remainder,
                max_scenes=args.learned_max_scenes,
            )
            target_budget = float(
                args.bcot_risk_budget
                if args.bcot_risk_budget is not None
                else cfg.get("planning", {}).get("candidate_transport_budget", 0.35)
            )
            metrics = min(
                sweep,
                key=lambda m: abs(float(m.get("bcot_risk_budget", target_budget)) - target_budget),
            )
            payload = {
                args.method: metrics,
                "mode": "learned_offline",
                "checkpoint": args.checkpoint,
                "ncf_gate_mode": args.ncf_gate_mode,
                "priority_hard_threshold": args.priority_hard_threshold,
                "offline_fallback": args.offline_fallback,
                "pair_witness_threshold": float(args.witness_threshold),
                "bcot_risk_budget_sweep": sweep,
            }
        elif method_list:
            thresholds = [float(args.witness_threshold)]
            if args.witness_threshold_sweep:
                thresholds.extend(float(x.strip()) for x in args.witness_threshold_sweep.split(",") if x.strip())
            multi_sweep = learned_offline_candidate_eval_methods(
                args.cache_dir or cfg["outputs"]["tensor_cache_dir"],
                args.checkpoint,
                cfg,
                methods=method_list,
                batch_size=args.batch_size,
                device=args.device,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                pin_memory=args.pin_memory,
                witness_thresholds=thresholds,
                bcot_risk_budget=args.bcot_risk_budget,
                progress=not args.no_progress,
                gate_mode=args.ncf_gate_mode,
                secondary_witness_threshold=args.secondary_witness_threshold,
                secondary_opr_alpha=args.secondary_opr_alpha,
                priority_hard_threshold=args.priority_hard_threshold,
                soft_ncf_penalty=args.soft_ncf_penalty,
                offline_fallback=args.offline_fallback,
                adaptive_frontier_margin=args.adaptive_frontier_margin,
                outcome_risk_penalty=args.outcome_risk_penalty,
                outcome_risk_threshold=args.outcome_risk_threshold,
                subset_modulo=args.learned_subset_modulo,
                subset_remainder=args.learned_subset_remainder,
                max_scenes=args.learned_max_scenes,
            )
            payload = {
                method_name: min(rows, key=lambda m: abs(float(m.get("witness_threshold", 0.5)) - float(args.witness_threshold)))
                for method_name, rows in multi_sweep.items()
            }
            payload.update({
                "mode": "learned_offline",
                "checkpoint": args.checkpoint,
                "ncf_gate_mode": args.ncf_gate_mode,
                "priority_hard_threshold": args.priority_hard_threshold,
                "offline_fallback": args.offline_fallback,
                "methods": method_list,
                "shared_model_pass": True,
            })
            if args.witness_threshold_sweep:
                payload["witness_threshold_sweep"] = multi_sweep
        elif args.witness_threshold_sweep:
            thresholds = [float(x.strip()) for x in args.witness_threshold_sweep.split(",") if x.strip()]
            thresholds.append(float(args.witness_threshold))
            sweep = learned_offline_candidate_eval_sweep(
                args.cache_dir or cfg["outputs"]["tensor_cache_dir"],
                args.checkpoint,
                cfg,
                batch_size=args.batch_size,
                device=args.device,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                pin_memory=args.pin_memory,
                witness_thresholds=thresholds,
                bcot_risk_budget=args.bcot_risk_budget,
                progress=not args.no_progress,
                gate_mode=args.ncf_gate_mode,
                secondary_witness_threshold=args.secondary_witness_threshold,
                secondary_opr_alpha=args.secondary_opr_alpha,
                priority_hard_threshold=args.priority_hard_threshold,
                soft_ncf_penalty=args.soft_ncf_penalty,
                method=args.method,
                offline_fallback=args.offline_fallback,
                adaptive_frontier_margin=args.adaptive_frontier_margin,
                outcome_risk_penalty=args.outcome_risk_penalty,
                outcome_risk_threshold=args.outcome_risk_threshold,
                subset_modulo=args.learned_subset_modulo,
                subset_remainder=args.learned_subset_remainder,
                max_scenes=args.learned_max_scenes,
            )
            metrics = min(sweep, key=lambda m: abs(float(m.get("witness_threshold", 0.5)) - float(args.witness_threshold)))
            payload = {args.method: metrics, "mode": "learned_offline", "checkpoint": args.checkpoint, "ncf_gate_mode": args.ncf_gate_mode, "priority_hard_threshold": args.priority_hard_threshold, "offline_fallback": args.offline_fallback, "witness_threshold_sweep": sweep}
        else:
            metrics = learned_offline_candidate_eval(
                args.cache_dir or cfg["outputs"]["tensor_cache_dir"],
                args.checkpoint,
                cfg,
                batch_size=args.batch_size,
                device=args.device,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                pin_memory=args.pin_memory,
                witness_threshold=args.witness_threshold,
                bcot_risk_budget=args.bcot_risk_budget,
                progress=not args.no_progress,
                gate_mode=args.ncf_gate_mode,
                secondary_witness_threshold=args.secondary_witness_threshold,
                secondary_opr_alpha=args.secondary_opr_alpha,
                priority_hard_threshold=args.priority_hard_threshold,
                soft_ncf_penalty=args.soft_ncf_penalty,
                method=args.method,
                offline_fallback=args.offline_fallback,
                adaptive_frontier_margin=args.adaptive_frontier_margin,
                outcome_risk_penalty=args.outcome_risk_penalty,
                outcome_risk_threshold=args.outcome_risk_threshold,
                subset_modulo=args.learned_subset_modulo,
                subset_remainder=args.learned_subset_remainder,
                max_scenes=args.learned_max_scenes,
            )
            payload = {args.method: metrics, "mode": "learned_offline", "checkpoint": args.checkpoint, "ncf_gate_mode": args.ncf_gate_mode, "priority_hard_threshold": args.priority_hard_threshold, "offline_fallback": args.offline_fallback}
    else:
        if args.policy_fn:
            policy_fn = import_policy_fn(args.policy_fn)
        elif args.checkpoint:
            policy_fn = make_cowp_policy(
                args.checkpoint,
                cfg,
                device=args.device,
                witness_threshold=args.witness_threshold,
                bcot_risk_budget=args.bcot_risk_budget,
                action_mode=args.waymax_action_mode,
                ncf_gate_mode=args.ncf_gate_mode,
                priority_hard_threshold=args.priority_hard_threshold,
                secondary_witness_threshold=args.secondary_witness_threshold,
                secondary_opr_alpha=args.secondary_opr_alpha,
                soft_ncf_penalty=args.soft_ncf_penalty,
                method=args.method,
                adaptive_frontier_margin=args.adaptive_frontier_margin,
                outcome_risk_penalty=args.outcome_risk_penalty,
                outcome_risk_threshold=args.outcome_risk_threshold,
                profile_policy_runtime=bool(args.profile_policy_runtime),
                profile_policy_runtime_sync=bool(args.profile_policy_sync),
            )
        else:
            raise ValueError("--mode waymax requires either --checkpoint for the built-in COWP policy wrapper or --policy-fn module:function.")
        horizon = args.rollout_horizon_steps or int(cfg.get("eval", {}).get("rollout_horizon_steps", cfg.get("time", {}).get("future_steps", 80)))
        scenario_ids = _load_scenario_ids_file(args.scenario_ids_file)
        rollouts = waymax_closed_loop_rollout(
            cfg,
            policy_fn,
            num_scenarios=args.num_scenarios,
            horizon_steps=horizon,
            progress=not args.no_progress,
            compute_standard_metrics=args.waymax_standard_metrics,
            action_mode=args.waymax_action_mode,
            keep_rollout_state=args.keep_rollout_state,
            clear_accelerator_cache=args.clear_accelerator_cache,
            split=args.waymax_split,
            tfexample_glob=args.tfexample_glob,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            reuse_env=bool(args.reuse_waymax_env),
            prefilter_shards=bool(args.prefilter_waymax_shards),
            jit_env=bool(args.jit_waymax_env),
            jit_standard_metrics=bool(args.jit_waymax_metrics),
            status_every=int(args.status_every),
            standard_metric_names=_parse_metric_names(args.waymax_standard_metric_names),
            scenario_ids=scenario_ids,
            tfexample_index_jsonl=args.tfexample_index_jsonl,
            profile_runtime=bool(args.profile_waymax_runtime),
            profile_runtime_sync=bool(args.profile_waymax_sync),
        )
        payload = {
            "mode": "waymax",
            "method": args.method,
            "checkpoint": args.checkpoint,
            "policy_fn": args.policy_fn,
            "ncf_gate_mode": args.ncf_gate_mode,
            "waymax_split": args.waymax_split,
            "tfexample_glob": args.tfexample_glob,
            "scenario_ids_file": args.scenario_ids_file,
            "scenario_ids_requested": int(len(scenario_ids)) if scenario_ids is not None else None,
            "scenario_ids_requested_on_shard": int(sum(1 for i in range(len(scenario_ids)) if i % max(int(args.num_shards), 1) == int(args.shard_index) % max(int(args.num_shards), 1))) if scenario_ids is not None else None,
            "scenario_ids_sha256": hashlib.sha256("\n".join(scenario_ids).encode("utf-8")).hexdigest() if scenario_ids is not None else None,
            "scenario_ids_resolved": [str(x.get("scenario_id")) for x in rollouts if x.get("scenario_id") is not None],
            "waymax_standard_metric_names": sorted(_parse_metric_names(args.waymax_standard_metric_names)) if isinstance(_parse_metric_names(args.waymax_standard_metric_names), set) else args.waymax_standard_metric_names,
            "reuse_waymax_env": bool(args.reuse_waymax_env),
            "prefilter_waymax_shards": bool(args.prefilter_waymax_shards),
            "jit_waymax_env": bool(args.jit_waymax_env),
            "jit_waymax_metrics": bool(args.jit_waymax_metrics),
            "shard_index": int(args.shard_index),
            "num_shards": int(args.num_shards),
            "num_rollouts": len(rollouts),
            "steps": [int(x.get("steps", 0)) for x in rollouts],
            "actual_non_ego_policy": actual_non_ego_policy,
            "reactive_mixture_implemented": reactive_mixture_implemented,
            "mechanism_ground_truth_available_online": False,
            "policy_diagnostic_summary": policy_diagnostic_summary(rollouts),
            "closed_loop_cowp_metric_summary": policy_diagnostic_episode_summary(rollouts),
            "scenario_diagnostics": policy_diagnostic_scenario_rows(rollouts),
            "physical_failure_attribution_summary": physical_failure_attribution_summary(rollouts),
            "tfexample_index_jsonl": args.tfexample_index_jsonl,
            "profile_waymax_runtime": bool(args.profile_waymax_runtime),
            "profile_waymax_sync": bool(args.profile_waymax_sync),
            "profile_policy_runtime": bool(args.profile_policy_runtime),
            "profile_policy_sync": bool(args.profile_policy_sync),
        }
        if args.waymax_standard_metrics:
            payload["standard_metrics"] = [x.get("standard_metrics", {}) for x in rollouts]
            payload["standard_metric_summary"] = aggregate_waymax_standard_metrics(rollouts)
        runtime_rows = [x.get("runtime_profile", {}) for x in rollouts if x.get("runtime_profile")]
        if runtime_rows:
            keys = ("data_next_s", "env_reset_s", "policy_s", "env_step_s", "metric_s", "scenario_total_s")
            runtime_summary = {}
            for key in keys:
                vals = [float(r.get(key, 0.0)) for r in runtime_rows]
                runtime_summary[f"mean/{key}"] = float(sum(vals) / max(len(vals), 1))
                runtime_summary[f"sum/{key}"] = float(sum(vals))
            total_component = sum(runtime_summary[f"sum/{k}"] for k in ("data_next_s", "env_reset_s", "policy_s", "env_step_s", "metric_s"))
            if total_component > 0.0:
                for key in ("data_next_s", "env_reset_s", "policy_s", "env_step_s", "metric_s"):
                    runtime_summary[f"fraction/{key}"] = float(runtime_summary[f"sum/{key}"] / total_component)
            runtime_summary["scenarios"] = int(len(runtime_rows))
            payload["waymax_runtime_profile_summary"] = runtime_summary
            payload["waymax_runtime_profiles"] = runtime_rows

    if isinstance(payload, dict):
        payload["bcot_risk_budget"] = float(
            cfg.get("planning", {}).get("candidate_transport_budget", 0.35)
            if args.bcot_risk_budget is None else args.bcot_risk_budget
        )
        payload["pair_witness_threshold"] = float(args.witness_threshold)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_safe)
    print(json.dumps(payload, indent=2, default=_json_safe))


if __name__ == "__main__":
    main()
