from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cowp.core.config import load_config
from cowp.waymax_eval.rollout import import_policy_fn, learned_offline_candidate_eval, learned_offline_candidate_eval_sweep, offline_candidate_eval, waymax_closed_loop_rollout
from cowp.waymax_eval.policy_wrapper import make_cowp_policy
from cowp.waymax_eval.metrics_cowp import policy_diagnostic_episode_summary, policy_diagnostic_summary
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
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--witness-threshold", type=float, default=0.5)
    ap.add_argument("--witness-threshold-sweep", default=None, help="Comma-separated thresholds for learned_offline diagnostic sweep, e.g. 0.1,0.2,0.3,0.5,0.7.")
    ap.add_argument("--ncf-gate-mode", choices=["hard", "priority", "soft"], default="hard", help="Candidate acceptance gate. hard reproduces the original universal veto; priority hard-vetoes only high-priority/severe coercion and penalizes the rest; soft uses no witness hard veto.")
    ap.add_argument("--priority-hard-threshold", type=float, default=0.55, help="Online priority proxy threshold for hard P-NCF rejection.")
    ap.add_argument("--secondary-witness-threshold", type=float, default=0.85, help="Severe witness threshold for priority/soft gate diagnostics.")
    ap.add_argument("--secondary-opr-alpha", type=float, default=0.10, help="Severe low-option-preservation threshold used with secondary witness threshold.")
    ap.add_argument("--soft-ncf-penalty", type=float, default=1.5, help="Score penalty weight for non-hard coercion evidence in priority/soft gates.")
    ap.add_argument("--method", default="cowp")
    ap.add_argument("--mode", choices=["offline", "learned_offline", "waymax"], default="offline")
    ap.add_argument("--policy-fn", default=None, help="For --mode waymax: optional Python callable spec 'module:function' returning Waymax actions. If omitted with --checkpoint, a COWP checkpoint policy is used.")
    ap.add_argument("--waymax-action-mode", choices=["delta_xy_yaw", "absolute_xy_yaw"], default="delta_xy_yaw")
    ap.add_argument("--waymax-device", choices=["auto", "cpu", "gpu"], default="auto", help="Runtime device preference for Waymax/JAX. Use cpu to keep Waymax off CUDA and leave the GPU to PyTorch.")
    ap.add_argument("--jax-visible-devices", default=None, help="Comma-separated JAX device ids, relative to CUDA_VISIBLE_DEVICES. Example: 0 keeps Waymax/JAX on the first visible GPU.")
    ap.add_argument("--jax-preallocate", choices=["true", "false"], default="false", help="Set XLA_PYTHON_CLIENT_PREALLOCATE before importing JAX. Default false avoids JAX reserving most GPU memory.")
    ap.add_argument("--jax-mem-fraction", type=float, default=None, help="Optional XLA_PYTHON_CLIENT_MEM_FRACTION, e.g. 0.35. Mainly useful if --jax-preallocate true.")
    ap.add_argument("--keep-rollout-state", action="store_true", help="Keep final Waymax SimulatorState objects in memory. Off by default to avoid GPU memory growth across scenarios.")
    ap.add_argument("--clear-accelerator-cache", action="store_true", help="Best-effort gc/JAX/PyTorch cache cleanup between scenarios. Slower, but useful for diagnosing OOM.")
    ap.add_argument("--num-scenarios", type=int, default=None)
    ap.add_argument("--rollout-horizon-steps", type=int, default=None)
    ap.add_argument("--waymax-standard-metrics", action="store_true")
    ap.add_argument("--output", default="outputs/eval_metrics.json")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()
    _configure_waymax_runtime(args)
    cfg = load_config(args.label_config, args.data_config, args.eval_config)

    if args.mode == "offline":
        metrics = offline_candidate_eval(args.labels_dir or cfg["outputs"]["labels_dir"], cfg, method=args.method, progress=not args.no_progress)
        payload = {args.method: metrics, "mode": "offline"}
    elif args.mode == "learned_offline":
        if not args.checkpoint:
            raise ValueError("--mode learned_offline requires --checkpoint")
        if args.witness_threshold_sweep:
            thresholds = [float(x.strip()) for x in args.witness_threshold_sweep.split(",") if x.strip()]
            thresholds.append(float(args.witness_threshold))
            sweep = learned_offline_candidate_eval_sweep(
                args.cache_dir or cfg["outputs"]["tensor_cache_dir"],
                args.checkpoint,
                cfg,
                batch_size=args.batch_size,
                device=args.device,
                witness_thresholds=thresholds,
                progress=not args.no_progress,
                gate_mode=args.ncf_gate_mode,
                secondary_witness_threshold=args.secondary_witness_threshold,
                secondary_opr_alpha=args.secondary_opr_alpha,
                soft_ncf_penalty=args.soft_ncf_penalty,
            )
            metrics = min(sweep, key=lambda m: abs(float(m.get("witness_threshold", 0.5)) - float(args.witness_threshold)))
            payload = {args.method: metrics, "mode": "learned_offline", "checkpoint": args.checkpoint, "ncf_gate_mode": args.ncf_gate_mode, "witness_threshold_sweep": sweep}
        else:
            metrics = learned_offline_candidate_eval(
                args.cache_dir or cfg["outputs"]["tensor_cache_dir"],
                args.checkpoint,
                cfg,
                batch_size=args.batch_size,
                device=args.device,
                witness_threshold=args.witness_threshold,
                progress=not args.no_progress,
                gate_mode=args.ncf_gate_mode,
                secondary_witness_threshold=args.secondary_witness_threshold,
                secondary_opr_alpha=args.secondary_opr_alpha,
                soft_ncf_penalty=args.soft_ncf_penalty,
            )
            payload = {args.method: metrics, "mode": "learned_offline", "checkpoint": args.checkpoint, "ncf_gate_mode": args.ncf_gate_mode}
    else:
        if args.policy_fn:
            policy_fn = import_policy_fn(args.policy_fn)
        elif args.checkpoint:
            policy_fn = make_cowp_policy(
                args.checkpoint,
                cfg,
                device=args.device,
                witness_threshold=args.witness_threshold,
                action_mode=args.waymax_action_mode,
                ncf_gate_mode=args.ncf_gate_mode,
                priority_hard_threshold=args.priority_hard_threshold,
                secondary_witness_threshold=args.secondary_witness_threshold,
                secondary_opr_alpha=args.secondary_opr_alpha,
                soft_ncf_penalty=args.soft_ncf_penalty,
            )
        else:
            raise ValueError("--mode waymax requires either --checkpoint for the built-in COWP policy wrapper or --policy-fn module:function.")
        horizon = args.rollout_horizon_steps or int(cfg.get("eval", {}).get("rollout_horizon_steps", cfg.get("time", {}).get("future_steps", 80)))
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
        )
        payload = {
            "mode": "waymax",
            "method": args.method,
            "checkpoint": args.checkpoint,
            "policy_fn": args.policy_fn,
            "ncf_gate_mode": args.ncf_gate_mode,
            "num_rollouts": len(rollouts),
            "steps": [int(x.get("steps", 0)) for x in rollouts],
            "policy_diagnostic_summary": policy_diagnostic_summary(rollouts),
            "closed_loop_cowp_metric_summary": policy_diagnostic_episode_summary(rollouts),
        }
        if args.waymax_standard_metrics:
            payload["standard_metrics"] = [x.get("standard_metrics", {}) for x in rollouts]
            payload["standard_metric_summary"] = aggregate_waymax_standard_metrics(rollouts)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_safe)
    print(json.dumps(payload, indent=2, default=_json_safe))


if __name__ == "__main__":
    main()
