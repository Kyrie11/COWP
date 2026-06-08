from __future__ import annotations

import argparse
import json
from pathlib import Path

from cowp.core.config import load_config
from cowp.waymax_eval.rollout import import_policy_fn, offline_candidate_eval, waymax_closed_loop_rollout


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
    ap = argparse.ArgumentParser(description="Evaluate COWP labels/planner offline or run a real Waymax closed-loop rollout.")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--label-config", default="configs/label.yaml")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--method", default="cowp")
    ap.add_argument("--mode", choices=["offline", "waymax"], default="offline")
    ap.add_argument("--policy-fn", default=None, help="For --mode waymax: Python callable spec 'module:function' returning Waymax actions.")
    ap.add_argument("--num-scenarios", type=int, default=None)
    ap.add_argument("--rollout-horizon-steps", type=int, default=None)
    ap.add_argument("--waymax-standard-metrics", action="store_true")
    ap.add_argument("--output", default="outputs/eval_metrics.json")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.label_config, args.data_config, args.eval_config)

    if args.mode == "offline":
        metrics = offline_candidate_eval(args.labels_dir or cfg["outputs"]["labels_dir"], cfg, method=args.method, progress=not args.no_progress)
        payload = {args.method: metrics, "mode": "offline"}
    else:
        if not args.policy_fn:
            raise ValueError("--mode waymax requires --policy-fn module:function so the rollout can produce Waymax actions.")
        policy_fn = import_policy_fn(args.policy_fn)
        horizon = args.rollout_horizon_steps or int(cfg.get("eval", {}).get("rollout_horizon_steps", cfg.get("time", {}).get("future_steps", 80)))
        rollouts = waymax_closed_loop_rollout(
            cfg,
            policy_fn,
            num_scenarios=args.num_scenarios,
            horizon_steps=horizon,
            progress=not args.no_progress,
            compute_standard_metrics=args.waymax_standard_metrics,
        )
        payload = {
            "mode": "waymax",
            "num_rollouts": len(rollouts),
            "steps": [int(x.get("steps", 0)) for x in rollouts],
        }
        if args.waymax_standard_metrics:
            payload["standard_metrics"] = [x.get("standard_metrics", {}) for x in rollouts]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_safe)
    print(json.dumps(payload, indent=2, default=_json_safe))


if __name__ == "__main__":
    main()
