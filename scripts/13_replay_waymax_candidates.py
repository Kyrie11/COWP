from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cowp.core.config import load_config
from cowp.waymax_eval.candidate_replay import replay_cache_candidates_to_jsonl


def _configure_waymax_runtime(device: str) -> None:
    device = str(device or "auto").lower()
    if device == "cpu":
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    elif device == "gpu":
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    else:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run real Waymax candidate replay for existing COWP tensor-cache files "
            "and write candidate-level outcomes as JSONL. Run scripts/12 afterward "
            "to attach the JSONL fields into the cache."
        )
    )
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--label-config", default="configs/label.yaml")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--tfexample-glob", default=None, help="WOMD tf.Example glob/path for the same split as --cache-dir.")
    ap.add_argument("--split", choices=["training", "validation", "testing"], default=None)
    ap.add_argument("--outcomes-jsonl", required=True)
    ap.add_argument("--candidate-selection", choices=["balanced", "selected", "all", "noncoercive", "false_safe", "conventional"], default="balanced")
    ap.add_argument("--max-candidates-per-scene", type=int, default=8, help="0 or negative means all selected candidates for the selection mode.")
    ap.add_argument("--rollout-horizon-steps", type=int, default=None)
    ap.add_argument("--limit-scenes", type=int, default=None)
    ap.add_argument("--waymax-action-mode", choices=["delta_xy_yaw", "absolute_xy_yaw"], default="delta_xy_yaw")
    ap.add_argument("--waymax-device", choices=["auto", "cpu", "gpu"], default="cpu")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite an existing outcomes JSONL instead of resuming it.")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    _configure_waymax_runtime(args.waymax_device)
    cfg = load_config(args.label_config, args.data_config, args.eval_config)
    horizon = int(args.rollout_horizon_steps if args.rollout_horizon_steps is not None else cfg.get("eval", {}).get("rollout_horizon_steps", cfg.get("time", {}).get("future_steps", 80)))
    data_cfg = load_config(args.data_config)
    summary = replay_cache_candidates_to_jsonl(
        cache_dir=args.cache_dir,
        data_config=data_cfg,
        outcomes_jsonl=args.outcomes_jsonl,
        cfg=cfg,
        tfexample_glob=args.tfexample_glob,
        split=args.split,
        candidate_selection=args.candidate_selection,
        max_candidates_per_scene=args.max_candidates_per_scene,
        horizon_steps=horizon,
        action_mode=args.waymax_action_mode,
        limit_scenes=args.limit_scenes,
        resume=not args.overwrite,
        progress=not args.no_progress,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
