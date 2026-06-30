from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cowp.core.config import load_config
from cowp.waymax_eval.candidate_replay import replay_cache_candidates_to_jsonl


def _configure_waymax_runtime(device: str) -> None:
    device = str(device or "auto").lower()
    # Candidate replay can instantiate many JAX buffers.  Disable XLA's greedy GPU
    # preallocation regardless of CPU/GPU mode; this is safe when the variable was
    # unset and prevents smoke tests from reserving nearly all GPU memory.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if device == "cpu":
        # CPU mode must override stale shell state such as CUDA_VISIBLE_DEVICES=0
        # from earlier training commands.
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    elif device == "gpu":
        # A previous dataset-building shell often exports JAX_PLATFORM_NAME=cpu.
        # The explicit --waymax-device gpu flag should not inherit that CPU lock.
        if os.environ.get("JAX_PLATFORM_NAME", "").lower() == "cpu":
            os.environ.pop("JAX_PLATFORM_NAME", None)


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
    ap.add_argument("--tfexample-index-jsonl", default=None, help="Optional scenario-id index from 02_build_tensor_cache. When provided, replay scans only TFRecord shards containing cache scene ids.")
    ap.add_argument("--split", choices=["training", "validation", "testing"], default=None)
    ap.add_argument("--outcomes-jsonl", required=True)
    ap.add_argument("--candidate-selection", choices=["balanced", "selected", "all", "noncoercive", "false_safe", "conventional"], default="balanced")
    ap.add_argument("--max-candidates-per-scene", type=int, default=8, help="0 or negative means all selected candidates for the selection mode.")
    ap.add_argument("--rollout-horizon-steps", type=int, default=None)
    ap.add_argument("--limit-scenes", type=int, default=None)
    ap.add_argument("--waymax-action-mode", choices=["delta_xy_yaw", "absolute_xy_yaw"], default="absolute_xy_yaw", help="absolute_xy_yaw is much faster because it avoids per-step host/device state extraction; delta_xy_yaw remains available for compatibility.")
    ap.add_argument("--waymax-device", choices=["auto", "cpu", "gpu"], default="gpu")
    ap.add_argument("--metric-set", choices=["safety", "safety_logdiv", "standard", "none"], default="safety", help="safety computes only overlap/offroad labels. safety_logdiv also computes log divergence. standard computes all available Waymax metrics.")
    ap.add_argument("--full-waymax-scan", action="store_true", help="Use the old full Waymax state generator. Default scans tf.Example ids cheaply and builds Waymax states only for cache scene ids.")
    ap.add_argument("--verify-cache-sid", action="store_true", help="Read scenario/id from each npz instead of trusting filename stem as scenario id.")
    ap.add_argument("--num-shards", type=int, default=1, help="Split cache files into this many deterministic shards for multiple parallel runs.")
    ap.add_argument("--shard-index", type=int, default=0, help="Shard index in [0, num_shards). Use a separate outcomes-jsonl per shard.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite an existing outcomes JSONL instead of resuming it.")
    ap.add_argument("--gc-every-scenes", type=int, default=16, help="Run Python GC every N matched scenes. Larger values reduce overhead; set 0 to disable explicit GC.")
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
        metric_set=args.metric_set,
        limit_scenes=args.limit_scenes,
        resume=not args.overwrite,
        progress=not args.no_progress,
        matched_only=not args.full_waymax_scan,
        verify_cache_sid=args.verify_cache_sid,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        tfexample_index_jsonl=args.tfexample_index_jsonl,
        gc_every_scenes=args.gc_every_scenes,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
