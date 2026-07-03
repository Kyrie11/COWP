from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cowp.core.config import load_config


def _configure_waymax_runtime(device: str) -> None:
    device = str(device or "auto").lower()
    # Candidate replay can instantiate many JAX buffers. Disable XLA's greedy GPU
    # preallocation by default. With this set, nvidia-smi may show only a small
    # memory increase even when kernels are really running on the GPU.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if device == "cpu":
        # CPU mode must override stale shell state such as CUDA_VISIBLE_DEVICES=0
        # from earlier training commands.
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    elif device == "gpu":
        # Do NOT set JAX_PLATFORMS=gpu here.  In some JAX 0.4.x plugin setups,
        # the generic value "gpu" can initialize both CUDA and ROCm backends; on
        # CUDA-only machines this may fail with:
        #   Unable to initialize backend 'rocm' ... GpuAllocatorConfig
        # Let JAX auto-select the visible accelerator, then verify that a GPU is
        # actually available in _print_jax_runtime.
        if os.environ.get("JAX_PLATFORM_NAME", "").strip().lower() in {"cpu", "gpu", "cuda"}:
            os.environ.pop("JAX_PLATFORM_NAME", None)
        platforms = os.environ.get("JAX_PLATFORMS")
        if platforms is not None:
            normalized = platforms.strip().lower()
            # Clear the stale CPU lock and the stale generic-gpu lock from older
            # commands.  A user-provided explicit platform list such as
            # "cuda,cpu" is preserved.
            if normalized in {"", "cpu", "gpu"}:
                os.environ.pop("JAX_PLATFORMS", None)
        # Restrict each worker process to one card with CUDA_VISIBLE_DEVICES in
        # the launch command.  If CUDA cannot initialize, the runtime check below
        # raises before any labels are written.


def _print_jax_runtime(device: str, *, require_gpu: bool = False) -> None:
    try:
        import jax  # type: ignore
    except Exception as exc:
        if require_gpu:
            raise
        print(f"[waymax-runtime] JAX unavailable: {exc}")
        return
    backend = jax.default_backend()
    devices = jax.devices()
    try:
        gpu_devices = jax.devices("gpu")
    except Exception:
        gpu_devices = []
    print(
        "[waymax-runtime] "
        f"JAX {getattr(jax, '__version__', '?')} default_backend={backend} "
        f"devices={devices} gpu_devices={gpu_devices} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
        f"JAX_PLATFORM_NAME={os.environ.get('JAX_PLATFORM_NAME', '<unset>')} "
        f"JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS', '<unset>')}"
    )
    if require_gpu and not gpu_devices:
        raise RuntimeError(
            "--waymax-device gpu was requested, but JAX did not initialize any GPU device. "
            "Check CUDA_VISIBLE_DEVICES, JAX_PLATFORMS/JAX_PLATFORM_NAME, NVIDIA driver, "
            "and jax/jaxlib/jax-cuda12-plugin installation."
        )


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
    ap.add_argument("--state-source", choices=["auto", "cache", "tfexample"], default="auto", help="Where to build Waymax SimulatorState from. auto/cache uses WOMD features already stored in tensor cache and avoids a second TFRecord scan; tfexample uses --tfexample-glob.")
    ap.add_argument("--full-waymax-scan", action="store_true", help="Use the old full Waymax state generator. Default scans tf.Example ids cheaply and builds Waymax states only for cache scene ids. Ignored when --state-source cache/auto uses cached WOMD tensors.")
    ap.add_argument("--verify-cache-sid", action="store_true", help="Read scenario/id from each npz instead of trusting filename stem as scenario id.")
    ap.add_argument("--num-shards", type=int, default=1, help="Split cache files into this many deterministic shards for multiple parallel runs.")
    ap.add_argument("--shard-index", type=int, default=0, help="Shard index in [0, num_shards). Use a separate outcomes-jsonl per shard.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite an existing outcomes JSONL instead of resuming it.")
    ap.add_argument("--gc-every-scenes", type=int, default=16, help="Run Python GC every N matched scenes. Larger values reduce overhead; set 0 to disable explicit GC.")
    ap.add_argument("--profile-replay-jsonl", default=None, help="Optional per-scene replay timing JSONL. Enables per-candidate timing breakdown fields in the outcomes JSONL and aggregated timing fields in the profile.")
    ap.add_argument("--jit-env-step", action="store_true", help="Best-effort JAX-jit wrapper around env.step. Preserves the same actions, metrics and done checks; falls back to eager step if tracing is unsupported.")
    ap.add_argument("--done-check-interval", type=int, default=1, help="Check Waymax state.done every N steps. Default 1 preserves previous behavior. 0 disables early-done checks and should only be used after validating equivalence on smoke runs.")
    ap.add_argument("--metric-eval-mode", choices=["final", "step", "sampled", "interval"], default="step", help="step preserves the exact per-step Waymax metric path. final computes safety metrics once at the final state and is fastest but can miss transient collisions. sampled/interval computes metrics at step 1, every --metric-eval-interval steps, and the final step as a quality/speed compromise.")
    ap.add_argument("--metric-eval-interval", type=int, default=5, help="Only used with --metric-eval-mode sampled/interval. Compute Waymax safety metrics at step 1, every N rollout steps, and the final step. Lower is more accurate but slower; 1 is equivalent to step mode.")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--no-jax-runtime-print", action="store_true", help="Do not print JAX backend/device information at startup.")
    args = ap.parse_args()

    _configure_waymax_runtime(args.waymax_device)
    if not args.no_jax_runtime_print:
        _print_jax_runtime(args.waymax_device, require_gpu=(str(args.waymax_device).lower() == "gpu"))

    # Import the replay implementation only after JAX-related environment
    # variables are cleaned.  This prevents future transitive imports of Waymax
    # or JAX from initializing the wrong backend before configuration.
    from cowp.waymax_eval.candidate_replay import replay_cache_candidates_to_jsonl

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
        state_source=args.state_source,
        profile_replay_jsonl=args.profile_replay_jsonl,
        jit_env_step=bool(args.jit_env_step),
        done_check_interval=int(args.done_check_interval),
        metric_eval_mode=str(args.metric_eval_mode),
        metric_eval_interval=int(args.metric_eval_interval),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
