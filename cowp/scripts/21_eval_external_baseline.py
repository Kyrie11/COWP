from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from cowp.core.config import load_config
from cowp.data.dataset import collate_torch
from cowp.external_baselines.adapters import ExternalCOWPDataset, label_from_batch_item, make_external_batch
from cowp.external_baselines.waymax_policy import build_external_model_from_checkpoint, make_external_waymax_policy
from cowp.utils.progress import tqdm_iter
from cowp.waymax_eval.metrics_cowp import policy_diagnostic_episode_summary, policy_diagnostic_summary
from cowp.waymax_eval.metrics_standard import aggregate_waymax_standard_metrics
from cowp.waymax_eval.rollout import _LearnedMetricsAccumulator, waymax_closed_loop_rollout


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def _safe_len(obj: Any) -> int | None:
    try:
        return int(len(obj))
    except Exception:
        return None


def _configure_waymax_runtime(args) -> None:
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
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def learned_offline_eval(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    _log(f"loading checkpoint {args.checkpoint}")
    model, baseline, model_cfg, ckpt_args, device = build_external_model_from_checkpoint(args.checkpoint, cfg, args.device)
    max_neighbors = int(args.max_neighbors or ckpt_args.get("max_neighbors", 10))
    max_candidates = int(args.max_candidates or ckpt_args.get("max_candidates", cfg.get("limits", {}).get("max_candidates", 30)))
    future_len = int(args.future_len or ckpt_args.get("future_len", cfg.get("time", {}).get("future_steps", 80)))
    _log(f"checkpoint ready baseline={baseline} device={device} max_neighbors={max_neighbors} max_candidates={max_candidates} future_len={future_len}")
    _log(f"loading learned-offline dataset from {args.cache_dir}")
    ds = ExternalCOWPDataset(args.cache_dir, include_waymax_outcomes=True)
    _log(f"learned-offline dataset ready scenes={len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_torch, pin_memory=False)
    total_batches = _safe_len(loader)
    _log(f"learned-offline loader ready batches={total_batches} batch_size={args.batch_size} workers={args.num_workers}")
    acc = _LearnedMetricsAccumulator(beta_default=float(cfg.get("label", {}).get("beta_default", 0.65)))
    selected_rows = []
    log_every = max(int(getattr(args, "log_every", 0) or 0), 0)
    iterator = tqdm_iter(loader, enabled=not args.no_progress, total=total_batches, desc=f"learned_offline {baseline}", unit="batch")
    seen = 0
    t0 = time.time()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(iterator, start=1):
            ext = make_external_batch(batch, model_cfg, device=device, max_neighbors=max_neighbors, max_candidates=max_candidates, horizon=future_len)
            if baseline == "gameformer":
                scores = model.score_candidates(ext.gameformer_inputs, ext.candidates, ext.candidate_valid)
            else:
                scores = model.score_candidates(ext.dtpp_inputs, ext.dtpp_candidate_tree, ext.candidate_valid, timesteps=future_len)
            accept = ext.candidate_valid
            if args.require_conventional_safe:
                accept2 = accept & ext.conventional_safe
                accept = torch.where(accept2.any(dim=1, keepdim=True), accept2, accept)
            masked_scores = torch.where(accept, scores, torch.full_like(scores, -1e9))
            selected = torch.argmax(masked_scores, dim=1)
            for i in range(selected.shape[0]):
                label = label_from_batch_item(batch, i)
                accepted_np = accept[i].detach().cpu().numpy().astype(bool)
                selected_i = int(selected[i].detach().cpu()) if accepted_np.any() else -1
                acc.add_selection(selected_i, accepted_np, label, cert=None)
                if args.dump_selections and len(selected_rows) < args.max_dump_rows:
                    selected_rows.append({"selected_idx": selected_i, "selected_score": float(scores[i, selected_i].detach().cpu()) if selected_i >= 0 else None})
            seen += int(selected.shape[0])
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(samples=seen, refresh=False)
            if log_every and (batch_idx == 1 or batch_idx % log_every == 0 or (total_batches is not None and batch_idx == total_batches)):
                elapsed = max(time.time() - t0, 1e-6)
                total_txt = str(total_batches) if total_batches is not None else "?"
                _log(f"learned_offline {baseline} batch={batch_idx}/{total_txt} samples={seen} batch_rate={batch_idx/elapsed:.3f}/s sample_rate={seen/elapsed:.1f}/s")
    metrics = acc.finish(auprc=0.0, rank_good=0, rank_total=0, witness_threshold=0.0)
    metrics.update({
        "baseline": baseline,
        "checkpoint": str(args.checkpoint),
        "max_neighbors": max_neighbors,
        "max_candidates": max_candidates,
        "future_len": future_len,
        "require_conventional_safe": bool(args.require_conventional_safe),
    })
    _log(f"learned_offline {baseline} done samples={seen}")
    return {"mode": "learned_offline", baseline: metrics, "selections_preview": selected_rows}


def waymax_eval(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    _log(f"building Waymax policy checkpoint={args.checkpoint}")
    policy = make_external_waymax_policy(
        args.checkpoint,
        cfg,
        device=args.device,
        action_mode=args.waymax_action_mode,
        require_conventional_safe=args.require_conventional_safe,
    )
    horizon = int(args.rollout_horizon_steps or cfg.get("eval", {}).get("rollout_horizon_steps", cfg.get("time", {}).get("future_steps", 80)))
    _log(
        f"waymax {policy.baseline} start split={args.waymax_split} tfexample_glob={args.tfexample_glob} "
        f"num_scenarios={args.num_scenarios} horizon={horizon} progress={not args.no_progress} standard_metrics={args.waymax_standard_metrics}"
    )
    t0 = time.time()
    rollouts = waymax_closed_loop_rollout(
        cfg,
        policy,
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
    )
    _log(f"waymax {policy.baseline} rollout done episodes={len(rollouts)} seconds={time.time()-t0:.1f}")
    payload: dict[str, Any] = {
        "mode": "waymax",
        "baseline": policy.baseline,
        "checkpoint": args.checkpoint,
        "waymax_split": args.waymax_split,
        "tfexample_glob": args.tfexample_glob,
        "num_rollouts": len(rollouts),
        "steps": [int(x.get("steps", 0)) for x in rollouts],
        "policy_diagnostic_summary": policy_diagnostic_summary(rollouts),
        "closed_loop_cowp_metric_summary": policy_diagnostic_episode_summary(rollouts),
    }
    if args.waymax_standard_metrics:
        payload["standard_metrics"] = [x.get("standard_metrics", {}) for x in rollouts]
        payload["standard_metric_summary"] = aggregate_waymax_standard_metrics(rollouts)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate GameFormer/DTPP external baselines on COWP learned-offline or real Waymax closed loop.")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--label-config", default="configs/label.yaml")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--mode", choices=["learned_offline", "waymax"], default="learned_offline")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-neighbors", type=int, default=None)
    ap.add_argument("--max-candidates", type=int, default=None)
    ap.add_argument("--future-len", type=int, default=None)
    ap.add_argument("--require-conventional-safe", action="store_true")
    ap.add_argument("--dump-selections", action="store_true")
    ap.add_argument("--max-dump-rows", type=int, default=50)
    ap.add_argument("--waymax-split", choices=["training", "validation", "testing"], default="validation")
    ap.add_argument("--tfexample-glob", default=None)
    ap.add_argument("--num-scenarios", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--rollout-horizon-steps", type=int, default=None)
    ap.add_argument("--waymax-action-mode", choices=["delta_xy_yaw", "absolute_xy_yaw"], default="delta_xy_yaw")
    ap.add_argument("--waymax-device", choices=["auto", "cpu", "gpu"], default="auto")
    ap.add_argument("--jax-visible-devices", default=None)
    ap.add_argument("--jax-preallocate", choices=["true", "false"], default="false")
    ap.add_argument("--jax-mem-fraction", type=float, default=None)
    ap.add_argument("--waymax-standard-metrics", action="store_true")
    ap.add_argument("--keep-rollout-state", action="store_true")
    ap.add_argument("--clear-accelerator-cache", action="store_true")
    ap.add_argument("--output", required=True)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--log-every", type=int, default=int(os.environ.get("LOG_EVERY", "25")))
    args = ap.parse_args()
    _log(f"external eval entry mode={args.mode} pid={os.getpid()} python={sys.executable} torch={torch.__version__}")
    _log(f"args={json.dumps(vars(args), sort_keys=True)}")
    _configure_waymax_runtime(args)
    cfg = load_config(args.label_config, args.data_config, args.eval_config)
    if args.mode == "learned_offline":
        if not args.cache_dir:
            raise ValueError("--mode learned_offline requires --cache-dir")
        payload = learned_offline_eval(args, cfg)
    else:
        payload = waymax_eval(args, cfg)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_safe)
    _log(f"wrote output {out}")
    print(json.dumps(payload, indent=2, default=_json_safe), flush=True)


if __name__ == "__main__":
    main()
