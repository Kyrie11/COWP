from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from cowp.core.config import load_config
from cowp.data.dataset import collate_torch
from cowp.external_baselines.adapters import ExternalCOWPDataset, label_from_batch_item
from cowp.external_baselines.rule_based import RULE_BASELINES, select_rule_indices
from cowp.external_baselines.rule_waymax_policy import make_rule_waymax_policy
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


def _reference_family(method: str) -> str:
    table = {
        "idm_lattice": "Treiber-Hennecke-Helbing IDM longitudinal car-following + local lattice candidate selection",
        "frenet_optimal": "Werling-Ziegler-Kammel-Thrun Frenet-frame optimal trajectory cost",
        "state_lattice": "Pivtoraiko-Kelly state lattice motion-primitive edge cost + progress heuristic",
    }
    return table.get(method, method)


def learned_offline_rule_eval(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    _log(f"loading rule learned-offline dataset from {args.cache_dir}")
    ds = ExternalCOWPDataset(args.cache_dir, include_waymax_outcomes=True)
    _log(f"rule learned-offline dataset ready baseline={args.baseline} scenes={len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_torch, pin_memory=False)
    total_batches = _safe_len(loader)
    _log(f"rule learned-offline loader ready baseline={args.baseline} batches={total_batches} batch_size={args.batch_size} workers={args.num_workers}")
    acc = _LearnedMetricsAccumulator(beta_default=float(cfg.get("label", {}).get("beta_default", cfg.get("burden", {}).get("beta0_vehicle", 0.65))))
    selected_rows = []
    log_every = max(int(getattr(args, "log_every", 0) or 0), 0)
    iterator = tqdm_iter(loader, enabled=not args.no_progress, total=total_batches, desc=f"learned_offline {args.baseline}", unit="batch")
    seen = 0
    t0 = time.time()
    for batch_idx, batch in enumerate(iterator, start=1):
        selected, accept, scores = select_rule_indices(
            batch,
            cfg,
            args.baseline,
            require_conventional_safe=not args.no_conventional_filter,
        )
        B = int(selected.shape[0])
        seen += B
        for i in range(B):
            label = label_from_batch_item(batch, i)
            accepted_np = np.asarray(accept[i], dtype=bool)
            selected_i = int(selected[i]) if selected[i] >= 0 else -1
            acc.add_selection(selected_i, accepted_np, label, cert=None)
            if args.dump_selections and len(selected_rows) < args.max_dump_rows:
                selected_rows.append({
                    "selected_idx": selected_i,
                    "selected_score": float(scores[i, selected_i]) if selected_i >= 0 else None,
                    "accepted_candidates": int(accepted_np.sum()),
                })
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(samples=seen, refresh=False)
        if log_every and (batch_idx == 1 or batch_idx % log_every == 0 or (total_batches is not None and batch_idx == total_batches)):
            elapsed = max(time.time() - t0, 1e-6)
            total_txt = str(total_batches) if total_batches is not None else "?"
            _log(f"rule learned_offline {args.baseline} batch={batch_idx}/{total_txt} samples={seen} batch_rate={batch_idx/elapsed:.3f}/s sample_rate={seen/elapsed:.1f}/s")
    metrics = acc.finish(auprc=0.0, rank_good=0, rank_total=0, witness_threshold=0.0)
    metrics.update({
        "baseline": args.baseline,
        "reference_family": _reference_family(args.baseline),
        "require_conventional_safe": not bool(args.no_conventional_filter),
    })
    _log(f"rule learned_offline {args.baseline} done samples={seen}")
    return {"mode": "learned_offline", args.baseline: metrics, "selections_preview": selected_rows}


def waymax_rule_eval(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    policy = make_rule_waymax_policy(
        args.baseline,
        cfg,
        action_mode=args.waymax_action_mode,
        require_conventional_safe=not args.no_conventional_filter,
    )
    horizon = int(args.rollout_horizon_steps or cfg.get("eval", {}).get("rollout_horizon_steps", cfg.get("time", {}).get("future_steps", 80)))
    _log(
        f"rule waymax {policy.baseline} start split={args.waymax_split} tfexample_glob={args.tfexample_glob} "
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
    _log(f"rule waymax {policy.baseline} rollout done episodes={len(rollouts)} seconds={time.time()-t0:.1f}")
    payload: dict[str, Any] = {
        "mode": "waymax",
        "baseline": policy.baseline,
        "reference_family": _reference_family(policy.baseline),
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
    ap = argparse.ArgumentParser(description="Evaluate rule-based external baselines on COWP learned-offline or real Waymax closed loop.")
    ap.add_argument("--baseline", choices=sorted(RULE_BASELINES), required=True)
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--label-config", default="configs/label.yaml")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--mode", choices=["learned_offline", "waymax"], default="learned_offline")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-conventional-filter", action="store_true", help="Allow all valid candidates instead of filtering to conventional-safe candidates.")
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
    _log(f"rule baseline eval entry mode={args.mode} baseline={args.baseline} pid={os.getpid()} python={sys.executable}")
    _log(f"args={json.dumps(vars(args), sort_keys=True)}")
    _configure_waymax_runtime(args)
    cfg = load_config(args.label_config, args.data_config, args.eval_config)
    if args.mode == "learned_offline":
        if not args.cache_dir:
            raise ValueError("--mode learned_offline requires --cache-dir")
        payload = learned_offline_rule_eval(args, cfg)
    else:
        payload = waymax_rule_eval(args, cfg)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_safe)
    _log(f"wrote output {out}")
    print(json.dumps(payload, indent=2, default=_json_safe), flush=True)


if __name__ == "__main__":
    main()
