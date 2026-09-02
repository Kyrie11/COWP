from __future__ import annotations

import argparse
import hashlib
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
from cowp.external_baselines.reference_metadata import baseline_reference_metadata
from cowp.external_baselines.rule_based import RULE_BASELINES, select_rule_indices
from cowp.external_baselines.rule_waymax_policy import make_rule_waymax_policy
from cowp.utils.progress import tqdm_iter
from cowp.utils.dataloader_runtime import configure_dataloader_runtime
from cowp.waymax_eval.metrics_cowp import (
    physical_failure_attribution_summary,
    policy_diagnostic_episode_summary,
    policy_diagnostic_scenario_rows,
    policy_diagnostic_summary,
)
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


def _parse_metric_names(value: str | None):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"all", "default"}:
        return None
    if text.lower() == "none":
        return []
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


def _scenario_ids_requested_on_shard(scenario_ids: list[str] | None, num_shards: int, shard_index: int) -> int | None:
    if scenario_ids is None:
        return None
    n = max(int(num_shards), 1)
    s = int(shard_index) % n
    return int(sum(1 for i in range(len(scenario_ids)) if (i % n) == s))


def _reference_family(method: str) -> str:
    table = {
        "idm_lattice": "Treiber-Hennecke-Helbing IDM longitudinal car-following + local lattice candidate selection",
        "frenet_optimal": "Werling-Ziegler-Kammel-Thrun Frenet-frame optimal trajectory cost",
        "state_lattice": "Pivtoraiko-Kelly state lattice motion-primitive edge cost + progress heuristic",
        "pdm_closed": "PDM-Closed-style predictive safety/progress/comfort rule scoring over the WOMD/COWP local proposal bank",
    }
    return table.get(method, method)


def _augment_policy_summary_with_external_rates(summary: dict[str, float], rollouts: list[dict]) -> dict[str, float]:
    rows: list[dict] = []
    for item in rollouts:
        rows.extend(item.get("policy_diagnostics", []) or [])
    if not rows:
        return summary
    direct = [1.0 if str(r.get("execution_trajectory_source", "candidate")) == "direct" else 0.0 for r in rows]
    if direct:
        summary["ClosedLoopDirectExecutionStepRate"] = float(sum(direct) / len(direct))
    return summary


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
    scenario_ids = _load_scenario_ids_file(args.scenario_ids_file)
    metric_names = _parse_metric_names(args.waymax_standard_metric_names)
    _log(
        f"rule waymax {policy.baseline} start split={args.waymax_split} tfexample_glob={args.tfexample_glob} "
        f"num_scenarios={args.num_scenarios} exact_ids={len(scenario_ids) if scenario_ids is not None else 0} "
        f"shard={args.shard_index}/{args.num_shards} horizon={horizon} progress={not args.no_progress} standard_metrics={args.waymax_standard_metrics}"
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
        reuse_env=bool(args.reuse_waymax_env),
        prefilter_shards=bool(args.prefilter_waymax_shards),
        jit_env=bool(args.jit_waymax_env),
        jit_standard_metrics=bool(args.jit_waymax_metrics),
        status_every=int(args.status_every),
        standard_metric_names=metric_names,
        scenario_ids=scenario_ids,
        tfexample_index_jsonl=args.tfexample_index_jsonl,
        profile_runtime=bool(args.profile_waymax_runtime),
        profile_runtime_sync=bool(args.profile_waymax_sync),
    )
    _log(f"rule waymax {policy.baseline} rollout done episodes={len(rollouts)} seconds={time.time()-t0:.1f}")
    policy_summary = _augment_policy_summary_with_external_rates(policy_diagnostic_summary(rollouts), rollouts)
    payload: dict[str, Any] = {
        "mode": "waymax",
        "baseline": policy.baseline,
        "method": policy.baseline,
        "reference_metadata": baseline_reference_metadata(policy.baseline),
        "reference_family": _reference_family(policy.baseline),
        "waymax_split": args.waymax_split,
        "tfexample_glob": args.tfexample_glob,
        "tfexample_index_jsonl": args.tfexample_index_jsonl,
        "scenario_ids_file": args.scenario_ids_file,
        "scenario_ids_requested": int(len(scenario_ids)) if scenario_ids is not None else None,
        "scenario_ids_requested_on_shard": _scenario_ids_requested_on_shard(scenario_ids, args.num_shards, args.shard_index),
        "scenario_ids_sha256": hashlib.sha256("\n".join(scenario_ids).encode("utf-8")).hexdigest() if scenario_ids is not None else None,
        "scenario_ids_resolved": [str(x.get("scenario_id")) for x in rollouts if x.get("scenario_id") is not None],
        "waymax_standard_metric_names": sorted(metric_names) if isinstance(metric_names, set) else args.waymax_standard_metric_names,
        "reuse_waymax_env": bool(args.reuse_waymax_env),
        "prefilter_waymax_shards": bool(args.prefilter_waymax_shards),
        "jit_waymax_env": bool(args.jit_waymax_env),
        "jit_waymax_metrics": bool(args.jit_waymax_metrics),
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "num_rollouts": len(rollouts),
        "steps": [int(x.get("steps", 0)) for x in rollouts],
        "external_policy_runtime_summary": policy_summary,
        "policy_diagnostic_summary": policy_summary,
        "closed_loop_cowp_metric_summary": policy_diagnostic_episode_summary(rollouts),
        "scenario_diagnostics": policy_diagnostic_scenario_rows(rollouts),
        "physical_failure_attribution_summary": physical_failure_attribution_summary(rollouts),
        "closed_loop_mechanism_metric_status": {
            "available": False,
            "reason": "Rule/PDM external rollout has no frozen counterfactual COWP auditor; run learned_offline audit for PBTR/FSR/OPR/BTE on cached labels.",
        },
        "profile_waymax_runtime": bool(args.profile_waymax_runtime),
        "profile_waymax_sync": bool(args.profile_waymax_sync),
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
    ap.add_argument("--sharing-strategy", choices=["auto", "current", "file_descriptor", "file_system"], default=None)
    ap.add_argument("--no-conventional-filter", action="store_true", help="Allow all valid candidates instead of filtering to conventional-safe candidates.")
    ap.add_argument("--dump-selections", action="store_true")
    ap.add_argument("--max-dump-rows", type=int, default=50)
    ap.add_argument("--waymax-split", choices=["training", "validation", "testing"], default="validation")
    ap.add_argument("--tfexample-glob", default=None)
    ap.add_argument("--tfexample-index-jsonl", default=None)
    ap.add_argument("--scenario-ids-file", default=None)
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
    ap.add_argument("--waymax-standard-metric-names", default=None)
    ap.add_argument("--reuse-waymax-env", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--prefilter-waymax-shards", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--jit-waymax-env", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--jit-waymax-metrics", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--profile-waymax-runtime", action="store_true")
    ap.add_argument("--profile-waymax-sync", action="store_true")
    ap.add_argument("--profile-policy-timing", action="store_true")
    ap.add_argument("--status-every", type=int, default=10)
    ap.add_argument("--keep-rollout-state", action="store_true")
    ap.add_argument("--clear-accelerator-cache", action="store_true")
    ap.add_argument("--output", required=True)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--log-every", type=int, default=int(os.environ.get("LOG_EVERY", "25")))
    args = ap.parse_args()
    loader_runtime = configure_dataloader_runtime(args.sharing_strategy)
    _log(f"rule baseline eval entry mode={args.mode} baseline={args.baseline} pid={os.getpid()} python={sys.executable}")
    _log(f"DataLoader IPC runtime={json.dumps(loader_runtime, sort_keys=True)}")
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
