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
import torch
from torch.utils.data import DataLoader

from cowp.core.config import load_config
from cowp.data.dataset import collate_torch
from cowp.external_baselines.adapters import (
    ExternalCOWPDataset,
    candidate_geometry_finite,
    external_map_topology_report,
    label_from_batch_item,
    make_external_batch,
)
from cowp.external_baselines.waymax_policy import build_external_model_from_checkpoint, make_external_waymax_policy
from cowp.external_baselines.reference_metadata import baseline_reference_metadata
from cowp.utils.progress import tqdm_iter
from cowp.utils.dataloader_runtime import configure_dataloader_runtime
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


def _validate_map_topology_contract(
    dataset: ExternalCOWPDataset,
    *,
    baseline: str,
    allow_legacy_flat_map: bool,
) -> None:
    """Ensure offline evaluation exercises the same V6 map contract as training."""
    size = len(dataset)
    if size <= 0:
        raise RuntimeError(f"Empty learned-offline dataset for external baseline {baseline}.")
    indices = sorted({0, size // 2, size - 1})
    reports: list[dict[str, object]] = []
    bad: list[tuple[int, dict[str, object]]] = []
    for idx in indices:
        report = external_map_topology_report(dataset[idx])
        reports.append(report)
        ready = bool(
            report.get("has_xy")
            and report.get("has_id")
            and report.get("has_type")
            and report.get("has_dir")
            and report.get("has_valid")
            and report.get("aligned")
        )
        if not ready:
            bad.append((idx, report))
    if bad and not allow_legacy_flat_map:
        raise RuntimeError(
            f"V6 learned-offline map-topology contract failed for baseline={baseline}: {bad}. "
            "Evaluation must use aligned WOMD roadgraph xyz(or x,y), id, type, dir, and valid so "
            "it cannot silently revert to the V5 flat-map representation."
        )
    mode = "legacy_flat_fallback_allowed" if bad else "womd_feature_id_topology"
    _log(
        f"map topology contract baseline={baseline} split=learned_offline mode={mode} "
        f"sample_indices={indices} reports={json.dumps(reports, sort_keys=True)}"
    )


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

def _external_policy_runtime_summary(rollouts: list[dict]) -> dict[str, float]:
    """Aggregate only diagnostics actually produced by an external policy.

    External baselines do not emit COWP witness/OPR/burden predictions online.
    Reporting the generic COWP diagnostic adapter would silently turn missing
    fields into FSR=0/OPR=1, so we intentionally expose only execution/runtime
    quantities and mark mechanism evidence unavailable.
    """
    rows: list[dict] = []
    for item in rollouts:
        rows.extend(item.get("policy_diagnostics", []) or [])
    if not rows:
        return {
            "ClosedLoopMechanismGroundTruthAvailable": 0.0,
            "ClosedLoopMechanismProxyAvailable": 0.0,
            "ClosedLoopPolicySteps": 0.0,
        }
    out: dict[str, float] = {
        "ClosedLoopMechanismGroundTruthAvailable": 0.0,
        "ClosedLoopMechanismProxyAvailable": 0.0,
        "ClosedLoopPolicySteps": float(len(rows)),
        "ClosedLoopFallbackStepRate": float(np.mean([bool(r.get("fallback", r.get("fallback_used", False))) for r in rows])),
    }
    for key in ("valid_candidates", "conventional_candidates", "selected_score"):
        vals = np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[f"ClosedLoopMean/{key}"] = float(vals.mean())
    timing_keys = sorted({str(k) for r in rows for k in r if str(k).startswith("timing_ms/")})
    for key in timing_keys:
        vals = np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[f"ClosedLoopMean/{key}"] = float(vals.mean())
            out[f"ClosedLoopP95/{key}"] = float(np.quantile(vals, 0.95))
    direct = np.asarray([str(r.get("execution_mode", "candidate")) == "direct" for r in rows], dtype=bool)
    out["ClosedLoopDirectExecutionStepRate"] = float(direct.mean())
    return out



def _external_scenario_rows(rollouts: list[dict]) -> list[dict[str, float]]:
    rows_out: list[dict[str, float]] = []
    for item in rollouts:
        rows = item.get("policy_diagnostics", []) or []
        steps = int(item.get("steps", len(rows)) or len(rows))
        fallback = [bool(r.get("fallback", r.get("fallback_used", False))) for r in rows]
        direct = [str(r.get("execution_mode", "candidate")) == "direct" for r in rows]
        rows_out.append({
            "steps": float(steps),
            "fallback_step_rate": float(np.mean(fallback)) if fallback else 0.0,
            "direct_execution_step_rate": float(np.mean(direct)) if direct else 0.0,
        })
    return rows_out

def learned_offline_eval(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    _log(f"loading checkpoint {args.checkpoint}")
    model, baseline, model_cfg, ckpt_args, device = build_external_model_from_checkpoint(args.checkpoint, cfg, args.device)
    max_neighbors = int(args.max_neighbors or ckpt_args.get("max_neighbors", 10))
    max_candidates = int(args.max_candidates or ckpt_args.get("max_candidates", cfg.get("limits", {}).get("max_candidates", 30)))
    future_len = int(args.future_len or ckpt_args.get("future_len", cfg.get("time", {}).get("future_steps", 80)))
    _log(f"checkpoint ready baseline={baseline} device={device} max_neighbors={max_neighbors} max_candidates={max_candidates} future_len={future_len}")
    _log(f"loading learned-offline dataset from {args.cache_dir}")
    ds = ExternalCOWPDataset(args.cache_dir, include_waymax_outcomes=True, baseline=baseline, purpose="audit")
    _log(f"learned-offline dataset ready scenes={len(ds)}")
    _validate_map_topology_contract(
        ds,
        baseline=baseline,
        allow_legacy_flat_map=bool(getattr(args, "allow_legacy_flat_map", False)),
    )
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
            ext = make_external_batch(
                batch, model_cfg, device=device, max_neighbors=max_neighbors, max_candidates=max_candidates,
                horizon=future_len, baseline=baseline, require_candidates=True, require_future=False,
            )
            if baseline == "gameformer":
                scores = model.score_candidates(ext.gameformer_inputs, ext.candidates, ext.candidate_valid)
            elif baseline == "dtpp":
                scores = model.score_candidates(ext.dtpp_inputs, ext.dtpp_candidate_tree, ext.candidate_valid, timesteps=future_len)
            elif baseline in {"pluto", "plant2"}:
                scores = model.score_candidates(ext.planner_inputs, ext.candidates, ext.candidate_valid)
            else:
                raise ValueError(baseline)
            # Never let NaN/Inf logits or malformed proposal geometry win argmax.
            candidate_finite = candidate_geometry_finite(ext.candidates)
            accept = ext.candidate_valid & candidate_finite & torch.isfinite(scores)
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
        "reference_metadata": baseline_reference_metadata(baseline),
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
        execution_mode=args.execution_mode,
        profile_timing=args.profile_policy_timing,
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
        scenario_ids=(Path(args.scenario_ids_file).read_text(encoding="utf-8").splitlines() if args.scenario_ids_file else None),
        tfexample_index_jsonl=args.tfexample_index_jsonl,
        standard_metric_names=({x.strip() for x in args.waymax_standard_metric_names.split(",") if x.strip()} if args.waymax_standard_metric_names else None),
    )
    _log(f"waymax {policy.baseline} rollout done episodes={len(rollouts)} seconds={time.time()-t0:.1f}")
    payload: dict[str, Any] = {
        "mode": "waymax",
        "baseline": policy.baseline,
        "method": f"external_{policy.baseline}",
        "reference_metadata": baseline_reference_metadata(policy.baseline),
        "checkpoint": args.checkpoint,
        "execution_mode": policy.execution_mode,
        "waymax_split": args.waymax_split,
        "scenario_ids_sha256": hashlib.sha256("\n".join(Path(args.scenario_ids_file).read_text(encoding="utf-8").splitlines()).encode("utf-8")).hexdigest() if args.scenario_ids_file else None,
        "scenario_ids_resolved": [str(x.get("scenario_id")) for x in rollouts if x.get("scenario_id") is not None],
        "scenario_diagnostics": _external_scenario_rows(rollouts),
        "tfexample_glob": args.tfexample_glob,
        "num_rollouts": len(rollouts),
        "steps": [int(x.get("steps", 0)) for x in rollouts],
        "external_policy_runtime_summary": _external_policy_runtime_summary(rollouts),
        "policy_diagnostic_summary": _external_policy_runtime_summary(rollouts),
        "closed_loop_mechanism_metric_status": {
            "available": False,
            "reason": "External planner rollout has no frozen counterfactual COWP auditor; run learned_offline audit for PBTR/FSR/OPR/BTE on cached labels.",
        },
    }
    if args.waymax_standard_metrics:
        payload["standard_metrics"] = [x.get("standard_metrics", {}) for x in rollouts]
        payload["standard_metric_summary"] = aggregate_waymax_standard_metrics(rollouts)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate learned external planning baselines on COWP learned-offline or real Waymax closed loop.")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--label-config", default="configs/label.yaml")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--mode", choices=["learned_offline", "waymax"], default="learned_offline")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--sharing-strategy", choices=["auto", "current", "file_descriptor", "file_system"], default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-neighbors", type=int, default=None)
    ap.add_argument("--max-candidates", type=int, default=None)
    ap.add_argument("--future-len", type=int, default=None)
    ap.add_argument(
        "--allow-legacy-flat-map",
        action="store_true",
        help="Debug compatibility only: permit learned-offline evaluation on caches missing WOMD roadgraph id/type/dir. Formal V6 evaluation should leave this disabled.",
    )
    ap.add_argument("--require-conventional-safe", action="store_true", help="Candidate-mode only: restrict scoring to COWP conventional-safe proposals when available.")
    ap.add_argument("--execution-mode", choices=["auto", "direct", "candidate"], default="auto", help="Waymax only. auto executes native direct heads for GameFormer/PLUTO/PlanT2 and candidate-tree scoring for DTPP.")
    ap.add_argument("--profile-policy-timing", action="store_true", help="Synchronize accelerator stages and write per-step planner timing diagnostics. Use for short profiling runs only.")
    ap.add_argument("--dump-selections", action="store_true")
    ap.add_argument("--max-dump-rows", type=int, default=50)
    ap.add_argument("--waymax-split", choices=["training", "validation", "testing"], default="validation")
    ap.add_argument("--tfexample-glob", default=None)
    ap.add_argument("--tfexample-index-jsonl", default=None, help="Optional exact scenario-id to TFExample shard index.")
    ap.add_argument("--scenario-ids-file", default=None, help="Exact scenario-ID manifest; every requested ID must resolve.")
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
    ap.add_argument("--waymax-standard-metric-names", default="OverlapMetric,OffroadMetric,WrongWayMetric,ProgressionMetric,OffRouteMetric,KinematicsInfeasibilityMetric,LogDivergenceMetric")
    ap.add_argument("--keep-rollout-state", action="store_true")
    ap.add_argument("--clear-accelerator-cache", action="store_true")
    ap.add_argument("--output", required=True)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--log-every", type=int, default=int(os.environ.get("LOG_EVERY", "25")))
    args = ap.parse_args()
    loader_runtime = configure_dataloader_runtime(args.sharing_strategy)
    _log(f"external eval entry mode={args.mode} pid={os.getpid()} python={sys.executable} torch={torch.__version__}")
    _log(f"DataLoader IPC runtime={json.dumps(loader_runtime, sort_keys=True)}")
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
