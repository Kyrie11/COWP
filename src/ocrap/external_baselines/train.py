from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ocrap.data.serialization import ensure_dir, write_json
from ocrap.external_baselines.data import ExternalGroupDataset, use_teacher_branch_context
from ocrap.external_baselines.models import build_model_from_cfg
from ocrap.utils.seed import seed_everything

try:  # tqdm is optional but strongly preferred on training machines.
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _distributed_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def _setup_distributed(cfg: dict[str, Any]) -> tuple[bool, int, int, int]:
    bcfg = cfg.get("external_baselines", {}) if isinstance(cfg.get("external_baselines", {}), dict) else {}
    tcfg = bcfg.get("training", {}) if isinstance(bcfg.get("training", {}), dict) else {}
    requested = str(tcfg.get("distributed", "auto")).lower()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = (requested in {"true", "1", "yes"}) or (requested == "auto" and world_size > 1)
    if not use_ddp:
        return False, 0, 0, 1
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training was requested but CUDA is not available")
    if not dist.is_initialized():
        dist.init_process_group(backend=str(tcfg.get("dist_backend", "nccl")), init_method="env://")
    rank = int(dist.get_rank())
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1)))
    world_size = int(dist.get_world_size())
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def _cleanup_distributed() -> None:
    if _distributed_available():
        dist.barrier()
        dist.destroy_process_group()


def _device(cfg: dict[str, Any], *, use_ddp: bool = False, local_rank: int = 0) -> torch.device:
    bcfg = cfg.get("external_baselines", {}) if isinstance(cfg.get("external_baselines", {}), dict) else {}
    tcfg = bcfg.get("training", {}) if isinstance(bcfg.get("training", {}), dict) else {}
    requested = str(tcfg.get("device", (cfg.get("training", {}) or {}).get("device", "auto")))
    if use_ddp:
        return torch.device(f"cuda:{int(local_rank)}")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool() & torch.isfinite(target)
    if not bool(mask.any()):
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[mask], target[mask])


def _reduce_totals(totals: dict[str, float], n: int, device: torch.device) -> tuple[dict[str, float], int]:
    if not _distributed_available():
        return totals, n
    keys = sorted(totals)
    vals = torch.tensor([totals[k] for k in keys] + [float(n)], dtype=torch.float64, device=device)
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    return {k: float(vals[i].item()) for i, k in enumerate(keys)}, int(vals[-1].item())


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _forward_model(model: torch.nn.Module, batch: dict[str, torch.Tensor], cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    deployable_only = not use_teacher_branch_context(cfg)
    return model(
        batch["x"].float(),
        batch["mask"].bool(),
        # Keep the branch encoder active on a fixed neutral context when the
        # teacher tensors are unavailable. This avoids both label leakage and
        # DDP unused-parameter failures.
        branch_margins=None if deployable_only else batch.get("branch_margins", None),
        root_features=None if deployable_only else batch.get("root_features", None),
        root_probs=None if deployable_only else batch.get("root_probs", None),
        root_valid=None if deployable_only else batch.get("root_valid", None),
        option_valid=batch.get("option_valid", None),
        topology_features=batch.get("topology_features", None),
        topology_mask=batch.get("topology_mask", None),
        ego_history=batch.get("ego_history", None),
        neighbor_history=batch.get("neighbor_history", None),
        neighbor_valid=batch.get("neighbor_valid", None),
        prefix_traj=batch.get("prefix_traj", None),
        prefix_valid=batch.get("prefix_valid", None),
        actor_topology_features=batch.get("actor_topology_features", None),
        actor_topology_mask=batch.get("actor_topology_mask", None),
        map_topology_features=batch.get("map_topology_features", None),
        map_topology_mask=batch.get("map_topology_mask", None),
    )




def _gameformer_traj_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Best-of-M Gaussian trajectory loss with deep level-k supervision."""
    traj_levels = out.get("gameformer_level_trajs")
    score_levels = out.get("gameformer_level_scores")
    if not isinstance(traj_levels, list) or not isinstance(score_levels, list) or "prefix_traj" not in batch:
        return out["logits"].sum() * 0.0
    gt = batch["prefix_traj"].float()
    valid = batch.get("prefix_valid", torch.ones_like(gt[..., 0])).bool() & batch["mask"].bool().unsqueeze(-1)
    losses = []
    for traj, scores in zip(traj_levels, score_levels):
        pred = traj[..., :2]
        log_sigma = traj[..., 2:4].clamp(-5.0, 3.0)
        B, N, M, T, _ = pred.shape
        target = gt[:, :, None, :T, :]
        mask = valid[:, :, None, :T]
        inv_var = torch.exp(-2.0 * log_sigma)
        point_nll = 0.5 * ((pred - target) ** 2 * inv_var).sum(dim=-1) + log_sigma.sum(dim=-1)
        mode_nll = (point_nll * mask.float()).sum(dim=-1) / mask.float().sum(dim=-1).clamp_min(1.0)
        best = mode_nll.argmin(dim=-1)
        bidx = torch.arange(B, device=pred.device)[:, None]
        nidx = torch.arange(N, device=pred.device)[None, :]
        best_nll = mode_nll[bidx, nidx, best]
        cand_mask = batch["mask"].bool()
        reg = best_nll[cand_mask].mean() if bool(cand_mask.any()) else pred.sum() * 0.0
        flat_mask = cand_mask.reshape(-1)
        cls = F.cross_entropy(scores.reshape(B * N, M)[flat_mask], best.reshape(-1)[flat_mask]) if bool(flat_mask.any()) else pred.sum() * 0.0
        losses.append(reg + 0.25 * cls)
    return torch.stack(losses).mean() if losses else out["logits"].sum() * 0.0


def _focal_topk_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, *, top_k_ratio: float = 0.25) -> torch.Tensor:
    # logits [B,N,K,1] or [B,N,K]; target/valid [B,N,K]
    if logits.dim() == target.dim() + 1:
        logits = logits.squeeze(-1)
    target = target.float()
    valid = valid.bool()
    if not bool(valid.any()):
        return logits.sum() * 0.0
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1.0 - p) * (1.0 - target)
    loss = ce * (1.0 - pt).pow(2.0)
    loss = loss.masked_fill(~valid, 0.0)
    B, N, K = loss.shape
    flat = loss.reshape(B, N * K)
    vflat = valid.reshape(B, N * K)
    k = max(1, int(float(top_k_ratio) * max(N * K, 1)))
    # Follow BeTop's hard-topology mining: sort all valid topology terms and keep the hardest top-k.
    flat = flat.masked_fill(~vflat, -1.0)
    vals = torch.topk(flat, k=min(k, flat.shape[-1]), dim=-1).values
    vals = vals.clamp_min(0.0)
    denom = (vals > 0).float().sum(dim=-1).clamp_min(1.0)
    return (vals.sum(dim=-1) / denom).mean()

def _loss_dict(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    bcfg = cfg.get("external_baselines", {}) if isinstance(cfg.get("external_baselines", {}), dict) else {}
    lw = bcfg.get("loss_weights", {}) if isinstance(bcfg.get("loss_weights", {}), dict) else {}
    mask = batch["mask"].bool()
    losses: dict[str, torch.Tensor] = {}
    losses["loss_policy"] = F.cross_entropy(out["logits"], batch["target_index"].long())
    # Deep supervision for GameFormer level-k decoders.  The original method
    # supervises every reasoning level; this keeps the gradient signal active at
    # level-0 and intermediate response decoders.
    level_logits = out.get("level_logits")
    if isinstance(level_logits, list) and len(level_logits) > 1:
        lev = [F.cross_entropy(x, batch["target_index"].long()) for x in level_logits]
        losses["loss_levelk"] = torch.stack(lev).mean()
        # Response consistency: later levels should not collapse to a noisier
        # distribution than the previous level.  KL is computed with stop-grad on
        # level k-1, mirroring the hierarchical response interpretation.
        kls = []
        for prev, cur in zip(level_logits[:-1], level_logits[1:]):
            p_prev = F.softmax(prev.detach(), dim=-1)
            log_cur = F.log_softmax(cur, dim=-1)
            kls.append(F.kl_div(log_cur, p_prev, reduction="batchmean"))
        losses["loss_level_response"] = torch.stack(kls).mean() if kls else out["logits"].sum() * 0.0
    else:
        losses["loss_levelk"] = out["logits"].sum() * 0.0
        losses["loss_level_response"] = out["logits"].sum() * 0.0
    losses["loss_utility"] = _masked_mse(out["utility"], batch["utility"].float(), mask)
    losses["loss_hard"] = _masked_mse(out["hard"], batch["hard"].float(), mask)
    losses["loss_harm"] = _masked_mse(out["harm"], batch["harm"].float(), mask)
    losses["loss_oracle_rec"] = _masked_mse(out["r_orc"], batch["r_orc"].float(), mask)
    losses["loss_deploy_rec"] = _masked_mse(out["r_dep"], batch["r_dep"].float(), mask)
    losses["loss_gameformer_traj"] = _gameformer_traj_loss(out, batch)
    topo_losses = []

    def _as_level_list(value: Any) -> list[torch.Tensor]:
        if isinstance(value, list):
            return [v for v in value if torch.is_tensor(v)]
        return [value] if torch.is_tensor(value) else []

    # BeTopNet-style topology reasoning has a topology decoder at every fusion
    # layer.  In the source-adapted model the earlier decoder logits are returned
    # as *_levels.  They must all participate in the loss; otherwise DDP sees the
    # early decoder parameters as unused and fails on the next iteration.
    actor_levels = _as_level_list(out.get("actor_topo_logits_levels")) or _as_level_list(out.get("actor_topo_logits"))
    map_levels = _as_level_list(out.get("map_topo_logits_levels")) or _as_level_list(out.get("map_topo_logits"))
    if actor_levels and "actor_topology_target" in batch and "actor_topology_mask" in batch:
        target = batch["actor_topology_target"].float()
        valid = batch["actor_topology_mask"].bool() & mask.unsqueeze(-1)
        for z in actor_levels:
            topo_losses.append(_focal_topk_loss(z, target, valid, top_k_ratio=float(lw.get("topology_topk_ratio", 0.25))))
    if map_levels and "map_topology_target" in batch and "map_topology_mask" in batch:
        target = batch["map_topology_target"].float()
        valid = batch["map_topology_mask"].bool() & mask.unsqueeze(-1)
        for z in map_levels:
            topo_losses.append(_focal_topk_loss(z, target, valid, top_k_ratio=float(lw.get("topology_topk_ratio", 0.25))))
    if topo_losses:
        losses["loss_topology"] = torch.stack(topo_losses).mean()
    else:
        topo_logits = out.get("topology_logits")
        if topo_logits is not None and "topology_target" in batch and "topology_mask" in batch and topo_logits.shape[-1] > 1:
            topo_mask = batch["topology_mask"].bool() & mask.unsqueeze(-1)
            losses["loss_topology"] = F.cross_entropy(topo_logits[topo_mask], batch["topology_target"].long()[topo_mask]) if bool(topo_mask.any()) else out["logits"].sum() * 0.0
        else:
            losses["loss_topology"] = out["logits"].sum() * 0.0
    losses["loss"] = (
        float(lw.get("policy", 1.0)) * losses["loss_policy"]
        + float(lw.get("levelk", 0.35)) * losses["loss_levelk"]
        + float(lw.get("level_response", 0.10)) * losses["loss_level_response"]
        + float(lw.get("topology", 0.0)) * losses["loss_topology"]
        + float(lw.get("gameformer_traj", 0.25)) * losses["loss_gameformer_traj"]
        + float(lw.get("utility", 0.10)) * losses["loss_utility"]
        + float(lw.get("hard", 0.50)) * losses["loss_hard"]
        + float(lw.get("harm", 0.25)) * losses["loss_harm"]
        + float(lw.get("oracle_rec", 0.50)) * losses["loss_oracle_rec"]
        + float(lw.get("deploy_rec", 0.25)) * losses["loss_deploy_rec"]
    )
    return losses


def _epoch(model, loader, opt, device, cfg: dict[str, Any], train: bool, *, rank: int = 0, epoch: int = 0) -> dict[str, float]:
    model.train(train)
    bcfg = cfg.get("external_baselines", {}) if isinstance(cfg.get("external_baselines", {}), dict) else {}
    tcfg = bcfg.get("training", {}) if isinstance(bcfg.get("training", {}), dict) else {}
    totals: dict[str, float] = {}
    n = 0
    iterator = loader
    show = bool(tcfg.get("tqdm", True)) and rank == 0 and tqdm is not None
    if show:
        iterator = tqdm(loader, desc=("train" if train else "val") + f" ep{epoch:03d}", leave=False, dynamic_ncols=True)
    for batch in iterator:
        batch = _batch_to_device(batch, device)
        with torch.set_grad_enabled(train):
            out = _forward_model(model, batch, cfg)
            losses = _loss_dict(out, batch, cfg)
            loss = losses["loss"]
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.get("grad_clip", 5.0)))
                opt.step()
        bs = int(batch["x"].shape[0])
        n += bs
        for name, val in losses.items():
            totals[name] = totals.get(name, 0.0) + float(val.detach().cpu()) * bs
        with torch.no_grad():
            pred = torch.argmax(out["logits"], dim=-1)
            acc = (pred == batch["target_index"]).float().mean()
            totals["target_acc"] = totals.get("target_acc", 0.0) + float(acc.detach().cpu()) * bs
        if show:
            iterator.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")
    totals, n = _reduce_totals(totals, n, device)
    return {k: v / max(n, 1) for k, v in totals.items()}


def _model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    module = model.module if isinstance(model, DDP) else model
    return module.state_dict()


def train_external_baseline(dataset: str, output: str, cfg: dict[str, Any], *, val_dataset: str | None = None, baseline: str | None = None) -> dict[str, Any]:
    deterministic = {
        "marc", "marc_lite", "marc_contingency",
        "racp", "racp_lite", "risk_aware_contingency",
        "expected_risk", "expected_risk_filter", "expected_risk_planner",
        "cvar_risk", "cvar_risk_filter", "cvar_planner",
        "dro_cvar", "dro_cvar_filter", "dro_cvar_safety_filter", "dr_cvar_filter",
        "predictive_safety_filter", "psf", "cbf_backup_filter", "predictive_cbf_backup", "backup_cbf_filter",
        "oracle_filter", "oracle_recovery_filter", "branchwise_oracle_filter", "oracle_branchwise_recovery",
        "postimpact_mpc", "postimpact_mpc_lite", "post_impact_mpc_lite", "postimpact_mpc_paper", "integrated_postimpact_mpc",
        "post_crash_braking", "post_crash_braking_rule", "stable_stop", "stable_stop_rule", "postcrash_stable_stop",
        "post_collision_restoration", "trajectory_restoration", "post_collision_trajectory_restoration", "post_collision_restoration_heuristic", "ackermann_restoration",
        "severity_minimization", "severity_minimization_planner", "unavoidable_collision_planner", "crash_mitigation_planner", "uc_severity_planner",
    }
    bcfg0 = cfg.setdefault("external_baselines", {})
    if baseline:
        bcfg0["baseline"] = baseline
    baseline_name0 = str(bcfg0.get("baseline", "route_bc_lite")).lower()
    if baseline_name0 in deterministic:
        out_dir = ensure_dir(output)
        # These papers define optimization/sampling/filter baselines rather than
        # learned policy networks.  "Training" registers the config and validates
        # that the OC-RAP grouped dataset can be read; thresholds are explicit in
        # the YAML so there is no hidden fitting to test labels.
        tcfg0 = bcfg0.get("training", {}) if isinstance(bcfg0.get("training", {}), dict) else {}
        validate_dataset = bool(tcfg0.get("validate_dataset", True))
        train_ds = ExternalGroupDataset(dataset, cfg, split="train", baseline=baseline_name0) if validate_dataset else None
        summary = {
            "baseline": baseline_name0,
            "training_mode": "non_learning_filter_or_planner",
            "dataset_validated": bool(validate_dataset),
            "num_train_groups": (len(train_ds) if train_ds is not None else None),
            "feature_dim": (int(train_ds.feature_dim) if train_ds is not None else None),
            "max_candidates": (int(train_ds.max_candidates) if train_ds is not None else int(bcfg0.get("max_candidates", 0))),
            "notes": "No neural weights are trained for MARC/RACP/risk/PSF/oracle filters; the paper core is an optimization or safety-filter rule over the candidate lattice.",
            "cfg": cfg,
        }
        write_json(summary, out_dir / "train_summary.json")
        return summary
    seed_everything(int(cfg.get("seed", 7)))
    use_ddp, rank, local_rank, world_size = _setup_distributed(cfg)
    try:
        bcfg = cfg.setdefault("external_baselines", {})
        if baseline:
            bcfg["baseline"] = baseline
        baseline_name = str(bcfg.get("baseline", "route_bc_lite"))
        out_dir = ensure_dir(output) if rank == 0 else Path(output)
        train_ds = ExternalGroupDataset(dataset, cfg, split="train", baseline=baseline_name)
        if val_dataset:
            val_ds = ExternalGroupDataset(val_dataset, cfg, split="val", baseline=baseline_name)
        else:
            try:
                val_ds = ExternalGroupDataset(dataset, cfg, split="val", baseline=baseline_name)
            except Exception:
                val_ds = train_ds
        tcfg = bcfg.get("training", {}) if isinstance(bcfg.get("training", {}), dict) else {}
        device = _device(cfg, use_ddp=use_ddp, local_rank=local_rank)
        model = build_model_from_cfg(train_ds.feature_dim, cfg).to(device)
        if use_ddp:
            fup = tcfg.get("find_unused_parameters", "auto")
            if isinstance(fup, str):
                fup_s = fup.lower()
                find_unused = baseline_name.lower() in {"betop", "betop_lite", "betopnet", "betopnet_lite"} if fup_s == "auto" else fup_s in {"1", "true", "yes"}
            else:
                find_unused = bool(fup)
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused)
        opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg.get("lr", 2.0e-4)), weight_decay=float(tcfg.get("weight_decay", 1.0e-4)))
        batch_size = int(tcfg.get("batch_size", 32))
        num_workers = int(tcfg.get("num_workers", 0))
        pin_memory = bool(tcfg.get("pin_memory", True)) and torch.cuda.is_available()
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False) if use_ddp else None
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if use_ddp else None
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=_collate, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, sampler=val_sampler, collate_fn=_collate, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))
        epochs = int(tcfg.get("epochs", 10))
        best_val = float("inf")
        best_epoch = 0
        history = []
        t0 = perf_counter()
        for ep in range(1, epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(ep)
            tr = _epoch(model, train_loader, opt, device, cfg, train=True, rank=rank, epoch=ep)
            with torch.no_grad():
                va = _epoch(model, val_loader, None, device, cfg, train=False, rank=rank, epoch=ep)
            row = {"epoch": ep, "train": tr, "val": va}
            if rank == 0:
                history.append(row)
                ckpt = {
                    "baseline": baseline_name,
                    "cfg": cfg,
                    "input_dim": int(train_ds.feature_dim),
                    "max_candidates": int(train_ds.max_candidates),
                    "num_roots": int(train_ds.num_roots),
                    "num_options": int(train_ds.num_options),
                    "root_feature_dim": int(train_ds.root_feature_dim),
                    "num_topology_agents": int(getattr(train_ds, "num_topology_agents", 0)),
                    "topology_feature_dim": int(getattr(train_ds, "topology_feature_dim", 0)),
                    "actor_topology_feature_dim": int(getattr(train_ds, "actor_topology_feature_dim", 0)),
                    "num_topology_map": int(getattr(train_ds, "num_topology_map", 0)),
                    "map_topology_feature_dim": int(getattr(train_ds, "map_topology_feature_dim", 0)),
                    "history_len": int(getattr(train_ds, "history_len", 0)),
                    "neighbors_to_predict": int(getattr(train_ds, "neighbors_to_predict", 0)),
                    "future_len": int(getattr(train_ds, "future_len", 0)),
                    "input_contract": {
                        "version": 2,
                        "use_teacher_branch_context": bool(use_teacher_branch_context(cfg)),
                        "deployable_feature_only": bool(not use_teacher_branch_context(cfg)),
                        "coordinate_frame": "current_ego_relative",
                        "selection_supervision": str((cfg.get("external_baselines", {}) or {}).get("supervision_target", "logged_nominal")),
                    },
                    "model_state": _model_state(model),
                    "epoch": int(ep),
                    "val_loss": float(va.get("loss", 0.0)),
                    "world_size": int(world_size),
                }
                torch.save(ckpt, out_dir / "latest.pt")
                if va.get("loss", float("inf")) <= best_val:
                    best_val = float(va.get("loss", float("inf")))
                    best_epoch = ep
                    torch.save(ckpt, out_dir / "best.pt")
                print({"event": "external_baseline_epoch", "baseline": baseline_name, "epoch": ep, "world_size": world_size, "train_loss": tr.get("loss"), "val_loss": va.get("loss"), "target_acc": va.get("target_acc")}, flush=True)
        summary = {
            "baseline": baseline_name,
            "num_train_groups": len(train_ds),
            "num_val_groups": len(val_ds),
            "feature_dim": int(train_ds.feature_dim),
            "max_candidates": int(train_ds.max_candidates),
            "num_roots": int(train_ds.num_roots),
            "num_options": int(train_ds.num_options),
            "root_feature_dim": int(train_ds.root_feature_dim),
            "num_topology_agents": int(getattr(train_ds, "num_topology_agents", 0)),
            "topology_feature_dim": int(getattr(train_ds, "topology_feature_dim", 0)),
            "actor_topology_feature_dim": int(getattr(train_ds, "actor_topology_feature_dim", 0)),
            "num_topology_map": int(getattr(train_ds, "num_topology_map", 0)),
            "map_topology_feature_dim": int(getattr(train_ds, "map_topology_feature_dim", 0)),
            "history_len": int(getattr(train_ds, "history_len", 0)),
            "neighbors_to_predict": int(getattr(train_ds, "neighbors_to_predict", 0)),
            "future_len": int(getattr(train_ds, "future_len", 0)),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "seconds": float(perf_counter() - t0),
            "world_size": int(world_size),
            "history": history,
        }
        if rank == 0:
            write_json(summary, out_dir / "train_summary.json")
        return summary
    finally:
        _cleanup_distributed()
