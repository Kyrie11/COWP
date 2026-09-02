from __future__ import annotations

import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from ocrap.data.serialization import ensure_dir, write_json
from ocrap.external_baselines.data import ExternalGroupDataset, use_teacher_branch_context
from ocrap.external_baselines.models import build_model_from_cfg
from ocrap.external_baselines.runtime import configure_cuda_runtime, resolve_amp_dtype
from ocrap.utils.seed import seed_everything

try:  # tqdm is optional but strongly preferred on training machines.
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None






def _nonfinite_gradient_report(model: torch.nn.Module, *, limit: int = 12) -> list[dict[str, Any]]:
    """Describe parameters with NaN/Inf gradients without mutating them."""
    module = model.module if isinstance(model, DDP) else model
    report: list[dict[str, Any]] = []
    for name, param in module.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        finite = torch.isfinite(grad.detach())
        if bool(finite.all()):
            continue
        bad = int((~finite).sum().item())
        total = int(finite.numel())
        finite_vals = grad.detach()[finite]
        max_abs = float(finite_vals.abs().max().item()) if finite_vals.numel() else None
        report.append({"parameter": name, "nonfinite": bad, "numel": total, "max_abs_finite": max_abs})
        if len(report) >= int(limit):
            break
    return report

def _stable_clip_grad_norm_(parameters, max_norm: float, eps: float = 1.0e-12) -> float:
    """Clip using a float64 norm so large finite float32 gradients do not overflow.

    ``torch.nn.utils.clip_grad_norm_`` can report ``inf`` when the individual
    gradients are finite but the float32 reduction overflows.  That turns the
    clipping coefficient into zero and silently erases the update.  Accumulate
    the norm in float64 and apply one finite coefficient in-place instead.
    """
    params = list(parameters)
    grads = [p.grad for p in params if getattr(p, "grad", None) is not None]
    if not grads:
        return 0.0
    total_sq = torch.zeros((), dtype=torch.float64, device=grads[0].device)
    for grad in grads:
        detached = grad.detach()
        if not bool(torch.isfinite(detached).all()):
            return float("inf")
        total_sq = total_sq + detached.double().square().sum()
    total_norm = torch.sqrt(total_sq)
    total = float(total_norm.item())
    limit = max(float(max_norm), 0.0)
    if total > limit and total > 0.0:
        coefficient = limit / (total + float(eps))
        for grad in grads:
            grad.mul_(coefficient)
    return total


class _DistributedEvalSampler(Sampler[int]):
    """Shard validation data without DistributedSampler's duplicate padding.

    Evaluation has no per-step gradient collective, so ranks may process one
    different number of examples. Final totals are reduced once in ``_epoch``.
    This preserves each validation group exactly once and therefore preserves
    best-checkpoint selection semantics.
    """

    def __init__(self, dataset, *, num_replicas: int, rank: int) -> None:
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        n = len(self.dataset) - self.rank
        return 0 if n <= 0 else (n + self.num_replicas - 1) // self.num_replicas


def _loader_kwargs(tcfg: dict[str, Any], *, num_workers: int, pin_memory: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(num_workers > 0 and tcfg.get("persistent_workers", True)),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(tcfg.get("prefetch_factor", 2))
    return kwargs


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
        return torch.nan_to_num(pred.float(), nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
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




def _zero_loss(out: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return a differentiable finite zero on the model device."""
    return torch.nan_to_num(out["logits"].float(), nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0


def _gameformer_traj_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Numerically-stable best-of-M Gaussian trajectory loss.

    GameFormer's heteroscedastic trajectory NLL is much more sensitive to AMP
    than the policy cross entropy.  In particular, exponentiation, squared
    residuals and masked reductions should not be performed in BF16/FP16.
    Keep the model forward under AMP, but explicitly evaluate this loss in
    float32.  Invalid/padded steps are removed with ``where`` before reduction
    so an ``inf`` at a masked position can never become ``0 * inf -> nan``.
    """
    traj_levels = out.get("gameformer_level_trajs")
    score_levels = out.get("gameformer_level_scores")
    if not isinstance(traj_levels, list) or not isinstance(score_levels, list) or "prefix_traj" not in batch:
        return _zero_loss(out)

    gt = batch["prefix_traj"].float()
    valid = batch.get("prefix_valid", torch.ones_like(gt[..., 0])).bool() & batch["mask"].bool().unsqueeze(-1)
    losses: list[torch.Tensor] = []
    # torch.autocast supports enabled=False on CUDA and CPU.  Do not use a
    # device-specific dtype here: every tensor is converted to float32 below.
    with torch.autocast(device_type=gt.device.type, enabled=False):
        gt32 = gt.float()
        cand_mask = batch["mask"].bool()
        for traj, scores in zip(traj_levels, score_levels):
            traj32 = traj.float()
            pred = traj32[..., :2]
            # The decoder already clamps log-sigma, but clamp again in FP32 at
            # the loss boundary so old checkpoints/alternate heads are safe.
            log_sigma = traj32[..., 2:4].clamp(-5.0, 3.0)
            B, N, M, T, _ = pred.shape
            target = gt32[:, :, None, :T, :]
            step_mask = valid[:, :, None, :T] & torch.isfinite(target).all(dim=-1)
            # Replace *both* sides before arithmetic.  Multiplying an invalid
            # NLL by zero after the fact is not safe because 0*inf is NaN, and
            # even a post-hoc ``where`` may leave NaN intermediates in backward.
            step_mask_xy = step_mask.unsqueeze(-1)
            safe_pred = torch.where(step_mask_xy, pred, torch.zeros_like(pred))
            safe_target = torch.where(step_mask_xy, target, torch.zeros_like(target))
            safe_log_sigma = torch.where(step_mask_xy, log_sigma, torch.zeros_like(log_sigma))
            inv_var = torch.exp((-2.0 * safe_log_sigma).clamp(min=-6.0, max=10.0))
            point_nll = 0.5 * ((safe_pred - safe_target).square() * inv_var).sum(dim=-1) + safe_log_sigma.sum(dim=-1)
            point_nll = torch.where(step_mask, point_nll, torch.zeros_like(point_nll))
            denom = step_mask.float().sum(dim=-1).clamp_min(1.0)
            mode_nll = point_nll.sum(dim=-1) / denom

            valid_candidate = cand_mask & valid[:, :, :T].any(dim=-1)
            if not bool(valid_candidate.any()):
                losses.append(_zero_loss(out))
                continue
            # Do not let padded candidates participate in best-mode mining.
            safe_mode_nll = mode_nll.masked_fill(~valid_candidate.unsqueeze(-1), float("inf"))
            best = safe_mode_nll.argmin(dim=-1)
            bidx = torch.arange(B, device=pred.device)[:, None]
            nidx = torch.arange(N, device=pred.device)[None, :]
            best_nll = mode_nll[bidx, nidx, best]
            reg = best_nll[valid_candidate].mean()
            flat_mask = valid_candidate.reshape(-1)
            cls = F.cross_entropy(scores.float().reshape(B * N, M)[flat_mask], best.reshape(-1)[flat_mask])
            losses.append(reg + 0.25 * cls)
    return torch.stack(losses).mean() if losses else _zero_loss(out)

def _focal_topk_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, *, top_k_ratio: float = 0.25) -> torch.Tensor:
    # logits [B,N,K,1] or [B,N,K]; target/valid [B,N,K]
    if logits.dim() == target.dim() + 1:
        logits = logits.squeeze(-1)
    target = target.float()
    valid = valid.bool()
    if not bool(valid.any()):
        return torch.nan_to_num(logits.float(), nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
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
    """Compute only losses whose configured weight is non-zero.

    This is both a correctness and a speed fix.  The old code evaluated every
    auxiliary loss and multiplied it by its weight afterwards.  IEEE floating
    point makes ``0 * NaN`` equal NaN, so an inactive auxiliary head could poison
    the total objective.  It also wasted substantial work for PlanTF/PDM-Hybrid,
    whose objectives are policy-only in the shipped configs.
    """
    bcfg = cfg.get("external_baselines", {}) if isinstance(cfg.get("external_baselines", {}), dict) else {}
    lw = bcfg.get("loss_weights", {}) if isinstance(bcfg.get("loss_weights", {}), dict) else {}
    mask = batch["mask"].bool()
    zero = _zero_loss(out)
    losses: dict[str, torch.Tensor] = {}

    defaults = {
        "policy": 1.0, "levelk": 0.35, "level_response": 0.10,
        "topology": 0.0, "gameformer_traj": 0.25, "pluto_contrastive": 0.0,
        "utility": 0.10, "hard": 0.50, "harm": 0.25,
        "oracle_rec": 0.50, "deploy_rec": 0.25,
    }
    weights = {name: float(lw.get(name, default)) for name, default in defaults.items()}
    active = lambda name: abs(weights[name]) > 0.0

    losses["loss_policy"] = (
        F.cross_entropy(out["logits"].float(), batch["target_index"].long()) if active("policy") else zero
    )

    level_logits = out.get("level_logits")
    if isinstance(level_logits, list) and len(level_logits) > 1 and (active("levelk") or active("level_response")):
        if active("levelk"):
            lev = [F.cross_entropy(x.float(), batch["target_index"].long()) for x in level_logits]
            losses["loss_levelk"] = torch.stack(lev).mean()
        else:
            losses["loss_levelk"] = zero
        if active("level_response"):
            kls = []
            for prev, cur in zip(level_logits[:-1], level_logits[1:]):
                # KL is another reduction that gains nothing from BF16.
                p_prev = F.softmax(prev.detach().float(), dim=-1)
                log_cur = F.log_softmax(cur.float(), dim=-1)
                kls.append(F.kl_div(log_cur, p_prev, reduction="batchmean"))
            losses["loss_level_response"] = torch.stack(kls).mean() if kls else zero
        else:
            losses["loss_level_response"] = zero
    else:
        losses["loss_levelk"] = zero
        losses["loss_level_response"] = zero

    losses["loss_utility"] = _masked_mse(out["utility"].float(), batch["utility"].float(), mask) if active("utility") else zero
    losses["loss_hard"] = _masked_mse(out["hard"].float(), batch["hard"].float(), mask) if active("hard") else zero
    losses["loss_harm"] = _masked_mse(out["harm"].float(), batch["harm"].float(), mask) if active("harm") else zero
    losses["loss_oracle_rec"] = _masked_mse(out["r_orc"].float(), batch["r_orc"].float(), mask) if active("oracle_rec") else zero
    losses["loss_deploy_rec"] = _masked_mse(out["r_dep"].float(), batch["r_dep"].float(), mask) if active("deploy_rec") else zero
    losses["loss_gameformer_traj"] = _gameformer_traj_loss(out, batch) if active("gameformer_traj") else zero

    pluto_logits = out.get("pluto_contrastive_logits")
    losses["loss_pluto_contrastive"] = (
        F.cross_entropy(pluto_logits.float(), batch["target_index"].long())
        if active("pluto_contrastive") and torch.is_tensor(pluto_logits) else zero
    )

    topo_losses: list[torch.Tensor] = []

    def _as_level_list(value: Any) -> list[torch.Tensor]:
        if isinstance(value, list):
            return [v for v in value if torch.is_tensor(v)]
        return [value] if torch.is_tensor(value) else []

    if active("topology"):
        actor_levels = _as_level_list(out.get("actor_topo_logits_levels")) or _as_level_list(out.get("actor_topo_logits"))
        map_levels = _as_level_list(out.get("map_topo_logits_levels")) or _as_level_list(out.get("map_topo_logits"))
        if actor_levels and "actor_topology_target" in batch and "actor_topology_mask" in batch:
            target = batch["actor_topology_target"].float()
            valid = batch["actor_topology_mask"].bool() & mask.unsqueeze(-1)
            for z in actor_levels:
                topo_losses.append(_focal_topk_loss(z.float(), target, valid, top_k_ratio=float(lw.get("topology_topk_ratio", 0.25))))
        if map_levels and "map_topology_target" in batch and "map_topology_mask" in batch:
            target = batch["map_topology_target"].float()
            valid = batch["map_topology_mask"].bool() & mask.unsqueeze(-1)
            for z in map_levels:
                topo_losses.append(_focal_topk_loss(z.float(), target, valid, top_k_ratio=float(lw.get("topology_topk_ratio", 0.25))))
        if topo_losses:
            losses["loss_topology"] = torch.stack(topo_losses).mean()
        else:
            topo_logits = out.get("topology_logits")
            if topo_logits is not None and "topology_target" in batch and "topology_mask" in batch and topo_logits.shape[-1] > 1:
                topo_mask = batch["topology_mask"].bool() & mask.unsqueeze(-1)
                losses["loss_topology"] = F.cross_entropy(topo_logits.float()[topo_mask], batch["topology_target"].long()[topo_mask]) if bool(topo_mask.any()) else zero
            else:
                losses["loss_topology"] = zero
    else:
        losses["loss_topology"] = zero

    key_map = {
        "policy": "loss_policy", "levelk": "loss_levelk", "level_response": "loss_level_response",
        "topology": "loss_topology", "gameformer_traj": "loss_gameformer_traj",
        "pluto_contrastive": "loss_pluto_contrastive", "utility": "loss_utility",
        "hard": "loss_hard", "harm": "loss_harm", "oracle_rec": "loss_oracle_rec",
        "deploy_rec": "loss_deploy_rec",
    }
    total = zero
    for name, key in key_map.items():
        if active(name):
            total = total + weights[name] * losses[key]
    # With DDP, inactive heads still need to appear in the autograd graph when
    # find_unused_parameters=False.  Attach a numerically sanitized zero only in
    # distributed mode; single-GPU training avoids this extra traversal entirely.
    if _distributed_available():
        graph_zero = zero
        def _attach(value: Any) -> None:
            nonlocal graph_zero
            if torch.is_tensor(value):
                graph_zero = graph_zero + torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
            elif isinstance(value, list):
                for item in value:
                    _attach(item)
        for value in out.values():
            _attach(value)
        total = total + graph_zero
    losses["loss"] = total
    return losses

def _epoch(model, loader, opt, device, cfg: dict[str, Any], train: bool, *, rank: int = 0, epoch: int = 0, scaler: Any | None = None, amp_dtype: torch.dtype | None = None) -> dict[str, float]:
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
        amp_enabled = bool(tcfg.get("amp", True)) and device.type == "cuda"
        effective_amp_dtype = amp_dtype or resolve_amp_dtype(tcfg, device)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, dtype=effective_amp_dtype, enabled=amp_enabled):
                out = _forward_model(model, batch, cfg)
                losses = _loss_dict(out, batch, cfg)
                loss = losses["loss"]
            bad_losses = {name: float(val.detach().float().cpu()) for name, val in losses.items() if not bool(torch.isfinite(val.detach()).all())}
            if bad_losses:
                raise FloatingPointError(
                    f"Non-finite external-baseline loss at epoch={epoch}, train={train}: {bad_losses}. "
                    "The optimizer step is intentionally aborted so NaNs cannot poison the checkpoint."
                )
            if train:
                opt.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    grad_norm = _stable_clip_grad_norm_(model.parameters(), float(tcfg.get("grad_clip", 5.0)))
                    if not math.isfinite(grad_norm):
                        bad_grads = _nonfinite_gradient_report(model)
                        opt.zero_grad(set_to_none=True)
                        raise FloatingPointError(
                            f"Non-finite gradient norm at epoch={epoch}; optimizer step aborted. "
                            f"nonfinite_parameters={bad_grads}"
                        )
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    grad_norm = _stable_clip_grad_norm_(model.parameters(), float(tcfg.get("grad_clip", 5.0)))
                    if not math.isfinite(grad_norm):
                        bad_grads = _nonfinite_gradient_report(model)
                        opt.zero_grad(set_to_none=True)
                        raise FloatingPointError(
                            f"Non-finite gradient norm at epoch={epoch}; optimizer step aborted. "
                            f"nonfinite_parameters={bad_grads}"
                        )
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
        "robust_scenario_mpc", "scenario_mpc", "batkovic_scenario_mpc",
        "dr_cvar_safety_filter", "distributionally_robust_cvar_filter", "safaoui_dr_cvar_filter",
        "conformal_predictive_safety_filter", "conformal_safety_filter", "cpsf",
        "pdm_closed", "pdm_closed_adapter", "idm", "idm_planner",
        "oracle_filter", "oracle_recovery_filter", "branchwise_oracle_filter", "oracle_branchwise_recovery",
        "postimpact_mpc", "postimpact_mpc_lite", "post_impact_mpc_lite", "postimpact_mpc_paper", "integrated_postimpact_mpc",
        "post_crash_braking", "post_crash_braking_rule", "stable_stop", "stable_stop_rule", "postcrash_stable_stop",
        "postimpact_motion_tvlqr", "postimpact_motion_planning", "wang2022_postimpact", "postimpact_tvlqr",
        "post_collision_restoration", "trajectory_restoration", "post_collision_trajectory_restoration", "post_collision_restoration_heuristic", "ackermann_restoration",
        "severity_minimization", "severity_minimization_planner", "unavoidable_collision_planner", "crash_mitigation_planner", "uc_severity_planner",
        "compensatory_postimpact_mpc", "cao_postimpact_mpc",
        "robust_postimpact_control", "postimpact_sliding_mode", "ao_postimpact_control",
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
        val_ds = (
            ExternalGroupDataset(val_dataset, cfg, split="val", baseline=baseline_name0)
            if validate_dataset and val_dataset
            else None
        )
        summary = {
            "baseline": baseline_name0,
            "training_mode": "non_learning_filter_or_planner",
            "dataset_validated": bool(validate_dataset),
            "train_dataset": str(dataset),
            "val_dataset": (str(val_dataset) if val_dataset else None),
            "num_train_groups": (len(train_ds) if train_ds is not None else None),
            "num_val_groups": (len(val_ds) if val_ds is not None else None),
            "feature_dim": (int(train_ds.feature_dim) if train_ds is not None else None),
            "max_candidates": (int(train_ds.max_candidates) if train_ds is not None else int(bcfg0.get("max_candidates", 0))),
            "notes": "This baseline is optimization/rule/filter based and has no neural training step. The train/val stage validates only the matching regime datasets; any calibration-only scalar is fitted separately on the matching calibration split, never on test.",
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
        runtime_info = configure_cuda_runtime(tcfg, device, log=(rank == 0))
        effective_amp_dtype = resolve_amp_dtype(tcfg, device)
        model = build_model_from_cfg(train_ds.feature_dim, cfg).to(device)
        if bool(tcfg.get("compile", False)) and hasattr(torch, "compile"):
            model = torch.compile(model, mode=str(tcfg.get("compile_mode", "reduce-overhead")), dynamic=bool(tcfg.get("compile_dynamic", False)))
        if use_ddp:
            fup = tcfg.get("find_unused_parameters", "auto")
            if isinstance(fup, str):
                fup_s = fup.lower()
                find_unused = baseline_name.lower() in {"betop", "betop_lite", "betopnet", "betopnet_lite"} if fup_s == "auto" else fup_s in {"1", "true", "yes"}
            else:
                find_unused = bool(fup)
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=find_unused,
                # The validation sampler intentionally does not pad/duplicate groups.
                # Disabling per-forward buffer broadcasts makes uneven validation
                # shard lengths safe; these architectures do not use BatchNorm.
                broadcast_buffers=False,
            )
        opt_kwargs = {"lr": float(tcfg.get("lr", 2.0e-4)), "weight_decay": float(tcfg.get("weight_decay", 1.0e-4))}
        if device.type == "cuda" and bool(tcfg.get("fused_optimizer", True)):
            opt_kwargs["fused"] = True
        try:
            opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)
        except (TypeError, RuntimeError):
            opt_kwargs.pop("fused", None)
            opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)
        amp_enabled = bool(tcfg.get("amp", True)) and device.type == "cuda"
        use_scaler = amp_enabled and effective_amp_dtype == torch.float16
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
        requested_batch_size = int(tcfg.get("batch_size", 32))
        global_batch_size = int(tcfg.get("global_batch_size", requested_batch_size * world_size if use_ddp else requested_batch_size))
        if use_ddp:
            if global_batch_size % world_size != 0:
                raise ValueError(f"external_baselines.training.global_batch_size={global_batch_size} must be divisible by world_size={world_size}")
            batch_size = global_batch_size // world_size
        else:
            batch_size = global_batch_size
        if use_ddp and len(train_ds) % world_size != 0:
            raise ValueError(
                f"Exact-result DDP requires num_train_groups ({len(train_ds)}) divisible by world_size ({world_size}); "
                "otherwise DistributedSampler would duplicate training groups. Use one GPU or rebuild/shard the split."
            )
        configured_workers = int(tcfg.get("num_workers_total", tcfg.get("num_workers", 0)))
        num_workers = max(0, configured_workers // world_size) if use_ddp else max(0, configured_workers)
        pin_memory = bool(tcfg.get("pin_memory", True)) and torch.cuda.is_available()
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False) if use_ddp else None
        val_sampler = _DistributedEvalSampler(val_ds, num_replicas=world_size, rank=rank) if use_ddp else None
        loader_kwargs = _loader_kwargs(tcfg, num_workers=num_workers, pin_memory=pin_memory)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=_collate, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, sampler=val_sampler, collate_fn=_collate, **loader_kwargs)
        epochs = int(tcfg.get("epochs", 10))
        best_val = float("inf")
        best_epoch = 0
        history = []
        t0 = perf_counter()
        for ep in range(1, epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(ep)
            tr = _epoch(model, train_loader, opt, device, cfg, train=True, rank=rank, epoch=ep, scaler=scaler, amp_dtype=effective_amp_dtype)
            with torch.no_grad():
                va = _epoch(model, val_loader, None, device, cfg, train=False, rank=rank, epoch=ep, scaler=None, amp_dtype=effective_amp_dtype)
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
                    "per_rank_batch_size": int(batch_size),
                    "global_batch_size": int(global_batch_size),
                }
                torch.save(ckpt, out_dir / "latest.pt")
                if va.get("loss", float("inf")) <= best_val:
                    best_val = float(va.get("loss", float("inf")))
                    best_epoch = ep
                    torch.save(ckpt, out_dir / "best.pt")
                print({"event": "external_baseline_epoch", "baseline": baseline_name, "epoch": ep, "world_size": world_size, "train_loss": tr.get("loss"), "val_loss": va.get("loss"), "target_acc": va.get("target_acc")}, flush=True)
        summary = {
            "baseline": baseline_name,
            "train_dataset": str(dataset),
            "val_dataset": (str(val_dataset) if val_dataset else None),
            "epochs_requested": int(epochs),
            "epochs_completed": int(history[-1]["epoch"] if history else 0),
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
            "per_rank_batch_size": int(batch_size),
            "global_batch_size": int(global_batch_size),
            "num_workers_per_rank": int(num_workers),
            "amp": bool(amp_enabled),
            "amp_dtype": str(effective_amp_dtype).replace("torch.", ""),
            "cuda_runtime": runtime_info,
            "fused_optimizer": bool(opt_kwargs.get("fused", False)),
            "torch_compile": bool(tcfg.get("compile", False)),
            "history": history,
        }
        if rank == 0:
            write_json(summary, out_dir / "train_summary.json")
        return summary
    finally:
        _cleanup_distributed()
