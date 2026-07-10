from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler, DistributedSampler, Sampler

from cowp.core.config import load_config
from cowp.data.dataset import TorchCOWPDataset, collate_torch
from cowp.models.cowp_model import COWPModel
from cowp.utils.progress import tqdm_iter
from cowp.models.losses import candidate_classification_loss, natural_loss, planner_imitation_loss, planner_outcome_loss, planner_outcome_supervision, planner_ranking_loss, priority_claim_loss, response_loss, witness_loss


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _distributed_env() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def _setup_distributed(device_arg: str) -> tuple[bool, int, int, int]:
    rank, world_size, local_rank = _distributed_env()
    distributed = world_size > 1
    if not distributed:
        return False, 0, 1, 0
    use_cuda = str(device_arg).lower() != "cpu" and torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    if use_cuda:
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return True, rank, world_size, local_rank


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def _rank0_print(*args, **kwargs) -> None:
    if _is_main_process():
        print(*args, **kwargs)


def _distributed_mean_metrics(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    if not (dist.is_available() and dist.is_initialized()):
        return metrics
    if not metrics:
        return metrics
    keys = sorted(metrics.keys())
    values = torch.tensor([float(metrics[k]) for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= float(dist.get_world_size())
    return {k: float(v) for k, v in zip(keys, values.detach().cpu().tolist())}


def _set_sampler_epoch(dl: DataLoader | None, epoch: int) -> None:
    if dl is None:
        return
    sampler = getattr(dl, "sampler", None)
    set_epoch = getattr(sampler, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(epoch)


def _set_runtime_defaults(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        if hasattr(torch, "set_float32_matmul_precision"):
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass




def _cuda_pin_memory_works(device: torch.device) -> bool:
    """Best-effort guard for explicit DataLoader pin_memory requests.

    Pinning is deliberately *not* enabled by default in this project.  Stage-B
    response training can load hundreds of MB per batch because
    ``cowp/response/traj`` is a dense [K,A,R,T,7] target.  With multi-worker
    prefetching, the pin-memory thread may try to page-lock several GB of CPU
    tensors and fail with ``CUDA error: invalid argument`` before the next batch
    reaches the model.  Pinning changes transfer mechanics only; it does not
    change the model, labels, loss, or predictions.  Therefore the stable default
    is no pinning, while ``--pin-memory`` remains available after this sanity
    check.
    """
    if device.type != "cuda":
        return False
    try:
        x = torch.empty(1)
        try:
            _ = x.pin_memory(device)
        except TypeError:
            _ = x.pin_memory()
        return True
    except Exception as exc:
        print(f"Warning: requested DataLoader pin_memory is unavailable; disabling it: {exc}")
        return False


def _heavy_label_stage(stage: str) -> bool:
    """Stages whose batches may contain very large dense supervision tensors."""
    return stage in {"response", "planner", "all"}

def _make_grad_scaler(enabled: bool):
    """Return a CUDA GradScaler that works across old and new PyTorch APIs.

    Some PyTorch versions expose AMP as ``torch.cuda.amp`` only, while newer
    versions prefer ``torch.amp``.  Importantly, when AMP is disabled we return
    ``None`` instead of constructing a scaler; this avoids crashes like
    ``AttributeError: module 'torch.amp' has no attribute 'GradScaler'`` even for
    non-AMP runs.
    """
    if not enabled:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:  # older signature: GradScaler(enabled=True)
            return torch.amp.GradScaler(enabled=True)
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=True)
    return None


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            try:
                return torch.amp.autocast("cuda", enabled=True)
            except TypeError:
                return torch.amp.autocast(device_type="cuda", enabled=True)
        if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
            return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def _dataset_signature(paths: list[Path]) -> dict[str, object]:
    if not paths:
        return {"num_files": 0, "mtime_ns_sum": 0, "size_sum": 0}
    mtime_ns_sum = 0
    size_sum = 0
    for p in paths:
        try:
            st = p.stat()
            mtime_ns_sum += int(st.st_mtime_ns)
            size_sum += int(st.st_size)
        except OSError:
            pass
    return {"num_files": len(paths), "mtime_ns_sum": int(mtime_ns_sum), "size_sum": int(size_sum)}


def _read_sample_weight_from_npz(path: Path, *, positive_weight: float, stress_weight: float) -> float:
    with np.load(path, allow_pickle=True) as data:
        keys = {k.replace("__", "/"): k for k in data.files}
        w = 1.0
        if "cowp/witness/exists" in keys and np.asarray(data[keys["cowp/witness/exists"]]).any():
            w *= positive_weight
        if "cowp/candidates/false_safe" in keys and np.asarray(data[keys["cowp/candidates/false_safe"]]).any():
            w *= stress_weight
        if "cowp/candidates/noncoercive_feasible" in keys and np.asarray(data[keys["cowp/candidates/noncoercive_feasible"]]).any():
            w *= 1.25
        return float(w)


def _sample_weights(
    ds: TorchCOWPDataset,
    *,
    positive_weight: float = 3.0,
    stress_weight: float = 2.0,
    progress: bool = True,
    cache: bool = True,
) -> torch.DoubleTensor:
    paths = [Path(p) for p in ds.base.paths]
    sig = _dataset_signature(paths)
    meta = {**sig, "positive_weight": float(positive_weight), "stress_weight": float(stress_weight), "version": 2}
    cache_path = Path(ds.base.cache_dir) / f".cowp_sampler_weights_pw{positive_weight:g}_sw{stress_weight:g}_v2.npz"
    if cache and cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=True) as cached:
                cached_meta = json.loads(str(cached["metadata"].item()))
                weights = cached["weights"].astype(np.float64)
            if cached_meta == meta and len(weights) == len(paths):
                print(f"Loaded sampler weights from {cache_path}")
                return torch.as_tensor(weights, dtype=torch.double)
        except Exception:
            pass

    print("Building sampler weights once. This scans cache metadata; pass --no-positive-oversampling to skip it.")
    weights: list[float] = []
    iterator = tqdm_iter(paths, enabled=progress, total=len(paths), desc="Build sampler weights", unit="file")
    for p in iterator:
        weights.append(_read_sample_weight_from_npz(p, positive_weight=positive_weight, stress_weight=stress_weight))
    arr = np.asarray(weights, dtype=np.float64)
    if cache:
        try:
            np.savez(cache_path, weights=arr, metadata=json.dumps(meta, ensure_ascii=False))
            print(f"Saved sampler weights to {cache_path}")
        except Exception as exc:
            print(f"Warning: failed to save sampler weights cache {cache_path}: {exc}")
    return torch.as_tensor(arr, dtype=torch.double)


class DistributedWeightedRandomSampler(Sampler[int]):
    """Weighted replacement sampler split deterministically across DDP ranks."""

    def __init__(
        self,
        weights: torch.Tensor,
        *,
        num_replicas: int,
        rank: int,
        replacement: bool = True,
        seed: int = 2026,
    ) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.weights) / max(self.num_replicas, 1)))
        self.total_size = int(self.num_samples * max(self.num_replicas, 1))

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.total_size, self.replacement, generator=g).tolist()
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def _make_loader(
    ds: TorchCOWPDataset,
    cfg: dict,
    batch_size: int,
    *,
    shuffle: bool,
    oversample: bool,
    use_cuda: bool,
    progress: bool,
    num_workers: int | None = None,
    sampler_cache: bool = True,
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 2026,
) -> DataLoader:
    sampler = None
    if distributed:
        if shuffle and oversample:
            sampler = DistributedWeightedRandomSampler(
                _sample_weights(ds, progress=progress, cache=sampler_cache),
                num_replicas=world_size,
                rank=rank,
                replacement=True,
                seed=seed,
            )
        else:
            sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed, drop_last=False)
        shuffle = False
    elif shuffle and oversample:
        sampler = WeightedRandomSampler(_sample_weights(ds, progress=progress, cache=sampler_cache), num_samples=len(ds), replacement=True)
        shuffle = False
    nw = int(cfg["train"].get("num_workers", 0) if num_workers is None else num_workers)
    kwargs: dict[str, Any] = {}
    if nw > 0:
        kwargs["persistent_workers"] = True
        pf = int(cfg["train"].get("prefetch_factor", 2) if prefetch_factor is None else prefetch_factor)
        kwargs["prefetch_factor"] = max(1, pf)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=nw,
        collate_fn=collate_torch,
        pin_memory=bool(use_cuda and pin_memory),
        **kwargs,
    )


def _to_device(batch: dict[str, Any], device: torch.device, *, non_blocking: bool = False) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=bool(non_blocking)) for k, v in batch.items() if torch.is_tensor(v)}


def _compute_losses(pred: dict[str, Any], batch: dict[str, torch.Tensor], stage: str, loss_weights: dict[str, float]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    losses: list[torch.Tensor] = []
    if stage in ("natural", "representation", "all"):
        nl = natural_loss(pred["natural"], batch, loss_weights)
        out.update({f"natural/{k}": v for k, v in nl.items() if k != "loss"})
        losses.append(nl["loss"])
    if stage in ("response", "all"):
        rl = response_loss(pred["response"], batch, loss_weights)
        out.update({f"response/{k}": v for k, v in rl.items() if k != "loss"})
        losses.append(rl["loss"])
    if stage in ("witness", "planner", "all"):
        wl = witness_loss(pred["witness"], batch, loss_weights)
        out.update({f"witness/{k}": v for k, v in wl.items() if k != "loss"})
        losses.append(wl["loss"])
    if stage in ("planner", "all"):
        rank = planner_ranking_loss(
            pred["planner_score"],
            batch["cowp/candidates/noncoercive_feasible"].bool(),
            batch["cowp/candidates/false_safe"].bool(),
            batch["cowp/candidates/valid"].bool(),
        )
        imitation = planner_imitation_loss(pred["planner_score"], batch)
        outcome_legacy = planner_outcome_loss(pred["planner_score"], batch)
        outcome = planner_outcome_supervision(pred.get("outcome"), pred["planner_score"], batch, loss_weights)
        priority_claim = priority_claim_loss(pred.get("priority_claim_logits"), batch, loss_weights)
        cls = candidate_classification_loss(pred["planner_score"], batch, loss_weights)
        out["planner/ranking"] = rank
        out["planner/imitation"] = imitation
        out["planner/outcome"] = outcome["loss"]
        out["planner/outcome_cls"] = outcome["cls"]
        out["planner/outcome_logdiv"] = outcome["logdiv"]
        out["planner/outcome_rank"] = outcome["rank"]
        out["planner/outcome_expected"] = outcome["expected_cost"]
        out["planner/outcome_legacy"] = outcome_legacy
        out["planner/priority_claim"] = priority_claim
        out.update({f"planner/{k}": v for k, v in cls.items() if k != "loss"})
        losses.append(
            loss_weights.get("ranking", 1.0) * rank
            + loss_weights.get("imitation", 1.0) * imitation
            + loss_weights.get("closed_loop", 0.0) * outcome["loss"]
            + loss_weights.get("closed_loop_legacy", 0.0) * outcome_legacy
            + loss_weights.get("priority_claim", 0.5) * priority_claim
            + cls["loss"]
        )
    if not losses:
        # Representation fallback: keep graph/planner path differentiable.
        losses.append(pred["planner_score"].mean() * 0.0 + torch.relu(pred["witness"]["opr"].mean() - 0.5))
    out["loss"] = sum(losses)
    return out


def _masked_batch_with_pred_critical(batch: dict[str, torch.Tensor], pred: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Make loss masks agree with model-visible critical-agent slots."""
    crit_mask = pred.get("critical_mask")
    if not torch.is_tensor(crit_mask) or "cowp/critical/valid" not in batch:
        return batch
    if torch.equal(batch["cowp/critical/valid"].bool(), crit_mask.bool()):
        return batch
    out = dict(batch)
    out["cowp/critical/valid"] = crit_mask.bool()
    return out



def _finite_scalar(x: torch.Tensor) -> bool:
    try:
        return bool(torch.isfinite(x.detach()).all().item())
    except Exception:
        return False

def _run_epoch(
    model: COWPModel,
    dl: DataLoader,
    device: torch.device,
    stage: str,
    loss_weights: dict[str, float],
    opt: torch.optim.Optimizer | None = None,
    *,
    epoch: int = 0,
    progress: bool = True,
    amp: bool = False,
    non_blocking_transfer: bool = False,
    decode_response_traj: bool = True,
) -> dict[str, float]:
    is_train = opt is not None
    model.train(is_train)
    sums: dict[str, float] = {}
    count = 0
    scaler_enabled = bool(amp and is_train and device.type == "cuda")
    scaler = _make_grad_scaler(scaler_enabled)
    context = torch.enable_grad() if is_train else torch.no_grad()
    desc = f"{'train' if is_train else 'val'} {stage} epoch {epoch}"
    iterator = tqdm_iter(dl, enabled=progress, total=len(dl), desc=desc, unit="batch")
    with context:
        for step, batch in enumerate(iterator, start=1):
            batch = _to_device(batch, device, non_blocking=non_blocking_transfer)
            if is_train:
                opt.zero_grad(set_to_none=True)
            autocast_enabled = bool(amp and device.type == "cuda")
            # Only the model forward runs under AMP.  Losses are intentionally
            # computed in fp32 outside autocast: this prevents CUDA AMP from
            # rejecting unsafe probability-space BCE kernels and keeps scatter /
            # masked reductions numerically stable.
            with _autocast_context(device, autocast_enabled):
                pred = model(batch, stage=stage, decode_response_traj=decode_response_traj)
            batch_for_loss = _masked_batch_with_pred_critical(batch, pred)
            losses = _compute_losses(pred, batch_for_loss, stage, loss_weights)
            loss = losses["loss"]
            if not _finite_scalar(loss):
                bad = []
                for k, v in losses.items():
                    if torch.is_tensor(v) and not _finite_scalar(v):
                        bad.append(k)
                print(f"Warning: non-finite loss at stage={stage} epoch={epoch} step={step}; skipped batch; bad_terms={bad[:8]}")
                if is_train:
                    opt.zero_grad(set_to_none=True)
                continue
            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
            bs = int(next(iter(batch.values())).shape[0]) if batch else 1
            count += bs
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach().cpu()) * bs
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}", seen=count, refresh=(step == 1 or step % 10 == 0))
    return {k: v / max(count, 1) for k, v in sums.items()}


def _model_state_dict_for_save(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a checkpoint state_dict that can be loaded without DDP/torch.compile wrappers."""
    inner = getattr(model, "module", model)
    inner = getattr(inner, "_orig_mod", inner)
    if isinstance(inner, torch.nn.Module):
        return inner.state_dict()
    return model.state_dict()


def _load_model_state_robust(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load checkpoints saved from eager/compiled models and tolerate new heads."""
    candidates = [state]
    if state and all(k.startswith("_orig_mod.") for k in state.keys()):
        candidates.insert(0, {k[len("_orig_mod."):]: v for k, v in state.items()})
    last_error: RuntimeError | None = None
    model_state = model.state_dict()
    for cand in candidates:
        try:
            model.load_state_dict(cand)
            return
        except RuntimeError as exc:
            last_error = exc
        compatible = {k: v for k, v in cand.items() if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)}
        missing = sorted(set(model_state) - set(compatible))
        unexpected = sorted(set(cand) - set(compatible))
        if compatible:
            model.load_state_dict(compatible, strict=False)
            if missing:
                print(f"Loaded checkpoint with {len(missing)} newly initialized/missing keys, e.g. {missing[:6]}")
            if unexpected:
                print(f"Ignored {len(unexpected)} incompatible/unexpected keys, e.g. {unexpected[:6]}")
            return
    assert last_error is not None
    raise last_error



def _make_adamw_optimizer(model: torch.nn.Module, *, lr: float, weight_decay: float, fused: bool = False) -> torch.optim.Optimizer:
    """Create AdamW, using fused CUDA implementation when available/requested."""
    kwargs: dict[str, Any] = {"lr": float(lr), "weight_decay": float(weight_decay)}
    if fused and torch.cuda.is_available():
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except TypeError:
            pass
        except RuntimeError as exc:
            print(f"Warning: fused AdamW unavailable, falling back to standard AdamW: {exc}")
    return torch.optim.AdamW(model.parameters(), **kwargs)


def _maybe_compile_model(model: COWPModel, enabled: bool, backend: str | None = None) -> torch.nn.Module:
    """Compile the model in a failure-tolerant way.

    torch.compile performs most real compilation at first forward.  Setting
    Dynamo suppress_errors lets unsupported Inductor/Triton subgraphs fall back
    to eager instead of aborting the training run.
    """
    if not enabled:
        return model
    import torch
    if not hasattr(torch, "compile"):
        print("Warning: torch.compile is not available in this PyTorch version; continuing without compile.")
        return model
    try:
        import torch._dynamo  # type: ignore[attr-defined]
        torch._dynamo.config.suppress_errors = True
    except Exception:
        pass
    try:
        backend_msg = f", backend={backend}" if backend else ""
        print(f"Compiling model with torch.compile(mode='reduce-overhead'{backend_msg}, suppress_errors=True) ...")
        kwargs: dict[str, Any] = {"mode": "reduce-overhead"}
        if backend:
            kwargs["backend"] = backend
        return torch.compile(model, **kwargs)
    except Exception as exc:
        print(f"Warning: torch.compile setup failed, continuing without compile: {exc}")
        return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Train COWP model stages on COWP tensor cache.")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--train-config", default="configs/train.yaml")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--val-cache-dir", default=None)
    ap.add_argument("--stage", choices=["representation", "natural", "response", "witness", "planner", "all"], default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--output-dir", default="outputs/checkpoints")
    ap.add_argument("--device", default=None, help="Training device: auto, cuda, cuda:0, cpu. Overrides configs/train.yaml.")
    ap.add_argument("--num-workers", type=int, default=None, help="Override DataLoader num_workers.")
    ap.add_argument("--prefetch-factor", type=int, default=None, help="Override DataLoader prefetch_factor when num_workers > 0. For response/planner/all stages, default is capped to 1 to avoid huge queued batches.")
    ap.add_argument("--no-positive-oversampling", action="store_true")
    ap.add_argument("--force-positive-oversampling", action="store_true", help="Force witness/candidate positive oversampling even for representation/natural stages.")
    ap.add_argument("--no-sampler-cache", action="store_true", help="Do not read/write cached sampler weights for positive oversampling.")
    ap.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision for lower memory and faster training.")
    ap.add_argument("--pin-memory", action="store_true", help="Force DataLoader pin_memory on when using CUDA.")
    ap.add_argument("--no-pin-memory", action="store_true", help="Force DataLoader pin_memory off. Useful for CUDA invalid-argument pinning errors.")
    ap.add_argument("--no-progress", action="store_true", help="Disable per-epoch tqdm progress bars.")
    ap.add_argument("--compile", action="store_true", help="Use torch.compile for faster repeated training on PyTorch 2.x. Unsupported subgraphs fall back to eager.")
    ap.add_argument("--compile-backend", default=None, help="Optional torch.compile backend, e.g. aot_eager for safer fallback when Inductor/Triton is unstable.")
    ap.add_argument("--fused-adamw", action="store_true", help="Use fused CUDA AdamW when supported.")
    ap.add_argument("--response-traj-weight", type=float, default=None, help="Override loss_weights.response_traj_l1. Set to 0 for fast response/witness/planner smoke training without dense response trajectory labels.")
    ap.add_argument("--no-response-traj", action="store_true", help="Shortcut for --response-traj-weight 0. Avoids loading cowp/response/traj and skips the response trajectory head.")
    ap.add_argument("--no-response-components", action="store_true", help="Do not load/supervise cowp/response/burden_components during response training.")
    ap.add_argument("--with-waymax-outcome-labels", action="store_true", help="For planner training, load optional waymax/candidate_{collision,offroad,log_divergence} labels if present. Default keeps broad waymax tensors out of batches.")
    ap.add_argument("--val-every", type=int, default=1, help="Run validation every N epochs. Use 0 to disable validation during quick smoke training.")
    args = ap.parse_args()

    cfg = load_config(args.model_config, args.train_config, args.data_config)
    tcfg = cfg["train"]
    stage = args.stage or tcfg.get("stage", "witness")
    loss_weights = dict(cfg.get("loss_weights", {}))
    if args.no_response_traj:
        loss_weights["response_traj_l1"] = 0.0
    if args.response_traj_weight is not None:
        loss_weights["response_traj_l1"] = float(args.response_traj_weight)
    if args.no_response_components:
        loss_weights["response_components_l1"] = 0.0

    device_arg = args.device or tcfg.get("device", "auto")
    distributed, rank, world_size, local_rank = _setup_distributed(str(device_arg))
    if distributed and str(device_arg).lower() != "cpu" and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = _device(device_arg)
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device.type == "cpu" and str(device_arg) == "auto" and visible_devices in {"-1", "none", "None", ""}:
        print(
            "Warning: CUDA is hidden by CUDA_VISIBLE_DEVICES="
            f"{visible_devices!r}; training will run on CPU. "
            "Run `unset CUDA_VISIBLE_DEVICES` or `export CUDA_VISIBLE_DEVICES=0` before training if a GPU is available."
        )
    if str(device_arg).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device was requested, but torch.cuda.is_available() is False. "
            "Check CUDA_VISIBLE_DEVICES and the PyTorch CUDA build."
        )
    _set_runtime_defaults(int(tcfg.get("seed", 2026)), device)
    batch_size = args.batch_size or int(tcfg.get("batch_size", 8))
    if args.no_pin_memory:
        pin_memory = False
    elif args.pin_memory:
        pin_memory = _cuda_pin_memory_works(device)
        if pin_memory and _heavy_label_stage(stage):
            print(
                f"Warning: pin_memory was forced on for heavy stage={stage}. "
                "If CUDA invalid-argument errors recur, rerun with --no-pin-memory."
            )
    else:
        # Stable default: keep pinning off.  This avoids Stage-B crashes caused by
        # page-locking dense response supervision batches.  Use --pin-memory only
        # after confirming the host/CUDA setup can sustain the memory pressure.
        pin_memory = False

    user_prefetch = args.prefetch_factor
    effective_prefetch = user_prefetch
    if effective_prefetch is None and _heavy_label_stage(stage):
        effective_prefetch = 1
    elif effective_prefetch is not None and effective_prefetch > 1 and _heavy_label_stage(stage):
        print(
            f"Warning: stage={stage} has large dense label tensors; "
            f"prefetch_factor={effective_prefetch} can queue very large CPU batches. "
            "Use --prefetch-factor 1 if host RAM or pinned-memory errors appear."
        )

    compile_enabled = bool(args.compile or tcfg.get("compile", False))
    # Respect response-head disabling flags in both response-only and all-head
    # training.  Without this, ``--stage all --response-traj-weight 0`` still
    # loaded ``cowp/response/traj`` and decoded the giant trajectory head.
    include_response_traj = not (stage in {"response", "all"} and float(loss_weights.get("response_traj_l1", 0.1)) == 0.0)
    include_response_components = not (stage in {"response", "all"} and float(loss_weights.get("response_components_l1", 0.25)) == 0.0)
    if distributed and compile_enabled:
        _rank0_print("Warning: disabling torch.compile under DDP for staged training stability.")
        compile_enabled = False
    _rank0_print(f"COWP train startup: stage={stage}, device={device}, distributed={distributed}, rank={rank}/{world_size}, cuda_available={torch.cuda.is_available()}, amp={args.amp}, compile={compile_enabled}, batch_size_per_gpu={batch_size}, global_batch_size={batch_size * world_size if distributed else batch_size}, pin_memory={pin_memory}, prefetch_factor={effective_prefetch if effective_prefetch is not None else tcfg.get('prefetch_factor', 2)}, response_traj_l1={float(loss_weights.get('response_traj_l1', 0.0))}, load_response_traj={include_response_traj}, load_waymax_outcomes={bool(args.with_waymax_outcome_labels)}")
    if device.type == "cuda":
        try:
            _rank0_print(f"CUDA device: {torch.cuda.get_device_name(device)}")
        except Exception:
            pass

    train_ds = TorchCOWPDataset(
        args.cache_dir or cfg["outputs"]["tensor_cache_dir"],
        stage=stage,
        include_response_traj=include_response_traj,
        include_response_components=include_response_components,
        include_waymax_outcomes=bool(args.with_waymax_outcome_labels),
    )
    val_ds = (
        TorchCOWPDataset(
            args.val_cache_dir,
            stage=stage,
            include_response_traj=include_response_traj,
            include_response_components=include_response_components,
            include_waymax_outcomes=bool(args.with_waymax_outcome_labels),
        )
        if args.val_cache_dir
        else None
    )
    _rank0_print(f"Loaded datasets: train={len(train_ds)}" + (f", val={len(val_ds)}" if val_ds is not None else ""))
    oversample_enabled = bool(tcfg.get("positive_pair_oversampling", True)) and not args.no_positive_oversampling
    # Representation/natural stages do not consume witness/candidate labels.
    # Scanning every npz for those labels can add many minutes before the first
    # batch and does not change the Stage-A objective.
    if stage in {"representation", "natural"} and not args.force_positive_oversampling:
        if oversample_enabled:
            _rank0_print(f"Skipping positive oversampling for stage={stage}; use --force-positive-oversampling to enable it.")
        oversample_enabled = False

    seed = int(tcfg.get("seed", 2026))
    if distributed and oversample_enabled and not args.no_sampler_cache and rank != 0:
        dist.barrier()
    train_dl = _make_loader(
        train_ds,
        cfg,
        batch_size,
        shuffle=True,
        oversample=oversample_enabled,
        use_cuda=device.type == "cuda",
        progress=(not args.no_progress) and _is_main_process(),
        num_workers=args.num_workers,
        sampler_cache=not args.no_sampler_cache,
        pin_memory=pin_memory,
        prefetch_factor=effective_prefetch,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )
    if distributed and oversample_enabled and not args.no_sampler_cache and rank == 0:
        dist.barrier()
    val_dl = _make_loader(
        val_ds, cfg, batch_size, shuffle=False, oversample=False, use_cuda=device.type == "cuda",
        progress=(not args.no_progress) and _is_main_process(), num_workers=args.num_workers, pin_memory=pin_memory,
        prefetch_factor=effective_prefetch, distributed=distributed, rank=rank, world_size=world_size, seed=seed,
    ) if val_ds is not None else None

    _rank0_print(f"DataLoader: num_workers={train_dl.num_workers}, pin_memory={train_dl.pin_memory}, prefetch_factor={getattr(train_dl, 'prefetch_factor', None)}, train_batches_per_rank={len(train_dl)}" + (f", val_batches_per_rank={len(val_dl)}" if val_dl is not None else ""))
    model = COWPModel(cfg).to(device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        _load_model_state_robust(model, ckpt["model"])
    model = _maybe_compile_model(model, compile_enabled, backend=args.compile_backend)
    if distributed:
        ddp_kwargs = {"find_unused_parameters": True}
        if device.type == "cuda":
            ddp_kwargs.update({"device_ids": [local_rank], "output_device": local_rank})
        model = DDP(model, **ddp_kwargs)
    opt = _make_adamw_optimizer(
        model,
        lr=float(args.lr if args.lr is not None else tcfg.get("lr", 3e-4)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
        fused=bool(args.fused_adamw or tcfg.get("fused_adamw", False)),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = args.epochs or int(tcfg.get("epochs", 10))
    history = []
    best_val = float("inf")
    try:
        for epoch in range(epochs):
            _set_sampler_epoch(train_dl, epoch)
            train_metrics = _run_epoch(
                model,
                train_dl,
                device,
                stage,
                loss_weights,
                opt,
                epoch=epoch,
                progress=(not args.no_progress) and _is_main_process(),
                amp=args.amp,
                non_blocking_transfer=pin_memory,
                decode_response_traj=include_response_traj,
            )
            train_metrics = _distributed_mean_metrics(train_metrics, device)
            row: dict[str, Any] = {"epoch": epoch, **{f"train/{k}": v for k, v in train_metrics.items()}}
            do_val = val_dl is not None and int(args.val_every) > 0 and ((epoch + 1) % int(args.val_every) == 0 or epoch == epochs - 1)
            if do_val:
                _set_sampler_epoch(val_dl, epoch)
                val_metrics = _run_epoch(
                    model,
                    val_dl,
                    device,
                    stage,
                    loss_weights,
                    None,
                    epoch=epoch,
                    progress=(not args.no_progress) and _is_main_process(),
                    amp=args.amp,
                    non_blocking_transfer=pin_memory,
                    decode_response_traj=include_response_traj,
                )
                val_metrics = _distributed_mean_metrics(val_metrics, device)
                row.update({f"val/{k}": v for k, v in val_metrics.items()})
                score_loss = float(val_metrics.get("loss", float("inf")))
                score_name = "val_loss"
            else:
                # If validation is intentionally disabled for smoke training, still
                # produce cowp_<stage>_best.pt so downstream stage commands can resume.
                score_loss = float(train_metrics.get("loss", float("inf"))) if (val_dl is None or int(args.val_every) == 0) else float("inf")
                score_name = "train_loss"
            if _is_main_process():
                if math.isfinite(score_loss) and score_loss < best_val:
                    best_val = score_loss
                    torch.save({"model": _model_state_dict_for_save(model), "cfg": cfg, "epoch": epoch, "stage": stage, score_name: score_loss}, output_dir / f"cowp_{stage}_best.pt")
                history.append(row)
                print(json.dumps(row, ensure_ascii=False))
                torch.save({"model": _model_state_dict_for_save(model), "cfg": cfg, "epoch": epoch, "stage": stage}, output_dir / f"cowp_{stage}_epoch{epoch:03d}.pt")
        if _is_main_process():
            with (output_dir / f"history_{stage}.json").open("w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    main()
