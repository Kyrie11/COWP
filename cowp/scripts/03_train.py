from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from cowp.core.config import load_config
from cowp.data.dataset import TorchCOWPDataset, collate_torch
from cowp.models.cowp_model import COWPModel
from cowp.utils.progress import tqdm_iter
from cowp.models.losses import candidate_classification_loss, natural_loss, planner_imitation_loss, planner_outcome_loss, planner_ranking_loss, response_loss, witness_loss


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


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
    """Best-effort guard for PyTorch/CUDA builds whose pin-memory path crashes.

    Some environments report ``torch.cuda.is_available() == True`` but raise
    ``CUDA error: invalid argument`` inside DataLoader's pin-memory thread.  That
    error happens before the batch reaches the model, so it is safer to detect it
    once and disable pinning than to fail mid-epoch.
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
        print(f"Warning: disabling DataLoader pin_memory because a test pin failed: {exc}")
        return False

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
) -> DataLoader:
    sampler = None
    if shuffle and oversample:
        sampler = WeightedRandomSampler(_sample_weights(ds, progress=progress, cache=sampler_cache), num_samples=len(ds), replacement=True)
        shuffle = False
    nw = int(cfg["train"].get("num_workers", 0) if num_workers is None else num_workers)
    kwargs: dict[str, Any] = {}
    if nw > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(cfg["train"].get("prefetch_factor", 2))
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


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}


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
        outcome = planner_outcome_loss(pred["planner_score"], batch)
        cls = candidate_classification_loss(pred["planner_score"], batch, loss_weights)
        out["planner/ranking"] = rank
        out["planner/imitation"] = imitation
        out["planner/outcome"] = outcome
        out.update({f"planner/{k}": v for k, v in cls.items() if k != "loss"})
        losses.append(
            loss_weights.get("ranking", 1.0) * rank
            + loss_weights.get("imitation", 1.0) * imitation
            + loss_weights.get("closed_loop", 0.0) * outcome
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
            batch = _to_device(batch, device)
            if is_train:
                opt.zero_grad(set_to_none=True)
            autocast_enabled = bool(amp and device.type == "cuda")
            with _autocast_context(device, autocast_enabled):
                pred = model(batch, stage=stage)
                batch_for_loss = _masked_batch_with_pred_critical(batch, pred)
                losses = _compute_losses(pred, batch_for_loss, stage, loss_weights)
                loss = losses["loss"]
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


def _maybe_compile_model(model: COWPModel, enabled: bool) -> torch.nn.Module:
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        print("Warning: torch.compile is not available in this PyTorch version; continuing without compile.")
        return model
    try:
        print("Compiling model with torch.compile(mode='max-autotune') ...")
        return torch.compile(model, mode="max-autotune")
    except Exception as exc:
        print(f"Warning: torch.compile failed, continuing without compile: {exc}")
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
    ap.add_argument("--no-positive-oversampling", action="store_true")
    ap.add_argument("--no-sampler-cache", action="store_true", help="Do not read/write cached sampler weights for positive oversampling.")
    ap.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision for lower memory and faster training.")
    ap.add_argument("--pin-memory", action="store_true", help="Force DataLoader pin_memory on when using CUDA.")
    ap.add_argument("--no-pin-memory", action="store_true", help="Force DataLoader pin_memory off. Useful for CUDA invalid-argument pinning errors.")
    ap.add_argument("--no-progress", action="store_true", help="Disable per-epoch tqdm progress bars.")
    ap.add_argument("--compile", action="store_true", help="Use torch.compile for faster repeated training on PyTorch 2.x. First epoch may be slower.")
    ap.add_argument("--fused-adamw", action="store_true", help="Use fused CUDA AdamW when supported.")
    args = ap.parse_args()

    cfg = load_config(args.model_config, args.train_config, args.data_config)
    tcfg = cfg["train"]
    stage = args.stage or tcfg.get("stage", "witness")
    device = _device(args.device or tcfg.get("device", "auto"))
    _set_runtime_defaults(int(tcfg.get("seed", 2026)), device)
    batch_size = args.batch_size or int(tcfg.get("batch_size", 8))
    if args.no_pin_memory:
        pin_memory = False
    elif args.pin_memory:
        pin_memory = True
    else:
        pin_memory = _cuda_pin_memory_works(device)

    print(f"COWP train startup: stage={stage}, device={device}, cuda_available={torch.cuda.is_available()}, amp={args.amp}, compile={args.compile or bool(tcfg.get('compile', False))}, batch_size={batch_size}, pin_memory={pin_memory}")
    if device.type == "cuda":
        try:
            print(f"CUDA device: {torch.cuda.get_device_name(device)}")
        except Exception:
            pass

    train_ds = TorchCOWPDataset(args.cache_dir or cfg["outputs"]["tensor_cache_dir"], stage=stage)
    val_ds = TorchCOWPDataset(args.val_cache_dir, stage=stage) if args.val_cache_dir else None
    print(f"Loaded datasets: train={len(train_ds)}" + (f", val={len(val_ds)}" if val_ds is not None else ""))
    train_dl = _make_loader(
        train_ds,
        cfg,
        batch_size,
        shuffle=True,
        oversample=bool(tcfg.get("positive_pair_oversampling", True)) and not args.no_positive_oversampling,
        use_cuda=device.type == "cuda",
        progress=not args.no_progress,
        num_workers=args.num_workers,
        sampler_cache=not args.no_sampler_cache,
        pin_memory=pin_memory,
    )
    val_dl = _make_loader(val_ds, cfg, batch_size, shuffle=False, oversample=False, use_cuda=device.type == "cuda", progress=not args.no_progress, num_workers=args.num_workers, pin_memory=pin_memory) if val_ds is not None else None

    print(f"DataLoader: num_workers={train_dl.num_workers}, pin_memory={train_dl.pin_memory}, train_batches={len(train_dl)}" + (f", val_batches={len(val_dl)}" if val_dl is not None else ""))
    model = COWPModel(cfg).to(device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
    model = _maybe_compile_model(model, bool(args.compile or tcfg.get("compile", False)))
    opt = _make_adamw_optimizer(
        model,
        lr=float(args.lr if args.lr is not None else tcfg.get("lr", 3e-4)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
        fused=bool(args.fused_adamw or tcfg.get("fused_adamw", False)),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = args.epochs or int(tcfg.get("epochs", 10))
    loss_weights = cfg.get("loss_weights", {})
    history = []
    best_val = float("inf")
    for epoch in range(epochs):
        train_metrics = _run_epoch(model, train_dl, device, stage, loss_weights, opt, epoch=epoch, progress=not args.no_progress, amp=args.amp)
        row: dict[str, Any] = {"epoch": epoch, **{f"train/{k}": v for k, v in train_metrics.items()}}
        if val_dl is not None:
            val_metrics = _run_epoch(model, val_dl, device, stage, loss_weights, None, epoch=epoch, progress=not args.no_progress, amp=args.amp)
            row.update({f"val/{k}": v for k, v in val_metrics.items()})
            val_loss = float(val_metrics.get("loss", float("inf")))
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "stage": stage, "val_loss": val_loss}, output_dir / f"cowp_{stage}_best.pt")
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "stage": stage}, output_dir / f"cowp_{stage}_epoch{epoch:03d}.pt")
    with (output_dir / f"history_{stage}.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
