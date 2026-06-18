from __future__ import annotations

import argparse
import json
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


def _sample_weights(ds: TorchCOWPDataset, *, positive_weight: float = 3.0, stress_weight: float = 2.0) -> torch.DoubleTensor:
    weights: list[float] = []
    # Read only light label arrays.  This runs once and makes the existing
    # positive_pair_oversampling config real instead of a no-op.
    for p in ds.base.paths:
        with np.load(p, allow_pickle=True) as data:
            keys = {k.replace("__", "/"): k for k in data.files}
            w = 1.0
            if "cowp/witness/exists" in keys and np.asarray(data[keys["cowp/witness/exists"]]).any():
                w *= positive_weight
            if "cowp/candidates/false_safe" in keys and np.asarray(data[keys["cowp/candidates/false_safe"]]).any():
                w *= stress_weight
            if "cowp/candidates/noncoercive_feasible" in keys and np.asarray(data[keys["cowp/candidates/noncoercive_feasible"]]).any():
                w *= 1.25
            weights.append(w)
    return torch.as_tensor(weights, dtype=torch.double)


def _make_loader(ds: TorchCOWPDataset, cfg: dict, batch_size: int, *, shuffle: bool, oversample: bool) -> DataLoader:
    sampler = None
    if shuffle and oversample:
        sampler = WeightedRandomSampler(_sample_weights(ds), num_samples=len(ds), replacement=True)
        shuffle = False
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        collate_fn=collate_torch,
        pin_memory=torch.cuda.is_available(),
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
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    context = torch.enable_grad() if is_train else torch.no_grad()
    desc = f"{'train' if is_train else 'val'} {stage} epoch {epoch}"
    iterator = tqdm_iter(dl, enabled=progress, total=len(dl), desc=desc, unit="batch")
    with context:
        for step, batch in enumerate(iterator, start=1):
            batch = _to_device(batch, device)
            if is_train:
                opt.zero_grad(set_to_none=True)
            autocast_enabled = bool(amp and device.type == "cuda")
            with torch.amp.autocast("cuda", enabled=autocast_enabled):
                pred = model(batch, stage=stage)
                batch_for_loss = _masked_batch_with_pred_critical(batch, pred)
                losses = _compute_losses(pred, batch_for_loss, stage, loss_weights)
                loss = losses["loss"]
            if is_train:
                if scaler_enabled:
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
    ap.add_argument("--no-positive-oversampling", action="store_true")
    ap.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision for lower memory and faster training.")
    ap.add_argument("--no-progress", action="store_true", help="Disable per-epoch tqdm progress bars.")
    args = ap.parse_args()

    cfg = load_config(args.model_config, args.train_config, args.data_config)
    tcfg = cfg["train"]
    stage = args.stage or tcfg.get("stage", "witness")
    device = _device(tcfg.get("device", "auto"))
    batch_size = args.batch_size or int(tcfg.get("batch_size", 8))

    train_ds = TorchCOWPDataset(args.cache_dir or cfg["outputs"]["tensor_cache_dir"])
    val_ds = TorchCOWPDataset(args.val_cache_dir) if args.val_cache_dir else None
    train_dl = _make_loader(
        train_ds,
        cfg,
        batch_size,
        shuffle=True,
        oversample=bool(tcfg.get("positive_pair_oversampling", True)) and not args.no_positive_oversampling,
    )
    val_dl = _make_loader(val_ds, cfg, batch_size, shuffle=False, oversample=False) if val_ds is not None else None

    model = COWPModel(cfg).to(device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr if args.lr is not None else tcfg.get("lr", 3e-4)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
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
