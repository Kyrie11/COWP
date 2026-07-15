from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from cowp.core.config import load_config
from cowp.data.dataset import collate_torch
from cowp.external_baselines.adapters import ExternalCOWPDataset, best_candidate_to_logged_ego, make_external_batch
from cowp.external_baselines.dtpp_cowp import COWPDTPP, dtpp_loss
from cowp.external_baselines.gameformer_cowp import COWPGameFormer, gameformer_loss
from cowp.utils.progress import tqdm_iter


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "set_float32_matmul_precision"):
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass


def _build_model(args: argparse.Namespace):
    if args.baseline == "gameformer":
        return COWPGameFormer(
            modalities=args.gameformer_modalities,
            neighbors_to_predict=args.max_neighbors,
            future_len=args.future_len,
            encoder_layers=args.gameformer_encoder_layers,
            decoder_levels=args.gameformer_decoder_levels,
        )
    if args.baseline == "dtpp":
        return COWPDTPP(neighbors=args.max_neighbors, max_branch=args.max_candidates, variable_cost=not args.dtpp_fixed_cost)
    raise ValueError(args.baseline)


def _state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    if isinstance(model, torch.nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def _run_epoch(model: torch.nn.Module, loader: DataLoader, cfg: dict[str, Any], args: argparse.Namespace, device: torch.device, optimizer: torch.optim.Optimizer | None, epoch: int) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    sums: dict[str, float] = {}
    n = 0
    iterator = tqdm_iter(loader, enabled=not args.no_progress, desc=("train" if train else "val") + f" {args.baseline} e{epoch}")
    for batch in iterator:
        try:
            ext = make_external_batch(batch, cfg, device=device, max_neighbors=args.max_neighbors, max_candidates=args.max_candidates, horizon=args.future_len)
            if args.baseline == "gameformer":
                outputs = model(ext.gameformer_inputs)
                loss, metrics = gameformer_loss(outputs, ext.ego_future_xy, ext.ego_future_valid, ext.neighbors_future_xy, ext.neighbors_future_valid)
            else:
                best_idx = best_candidate_to_logged_ego(ext.candidates, ext.candidate_valid, ext.ego_future_xy, ext.ego_future_valid)
                loss, metrics = dtpp_loss(model, ext.dtpp_inputs, ext.dtpp_candidate_tree, ext.candidate_valid, best_idx, ext.ego_future_xy, ext.ego_future_valid, ext.neighbors_future_xy, ext.neighbors_future_valid, timesteps=args.future_len)
            if not torch.isfinite(loss):
                continue
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            bs = int(ext.candidate_valid.shape[0])
            n += bs
            sums["loss"] = sums.get("loss", 0.0) + float(loss.detach().cpu()) * bs
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + float(v) * bs
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(loss=sums["loss"] / max(n, 1), refresh=False)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if args.strict:
                raise
            if n == 0:
                print(f"Warning: skipped malformed first batch: {exc}")
            continue
    return {k: v / max(n, 1) for k, v in sums.items()} | {"num_samples": float(n)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train GameFormer/DTPP external baselines on COWP tensor-cache data.")
    ap.add_argument("--baseline", choices=["gameformer", "dtpp"], required=True)
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--label-config", default="configs/label.yaml")
    ap.add_argument("--train-config", default="configs/train.yaml")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--val-cache-dir", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-neighbors", type=int, default=10)
    ap.add_argument("--max-candidates", type=int, default=30)
    ap.add_argument("--future-len", type=int, default=80)
    ap.add_argument("--gameformer-modalities", type=int, default=6)
    ap.add_argument("--gameformer-encoder-layers", type=int, default=6)
    ap.add_argument("--gameformer-decoder-levels", type=int, default=4)
    ap.add_argument("--dtpp-fixed-cost", action="store_true")
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.label_config, args.data_config, args.train_config)
    device = _device(args.device)
    _seed(args.seed, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ExternalCOWPDataset(args.cache_dir, include_waymax_outcomes=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_torch, pin_memory=(device.type == "cuda" and args.num_workers > 0), drop_last=False)
    val_loader = None
    if args.val_cache_dir:
        val_ds = ExternalCOWPDataset(args.val_cache_dir, include_waymax_outcomes=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_torch, pin_memory=False, drop_last=False)

    model = _build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_metric = math.inf
    history = []
    best_path = out_dir / f"external_{args.baseline}_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, cfg, args, device, optimizer, epoch)
        val_metrics = _run_epoch(model, val_loader, cfg, args, device, None, epoch) if val_loader is not None else {}
        metric = float(val_metrics.get("plannerADE", val_metrics.get("loss", train_metrics.get("loss", math.inf))))
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, indent=2))
        ckpt = {
            "baseline": args.baseline,
            "model": _state_dict(model),
            "cfg": cfg,
            "args": vars(args),
            "epoch": epoch,
            "metrics": record,
        }
        torch.save(ckpt, out_dir / f"external_{args.baseline}_epoch{epoch}.pt")
        if metric < best_metric:
            best_metric = metric
            torch.save(ckpt, best_path)
        with (out_dir / f"external_{args.baseline}_history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    print(json.dumps({"best_checkpoint": str(best_path), "best_metric": best_metric}, indent=2))


if __name__ == "__main__":
    main()
