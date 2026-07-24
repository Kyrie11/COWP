from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
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
from cowp.utils.dataloader_runtime import configure_dataloader_runtime


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def _safe_len(obj: Any) -> int | None:
    try:
        return int(len(obj))
    except Exception:
        return None


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


def _num_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _make_grad_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:  # older PyTorch
        return torch.cuda.amp.GradScaler(enabled=True)


def _autocast(device: torch.device, enabled: bool):
    try:
        return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)
    except Exception:  # pragma: no cover - older PyTorch fallback
        return torch.cuda.amp.autocast(enabled=enabled)


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    scaler=None,
) -> dict[str, float]:
    train = optimizer is not None
    phase = "train" if train else "val"
    model.train(train)
    sums: dict[str, float] = {}
    n = 0
    skipped = 0
    total_batches = _safe_len(loader)
    log_every = max(int(getattr(args, "log_every", 0) or 0), 0)
    amp_enabled = bool(getattr(args, "amp", False) and device.type == "cuda")
    _log(f"{phase} {args.baseline} epoch={epoch} start batches={total_batches if total_batches is not None else 'unknown'} batch_size={args.batch_size} workers={args.num_workers} amp={amp_enabled}")
    iterator = tqdm_iter(
        loader,
        enabled=not args.no_progress,
        total=total_batches,
        desc=f"{phase} {args.baseline} e{epoch}",
        unit="batch",
    )
    t0 = time.time()
    last_log = t0
    for batch_idx, batch in enumerate(iterator, start=1):
        try:
            ext = make_external_batch(batch, cfg, device=device, max_neighbors=args.max_neighbors, max_candidates=args.max_candidates, horizon=args.future_len)
            with _autocast(device, amp_enabled):
                if args.baseline == "gameformer":
                    outputs = model(ext.gameformer_inputs)
                    loss, metrics = gameformer_loss(outputs, ext.ego_future_xy, ext.ego_future_valid, ext.neighbors_future_xy, ext.neighbors_future_valid)
                else:
                    best_idx = best_candidate_to_logged_ego(ext.candidates, ext.candidate_valid, ext.ego_future_xy, ext.ego_future_valid)
                    loss, metrics = dtpp_loss(model, ext.dtpp_inputs, ext.dtpp_candidate_tree, ext.candidate_valid, best_idx, ext.ego_future_xy, ext.ego_future_valid, ext.neighbors_future_xy, ext.neighbors_future_valid, timesteps=args.future_len)
            if not torch.isfinite(loss):
                skipped += 1
                if log_every and (skipped <= 3 or skipped % log_every == 0):
                    _log(f"{phase} {args.baseline} epoch={epoch} batch={batch_idx} skipped non-finite loss")
                continue
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
            bs = int(ext.candidate_valid.shape[0])
            n += bs
            sums["loss"] = sums.get("loss", 0.0) + float(loss.detach().cpu()) * bs
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + float(v) * bs
            mean_loss = sums["loss"] / max(n, 1)
            if hasattr(iterator, "set_postfix"):
                postfix = {"loss": f"{mean_loss:.4f}", "samples": n, "skipped": skipped}
                for mk in ("plannerADE", "plannerFDE", "score_ce", "neighbor_cmp", "ego_reg"):
                    if mk in metrics:
                        try:
                            postfix[mk] = f"{float(metrics[mk]):.4f}"
                        except Exception:
                            pass
                iterator.set_postfix(postfix, refresh=False)
            now = time.time()
            should_log = bool(log_every and (batch_idx == 1 or batch_idx % log_every == 0 or (total_batches is not None and batch_idx == total_batches)))
            if should_log:
                elapsed = max(now - t0, 1e-6)
                batch_rate = batch_idx / elapsed
                sample_rate = n / elapsed
                total_txt = str(total_batches) if total_batches is not None else "?"
                _log(
                    f"{phase} {args.baseline} epoch={epoch} batch={batch_idx}/{total_txt} "
                    f"samples={n} skipped={skipped} loss={mean_loss:.6f} "
                    f"batch_rate={batch_rate:.3f}/s sample_rate={sample_rate:.1f}/s"
                )
                last_log = now
            elif log_every == 0 and batch_idx == 1 and now - last_log > 30:
                _log(f"{phase} {args.baseline} epoch={epoch} first batch finished samples={n} loss={mean_loss:.6f}")
                last_log = now
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            skipped += 1
            if args.strict:
                raise
            if skipped <= 5 or (log_every and skipped % max(log_every, 1) == 0):
                _log(f"Warning: skipped malformed batch phase={phase} baseline={args.baseline} epoch={epoch} batch={batch_idx}: {type(exc).__name__}: {exc}")
            continue
    elapsed = max(time.time() - t0, 1e-6)
    if n == 0:
        raise RuntimeError(f"No usable samples in {phase} epoch {epoch}. skipped_batches={skipped}. Set --strict to expose the first malformed batch.")
    out = {k: v / max(n, 1) for k, v in sums.items()} | {"num_samples": float(n), "num_batches": float(total_batches or 0), "skipped_batches": float(skipped), "seconds": float(elapsed)}
    _log(f"{phase} {args.baseline} epoch={epoch} done samples={n} skipped={skipped} seconds={elapsed:.1f} loss={out.get('loss', float('nan')):.6f}")
    return out


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
    ap.add_argument("--val-num-workers", type=int, default=2)
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
    ap.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision during training/validation.")
    ap.add_argument("--prefetch-factor", type=int, default=int(os.environ.get("PREFETCH_FACTOR", "2")))
    ap.add_argument("--val-prefetch-factor", type=int, default=1)
    ap.add_argument("--sharing-strategy", choices=["auto", "current", "file_descriptor", "file_system"], default=None)
    ap.add_argument("--no-persistent-workers", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--log-every", type=int, default=int(os.environ.get("LOG_EVERY", "25")), help="Emit one line per N batches even when tqdm is disabled. Use 0 to disable heartbeat lines.")
    args = ap.parse_args()
    loader_runtime = configure_dataloader_runtime(args.sharing_strategy)

    _log(f"external training entry baseline={args.baseline} pid={os.getpid()} python={sys.executable} torch={torch.__version__}")
    _log(f"DataLoader IPC runtime={json.dumps(loader_runtime, sort_keys=True)}")
    _log(f"args={json.dumps(vars(args), sort_keys=True)}")
    cfg = load_config(args.label_config, args.data_config, args.train_config)
    device = _device(args.device)
    _seed(args.seed, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"device={device} cuda_available={torch.cuda.is_available()} output_dir={out_dir}")

    _log(f"loading train dataset from {args.cache_dir}")
    train_ds = ExternalCOWPDataset(args.cache_dir, include_waymax_outcomes=False)
    _log(f"train dataset ready scenes={len(train_ds)}")
    loader_kwargs = {
        "num_workers": args.num_workers,
        "collate_fn": collate_torch,
        "pin_memory": (device.type == "cuda" and args.num_workers > 0),
        "drop_last": False,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = not args.no_persistent_workers
        loader_kwargs["prefetch_factor"] = max(int(args.prefetch_factor), 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    _log(f"train loader ready batches={len(train_loader)}")
    val_loader = None
    if args.val_cache_dir:
        _log(f"loading val dataset from {args.val_cache_dir}")
        val_ds = ExternalCOWPDataset(args.val_cache_dir, include_waymax_outcomes=True)
        _log(f"val dataset ready scenes={len(val_ds)}")
        val_loader_kwargs = dict(loader_kwargs)
        val_workers = max(int(args.val_num_workers), 0)
        val_loader_kwargs["num_workers"] = val_workers
        val_loader_kwargs["pin_memory"] = False
        if val_workers > 0:
            val_loader_kwargs["persistent_workers"] = False
            val_loader_kwargs["prefetch_factor"] = max(int(args.val_prefetch_factor), 1)
        else:
            val_loader_kwargs.pop("persistent_workers", None)
            val_loader_kwargs.pop("prefetch_factor", None)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **val_loader_kwargs)
        _log(f"val loader ready batches={len(val_loader)}")

    model = _build_model(args).to(device)
    total_params, trainable_params = _num_parameters(model)
    _log(f"model built baseline={args.baseline} total_params={total_params} trainable_params={trainable_params}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = _make_grad_scaler(bool(args.amp and device.type == "cuda"))
    if scaler is not None:
        _log("AMP GradScaler enabled")
    best_metric = math.inf
    history = []
    best_path = out_dir / f"external_{args.baseline}_best.pt"

    for epoch in range(1, args.epochs + 1):
        _log(f"epoch {epoch}/{args.epochs} begin baseline={args.baseline}")
        train_metrics = _run_epoch(model, train_loader, cfg, args, device, optimizer, epoch, scaler=scaler)
        val_metrics = _run_epoch(model, val_loader, cfg, args, device, None, epoch, scaler=None) if val_loader is not None else {}
        metric = float(val_metrics.get("plannerADE", val_metrics.get("loss", train_metrics.get("loss", math.inf))))
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        _log("epoch summary " + json.dumps(record, sort_keys=True))
        ckpt = {
            "baseline": args.baseline,
            "model": _state_dict(model),
            "cfg": cfg,
            "args": vars(args),
            "epoch": epoch,
            "metrics": record,
        }
        epoch_path = out_dir / f"external_{args.baseline}_epoch{epoch}.pt"
        torch.save(ckpt, epoch_path)
        _log(f"saved checkpoint {epoch_path}")
        if metric < best_metric:
            best_metric = metric
            torch.save(ckpt, best_path)
            _log(f"updated best checkpoint {best_path} best_metric={best_metric:.6f}")
        with (out_dir / f"external_{args.baseline}_history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    _log(json.dumps({"best_checkpoint": str(best_path), "best_metric": best_metric}, indent=2))


if __name__ == "__main__":
    main()
