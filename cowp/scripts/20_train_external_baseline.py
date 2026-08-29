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
from cowp.external_baselines.pluto_cowp import COWPPLUTO, pluto_loss
from cowp.external_baselines.plant2_cowp import COWPPlanT2, plant2_loss
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
        variable_cost = bool(getattr(args, "dtpp_variable_cost", False)) and not bool(getattr(args, "dtpp_fixed_cost", False))
        return COWPDTPP(neighbors=args.max_neighbors, max_branch=args.max_candidates, variable_cost=variable_cost)
    if args.baseline == "pluto":
        return COWPPLUTO(
            future_len=args.future_len,
            d_model=args.pluto_d_model,
            num_heads=args.pluto_num_heads,
            encoder_layers=args.pluto_encoder_layers,
            lateral_queries=args.pluto_lateral_queries,
            longitudinal_queries=args.pluto_longitudinal_queries,
        )
    if args.baseline == "plant2":
        return COWPPlanT2(
            future_len=args.future_len,
            d_model=args.plant2_d_model,
            num_heads=args.plant2_num_heads,
            layers=args.plant2_layers,
        )
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


def _autocast(device: torch.device, enabled: bool, dtype_name: str = "auto"):
    if dtype_name == "auto":
        use_bf16 = bool(device.type == "cuda" and torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    try:
        return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)
    except Exception:  # pragma: no cover - older PyTorch fallback
        return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)


def _clip_grad_norm_stable(parameters, max_norm: float) -> tuple[torch.Tensor, list[str], bool]:
    """Clip global gradient norm with a float64 fallback for fp32 norm overflow.

    Returns ``(pre_clip_norm, bad_gradient_parameter_names, used_fp64_fallback)``.
    True NaN/Inf entries are reported and left unhidden; finite but extremely large
    gradients are clipped using an fp64 accumulator so AMP/fp32 training can recover.
    """
    if isinstance(parameters, torch.nn.Module):
        named = list(parameters.named_parameters())
    else:
        named = [(str(i), p) for i, p in enumerate(parameters)]
    params = [p for _, p in named if p.grad is not None]
    if not params:
        ref = named[0][1] if named else None
        device = ref.device if ref is not None else torch.device("cpu")
        return torch.zeros((), dtype=torch.float32, device=device), [], False

    def _bad_paths() -> list[str]:
        bad: list[str] = []
        for name, param in named:
            grad = param.grad
            if grad is None:
                continue
            try:
                if not bool(torch.isfinite(grad.detach()).all().item()):
                    bad.append(name)
            except Exception:
                bad.append(name)
        return bad

    max_norm = max(float(max_norm), 1.0e-6)
    try:
        norm = torch.nn.utils.clip_grad_norm_(params, max_norm, error_if_nonfinite=True)
        return norm if torch.is_tensor(norm) else torch.tensor(float(norm), device=params[0].device), [], False
    except TypeError:
        bad = _bad_paths()
        if bad:
            return torch.full((), float("nan"), device=params[0].device), bad, False
    except RuntimeError:
        bad = _bad_paths()
        if bad:
            return torch.full((), float("inf"), device=params[0].device), bad, False

    device = params[0].device
    total_sq = torch.zeros((), dtype=torch.float64, device=device)
    for param in params:
        total_sq = total_sq + param.grad.detach().to(torch.float64).square().sum()
    norm64 = total_sq.sqrt()
    if not bool(torch.isfinite(norm64).item()):
        return norm64, ["<finite entries but non-finite float64 norm>"], True
    clip_coef = min(1.0, max_norm / (float(norm64.item()) + 1.0e-6))
    if clip_coef < 1.0:
        for param in params:
            param.grad.mul_(clip_coef)
    return norm64, [], True


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
    amp_dtype = str(getattr(args, "amp_dtype", "auto"))
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
            require_candidates = args.baseline == "dtpp"
            ext = make_external_batch(batch, cfg, device=device, max_neighbors=args.max_neighbors, max_candidates=args.max_candidates, horizon=args.future_len, baseline=args.baseline, require_candidates=require_candidates, require_future=True)
            with _autocast(device, amp_enabled, amp_dtype):
                if args.baseline == "gameformer":
                    outputs = model(ext.gameformer_inputs)
                    loss, metrics = gameformer_loss(outputs, ext.ego_future_xy, ext.ego_future_valid, ext.neighbors_future_xy, ext.neighbors_future_valid)
                elif args.baseline == "dtpp":
                    best_idx = best_candidate_to_logged_ego(ext.candidates, ext.candidate_valid, ext.ego_future_xy, ext.ego_future_valid)
                    loss, metrics = dtpp_loss(model, ext.dtpp_inputs, ext.dtpp_candidate_tree, ext.candidate_valid, best_idx, ext.ego_future_xy, ext.ego_future_valid, ext.neighbors_future_xy, ext.neighbors_future_valid, timesteps=args.future_len)
                elif args.baseline == "pluto":
                    loss, metrics = pluto_loss(model, ext.planner_inputs, ext.ego_future_xy, ext.ego_future_valid)
                elif args.baseline == "plant2":
                    loss, metrics = plant2_loss(model, ext.planner_inputs, ext.ego_future_xy, ext.ego_future_valid)
                else:
                    raise ValueError(f"Unsupported baseline: {args.baseline}")
            if not torch.isfinite(loss):
                skipped += 1
                if skipped <= 5 or (log_every and skipped % max(log_every, 1) == 0):
                    _log(
                        f"{phase} {args.baseline} epoch={epoch} batch={batch_idx} skipped non-finite loss "
                        f"amp={amp_enabled} amp_dtype={amp_dtype}"
                    )
                continue
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    _clip_grad_norm_stable(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    _clip_grad_norm_stable(model.parameters(), args.grad_clip)
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
    skip_fraction = float(skipped / max(int(total_batches or (skipped + 1)), 1))
    max_skip_fraction = float(getattr(args, "max_skip_fraction", 0.02))
    if skip_fraction > max_skip_fraction:
        raise RuntimeError(
            f"External baseline {args.baseline} {phase} epoch {epoch} skipped {skipped}/{total_batches} batches "
            f"({skip_fraction:.3%}) > max_skip_fraction={max_skip_fraction:.3%}. "
            "Do not publish or continue training a checkpoint learned from a tiny surviving subset. "
            "Use ego-frame inputs and BF16/FP32 loss computation; rerun with --strict to expose malformed batches."
        )
    out = {k: v / max(n, 1) for k, v in sums.items()} | {"num_samples": float(n), "num_batches": float(total_batches or 0), "skipped_batches": float(skipped), "skip_fraction": skip_fraction, "seconds": float(elapsed)}
    _log(f"{phase} {args.baseline} epoch={epoch} done samples={n} skipped={skipped} seconds={elapsed:.1f} loss={out.get('loss', float('nan')):.6f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train source-faithful external baselines on COWP tensor-cache data.")
    ap.add_argument("--baseline", choices=["gameformer", "dtpp", "pluto", "plant2"], required=True)
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
    ap.add_argument("--dtpp-variable-cost", action="store_true", help="Enable DTPP variable-cost head/weights; default off to match the public-source reference recipe used by the launcher.")
    ap.add_argument("--pluto-d-model", type=int, default=128)
    ap.add_argument("--pluto-num-heads", type=int, default=8)
    ap.add_argument("--pluto-encoder-layers", type=int, default=4)
    ap.add_argument("--pluto-lateral-queries", type=int, default=4)
    ap.add_argument("--pluto-longitudinal-queries", type=int, default=6)
    ap.add_argument("--plant2-d-model", type=int, default=128)
    ap.add_argument("--plant2-num-heads", type=int, default=8)
    ap.add_argument("--plant2-layers", type=int, default=4)
    ap.add_argument("--checkpoint-every", type=int, default=1)
    ap.add_argument("--contract-version", default=None)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision during training/validation.")
    ap.add_argument("--amp-dtype", choices=["auto", "bfloat16", "float16"], default="auto", help="AMP dtype; auto prefers BF16 to avoid FP16 overflow in trajectory/GMM losses.")
    ap.add_argument("--max-skip-fraction", type=float, default=0.02, help="Fail an epoch when non-finite/malformed batches exceed this fraction.")
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
    complete_payload = {
        "completed": True,
        "baseline": args.baseline,
        "best_checkpoint": str(best_path),
        "best_metric": float(best_metric),
        "epochs": int(args.epochs),
        "training_signature": {
            "contract_version": str(args.contract_version or os.environ.get("EXTERNAL_TRAINING_CONTRACT_VERSION", "v6_womd_map_topology_source_fidelity_20260827")),
            "amp": bool(args.amp),
            "amp_dtype": str(args.amp_dtype),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "dtpp_variable_cost": bool(args.baseline == "dtpp" and args.dtpp_variable_cost and not args.dtpp_fixed_cost),
            "max_neighbors": int(args.max_neighbors),
            "max_candidates": int(args.max_candidates),
            "future_len": int(args.future_len),
        },
    }
    complete_path = out_dir / f"external_{args.baseline}_training_complete.json"
    with complete_path.open("w", encoding="utf-8") as f:
        json.dump(complete_payload, f, indent=2)
    _log(f"wrote training completion marker {complete_path}")
    _log(json.dumps({"best_checkpoint": str(best_path), "best_metric": best_metric}, indent=2))


if __name__ == "__main__":
    main()
