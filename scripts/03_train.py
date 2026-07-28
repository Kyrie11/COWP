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
from torch.utils.data import DataLoader, WeightedRandomSampler

from cowp.core.config import load_config
from cowp.data.dataset import TorchCOWPDataset, collate_torch
from cowp.models.cowp_model import COWPModel
from cowp.utils.progress import tqdm_iter
from cowp.utils.dataloader_runtime import configure_dataloader_runtime
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
) -> DataLoader:
    sampler = None
    if shuffle and oversample:
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
    """Return a checkpoint state_dict that can be loaded without torch.compile."""
    inner = getattr(model, "_orig_mod", None)
    if inner is not None and isinstance(inner, torch.nn.Module):
        return inner.state_dict()
    return model.state_dict()


def _load_model_state_robust(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load checkpoints saved from either eager or compiled models."""
    first_error: RuntimeError | None = None
    try:
        model.load_state_dict(state)
        return
    except RuntimeError as exc:
        first_error = exc
    if state and all(k.startswith("_orig_mod.") for k in state.keys()):
        stripped = {k[len("_orig_mod."):]: v for k, v in state.items()}
        model.load_state_dict(stripped)
        return
    assert first_error is not None
    raise first_error


def _score_from_history_row(row: dict[str, Any]) -> float:
    """Return the scalar used for best-checkpoint selection for a history row."""
    if "val/loss" in row:
        return float(row.get("val/loss", float("inf")))
    if "train/loss" in row:
        return float(row.get("train/loss", float("inf")))
    return float("inf")


def _load_existing_history(path: Path, *, before_epoch: int | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Warning: failed to read existing history {path}: {exc}")
        return []
    if not isinstance(data, list):
        return []
    rows = [r for r in data if isinstance(r, dict)]
    if before_epoch is not None:
        rows = [r for r in rows if int(r.get("epoch", -1)) < int(before_epoch)]
    return rows


def _best_score_from_history(history: list[dict[str, Any]]) -> float:
    best = float("inf")
    for row in history:
        score = _score_from_history_row(row)
        if math.isfinite(score) and score < best:
            best = score
    return best


def _checkpoint_stage(ckpt: dict[str, Any]) -> str | None:
    stage = ckpt.get("stage")
    return str(stage) if stage is not None else None


def _checkpoint_epoch(ckpt: dict[str, Any]) -> int | None:
    epoch = ckpt.get("epoch")
    if epoch is None:
        return None
    try:
        return int(epoch)
    except Exception:
        return None


def _make_checkpoint_payload(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    opt: torch.optim.Optimizer,
    *,
    epoch: int,
    stage: str,
    best_val: float,
    save_optimizer: bool,
    extra: dict[str, float] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": _model_state_dict_for_save(model),
        "cfg": cfg,
        "epoch": int(epoch),
        "stage": stage,
        "best_val": float(best_val),
    }
    if save_optimizer:
        payload["optimizer"] = opt.state_dict()
    if extra:
        payload.update(extra)
    return payload


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
    ap.add_argument("--resume-training", action="store_true", help="When --resume points to a checkpoint from the same stage, continue epoch numbering/history instead of treating it as a warm start.")
    ap.add_argument("--no-save-optimizer", action="store_true", help="Do not store optimizer state in epoch/best checkpoints. This saves disk but weakens exact crash resume.")
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
    configure_dataloader_runtime()

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
    include_response_traj = not (stage == "response" and float(loss_weights.get("response_traj_l1", 0.1)) == 0.0)
    include_response_components = not (stage == "response" and float(loss_weights.get("response_components_l1", 0.25)) == 0.0)
    print(f"COWP train startup: stage={stage}, device={device}, cuda_available={torch.cuda.is_available()}, amp={args.amp}, compile={compile_enabled}, batch_size={batch_size}, pin_memory={pin_memory}, prefetch_factor={effective_prefetch if effective_prefetch is not None else tcfg.get('prefetch_factor', 2)}, response_traj_l1={float(loss_weights.get('response_traj_l1', 0.0))}, load_response_traj={include_response_traj}, load_waymax_outcomes={bool(args.with_waymax_outcome_labels)}")
    if device.type == "cuda":
        try:
            print(f"CUDA device: {torch.cuda.get_device_name(device)}")
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
    print(f"Loaded datasets: train={len(train_ds)}" + (f", val={len(val_ds)}" if val_ds is not None else ""))
    oversample_enabled = bool(tcfg.get("positive_pair_oversampling", True)) and not args.no_positive_oversampling
    # Representation/natural stages do not consume witness/candidate labels.
    # Scanning every npz for those labels can add many minutes before the first
    # batch and does not change the Stage-A objective.
    if stage in {"representation", "natural"} and not args.force_positive_oversampling:
        if oversample_enabled:
            print(f"Skipping positive oversampling for stage={stage}; use --force-positive-oversampling to enable it.")
        oversample_enabled = False

    train_dl = _make_loader(
        train_ds,
        cfg,
        batch_size,
        shuffle=True,
        oversample=oversample_enabled,
        use_cuda=device.type == "cuda",
        progress=not args.no_progress,
        num_workers=args.num_workers,
        sampler_cache=not args.no_sampler_cache,
        pin_memory=pin_memory,
        prefetch_factor=effective_prefetch,
    )
    val_dl = _make_loader(val_ds, cfg, batch_size, shuffle=False, oversample=False, use_cuda=device.type == "cuda", progress=not args.no_progress, num_workers=args.num_workers, pin_memory=pin_memory, prefetch_factor=effective_prefetch) if val_ds is not None else None

    print(f"DataLoader: num_workers={train_dl.num_workers}, pin_memory={train_dl.pin_memory}, prefetch_factor={getattr(train_dl, 'prefetch_factor', None)}, train_batches={len(train_dl)}" + (f", val_batches={len(val_dl)}" if val_dl is not None else ""))
    model = COWPModel(cfg).to(device)
    resume_ckpt: dict[str, Any] | None = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device)
        _load_model_state_robust(model, resume_ckpt["model"])
    model = _maybe_compile_model(model, compile_enabled, backend=args.compile_backend)
    opt = _make_adamw_optimizer(
        model,
        lr=float(args.lr if args.lr is not None else tcfg.get("lr", 3e-4)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
        fused=bool(args.fused_adamw or tcfg.get("fused_adamw", False)),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = args.epochs or int(tcfg.get("epochs", 10))
    history_path = output_dir / f"history_{stage}.json"
    history: list[dict[str, Any]] = []
    best_val = float("inf")
    start_epoch = 0
    if args.resume_training:
        if not args.resume or resume_ckpt is None:
            raise ValueError("--resume-training requires --resume to point to a same-stage checkpoint")
        ckpt_stage = _checkpoint_stage(resume_ckpt)
        ckpt_epoch = _checkpoint_epoch(resume_ckpt)
        if ckpt_epoch is None:
            raise ValueError(f"Checkpoint {args.resume} has no integer epoch; cannot use --resume-training")
        if ckpt_stage is not None and ckpt_stage != stage:
            raise ValueError(f"--resume-training expected stage={stage}, but checkpoint stage={ckpt_stage}; use plain --resume for cross-stage warm start")
        start_epoch = ckpt_epoch + 1
        history = _load_existing_history(history_path, before_epoch=start_epoch)
        best_val = _best_score_from_history(history)
        ckpt_best = resume_ckpt.get("best_val")
        if ckpt_best is not None:
            try:
                best_val = min(best_val, float(ckpt_best))
            except Exception:
                pass
        if not math.isfinite(best_val):
            best_val = float("inf")
        if "optimizer" in resume_ckpt:
            try:
                opt.load_state_dict(resume_ckpt["optimizer"])
                print(f"Resumed optimizer state from {args.resume}")
            except Exception as exc:
                print(f"Warning: failed to load optimizer state from {args.resume}; continuing with a fresh optimizer: {exc}")
        else:
            print(f"Warning: checkpoint {args.resume} has no optimizer state; continuing model weights from epoch {ckpt_epoch} with a fresh optimizer.")
        print(f"Resume-training stage={stage}: checkpoint_epoch={ckpt_epoch}, next_epoch={start_epoch}, target_epochs={epochs}, previous_history_rows={len(history)}")
    if start_epoch >= epochs:
        print(f"Stage {stage} already reached target epochs: checkpoint next_epoch={start_epoch}, target_epochs={epochs}")
        return
    for epoch in range(start_epoch, epochs):
        train_metrics = _run_epoch(
            model,
            train_dl,
            device,
            stage,
            loss_weights,
            opt,
            epoch=epoch,
            progress=not args.no_progress,
            amp=args.amp,
            non_blocking_transfer=pin_memory,
            decode_response_traj=include_response_traj,
        )
        row: dict[str, Any] = {"epoch": epoch, **{f"train/{k}": v for k, v in train_metrics.items()}}
        do_val = val_dl is not None and int(args.val_every) > 0 and ((epoch + 1) % int(args.val_every) == 0 or epoch == epochs - 1)
        if do_val:
            val_metrics = _run_epoch(
                model,
                val_dl,
                device,
                stage,
                loss_weights,
                None,
                epoch=epoch,
                progress=not args.no_progress,
                amp=args.amp,
                non_blocking_transfer=pin_memory,
                decode_response_traj=include_response_traj,
            )
            row.update({f"val/{k}": v for k, v in val_metrics.items()})
            score_loss = float(val_metrics.get("loss", float("inf")))
            score_name = "val_loss"
        else:
            # If validation is intentionally disabled for smoke training, still
            # produce cowp_<stage>_best.pt so downstream stage commands can resume.
            score_loss = float(train_metrics.get("loss", float("inf"))) if (val_dl is None or int(args.val_every) == 0) else float("inf")
            score_name = "train_loss"
        if math.isfinite(score_loss) and score_loss < best_val:
            best_val = score_loss
            torch.save(
                _make_checkpoint_payload(
                    model,
                    cfg,
                    opt,
                    epoch=epoch,
                    stage=stage,
                    best_val=best_val,
                    save_optimizer=not args.no_save_optimizer,
                    extra={score_name: score_loss},
                ),
                output_dir / f"cowp_{stage}_best.pt",
            )
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        torch.save(
            _make_checkpoint_payload(
                model,
                cfg,
                opt,
                epoch=epoch,
                stage=stage,
                best_val=best_val,
                save_optimizer=not args.no_save_optimizer,
            ),
            output_dir / f"cowp_{stage}_epoch{epoch:03d}.pt",
        )
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
